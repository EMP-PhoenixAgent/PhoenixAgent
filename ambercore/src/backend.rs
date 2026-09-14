//! Compute backend abstraction.
//!
//! The [`Backend`] trait decouples the *compute target* (CPU, CUDA, potentially
//! Metal/Vulkan later) from the model and pipeline code. Candle's own
//! [`candle_core::Device`][Device] enum already abstracts over CPU/CUDA/Metal
//! via Cargo feature flags — the backends here hand out the right `Device` for
//! the chosen target.
//!
//! ## Backends
//!
//! - [`CpuBackend`] — always available. The default.
//! - [`CudaBackend`] — gated behind the `cuda` Cargo feature. Requires the CUDA
//!   toolkit (`nvcc`) at build time and an NVIDIA GPU at runtime.
//! - [`MetalBackend`] — gated behind the `metal` Cargo feature + `target_os =
//!   "macos"`. Requires an Apple GPU.
//! - [`AmdBackend`] — gated behind the `rocm` Cargo feature. **Always errors at
//!   resolve time**: candle has no stable ROCm backend, so this is a stub that
//!   keeps the surface ready for the day upstream lands.
//!
//! ## Choosing at runtime
//!
//! [`DeviceChoice`] (`cpu` / `cuda` / `metal` / `amd` / `auto`) is parsed from
//! the CLI and resolved to a concrete `Box<dyn Backend>` via [`resolve_backend`].
//! `auto` tries each compiled-in GPU backend in turn (CUDA → Metal) before
//! falling back to CPU.
//!
//! [Device]: https://docs.rs/candle-core/latest/candle_core/enum.Device.html

use crate::error::Result;
use candle_core::Device;

/// Live GPU status reported by a GPU backend. `None` on CPU.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuInfo {
    /// Device name, e.g. `"NVIDIA GeForce RTX 3050"`.
    pub name: String,
    /// Total VRAM in MB.
    pub vram_total_mb: Option<u64>,
    /// Used VRAM in MB (live reading at call time).
    pub vram_used_mb: Option<u64>,
    /// Free VRAM in MB (live reading at call time) — powers the model-load
    /// headroom warning (`check_vram_headroom`).
    pub vram_free_mb: Option<u64>,
}

// ─────────────────────── CUDA error translation ────────────────────────────
//
// candle embeds its GPU kernels as PTX and JIT-compiles them through the
// installed NVIDIA driver at runtime. A driver older than the toolkit that
// built the binary rejects the PTX with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
// — surfacing deep inside a model load with no hint about the actual cause.
// These helpers turn that (and its close cousins) into an actionable message,
// and are applied both at backend construction (kernel warm-up) and at model
// build time as defense in depth.

/// Minimum NVIDIA driver version per CUDA toolkit (Windows). Used to make
/// PTX-mismatch errors actionable.
fn min_driver_for(toolkit: &str) -> &'static str {
    match toolkit {
        "12.0" => "525.60",
        "12.1" => "531.14",
        "12.2" => "536.25",
        "12.3" => "545.84",
        "12.4" => "550.54",
        "12.5" => "555.42",
        "12.6" => "560.76",
        "12.7" => "565.57",
        "12.8" => "570.51",
        "13.0" => "580.65",
        // Point releases within a major CUDA line: the major line's first
        // driver is the baseline; newer PTX may need a newer driver within it.
        t if t.starts_with("13.") => "580.65 or newer (CUDA 13 era)",
        t if t.starts_with("12.") => "570.51 or newer (late CUDA 12 era)",
        _ => "the driver release that shipped with this CUDA version",
    }
}

/// The installed driver's CUDA version (e.g. `12080` → `"12.8"`), best-effort.
fn driver_cuda_version() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc;
        // SAFETY: cuDriverGetVersion only writes the i32 out-param.
        let mut v: std::ffi::c_int = 0;
        let rc = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) };
        if rc as u32 == 0 && v > 0 {
            Some(format!("{}.{}", v / 1000, (v % 1000) / 10))
        } else {
            None
        }
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// Expand a CUDA driver-mismatch error into an actionable message. Returns the
/// original text unchanged for unrelated errors.
pub fn translate_cuda_error(err: &str) -> String {
    let ptx = err.contains("UNSUPPORTED_PTX_VERSION") || err.contains("unsupported toolchain");
    let no_binary = err.contains("NO_BINARY_FOR_GPU") || err.contains("no kernel image");
    if !ptx && !no_binary {
        return err.to_string();
    }
    let toolkit = option_env!("AMBERCORE_CUDA_TOOLKIT");
    let driver = driver_cuda_version();
    let cause = if ptx {
        "this build's GPU kernels were compiled with a newer CUDA toolkit than \
         your NVIDIA driver supports"
    } else {
        "this build's GPU kernels target a newer GPU generation than the one \
         installed"
    };
    let fix = if ptx {
        "update your NVIDIA driver (https://www.nvidia.com/drivers), or use the \
         universal CPU build"
    } else {
        "use the universal CPU build (this GPU generation predates the kernels)"
    };
    let toolkit_note = toolkit
        .map(|t| format!("built with CUDA {t} (needs driver {})", min_driver_for(t)))
        .unwrap_or_else(|| "built with an unknown CUDA toolkit version".into());
    let driver_note = driver
        .map(|d| format!("your driver reports CUDA {d}"))
        .unwrap_or_else(|| "your driver's CUDA version could not be read".into());
    format!(
        "{err}\n\nCUDA kernel mismatch: {cause} ({toolkit_note}; {driver_note}). To fix: {fix}."
    )
}

// ─────────────────────── VRAM headroom warning ─────────────────────────────
//
// Under WDDM (Windows; Linux behaves similarly with pageable device memory),
// an oversized CUDA allocation never FAILS — the driver spills it to shared
// system memory and pages it over PCIe on every step, collapsing throughput
// ~12x (observed: qwen3-8b Q4_K_M went 26.6 → 2.2 tok/s when the 8 GiB card
// also hosted the app's WebView + Discord + a browser). So we WARN at model
// load, before the crawl can surprise anyone. Best-effort: `nvidia-smi` is
// queried (no new dependency, context-free — NVML needs no current CUDA
// context unlike mem_get_info); a missing answer just means no warning.

/// Estimated resident size of a GGUF on the GPU: file × 1.5 + 300 MiB fixed
/// (KV cache + activations + CUDA context + fragmentation). Calibrated on
/// qwen3-8b Q4_K_M: 4.7 GiB file → 7.1 GiB resident on an RTX 3050. The
/// factor deliberately overshoots slightly — borderline refusals err on the
/// safe side.
fn estimated_resident_mb(file_mb: u64) -> u64 {
    file_mb * 3 / 2 + 300
}

/// Pre-flight verdict for a GGUF about to load.
#[derive(Debug, Clone)]
pub struct VramCheck {
    /// `Some(true)`: fits. `Some(false)`: would spill — **the load is
    /// REFUSED** (pre-flight refusal). `None`: free VRAM couldn't be queried
    /// (CPU backend, no nvidia-smi, parse failure) — loads proceed unjudged.
    pub fits: Option<bool>,
    /// Estimated resident MB needed (weights + KV + activations + overhead).
    pub needed_mb: u64,
    /// Free VRAM MB at check time (`None` when unqueryable).
    pub free_mb: Option<u64>,
    /// Human line for the chat — "fits" or "refused, would spill" — when a
    /// verdict could be reached.
    pub message: Option<String>,
}

/// Live total + free VRAM (MiB) via nvidia-smi — context-free, so any thread
/// (UI commands, search) can call it. The search modal displays the GPU's
/// MAX (total) VRAM while the fit coloring compares against free.
pub fn query_vram_mb() -> Option<(u64, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut nums = text
        .lines()
        .next()?
        .split(',')
        .filter_map(|n| n.trim().parse::<u64>().ok());
    let total = nums.next()?;
    let free = nums.next()?;
    Some((total, free))
}

/// Live free-VRAM reading (MiB) via nvidia-smi — context-free, so any thread
/// (UI commands, search) can call it. Powers the model-search cards' fit
/// coloring alongside `check_vram`.
pub fn query_free_vram_mb() -> Option<u64> {
    query_vram_mb().map(|(_, free)| free)
}

/// Pre-flight VRAM check for a GGUF (BEFORE loading it). Callers refuse the
/// load on `fits == Some(false)`: under WDDM an oversized CUDA allocation
/// never fails — the driver spills it to system RAM and pages it over PCIe
/// on every step (observed: 26.6 → 2.2 tok/s, ~12x). Best-effort via
/// `nvidia-smi` (context-free — NVML-style queries need no current CUDA
/// context, unlike `mem_get_info`, so async load threads can call it).
pub fn check_vram(gguf_path: &std::path::Path) -> VramCheck {
    let file_mb = match std::fs::metadata(gguf_path) {
        Ok(m) => {
            let mut bytes = m.len();
            // Sharded safetensors: the entry is only the first shard — sum
            // the siblings the index.json maps so the estimate covers the
            // whole checkpoint (the F16 loader mmaps them all).
            if gguf_path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("safetensors")).unwrap_or(false) {
                if let Ok(index_raw) = std::fs::read_to_string(
                    gguf_path.parent().unwrap_or(std::path::Path::new(".")).join("index.json"),
                ) {
                    if let Ok(idx) = serde_json::from_str::<serde_json::Value>(&index_raw) {
                        if let Some(wm) = idx.get("weight_map").and_then(|v| v.as_object()) {
                            let parent = gguf_path.parent().unwrap_or(std::path::Path::new("."));
                            for shard in wm.values().filter_map(|v| v.as_str()) {
                                if let Ok(sm) = std::fs::metadata(parent.join(shard)) {
                                    bytes = bytes.saturating_add(sm.len());
                                }
                            }
                            // The entry shard was counted twice (once as the
                            // file itself, once via the map).
                            bytes = bytes.saturating_sub(m.len());
                        }
                    }
                }
            }
            bytes / (1024 * 1024)
        }
        Err(_) => {
            return VramCheck { fits: None, needed_mb: 0, free_mb: None, message: None };
        }
    };
    let needed_mb = estimated_resident_mb(file_mb);
    let free_mb = query_free_vram_mb();
    let name = gguf_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| gguf_path.display().to_string());
    match free_mb {
        Some(free) if needed_mb <= free => VramCheck {
            fits: Some(true),
            needed_mb,
            free_mb: Some(free),
            message: Some(format!(
                "{name}: contained in VRAM — needs ~{needed_mb} MiB, {free} MiB free."
            )),
        },
        Some(free) => VramCheck {
            fits: Some(false),
            needed_mb,
            free_mb: Some(free),
            message: Some(format!(
                "{name}: load REFUSED — needs ~{needed_mb} MiB resident but only {free} MiB \
                 free. Loading it anyway would spill to system RAM and run ~10x slower. \
                 {}{}",
                igpu_advice(),
                format_args!(
                    "Close GPU-heavy apps (top users now: {}) or pick a smaller model/quant.",
                    top_gpu_processes(3)
                )
            )),
        },
        None => VramCheck { fits: None, needed_mb, free_mb: None, message: None },
    }
}

/// Advice line for the spill refusal when a secondary (non-NVIDIA) GPU —
/// typically the Intel/AMD iGPU on a hybrid system — is present.
///
/// It canNOT host the spill itself: CUDA allocations only live on NVIDIA
/// devices (candle has no iGPU backend), and iGPU memory is system RAM on the
/// far side of the same PCIe bus — no bandwidth to gain over the driver's own
/// spill. What it CAN do is take display + app-rendering duty, freeing dGPU
/// VRAM — often the difference between fit and spill for an 8B-class model.
fn igpu_advice() -> String {
    match detect_secondary_gpu() {
        Some(igpu) => format!(
            "An integrated GPU ({igpu}) is present: it cannot hold CUDA spill memory, but \
             moving your display and GPU-heavy apps to it frees this GPU's VRAM (Windows \
             Settings > System > Display > Graphics, or plug the monitor into the \
             motherboard's output) — that often makes the model fit. "
        ),
        None => String::new(),
    }
}

/// Best-effort detection of a secondary (non-NVIDIA) display adapter — the
/// iGPU on hybrid systems. `None` when absent or unqueryable. Also used by
/// Phoenix to pin the app's own rendering to the iGPU (freeing dGPU VRAM for
/// the model — see main.rs `pin_rendering_to_igpu`).
pub fn detect_secondary_gpu() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            // Skip the NVIDIA dGPU(s) and Windows' fallback render device.
            .find(|l| !l.contains("NVIDIA") && !l.contains("Basic Display"))
            .map(str::to_string)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let out = std::process::Command::new("lspci").output().ok()?;
        if !out.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .filter(|l| l.contains("VGA ") || l.contains("Display controller"))
            .find(|l| !l.contains("NVIDIA"))
            .and_then(|l| l.split(':').next_back())
            .map(|s| s.trim().to_string())
    }
}

/// Top-N GPU-memory-consuming processes, best-effort (rows with unreadable
/// memory, e.g. permission-limited "[N/A]", are skipped). "unknown" when the
/// query itself fails.
fn top_gpu_processes(n: usize) -> String {
    let out = match std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return "unknown".to_string(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut procs: Vec<(u64, String)> = text
        .lines()
        .filter_map(|line| {
            let mut cols = line.split(',');
            let _pid = cols.next()?;
            let full_name = cols.next()?.trim().to_string();
            // Bare executable name — full paths are noisy in a warning line.
            let name = full_name.rsplit(['\\', '/']).next()?.to_string();
            let mem_mib: u64 = cols.next()?.trim().parse().ok()?;
            Some((mem_mib, name))
        })
        .collect();
    if procs.is_empty() {
        return "unknown".to_string();
    }
    procs.sort_by(|a, b| b.0.cmp(&a.0));
    procs
        .into_iter()
        .take(n)
        .map(|(mem, name)| format!("{name} ({mem} MiB)"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A compute target. Implementations hand out the candle [`Device`] that models
/// and tensors live on, and own any device-specific resources or limits.
pub trait Backend: Send + Sync {
    /// Human-readable backend name, e.g. `"cpu"` or `"cuda:0"`.
    fn name(&self) -> &str;

    /// The candle device tensors should be placed on for this backend.
    fn device(&self) -> Result<Device>;

    /// Live GPU status where the backend can report one (CUDA: device name +
    /// VRAM). `None` on CPU, or when the driver query fails — purely
    /// informational, never load-bearing.
    fn gpu_info(&self) -> Option<GpuInfo> {
        None
    }
}

/// CPU compute backend. Always available.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn device(&self) -> Result<Device> {
        Ok(Device::Cpu)
    }
}

/// Which device to run on. Parsed from the CLI
/// (`--device cpu|cuda|metal|amd|auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
#[value(rename_all = "lower")]
pub enum DeviceChoice {
    /// Force CPU (the default; always works).
    Cpu,
    /// Force CUDA. Errors if the `cuda` feature is off or no NVIDIA GPU is present.
    Cuda,
    /// Force Metal (Apple). Errors if the `metal` feature is off, this isn't
    /// macOS, or no Apple GPU is present.
    Metal,
    /// Force AMD/ROCm. Always errors: candle has no stable ROCm backend
    /// (see ACRoad.md §7).
    Amd,
    /// Try each compiled-in GPU backend in turn (CUDA → Metal), then fall back
    /// to CPU.
    Auto,
}

impl Default for DeviceChoice {
    fn default() -> Self {
        DeviceChoice::Cpu
    }
}

impl DeviceChoice {
    /// As the lowercase string used on the CLI / in logs.
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceChoice::Cpu => "cpu",
            DeviceChoice::Cuda => "cuda",
            DeviceChoice::Metal => "metal",
            DeviceChoice::Amd => "amd",
            DeviceChoice::Auto => "auto",
        }
    }
}

/// Resolve a [`DeviceChoice`] to a concrete backend.
///
/// - `cpu` → [`CpuBackend`] (always).
/// - `cuda` → [`CudaBackend`] when the `cuda` feature is on and a GPU is present;
///   an informative error otherwise.
/// - `metal` → [`MetalBackend`] when the `metal` feature is on and this is macOS
///   with an Apple GPU; an informative error otherwise.
/// - `amd` → always an error (candle has no stable ROCm backend; see ACRoad.md §7).
/// - `auto` → each compiled-in GPU backend in turn (CUDA → Metal), else [`CpuBackend`].
pub fn resolve_backend(choice: DeviceChoice) -> Result<Box<dyn Backend>> {
    match choice {
        DeviceChoice::Cpu => Ok(Box::new(CpuBackend::new())),
        DeviceChoice::Cuda => try_cuda(),
        DeviceChoice::Metal => try_metal(),
        DeviceChoice::Amd => Err(amd_unavailable()),
        DeviceChoice::Auto => try_auto(),
    }
}

/// Attempt to build the CUDA backend. Errors if the `cuda` feature is off or no
/// GPU is available at runtime.
fn try_cuda() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new(0) {
            Ok(b) => Ok(Box::new(b)),
            Err(e) => Err(e),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(crate::error::Error::Backend(
            "CUDA requested but AmberCore was built without the `cuda` feature. \
             Rebuild with `cargo build --release --features cuda` (requires the \
             CUDA toolkit / nvcc + MSVC `cl.exe` on Windows)."
                .into(),
        ))
    }
}

/// Attempt to build the Metal backend. Errors if the `metal` feature is off or
/// this isn't macOS with an Apple GPU.
fn try_metal() -> Result<Box<dyn Backend>> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match MetalBackend::new(0) {
            Ok(b) => Ok(Box::new(b)),
            Err(e) => Err(e),
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        Err(crate::error::Error::Backend(
            "Metal requested but AmberCore was built without Metal support. Build \
             on macOS with `cargo build --release --features metal` (requires an \
             Apple GPU)."
                .into(),
        ))
    }
}

/// Resolve `auto`: try each compiled-in GPU backend in turn (CUDA → Metal),
/// falling back to CPU if none is available.
fn try_auto() -> Result<Box<dyn Backend>> {
    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new(0) {
            Ok(b) => {
                tracing::info!("auto: CUDA available; using CUDA");
                return Ok(Box::new(b));
            }
            Err(e) => tracing::warn!("auto: CUDA unavailable ({e}); trying next backend"),
        }
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match MetalBackend::new(0) {
            Ok(b) => {
                tracing::info!("auto: Metal available; using Metal");
                return Ok(Box::new(b));
            }
            Err(e) => tracing::warn!("auto: Metal unavailable ({e}); trying next backend"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        tracing::info!("auto: built without `cuda` feature");
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        tracing::info!("auto: built without `metal` feature / not on macOS");
    }
    tracing::info!("auto: falling back to CPU");
    Ok(Box::new(CpuBackend::new()))
}

// ─────────────────────────── CUDA backend (feature-gated) ──────────────────────

#[cfg(feature = "cuda")]
mod cuda {
    use super::{Backend, Device};
    use crate::error::{Error, Result};

    /// CUDA compute backend. Requires the `cuda` Cargo feature (build time:
    /// nvcc; runtime: an NVIDIA GPU + driver).
    pub struct CudaBackend {
        ordinal: usize,
        device: Device,
    }

    impl CudaBackend {
        /// Create a CUDA backend for GPU `ordinal` (0 = first GPU).
        ///
        /// Beyond creating the device, this **warms up one kernel immediately**:
        /// candle JIT-compiles its embedded PTX through the driver at first
        /// kernel use, so a driver/toolkit skew (the
        /// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` a too-old driver produces)
        /// surfaces HERE — with the translated, actionable message — instead of
        /// crashing a model load halfway through. Under `auto` the resolver
        /// catches this and falls back to CPU, so the app still runs.
        pub fn new(ordinal: usize) -> Result<Self> {
            // `new_cuda_with_stream`: ONE cudarc-managed stream shared by every
            // thread + cudarc's event tracking. `Device::new_cuda` instead hands
            // each OS thread its own CUDA per-thread default stream — and
            // AmberCore loads models and runs generation on different tokio
            // blocking threads, so tensor alloc/free would cross streams with
            // no ordering and the context dies mid-generation with
            // CUDA_ERROR_INVALID_CONTEXT (seen on RTX 3050, v0.8.4 installers).
            let device = Device::new_cuda_with_stream(ordinal)
                .map_err(|e| Error::Backend(format!("CUDA init (ordinal {ordinal}): {e}")))?;
            warm_up_kernel(&device)
                .map_err(|e| Error::Backend(super::translate_cuda_error(&e.to_string())))?;
            Ok(Self { ordinal, device })
        }
    }

    /// Run one trivial kernel so the PTX modules actually load + JIT now.
    fn warm_up_kernel(device: &Device) -> std::result::Result<(), candle_core::Error> {
        use candle_core::Tensor;
        let a = Tensor::new(&[1.0f32, 2.0], device)?;
        let b = Tensor::new(&[3.0f32, 4.0], device)?;
        let _ = &a + &b;
        Ok(())
    }

    impl Backend for CudaBackend {
        fn name(&self) -> &str {
            "cuda"
        }

        fn device(&self) -> Result<Device> {
            // The device is created once at construction; hand out a reference.
            // candle::Device is cheaply cloneable (it's an enum wrapping an
            // Arc for the CUDA handle).
            Ok(self.device.clone())
        }

        fn gpu_info(&self) -> Option<super::GpuInfo> {
            // Only query the driver when a CUDA device is actually up (candle
            // has run cuInit by then). Uses candle's own cudarc re-export so
            // the driver API is the exact instance candle links against.
            let Device::Cuda(_) = &self.device else {
                return None;
            };
            use candle_core::cuda_backend::cudarc;
            let dev = cudarc::driver::result::device::get(self.ordinal as i32).ok()?;
            let name = cudarc::driver::result::device::get_name(dev)
                .unwrap_or_else(|_| "CUDA device".to_string());
            // Device-level queries (no CUDA context needed). SAFETY: `dev` is
            // a live CUdevice handle returned by cuDeviceGet above.
            let vram_total_mb = unsafe {
                cudarc::driver::result::device::total_mem(dev)
            }
            .ok()
            .map(|b| b as u64 / (1024 * 1024));
            // Used/free VRAM needs a *current* context; UI threads don't have
            // one, so this is best-effort and may read as "—".
            let (vram_free_mb, vram_used_mb) = cudarc::driver::result::mem_get_info()
                .ok()
                .map(|(free, total)| {
                    (
                        Some(free as u64 / (1024 * 1024)),
                        Some((total - free) as u64 / (1024 * 1024)),
                    )
                })
                .unwrap_or((None, None));
            Some(super::GpuInfo {
                name,
                vram_total_mb,
                vram_used_mb,
                vram_free_mb,
            })
        }
    }

    impl std::fmt::Debug for CudaBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CudaBackend")
                .field("ordinal", &self.ordinal)
                .finish()
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

// ─────────────────────────── Metal backend (feature + macOS gated) ────────────

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::{Backend, Device};
    use crate::error::{Error, Result};

    /// Metal compute backend (Apple GPU). Requires the `metal` Cargo feature and
    /// macOS at build/runtime + an Apple GPU. Mirrors [`super::CudaBackend`].
    ///
    /// *Caveat:* candle issue #2818 — a Metal embedding-generation panic on some
    /// shapes. Verify against Qwen2 before declaring it done.
    pub struct MetalBackend {
        ordinal: usize,
        device: Device,
    }

    impl MetalBackend {
        /// Create a Metal backend for GPU `ordinal` (0 = first GPU).
        pub fn new(ordinal: usize) -> Result<Self> {
            let device = Device::new_metal(ordinal)
                .map_err(|e| Error::Backend(format!("Metal init (ordinal {ordinal}): {e}")))?;
            Ok(Self { ordinal, device })
        }
    }

    impl Backend for MetalBackend {
        fn name(&self) -> &str {
            "metal"
        }

        fn device(&self) -> Result<Device> {
            // candle::Device is cheaply cloneable (Arc-backed for Metal).
            Ok(self.device.clone())
        }
    }

    impl std::fmt::Debug for MetalBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MetalBackend")
                .field("ordinal", &self.ordinal)
                .finish()
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::MetalBackend;

// ─────────────────────────── AMD/ROCm stub (feature-gated) ────────────────────
//
// candle has NO stable ROCm backend (only stale, self-described "AI-generated,
// unsafe" PRs #3424/#3801). Rather than track a fragile git fork, AmberCore ships
// a stub that always errors cleanly. The struct + feature exist so the surface
// (DeviceChoice::Amd, `--device amd`, the resolve path) is ready the day upstream
// candle merges official ROCm. Zero runtime cost, no experimental dependency.

/// The error returned for any AMD/ROCm device request — always, regardless of the
/// `rocm` feature, because candle has no stable ROCm backend to wire up.
fn amd_unavailable() -> crate::error::Error {
    crate::error::Error::Backend(
        "AMD/ROCm support is blocked on upstream candle (no stable ROCm backend — \
         see ACRoad.md §7). AmberCore will wire it the day candle merges official \
         ROCm."
            .into(),
    )
}

#[cfg(feature = "rocm")]
mod amd {
    use super::{Backend, Device};
    use crate::error::Result;

    /// AMD/ROCm compute backend — **stub only**. Always errors: candle has no
    /// stable ROCm backend. Present so the `--device amd` surface is wired and
    /// ready for the day upstream candle merges ROCm; drop in real
    /// `Device::new_amd`-style construction here when that lands.
    pub struct AmdBackend {
        _priv: (),
    }

    impl AmdBackend {
        /// Always fails — ROCm is not supported upstream.
        pub fn new(_ordinal: usize) -> Result<Self> {
            Err(super::amd_unavailable())
        }
    }

    impl Backend for AmdBackend {
        fn name(&self) -> &str {
            "amd"
        }

        fn device(&self) -> Result<Device> {
            // Unreachable: `new` always errors, so no device is ever handed out.
            Err(super::amd_unavailable())
        }
    }
}

#[cfg(feature = "rocm")]
pub use amd::AmdBackend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_returns_cpu_device() {
        let b = CpuBackend::new();
        assert_eq!(b.name(), "cpu");
        let d = b.device().unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn resolve_cpu_choice_gives_cpu() {
        let b = resolve_backend(DeviceChoice::Cpu).unwrap();
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn resolve_auto_without_cuda_falls_back_to_cpu() {
        // Without the `cuda` feature compiled in, `auto` → CPU.
        let b = resolve_backend(DeviceChoice::Auto).unwrap();
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn resolve_cuda_without_feature_errors_cleanly() {
        // Without the `cuda` feature, an explicit `cuda` request must error
        // with a helpful message (not panic). Use match (not unwrap_err) so we
        // don't require `Box<dyn Backend>: Debug`.
        match resolve_backend(DeviceChoice::Cuda) {
            Ok(b) => panic!("expected CUDA error, got backend: {}", b.name()),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("cuda"), "msg was: {msg}");
            }
        }
    }

    #[test]
    fn resolve_metal_without_feature_errors_cleanly() {
        // Without the `metal` feature (or off macOS), an explicit `metal`
        // request must error with a helpful message (not panic).
        match resolve_backend(DeviceChoice::Metal) {
            Ok(b) => panic!("expected Metal error, got backend: {}", b.name()),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("metal"), "msg was: {msg}");
            }
        }
    }

    #[test]
    fn resolve_amd_always_errors_cleanly() {
        // AMD/ROCm always errors — candle has no stable ROCm backend. The error
        // must be informative regardless of the (empty) `rocm` feature.
        match resolve_backend(DeviceChoice::Amd) {
            Ok(b) => panic!("expected AMD error, got backend: {}", b.name()),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("ROCm") || msg.contains("AMD"),
                    "msg was: {msg}"
                );
            }
        }
    }

    #[test]
    fn ptx_mismatch_error_gets_actionable_guidance() {
        let msg = translate_cuda_error(
            "DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION, \"the provided PTX was \
             compiled with an unsupported toolchain.\")",
        );
        assert!(msg.contains("update your NVIDIA driver"), "msg was: {msg}");
        assert!(msg.contains("CUDA kernel mismatch"), "msg was: {msg}");
    }

    #[test]
    fn unrelated_cuda_errors_pass_through_unchanged() {
        let original = "some unrelated driver hiccup";
        assert_eq!(translate_cuda_error(original), original);
    }

    #[test]
    fn min_driver_table_covers_the_shipped_toolkits() {
        assert_eq!(min_driver_for("12.8"), "570.51");
        assert_eq!(min_driver_for("13.0"), "580.65");
        // Point releases map to their major line's baseline.
        assert!(min_driver_for("13.3").contains("580.65"));
        assert!(min_driver_for("12.9").contains("570.51"));
        assert!(min_driver_for("42.0").contains("driver release"));
    }
}
