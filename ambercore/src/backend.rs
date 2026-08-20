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

/// A compute target. Implementations hand out the candle [`Device`] that models
/// and tensors live on, and own any device-specific resources or limits.
pub trait Backend: Send + Sync {
    /// Human-readable backend name, e.g. `"cpu"` or `"cuda:0"`.
    fn name(&self) -> &str;

    /// The candle device tensors should be placed on for this backend.
    fn device(&self) -> Result<Device>;
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
        pub fn new(ordinal: usize) -> Result<Self> {
            let device = Device::new_cuda(ordinal)
                .map_err(|e| Error::Backend(format!("CUDA init (ordinal {ordinal}): {e}")))?;
            Ok(Self { ordinal, device })
        }
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
}
