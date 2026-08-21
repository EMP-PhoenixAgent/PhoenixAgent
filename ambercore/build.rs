//! Build script — records the CUDA toolkit version used to compile the PTX
//! kernels (only under the `cuda` feature) so runtime driver-mismatch errors
//! can tell the user exactly which driver version they need.
//!
//! candle embeds its GPU kernels as PTX and JIT-compiles them through the
//! *installed NVIDIA driver* at runtime. A driver older than the build
//! toolkit rejects that PTX with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` — the
//! message built here lets us say "compiled with CUDA 12.8 → driver ≥ 570.51"
//! instead of leaving the user with a cryptic error.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if let Some(ver) = detect_nvcc_version() {
            println!("cargo:rustc-env=AMBERCORE_CUDA_TOOLKIT={ver}");
            println!("cargo:warning=AmberCore CUDA kernels: built with toolkit {ver}");
        } else {
            println!("cargo:warning=AmberCore CUDA build: nvcc not found on PATH — the runtime driver-mismatch hint will not name a toolkit version");
        }
    }
}

/// Run `nvcc --version` and pull the `release X.Y` version out of its banner:
/// `Cuda compilation tools, release 12.8, V12.8.93` → `12.8`.
fn detect_nvcc_version() -> Option<String> {
    let out = std::process::Command::new("nvcc").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().rev().find_map(|line| {
        let idx = line.find("release ")?;
        let rest = &line[idx + "release ".len()..];
        let end = rest.find([',', ' ']).unwrap_or(rest.len());
        let ver = &rest[..end];
        // Sanity: must look like major.minor digits.
        ver.split_once('.').filter(|(a, b)| {
            !a.is_empty() && !b.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && b.bytes().all(|c| c.is_ascii_digit())
        })
        .map(|_| ver.to_string())
    })
}
