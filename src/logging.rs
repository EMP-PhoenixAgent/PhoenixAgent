//! Logging initialization (tracing → rolling file in the data dir).

use std::path::Path;
use tracing_appender::rolling;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize tracing to write a rotating log file under `logs_dir`.
///
/// Returns a guard that must be held for the lifetime of the program to keep
/// the file flush working.
pub fn init(logs_dir: &Path) -> tracing_appender::non_blocking::WorkerGuard {
    std::fs::create_dir_all(logs_dir).ok();
    let file_appender = rolling::daily(logs_dir, "phoenix.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let env = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(env)
        .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
        .init();
    guard
}
