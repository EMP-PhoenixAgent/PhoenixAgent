//! Prometheus telemetry — push completed runs to the Phoenix collector.
//!
//! This is the alpha-test data path. After each `/api/chat` request finishes,
//! the server builds a Prometheus-shaped payload (run id + hardware snapshot +
//! microsecond timeline + derived metrics) and pushes it to
//! `PHOENIX_COLLECTOR_URL` (the phoenix-web collector's `/api/ingest`).
//!
//! ## Non-blocking guarantee
//!
//! All telemetry work happens in a **detached** tokio task spawned *after* the
//! response stream has completed. The inference path and the client response
//! are never gated on the collector. If the env var is unset, `submit` is a
//! no-op. If the push fails, the error is logged once and dropped.
//!
//! ## Timeline events
//!
//! We emit the Prometheus "Chronos" events: `REQ_REC`, `PREP_START`,
//! `KV_CACHE_ALLOC`, `PROMPT_PROC`, `TTFT`, `GEN_STEP`, `RES_FIN`. Timings are
//! captured at boundaries already measured in `chat.rs` / `pipeline/mod.rs`.

use crate::pipeline::GenStats;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Read the collector URL from the environment (set by the operator, e.g.
/// `PHOENIX_COLLECTOR_URL=https://collector.phoenix-agent.io/api/ingest`).
pub fn collector_url() -> Option<String> {
    match std::env::var("PHOENIX_COLLECTOR_URL") {
        Ok(u) if !u.trim().is_empty() => Some(u),
        _ => None,
    }
}

/// One Prometheus timeline event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Microsecond offset from run start.
    pub ts: i64,
    pub event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Hardware snapshot — cached once at startup (sysinfo is heavy to poll).
#[derive(Debug, Clone, Serialize)]
pub struct Hardware {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_used_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_cores: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_total_mb: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

/// Capture the hardware snapshot once. Cheap to clone thereafter.
///
/// `backend` is the AmberCore backend name ("cpu" / "cuda"). VRAM/driver are
/// only populated for the cuda backend (NVML is gated behind the `cuda`
/// feature to avoid a runtime dependency on NVIDIA drivers for CPU builds).
pub fn hardware_snapshot(backend: &str) -> Hardware {
    use sysinfo::System;
    let mut hw = Hardware {
        gpu: None,
        vram_total_mb: None,
        vram_used_mb: None,
        cpu: None,
        cpu_cores: None,
        ram_total_mb: None,
        os: None,
        driver: None,
    };

    let mut sys = System::new_all();
    sys.refresh_cpu_all();
    sys.refresh_memory();

    if let Some(cpu) = sys.cpus().first() {
        hw.cpu = Some(cpu.brand().to_string());
    }
    hw.cpu_cores = Some(sys.cpus().len() as i64);
    hw.ram_total_mb = Some((sys.total_memory() / 1024) as i64);
    hw.os = Some(System::name().unwrap_or_else(|| std::env::consts::OS.to_string()));

    // CUDA: surface the GPU as the backend's compute device. We don't pull in
    // nvml-wrapper here (it adds a runtime NVIDIA-driver dependency even on CPU
    // builds); instead we report the backend as the GPU descriptor when cuda is
    // active. A richer NVML snapshot can be added behind the `cuda` feature
    // later without changing this shape.
    if backend.eq_ignore_ascii_case("cuda") {
        hw.gpu = Some("CUDA device".to_string());
    }

    hw
}

/// Full Prometheus payload submitted to the collector.
#[derive(Debug, Serialize)]
struct Payload {
    run_id: String,
    metadata: PayloadMetadata,
    hardware: Hardware,
    timeline: Vec<TimelineEvent>,
    metrics: PayloadMetrics,
    source: &'static str,
}

#[derive(Debug, Serialize)]
struct PayloadMetadata {
    model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    backend: String,
}

#[derive(Debug, Serialize)]
struct PayloadMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    tft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tbt_avg_ms: Option<f64>,
    tokens_per_sec: f64,
    prompt_tokens: i64,
    output_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_time_ms: Option<f64>,
}

/// Build the Prometheus timeline from the boundary instants captured in
/// `chat.rs`. All offsets are microseconds relative to `req_start`.
fn build_timeline(
    req_start: Instant,
    prefill_start: Instant,
    prefill_done: Instant,
    first_token: Instant,
    gen_steps: &[(Instant, usize)],
    finished: Instant,
    prompt_tokens: usize,
) -> Vec<TimelineEvent> {
    let us = |t: Instant| t.saturating_duration_since(req_start).as_micros() as i64;
    let mut tl = Vec::with_capacity(7 + gen_steps.len());

    tl.push(TimelineEvent {
        ts: 0,
        event: "REQ_REC",
        data: Some(serde_json::json!({ "prompt_len": prompt_tokens })),
    });
    tl.push(TimelineEvent { ts: us(prefill_start), event: "PREP_START", data: None });
    tl.push(TimelineEvent { ts: us(prefill_start), event: "KV_CACHE_ALLOC", data: None });
    tl.push(TimelineEvent {
        ts: us(prefill_done),
        event: "PROMPT_PROC",
        data: Some(serde_json::json!({ "tokens": prompt_tokens })),
    });
    tl.push(TimelineEvent { ts: us(first_token), event: "TTFT", data: None });
    // GEN_STEP every N tokens (caller already subsampled). Each carries its
    // absolute token index.
    for (t, idx) in gen_steps {
        tl.push(TimelineEvent {
            ts: us(*t),
            event: "GEN_STEP",
            data: Some(serde_json::json!({ "token_idx": idx })),
        });
    }
    tl.push(TimelineEvent { ts: us(finished), event: "RES_FIN", data: None });
    tl
}

/// Arguments needed to assemble + push a run's telemetry.
///
/// `prefill_done` is when the prefill forward pass returned (== TTFT minus
/// sampling, approximately). `gen_steps` is an already-subsampled list of
/// `(instant, token_index)` pairs — one every N tokens — to keep payloads small.
pub struct TelemetryReport<'a> {
    pub model_name: &'a str,
    pub quantization: Option<&'a str>,
    pub backend: &'a str,
    pub stats: &'a GenStats,
    pub req_start: Instant,
    pub prefill_start: Instant,
    pub prefill_done: Instant,
    pub first_token: Instant,
    pub gen_steps: &'a [(Instant, usize)],
    pub finished: Instant,
}

/// Build the payload + push it to the collector in a detached task. Returns
/// immediately. No-op if `PHOENIX_COLLECTOR_URL` is unset.
pub fn submit(report: TelemetryReport<'_>, hardware: Arc<Hardware>) {
    let url = match collector_url() {
        Some(u) => u,
        None => return, // telemetry disabled — common in dev
    };

    let timeline = build_timeline(
        report.req_start,
        report.prefill_start,
        report.prefill_done,
        report.first_token,
        report.gen_steps,
        report.finished,
        report.stats.prompt_tokens,
    );

    let total_time_ms = Some(report.finished.saturating_duration_since(report.req_start).as_secs_f64() * 1000.0);

    let payload = Payload {
        run_id: uuid::Uuid::new_v4().to_string(),
        metadata: PayloadMetadata {
            model_name: report.model_name.to_string(),
            quantization: report.quantization.map(|s| s.to_string()),
            backend: report.backend.to_string(),
        },
        hardware: (*hardware).clone(),
        timeline,
        metrics: PayloadMetrics {
            tft_ms: report.stats.ttft_ms,
            tbt_avg_ms: report.stats.tbt_avg_ms(),
            tokens_per_sec: report.stats.tokens_per_sec(),
            prompt_tokens: report.stats.prompt_tokens as i64,
            output_tokens: report.stats.output_tokens as i64,
            total_time_ms,
        },
        source: "ambercore",
    };

    // Fire-and-forget. This runs AFTER the response stream finished — it never
    // blocks inference or the client.
    tokio::spawn(async move {
        let client = match reqwest::Client::builder().timeout(std::time::Duration::from_secs(10)).build() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "telemetry: failed to build http client");
                return;
            }
        };
        match client.post(&url).json(&payload).send().await {
            Ok(r) if r.status().is_success() => {
                tracing::debug!(run_id = %payload.run_id, "telemetry: pushed run to collector");
            }
            Ok(r) => {
                tracing::warn!(
                    run_id = %payload.run_id,
                    status = %r.status(),
                    "telemetry: collector rejected run"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "telemetry: push failed (collector unreachable)");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn build_timeline_orders_events() {
        let req_start = Instant::now();
        let prefill_start = req_start;
        let prefill_done = req_start + Duration::from_micros(400);
        let first_token = req_start + Duration::from_micros(500);
        let finished = req_start + Duration::from_micros(5000);
        let tl = build_timeline(
            req_start,
            prefill_start,
            prefill_done,
            first_token,
            &[(req_start + Duration::from_micros(2000), 10)],
            finished,
            50,
        );
        // First event is REQ_REC at 0, last is RES_FIN.
        assert_eq!(tl.first().unwrap().event, "REQ_REC");
        assert_eq!(tl.last().unwrap().event, "RES_FIN");
        // Timestamps are monotonically non-decreasing.
        let mut prev = 0;
        for ev in &tl {
            assert!(ev.ts >= prev, "timeline not monotonic: {} < {}", ev.ts, prev);
            prev = ev.ts;
        }
        // TTFT is present.
        assert!(tl.iter().any(|e| e.event == "TTFT"));
    }

    #[test]
    fn hardware_snapshot_populates_cpu_ram() {
        let hw = hardware_snapshot("cpu");
        assert!(hw.cpu.is_some(), "cpu brand should be populated");
        assert!(hw.cpu_cores.unwrap_or(0) > 0, "should report >0 cores");
        assert!(hw.ram_total_mb.unwrap_or(0) > 0, "should report >0 RAM");
        assert!(hw.gpu.is_none(), "cpu backend has no gpu");
    }
}
