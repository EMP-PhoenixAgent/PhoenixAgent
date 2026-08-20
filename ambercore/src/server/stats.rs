//! `GET /api/stats` handler.
//!
//! Returns live server throughput so a client (e.g. Phoenix's health bar) can
//! display tokens/second. AmberCore records the tok/s at the end of each
//! `/api/chat` generation; this endpoint exposes the last measured value.

use crate::server::ServerState;
use axum::extract::State;
use axum::Json;
use serde::Serialize;

/// Response body for `GET /api/stats`.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    /// Last measured generation throughput in tokens/second. `null` until the
    /// first generation completes.
    pub tokens_per_sec: Option<f64>,
    /// The compute backend name (e.g. "cpu", "cuda").
    pub backend: String,
}

/// `GET /api/stats`.
pub async fn stats(State(state): State<ServerState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        tokens_per_sec: state.last_tokens_per_sec().await,
        backend: state.backend_name().to_string(),
    })
}
