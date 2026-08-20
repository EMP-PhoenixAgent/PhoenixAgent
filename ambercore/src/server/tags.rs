//! `GET /api/tags` handler.
//!
//! Lists every tag in the catalog as `{"models":[{"name":<tag>},...]}`. This is
//! Phoenix's sole health probe + model-availability signal, so it must return
//! HTTP 200 with the catalog's tags verbatim.

use crate::server::protocol::{TagEntry, TagsResponse};
use crate::server::ServerState;
use axum::extract::State;
use axum::Json;

/// Build the `/api/tags` response from a catalog's tag list.
pub fn tags_response(tags: Vec<String>) -> TagsResponse {
    TagsResponse {
        models: tags.into_iter().map(|name| TagEntry { name }).collect(),
    }
}

/// `GET /api/tags` — live catalog.
pub async fn list(State(state): State<ServerState>) -> Json<TagsResponse> {
    let tags = state.tags().await;
    Json(tags_response(tags))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_response_shape() {
        let resp = tags_response(vec!["qwen2.5-coder:7b".into(), "llama3:8b".into()]);
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(
            json,
            r#"{"models":[{"name":"qwen2.5-coder:7b"},{"name":"llama3:8b"}]}"#
        );
    }

    #[test]
    fn tags_response_empty_is_valid() {
        // Phoenix tolerates an empty models array — server still "up".
        let resp = tags_response(vec![]);
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, r#"{"models":[]}"#);
    }
}
