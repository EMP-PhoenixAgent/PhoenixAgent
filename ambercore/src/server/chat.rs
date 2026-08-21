//! Chat generation — the shared in-process core + the `POST /api/chat` handler.
//!
//! The real generation logic lives in [`generate_events`], which streams
//! semantic [`GenEvent`]s (reasoning / content / tool calls / done). It has two
//! consumers:
//!
//! - the **HTTP handler** ([`chat`]) maps events to the Ollama-compatible NDJSON
//!   [`ChatChunk`] wire format, and
//! - **Phoenix Agent's embedded provider** maps them directly to its own
//!   `ChatEvent`s in-process — same engine, no HTTP hop.
//!
//! ## Flow (inside `generate_events`)
//!
//! 1. Resolve the model tag → [`LoadedEntry`] (load + cache on first hit).
//! 2. Map messages → Qwen ChatML template → prompt string (tools rendered into
//!    the system prompt in Hermes format).
//! 3. Spawn generation on `spawn_blocking`. The `on_token` callback routes
//!    Qwen3 `<think>` reasoning vs answer text through the [`ThinkSplitter`].
//! 4. With tools active, the full text is buffered and post-processed: the think
//!    block is split out and Hermes `<tool_call>` markers parsed into structured
//!    calls. The terminal event carries the token counts.
//! 5. Prometheus telemetry + throughput recording run in detached tasks after
//!    the stream completes (never on the inference path).

use crate::error::{Error, Result};
use crate::pipeline::{Pipeline, SampleParams, StopCondition};
use crate::server::protocol::{ChatChunk, ChatRequest, ToolCall};
use crate::server::ServerState;
use crate::tokenizer::{format_chat_prompt, pick_template, ChatTurn, Role};
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;

/// Map a Phoenix `role` string to our internal [`Role`].
fn map_role(role: &str) -> Option<Role> {
    match role {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        // `tool` messages carry tool results; M2 renders them as a user-turn
        // observation so the model can continue. M3 (tool-calling) will handle
        // these explicitly.
        _ => None,
    }
}

/// Convert Phoenix chat messages into ChatML turns.
///
/// `tool`-role messages (tool results) are folded into a synthetic user turn
/// labelled with the tool name, so the model sees the result as context.
pub fn to_chat_turns(messages: &[crate::server::protocol::ChatMessage]) -> Vec<ChatTurn> {
    let mut turns = Vec::with_capacity(messages.len());
    for m in messages {
        match map_role(&m.role) {
            Some(role) => turns.push(ChatTurn {
                role,
                content: m.content.clone(),
            }),
            None => {
                // role == "tool": render as a user observation.
                let label = m.tool.as_deref().unwrap_or("tool");
                turns.push(ChatTurn {
                    role: Role::User,
                    content: format!("[tool result from {label}]\n{}", m.content),
                });
            }
        }
    }
    turns
}

// ─────────────────────── semantic generation events ────────────────────────

/// One semantic step of a generation, streamed by [`generate_events`].
///
/// Backend-agnostic: the HTTP handler serializes these to NDJSON `ChatChunk`s,
/// and Phoenix's embedded provider maps them straight to its `ChatEvent`s.
/// The stream ends after `Done` (success) or `Error` (failure).
#[derive(Debug, Clone)]
pub enum GenEvent {
    /// A piece of the model's reasoning / "thinking" (Qwen3 `<think>` content,
    /// already split out of the answer text).
    Reasoning(String),
    /// A piece of the visible answer text.
    Content(String),
    /// The structured tool calls the model requested (emitted once, just before
    /// `Done`, only when non-empty).
    ToolCalls(Vec<ToolCall>),
    /// Terminal success event with the token counts.
    Done { prompt_tokens: i64, output_tokens: i64 },
    /// Terminal failure event. No `Done` follows.
    Error(String),
}

/// Run one chat generation against the engine, streaming [`GenEvent`]s.
///
/// This is the whole generation pipeline minus any transport: the HTTP handler
/// and Phoenix's in-process provider both drive the engine through it. Model
/// replicas are acquired from the shared pool (built lazily on first use), and
/// the telemetry/throughput bookkeeping runs exactly as it does over HTTP.
pub async fn generate_events(
    state: &ServerState,
    req: &ChatRequest,
) -> Result<tokio::sync::mpsc::Receiver<GenEvent>> {
    // 1. Resolve the model (acquire a replica — load + cache on the cold path,
    //    queue fairly if the pool is at capacity).
    let handle = state.acquire_replica(&req.model).await?;

    // 2. Map messages → chat turns → prompt string, formatted with the
    //    template the model's architecture was trained on (ChatML for Qwen,
    //    Gemma/Phi3/GLM4/Llama3/Mistral for their families — `llama` is
    //    disambiguated from the tokenizer's special tokens).
    //    If tools were supplied, render them into the system prompt (Hermes
    //    format) so the model knows how to emit calls.
    let turns = to_chat_turns(&req.messages);
    if turns.is_empty() {
        return Err(Error::InvalidInput("chat request has no messages".into()));
    }
    let template = {
        let replica = handle.replica().lock().unwrap();
        pick_template(&replica.arch, &replica.tokenizer)
    };
    let tools_section = crate::server::tools::render_tools_section(&req.tools);
    let system = if tools_section.is_empty() {
        None
    } else {
        Some(tools_section.leak() as &'static str)
    };
    let prompt = format_chat_prompt(template, &turns, system);
    let has_tools = !req.tools.is_empty();

    // 3. Sampler params + stop condition. No max-tokens cap — AmberCore is fully
    //    local, so generation runs until the model's natural EOS / `<|im_end|>`
    //    (the real end-of-turn signal).
    let temperature = req.temperature.unwrap_or(0.8);
    let params = SampleParams {
        temperature,
        ..Default::default()
    };
    let stop_tokens = handle.replica().lock().unwrap().stop_tokens.clone();
    let stop = StopCondition {
        max_tokens: None,
        stop_tokens,
    };

    // Identity for the telemetry payload (all event timestamps are µs offsets
    // from `req_start`).
    let req_start = std::time::Instant::now();
    let model_name = req.model.clone();
    let backend_name = state.backend_name().to_string();
    let hardware = state.hardware();
    // Quantization isn't a first-class field in AmberCore yet (it's implicit in
    // the GGUF tensor data), so we don't report it. The collector stores null.
    let quantization: Option<&str> = None;

    let (tx, rx) = tokio::sync::mpsc::channel::<GenEvent>(32);
    let state_for_stats = state.clone();

    // The replica is held for the whole generation; `handle` releases it back to
    // the pool (or hands it to the next waiter) when dropped at the end of the
    // blocking task. `replica` is a clone of the Arc so we can split the borrow.
    let replica = handle.replica().clone();

    tokio::task::spawn_blocking(move || {
        // `_handle` is declared first → drops LAST, so the entry mutex is
        // unlocked before the replica is returned to the pool.
        let _handle = handle;
        let mut entry_guard = replica.lock().unwrap();

        // Split the borrow: bind the tokenizer ref first so the mutable model
        // borrow doesn't conflict with it.
        let entry: &mut crate::server::LoadedEntry = &mut entry_guard;
        let tokenizer = &entry.tokenizer;
        let model: &mut dyn crate::model::DynModel = entry.model.as_mut();
        let device = &entry.device;
        let context_length = entry.context_length;
        let mut pipeline = Pipeline {
            model,
            tokenizer,
            device,
            context_length,
        };

        // When tools are active we buffer all text (the model may emit
        // `<tool_call>` blocks that must not leak as content), then
        // post-process. `Rc<RefCell<...>>` is safe here — this closure runs on
        // one blocking thread; no cross-thread sharing.
        let full_text = std::rc::Rc::new(std::cell::RefCell::new(String::new()));

        // Streaming `<think>` splitter: routes Qwen3 reasoning into `Reasoning`
        // events so consumers can render a collapsible thinking block.
        let splitter = std::rc::Rc::new(std::cell::RefCell::new(ThinkSplitter::new()));

        // Telemetry timeline capture: first-token instant + a subsampled list
        // of (instant, token_index) GEN_STEP markers (one every 25 tokens).
        let first_token = std::rc::Rc::new(std::cell::RefCell::new(None::<std::time::Instant>));
        let gen_steps = std::rc::Rc::new(std::cell::RefCell::new(Vec::<(std::time::Instant, usize)>::new()));
        let token_count = std::rc::Rc::new(std::cell::RefCell::new(0usize));

        let result = {
            let tx_for_events = tx.clone();
            let full_text_inner = full_text.clone();
            let first_token_inner = first_token.clone();
            let gen_steps_inner = gen_steps.clone();
            let token_count_inner = token_count.clone();
            let splitter_inner = splitter.clone();
            pipeline.generate(&prompt, &params, &stop, move |delta| {
                full_text_inner.borrow_mut().push_str(&delta);

                // Telemetry: stamp the first-token instant, and record a
                // GEN_STEP every 25 tokens. `on_token` fires per *text delta*
                // (not per token), so we count deltas as a proxy for token
                // boundaries — exact enough for the timeline's coarse grain.
                let now = std::time::Instant::now();
                {
                    let mut ft = first_token_inner.borrow_mut();
                    if ft.is_none() {
                        *ft = Some(now);
                    }
                }
                let mut count = token_count_inner.borrow_mut();
                *count += 1;
                if *count % 25 == 0 {
                    gen_steps_inner.borrow_mut().push((now, *count));
                }

                if has_tools {
                    // With tools active, DON'T stream during generation — the
                    // model may emit `<tool_call>` blocks mid-stream that are
                    // only safely filtered after the fact (below).
                    return;
                }

                // No tools: split Qwen3 `<think>` reasoning from answer content
                // and stream each piece as the right event.
                for piece in splitter_inner_feed(&splitter_inner, &delta) {
                    let ev = match piece {
                        Piece::Think(t) => GenEvent::Reasoning(t),
                        Piece::Content(t) => GenEvent::Content(t),
                    };
                    // Send failures (consumer gone) are ignored — generation
                    // completes regardless, matching the HTTP behavior.
                    let _ = tx_for_events.blocking_send(ev);
                }
            })
        };

        // Flush any think/content text left buffered in the splitter (the final
        // delta often ends mid-classification with a partial marker held back).
        if !has_tools {
            for piece in splitter.borrow_mut().finish() {
                let ev = match piece {
                    Piece::Think(t) => GenEvent::Reasoning(t),
                    Piece::Content(t) => GenEvent::Content(t),
                };
                let _ = tx.blocking_send(ev);
            }
        }

        // With tools active: split the think block out, parse the Hermes
        // `<tool_call>` markers into structured calls, and stream the clean
        // content in one shot.
        let tool_calls = if has_tools {
            let (think_text, content_text) = split_think(&full_text.borrow());
            if !think_text.is_empty() {
                let _ = tx.blocking_send(GenEvent::Reasoning(think_text));
            }
            let parsed = crate::server::tools::parse_tool_calls(&content_text);
            if !parsed.remaining_text.is_empty() {
                let _ = tx.blocking_send(GenEvent::Content(parsed.remaining_text));
            }
            parsed.tool_calls
        } else {
            Vec::new()
        };

        // Terminal event + bookkeeping.
        let finished = std::time::Instant::now();
        match &result {
            Ok((stats, _)) => {
                // Record throughput for /api/stats (async mutex → spawn a task).
                let tps = stats.tokens_per_sec();
                let s = state_for_stats.clone();
                tokio::task::spawn(async move {
                    s.record_tokens_per_sec(tps).await;
                });

                // Prometheus telemetry push (fire-and-forget, detached — never
                // blocks the stream). We reconstruct prefill_done from
                // req_start + prefill_secs; first_token + gen_steps were
                // captured in the on_token closure above.
                let first_token_inst = *first_token.borrow();
                let gen_steps_vec = gen_steps.borrow().clone();
                let prefill_done = req_start + std::time::Duration::from_secs_f64(stats.prefill_secs);
                let report = crate::server::telemetry::TelemetryReport {
                    model_name: &model_name,
                    quantization,
                    backend: &backend_name,
                    stats,
                    req_start,
                    prefill_start: req_start,
                    prefill_done,
                    first_token: first_token_inst.unwrap_or(finished),
                    gen_steps: &gen_steps_vec,
                    finished,
                };
                crate::server::telemetry::submit(report, hardware.clone());

                if !tool_calls.is_empty() {
                    let _ = tx.blocking_send(GenEvent::ToolCalls(tool_calls));
                }
                let _ = tx.blocking_send(GenEvent::Done {
                    prompt_tokens: stats.prompt_tokens as i64,
                    output_tokens: stats.output_tokens as i64,
                });
            }
            Err(e) => {
                tracing::warn!(error = %e, "generation failed mid-stream");
                let _ = tx.blocking_send(GenEvent::Error(e.to_string()));
            }
        }
    });

    Ok(rx)
}

/// Helper so the `on_token` closure can feed the shared splitter without
/// borrowing it across the closure boundary twice.
fn splitter_inner_feed(
    splitter: &std::rc::Rc<std::cell::RefCell<ThinkSplitter>>,
    delta: &str,
) -> Vec<Piece> {
    splitter.borrow_mut().feed(delta)
}

// ─────────────────────────── HTTP handler ───────────────────────────────────

/// `POST /api/chat` — streaming NDJSON. Drives [`generate_events`] and maps each
/// [`GenEvent`] to the Ollama-compatible wire format.
pub async fn chat(
    State(state): State<ServerState>,
    axum::Json(req): axum::Json<ChatRequest>,
) -> Response {
    match chat_inner(state, req).await {
        Ok(resp) => resp,
        Err(e) => {
            let msg = format!("ambercore error: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
        }
    }
}

async fn chat_inner(state: ServerState, req: ChatRequest) -> Result<Response> {
    let mut ev_rx = generate_events(&state, &req).await?;

    // Map semantic events → NDJSON lines. Tool calls ride on the terminal
    // chunk per the wire contract.
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<Bytes, String>>(32);
    tokio::spawn(async move {
        let mut pending_tools: Option<Vec<ToolCall>> = None;
        while let Some(ev) = ev_rx.recv().await {
            let Some(line) = event_to_ndjson_line(ev, &mut pending_tools) else {
                continue;
            };
            if tx.send(Ok(Bytes::from(format!("{line}\n")))).await.is_err() {
                break; // client went away
            }
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .map_err(|e| Error::Server(format!("build response: {e}")))?)
}

/// Map one [`GenEvent`] to one NDJSON line (or `None` for the buffered
/// tool-calls event, which merges into the terminal chunk). Pure — unit-tested.
fn event_to_ndjson_line(ev: GenEvent, pending_tools: &mut Option<Vec<ToolCall>>) -> Option<String> {
    match ev {
        GenEvent::Reasoning(t) => Some(ChatChunk::think(t).to_ndjson().unwrap_or_default()),
        GenEvent::Content(t) => Some(ChatChunk::delta(t).to_ndjson().unwrap_or_default()),
        GenEvent::ToolCalls(tc) => {
            *pending_tools = Some(tc);
            None
        }
        GenEvent::Done { prompt_tokens, output_tokens } => Some(
            ChatChunk::done(
                pending_tools.take().filter(|t| !t.is_empty()),
                output_tokens,
                prompt_tokens,
            )
            .to_ndjson()
            .unwrap_or_else(|_| "{\"done\":true}".into()),
        ),
        GenEvent::Error(e) => {
            tracing::warn!("generation failed mid-stream: {e}");
            Some(
                ChatChunk::done(None, 0, 0)
                    .to_ndjson()
                    .unwrap_or_else(|_| "{\"done\":true}".into()),
            )
        }
    }
}

// ─────────────────────── <think> streaming splitter ────────────────────────
//
// Qwen3-style models emit an optional `<think>…</think>` reasoning block (often
// at the start of their output). This splitter classifies streamed text deltas
// into reasoning (`Think`) vs answer (`Content`) pieces, robust to the
// `<think>` / `</think>` markers being split across delta boundaries.

enum Piece {
    Think(String),
    Content(String),
}

struct ThinkSplitter {
    in_think: bool,
    /// Unprocessed tail that may still be part of a partial marker.
    tail: String,
}

impl ThinkSplitter {
    fn new() -> Self {
        Self { in_think: false, tail: String::new() }
    }

    fn feed(&mut self, delta: &str) -> Vec<Piece> {
        self.tail.push_str(delta);
        let mut out = Vec::new();
        loop {
            let marker = if self.in_think { "</think>" } else { "<think>" };
            // A full marker is present → emit the text before it on the current
            // channel, consume the marker, and flip think↔content.
            if let Some(idx) = self.tail.find(marker) {
                if idx > 0 {
                    let before: String = self.tail.drain(..idx).collect();
                    out.push(if self.in_think { Piece::Think(before) } else { Piece::Content(before) });
                }
                self.tail.drain(..marker.len());
                self.in_think = !self.in_think;
                continue;
            }
            // No full marker yet. Emit everything except a trailing suffix that
            // could be the start of a marker split across deltas (hold it back).
            let safe = safe_emit_len(&self.tail, marker);
            if safe > 0 {
                let piece: String = self.tail.drain(..safe).collect();
                out.push(if self.in_think { Piece::Think(piece) } else { Piece::Content(piece) });
            }
            break;
        }
        out
    }

    /// Drain any buffered tail at end of stream on its current channel.
    fn finish(&mut self) -> Vec<Piece> {
        if self.tail.is_empty() {
            Vec::new()
        } else {
            let s = std::mem::take(&mut self.tail);
            vec![if self.in_think { Piece::Think(s) } else { Piece::Content(s) }]
        }
    }
}

/// Largest byte length emittable now: all of `tail` except the longest trailing
/// suffix that is a *proper* prefix of `marker`. Markers are ASCII, so any
/// matching suffix is ASCII and the split point is a valid UTF-8 boundary.
fn safe_emit_len(tail: &str, marker: &str) -> usize {
    let tb = tail.as_bytes();
    let mb = marker.as_bytes();
    let max_k = tb.len().min(mb.len().saturating_sub(1));
    let mut k = 0;
    for cand in (1..=max_k).rev() {
        if &tb[tb.len() - cand..] == &mb[..cand] {
            k = cand;
            break;
        }
    }
    tb.len() - k
}

/// One-shot split of a complete text into `(reasoning, content)`.
fn split_think(text: &str) -> (String, String) {
    let mut sp = ThinkSplitter::new();
    let mut think = String::new();
    let mut content = String::new();
    for p in sp.feed(text) {
        match p {
            Piece::Think(s) => think.push_str(&s),
            Piece::Content(s) => content.push_str(&s),
        }
    }
    for p in sp.finish() {
        match p {
            Piece::Think(s) => think.push_str(&s),
            Piece::Content(s) => content.push_str(&s),
        }
    }
    (think, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::protocol::ChatMessage;

    #[test]
    fn to_chat_turns_maps_standard_roles() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "be helpful".into(),
                tool_calls: None,
                tool: None,
            },
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                tool_calls: None,
                tool: None,
            },
        ];
        let turns = to_chat_turns(&msgs);
        assert_eq!(turns.len(), 2);
        assert!(matches!(turns[0].role, Role::System));
        assert!(matches!(turns[1].role, Role::User));
    }

    #[test]
    fn to_chat_turns_folds_tool_results_as_user() {
        let msgs = vec![ChatMessage {
            role: "tool".into(),
            content: "ls output here".into(),
            tool_calls: None,
            tool: Some("list_dir".into()),
        }];
        let turns = to_chat_turns(&msgs);
        assert_eq!(turns.len(), 1);
        assert!(matches!(turns[0].role, Role::User));
        assert!(turns[0].content.contains("list_dir"));
        assert!(turns[0].content.contains("ls output here"));
    }

    #[test]
    fn split_think_extracts_leading_block() {
        let (think, content) = split_think("<think>reasoning here</think>the answer");
        assert_eq!(think, "reasoning here");
        assert_eq!(content, "the answer");
    }

    #[test]
    fn split_think_no_block_is_all_content() {
        let (think, content) = split_think("just an answer");
        assert!(think.is_empty());
        assert_eq!(content, "just an answer");
    }

    #[test]
    fn split_think_marker_split_across_deltas() {
        // Feed the text one char at a time to stress marker-boundary handling.
        let mut sp = ThinkSplitter::new();
        let mut think = String::new();
        let mut content = String::new();
        for ch in "<think>hi</think>yo".chars() {
            for p in sp.feed(&ch.to_string()) {
                match p {
                    Piece::Think(s) => think.push_str(&s),
                    Piece::Content(s) => content.push_str(&s),
                }
            }
        }
        for p in sp.finish() {
            match p {
                Piece::Think(s) => think.push_str(&s),
                Piece::Content(s) => content.push_str(&s),
            }
        }
        assert_eq!(think, "hi");
        assert_eq!(content, "yo");
    }

    #[test]
    fn split_think_literal_angle_bracket_is_content() {
        // A "<" that isn't part of a marker must stream as content, not hang.
        let (think, content) = split_think("< not a tag > done");
        assert!(think.is_empty());
        assert_eq!(content, "< not a tag > done");
    }

    #[test]
    fn event_to_ndjson_maps_semantics_to_wire() {
        // Reasoning → message.think, Content → message.content.
        let mut pending = None;
        let think_line = event_to_ndjson_line(GenEvent::Reasoning("why".into()), &mut pending).unwrap();
        assert!(think_line.contains("\"think\":\"why\""));
        let content_line = event_to_ndjson_line(GenEvent::Content("ans".into()), &mut pending).unwrap();
        assert!(content_line.contains("\"content\":\"ans\""));

        // ToolCalls buffers onto the terminal Done chunk, per the wire contract.
        let call = ToolCall {
            id: None,
            function: crate::server::protocol::FunctionRef {
                name: "read_file".into(),
                arguments: "{}".into(),
            },
        };
        assert!(event_to_ndjson_line(GenEvent::ToolCalls(vec![call]), &mut pending).is_none());
        let done_line = event_to_ndjson_line(
            GenEvent::Done { prompt_tokens: 12, output_tokens: 5 },
            &mut pending,
        )
        .unwrap();
        assert!(done_line.contains("\"done\":true"));
        assert!(done_line.contains("\"eval_count\":5"));
        assert!(done_line.contains("\"prompt_eval_count\":12"));
        assert!(done_line.contains("read_file"));

        // Error → a clean terminal chunk with zero counts.
        let err_line = event_to_ndjson_line(GenEvent::Error("boom".into()), &mut pending).unwrap();
        assert!(err_line.contains("\"done\":true"));
        assert!(!err_line.contains("tool_calls"));
    }
}
