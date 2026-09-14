//! Tokenizer wrapper.
//!
//! Thin wrapper around the HuggingFace [`tokenizers`] crate. AmberCore loads a
//! tokenizer from a sibling `tokenizer.json` (HF format) next to the GGUF file.
//! This is the canonical arrangement: the GGUF carries the weights; the HF
//! `tokenizer.json` carries the BPE merges + special tokens.
//!
//! For M0 the tokenizer must be supplied alongside the model. A future milestone
//! can reconstruct it from the GGUF's inline `tokenizer.ggml.*` metadata.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Encoded text: token ids.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub ids: Vec<u32>,
}

/// A loaded tokenizer. Owns the underlying HF tokenizer.
#[derive(Debug)]
pub struct TokenizerWrapper {
    pub inner: Tokenizer,
    /// Vocab size (best-effort; from the tokenizer's vocab).
    pub vocab_size: usize,
}

impl TokenizerWrapper {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn load(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| Error::Tokenizer(format!("load {}: {e}", path.display())))?;
        let vocab_size = inner.get_vocab_size(false);
        Ok(Self { inner, vocab_size })
    }

    /// Resolve the tokenizer path for a given GGUF. Tries, in order:
    /// 1. `<model_stem>.tokenizer.json` — model-specific (lets multiple models
    ///    with different vocabularies share one directory).
    /// 2. `tokenizer.json` — the generic fallback.
    /// Returns the first match.
    pub fn resolve_next_to(gguf_path: &Path) -> Option<PathBuf> {
        let dir = gguf_path.parent()?;
        // 1. Model-specific: e.g. `qwen3-8b-q4_k_m.tokenizer.json`.
        if let Some(stem) = gguf_path.file_stem().and_then(|s| s.to_str()) {
            let specific = dir.join(format!("{stem}.tokenizer.json"));
            if specific.is_file() {
                return Some(specific);
            }
        }
        // 2. Generic fallback.
        let generic = dir.join("tokenizer.json");
        if generic.is_file() {
            Some(generic)
        } else {
            None
        }
    }

    /// Load the tokenizer for a GGUF, using [`resolve_next_to`]'s lookup order.
    pub fn load_next_to(gguf_path: &Path) -> Result<Self> {
        let path = Self::resolve_next_to(gguf_path).ok_or_else(|| {
            Error::Tokenizer(format!(
                "no tokenizer found for {} (place a `<stem>.tokenizer.json` or \
                 `tokenizer.json` next to it)",
                gguf_path.display()
            ))
        })?;
        Self::load(&path)
    }

    /// Encode a text prompt into token ids.
    pub fn encode(&self, text: &str) -> Result<Encoded> {
        let enc = self
            .inner
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(format!("encode: {e}")))?;
        Ok(Encoded {
            ids: enc.get_ids().to_vec(),
        })
    }

    /// Decode a slice of token ids back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(format!("decode: {e}")))
    }

    /// Look up a special-token id by string (e.g. `"<|endoftext|>"`).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// A streaming token decoder.
///
/// Solves the BPE partial-token problem: when generating token-by-token, a
/// single new token may *change the decoding of earlier tokens* (because BPE
/// merges span token boundaries). To stream safely, we accumulate tokens and
/// only emit the stable prefix — the text that won't change as more tokens
/// arrive. The final [`decode_rest`](Self::decode_rest) flushes whatever
/// remains once generation stops.
///
/// Inlined from candle's `TokenOutputStream` (avoiding a `candle-examples`
/// dependency) — same logic, fewer allocations.
pub struct StreamingDecoder {
    tokens: Vec<u32>,
    /// Index into `tokens` of the first byte of the text we've already emitted.
    prev_index: usize,
    /// Index into `tokens` up to which the decoded text is stable.
    current_index: usize,
}

impl StreamingDecoder {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    /// Feed the next token and return any newly-stable text to emit.
    ///
    /// Returns `Some(delta)` when a new alphanumeric char has landed (meaning
    /// the prefix up to it is stable), or `None` when the new token only
    /// modifies trailing bytes (e.g. whitespace/punctuation that a future merge
    /// could still change).
    pub fn next_token(&mut self, tokenizer: &TokenizerWrapper, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let toks = &self.tokens[self.prev_index..self.current_index];
            tokenizer.decode(toks)?
        };
        self.tokens.push(token);
        let text = tokenizer.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len()
            && text.chars().last().map(|c| c.is_alphanumeric()).unwrap_or(false)
        {
            let delta = text.split_at(prev_text.len()).1.to_string();
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(delta))
        } else {
            Ok(None)
        }
    }

    /// Flush any remaining un-emitted text (call once at end of generation).
    pub fn decode_rest(&self, tokenizer: &TokenizerWrapper) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let toks = &self.tokens[self.prev_index..self.current_index];
            tokenizer.decode(toks)?
        };
        let text = tokenizer.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            Ok(Some(text.split_at(prev_text.len()).1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Reset for a new session.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A role in a chat turn.
#[derive(Debug, Clone, Copy)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One turn in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
}

/// Format chat turns into the **Qwen `<|im_start|>` ChatML template** — the
/// format Qwen2-Instruct models are trained on.
///
/// Produces:
/// ```text
/// <|im_start|>system
/// <system text><|im_end|>
/// <|im_start|>user
/// <user text><|im_end|>
/// <|im_start|>assistant
/// ```
///
/// The trailing `<|im_start|>assistant\n` (no `<|im_end|>`) primes the model to
/// generate the assistant's reply. If no system turn is supplied, a minimal
/// default is inserted.
pub fn format_qwen_chatml(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    let mut have_system = false;
    for turn in turns {
        if matches!(turn.role, Role::System) {
            have_system = true;
        }
        out.push_str("<|im_start|>");
        out.push_str(turn.role.as_str());
        out.push('\n');
        out.push_str(&turn.content);
        out.push_str("<|im_end|>\n");
    }
    if !have_system {
        // Prepend a default system turn so the model behaves as an assistant.
        let sys = default_system.unwrap_or("You are a helpful assistant.");
        let mut prefixed = String::new();
        prefixed.push_str("<|im_start|>system\n");
        prefixed.push_str(sys);
        prefixed.push_str("<|im_end|>\n");
        prefixed.push_str(&out);
        out = prefixed;
    }
    // Prime the assistant reply.
    out.push_str("<|im_start|>assistant\n");
    out
}

// ─────────────────────── per-architecture chat templates ────────────────────
//
// Different model families are trained on different chat markups; using the
// wrong one degrades output badly. These formats follow the vendors' official
// templates (the same ones llama.cpp hardcodes per arch).

/// The chat prompt format a model family was trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `<|im_start|>role\n…<|im_end|>` — Qwen, StarCoder2-ChatML, InternLM2,
    /// Hermes-style Llama finetunes, LFM2.
    ChatMl,
    /// `<start_of_turn>user/model … <end_of_turn>` — Gemma 1/2/3. No system
    /// role: a system turn is folded into the first user turn.
    Gemma,
    /// `<|user|>…<|end|>` / `<|assistant|>…<|end|>` — Phi-3 and Phi-4.
    Phi3,
    /// `[gMASK]<sop><|user|>…<|assistant|>` — GLM-4.
    Glm4,
    /// `[INST] … [/INST]` — Mixtral and Llama-2-style conversions.
    MistralInst,
    /// `<|start_header_id|>role<|end_header_id|>` — Llama-3.
    Llama3,
    /// `<|start_of_role|>role<|end_of_role|>…<|end_of_text|>` — IBM Granite.
    Granite,
    /// `<｜User｜>…<｜Assistant｜>` with `<｜end▁of▁sentence｜>` — DeepSeek
    /// V2/V3/R1 (the separators are full-width characters, kept verbatim).
    DeepSeek,
    /// `<|im_user|>user<|im_middle|>…<|im_end|>` — Kimi K2 (ships on the
    /// `deepseek2` arch but speaks its own markup).
    Kimi,
    /// `]~b]system/user/ai … [e~[` — MiniMax M2's punctuation-soup tokens
    /// (each is a single vocab entry, not a typo).
    MiniMax,
}

impl ChatTemplate {
    /// The template an architecture trains on. `"llama"` is ambiguous
    /// (Llama-2, Llama-3, Mistral conversions and ChatML finetunes all report
    /// the `llama` arch) — [`pick_template`] resolves it from the tokenizer.
    pub fn default_for_arch(arch: &str) -> ChatTemplate {
        match arch {
            "gemma" | "gemma2" | "gemma3" | "gemma4" => ChatTemplate::Gemma,
            "phi3" => ChatTemplate::Phi3,
            "glm4" => ChatTemplate::Glm4,
            "mixtral" => ChatTemplate::MistralInst,
            "granite" | "granitemoe" | "granite_swa" => ChatTemplate::Granite,
            // The deepseek2 family splits by tokenizer: Kimi K2 carries
            // `<|im_middle|>`, DeepSeek does not (see [`pick_template`]).
            "deepseek2" | "deepseek32" | "kimi_k2" => ChatTemplate::DeepSeek,
            "minimax-m2" => ChatTemplate::MiniMax,
            // qwen2/qwen2_v2/qwen3/qwen3moe/phi2/starcoder2/internlm2/lfm2/
            // nemotron + default — nemotron is disambiguated from the
            // tokenizer like `llama` (see [`pick_template`]).
            _ => ChatTemplate::ChatMl,
        }
    }
}

/// Disambiguate the `llama` arch using the tokenizer's special tokens (BOS is
/// added by the tokenizer itself; only the chat markup differs here).
fn resolve_llama_family(tok: &TokenizerWrapper) -> ChatTemplate {
    if tok.token_to_id("<|start_header_id|>").is_some() {
        ChatTemplate::Llama3
    } else if tok.token_to_id("<|im_start|>").is_some() {
        ChatTemplate::ChatMl
    } else {
        // Llama-2 / Mistral conversions speak [INST].
        ChatTemplate::MistralInst
    }
}

/// Pick the chat template for a loaded model: per-arch default, with the
/// ambiguous archs resolved from the tokenizer's special tokens — `llama` and
/// `nemotron` (Llama-2/3/ChatML finetunes all report those archs), plus the
/// `deepseek2` family (DeepSeek vs Kimi K2 markup).
pub fn pick_template(arch: &str, tok: &TokenizerWrapper) -> ChatTemplate {
    match arch {
        "llama" | "nemotron" => resolve_llama_family(tok),
        "deepseek2" | "deepseek32" | "kimi_k2" => {
            if tok.token_to_id("<|im_middle|>").is_some() {
                ChatTemplate::Kimi
            } else {
                ChatTemplate::DeepSeek
            }
        }
        _ => ChatTemplate::default_for_arch(arch),
    }
}

/// Format chat turns with the given template. Same contract as
/// [`format_qwen_chatml`]: an open-ended assistant prime ends the prompt, and
/// `default_system` (e.g. the rendered tools section) overrides the family's
/// default system handling.
pub fn format_chat_prompt(
    template: ChatTemplate,
    turns: &[ChatTurn],
    default_system: Option<&str>,
) -> String {
    match template {
        ChatTemplate::ChatMl => format_qwen_chatml(turns, default_system),
        ChatTemplate::Gemma => format_gemma_chat(turns, default_system),
        ChatTemplate::Phi3 => format_phi3_chat(turns, default_system),
        ChatTemplate::Glm4 => format_glm4_chat(turns, default_system),
        ChatTemplate::MistralInst => format_mistral_inst(turns, default_system),
        ChatTemplate::Llama3 => format_llama3_chat(turns, default_system),
        ChatTemplate::Granite => format_granite_chat(turns, default_system),
        ChatTemplate::DeepSeek => format_deepseek_chat(turns, default_system),
        ChatTemplate::Kimi => format_kimi_chat(turns, default_system),
        ChatTemplate::MiniMax => format_minimax_chat(turns, default_system),
    }
}

/// Extract the system text (an explicit system turn or `default_system`),
/// if any.
fn system_text(turns: &[ChatTurn], default_system: Option<&str>) -> Option<String> {
    turns
        .iter()
        .find(|t| matches!(t.role, Role::System))
        .map(|t| t.content.clone())
        .or_else(|| default_system.map(|s| s.to_string()))
}

/// Gemma: no system role — a system is folded into the first user turn.
/// `<start_of_turn>user\n…<end_of_turn>\n` / `<start_of_turn>model\n…<end_of_turn>\n`,
/// primed with an open `<start_of_turn>model\n`.
fn format_gemma_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let system = system_text(turns, default_system);
    let mut out = String::new();
    let mut first_user_seen = false;
    for turn in turns {
        match turn.role {
            Role::System => {}
            Role::User => {
                out.push_str("<start_of_turn>user\n");
                if !first_user_seen {
                    first_user_seen = true;
                    if let Some(sys) = &system {
                        out.push_str(sys);
                        out.push_str("\n\n");
                    }
                }
                out.push_str(&turn.content);
                out.push_str("<end_of_turn>\n");
            }
            Role::Assistant => {
                out.push_str("<start_of_turn>model\n");
                out.push_str(&turn.content);
                out.push_str("<end_of_turn>\n");
            }
        }
    }
    out.push_str("<start_of_turn>model\n");
    out
}

/// Phi-3/Phi-4: `<|role|>\n…<|end|>\n`, primed with `<|assistant|>\n`.
fn format_phi3_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str("<|system|>\n");
        out.push_str(&sys);
        out.push_str("<|end|>\n");
    }
    for turn in turns {
        if matches!(turn.role, Role::System) {
            continue;
        }
        let tag = match turn.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => unreachable!(),
        };
        out.push_str(&format!("<|{tag}|>\n{}<|end|>\n", turn.content));
    }
    out.push_str("<|assistant|>\n");
    out
}

/// GLM-4: `[gMASK]<sop>` once, then `<|system|>\n…`/`<|user|>\n…`/`<|assistant|>\n…`
/// blocks (assistant replies are NOT terminated — the next `<|user|>` closes
/// them), primed with `<|assistant|>\n`.
fn format_glm4_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::from("[gMASK]<sop>");
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str(&format!("<|system|>\n{sys}"));
    }
    for turn in turns {
        match turn.role {
            Role::System => {}
            Role::User => out.push_str(&format!("<|user|>\n{}", turn.content)),
            Role::Assistant => out.push_str(&format!("<|assistant|>\n{}", turn.content)),
        }
    }
    out.push_str("<|assistant|>\n");
    out
}

/// Mistral / Llama-2: `[INST] … [/INST]` with the system folded into the first
/// instruction; prior assistant replies close with `</s>`.
fn format_mistral_inst(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let system = system_text(turns, default_system);
    let mut out = String::new();
    let mut first_user_seen = false;
    let mut pending_open_inst = false;
    for turn in turns {
        match turn.role {
            Role::System => {}
            Role::User => {
                out.push_str("[INST] ");
                if !first_user_seen {
                    first_user_seen = true;
                    if let Some(sys) = &system {
                        out.push_str(sys);
                        out.push_str("\n\n");
                    }
                }
                out.push_str(&turn.content);
                out.push_str(" [/INST]");
                pending_open_inst = true;
            }
            Role::Assistant => {
                if pending_open_inst {
                    out.push(' ');
                    pending_open_inst = false;
                }
                out.push_str(&turn.content);
                out.push_str("</s>");
            }
        }
    }
    out
}

/// Llama-3: `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>` blocks,
/// primed with the assistant header. `<|begin_of_text|>` is intentionally not
/// emitted — the tokenizer's post-processor adds BOS on encode.
fn format_llama3_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str(&format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
        ));
    }
    for turn in turns {
        if matches!(turn.role, Role::System) {
            continue;
        }
        let role = match turn.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => unreachable!(),
        };
        out.push_str(&format!(
            "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
            turn.content
        ));
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    out
}

/// Granite: `<|start_of_role|>role<|end_of_role|>content<|end_of_text|>` turns,
/// primed with an open assistant role. BOS is left to the tokenizer's
/// post-processor, matching the other llama-derived families.
fn format_granite_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str(&format!("<|start_of_role|>system<|end_of_role|>{sys}<|end_of_text|>"));
    }
    for turn in turns {
        let role = match turn.role {
            Role::System => continue,
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!(
            "<|start_of_role|>{role}<|end_of_role|>{}<|end_of_text|>",
            turn.content
        ));
    }
    out.push_str("<|start_of_role|>assistant<|end_of_role|>");
    out
}

/// DeepSeek: system text first (no markers), then
/// `<｜User｜>…`/`<｜Assistant｜>reply<｜end▁of▁sentence｜>` per turn, primed
/// with `<｜Assistant｜>`. BOS is left to the tokenizer's post-processor. The
/// separators are full-width Unicode, matching the trained vocab exactly.
fn format_deepseek_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str(&sys);
    }
    for turn in turns {
        match turn.role {
            Role::System => {}
            Role::User => out.push_str(&format!("<｜User｜>{}", turn.content)),
            Role::Assistant => out.push_str(&format!(
                "<｜Assistant｜>{}<｜end▁of▁sentence｜>",
                turn.content
            )),
        }
    }
    out.push_str("<｜Assistant｜>");
    out
}

/// Kimi K2: `<|im_{role}|>role<|im_middle|>content<|im_end|>` turns over the
/// `<|im_system|>` header, primed with the assistant header (verbatim from
/// moonshotai/Kimi-K2-Instruct's chat_template.jinja). Tools ride a dedicated
/// `tool_declare` turn in the official template; AmberCore injects tools
/// through `default_system`, which lands in the system header here.
fn format_kimi_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::new();
    if !turns.iter().any(|t| matches!(t.role, Role::System)) {
        let sys = default_system
            .unwrap_or("You are Kimi, an AI assistant created by Moonshot AI.");
        out.push_str(&format!("<|im_system|>system<|im_middle|>{sys}<|im_end|>\n"));
    }
    for turn in turns {
        match turn.role {
            Role::System => {
                out.push_str(&format!(
                    "<|im_system|>system<|im_middle|>{}<|im_end|>\n",
                    turn.content
                ));
            }
            Role::User => {
                out.push_str(&format!("<|im_user|>user<|im_middle|>{}<|im_end|>\n", turn.content));
            }
            Role::Assistant => {
                out.push_str(&format!(
                    "<|im_assistant|>assistant<|im_middle|>{}<|im_end|>\n",
                    turn.content
                ));
            }
        }
    }
    out.push_str("<|im_assistant|>assistant<|im_middle|>");
    out
}

/// MiniMax M2: `]~!b[]~b]system\n…[e~[` header, then `]~b]user/ai\n…[e~[`
/// turns, primed with `]~b]ai\n` (verbatim from MiniMaxAI/MiniMax-M2's
/// chat_template.jinja — the punctuation-soup tokens are real single-token
/// vocab entries). MiniMax always trains with a system block, so the header is
/// unconditional and empty when no system text is supplied.
fn format_minimax_chat(turns: &[ChatTurn], default_system: Option<&str>) -> String {
    let mut out = String::from("]~!b[]~b]system\n");
    if let Some(sys) = system_text(turns, default_system) {
        out.push_str(&sys);
    }
    out.push_str("[e~[\n");
    for turn in turns {
        match turn.role {
            Role::System => {}
            Role::User => out.push_str(&format!("]~b]user\n{}[e~[\n", turn.content)),
            Role::Assistant => out.push_str(&format!("]~b]ai\n{}[e~[\n", turn.content)),
        }
    }
    out.push_str("]~b]ai\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_formats_a_user_turn_with_default_system() {
        let prompt = format_qwen_chatml(
            &[ChatTurn {
                role: Role::User,
                content: "Hi".into(),
            }],
            None,
        );
        // Default system turn is injected, then the user turn, then the
        // assistant prime (open-ended, no <|im_end|>).
        assert!(prompt.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_respects_an_explicit_system_turn() {
        let prompt = format_qwen_chatml(
            &[
                ChatTurn {
                    role: Role::System,
                    content: "Be terse.".into(),
                },
                ChatTurn {
                    role: Role::User,
                    content: "Hi".into(),
                },
            ],
            None,
        );
        // No default system injected when one is already present.
        assert!(prompt.contains("<|im_start|>system\nBe terse.<|im_end|>"));
        assert!(!prompt.contains("You are a helpful assistant."));
    }

    #[test]
    fn chatml_includes_prior_assistant_turns_for_multi_turn() {
        let prompt = format_qwen_chatml(
            &[
                ChatTurn {
                    role: Role::User,
                    content: "Hello".into(),
                },
                ChatTurn {
                    role: Role::Assistant,
                    content: "Hi there".into(),
                },
                ChatTurn {
                    role: Role::User,
                    content: "Bye".into(),
                },
            ],
            None,
        );
        // The prior assistant reply is closed with <|im_end|>.
        assert!(prompt.contains("<|im_start|>assistant\nHi there<|im_end|>"));
        // And the final assistant prime is open-ended.
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn default_for_arch_maps_each_family() {
        assert_eq!(ChatTemplate::default_for_arch("qwen2"), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::default_for_arch("qwen3"), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::default_for_arch("qwen3moe"), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::default_for_arch("starcoder2"), ChatTemplate::ChatMl);
        assert_eq!(ChatTemplate::default_for_arch("gemma"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::default_for_arch("gemma3"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::default_for_arch("phi3"), ChatTemplate::Phi3);
        assert_eq!(ChatTemplate::default_for_arch("glm4"), ChatTemplate::Glm4);
        assert_eq!(ChatTemplate::default_for_arch("mixtral"), ChatTemplate::MistralInst);
    }

    #[test]
    fn gemma_template_folds_system_into_first_user_turn() {
        let prompt = format_chat_prompt(
            ChatTemplate::Gemma,
            &[
                ChatTurn { role: Role::System, content: "Be terse.".into() },
                ChatTurn { role: Role::User, content: "Hi".into() },
            ],
            None,
        );
        assert!(prompt.contains("<start_of_turn>user\nBe terse.\n\nHi<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        assert!(!prompt.contains("<start_of_turn>system"));
    }

    #[test]
    fn phi3_template_uses_end_markers_and_assistant_prime() {
        let prompt = format_chat_prompt(
            ChatTemplate::Phi3,
            &[
                ChatTurn { role: Role::System, content: "Be terse.".into() },
                ChatTurn { role: Role::User, content: "Hi".into() },
            ],
            None,
        );
        assert!(prompt.contains("<|system|>\nBe terse.<|end|>"));
        assert!(prompt.contains("<|user|>\nHi<|end|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn glm4_template_uses_gmask_sop_prefix() {
        let prompt = format_chat_prompt(
            ChatTemplate::Glm4,
            &[ChatTurn { role: Role::User, content: "Hi".into() }],
            Some("Be terse."),
        );
        assert!(prompt.starts_with("[gMASK]<sop><|system|>\nBe terse."));
        assert!(prompt.contains("<|user|>\nHi"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn mistral_template_wraps_users_in_inst() {
        let prompt = format_chat_prompt(
            ChatTemplate::MistralInst,
            &[
                ChatTurn { role: Role::User, content: "Hello".into() },
                ChatTurn { role: Role::Assistant, content: "Hi".into() },
                ChatTurn { role: Role::User, content: "Bye".into() },
            ],
            Some("Be terse."),
        );
        // System folded into the first instruction.
        assert!(prompt.contains("[INST] Be terse.\n\nHello [/INST] Hi</s>"));
        // Final turn: an open instruction awaiting the reply.
        assert!(prompt.ends_with("Bye [/INST]"));
    }

    #[test]
    fn llama3_template_uses_headers_and_eot() {
        let prompt = format_chat_prompt(
            ChatTemplate::Llama3,
            &[
                ChatTurn { role: Role::System, content: "Be terse.".into() },
                ChatTurn { role: Role::User, content: "Hi".into() },
            ],
            None,
        );
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
        // BOS is left to the tokenizer's post-processor.
        assert!(!prompt.contains("<|begin_of_text|>"));
    }

    #[test]
    fn granite_template_uses_start_of_role_blocks() {
        let prompt = format_chat_prompt(
            ChatTemplate::Granite,
            &[
                ChatTurn { role: Role::System, content: "Be terse.".into() },
                ChatTurn { role: Role::User, content: "Hi".into() },
            ],
            None,
        );
        assert!(prompt.contains("<|start_of_role|>system<|end_of_role|>Be terse.<|end_of_text|>"));
        assert!(prompt.contains("<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>"));
        assert!(prompt.ends_with("<|start_of_role|>assistant<|end_of_role|>"));
    }

    #[test]
    fn deepseek_template_uses_fullwidth_separators() {
        let prompt = format_chat_prompt(
            ChatTemplate::DeepSeek,
            &[
                ChatTurn { role: Role::System, content: "Be terse.".into() },
                ChatTurn { role: Role::User, content: "Hi".into() },
                ChatTurn { role: Role::Assistant, content: "Hello".into() },
                ChatTurn { role: Role::User, content: "Bye".into() },
            ],
            None,
        );
        // System text is bare (no markers), before the first turn.
        assert!(prompt.starts_with("Be terse.<｜User｜>Hi"));
        // Prior assistant reply closes with the full-width end-of-sentence.
        assert!(prompt.contains("<｜Assistant｜>Hello<｜end▁of▁sentence｜>"));
        assert!(prompt.ends_with("<｜Assistant｜>"));
    }

    #[test]
    fn kimi_template_uses_im_middle_turns() {
        let prompt = format_chat_prompt(
            ChatTemplate::Kimi,
            &[ChatTurn { role: Role::User, content: "Hi".into() }],
            None,
        );
        // Kimi's own default system, not the generic one.
        assert!(prompt.contains(
            "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
        ));
        assert!(prompt.contains("<|im_user|>user<|im_middle|>Hi<|im_end|>"));
        assert!(prompt.ends_with("<|im_assistant|>assistant<|im_middle|>"));
    }

    #[test]
    fn minimax_template_uses_punctuation_soup_tokens() {
        let prompt = format_chat_prompt(
            ChatTemplate::MiniMax,
            &[ChatTurn { role: Role::User, content: "Hi".into() }],
            Some("Be terse."),
        );
        assert!(prompt.starts_with("]~!b[]~b]system\nBe terse.[e~[\n"));
        assert!(prompt.contains("]~b]user\nHi[e~[\n"));
        assert!(prompt.ends_with("]~b]ai\n"));
    }

    #[test]
    fn new_families_map_in_default_for_arch() {
        assert_eq!(ChatTemplate::default_for_arch("gemma4"), ChatTemplate::Gemma);
        assert_eq!(ChatTemplate::default_for_arch("granite"), ChatTemplate::Granite);
        assert_eq!(ChatTemplate::default_for_arch("granitemoe"), ChatTemplate::Granite);
        assert_eq!(ChatTemplate::default_for_arch("granite_swa"), ChatTemplate::Granite);
        assert_eq!(ChatTemplate::default_for_arch("deepseek2"), ChatTemplate::DeepSeek);
        assert_eq!(ChatTemplate::default_for_arch("deepseek32"), ChatTemplate::DeepSeek);
        assert_eq!(ChatTemplate::default_for_arch("kimi_k2"), ChatTemplate::DeepSeek);
        assert_eq!(ChatTemplate::default_for_arch("minimax-m2"), ChatTemplate::MiniMax);
        // nemotron falls through to ChatMl but is resolved per-tokenizer.
        assert_eq!(ChatTemplate::default_for_arch("nemotron"), ChatTemplate::ChatMl);
    }
}
