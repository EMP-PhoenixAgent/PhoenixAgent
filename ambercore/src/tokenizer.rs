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
}
