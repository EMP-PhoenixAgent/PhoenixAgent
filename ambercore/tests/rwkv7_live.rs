//! Live RWKV7 validation against a real GGUF (ignored by default).
//!
//! Loads an `rwkv7` model through the normal registry path, runs the world
//! chat template, and checks the generation is coherent (an oracle completion
//! plus finiteness). Run with:
//! ```text
//! PA_TEST_RWKV7=<dir containing rwkv7-*.gguf> \
//!   cargo test --test rwkv7_live -- --ignored --nocapture
//! ```

use ambercore::model::registry;
use ambercore::pipeline::{Pipeline, SampleParams, StopCondition};
use ambercore::tokenizer::{format_chat_prompt, pick_template, ChatTurn, Role, TokenizerWrapper};

#[test]
#[ignore = "needs a real rwkv7 GGUF on disk (PA_TEST_RWKV7 env var)"]
fn rwkv7_generates_coherent_text() {
    let dir = std::env::var("PA_TEST_RWKV7").expect("set PA_TEST_RWKV7 to a dir with rwkv7 GGUFs");
    let gguf = std::fs::read_dir(&dir)
        .expect("readable dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        })
        .expect("a .gguf file in the dir");

    let device = candle_core::Device::Cpu;
    let mut loaded = ambercore::model::LoadedModel::load(&gguf).expect("gguf loads");
    assert_eq!(loaded.arch, "rwkv7", "unexpected arch {}", loaded.arch);
    let mut model = registry::build(&mut loaded, &device).expect("rwkv7 builds");

    // Tokenizer must sit next to the GGUF as `<stem>.tokenizer.json`.
    let tokenizer = TokenizerWrapper::load_next_to(&gguf).expect("tokenizer loads");
    let template = pick_template("rwkv7", &tokenizer);
    assert!(matches!(template, ambercore::tokenizer::ChatTemplate::RwkvWorld));

    // Oracle: raw completion (no chat markup) — world models nail this even at 191M.
    let prompt = "The capital of France is";
    let context_length = 4096;
    let mut pipeline = Pipeline {
        model: model.as_mut(),
        tokenizer: &tokenizer,
        device: &device,
        context_length,
    };
    let params = SampleParams { temperature: 0.0, ..Default::default() };
    let stop = StopCondition { max_tokens: Some(30), stop_tokens: Vec::new() };
    let (stats, text) = pipeline
        .generate(prompt, &params, &stop, |_| {})
        .expect("generation runs");
    eprintln!("[completion] {text:?}");
    eprintln!("[stats] {} tok, {:.2} tok/s", stats.output_tokens, stats.tokens_per_sec());
    assert!(
        text.to_lowercase().contains("paris"),
        "oracle failed — completion was: {text:?}"
    );
    assert!(stats.output_tokens >= 5, "suspiciously short generation");

    // Chat path: world template + a multi-turn conversation stays finite.
    let turns = vec![
        ChatTurn { role: Role::User, content: "Hello! Who are you?".into() },
    ];
    let prompt = format_chat_prompt(template, &turns, None);
    eprintln!("[chat prompt] {prompt:?}");
    let mut pipeline = Pipeline {
        model: model.as_mut(),
        tokenizer: &tokenizer,
        device: &device,
        context_length,
    };
    let stop = StopCondition { max_tokens: Some(60), stop_tokens: Vec::new() };
    let (stats, chat) = pipeline
        .generate(&prompt, &params, &stop, |_| {})
        .expect("chat generation runs");
    eprintln!("[chat out] {chat:?}");
    assert!(chat.chars().count() > 5, "chat generation empty");
    assert!(
        chat.chars().all(|c| !c.is_control() || c == '\n' || c == '\r' || c == '\t'),
        "control characters leaked: {chat:?}"
    );
}
