# AmberCore

> *illuminated by Candle.*

A **fully-Rust LLM runner** built on [Candle](https://github.com/huggingface/candle).
Loads quantized **GGUF** models and serves them over an HTTP API that is
**wire-compatible with Ollama**, so it can replace Ollama as the model backend for the
[Phoenix Agent](../PROJECT_CONTEXT.md).

> **Status:** v0.0.0 — scaffold only. M0 (load GGUF → decode one token on CPU) in progress.
> See [`ACRoad.md`](./ACRoad.md) for the full roadmap and dev log.

---

## Why?

Python/C++ LLM runners pay interpreter overhead, GIL contention, GC pauses, and a heavy
runtime stack. AmberCore is **pure Rust**: zero-cost abstraction, no GC, true parallel
batching, and a single static binary. The win isn't beating `llama.cpp` on raw AVX kernel
speed — it's in the **system/serving layer**, where Rust's strengths compound. We reuse
Candle's kernels and compete on scheduling, memory, and concurrency.

---

## How to point Phoenix Agent at AmberCore

AmberCore speaks the **exact** Ollama protocol Phoenix already uses (only
`GET /api/tags` + `POST /api/chat`, NDJSON streaming). So switching is a one-line config
edit — **zero Phoenix code changes**.

In Phoenix Agent's `config.toml`:

```toml
ollama_url = "http://localhost:42069"
```

Then run AmberCore and (re)start Phoenix. That's it.

> Phoenix's default `ollama_url` is Ollama's `http://localhost:11434`. AmberCore listens
> on **port 42069** by default. The port is the only thing you change in Phoenix.

---

## Quick start (once M0 lands)

```bash
# Register a local GGUF model under a tag
ambercore register --file path/to/qwen2.5-coder-7b.gguf --tag qwen2.5-coder:7b

# Run a one-off chat in the terminal
ambercore run --model qwen2.5-coder:7b

# Serve the Ollama-compatible HTTP API (default port 42069)
ambercore serve
```

---

## Design

- **Pure Rust on Candle** — no Python, no C++ FFI.
- **Lib-first** — the real API is `src/lib.rs`. The HTTP server is a thin binary. A future
  milestone can compile AmberCore *in-process* into Phoenix Agent (no HTTP hop).
- **Multi-backend from day 1** — a `Backend` trait abstracts the compute target. CPU now,
  CUDA next (behind a `cuda` feature flag). No rewrites when adding GPU.
- **Generic architecture dispatch** — GGUF metadata selects the architecture; new models
  are a registry entry, not a rewrite.
- **Drop-in for Phoenix** — speak the protocol Phoenix already expects.

Full decisions, the exact Phoenix wire contract, and the milestone roadmap live in
[`ACRoad.md`](./ACRoad.md).

---

## Layout

```
AmberCore/
├── ACRoad.md              ← persistent dev log + roadmap
├── Cargo.toml
├── src/
│   ├── lib.rs             ← public API (Engine, Backend, catalog, DEFAULT_PORT)
│   ├── backend.rs         ← Backend trait + CpuBackend
│   ├── model/             ← GGUF loader + architecture registry + per-arch wrappers
│   ├── pipeline/          ← generation loop, KV cache, sampler
│   ├── tokenizer.rs       ← wraps the `tokenizers` crate
│   ├── catalog.rs         ← model tag registry (dir scan + manifest.json)
│   ├── server/            ← axum: /api/tags + /api/chat (NDJSON)
│   └── bin/ambercore.rs    ← CLI: serve / run / register
└── tests/
```
