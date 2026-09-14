# AmberCore — Development Roadmap & Log

> **This file is AmberCore's persistent dev log.** Read at the start of any AmberCore
> session to restore full context. Update it whenever a milestone lands or a decision
> changes. Mirrors the role of `PROJECT_CONTEXT.md` for Phoenix Agent itself.

**Last updated:** 2026-08-27
**Version:** v0.8.0 (M5b replica pool + qwen3 KV-reset fix; M6 CUDA; M7 Metal; M8 AMD)
**Status:** **M6 CUDA verified on GPU + M7/M8 landed.** The `cuda` feature propagates to all
three candle crates and **builds green against CUDA 13.3 + MSVC** (`CL=/Zc:preprocessor /std:c++17`
needed — CUDA 13's CCCL mandates the conforming preprocessor; `CUDA_COMPUTE_CAP=86`; 63 MB binary).
**Verified on the RTX 3050** (driver 610.88): Qwen2-0.5B = **57.98 tok/s** (vs 34.4 CPU),
Qwen3-8B = **26.61 tok/s** (vs ~3 CPU) at **7097/8192 MiB VRAM** (86% — the 8B Q4_K_M fits, tight).
Driver note: needed updating from 591.86 (CUDA 13.1) to 610.88 (13.3) to clear a
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` skew (no rebuild). **M7** (Metal) + **M8** (AMD stub)
code-complete: `MetalBackend`/`AmdBackend`, `DeviceChoice::Metal`/`Amd`, `auto` = CUDA → Metal → CPU.
**M5b — replica pool + fair scheduler** (`--max-replicas N` lifts the per-tag serialization; lazy
growth + FIFO queueing) **landed**, and with it a fix for a latent **qwen3 server KV bug** (see §7).
`cargo test` = **40/40 (default)** / **38/38 (--features cuda)**. Next: more architectures / live
multi-replica soak; true token-level batching stays blocked on candle's single-sequence quantized KV.

---

## 1. What is AmberCore?

A **fully-Rust LLM runner** built on [Candle](https://github.com/huggingface/candle)
(HuggingFace's minimalist ML framework). It loads quantized **GGUF** models and serves
them over an HTTP API that is **wire-compatible with Ollama**'s `/api/tags` and
`/api/chat`, so it can be a **drop-in replacement for Ollama** as the model backend of
the Phoenix Agent.

**Core principles:**
- **Pure Rust** — no Python interpreter, no GIL, no C++ FFI. Zero-cost abstraction +
  safety + a single static binary.
- **Lib-first** — the real API lives in `lib.rs`. The HTTP server is a thin binary
  wrapper. A future milestone can compile AmberCore *in-process* into Phoenix Agent,
  eliminating the HTTP hop entirely.
- **Multi-backend from day 1** — a `Backend` trait abstracts the compute target (CPU now,
  CUDA next, no rewrites). candle's `Device` enum does the heavy lifting under the hood.
- **Drop-in for Phoenix** — speak the exact protocol Phoenix already expects, so Phoenix
  needs **zero code changes** — only a one-line config edit.

**Why Rust over Python/C++ runners:** the win is not in beating `llama.cpp` on raw AVX
kernel speed (a multi-year effort). The win is in the **system/serving layer**: no
interpreter overhead, true parallel batching, lower memory footprint, no GC pauses, and a
single self-contained binary. We reuse Candle's kernels and compete on the layer where
Rust's strengths compound.

---

## 2. Locked Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Core compute | **Candle** (`candle-core`, `candle-nn`, `candle-transformers`) | Pure-Rust ML framework; no FFI; first-class GGUF support |
| Mission | **Drop-in replacement for Ollama**, serving Phoenix Agent | Narrow, testable target; tight integration |
| Backend abstraction | **`Backend` trait from day 1** (CPU now, CUDA later) | Decouples compute target from inference code; no rewrites when adding GPU |
| First milestone | **Load GGUF → decode ONE token on CPU** | Smallest end-to-end proof of loader/dispatch/backend/tokenizer pipeline |
| Model source | **Local files now**; `pull` deferred (Phoenix will manage downloads itself) | Keeps v0.1 scope minimal; aligns with Phoenix owning download UX |
| Architecture scope | **Generic dispatch** (GGUF metadata → registry → any candle model) | Not hardcoded to one arch; new models are a registry entry, not a rewrite |
| Deployment | **Lib-first** — `lib.rs` is the API; HTTP server is a thin binary | Enables future zero-overhead in-process integration with Phoenix |
| **Default port** | **42069** | User-chosen |

---

## 3. The Phoenix Contract (the exact wire protocol AmberCore must serve)

Derived from a direct audit of `phoenix-agent/src/model/ollama.rs` + `config.rs` +
`health.rs`. This is the **entire** surface Phoenix touches — only 2 endpoints.

### 3.1 Base URL
- Phoenix's `config.toml::ollama_url` defaults to Ollama's `http://localhost:11434`.
- **To use AmberCore: set `ollama_url = "http://localhost:42069"` in Phoenix's config.**
  One line. Zero Phoenix code changes.
- AmberCore listens on `0.0.0.0:42069` by default (`DEFAULT_PORT = 42069` in `lib.rs`).

### 3.2 `GET /api/tags` — model listing + health signal
Response body:
```json
{ "models": [ { "name": "qwen2.5-coder:7b" }, ... ] }
```
- Phoenix reads **only** the `name` field of each entry. The `models` array defaults to
  empty if absent.
- **This endpoint is also Phoenix's sole health probe** (`health.rs::probe_all`):
  - HTTP 200 → Ollama considered "up"; detail = `"{n} model(s)"`.
  - Non-200 / connection error → Ollama = `Down`.
  - The active model is "available" iff its exact string appears in the names list.
- Phoenix uses `/api/tags` — **not** OpenAI's `/v1/models`.

### 3.3 `POST /api/chat` — the chat call (streaming)
Request body AmberCore must accept:
```json
{
  "model": "qwen2.5-coder:7b",
  "messages": [ {"role":"system|user|assistant|tool", "content":"...", "tool_calls":[...]?} ],
  "tools": [ {"type":"function","function":{"name":"...","description":"...","parameters":{...JSON Schema...}}} ],
  "stream": true,
  "temperature": 0.2
}
```
- `tools` is **omitted** when empty (Phoenix uses `skip_serializing_if`).
- `temperature` always present.
- Tool-result messages carry the tool name in a **top-level** `tool` field (not nested).

### 3.4 Streaming format — **NDJSON** (one JSON object per line, NOT SSE)
- No `data:` prefixes. Phoenix buffers the byte stream and splits on `\n`, parsing each
  non-empty line as JSON; unparseable lines are silently skipped.
- Each line's shape:
  ```json
  { "message": { "content": "<delta>", "tool_calls": [...]? }, "done": false }
  ```
  - `message.content` is the **delta** (Phoenix accumulates itself).
  - Phoenix does **not** read the `role` field on chunks.
- **Terminal line** (the one with `done: true`) must additionally carry:
  - `message.tool_calls` (if any),
  - `eval_count` (output token count),
  - `prompt_eval_count` (input token count).

### 3.5 Tool-calling wire details
- Tools sent to AmberCore: `tools: [{type:"function", function:{name, description, parameters}}]`.
- Tool calls AmberCore returns: `message.tool_calls: [{id?, function:{name, arguments}}]`
  where **`arguments` is a JSON object serialized as a STRING** (OpenAI/Ollama convention).
  `id` is optional.
- AmberCore emits `message.tool_calls` **only on the terminal `done:true` line**.

### 3.6 Model identity
- The `model` string is whatever Phoenix has in `config.toml::model` (default
  `qwen2.5-coder:7b`) — a full Ollama tag including `:tag`.
- **No normalization anywhere.** Phoenix compares tag strings by exact equality against
  `/api/tags`. AmberCore must **accept any model string verbatim** and **echo that exact
  string back** in `/api/tags`, or Phoenix's health check marks it "not pulled".

---

## 4. Roadmap

| Milestone | Goal | Status |
|---|---|---|
| **M0** — Scaffolding + hello-token on CPU | Cargo project compiles; `Backend` trait + `CpuBackend`; GGUF loader; qwen2 dispatch; forward pass → decode **one** token → print. Proves loader/dispatch/backend/candle/tokenizer pipeline. | ✅ **DONE** — verified against real Qwen2-0.5B GGUF |
| **M1** — Full generation pipeline | KV cache; sampling (greedy → temperature/top-k/top-p); multi-token streaming; EOS handling; `ambercore run` CLI chat. | ✅ **DONE** — verified streaming chat against Qwen2-0.5B |
| **M2** — HTTP server (drop-in for Phoenix) | axum app; `GET /api/tags` from catalog; `POST /api/chat` NDJSON matching §3 exactly; model catalog (scan dir + optional `manifest.json`); `ambercore serve` (default port **42069**). | ✅ **DONE** — verified full Phoenix wire contract via curl |
| **M3** — Tool/function-calling | Parse `tools`; inject into prompt; emit `message.tool_calls` on the terminal line with `arguments` as a JSON string. | ✅ **DONE** — verified Hermes-format tool calls parse to Phoenix's wire contract |
| **M4** — CUDA backend | `CudaBackend` behind a `cuda` Cargo feature; runtime device selection. | ✅ **DONE** — CUDA path implemented + feature-gated; CPU + auto-fallback verified |
| **M5a** — CPU SIMD perf | `target-cpu=native` → AVX2/F16C enabled; 3x throughput (11.5→34.4 tok/s) | ✅ **DONE** — baked into `.cargo/config.toml` |
| **M6** — CUDA verified on GPU | Flip `cuda` feature on across candle-core/nn/transformers; install CUDA toolkit; verify Q4_K_M GGUFs run on the RTX 3050 with a real tok/s measurement. (M4 wrote the code but never ran it — no nvcc.) | ✅ **Done** — builds green (CUDA 13.3 + MSVC; `CL=/Zc:preprocessor`); verified on RTX 3050 (drv 610.88): Qwen2-0.5B **58 tok/s**, Qwen3-8B **26.6 tok/s** @ 7.1/8 GB VRAM. |
| **M7** — Metal backend (Apple) | New `MetalBackend` behind a `metal` feature mirroring the CUDA pattern; `Device::new_metal(0)`; macOS-only. Quantized GGUF kernels ship in candle (`quantized/metal.rs`), so Q4_K_M models run on Apple GPU. | ✅ **Code-complete** — `MetalBackend` + `metal` feature + `DeviceChoice::Metal`; `auto`=CUDA→Metal→CPU. macOS verification deferred (no Apple GPU here). |
| **M8** — AMD stub | `AmdBackend` behind a `rocm` feature that errors cleanly ("ROCm support is experimental upstream"). Keeps `DeviceChoice` + CLI ready the day candle merges official ROCm. No fragile fork. | ✅ **Done** — `AmdBackend` stub + empty `rocm` feature + `DeviceChoice::Amd`; always errors cleanly. |
| **M5b** — Replica pool (done); continuous batching (deferred) | Lift per-tag serialization via a model-replica pool (`--max-replicas N`, lazy growth + FIFO queueing); fix the qwen3 KV-cache reset bug. True token-level batching (one matmul across sequences) needs candle's quantized KV rewritten (single-sequence today) — deferred. | ✅ Pool + KV fix landed (40/40 & 38/38 tests); 🔜 true batching (blocked upstream) |

---

## 5. Directory Layout

```
N:\Phoenix Agent\AmberCore\
├── ACRoad.md              ← THIS FILE (AmberCore persistent memory)
├── Cargo.toml             ← deps + `cuda` feature + DEFAULT_PORT constant
├── README.md              ← orientation + Phoenix drop-in instructions
├── src/
│   ├── lib.rs             ← public API: Engine, Backend, catalog, DEFAULT_PORT=42069
│   ├── error.rs           ← Error type + Result alias
│   ├── backend.rs         ← Backend trait + CpuBackend (wraps candle Device)
│   ├── tokenizer.rs       ← wraps `tokenizers` crate
│   ├── catalog.rs         ← model registry (scan models/ dir + optional manifest.json)
│   ├── model/
│   │   ├── mod.rs         ← generic dispatch: arch name → model builder
│   │   ├── gguf.rs        ← load GGUF via candle_transformers::quantized::gguf_file
│   │   ├── registry.rs    ← arch → constructor table (qwen2 first, then llama, ...)
│   │   ├── qwen2.rs       ← wraps candle_transformers::models::qwen2
│   │   └── llama.rs       ← (stub — M1+)
│   ├── pipeline/
│   │   ├── mod.rs         ← generation loop (prefill + decode steps)
│   │   ├── kv_cache.rs    ← KV cache management
│   │   └── sampler.rs     ← greedy first, then temperature/top-k/top-p
│   ├── server/
│   │   ├── mod.rs         ← axum app + serve(port)
│   │   ├── tags.rs        ← GET /api/tags
│   │   ├── chat.rs        ← POST /api/chat (NDJSON streaming)
│   │   └── protocol.rs    ← wire types matching §3 exactly
│   └── bin/
│       └── ambercore.rs    ← clap CLI: `serve` (default 42069), `run`, `register`
└── tests/                 ← protocol-shape / round-trip tests (M2)
```

---

## 6. Dependencies

| Crate | Role |
|---|---|
| `candle-core` | Tensor library + `Device` enum (CPU/CUDA/Metal) |
| `candle-nn` | NN building blocks |
| `candle-transformers` | Model architectures (qwen2, llama, ...) + GGUF loader |
| `tokenizers` | HuggingFace fast Rust tokenizer |
| `axum` | HTTP server (drop-in for Phoenix's Ollama client) |
| `tokio` | Async runtime |
| `serde` / `serde_json` | (De)serialization for the wire protocol |
| `anyhow` / `thiserror` | Error handling |
| `clap` | CLI (`serve` / `run` / `register`) |
| `tracing` / `tracing-subscriber` | Structured logging |
| `dirs` | Locate the models directory across platforms |

The `cuda` Cargo feature gates CUDA backend support (M4).

---

## 6b. Adapter Authoring Rules — broadcast views, strides, and head-count forks

*(Born from the 2026-09-05 Qwen3.5-4B incident: the model NaN'd at sample on
CUDA and derailed on every device past ~10 tokens, while the validated 0.8B
was perfect. Two silent stride bugs + one convention mismatch, all invisible
to the synthetic tests. Check EVERY item below before calling an adapter
done.)*

1. **`broadcast_as` views are stride-0. Never hand one to `reshape` or
   `matmul` without `.contiguous()`.** When you must reshape through a
   broadcast view, remember the flatten follows the LOGICAL dim order: where
   you put the copy axis changes the result. Growing k-heads to v-heads via
   `(h, t, 1, d) → broadcast (h, t, rep, d) → reshape (h·rep, t, d)` silently
   interleaves the copy axis with TIME; the axis must go where the target
   layout says (see rule 5).
2. **`.contiguous()` is lenient — it no-ops on tensors that merely LOOK
   row-major.** Extent-1 dims (seq == 1 on every decode step!) make even
   reversed strides pass candle's check, and a `cat` of strided inputs can
   come back with head-interleaved strides on CUDA. After any
   `narrow → cat` (partial rope is the classic), force `.contiguous()` on
   BOTH the inputs and the result. Cheap insurance; prefill is unaffected.
3. **CPU and CUDA disagree on bad strides: CPU matmul silently copies
   (wrong-but-finite numbers), CUDA either hard-errors
   ("matmul is only supported for contiguous tensors") or reads garbage
   into NaN logits that surface at `sample: A weight is negative…`
   (rand_distr rejecting NaN softmax weights).** So a CPU-validated adapter
   proves nothing about its CUDA path — validate BOTH devices, and generate
   past ~30 tokens: the 4B looked fine for 8 tokens and derailed after.
   Fastest coherence oracle: `--raw --prompt "The capital of France is"`
   must yield " Paris".
4. **Config forks only run on the first model that has that shape.** The
   n_k ≠ n_v head-grow path never executed on any validated model until the
   4B shipped it. When an adapter branches on head counts, groups, MoE,
   tied/untied heads, SWA intervals… pick a REAL validation model that
   exercises every branch — synthetic tests use the symmetric case by
   default (that's how they're simplest to write).
5. **GGUF and safetensors speak different repeat conventions.** llama.cpp's
   `ggml_repeat` TILES (`k0..kN, k0..kN` — and the HF→GGUF converter
   permutes the v/z channel order to match its runtime), while
   transformers' `repeat_interleave` BLOCKS (`k0, k0, k1, k1`). A GGUF-fed
   adapter must use the tiled layout, a safetensors-fed one the block
   layout — with equal head counts the two coincide, which is exactly why
   the 0.8B couldn't catch it. Carry the weight source (GGUF vs HF) into
   the adapter and branch on it at the repeat site.
6. **Debug playbook that worked, in order:** (a) env-gated per-stage
   `tensor.layout().stride()` prints — one run pinpoints the exact op that
   corrupts the layout (do it at decode, seq == 1, where extent-1 dims
   hide); (b) the raw factual completion as the cheap end-to-end oracle;
   (c) a second quant of the same model to rule the file out; (d) the
   llama.cpp source for that arch (`src/models/<arch>.cpp`) as the
   GGUF-layout ground truth — transformers source only for safetensors.
7. **Windows CUDA builds: `cargo build --release --features cuda` from a VS
   developer prompt** (vcvars64 — nvcc needs `cl.exe`); a bare Git-Bash
   build fails with a bare "nvcc error" on `affine.cu`, and a running
   `ambercore.exe` locks the link ("failed to remove file … os error 5") —
   kill the server first. And check binary-vs-source timestamps before
   trusting a repro (today's first "repro failure" was a stale exe).

---

## 7. Development Log

### 2026-08-27 (final) — VRAM pre-flight REFUSAL + iGPU render pin + admin elevation

**Spec evolution:** the headroom warning became a **hard pre-flight refusal**
(`backend::check_vram` → `VramCheck { fits: Option<bool>, needed_mb, free_mb, message }`,
run in `build_loaded_entry` BEFORE the build takes its share, on a blocking thread):
`fits == Some(false)` → the load is refused with the actionable message (estimate =
file ×1.5 + 300 MiB, deliberately overshooting so borderline refusals err safe);
`Some(true)` → a "contained in VRAM — needs ~X, Y free" line; `None` (no nvidia-smi /
CPU backend) → proceeds unjudged. Both verdicts are stored (`last_vram_warning` /
`last_vram_info`) and Phoenix emits whichever is set to the chat as a status message
after warm-up/`run_ambercore` (`emit_vram_status`) — the user sees contained-or-refused
in every case.

**iGPU: what's real and what isn't.** "Use the iGPU for the spill" is not implementable
and would gain nothing: CUDA allocations live only on NVIDIA devices (no candle iGPU
backend), and iGPU memory is system RAM on the far side of the same PCIe — identical
bandwidth to the driver's own spill. What IS real on hybrid systems: move RENDERING off
the dGPU. So (1) the refusal message detects a secondary adapter
(`detect_secondary_gpu`, pub — Win32_VideoController / lspci) and advises moving the
display + GPU-heavy apps to it; (2) **Phoenix pins its own UI to the iGPU**
(`main.rs::pin_rendering_to_igpu`, GUI launch, pre-runtime): Chromium's
`--prefer-integrated-gpu` via `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` (set before
WebView2 spawns; CUDA unaffected — device enumeration ignores DXGI adapter preference)
+ the persistent per-exe Windows GPU preference (`reg add HKCU\...\UserGpuPreferences
→ GpuPreference=1;`). The chat UI needs nothing from the RTX; the model gets the VRAM.
Remaining fixed cost: if the monitor is cabled to the dGPU, DWM keeps ~0.5 GiB there —
plug the display into the motherboard output to reclaim that too.

**Admin elevation** (`phoenix-agent/app.manifest` via tauri-build
`WindowsAttributes::app_manifest`, `build.rs`): `requireAdministrator` permanently marks
the exe — Windows always launches it elevated (the OS-side "remembering"; the UAC
consent dialog itself still appears per launch unless UAC is set to elevate silently —
apps cannot opt out, by design). Needed because per-process GPU memory via nvidia-smi
is only readable elevated. RUN-PA.bat self-elevates (`net session` check +
`Start-Process -Verb RunAs`) — plain `cargo run` would die with error 740 on a
requireAdministrator exe. All 58 engine tests + app `cargo check` +
`cargo check --features cuda` green; mirrors synced.

### 2026-08-27 (later) — reasoning streams during tool turns + VRAM headroom warning

**Why:** with tools active (every Phoenix agent call), `/api/chat` buffered the WHOLE
generation — so the UI's working indicator showed nothing for the entire think, TTFT
measured "full generation time" (154 s observed), and a qwen3-8b run that crawled at
2.2 tok/s (vs the 26.6 the CLI had verified) gave no hint WHY.

**(1) Reasoning now streams live even with tools** (`server/chat.rs`): a `<think>` block
can never contain a `<tool_call>`, so the streaming `ThinkSplitter` routes Think pieces to
`GenEvent::Reasoning` immediately; only CONTENT stays buffered (tool calls must be parsed
post-hoc, never leaked as text). The splitter's end-of-stream flush now emits Reasoning in
both paths, and the post-gen think re-emit was removed (it would duplicate). TTFT is real
again (first reasoning delta) and the Phoenix working indicator shows the model thinking
throughout tool turns.

**(2) VRAM headroom warning** (`backend.rs::vram_headroom_warning`, wired in
`server/mod.rs::build_loaded_entry`): before each replica build, estimate resident size
(file × 1.5 + 300 MiB — calibrated on qwen3-8b Q4_K_M: 4.7 GiB file → 7.1 GiB resident on
the RTX 3050) and compare against free VRAM via **`nvidia-smi`** (context-free — unlike
`mem_get_info` it needs no current CUDA context, so async load threads can call it; no new
dependency, deliberately not nvml-wrapper per the telemetry note). Oversized → a
human-readable warning (needs vs free + top-3 GPU processes) is logged AND stored in
`ServerState::last_vram_warning` (surfaced via `hardware_status().vram_warning`). Phoenix's
boot warm-up + `run_ambercore` emit it to the chat as a status message. Rationale: under
WDDM an oversized CUDA allocation never fails — it spills to system RAM and pages over
PCIe per step (the observed 26.6 → 2.2 tok/s, ~12x). `GpuInfo` gained `vram_free_mb`
(mem_get_info now yields both free + used). 58/58 engine tests + app `cargo check` green;
mirrors synced.

### 2026-08-27 — CUDA mid-generation `INVALID_CONTEXT` fix: one shared stream, not per-thread

**Symptom (v0.8.4 CUDA installers, RTX 3050):** the embedded engine loads + warms on GPU fine,
streams a generation for ~2–4 minutes, then dies with
`candle error: DriverError(CUDA_ERROR_INVALID_CONTEXT, "invalid device context")`
(logged by `server::chat` as "generation failed mid-stream"). No TDR events in the Windows System
log, so not a driver reset; not OOM or PTX skew either (different errors, and the model had been
running fine).

**Root cause:** candle 0.11's `Device::new_cuda` builds its `CudaDevice` on the CUDA
**per-thread default stream** (`cudaStreamPerThread`, cudarc's `per_thread_stream()`). In that
mode every OS thread submits on its own stream and cudarc's cross-stream event tracking is off —
its docs require callers to guarantee no cross-stream tensor lifetime overlap. But AmberCore
boots/warm-ups, loads models, and runs each generation on **different tokio `spawn_blocking`
threads** (boot thread ≠ `build_loaded_entry` thread ≠ generation thread): weights allocated on
the load thread's stream get read, re-allocated around (KV-cache growth, temporaries) and freed on
the generation thread's stream with no cross-stream ordering. The stream-ordered allocator
eventually frees/reuses memory another stream still depends on → the context goes invalid from
then on. Matches the timing (fine for minutes, dies once a cross-thread realloc/free pattern
hits) and both qwen2-0.5b + qwen3-8b.

**Fix:** `CudaBackend::new` now uses `Device::new_cuda_with_stream` — ONE cudarc-managed
non-blocking stream shared by every thread, with cudarc's event tracking inserting the
alloc/free cross-stream dependencies. One line (plus the comment) in `backend.rs`; costs nothing
in practice since the default replica count is 1 anyway. Belt-and-braces follow-up if it ever
resurfaces: pin load + generation to one dedicated engine thread.

**Also note (context, not a bug here):** candle 0.11's new `ug` GGUF path pulls a **second cudarc
copy (0.17.8)** alongside candle-core's 0.19.9 — both retain the same device primary context, so
they interoperate, but the version skew is worth remembering when debugging context/stream issues.

### 2026-08-12 — M5b: replica pool + fair scheduler (and a qwen3 KV-reset bug fix)

**Direction chosen:** replica-pool concurrency + fair scheduler, *not* true token-level continuous
batching. The deciding constraint: candle 0.11's `quantized_qwen2`/`quantized_qwen3` `forward` is
`&mut self` with a **single-scalar `index_pos`** and one linear append-only KV cache per layer — no
per-sequence positions, no paged KV — so iteration-level batching (the real throughput win) would
require rewriting/forking candle's quantized attention. Research-grade; out of scope for now. The
pool instead lifts the goal the server's own doc comment flagged: *"M5+ will lift the per-tag
serialization."*

**The latent qwen3 bug this surfaced (found + fixed).** Auditing candle's cache lifecycle:
- `quantized_qwen2` **implicitly** drops its KV cache when a prefill runs at `index_pos == 0`
  (`if index_pos == 0 { (k, v) }`). This is the *only* reason `ambercore serve` reusing one model
  across requests ever produced coherent qwen2 output.
- `quantized_qwen3` does **not** self-reset — its cache appends unconditionally. Reusing one qwen3
  instance across sessions leaks the previous sequence's K/V into the next (garbage output) **and**
  grows the cache without bound (memory leak). The reference Qwen3-8B had this bug under `serve`
  with >1 request. (Not hit in M6 because `ambercore run` builds a fresh model each time.)
**Fix:** exposed `clear_kv_cache()` through AmberCore's `DynModel` trait (default no-op; qwen2/qwen3
forward to candle's public `ModelWeights::clear_kv_cache()`), and the pipeline calls it at the start
of every `generate()` / `decode_one()`. Both arches now start each session clean, and the
qwen2-vs-qwen3 asymmetry no longer matters.

**Replica pool.** Per-tag `ReplicaPool<T>` (generic over the handle so its logic is unit-tested with
a trivial stand-in; the server instantiates it at `T = Arc<Mutex<LoadedEntry>>`):
- Up to `max_replicas` built models per tag; **lazy growth** (empty until demand arrives) — no K×
  memory cost when there's no concurrency.
- **FIFO fair queueing** past the cap: at-capacity requests await a replica and wake in arrival order
  as replicas are released (no head-of-line blocking).
- Model builds (seconds + GiB) always run **outside** the pool's lock, coordinated via an
  `AcquireOutcome::Build` → `adopt` handshake with an `in_flight_builds` counter to prevent
  over-allocation. `release` is sync so the blocking generation thread returns a replica directly.
- `--max-replicas N` on `serve` (default **1** = the old per-tag serialization, zero behavior change;
  raise for concurrency). `ReplicaHandle` is RAII — drop releases.

**Reality check:** the pool adds concurrency (good for multi-request / multi-user workloads), **not**
per-token throughput (the matmul-sharing win needs true batching). On the 8 GB GPU the 8B Q4_K_M
(≈5 GB) pins `max-replicas` to 1; the win applies to CPU (RAM permitting) and the 0.5B. The scheduler
built here is the seam true batching would later plug into.

**Tests:** `cargo test` = **40/40 (default)** / **38/38 (--features cuda)**. New: 6 pool-logic tests
(build/free/reuse, lazy growth + cap, build-failed slot release, async FIFO waiter ordering) + 1
`DynModel::clear_kv_cache` trait test.

---

### 2026-08-12 — M6/M7/M8 GPU-backend push: CUDA build green; Metal + AMD landed

**Scope:** wire the three GPU backends. The environment is now ready — `nvcc` (CUDA 13.3)
and the RTX 3050 8 GB (driver 591.86) are both present, the exact hardware M6 was waiting on.

**M6 — CUDA: feature wired + build green; on-GPU run pending a driver update.**
1. `Cargo.toml`: `cuda = []` → `cuda = ["candle-core/cuda","candle-nn/cuda","candle-transformers/cuda"]`.
   The `CudaBackend` code from M4 was already correct; this propagation is what actually pulls
   candle's quantized CUDA kernels in.
2. **Build hurdles (all solved, Windows-specific):**
   - `nvcc` couldn't find `cl.exe` → load the MSVC env first: `vcvars64.bat` (VS 2022 / MSVC 14.44).
   - CUDA 13's CCCL headers reject MSVC's traditional preprocessor as a **fatal C1189** →
     `set CL=/Zc:preprocessor /std:c++17`. (cl.exe reads the `CL` env var; nvcc spawns cl.exe as
     its host compiler, so this injects the flag into every kernel compile without patching
     candle/cudaforge.)
   - `CUDA_COMPUTE_CAP=86` for the RTX 3050 (Ampere).
   - Result: `cargo build --release --features cuda` → **green in 3m20s**, **63 MB binary** (vs 9.7 MB CPU-only).
3. **On-GPU run: VERIFIED (after a driver update).** First attempt hit
   `DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)` — driver **591.86 supported only CUDA 13.1**,
   but nvcc is **13.3** (PTX ISA 8.8 > the driver's 8.7). **Fix:** updated the driver to **610.88**
   (CUDA 13.3) — no rebuild needed (same binary ran). Then on the RTX 3050 8 GB:
   - **Qwen2-0.5B (Q4_K_M): 57.98 tok/s** (CPU baseline 34.4 → **1.7x**). Small model is
     memory-bandwidth/launch-overhead bound, so the GPU edge is modest.
   - **Qwen3-8B (Q4_K_M): 26.61 tok/s** (CPU ~3 → **~8-9x** — the dramatic uplift). Loads in 15.3s;
     Qwen3 reasons in `<think>` mode as expected.
   - **VRAM: 7097 / 8192 MiB (86%)** for the 8B — the Q4_K_M (~5 GB weights + KV + activations)
     fits the 8 GB card but tight; long contexts will approach the ceiling.
4. **Done criterion met** — real on-GPU tok/s numbers recorded above.

**M7 — Metal: code-complete (macOS verification deferred).** New `MetalBackend` behind
`#[cfg(all(feature = "metal", target_os = "macos"))]` mirroring `CudaBackend`
(`Device::new_metal(0)`); the `metal` feature propagates to the three candle crates; new
`DeviceChoice::Metal`; `auto` now tries **CUDA → Metal → CPU**. No Apple GPU on this machine, so
runtime verification is deferred (anticipated in the original M7 plan). Caveat to test on macOS:
candle issue #2818 (Metal embedding panic).

**M8 — AMD: done (clean stub).** `AmdBackend` behind an **empty** `rocm` feature (deliberately
empty — candle has no stable ROCm to propagate; no fragile fork). `DeviceChoice::Amd` always
errors cleanly: *"AMD/ROCm support is blocked on upstream candle … will wire it the day candle
merges official ROCm."* `ambercore serve --device amd` now gives an informative message instead of
"unknown device".

**Tests:** `cargo test` (default features) = **33/33 pass**, including two new backend tests
(`resolve_metal_without_feature_errors_cleanly`, `resolve_amd_always_errors_cleanly`).
`cargo test --features cuda` = **31/31 pass** (the two `#[cfg(not(feature = "cuda"))]` backend
tests are correctly excluded under the cuda feature — they assert the no-cuda behavior).

**Build env recap:** `vcvars64.bat` → `set CL=/Zc:preprocessor /std:c++17` →
`set CUDA_COMPUTE_CAP=86` → `cargo build --release --features cuda` (or reuse the helper at
`%TEMP%\ambercore_cuda_build.bat`). Runtime just needs the 610.88+ driver.

---

### 2026-08-06 — GPU backend plan: CUDA (M6) + Metal (M7) + AMD stub (M8)

**Goal:** wire AmberCore to CUDA (NVIDIA), Metal (Apple), and AMD. Researched
candle-core 0.11's actual capabilities (verified against its `Cargo.toml`,
`src/quantized/`, and upstream issues/PRs). Plan locked with the user; execution
scheduled for next week.

**The key finding — quantized GGUF runs on GPU.** The biggest risk was whether
candle's quantized (`Q4_K_M` etc.) kernels are CPU-only. They are **not**:
`candle-core/src/quantized/` ships real `cuda.rs` and `metal.rs` kernels
(`QStorage::Cuda` / `QStorage::Metal` variants), plus the `fast_mmq`/`fast_mmvq`
GPU-optimized matmul/matvec paths. So `quantized_qwen2` / `quantized_qwen3` will
run on a CUDA or Metal device — not just CPU. The feature must be enabled on
**all three** crates (`candle-core` + `candle-nn` + `candle-transformers`).

| Backend | candle feature | AmberCore status | Build req | Plan |
|---|---|---|---|---|
| **CUDA (NVIDIA)** | `cuda` | Code written (M4), **never run** (no nvcc) | CUDA toolkit + `nvcc` | **M6:** enable feature on 3 crates, install CUDA toolkit, verify on RTX 3050 (8 GB, driver 591.86), measure tok/s |
| **Metal (Apple)** | `metal` | Not started | macOS only | **M7:** new `MetalBackend` mirroring `CudaBackend`; `Device::new_metal(0)`; macOS CI/build only |
| **AMD (ROCm/HIP)** | *(none upstream)* | Not started | ROCm toolkit, Linux | **M8:** stub only — candle has **no stable ROCm support** (only stale experimental PRs #3424/#3801, self-described "AI-generated, needs cleanup, unsafe"). Stub errors cleanly so `DeviceChoice` is ready when upstream lands. No fragile fork. |

#### M6 — CUDA (the priority — code exists, just needs running)

The `CudaBackend` in `backend.rs` is already fully written and gated behind
`#[cfg(feature = "cuda")]`. It calls `Device::new_cuda(ordinal)` and hands out
the device. The feature in `Cargo.toml` is currently empty (`cuda = []`).

**Work:**
1. `Cargo.toml`: change `cuda = []` → `cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]` so the feature propagates to all three candle crates.
2. Install the **CUDA toolkit** on this machine (RTX 3050 8 GB, driver 591.86 — driver is current; only the toolkit/`nvcc` is missing). Match the driver-supported toolkit version.
3. `cargo build --release --features cuda` with `CUDA_COMPUTE_CAP` set for the 3050 (compute capability 8.6 → Ampere).
4. `ambercore serve --device cuda` → verify `qwen2:0.5b` and `qwen3:8b` (Q4_K_M) load onto the GPU and generate. Measure tok/s (expect a large uplift over the 34 tok/s CPU baseline for the 0.5B; the 8B should move from ~3 tok/s CPU into a usable range).
5. Watch VRAM: the 8B Q4_K_M is ~5 GB weights + KV cache + activations; the 8 GB 3050 is tight. If it OOMs, document the ceiling and test the 0.5B + 1.5B/3B instead.

**Done =** a real on-GPU tok/s number in this dev log + the `cuda` feature flips on cleanly.

#### M7 — Metal (new backend, mirrors CUDA)

`Device::new_metal(0)` + the `metal` feature on the three candle crates. New
`MetalBackend` struct behind `#[cfg(feature = "metal")]`, identical shape to
`CudaBackend`. `DeviceChoice::Metal` added to the enum + CLI. macOS-only build
(`#[cfg(target_os = "macos")]` guard so it doesn't break Linux/Windows builds).

**Caveat to test:** candle issue #2818 — a Metal embedding-generation panic on
some shapes. Check against Qwen2 before declaring it done.

**Done =** `ambercore serve --device metal` runs on an Apple GPU with a tok/s
number. (Needs a macOS machine for the final verification — may be code-complete
this week, verified later.)

#### M8 — AMD stub (clean error, no fork)

New `AmdBackend` behind a `rocm` feature that **always errors** at resolve time:
*"AMD/ROCm support is blocked on upstream candle (no stable ROCm backend — see
ACRoad.md §7). AmberCore will wire it the day candle merges official ROCm."*
`DeviceChoice::Amd` added to the enum + CLI so the surface is ready. Zero
runtime cost, no experimental dependency.

**Why not the fork:** candle PRs #3424/#3801 are the only ROCm attempts — both
open, both stale, the author calls the code AI-generated and unsafe. Tracking it
as a git dep would risk the whole build for an unverified path. The stub keeps
the door open without the fragility.

**Done =** `ambercore serve --device amd` gives a clean, informative error
instead of "unknown device".

#### Build-matrix note

After M6/M7, AmberCore ships **three feature-gated GPU backends** (cuda/metal/
rocm-stub) + the always-on CPU backend. A single binary can't include all GPU
backends at once on one platform (CUDA needs Linux/Windows + nvcc; Metal needs
macOS), so releases are per-platform feature builds — same pattern candle itself
uses. The `auto` device choice should try each compiled-in GPU backend in turn
(CUDA → Metal) before falling back to CPU.

---

### 2026-07-30 — M5a perf + unlimited tokens (v0.6.0)
**Two changes, both in the spirit of "fully local, no excuses":**

**1. 3x CPU speedup via SIMD (the big M5 win).** Every prior test showed
`avx: false, f16c: false` — candle's quantized kernels have AVX2 code paths that simply
weren't being compiled in. Adding `target-cpu=native` to the build flips them on for the
building machine's exact CPU. Measured on Qwen2-0.5B / i5-12500 (which supports AVX2+F16C):

| Build | `avx` flag | tok/s | vs baseline |
|---|---|---|---|
| default (no SIMD) | false | **11.5** | 1.0x |
| `target-cpu=native` | true | **34.4** | **3.0x** |

Same output (greedy is deterministic), 3x faster. Baked into **`.cargo/config.toml`** so a
plain `cargo build --release` gets it automatically — no manual RUSTFLAGS needed. Trade-off
documented: the binary is no longer portable (crashes on CPUs lacking AVX2), which is fine
for local use but matters for distribution.

**2. Unlimited tokens (response + context).** AmberCore is local — there's no per-token
billing or remote quota, so arbitrary caps make no sense. Changed:
- `StopCondition.max_tokens: Option<usize>` — `None` = unlimited (stop only on EOS /
  `<|im_end|>`). The default is now `None`.
- The server (`/api/chat`) sets no cap — generation runs to the model's natural end-of-turn.
- The CLI `--max-tokens` is now optional (`Option<usize>`) — omit it for unlimited, or pass
  `--max-tokens N` for an explicit cap.
- Added **context-length awareness** to `Pipeline`: when a prompt exceeds the model's trained
  context (surfaced from GGUF `<arch>.context_length`), it `tracing::warn`s but does **not**
  block — the user can still send a long prompt, they just know quality may degrade.
- `LoadedEntry` + the CLI now surface `context_length` (e.g. "ctx 32768 tokens" in the build
  log line).

Verified: unlimited run stops cleanly on EOS ("2+2 equals 4." = 8 tokens, no cap);
explicit `--max-tokens 5` still caps ("Sure, I'd love" = exactly 5).

**Files touched:** `.cargo/config.toml` (new), `pipeline/mod.rs` (`StopCondition` +
`Pipeline.context_length` + the warning), `server/mod.rs` (`LoadedEntry.context_length`),
`server/chat.rs` (unlimited + passes context_length), `bin/ambercore.rs` (optional
`--max-tokens`, surfaces context_length).

### 2026-07-29 — M4 done (v0.5.0) — CUDA backend + device selection
**AmberCore now abstracts the compute target.** The `Backend` trait hands out the candle
`Device`; `CpuBackend` is always available, `CudaBackend` is feature-gated, and
`--device cpu|cuda|auto` selects at runtime.

```
$ ambercore serve --device cpu    # force CPU (default; always works)
$ ambercore serve --device cuda   # force CUDA (needs --features cuda + GPU)
$ ambercore serve --device auto   # try CUDA, fall back to CPU
```

Verified on this machine (no CUDA toolkit, so CPU-only):
- `--device cpu` → works (12 tok/s on Qwen2-0.5B)
- `--device auto` → logs "built without `cuda` feature; using CPU", falls back cleanly
- `--device cuda` → clean error: *"CUDA requested but AmberCore was built without the
  `cuda` feature. Rebuild with `cargo build --release --features cuda`"*

The CUDA code is **implemented and feature-gated** but **not yet run** — this machine has
an RTX 3050 8GB + current driver (591.86) but no CUDA toolkit (`nvcc`), which candle
requires at build time. The build instructions in §8 cover enabling it.

**What landed:**
- `backend.rs`: **rewritten.** `DeviceChoice` enum (`Cpu`/`Cuda`/`Auto`, `clap::ValueEnum`)
  parsed from the CLI. `resolve_backend(choice)` → `Box<dyn Backend>`: `Cpu` always; `Cuda`
  requires the feature (errors cleanly if off); `Auto` tries CUDA, warns + falls back to
  CPU on any failure. `CudaBackend` lives in a `#[cfg(feature = "cuda")] mod cuda` submodule
  and wraps `Device::new_cuda(ordinal)`. 4 new unit tests (CPU device, resolve cpu/auto,
  cuda-without-feature errors cleanly).
- `pipeline/mod.rs`: **refactored** `Pipeline` to hold `device: &candle_core::Device`
  instead of `backend: &dyn Backend`. The pipeline only ever needed the `Device`, so this
  removes an indirection and the awkward per-call `CpuBackend::new()` shim. The model's
  tensors already live on the right device (set at load time), so the pipeline just places
  input tensors on the same device.
- `server/mod.rs`: `ServerState::new(catalog, backend)` now takes the resolved backend;
  `LoadedEntry` gained a `device: candle_core::Device` field so the chat handler knows
  which device to place tensors on (was hardcoded CPU). `backend_name()` exposes it.
- `server/chat.rs`: removed the hardcoded `CpuBackend` — the blocking task now reads
  `device` from the `LoadedEntry`.
- `bin/ambercore.rs`: `--device` flag on both `serve` and `run` (default `cpu`).

**Bug found + fixed (not M4-related):** Qwen2-0.5B generation had started failing with
`index-select invalid index 248045 with dim size 151936`. Root cause: an earlier Qwen3.5
tokenizer download had **overwritten** the Qwen2 `tokenizer.json`, so the Qwen2 model was
being tokenized with the wrong (larger) vocabulary. Fixed by re-downloading the correct
Qwen2 tokenizer as `qwen2-0_5b-instruct-q4_k_m.tokenizer.json` (model-specific name) — the
model-specific tokenizer lookup I added in v0.4.1 (`<stem>.tokenizer.json`) prevents this
collision going forward.

**Key decision — `Pipeline` takes `Device`, not `Backend`:** the original design threaded a
`&dyn Backend` through the pipeline, but the pipeline only ever called `backend.device()`.
Switching to `&Device` directly is simpler, removes a trait-object indirection in the hot
path, and makes it obvious the model + input tensors must share a device. The `Backend`
trait is now purely a *selection* concern (resolved once at startup, then its `Device` is
what flows through the system).

### 2026-07-29 — Qwen3-8B verified + Qwen3.5 SSM finding (v0.4.1)
**A real, modern, tool-trained model now runs through AmberCore.** Registered
Qwen3-8B (4.7GB, Q4_K_M) from `N:/AI Models` and verified full chat + tool-calling:

```
POST /api/chat (tools: [get_weather])
→ {"message":{"content":"<think>...I need to use get_weather with Paris...</think>"},"done":false}
→ {"message":{"tool_calls":[{"function":{"name":"get_weather",
     "arguments":"{\"city\":\"Paris\"}"}}]},"done":true,"eval_count":99,"prompt_eval_count":146}
```

The model reasons in `<think>` mode, then emits a clean structured tool call — no marker
leakage in the stream, `arguments` as a JSON string (Phoenix contract).

**What landed:**
- `model/qwen3.rs` (new): wraps `candle_transformers::models::quantized_qwen3::ModelWeights`.
  Registered in `registry.rs` for arch `qwen3` and `qwen35`.
- `model/qwen3.rs` Qwen3.5 key-remap: when `arch == "qwen35"`, remaps metadata keys
  `qwen35.*` → `qwen3.*` before handing `Content` to candle's builder (candle hardcodes
  the `qwen3.` prefix). This fixes the metadata lookup for Qwen3.5 GGUFs.
- `tokenizer.rs`: `resolve_next_to` now tries `<model_stem>.tokenizer.json` (model-specific)
  before the generic `tokenizer.json` fallback — so multiple models with different
  vocabularies can share one directory.
- `server/chat.rs`: **simplified the tool-call streaming filter.** When tools are active,
  the generation callback buffers the full text and streams nothing during generation;
  after generation, `parse_tool_calls` strips the markers and the clean non-tool-call text
  is streamed in one shot. This replaced a fragile incremental filter that leaked closing
  markers. Phoenix doesn't need incremental streaming for tool-call turns (it waits for the
  terminal `tool_calls` event), so this is correct + simpler.

**⚠️ Qwen3.5 (4B/9B) CANNOT LOAD — hybrid SSM architecture.** Discovered by dumping the
tensor names: Qwen3.5 GGUFs contain `blk.N.ssm_conv1d`, `ssm_dt`, `ssm_alpha`, `ssm_beta`,
`attn_gate` — the signature of a **state-space model (Mamba-style) hybrid**, not a standard
transformer. candle 0.11's `quantized_qwen3` implements only the standard transformer
(`attn_q/k/v`, `ffn_gate/up/down`, `ffn_norm`), so it fails with "cannot find tensor
blk.0.ffn_norm.weight". **No tensor-name remap can fix this** — the architecture is
fundamentally different; it would require writing the entire Qwen3.5 hybrid SSM forward pass
from scratch (M6+ scale). Confirmed via the GGUF tensor dump + the
[EricLBuehler/candle-vllm#387](https://github.com/EricLBuehler/candle-vllm/issues/387)
report ("Model arch qwen35 not supported"). The `qwen35 → qwen3` key-remap I added is kept
(metadata-level) but insufficient on its own.

**Decision: Qwen3-8B is the reference tool-trained model.** It's a plain `qwen3`
transformer that candle supports. Qwen3.5 support deferred until candle upstream adds a
hybrid SSM model (or AmberCore implements one in M6+).

**Performance note:** Qwen3-8B runs at ~2.4 tok/s on CPU with no SIMD (vs ~12 tok/s for
the 0.5B). Model build takes ~20s (vs 0.5s). Enabling AVX2 + CUDA (M4) will improve both
substantially. The 8B is the right model for correctness verification; the 0.5B remains
the fast dev-iteration model.

### 2026-07-29 — M3 done (v0.4.0)
**AmberCore now supports Phoenix's full ReAct agent loop** via tool/function-calling.

Phoenix sends OpenAI-style `tools` and expects structured `tool_calls` back — it does
**not** parse tool calls from text (it relies on the provider's native function-calling).
Since candle gives us only raw forward passes, AmberCore implements a **text-protocol shim**
using the **Hermes `<tool_call>` format** that Qwen2.5/Qwen3 models are trained on:

```
$ curl -X POST .../api/chat -d '{
    "messages":[{"role":"user","content":"What is the weather in Tokyo? Use get_weather."}],
    "tools":[{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
    "stream":true,"temperature":0
}'
{"message":{"content":">"},"done":false}
{"message":{"tool_calls":[{"function":{"name":"get_weather",
    "arguments":"{\"city\":\"Tokyo\"}"}}]},"done":true,"eval_count":26,"prompt_eval_count":165}
```

The terminal line carries `tool_calls` with `arguments` as a **JSON string** (Phoenix's
exact wire contract — verified it parses cleanly with Phoenix's `parse_arguments`).

**What landed:**
- `server/tools.rs` (new): `render_tools_section(tools)` — injects a `## Tools` section
  into the system prompt with the Hermes emit format + each tool's full JSON Schema
  (name, description, pretty-printed parameters). `parse_tool_calls(text)` — extracts
  `<tool_call>{...}</tool_call>` blocks from the generated text, normalizes `arguments`
  to a JSON string, handles multiple calls, and drops unclosed/malformed blocks safely.
  9 unit tests covering rendering + all parse edge cases.
- `server/chat.rs`: when `req.tools` is non-empty, (1) the tools section is rendered into
  the system prompt, (2) the streaming callback runs a **filter** that suppresses
  `<tool_call>` markers from the content stream (buffering via `Rc<RefCell<String>>` +
  tracking in/out of marker blocks + holding back partial marker prefixes like `<tool_c`),
  (3) post-generation `parse_tool_calls` extracts the calls, (4) the terminal chunk emits
  them as `message.tool_calls`.
- `server/mod.rs`: registered the new `tools` module.

**Key decisions / findings:**
1. **Phoenix does zero text parsing for tool calls.** It sends `tools` and reads
   `tool_calls` — both structured. So AmberCore must own the full text↔structure
   translation. Confirmed by auditing `runtime.rs` + `tools/mod.rs`.
2. **Hermes format chosen** (`<tool_call>{...}</tool_call>`) because it's what Qwen2.5
   (Phoenix's default `qwen2.5-coder:7b`) and Qwen3 are trained on — this is vLLM's
   `--tool-call-parser hermes` format. Phoenix's actual models will emit it natively.
3. **Test-model caveat (honest):** the Qwen2-0.5B-Instruct we have for testing was NOT
   trained on tool tokens, so it's inconsistent — it emitted a correct `get_weather`
   call for the weather question but rambled past the file-read question. This is the
   *model's* limitation, not AmberCore's. The pipeline proved correct: when the model
   does emit the format, AmberCore parses + wires it perfectly. A Qwen2.5/Qwen3 model
   will be reliable.
4. **Streaming filter is conservative.** It holds back text once a `<tool_call>` marker
   begins, so the markers + inner JSON never reach the client as `content`. A stray
   partial char (`>`) can leak when the marker spans a token boundary — minor cosmetic
   artifact, doesn't affect the structured `tool_calls` output Phoenix consumes.
5. **`arguments` normalization:** the parser accepts `arguments` as either an object
   (re-serializes to string) or already a string (passes through), matching both how
   trained models emit it and how some fine-tunes do. Missing `arguments` → `"{}"`.

**Phoenix ReAct loop now works:** user message → AmberCore streams text or a `tool_calls`
event → Phoenix executes the tool → Phoenix sends back a `role:"tool"` message → AmberCore
folds it as a labelled user observation → model continues. The `to_chat_turns` mapping
(folding `tool`-role results) was already in place from M2.

### 2026-07-29 — M2 done (v0.3.0)
**AmberCore is now a drop-in Ollama replacement.** The HTTP server serves the exact
Phoenix wire contract on port 42069. Verified end-to-end with curl:

```
$ curl http://localhost:42069/api/tags
{"models":[{"name":"qwen2-0_5b-instruct-q4_k_m:latest"},{"name":"qwen2:0.5b"}]}

$ curl -X POST http://localhost:42069/api/chat -H "Content-Type: application/json" \
    -d '{"model":"qwen2:0.5b","messages":[{"role":"user","content":"What is 2+2?"}],"stream":true,"temperature":0}'
{"message":{"content":"2"},"done":false}
{"message":{"content":"+2"},"done":false}
{"message":{"content":" equals"},"done":false}
{"message":{"content":" 4"},"done":false}
{"message":{"content":"."},"done":false}
{"message":{},"done":true,"eval_count":8,"prompt_eval_count":32}
```

This is **exactly** the NDJSON shape Phoenix's `pump_stream` parser consumes: one JSON
object per line (no SSE `data:` prefixes), deltas on non-terminal lines, and a terminal
line carrying `done:true` + `eval_count` + `prompt_eval_count`. The `prompt_eval_count`
varies (27 with a system turn, 32 without) confirming the system message is folded into
the prompt correctly.

**What landed:**
- Activated `axum 0.8` + `tokio-stream` + `bytes` + `http`.
- `server/mod.rs`: **`ServerState`** — shared state holding the catalog (async mutex) +
  a lazily-populated model cache keyed by tag. `entry_for(tag)` does the cold-path
  load+build on `spawn_blocking` (candle parsing is sync I/O + CPU) and caches the
  result in an `Arc<std::sync::Mutex<LoadedEntry>>`. `app(state)` builds the router;
  `serve(port, state)` runs axum on `0.0.0.0:42069`.
- `server/tags.rs`: `GET /api/tags` wired to the live catalog (returns the tags verbatim
  — Phoenix compares by exact string equality).
- `server/chat.rs`: `POST /api/chat` — converts `protocol::ChatMessage` → `ChatTurn`
  (system/user/assistant mapped; `tool` results folded as labelled user observations
  for M2), formats ChatML, spawns generation on a blocking thread, streams
  `ChatChunk::delta(...)` lines as NDJSON, emits the terminal `done` chunk with counts.
- `bin/ambercore.rs serve`: now runs the real axum server via a multi-thread tokio runtime.

**Key decisions / issues found (and how resolved):**
1. **Concurrency model: per-tag std mutex + spawn_blocking.** candle's quantized forward
   is `&mut self` and CPU-bound, so a single model can't serve concurrent requests and
   mustn't run on the async executor. Each tag gets an `Arc<std::sync::Mutex<LoadedEntry>>`;
   `chat` locks it for the duration of `generate()` on a `spawn_blocking` thread. This
   serializes same-tag requests (acceptable for v1) while letting different tags run in
   parallel. M5's continuous batching will lift the per-tag serialization.
2. **Borrow-splitting in the generation closure.** Can't take `&mut model` and
   `&tokenizer` from the same `LoadedEntry` guard in one struct literal — the borrow
   checker rejects the overlapping borrow. Resolved by binding `&mut entry` once, then
   deriving the two field refs from it (the documented split-borrow pattern).
3. **axum 0.8 handler signature.** Handlers must use `State(state): State<ServerState>`
   (the extractor), not take `ServerState` directly — the direct form doesn't satisfy
   `Handler`. First attempt had `tags::list(state: ServerState)`; fixed to the extractor form.
4. **`Bytes::from(format!(...))`** for NDJSON lines (there's no `Bytes::from_owner`).
   Each line is a fresh allocation; fine for M2's throughput. M5 may pool buffers.
5. **Error handling.** Model-not-found / generation-failure both produce a clean HTTP 500
   with a message. Mid-stream generation failures still emit a terminal `done` chunk so
   the client unblocks (Phoenix's `pump_stream` waits for a `done:true` line).

**Phoenix integration (zero code change):** set `ollama_url = "http://localhost:42069"`
in Phoenix's `config.toml`. Phoenix's `health.rs` will probe `/api/tags`, see the tags
verbatim, and mark the model available; its `ollama.rs` will stream `/api/chat` NDJSON
unchanged. **AmberCore now replaces Ollama for Phoenix.**

### 2026-07-28 — M1 done (v0.2.0)
**AmberCore runs full streaming chat.** Verified end-to-end against Qwen2-0.5B-Instruct:

```
$ ambercore run --model qwen2:0.5b --prompt "What does the Rust function println! do?" --temperature 0
--- generation ---
The `println!` function in Rust is used to print a string to the standard output
stream (usually the standard output device, such as the console or the terminal).
It takes a string as its argument and prints it to the standard output stream.
--- end (51, 12.08 tok/s) ---
```

The model produces coherent multi-sentence answers, **stops cleanly on EOS**
(`<|im_end|>`), and streams token-by-token. Three test prompts all gave correct,
well-formed responses ("2 + 2 equals 4.", "Hello! How can I assist you today?",
and the println! description above).

**What landed:**
- `pipeline/sampler.rs`: **rewrote** around `candle_transformers::generation::LogitsProcessor`
  + `Sampling` enum. `SampleParams { temperature, top_k, top_p, seed }` translates to the
  correct `Sampling` variant (ArgMax / All / TopK / TopP / TopKThenTopP). `Sampler` owns the
  processor so RNG state advances correctly across tokens. Replaces the M0 hand-rolled greedy.
- `pipeline/mod.rs`: new **`generate()`** streaming loop. Prefills the prompt once at
  `index_pos=0`, then decodes one token per step at `index_pos = prompt_len + step - 1` —
  the model's internal KV cache makes each step O(1) in sequence length. Streams stable
  text via `StreamingDecoder`. Stops on `StopCondition { max_tokens, stop_tokens }`.
  Returns `GenStats { output_tokens, prompt_tokens, prefill_secs, decode_secs }` +
  `tokens_per_sec()`. Kept `decode_one()` for M0 compatibility.
- `tokenizer.rs`: added **`StreamingDecoder`** (inlined from candle's `TokenOutputStream` —
  same logic, no `candle-examples` dep). Solves the BPE partial-token problem: only emits
  the stable text prefix as tokens arrive. Added **`format_qwen_chatml()`** — the ChatML
  `<|im_start|>` template Qwen2-Instruct is trained on, with default system turn injection
  + open-ended assistant prime.
- `error.rs`: added `Error::Candle(#[from] candle_core::Error)` so `?` works on candle
  Results directly (cleaner than `.map_err` everywhere).
- `bin/ambercore.rs run`: full streaming chat — flags for `--temperature`, `--top-k`,
  `--top-p`, `--seed`, `--max-tokens`, `--raw`. Resolves `<|im_end|>` + `<|endoftext|>`
  as stop tokens from the tokenizer. Streams deltas to stdout, prints tok/s at the end.

**Key decisions:**
1. **KV cache lives inside the model, not in `kv_cache.rs`.** candle's quantized qwen2
   caches K/V internally keyed by `index_pos`; our pipeline just passes the advancing
   position each step. The `pipeline/kv_cache.rs` module is vestigial — kept as a stub
   for a future explicit paged-cache (M5+, for continuous batching). Documented this.
2. **Used candle's `LogitsProcessor` instead of hand-rolling sampling.** Correct softmax +
   multinomial numerics, battle-tested, and free. Our `SampleParams` is just a friendly
   config layer over the `Sampling` enum. A unit test verifies seeded reproducibility.
3. **Inlined `TokenOutputStream` as `StreamingDecoder`** rather than depending on
   `candle-examples` (which is an examples crate, not a library dep). Same algorithm,
   ~60 lines, no extra dependency.
4. **ChatML formatting is Qwen-specific for now.** `format_qwen_chatml` hardcodes the
   `<|im_start|>` template. When we add other architectures (llama, etc.), this becomes
   a per-architecture template function — likely loaded from the GGUF's
   `tokenizer.chat_template` metadata in a future milestone.

**Performance note:** ~12 tok/s on CPU with **no SIMD** active (`avx: false`). The candle
build didn't enable AVX2 for this target. Enabling it (RUSTFLAGS=`-C target-cpu=native`
or a CPU-detection feature) should roughly double throughput. M5 will tackle this.

### 2026-07-28 — M0 done (v0.1.0)
**AmberCore decodes real tokens from a real GGUF on CPU.** Verified end-to-end against
Qwen2-0.5B-Instruct (Q4_K_M, 380MB):

```
$ ambercore run --model qwen2:0.5b --prompt "The capital of France is"
loaded GGUF: arch=qwen2 name=Some("qwen2-0_5b-instruct") (in 1.29s)
built qwen2 model (in 0.50s)
decoded 1 token in 0.374s (5 prompt tokens) → id=12095 text=" Paris"
```

All four test prompts produced the correct next token (` Paris`, ` time`, ` `, `,`),
proving the full pipeline: **GGUF load → tensor read → qwen2 build → tokenize →
forward → argmax → decode**.

**What landed:**
- Activated the Candle stack: `candle-core/nn/transformers 0.11`, `tokenizers 0.21`,
  `hf-hub 0.3`.
- `backend.rs`: `Backend::device()` now returns the real `candle_core::Device`.
  `CpuBackend` returns `Device::Cpu`.
- `model/gguf.rs`: real GGUF loading via `candle::quantized::gguf_file::Content::read`.
  `LoadedModel` owns the open file + parsed `Content`, surfaces `arch` + key metadata
  (context_length, eos_token_id, block_count, embedding_length) as strings.
- `model/registry.rs`: `DynModel` trait now carries `forward(&mut self, &Tensor, index_pos)
  -> Result<Tensor>`. `build(&mut LoadedModel, &Device)` dispatches on architecture.
- `model/qwen2.rs`: wraps `candle_transformers::models::quantized_qwen2::ModelWeights`,
  built via `from_gguf(content, &mut file, device)`.
- `tokenizer.rs`: renamed to `TokenizerWrapper`, loads a sibling `tokenizer.json` via the
  `tokenizers` crate. `encode` / `decode` / `token_to_id` live.
- `pipeline/mod.rs`: real `decode_one` — encode → `[1, seq]` tensor → forward → squeeze
  batch → f32 → argmax → decode.
- `pipeline/sampler.rs`: real `greedy` argmax.
- `bin/ambercore.rs run`: full wiring — load → build → tokenize → decode_one → print,
  with load/build/decode timings + SIMD capability reporting.
- Added `.gitignore` (`/target/`, `*.gguf`, `tokenizer.json`, `manifest.json`).

**Key decisions / issues found (and how resolved):**
1. **`DynModel::forward` returns `[batch, vocab]`, not `[batch, seq, vocab]`.** The
   quantized qwen2 `forward` already slices to the last token internally
   (`x.i((.., seq_len-1, ..))`). First attempt double-sliced and failed
   (`to_vec1: unexpected rank, expected 1, got 0`). Fixed by documenting the contract
   on `DynModel::forward` — each architecture is responsible for returning last-position
   logits as `[batch, vocab]`. Future architectures (llama, etc.) must match this.
2. **API path corrections from research** (not the versions guessed in the scaffold):
   - `gguf_file` lives in `candle-core::quantized` (re-exported as
     `candle::quantized::gguf_file`), NOT `candle_transformers`.
   - The quantized model is `candle_transformers::models::quantized_qwen2::ModelWeights`
     (not `models::qwen2` — that's the unquantized one).
   - Latest published versions: candle-core/nn/transformers all at **0.11.0**
     (scaffold guessed 0.8).
3. **`max_by` returns the LAST element on ties** (Rust stdlib behavior). A unit test
   caught this — fixed the test expectation. Behavior is deterministic and fine.
4. **Tokenizer is loaded from a sibling `tokenizer.json`** (HF format), not reconstructed
   from GGUF metadata. This is the canonical arrangement and matches how Phoenix's
   future `pull` feature will lay out files. Reconstructing from `tokenizer.ggml.*`
   metadata is a possible future enhancement.

**Test model:** Qwen2-0.5B-Instruct Q4_K_M lives at `~/.ambercore/models/` (user data
dir, outside the repo) registered as `qwen2:0.5b`. The `tokenizer.json` sits next to it.
Used for M0 verification; safe to delete.

### 2026-07-28 — Scaffold (v0.0.0)
- Created `AmberCore/` at project root.
- Wrote this `ACRoad.md` (decisions, Phoenix contract, roadmap, layout, deps).
- Wrote `README.md` with Phoenix drop-in instructions.
- Created the full module tree as **compiling stubs** with documented responsibilities.
- Implemented the real `Backend` trait + `CpuBackend` shell.
- Wrote the `ambercore` CLI skeleton (`serve` defaulting to **42069**, `run`, `register`)
  with subcommands wired but stubbed.
- `Cargo.toml` configured with the dependency list above + a `cuda` feature.
- Verified `cargo check` passes cleanly.

**Next (M1):** full generation loop — KV cache reuse across decode steps, stochastic
sampling (temperature/top-k/top-p), streaming token output, EOS handling, multi-token
`ambercore run` chat.

---

## 8. Building with CUDA (M4)

The `cuda` Cargo feature gates the CUDA backend. It requires the **CUDA toolkit** (`nvcc`)
at build time — the GPU driver alone is not enough.

### Prerequisites
1. An **NVIDIA GPU** (this machine: RTX 3050 8GB) + a current driver.
2. The **CUDA Toolkit** installed (provides `nvcc`). Download from NVIDIA; the default
   install path on Windows is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`.
3. `nvcc` on `PATH` (verify: `nvcc --version`).
4. Optionally set `CUDA_COMPUTE_CAP=86` for the RTX 3050 (compute capability 8.6) so candle
   compiles kernels for your exact GPU. Other cards: see
   <https://developer.nvidia.com/cuda-gpus>.

### Build
```bash
# From the AmberCore directory:
cargo build --release --features cuda
```

> **M6 (2026-08-12): DONE — feature propagates + builds green.** `cuda` now reads
> `["candle-core/cuda","candle-nn/cuda","candle-transformers/cuda"]`. On Windows the CUDA
> build additionally requires (1) the MSVC env loaded (`vcvars64.bat`) so nvcc finds `cl.exe`,
> and (2) `set CL=/Zc:preprocessor /std:c++17`, because CUDA 13's CCCL headers reject MSVC's
> traditional preprocessor as a fatal error (C1189). Build:
> `CUDA_COMPUTE_CAP=86 cargo build --release --features cuda` → 63 MB binary, 3m20s.
> **Verified:** the build host runs CUDA **13.3**; the driver was updated from 591.86 (13.1) to
> **610.88 (13.3)** to clear `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (no rebuild). On-GPU results:
> Qwen2-0.5B **58 tok/s**, Qwen3-8B **26.6 tok/s** @ 7.1/8 GB VRAM.

### Run
```bash
# Force CUDA (errors if no GPU / driver mismatch):
ambercore serve --device cuda
# Or auto-select (CUDA if available, else CPU):
ambercore serve --device auto
ambercore run --model qwen3:8b --device auto --prompt "Hello"
```

### Status (2026-08-12)
**M6 DONE — verified on the RTX 3050 (driver 610.88 / CUDA 13.3).** Qwen2-0.5B = 58 tok/s,
Qwen3-8B = 26.6 tok/s @ 7097/8192 MiB VRAM (86%). `cuda` propagates to all three candle crates and
compiles against CUDA 13.3 + MSVC (`CL=/Zc:preprocessor /std:c++17`; 63 MB binary). The original
driver (591.86 / CUDA 13.1) was updated to clear a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` skew (no
rebuild). Quantized GGUF kernels ship in candle's `quantized/cuda.rs`, so Q4_K_M models run on GPU.
VRAM watch: the 8B Q4_K_M (~5 GB)
is tight on 8 GB — test the 0.5B/1.5B/3B first if the 8B OOMs.

---

### 2026-08-20 — `qwen35` mapping removed + pull-time architecture validation

**Bug found:** the `"qwen3" | "qwen35"` registry mapping added 2026-08-12 was
wrong. A real `qwen35` GGUF (Qwen3.5 hybrid SSM — block tensors are `ssm_*`,
fused `attn_qkv`, `attn_gate`, `post_attention_norm`; there is no `ffn_norm`
and no separate q/k/v) is NOT qwen3-layout compatible and crashed inside
`qwen3::from_gguf` with `cannot find tensor info for blk.0.ffn_norm.weight`.
It had never actually been exercised until Phoenix pulled `Qwen3.8-9B-Q4_K_M`.

**Fixes:**
1. `model/registry.rs`: removed `"qwen35"` from the qwen3 arm → clean
   `unsupported architecture: qwen35` failure again (hybrid SSM support stays
   deferred — candle lacks the kernels; see the earlier qwen35 note).
2. `model/registry.rs`: new `SUPPORTED_ARCHS` const + `is_supported()` — the
   single source of truth, kept beside `build()` so pull-time validation and
   load-time dispatch can never drift apart.
3. `model/gguf.rs`: new `probe_arch(path)` — reads only the GGUF header and
   returns `general.architecture` (cheap at any model size; tensor data is
   never touched).
4. Phoenix's `pull_ambercore_model` probes the architecture right after the
   GGUF download and rejects unsupported ones with a plain-language error
   BEFORE the tokenizer fetch + registration. The file is kept on disk
   (usable if support lands later) but never registered, so it never shows a
   broken Run button.

Tests: `supported_archs_match_build_arms` + `probe_arch_reads_minimal_gguf_header`
(hand-crafts the smallest valid GGUF: magic + version 2 + 0 tensors + one
string KV).

---

### 2026-08-20 — GPU/hardware status API (retroactive entry; synced 2026-08-22)

Landed 2026-08-20 alongside the CUDA installers but was never logged here:

1. `backend.rs`: new `GpuInfo { name, vram_total_mb, vram_used_mb }` (serde) +
   `Backend::gpu_info()` default trait method (`None` on CPU; the CUDA override
   queries device name + memory via candle's `cudarc` re-export).
2. `server/telemetry.rs`: new `HardwareStatus { backend, cpu, cpu_cores,
   ram_total_mb, os, gpu: Option<GpuInfo> }` (`skip_serializing_if` on the
   optional fields).
3. `server/mod.rs`: `ServerState::hardware_status()` merges the boot-time
   CPU/RAM/OS snapshot with the live `gpu_info()` per call.
4. Phoenix surfaces it in-app via the `get_hardware_status` Tauri command
   (Telemetry panel).

### 2026-08-22 — all four AmberCore copies synced

The engine exists in 4 trees (embedded `phoenix-agent/ambercore`, sibling
`AmberCore/`, `ALPHA/PhoenixAgent/ambercore`, `ALPHA/AmberCore-Server/ambercore`)
and had drifted: the sibling + server copies lacked the GPU-status work above,
and the **server copy still carried the buggy `"qwen3" | "qwen35"` mapping**
fixed 2026-08-20. Verified there are no intentional per-copy differences
(identical Cargo.toml/features), then synced everything to the embedded tree
as the single source of truth. `cargo test` green in the synced copies.

---

### 2026-08-22 — architecture expansion: 15 archs (gemma, phi, glm4, mixtral, llama, MoE, ...)

AmberCore went from 4 architectures (qwen2, qwen2_v2, qwen3, llama-stub) to
**15 fully working ones**, all pure Rust on candle 0.11:

| Family | Arch strings | Path |
|---|---|---|
| Qwen | `qwen2`, `qwen2_v2`, `qwen3` | candle direct (unchanged) |
| Qwen MoE | `qwen3moe` | **ported copy** of candle's quantized_qwen3_moe + the KV-cache clear candle 0.11 lacks (same replica-reuse bug class as dense qwen3) |
| Llama | `llama` | candle quantized_llama — **the old stub now really loads** (Llama 1/2/3, Mistral-7B conversions, TinyLlama, Yi, SmolLM) |
| Mixtral | `mixtral` | metadata remap `mixtral.*`→`llama.*` into candle llama's **MoE path** (router `ffn_gate_inp` + per-expert FFNs; 8x7B / 8x22B) |
| Gemma | `gemma`, `gemma2`, `gemma3` | candle quantized_gemma3 (probes all three prefixes itself) |
| Phi | `phi2`, `phi3` | candle quantized_phi / quantized_phi3 — **`phi3` also covers Phi-4** (converts with the phi3 arch incl. long-rope) |
| GLM | `glm4` | candle quantized_glm4 (F32) |
| Liquid | `lfm2` | candle quantized_lfm2 |
| Qwen2-layout | `starcoder2`, `internlm2` | metadata remap into candle qwen2 (same tensors, different namespace) |

**Chat templates per family** (`tokenizer.rs`): ChatML (Qwen + relatives),
Gemma (`<start_of_turn>`, system folded into first user turn), Phi3
(`<|user|>…<|end|>`), GLM4 (`[gMASK]<sop>`), Mistral/Llama-2 (`[INST]`),
Llama-3 (headers + `<|eot_id>`). The ambiguous `llama` arch resolves its
template from the tokenizer's special tokens (`<|start_header_id|>` → Llama3,
`<|im_start|>` → ChatML/Hermes, else `[INST]`). **Stop tokens are arch-aware
now**: the GGUF's `<arch>.eos_token_id` (by id) + per-family markers
(`<end_of_turn>`, `<|end|>`, `<|user|>`/`<|observation|>`, `<|eot_id|>`, `</s>`).

**Still unsupported, by design** (hybrid-SSM / MLA-MoE families candle has no
kernels for): `qwen35`, `deepseek2`/`deepseek_v3`, `kimi_k2`/`kimi_linear`,
`gemma3n`, `granite`, `olmo`, `nemotron`, `exaone`, `hunyuan`, `llama4`. They
fail at load with a clean error listing the supported set — same honest-refusal
policy as qwen35. Porting deepseek/kimi (MLA + MoE) is the big remaining item.

Tests: 54 passing (+7: per-template formatting, arch map, mixtral remap).

---

### 2026-08-22 (b) — CUDA driver-mismatch guard + per-model folder layout

**The alpha bug:** a tester on an older NVIDIA driver hit
`qwen2 from_gguf: DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION, "the
provided PTX was compiled with an unsupported toolchain.")` mid-model-load.
Cause: candle embeds its kernels as **PTX** JIT-compiled by the *installed
driver* at runtime; the build toolkit (12.8 for `-CUDA12.exe`, 13.x for
`-CUDA.exe`) was newer than the tester's driver, so the JIT refused the PTX.

**Fixes (engine side):**
1. `build.rs` (new): under the `cuda` feature, records the toolkit version
   from `nvcc --version` into `AMBERCORE_CUDA_TOOLKIT` at build time.
2. `CudaBackend::new` now **warms up one kernel immediately** (a 2-element
   add) — the PTX JIT happens at construction, not at first model load. A
   mismatch fails with a translated message naming the toolkit, the minimum
   driver (12.8 → ≥ 570.51, 13.0 → ≥ 580.65, table in `backend.rs`), and the
   driver's own reported CUDA version (`cuDriverGetVersion`). Under
   `auto` (the embedded engine's default) the resolver catches it and **falls
   back to CPU**, so the app still runs — the Telemetry panel shows `cpu`.
3. Model-build errors get the same translation as defense in depth
   (`server::build_loaded_entry`).
4. Note: the PTX targets the build GPU's class (compute 8.6) and JITs forward
   to newer GPUs (RTX 40/50 fine); pre-Ampere GTX needs the CPU build.
   Multi-arch PTX would need a candle-kernels fork — deferred.

**Per-model folder layout (catalog):** `Catalog::load` now scans one level of
subfolders (`<models_dir>/<model>/<model>.gguf`, stored folder-relative in the
manifest) alongside flat files. Phoenix pulls (v0.8.2+) create the subfolder
per model, so models with different vocabularies can never share a tokenizer;
flat layouts keep working. Split/sharded GGUFs are rejected at pull time.
Tests: 58 (engine) / 31 (Phoenix) green.

---

### 2026-08-28 — the architecture expansion: 9 new archs (gemma4, granite×3, nemotron, minimax-m2, deepseek2×2)

**Trigger:** `Error: Model error: ambercore: not found: model tag
'gemma-4-E2B-it-Q4_K_M' not in catalog`. Root cause chain: Phoenix's
`pull_ambercore_model` probes the GGUF arch → `is_supported("gemma4")` was
false → pull aborted before registering → tag never entered `manifest.json` →
chat failed with NotFound. The fix is real adapters, not tag tricks.

**New pattern (#4 — hand-built):** the first three patterns are direct use /
remap / port of candle-transformers models. These nine are built from
llama.cpp's reference graphs (`llama-graph.cpp` per-arch build functions)
directly over `QMatMul` + `ConcatKvCache`, sharing two new modules:
- `model/common.rs` — causal mask (optional sliding window), namespaced GGUF
  metadata reader (`Meta`), RoPE scaling parse (linear + YaRN/llama3 blend;
  llama3 `low/high_freq_factor` keys take priority over YaRN betas), cos/sin
  tables with both apply conventions (interleaved `rope_i` = GGUF
  pre-permuted Q/K for granite/deepseek/nemotron; half-split `rope` = neoX
  for gemma4/minimax), and a test-only synthetic-GGUF writer.
- `model/moe.rs` — routed MoE (softmax/sigmoid gating, optional route bias,
  top-k renorm/scale) over grouped expert stacks. **Discovery:** candle's
  `moe_gemm_gguf` is CUDA-only — meaning the pre-existing qwen3_moe adapter
  never worked on CPU builds. Fixed here with a per-expert path that slices
  each expert's quantized bytes straight out of the 3D stack via
  `QStorage::from_data` + `QTensor::new` (stays quantized, no dequant blowup),
  then gathers routed tokens per expert and scatter-adds with `index_add`.

**The adapters** (each with synthetic-GGUF end-to-end tests — write tiny
Q8_0/F32 GGUF → `LoadedModel::load` → registry build → forward+decode pass):
1. `granite.rs` — granite / granitemoe / granite_swa: llama layout + IBM's
   scale multipliers (embedding×emb_scale, attn/ffn outputs ×residual_scale
   before the residual add, logits ÷logit_scale, optional attention.scale
   overriding 1/√d). Interleaved rope. Optional per-layer sliding window.
2. `nemotron.rs` — LayerNorm (weights+bias, F32) attention + squared-ReLU FFN
   (no gate tensor). Interleaved rope. The "Super" latent-FFN variant
   (narrow→wide up-proj then expand) is rejected with a clear message.
3. `minimax_m2.rs` — QK-norm over the FULL projected dim before head split
   (not per-head like qwen3), half-split rope, sigmoid-gated MoE with the
   required `ffn_exp_probs_b.bias`, no shared expert.
4. `gemma4.rs` — the arch from the bug report. Embeddings ×√hidden;
   per-layer embeddings (PLE): `per_layer_tok_embd` stays quantized and rows
   are fetched lazily per forward via `QTensor::embedding` (dequantizing a
   6B-token-scale PLE table would blow RAM); mixed =
   (proj_norm(model_proj(x)/√h) + √ple_dim·embd)/√2; per-layer
   inp_gate→gelu→proj ×ple_l + post_norm sandwich. Cross-layer KV sharing
   (`attention.shared_kv_layers`: layers ≥ n reuse slot n-1 full / n-2 swa)
   implemented with a custom `KvSlot` (ConcatKvCache has no read-without-
   append). Dual head dims + dual rope bases (1M global / 10k SWA) with the
   `sliding_window_pattern` bool-array. Weightless RMSNorm on V, tanh
   softcapping via `1 − 2/(e²ˣ+1)`. gemma4-MoE (ffn_gate_inp present)
   rejected cleanly for now.
5. `deepseek2.rs` — MLA in absorbed form: q_nope absorbed through a
   dequantized `wk_b` (n_head, kv_lora, qk_nope); one shared latent KV head
   [kv_cmpr | k_pe]; V = kv_cmpr decompressed by `wv_b` after attention;
   kq_scale = mscale²/√key_length_mla with YaRN mscale compensation from
   `rope.yarn_log_mul`. Q-projection via q_lora (wq_a→norm→wq_b) or direct.
   Leading dense blocks (`leading_dense_block_count`, V3.1+), sigmoid/softmax
   MoE + shared expert, `ffn_exp_probs_b.bias`. Covers `deepseek2` (V2/V3/R1
   distilled + **Kimi K2**, whose GGUFs report deepseek2) and `deepseek32`
   (V3.2 — its DSA sparse-attention indexer tensors are detected, warned,
   and ignored; full DSA support deferred).

**Registry:** `SUPPORTED_ARCHS` 15 → 24. Chat templates: new Granite,
DeepSeek (full-width `<｜User｜>`/`<｜Assistant｜>`/`<｜end▁of▁sentence｜>`),
Kimi (`<|im_user|>user<|im_middle|>…<|im_end|>`, disambiguated from
deepseek2 arch by probing the tokenizer for `<|im_middle|>`) and MiniMax
(`]~!b[]~b]system … [e~[` punctuation-soup tokens) — all verified verbatim
against the vendors' chat_template.jinja files. gemma4→Gemma template,
nemotron resolved llama-style by tokenizer probe. Stop markers extended for
every new family.

**Tag-spelling fix:** Phoenix registers pulled models under the bare file
stem while the catalog scan derives `<stem>:latest` — a `:latest`-less
lookup could miss a scan-registered model (the error message above reads
exactly like that failure mode). `Catalog::resolve` now tries exact → stem
→ `:latest`; the server canonicalizes tags before pooling so alias
spellings share one replica pool, and the NotFound error lists known tags.

**Deliberately unsupported** (clean error, documented in the registry test):
hybrid/SSM families whose recurrent kernels candle lacks — qwen35/qwen35moe
(**this is where Ornith ships**), kimi-k3, kimi-linear, nemotron_h(+moe),
granitehybrid, graniteswitch, deepseek4, minimax-01, minimax-m3 (DSA
indexer), qwen3next, gpt-oss (MXFP4), llama4, gemma3n. These need new
kernel work in candle itself, not adapters.

**Known follow-ups:** gemma4-MoE, qwen2moe/glm4moe over the new RoutedMoe,
gemma3n, minimax-m3 DSA. Kimi/MiniMax template exactness validated against
their published jinja; worth a smoke-test against a real pull.

Tests: 79 (engine, all 4 trees synced and green) / cargo check --features
cuda clean.

---

### 2026-08-30 — qwen35: the hybrid gated-delta-net adapter (Qwen 3.5, incl. Ornith)

**Pattern #4's hardest resident.** Qwen 3.5 mixes two layer kinds per
`attention.recurrent_layers` / `full_attention_interval` (every 4th layer is
full attention — 6 of 24 on the 0.8B): GDN **gated delta net** recurrent
layers (fused `attn_qkv` + `attn_gate` z-gate, causal depthwise `ssm_conv1d`
+ silu, L2-normalized per-head q/k, per-head scalar decay
`softplus(α(x)+dt)·ssm_a`, `β = sigmoid`, per-head gated RMSNorm, `ssm_out`)
and full-attention layers (fused **Q+gate** `attn_q` — 2·head_dim per head —
with `sigmoid(gate)` output scaling, GQA 8/2 heads × 256 dims, per-head q/k
norms, partial rope over 64 of 256 dims, base 1e7; the MRoPE sections
degenerate to one contiguous rope for text). Post-norm sandwich
(`attn_norm → mix → +res → post_attention_norm → FFN → +res`); tied
embedding/output head kept quantized (`QTensor::embedding` lazy rows).

**Two bugs found by real-model validation** (synthetic tests passed both —
the oracle GGUF was unsloth's `Qwen3.5-0.8B-Q4_K_M`, plus Qwen's config.json
and the transformers `Qwen3_5` kernel as ground truth):

1. **The recurrence is a DELTA RULE, not an additive SSM.** The state update
   subtracts what the old state already predicts — `S ← exp(g)·S +
   k⊗((v − Sᵀk)·β)` — and q is scaled by `head_dim^-½`. My first version used
   the Mamba-style additive form (`S ← exp(g)·S + β·k⊗v`): loads fine,
   generates pure garbage. Prefill runs the chunked **UT-transform kernel**
   (WY representation from the FLA/transformers reference):
   `(I+M)⁻¹ = Σ(−M)^k` via log₂(L) nilpotent doublings, `v_new = u −
   k_cumdecay·S`, `o = exp(G)·(QS) + ((QKᵀ)⊙D)·v_new`,
   `S ← exp(G_L)·S + k_decayᵀ·v_new`. Decode = the stepwise form (the
   `len == 1` branch). `chunked_delta_rule_matches_stepwise` pins the two
   paths equal on random data.
2. **`eye(n).cumsum(1)` is UPPER-triangular.** Causal needs `cumsum(0)`
   (1 where j ≤ i). One argument, total garbage out.

Plus the sneaky one: a stride-0 `broadcast_as` view of the identity fed into
**batched matmul** silently corrupts all heads but the first —
`.contiguous()` before `broadcast_matmul` (single-head probes all passed
while batched runs diverged; that cost an hour).

**Numbers (0.8B Q4_K_M, CPU):** debug 0.95 tok/s → **release 23.9 tok/s**
decode; coherent chat with proper `<think>` block handling and clean
`<|im_end|>` stops; model+tokenizer pulled straight from HF into
`~/.ambercore-dev/models` for the validation. Registry: `SUPPORTED_ARCHS` 24
→ **25** (`qwen35`; `qwen35moe` still rejected — no released GGUF to pin its
FFN names). `KvSlot` promoted from gemma4 into `common.rs` (sparse hybrid KV:
only the 6 full-attention layers allocate). Template: ChatML (verified
against the GGUF's embedded template). MTP/NextN tensors warned + ignored.
Engine tests **83** ×4 trees green; `cargo check --features cuda` clean.

---

### 2026-09-05 — native F16 safetensors loading (the content shim)

**Zero-rewrite F16 path.** Instead of a parallel dense-weight surface for
every adapter, loading a `.safetensors` model synthesizes an in-memory GGUF
`Content` whose tensor entries point **straight at the safetensors mmap**
(`st_shim.rs`): weights stay F16/BF16 in place, `QMatMul` handles unquantized
GGUF dtypes natively, and every adapter works unchanged. Hparams come from the
sibling `config.json` (synthesized into the arch's usual metadata keys), with
per-arch HF→GGUF tensor name maps for **qwen3** and **qwen35** (qwen35's
`q_proj` already carries the fused attention gate — zero-copy; `conv1d`
`(c,1,k)` is a metadata-only squeeze). Tensors needing real math before they
match the GGUF layout (`A_log → −exp(A_log)`, dtype-aware F16/F32 decode)
materialize into a small in-memory **patch region** that sits in front of the
file in the virtual address space (`PatchedReader` routes reads by offset).

Two shim bugs found by the synthetic e2e tests: (1) a directly-constructed
`Content` stores **candle** dims — `Content::read` is what reverses GGUF ne
order, so the HF (out, in) layout passes through as-is; (2) zero-copy offsets
bake in the patch length, so **transforms map first** (pass 0) and zero-copy
tensors second — mapping interleaved shifted the file region by the patch
size and corrupted exactly 4 bytes of the first F16 tensor (one NaN logit was
the tell).

**Plumbing:** `LoadedModel.file` became `ModelFile` (GGUF file or patched
reader — adapters are generic over `Read+Seek`, so nothing else changed);
`probe_arch` reads `model_type` from config.json (qwen3, qwen3_5_text →
qwen35); engine catalog + Phoenix collect/delete accept `.safetensors`; the
pull flow downloads the repo's `config.json` next to F16 weights and rejects
sharded checkpoints for both formats (`-00001-of-000NN`).

**Validated on the real thing:** Qwen3-0.6B `model.safetensors` (F16, 1.5 GB,
310 tensors, zero copies) → coherent chat with proper `<think>` handling at
**7.6 tok/s** CPU (release; the Q4_K_M GGUF of the same model is ~2× faster
but 2.5× larger — the memory/speed trade you'd expect). Engine tests **86**
×4 trees green; cuda check clean. Follow-ups: sharded repos (multi-region
reader), more arch maps (llama/gemma3), quantize-on-load to recover Q4
footprints, BF16 CUDA path.

### 2026-09-05 (b) — sharded safetensors repos + download ETA on the Models nav

**Multi-region `PatchedReader`.** The virtual address space became
`[patch | shard 1 | … | shard N]`: `index.json`'s `weight_map` next to the
entry file yields the sorted shard list, every shard's header is parsed, and
tensor offsets bake in `patch.len() + shard base + data_start` — so a tensor
in shard 7 is a plain `seek` into shard 7, no merging, no copies. Shards
without an `index.json` still load as single files. `check_vram` sums sibling
shards via the same weight map (the pre-flight was seeing only shard 1's size
and under-refusing); the catalog registers only `-00001-of-` shards so the
models list doesn't show every shard as a model.

**Phoenix pull** (`shard_pull.rs`): F16/safetensors URLs no longer stop at
"sharded checkpoints rejected" — the pull fetches `index.json`, HEADs every
shard for a known total, then downloads each shard with one cumulative
progress stream (per-pull bar + the nav ETA see one smooth total, not N
restarts), then `config.json`, tokenizer, validation, and registration of the
first shard. Honest limits: sibling fetches assume the repo's `main` revision
(no rev pinning), and a missing `index.json` is a hard error for sharded
repos rather than a blind filename sweep.

**Validated on real weights:** the real Qwen3-0.6B safetensors split by layer
parity into 2 shards (157 + 154 tensors) with a generated `index.json` →
release CLI loads all 310 tensors across both files and chats coherently
at 7.2 tok/s CPU. (First attempt "failed" because the release binary
predated the code — timestamps before conclusions.) Synthetic 2-shard e2e
test + reader-routing unit test in-tree.

### 2026-09-05 (c) — the Qwen3.5-4B incident: three silent bugs, ACRoad §6b born

User hit `sample: A weight is negative, too large or not a valid number`
(rand_distr rejecting NaN softmax weights) running the pulled
**Qwen3.5-4B-Q4_K_S** through the app on CUDA. Three separate defects, all
invisible to the 0.8B validation and the synthetic tests:

1. **Partial-rope `cat` corruption at decode (CUDA).** The
   `narrow → rope → cat` in the full-attn layers returns a tensor with
   head-interleaved strides (`[1,1,1,16]`) when seq == 1 — extent-1 dims
   make `.contiguous()` a no-op and candle's CUDA `cat` doesn't sanitize.
   CPU matmuls silently copy such views; CUDA either hard-errors
   ("matmul is only supported for contiguous tensors" — the CLI repro) or
   reads garbage into NaN logits (the app's sampler failure). Fix: force
   `.contiguous()` on both halves and the cat result.
2. **GDN k→v head grow scrambled (all devices).** The 4B is the first
   qwen35 with `n_v_heads (32) ≠ n_k_heads (16)`; the grow did
   `broadcast_as` through `(h, t, rep, d)`, interleaving the copy axis
   with TIME instead of heads. Output: plausible for ~8 tokens, then
   "The user is asking me to." loops forever. Found with the raw oracle
   (`The capital of France is` → ` the, B, B is the:`; 0.8B → ` Paris`).
3. **GGUF vs HF repeat conventions differ.** llama.cpp's converter
   permutes v/z channels for `ggml_repeat`'s TILED layout (`[k0..kN,
   k0..kN]`); HF safetensors pair via `repeat_interleave` BLOCKS
   (`[k0, k0, k1, k1]`). Equal head counts — every model until the 4B —
   make the two coincide. Fix: `GdnAttn.kv_repeat_tiled` set from the
   weight source (`ModelFile::Gguf` → tiled, safetensors → block), the
   grow branches on it (verified against `src/models/qwen35.cpp` +
   transformers Qwen3_5 source, and codified in the new
   `kv_head_grow_layouts` test).

**After the fix:** 4B Q4_K_S on CUDA → ` Paris, not any other city` +
coherent structured thinking at **23-25 tok/s** (CPU 5.8-6.6); 0.8B
unaffected. Also while here: st_shim aliases `model.language_model.*`
(official Qwen3.5-4B is a VL wrapper repo — its real sharded safetensors,
2 × BF16 shards + nested `text_config`, now load through the multi-region
reader: 427 tensors, A_log patch fires; full F32 run OOMs on 16 GB —
the known F16-vs-RAM trade, noted not fixed). Engine **88 tests ×4
trees**. New permanent section: **§6b Adapter Authoring Rules** — the
broadcast-view/stride checklist every future adapter gets checked against
(user-requested). Windows CUDA notes: build from a vcvars64 prompt, kill
running `ambercore.exe` before linking, check exe-vs-source timestamps
before trusting a repro.

---

## 2026-09-13 — the ~500-token NaN: ill-conditioned WY solve in the GDN prefill (Qwen3.5-4B, CUDA first)

**Symptom.** The Phase C encapsulated build, first real session: pulled
Qwen3.5-4B-Q4_K_M (into the exe-side `models/` folder — the new layout
itself worked), warm-up fine, VRAM fine (~4.2/6.7 GiB), first chat died
with `sample: A weight is negative, too large or not a valid number`.

**The diagnosis ladder (each step killed a hypothesis):**
1. Engine sources diffed IDENTICAL across all 4 trees — not stale-code.
2. CPU CLI on the user's exact GGUF: clean at every length → not the file,
   not the quant, not the adapter logic.
3. CUDA CLI: clean ≤ ~390 tokens, **deterministic NaN ≥ ~480**, hard OOM at
   ~2500 (separate, see below) → length-dependent, CUDA-surfaced.
4. Primitive suspects tested exact CUDA-vs-CPU (eye→cumsum masks, the
   (n_v, L, 1) batched cumsum, the nilpotent doubling at L ≤ 65) → NOT a
   candle stride/cumsum bug (cumsum in candle = matmul against triu ones —
   no dedicated CUDA kernel to be broken).
5. Layer instrumentation (env-gated `AC_DEBUG_NAN`): layer 24, a GDN layer,
   at prefill seq=480 — inputs finite, output NaN.
6. Chunk instrumentation: `v_new` hit inf at chunk@192 with the state
   running `0 → 8e4 → 4e11 → 2e18 → inf` **while every decay ≤ 1**. The
   amplification lives in the WY solve: `(I + M)⁻¹ = Σ(−M)^k` entries
   reach ~1e15 when a chunk's keys are strongly correlated, and `v_new =
   (I+M)⁻¹v_β − (I+M)⁻¹(k_β e^g)·S` then destroys f32 cancellation. The
   formulas were verified line-by-line against FLA's
   `naive_chunk_gated_delta_rule` — identical; this is numerical
   conditioning, not a transcription bug.

**Why nobody saw it before:** every prior validation (0.8B chat, 4B
short-prompt oracle, 4B " Paris" runs) prefilled < ~64 tokens. The
full-attention layers upstream make a *long* context's deep-layer GDN
inputs value-correlated in ways short prompts never produce; the chunked
solve is exact in real arithmetic but catastrophically ill-conditioned in
f32 on those. CUDA surfaced it before CPU only because cuBLAS rounding
tips the borderline chunks sooner.

**Fix (`model/qwen35.rs`, all 4 trees).** The chunk loop is factored into
`chunked_delta_rule` (now returning `(o, final_state)` — the model forward
calls it, deleting the duplicated inline copy) with a **conditioning
guard**: after the doubling, `max|(I+M)⁻¹| > 1e4` (or non-finite) → that
chunk is redone with `step_delta`, the token-recurrent form the decode
path already uses — mathematically identical, unconditionally stable (and
what llama.cpp uses exclusively for this architecture). Belt-and-braces:
even a passing solve whose `o`/`S` came out non-finite is redone
stepwise. Well-conditioned chunks keep the fast path; throughput is
unchanged (~24 tok/s decode, prefill unaffected in the runs above).

**§6b rule 8 (new):** chunked/WY forms of delta rules need a conditioning
guard in f32 — check the solve's magnitude before trusting it, fall back
to the stepwise form. Short-prompt oracles do not cover prefill numerics;
validation prompts must exceed a full chunk×several.

**Known limit (pre-existing, NOT this bug):** ~2.5k-token prefills OOM on
8 GiB — the full-attention layers materialize (h, L, L) scores+probs
(quadratic, no flash-attention path in candle). Long-context work needs
chunked/masked attention or an SDPA-style kernel; documented, not fixed.

Tests: +`ill_conditioned_chunks_stay_finite` (near-singular input stays
finite; forced fallback is bitwise the chunk-1 path via the injectable
`chunked_delta_rule_guarded`) — engine **91 ×2 trees** (dev + Encaps;
mirror + ALPHA synced verbatim). All failing prompts now return coherent
` Paris, and the capital of Germany is Berlin.` at 23-24 tok/s.
