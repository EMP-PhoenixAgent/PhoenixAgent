# AmberCore — Development Roadmap & Log

> **This file is AmberCore's persistent dev log.** Read at the start of any AmberCore
> session to restore full context. Update it whenever a milestone lands or a decision
> changes. Mirrors the role of `PROJECT_CONTEXT.md` for Phoenix Agent itself.

**Last updated:** 2026-08-22
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

## 7. Development Log

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
