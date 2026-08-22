# Phoenix Agent

> **ALPHA — v0.8.0.** The reviewed, public source for the Phoenix Agent
> desktop app: fully local, encrypted at rest, autonomous. Models run through
> the **embedded AmberCore engine** (compiled in-process; CPU or NVIDIA CUDA),
> [Ollama](https://ollama.com), or any OpenAI-compatible cloud provider.

A fully-local, encrypted, autonomous coding & research agent. Everything runs
on your machine — your code, your conversations, and your memory never leave
the box. The memory database is AES-256 encrypted (SQLCipher) and unlocks only
with your launch password.

## What it does

- **Autonomous agent** — a ReAct reasoning loop with an approval gate: the
  agent reads/writes files, searches code, and runs shell commands; anything
  that mutates state asks first.
- **Visible reasoning** — the chat reads as a pipeline: collapsible *thinking*
  blocks (reasoning kept separate from answers, streaming with a flame glow),
  expandable tool cards (arguments · result · duration · inline Approve/Deny),
  nested sub-agent cards, and a live phase pill.
- **Encrypted memory** — every session and message lands in a
  SQLCipher-encrypted database. A two-password model gates it: the launch
  password you type at startup unwraps the database key; optional TOTP 2FA
  adds a recovery path.
- **Science workbench** — profiles bundle a model + working directory + the
  sidebar panels: **Skills** (markdown knowledge injected into the prompt),
  **Tools** (user scripts the agent can call like built-ins), **Context**
  (project ground-truth files), **Memory** (MCP server connections), and
  specialized sub-agents.
- **Hardware check-up** — Main Menu → **Telemetry** shows this machine's
  baseline at launch: CPU · cores · RAM · OS, the active compute backend, and
  the GPU + VRAM when a GPU backend is in use.

## Models

The **Models** panel offers three interchangeable backends — switch live, no
restart:

- **AmberCore** — the pure-Rust LLM runner (Candle), **embedded in-process**:
  the engine ships inside the app binary, launches with it, and preloads your
  last-used model so the first message answers fast. A remote AmberCore server
  can be linked instead by entering its `URL:port` (see
  [`AmberCore-Server`](../AmberCore-Server/README.md)).
- **Ollama** — local; auto-install and model pulls from the panel.
- **Provider API** — any OpenAI-compatible cloud endpoint. API keys are
  stored encrypted in the database and always blurred in the UI.

### Pulling models (AmberCore)

Paste a model URL — any Hugging Face `…/resolve/…/<model>.gguf` link (page
`/blob/` links and `?download=true` suffixes are handled) — and click
**Pull**. Phoenix downloads the GGUF **and fetches its tokenizer
automatically**: the same repo/revision first, then the base model repo, with
an optional Tokenizer URL field for other sources. The progress bar shows each
file as it downloads, and re-pulls skip files already on disk.

Pulls are validated before anything is registered:

- The GGUF's header is probed and **unsupported architectures are rejected
  immediately** with a plain-language error (supported today: `qwen2`,
  `qwen3`, `llama` — Qwen3.5's hybrid `qwen35` SSM architecture is not yet
  supported).
- A model **without a tokenizer never registers** — the pull fails listing
  the URLs it tried, and a retry (or a manual tokenizer URL) only downloads
  the missing file.

Models live in `<install folder>/models/` by default (configurable in the
panel) as per-model subfolders: `<model>/<model>.gguf` +
`<model>/<model>.tokenizer.json`.

### GPU acceleration

The embedded engine **prefers the GPU automatically** — CUDA when compiled
in, CPU otherwise; there is nothing to configure, and the active backend is
shown in Main Menu → Telemetry. The default build embeds the portable CPU
backend; rebuild with `--features ambercore-cuda` (NVIDIA; CUDA toolkit +
MSVC) or `--features ambercore-metal` (macOS Apple GPU, experimental) for
hardware acceleration. Ready-made Windows installers ship in both flavors —
see [`Installers`](../Installers/README.md).

## Build & run

### Prerequisites (one-time)

Phoenix needs the Rust toolchain, the Tauri CLI, and MSVC C++ build tools (to
compile the bundled SQLCipher + OpenSSL from source). On Windows, install with
[winget](https://learn.microsoft.com/windows/package-manager/winget/):

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools   # then add the "Desktop development with C++" workload
winget install StrawberryPerl.StrawberryPerl            # for the bundled OpenSSL build
winget install NASM.NASM                                 # OpenSSL assembler
cargo install tauri-cli --version "^2.0" --locked
```

> **Build trouble?** If NASM causes issues, the vendored OpenSSL can be built
> without assembly by setting `OPENSSL_NO_ASM=1`. Windows CUDA builds
> additionally need `set CL=/Zc:preprocessor /std:c++17` and
> `set CUDA_COMPUTE_CAP=<yy>` — see `ambercore/ACRoad.md` §8.

### Run

```powershell
cargo tauri dev
```

The first run shows a setup screen — choose a **launch password** (it derives
the keys that wrap the encrypted database). Then open **Models**, pull a
model, e.g.:

```
https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

click **Run**, and chat. A `phoenix doctor` CLI subcommand diagnoses the
toolchain and database if something misbehaves.

To bundle an installer: `cargo tauri build` (add `--features
ambercore-cuda` for the GPU build).

## Project layout

```
phoenix-agent/
├── src/
│   ├── main.rs            GUI entry (+ `doctor` subcommand)
│   ├── config.rs          paths + config.toml
│   ├── model/             ModelProvider trait + dispatch + Ollama / OpenAI /
│   │                      embedded-AmberCore adapters
│   ├── agent/             ReAct loop, prompt, skills, MCP client, tools
│   │                      (fs/search/shell + user scripts + sub-agents)
│   ├── backend/           process manager (Ollama serve)
│   ├── web/               Tauri layer: commands, events, state
│   ├── db/                SQLCipher connection + memory store + migrations
│   ├── crypto/            Argon2id KDF, key wrapping (AES-256-GCM), TOTP
│   └── health.rs          live backend/model health monitor
├── ambercore/             the embedded LLM engine (pure Rust, Candle)
├── frontend/              static HTML/CSS/JS UI (no build step)
├── migrations/            SQL schema (0001–0007)
└── installer.nsi          legacy NSIS script (Tauri's bundler drives builds)
```

Phoenix Agent is **fully portable**: everything the app creates — the
encrypted database, the wrapped-key bundle `keys.phx`, `config.toml`, logs,
and models — lives inside the **installation folder you chose in the
installer**. Nothing is scattered in your home directory. (Upgrading from a
pre-v0.8.3 install? The app copies your old `~/.phoenix` + `~/.ambercore`
data into the install folder automatically on first run — the originals are
left untouched.)

## Security notes (honest)

- Strong for a **local, single-user** tool: passphrase-gated DB, approval-gated
  writes, keys in process memory only.
- It is **not** a hardened multi-user or secrets-vault design. The passphrase
  protects the memory DB at rest; it does not encrypt the project files the
  agent reads/writes on your disk.
- Shell commands run with **your user privileges** in the working directory —
  that's why they're approval-gated. Review each command before approving.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
