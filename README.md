# Phoenix Agent

> **ALPHA build notes (read first).** This is the reviewed, public source for the
> Phoenix Agent desktop app. Current state: v0.8.0 — a Tauri 2 desktop GUI with a
> dark glassmorphism UI, SQLCipher-encrypted memory + two-password/2FA model, a
> ReAct agent with an approval gate, a **visible reasoning/tool pipeline**
> (collapsible thinking block, expandable tool cards, sub-agent cards), and a
> Models panel with three interchangeable backends.
>
> **Model backends** (Models panel):
> - **AmberCore** (pure-Rust runner) — **embedded in-process**: the engine is
>   compiled into this binary (`ambercore/`), launches with the app, and the
>   last-used model is preloaded at startup. No separate server or binary. A
>   **remote** AmberCore server can be linked instead by entering its URL:port
>   (see [`AmberCore-Server`](../AmberCore-Server/README.md)).
> - **Ollama** (local).
> - **Provider API** — any OpenAI-compatible cloud endpoint.
>
> **GPU acceleration:** the default build embeds AmberCore's portable **CPU**
> backend. Rebuild with `--features ambercore-cuda` (NVIDIA + CUDA toolkit) or
> `--features ambercore-metal` (macOS Apple GPU, experimental) for hardware
> acceleration.
>
> **Build:** `cargo tauri dev` to run, `cargo tauri build` to bundle. Windows
> needs Strawberry Perl on `PATH` for the SQLCipher/OpenSSL compile
> (`set PATH=C:\Strawberry\perl\bin;%PATH%`). (The historical v0.1 notes below are
> retained for context; the TUI they mention was replaced by the Tauri GUI.)

---

A fully-local, encrypted, autonomous coding agent. Runs entirely on your
machine — your code, your conversations, and your memory never leave the box.
Your memory database is AES-256 encrypted at rest and unlocks only with your
passphrase.

```
   Phoenix Agent v0.1 — local · encrypted · autonomous
```

## Features (v0.1)

- **Full-screen TUI** — chat with the agent, watch it stream responses and run
  tools live, with a status bar and approval prompts.
- **Local models via Ollama** — talk to any Ollama model (default
  `qwen2.5-coder:7b`). The model layer is abstracted so other backends can be
  added later.
- **Encrypted memory (SQLCipher)** — every conversation turn, tool call, and
  tool result is persisted to a SQLCipher-encrypted SQLite database. Locked when
  Phoenix isn't running; unlocked only with your passphrase.
- **Autonomous coding tools** — the agent can read/write/edit files, search
  code (ripgrep or pure-Rust fallback), and run shell commands. Write actions
  require your approval before they execute.
- **ReAct reasoning loop** — the agent reasons, acts, observes, and iterates
  (capped) until the task is done.

## Quick start

### 1. Prerequisites (one-time)

Phoenix needs the Rust toolchain and MSVC C++ build tools (to compile the
bundled SQLCipher + OpenSSL from source). On Windows, install with
[winget](https://learn.microsoft.com/windows/package-manager/winget/):

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools   # then add the "Desktop development with C++" workload
winget install StrawberryPerl.StrawberryPerl            # for the bundled OpenSSL build
winget install NASM.NASM                                 # OpenSSL assembler
```

Open a **fresh** terminal after installing, then verify:
```powershell
cargo --version   # should print cargo 1.x
cl                # should find the MSVC compiler (run from a Developer prompt or after vcvars)
perl --version    # Strawberry Perl
nasm -v           # NASM
```

> **Build trouble?** If NASM causes issues, the vendored OpenSSL can be built
> without assembly by setting `OPENSSL_NO_ASM=1` in your environment before
> `cargo build`. The crypto will be slower but functionally fine for a local DB.

### 2. Install Ollama + a model

```powershell
# Ollama runs the models locally.
ollama serve             # start the server (leave it running)
ollama pull qwen2.5-coder:7b
```

### 3. Build Phoenix

```powershell
cd phoenix-agent
cargo build --release
# binary: target\release\phoenix.exe
```

### 4. Initialize

```powershell
phoenix init
```

This sets your encryption passphrase, creates the encrypted database, and
confirms your model is available.

### 5. Run

```powershell
phoenix            # launch the TUI in the current directory
phoenix doctor     # diagnose Ollama / DB / toolchain
```

## Usage

Inside the TUI:

| Key | Action |
|-----|--------|
| Type + `Enter` | Send a message |
| `Shift+Enter` | Newline (multiline input) |
| `Ctrl+C` | Clear input (or quit if empty) |
| `Ctrl+U` | Clear line |
| `PageUp` / `PageDown` | Scroll chat history |
| `y` / `n` | Approve / deny a pending tool call |

Slash commands:

| Command | Action |
|---------|--------|
| `/model <name>` | Switch model |
| `/new` | Start a fresh session |
| `/clear` | Clear the screen |
| `/note <text>` | Save a note |
| `/help` | Show commands |
| `/quit` | Exit |

## How it works

### Encryption

- On `init`, you set a passphrase. Phoenix generates a random 16-byte salt
  (`~/.phoenix/salt.bin`, not secret) and derives a 32-byte key with
  **Argon2id** (64 MiB · 3 passes · 4 lanes).
- The key is passed to **SQLCipher**, which performs transparent AES-256
  encryption over the entire database file.
- The derived key lives **only in process memory** for the session and is
  zeroized on drop. Nothing secret is written to disk.
- When Phoenix exits, the connection closes and the database file is opaque
  ciphertext. Without the passphrase, it cannot be read.

### Memory

Every turn is persisted:

- **sessions** — one per conversation, with project, model, timestamps.
- **messages** — every system/user/assistant/tool message, including tool-call
  requests and results, token counts, and the model that produced it.
- **notes** — pinned facts and manual notes.

Use `phoenix doctor` (with passphrase) to inspect session counts, or extend
the code to add search/recall features.

### Tool approval

Read-only tools (`read_file`, `list_dir`, `grep`) run automatically. Tools that
mutate state (`write_file`, `edit_file`, `run_command`) trigger an approval
prompt in the TUI — press `y` to allow or `n` to deny. The policy is
configurable in `config.toml`:

```toml
[approval_policy]  # one of: all, writes_only, never
# (default: writes_only)
```

## Configuration

`~/.phoenix/config.toml`:

```toml
model = "qwen2.5-coder:7b"
ollama_url = "http://localhost:11434"
db_filename = "memory.db"
approval_policy = "writes_only"
max_iterations = 25
context_window = 50
```

Override the data directory with `--data-dir` or the `PHOENIX_DATA_DIR`
environment variable.

## Project layout

```
phoenix-agent/
├── src/
│   ├── main.rs           CLI entry (init / doctor / chat)
│   ├── config.rs         paths + config.toml
│   ├── crypto.rs         passphrase → Argon2 → key
│   ├── error.rs          unified error types
│   ├── db/               SQLCipher connection + memory store
│   ├── model/            ModelProvider trait + Ollama adapter
│   ├── agent/            ReAct loop, prompt, tools (fs/search/shell)
│   └── tui/              ratatui full-screen UI
└── migrations/           SQL schema
```

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
