//! Backend process lifecycle management.
//!
//! Manages long-lived server processes (AmberCore's `serve`, Ollama's `serve`)
//! spawned by Phoenix so the Models panel can start a backend, switch to it, and
//! stop the previously-active one. Before this, Phoenix assumed those servers
//! were already running externally.

pub mod process;
