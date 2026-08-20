//! Database layer: encrypted SQLCipher connection management and the memory store.

pub mod connection;
pub mod store;

pub use connection::open_encrypted;
pub use store::{
    ContextFile, MemorySource, MemoryStore, Profile, ProfileContext, ProfileMemory, Provider,
    ProfileSkill, ProfileTool, SessionSummary, Skill, ToolRow,
};
