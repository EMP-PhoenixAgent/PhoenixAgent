//! Llama architecture.
//!
//! Wraps [`candle_transformers::models::quantized_llama`]. Stub for now — qwen2
//! is M0's focus (Phoenix's default). Llama lands in M1+ once the Qwen2 path
//! proves the pipeline end-to-end.
//!
//! [`candle_transformers::models::quantized_llama`]: https://docs.rs/candle-transformers

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};

/// A constructed Llama model. Stub.
pub struct LlamaModel {
    arch: String,
}

impl DynModel for LlamaModel {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, _input: &Tensor, _index_pos: usize) -> Result<Tensor> {
        Err(Error::Model("llama forward: not implemented (M1+)".into()))
    }
}

/// Registry entry point. Not implemented (M1+).
pub fn build(loaded: &mut LoadedModel, _device: &Device) -> Result<Box<dyn DynModel>> {
    let _ = loaded.take_content(); // consume so the file handle is free
    Err(Error::Model("llama build: not implemented yet (M1+)".into()))
}
