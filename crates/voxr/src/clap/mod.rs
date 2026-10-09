//! CLAP scoring of audio against text: a checkpoint directory loaded into
//! boostr's `ClapModel`, its tokenizer, and the cosine scores between
//! clips and texts.

pub mod bundle;
pub mod weights;

pub use bundle::ClapBundle;
pub use weights::{PYTORCH_BIN, SAFETENSORS, load_weights};
