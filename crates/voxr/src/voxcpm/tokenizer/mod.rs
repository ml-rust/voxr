//! VoxCPM2's text path: [`load`] reads `tokenizer.json` into a
//! [`VoxCpm2Tokenizer`] giving the reference model's ids, and [`text`]
//! holds the whitespace collapse and the [`paren`] style prefix.

pub mod load;
pub mod text;

pub use load::{VoxCpm2Tokenizer, from_json, load_tokenizer, request_ids, tokenize};
pub use text::{is_blank, is_python_whitespace, normalize_whitespace, paren};
