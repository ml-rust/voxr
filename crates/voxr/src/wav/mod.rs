//! RIFF/WAVE: the one container this crate reads and writes without the
//! `decode` feature. [`decode`] parses PCM and IEEE-float files through hound,
//! [`decode_f64`] reads them to `f64` as libsndfile does,
//! [`encode`] writes them, [`stream`] writes the unknown-length header a
//! streamed response starts with, [`format`] holds the format tags and
//! canonical header layout all sides agree on, and `layout` parses the `fmt `
//! chunk and the channel layout it names.

pub mod decode;
pub mod decode_f64;
pub mod encode;
pub mod format;
pub(crate) mod layout;
pub mod stream;

pub use decode::{WavData, decode_wav, to_mono};
pub use decode_f64::{WavF64, decode_wav_f64};
pub use encode::{
    encode_f32_raw, encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16,
    encode_wav_pcm16_multichannel, encode_wav_pcm16_values,
};
pub use format::{FORMAT_EXTENSIBLE, FORMAT_IEEE_FLOAT, FORMAT_PCM, HEADER_LEN};
pub use stream::{STREAM_SIZE_UNKNOWN, wav_stream_header_f32, wav_stream_header_pcm16};
