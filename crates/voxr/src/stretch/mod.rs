//! Tempo change without a pitch change, by waveform-similarity overlap-add.
//!
//! [`atempo`] and its siblings reproduce the output of ffmpeg's `atempo`
//! filter for tempos in `[MIN_TEMPO, MAX_TEMPO]`: same length, same samples.
//!
//! The method is WSOLA (Verhelst & Roelands, ICASSP 1993). The
//! implementation follows Pavel Koshevoy's MIT-licensed yae audio tempo
//! filter. Its copyright and permission notice are in `LICENSE-aeyae` beside
//! this module.

mod atempo;
mod correlate;
mod sample;
mod wsola;

pub use atempo::{MAX_TEMPO, MIN_TEMPO, atempo, atempo_interleaved, atempo_pcm16};
pub use sample::PcmSample;
