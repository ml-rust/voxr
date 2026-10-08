//! The output of `librosa.resample` at its default quality, and so of the
//! resampling inside `librosa.load(path, sr=...)`.
//!
//! Provenance: an independent windowed-sinc design, checked against librosa
//! outputs. The method is a textbook polyphase Kaiser-windowed sinc
//! interpolator (J. F. Kaiser, "Nonrecursive digital filter design using
//! the I0-sinh window function", 1974; R. E. Crochiere and L. R. Rabiner,
//! "Multirate Digital Signal Processing", 1983). Its targets are the
//! documented traits of the soxr_hq quality librosa selects: linear phase,
//! a passband to 0.913 of the Nyquist limit, a stopband from 1.0, and 20-bit
//! precision. The cutoff, window width and β fit the impulse response of
//! librosa's outputs, measured as a black box. No resampler source was read
//! or translated.
//!
//! - [`resample_like_librosa`]: `librosa.resample(y, orig_sr=from, target_sr=to)`
//! - [`librosa_output_len`]: the length of its result

mod convert;
mod filter;

pub use convert::{librosa_output_len, resample_like_librosa};
