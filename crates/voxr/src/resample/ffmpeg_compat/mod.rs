//! The output of ffmpeg's default audio resampler: `-ar N`, or `aresample`
//! with every resampler option at its documented default, and the mono mix
//! of `-ac 1`.
//!
//! The method is a textbook polyphase Kaiser-windowed sinc interpolator
//! (R. E. Crochiere and L. R. Rabiner, "Multirate Digital Signal
//! Processing", 1983). The parameters are the documented defaults of the
//! ffmpeg resampler options: `filter_size` 32, `phase_shift` 10, `cutoff`
//! 0.97, `filter_type` kaiser, `kaiser_beta` 9, `exact_rational` and
//! `linear_interp` on. The bank layout, edge handling, output length and
//! arithmetic order come from black-box measurement of the ffmpeg binary:
//! impulses, ramps, noise, and length sweeps. Each file documents its part.

mod bank;
mod convert;
mod downmix;
mod kernel;
mod length;

pub use convert::{resample_interleaved_like_ffmpeg, resample_like_ffmpeg};
pub use downmix::{ffmpeg_default_layout, to_mono_like_ffmpeg};
pub use length::ffmpeg_resampled_len;
