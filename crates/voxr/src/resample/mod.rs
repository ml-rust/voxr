//! Sample-rate conversion for `f32` audio.
//!
//! - [`resample`] and its siblings: a general windowed-sinc converter run by
//!   rubato.
//! - [`resample_like_ffmpeg`]: the output of ffmpeg's default audio resampler
//!   (`-ar N`), sample for sample.
//! - [`to_mono_like_ffmpeg`]: the mono mix of ffmpeg's `-ac 1`.

mod ffmpeg_compat;
mod sinc;

pub use ffmpeg_compat::{
    ffmpeg_default_layout, ffmpeg_resampled_len, resample_interleaved_like_ffmpeg,
    resample_like_ffmpeg, to_mono_like_ffmpeg,
};
pub use sinc::{
    DEFAULT_TAPS_PER_PHASE, MAX_FILTER_TAPS, resample, resample_with_taps, to_mono_at_rate,
};
