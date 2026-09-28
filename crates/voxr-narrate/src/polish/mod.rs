//! The voice layer's polish of a take: sentence levels evened, a gentle
//! compressor, a de-esser and peak normalizing, and the 16-bit samples a
//! polished take is written as.

mod fft;

pub mod chain;
pub mod compressor;
pub mod de_ess;
pub mod level;
pub mod normalize;

pub use chain::{POLISH_PEAK_DB, PolishReport, pcm16_like_js, polish};
pub use compressor::{CompressorOptions, compress_voice};
pub use de_ess::{DeEssStats, de_ess};
pub use level::{LevelReport, Span, even_levels, speech_level};
pub use normalize::normalize_peak;
