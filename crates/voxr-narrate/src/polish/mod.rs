//! The voice layer's polish of a take: sentence levels evened, a gentle
//! compressor, a de-esser and peak normalizing, and the 16-bit samples a
//! polished take is written as. Also a reference voice's preparation for
//! cloning: breaths turned down and a thin voice's bass lifted.

mod fft;
mod filter;

pub mod bass;
pub mod breath;
pub mod chain;
pub mod compressor;
pub mod de_ess;
pub mod level;
pub mod normalize;
pub mod reference;

pub use bass::{BassReport, bass_balance, lift_bass};
pub use breath::{BreathReport, breath_runs, quiet_breaths};
pub use chain::{POLISH_PEAK_DB, PolishReport, pcm16_like_js, polish};
pub use compressor::{CompressorOptions, compress_voice};
pub use de_ess::{DeEssStats, de_ess};
pub use level::{LevelReport, Span, even_levels, speech_level};
pub use normalize::normalize_peak;
pub use reference::{REFERENCE_PEAK_DB, prepare_reference};
