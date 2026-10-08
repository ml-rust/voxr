//! Narration scripts: phrases and chunks a voice speaks, tags, subtitle and
//! timed script files, the written script, numbers read back from words and
//! digits, the check of a spoken take against its script, its glitches, and
//! the polish of its voice. Also a spoken take read into timed words, and
//! the take loop's cache keys, file names and take-record files, and its
//! pick of each part's best take through a speech engine and a judge, the
//! designed or prepared voice every take clones, a whole VoxCPM2 run from
//! script to voiceover, the per-phrase run of every other engine, a timed
//! script's lines placed at their times, the spoken track time-stretched to
//! a pace, and the skill's WAV files. Also, behind features, the real
//! engines behind the take loop's seams: Kokoro, VITS and VoxCPM2 as speech
//! engines, Whisper as the hearing, and the runs a caller hands loaded
//! models to.

mod error;
mod js;
mod js_math;
mod pattern;

#[cfg(any(
    feature = "whisper",
    feature = "kokoro",
    feature = "vits",
    feature = "voxcpm"
))]
pub mod adapters;
pub mod check;
pub mod glitch;
pub mod polish;
pub mod take;
pub mod text;
pub mod transcript;
pub mod wav;

#[cfg(any(
    feature = "whisper",
    feature = "kokoro",
    feature = "vits",
    feature = "voxcpm"
))]
pub use adapters::*;
pub use check::*;
pub use error::{Error, Result};
pub use glitch::*;
pub use polish::*;
pub use take::*;
pub use text::*;
pub use transcript::*;
pub use wav::{
    f32_wav, pcm16_round_trip, pcm16_wav, pcm16_wav_raw, probed_duration, read_wav, read_wav_file,
};
