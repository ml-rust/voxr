//! The motion-video skill's developer sound review (`sound-review.py`):
//! numeric limits on every rendered sound, and recognizability of the base
//! variants against their `says` texts by an audio-text model.

pub mod index;
pub mod measure;
pub mod pyfmt;
pub mod rank;
pub mod report;
pub mod run;
pub mod usage;

pub use index::{SoundEntry, SoundIndex};
pub use measure::RawMeasures;
pub use rank::{AudioTextScorer, MODEL_RATE, Rank, model_clip, rank};
pub use report::{HISS_FAIL, RANK_WARN, Row, SHARP_FAIL, TAIL_FAIL, TP_FAIL, header, summary};
pub use run::{ReviewOutcome, run_review};
pub use usage::usage;
