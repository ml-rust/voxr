//! The real engines behind the take loop's seams: voxr's Kokoro, VITS and
//! VoxCPM2 models as [`SpeechEngine`](crate::SpeechEngine)s, Whisper as the
//! [`Hearing`](crate::Hearing), and [`NarrationRuns`], the
//! [`SpeakRuns`](crate::SpeakRuns) a caller hands loaded models to.
//!
//! Each engine sits behind the voxr-narrate feature of the same name. The
//! hearing and the runs need `whisper`. Nothing here resolves a model
//! directory: the caller loads each model and passes it in.

mod seed;

#[cfg(feature = "whisper")]
pub mod audio;
#[cfg(feature = "kokoro")]
pub mod kokoro;
#[cfg(feature = "whisper")]
pub mod runs;
#[cfg(feature = "vits")]
pub mod vits;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;
#[cfg(feature = "voxcpm")]
pub mod voxcpm_text;
#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisper")]
mod work_dir;

#[cfg(feature = "whisper")]
pub use audio::{decode_16k, decode_48k, probe_duration, to_16k};
#[cfg(feature = "kokoro")]
pub use kokoro::KokoroSpeech;
#[cfg(feature = "whisper")]
pub use runs::{NarrationRuns, NoEngine, Progress, Spawn};
#[cfg(feature = "vits")]
pub use vits::VitsSpeech;
#[cfg(feature = "voxcpm")]
pub use voxcpm::{VoxcpmSpeech, VoxcpmVoice};
#[cfg(feature = "voxcpm")]
pub use voxcpm_text::{VoxcpmText, voxcpm_text};
#[cfg(feature = "whisper")]
pub use whisper::{RecognizerSource, WhisperHearing};
