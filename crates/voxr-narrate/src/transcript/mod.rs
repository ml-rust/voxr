//! A narration take read into timed words: the take's loudness, the pieces
//! a recognizer reads, recognizer loops collapsed, word times settled on
//! the sound, script words placed phrase by phrase, and the transcript
//! files. With the `whisper` feature, Whisper as the recognizer.

pub mod envelope;
pub mod even;
pub mod fit;
pub mod loops;
pub mod onsets;
pub mod output;
pub mod phrase_align;
pub mod pieces;
pub mod transcriber;
#[cfg(feature = "whisper")]
pub mod whisper_transcriber;

pub use envelope::{Envelope, HOP, TRANSCRIPT_RATE, envelope};
pub use even::even_out;
pub use fit::{PhraseGroups, fit_file, fit_to_sound, last_run};
pub use loops::{Collapsed, collapse_loops};
pub use onsets::{hand_back_tails, snap_ends, snap_onsets};
pub use output::{
    AUDIO_LEAD, AUDIO_TAIL, Narration, NarrationManifest, SceneNarration, Spoken, TimelineWords,
    Voiceover, WordsFile, js_json, narrations, spoken_words, words_file, write_timeline,
    write_words_file,
};
pub use phrase_align::{PHRASE_MATCH, PhraseHit, PhraseWords, align_phrases, find_phrase};
pub use pieces::pieces;
pub use transcriber::{
    DEFAULT_LANGUAGE, TranscribeOptions, Transcribed, Transcriber, transcribe_narration,
};
#[cfg(feature = "whisper")]
pub use whisper_transcriber::{WhisperTranscriber, WhisperWords, read_take_16k};
