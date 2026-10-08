//! The take loop's files, seeds and cache keys: the seed a part's own text
//! draws, the VoxCPM2 cache key builders and the file names they carry,
//! and the `.takes.json`/`.phrases.json`/script text files a take reads
//! and writes. Also a take's speaking pace: syllables per second and the
//! verdict against a target. Also the seams a take is spoken, stored and
//! judged through, the pick of each part's best take, and the split of a
//! part that fails whole. Also the voice every take clones: designed from
//! a description or prepared from a user's recording, and kept in the cache.
//! Also the kept takes shaped and joined into the voiceover, with each
//! phrase's span and each part's check, heat and marks. Also the take
//! cache on disk, and a whole VoxCPM2 run from script to voiceover. Also
//! which engine, model, voice and speed a run resolves to speak with, and
//! the per-phrase run of every other engine, the user's TTS command among
//! them. Also a timed script's lines placed at their times, and, with the
//! `stretch` feature, the spoken track time-stretched to a pace. Also a
//! whole speak run from script to voiceover, which picks the branch: one
//! engine run, a time-stretch, or a timed placement.

pub mod anchor;
pub mod anchor_files;
mod anchor_step;
pub mod cache;
pub mod command;
pub mod dispatch;
mod dispatch_run;
pub mod dispatch_setup;
pub mod engine;
pub mod engine_choice;
pub mod files;
pub mod hear;
pub mod join;
pub mod join_checks;
pub mod join_heat;
pub mod judge;
pub mod pace;
pub mod pick;
pub mod place;
pub mod seed;
pub mod simple;
pub mod simple_setup;
pub mod speak;
pub mod speak_opts;
pub mod speak_resolve;
pub mod speak_setup;
pub mod speak_step;
pub mod split;
pub mod step;
pub mod store;
#[cfg(feature = "stretch")]
pub mod stretch;

pub use anchor::{
    ANCHOR_RATE, AnchorStep, DesignSetup, DesignedVoice, OPENING_WORDS, SLOWER, design_voice,
    opening,
};
pub use anchor_files::{
    Anchor, AnchorSetup, CloneFile, ReferencePrep, clone_file, copy_voice_beside, designed_anchor,
};
pub use cache::{
    KeyPart, LEGACY_CFG, LEGACY_STEPS, TakeKeyInput, VOXCPM, anchor_key, cache_key, content_hash,
    mtime_millis, prepared_marker, prepared_reference_key, ref_id, reference_file, take_file,
    take_key, voice_design_file, voice_file, voice_transcript_file,
};
pub use command::{CommandEngine, OUT_FILE, TEXT_FILE, command_line, shell_quote};
pub use dispatch::speak_script;
pub use dispatch_setup::{ScriptChecks, ScriptStep, SpeakRuns, SpeakTarget, SpokenScript};
pub use engine::{
    CloneReference, Graded, SpeakRequest, SpeechEngine, Take, TakeJudge, TakeSpec, TakeStore,
    VoiceSource,
};
pub use engine_choice::{DEFAULT_VOICE, DefaultEngineOpts, Engine, KOKORO_MODEL, default_engine};
pub use files::{
    DEFAULT_SEED, PhrasesFile, ReadTakes, ResolvedSeed, TakesFile, chosen_takes, output_text_file,
    phrases_file, read_takes_file, resolve_seed, script_text_file, takes_file, voice_output_file,
};
pub use hear::{
    DEFAULT_ASR_MODEL, HeardCheck, Hearing, Recognizer, SMALL_ASR_MODEL, TranscriberHearing,
    asr_model, hear_16k, heard_check,
};
pub use join::{JOIN_RATE, JoinPart, JoinedTrack, ShapedTake, join_takes, shape_take};
pub use join_checks::PartCheck;
pub use join_heat::{PartHeat, part_heat};
pub use judge::{HeardJudge, JudgedTake, ReferenceHearing};
pub use pace::{
    DEFAULT_TARGET_PACE, PaceComparison, PaceSpan, PaceVerdict, REF_PACE, pace, pace_verdict,
    sounding, syllables,
};
pub use pick::{ATTEMPTS, PickSetup, RETRY_STEP, TakeLoop};
pub use place::{
    CueUnits, LINE_GAP, PLACE_RATE, PlacedTrack, TIMING_SLACK, UNPLACED_PROBLEM, cue_script,
    place_on_times,
};
pub use seed::text_seed;
pub use simple::speak_phrases;
pub use simple_setup::{SimpleRun, SimpleSetup, SimpleStep, WholeCheck, check_kokoro_voice};
pub use speak::{WHOLE_MAX, speak_voxcpm};
pub use speak_opts::{SpeakOpts, TtsConfig};
pub use speak_resolve::{EngineChoice, SpeakResolution, resolve_speak};
pub use speak_setup::{VoiceReference, VoxcpmRun, VoxcpmSetup};
pub use speak_step::SpeakStep;
pub use split::{BestTakes, SPLIT_ROUNDS, break_up};
pub use step::{PartTake, TakeStep};
pub use store::{FileStore, TakeKeys};
#[cfg(feature = "stretch")]
pub use stretch::{
    STRETCH_RATE, StretchedTrack, resample_48k, stretch_checks, stretch_phrases, stretch_track,
};
