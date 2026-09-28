//! The take loop's files, seeds and cache keys: the seed a part's own text
//! draws, the VoxCPM2 cache key builders and the file names they carry,
//! and the `.takes.json`/`.phrases.json`/script text files a take reads
//! and writes. Also a take's speaking pace: syllables per second and the
//! verdict against a target. Also the seams a take is spoken, stored and
//! judged through, the pick of each part's best take, and the split of a
//! part that fails whole.

pub mod cache;
pub mod engine;
pub mod files;
pub mod hear;
pub mod judge;
pub mod pace;
pub mod pick;
pub mod seed;
pub mod split;
pub mod step;

pub use cache::{
    KeyPart, LEGACY_CFG, LEGACY_STEPS, TakeKeyInput, VOXCPM, anchor_key, cache_key, content_hash,
    mtime_millis, prepared_marker, prepared_reference_key, ref_id, reference_file, take_file,
    take_key, voice_file,
};
pub use engine::{
    CloneReference, Graded, SpeakRequest, SpeechEngine, Take, TakeJudge, TakeSpec, TakeStore,
    VoiceSource,
};
pub use files::{
    DEFAULT_SEED, PhrasesFile, ReadTakes, ResolvedSeed, TakesFile, chosen_takes, output_text_file,
    phrases_file, read_takes_file, resolve_seed, script_text_file, takes_file,
};
pub use hear::{HeardCheck, Hearing, Recognizer, TranscriberHearing, asr_model, heard_check};
pub use judge::{HeardJudge, JudgedTake};
pub use pace::{
    DEFAULT_TARGET_PACE, PaceComparison, PaceSpan, PaceVerdict, REF_PACE, pace, pace_verdict,
    sounding, syllables,
};
pub use pick::{ATTEMPTS, PickSetup, RETRY_STEP, TakeLoop};
pub use seed::text_seed;
pub use split::{BestTakes, SPLIT_ROUNDS, break_up};
pub use step::{PartTake, TakeStep};
