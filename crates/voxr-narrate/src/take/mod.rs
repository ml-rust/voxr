//! The take loop's files, seeds and cache keys: the seed a part's own text
//! draws, the VoxCPM2 cache key builders and the file names they carry,
//! and the `.takes.json`/`.phrases.json`/script text files a take reads
//! and writes.

pub mod cache;
pub mod files;
pub mod seed;

pub use cache::{
    KeyPart, LEGACY_CFG, LEGACY_STEPS, TakeKeyInput, VOXCPM, anchor_key, cache_key, content_hash,
    mtime_millis, prepared_marker, prepared_reference_key, ref_id, reference_file, take_file,
    take_key, voice_file,
};
pub use files::{
    DEFAULT_SEED, PhrasesFile, ReadTakes, ResolvedSeed, TakesFile, chosen_takes, output_text_file,
    phrases_file, read_takes_file, resolve_seed, script_text_file, takes_file,
};
pub use seed::text_seed;
