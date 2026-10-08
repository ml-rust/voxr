//! Audio as the skill's ffmpeg decode reads it: a take resampled to the
//! 16 kHz a recognizer reads, a recording decoded to 48 kHz or 16 kHz mono,
//! and an audio file's duration as ffprobe reports it.
//!
//! voxr decodes first. A file voxr refuses as unsupported, such as
//! multichannel or HE-AAC, goes through the skill's own ffmpeg command
//! instead. The skill needs ffmpeg for every run, so a file it reads is
//! never refused here. A corrupt or truncated file keeps voxr's error.

mod decode;
mod ffmpeg;
mod tools;

pub use decode::{decode_16k, decode_48k, probe_duration, to_16k};
