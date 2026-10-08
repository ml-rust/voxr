//! Compressed-audio decoding via symphonia: WAV, FLAC, MP3, OGG/Vorbis,
//! AAC and ALAC in MP4/M4A/MOV, and Opus or Vorbis in WebM/MKV/OGG, video
//! files' audio tracks included. Each file decodes as ffmpeg decodes it:
//! encoder priming and padding trimmed.

mod api;
mod frames;
mod layout;
mod mkv_discard;
mod mp4_edit;
mod opus;
mod stream;
mod trims;

pub use api::{
    decode_audio, decode_audio_file_mono_at, decode_audio_like_ffmpeg, decode_audio_mono_at,
    extension_hint,
};
