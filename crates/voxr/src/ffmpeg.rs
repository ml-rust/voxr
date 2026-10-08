//! The ffmpeg programs voxr's ffmpeg-matching code is checked against:
//! `$FFMPEG` and `$FFPROBE`, else `ffmpeg` and `ffprobe` on `PATH`.
//!
//! voxr itself runs no process. Its tests compare against these programs,
//! and a caller that falls back to ffmpeg for a file voxr does not decode
//! runs the same ones.

use std::ffi::OsString;

/// `$name` when it is set and not empty, else `default`.
fn program(name: &str, default: &str) -> OsString {
    std::env::var_os(name)
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| OsString::from(default))
}

/// The ffmpeg program: `$FFMPEG`, else `ffmpeg` on `PATH`.
pub fn ffmpeg_program() -> OsString {
    program("FFMPEG", "ffmpeg")
}

/// The ffprobe program: `$FFPROBE`, else `ffprobe` on `PATH`.
pub fn ffprobe_program() -> OsString {
    program("FFPROBE", "ffprobe")
}
