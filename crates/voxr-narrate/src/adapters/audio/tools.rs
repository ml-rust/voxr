//! The ffmpeg and ffprobe programs the fallback for a file voxr does not
//! decode runs, and running one of them.

use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::{Error, Result};

/// The programs the fallback runs.
#[derive(Debug, Clone)]
pub(super) struct Tools {
    /// The ffmpeg program.
    pub(super) ffmpeg: OsString,
    /// The ffprobe program.
    pub(super) ffprobe: OsString,
}

impl Tools {
    /// `$FFMPEG` and `$FFPROBE`, else `ffmpeg` and `ffprobe` on `PATH`.
    pub(super) fn from_env() -> Self {
        Self {
            ffmpeg: voxr::ffmpeg_program(),
            ffprobe: voxr::ffprobe_program(),
        }
    }
}

/// Runs `program` with `args`, its standard output on success. The error
/// names `path`, voxr's `refusal` and what went wrong, with the next
/// action.
pub(super) fn run(program: &OsStr, args: &[&OsStr], path: &Path, refusal: &str) -> Result<Vec<u8>> {
    let name = program.to_string_lossy();
    let failed = |what: String| Error::DataError {
        reason: format!(
            "decoding {}: voxr refuses it ({refusal}), and {what}",
            path.display()
        ),
    };
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            failed(format!(
                "{name}, which reads what voxr does not, did not run: {e}. Install ffmpeg, or set FFMPEG and FFPROBE to its programs"
            ))
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(failed(format!(
            "{name} cannot read it either: {}. Pass a WAV, FLAC, MP3, M4A, MP4, MOV, MKV, WebM, Ogg or Opus file",
            stderr.trim()
        )));
    }
    Ok(out.stdout)
}
