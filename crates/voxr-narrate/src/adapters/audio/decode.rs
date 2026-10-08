//! Audio decode and resample, with voxr first and the ffmpeg fallback second.

use std::path::Path;

use super::ffmpeg;
use super::tools::Tools;
use crate::error::{Error, Result};

/// The rate a recognizer reads.
const HEARING_RATE: u32 = 16_000;

/// The rate a recording and a command's output are decoded at: the skill's
/// `samples()`.
const DECODE_RATE: u32 = 48_000;

/// `samples` at `rate` brought to 16 kHz as ffmpeg's `-ar 16000` brings
/// them, through [`voxr::resample_like_ffmpeg`]. A 16 kHz take comes back
/// unchanged.
pub fn to_16k(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    resample(samples, rate, HEARING_RATE)
}

/// The audio file at `path` as mono samples at 48 kHz: the skill's
/// `samples()`, which decodes through `ffmpeg -ac 1 -ar 48000`.
///
/// Any container voxr decodes is read: WAV, FLAC, MP3, AAC and ALAC in
/// MP4/M4A/MOV, and Opus and Vorbis in Ogg, WebM and MKV. The channels mix
/// through [`voxr::to_mono_like_ffmpeg`], then resample through
/// [`voxr::resample_like_ffmpeg`]. A file voxr does not decode goes
/// through that ffmpeg command, `$FFMPEG` or `ffmpeg` on `PATH`.
pub fn decode_48k(path: &Path) -> Result<Vec<f32>> {
    decode_at(path, DECODE_RATE, &Tools::from_env())
}

/// The audio file at `path` as mono samples at 16 kHz: the skill's
/// transcribe decode, `ffmpeg -ac 1 -ar 16000`, straight from the file's
/// own rate. A file voxr does not decode goes through that command.
pub fn decode_16k(path: &Path) -> Result<Vec<f32>> {
    decode_at(path, HEARING_RATE, &Tools::from_env())
}

/// The duration ffprobe reports for the audio file at `path`: its frames
/// at its own rate, rounded to whole microseconds. A file voxr does not
/// decode goes through the skill's ffprobe command, `$FFPROBE` or
/// `ffprobe` on `PATH`.
pub fn probe_duration(path: &Path) -> Result<f64> {
    probe_with(path, &Tools::from_env())
}

/// The file at `path` as mono samples at `rate`: voxr's decode, else
/// `tools`' ffmpeg when voxr refuses it as unsupported.
pub(super) fn decode_at(path: &Path, rate: u32, tools: &Tools) -> Result<Vec<f32>> {
    match decode_mono(path)? {
        Native::Mono(mono, from) => resample(&mono, from, rate),
        Native::Unsupported(why) => ffmpeg::decode(tools, path, rate, &why),
    }
}

/// [`probe_duration`] with `tools` as the fallback.
pub(super) fn probe_with(path: &Path, tools: &Tools) -> Result<f64> {
    match decode_mono(path)? {
        Native::Mono(mono, rate) => Ok(crate::wav::probed_duration(mono.len(), rate)),
        Native::Unsupported(why) => ffmpeg::duration(tools, path, &why),
    }
}

/// What voxr makes of a file.
pub(super) enum Native {
    /// Mono samples, at the file's own rate.
    Mono(Vec<f32>, u32),
    /// voxr's refusal of a well-formed file it does not decode.
    Unsupported(String),
}

/// The audio file at `path` decoded and mixed to mono as `ffmpeg -ac 1`
/// mixes it, at its own rate, or voxr's "unsupported" refusal. Every
/// other decode error is returned as it is.
pub(super) fn decode_mono(path: &Path) -> Result<Native> {
    let bytes = std::fs::read(path).map_err(|e| Error::File {
        reason: format!(
            "reading {}: {e}. Pass a readable audio file",
            path.display()
        ),
    })?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(voxr::extension_hint);
    let data = match voxr::decode_audio_like_ffmpeg(&bytes, hint) {
        Ok(data) => data,
        Err(voxr::Error::Unsupported { reason }) => return Ok(Native::Unsupported(reason)),
        Err(e) => {
            return Err(Error::DataError {
                reason: format!(
                    "decoding {}: {e}. Pass a WAV, FLAC, MP3, M4A, MP4, MOV, MKV, WebM, Ogg or Opus file",
                    path.display()
                ),
            });
        }
    };
    let mono =
        voxr::to_mono_like_ffmpeg(&data.samples, data.channels, data.layout).map_err(|e| {
            Error::DataError {
                reason: format!(
                    "mixing the {} channels of {} to mono: {e}",
                    data.channels,
                    path.display()
                ),
            }
        })?;
    Ok(Native::Mono(mono, data.sample_rate))
}

/// `samples` from `from` Hz to `to` Hz, as ffmpeg resamples.
fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    voxr::resample_like_ffmpeg(samples, from, to).map_err(|e| Error::DataError {
        reason: format!(
            "resampling {} samples from {from} Hz to {to} Hz: {e}. Pass audio at a common rate such as 16, 24, 44.1 or 48 kHz",
            samples.len()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::pcm16_wav;

    #[test]
    fn a_16k_take_is_unchanged_and_a_48k_one_a_third_as_long() -> Result<()> {
        let take: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        assert_eq!(to_16k(&take, 16_000)?, take);
        assert_eq!(to_16k(&take, 48_000)?.len(), 1600);
        Ok(())
    }

    #[test]
    fn a_wav_decodes_to_48k_mono() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let path = dir.path().join("take.wav");
        let take: Vec<f32> = (0..2400).map(|i| (i as f32 * 0.02).sin() * 0.25).collect();
        std::fs::write(&path, pcm16_wav(&take, 24_000)?).map_err(|e| Error::File {
            reason: e.to_string(),
        })?;
        assert_eq!(decode_48k(&path)?.len(), 4800);
        assert_eq!(decode_16k(&path)?.len(), 1600);
        assert_eq!(probe_duration(&path)?, 0.1);
        Ok(())
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let err = decode_48k(Path::new("/no/such/take.wav")).err();
        assert!(err.is_some_and(|e| e.to_string().contains("/no/such/take.wav")));
    }
}
