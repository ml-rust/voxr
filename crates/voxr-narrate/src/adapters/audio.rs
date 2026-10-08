//! Audio as the skill's ffmpeg decode reads it: a take resampled to the
//! 16 kHz a recognizer reads, and a recording decoded to 48 kHz mono.

use std::path::Path;

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
/// Any container voxr decodes (WAV, FLAC, MP3, OGG/Vorbis) is read. The
/// channels are averaged, then resampled through
/// [`voxr::resample_like_ffmpeg`].
pub fn decode_48k(path: &Path) -> Result<Vec<f32>> {
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
    let data = voxr::decode_audio(&bytes, hint).map_err(|e| Error::DataError {
        reason: format!(
            "decoding {}: {e}. Pass a WAV, FLAC, MP3 or OGG/Vorbis file",
            path.display()
        ),
    })?;
    let mono = voxr::to_mono(&data.samples, data.channels).map_err(|e| Error::DataError {
        reason: format!(
            "mixing the {} channels of {} to mono: {e}. Pass a file with whole frames",
            data.channels,
            path.display()
        ),
    })?;
    resample(&mono, data.sample_rate, DECODE_RATE)
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
        Ok(())
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let err = decode_48k(Path::new("/no/such/take.wav")).err();
        assert!(err.is_some_and(|e| e.to_string().contains("/no/such/take.wav")));
    }
}
