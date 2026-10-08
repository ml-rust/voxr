//! Mono WAV files as the skill writes and reads them: 16-bit samples behind
//! a 44-byte header, written with the skill's own rounding, and read back
//! as ffmpeg decodes them.
//!
//! voxr writes every file. Its `f32` PCM16 encoder rounds a half away from
//! zero and scales in `f32`, so its bytes differ from the skill's on some
//! samples ([`pcm16_like_js`]). A cache file shared with the skill, and a
//! prepared reference's content hash, need the skill's bytes exactly, so
//! this module rounds as the skill does and hands voxr the 16-bit values.
//!
//! The reader stays here: it decodes 16-bit samples as `v / 32768`, as
//! ffmpeg does, where voxr's decoder scales by `i16::MAX`. It also reads a
//! data chunk cut short up to the end of the bytes, as ffmpeg does.

use std::path::Path;

use voxr::wav::{FORMAT_EXTENSIBLE, FORMAT_IEEE_FLOAT, FORMAT_PCM};

use crate::error::{Error, Result, file_error};
use crate::polish::pcm16_like_js;
use crate::take::Take;

/// A 16-bit value `v` decodes as `v / 32768`, as ffmpeg's `s16` to `flt`.
const PCM16_READ: f32 = 1.0 / 32768.0;

/// The skill's `wav([x], rate).buf`: a mono 16-bit WAV of `x` at `rate` Hz.
///
/// Each sample is written as [`pcm16_like_js`] rounds it. More than 2^31
/// samples overflow the header's 32-bit sizes, and the skill's writer
/// throws on them too. A rate of 0 is refused.
pub fn pcm16_wav(x: &[f32], rate: u32) -> Result<Vec<u8>> {
    pcm16_wav_raw(&pcm16_like_js(x), rate)
}

/// A mono 16-bit WAV of the 16-bit values `x` at `rate` Hz, behind the
/// 44-byte header [`pcm16_wav`] writes. More than 2^31 samples overflow the
/// header's 32-bit sizes. A rate of 0 is refused.
pub fn pcm16_wav_raw(x: &[i16], rate: u32) -> Result<Vec<u8>> {
    voxr::wav::encode_wav_pcm16_values(x, rate).map_err(written)
}

/// The duration ffprobe reports for a WAV of `frames` frames at `rate` Hz
/// (`-show_entries format=duration`), read back as a number: the length
/// rounded to whole microseconds, a half rounded up. A rate of 0 gives 0.
pub fn probed_duration(frames: usize, rate: u32) -> f64 {
    if rate == 0 {
        return 0.0;
    }
    let rate = u128::from(rate);
    let micros = (frames as u128 * 1_000_000 + rate / 2) / rate;
    // ffprobe prints `%f`: the whole seconds and six decimals.
    let text = format!("{}.{:06}", micros / 1_000_000, micros % 1_000_000);
    text.parse().unwrap_or(f64::NAN)
}

/// A mono 32-bit float WAV of `x` at `rate` Hz, behind a 44-byte header.
/// It reads back ([`read_wav`]) as `x` exactly.
///
/// More than 2^30 samples overflow the header's 32-bit sizes. A rate of 0
/// is refused.
pub fn f32_wav(x: &[f32], rate: u32) -> Result<Vec<u8>> {
    voxr::wav::encode_wav_f32(x, rate).map_err(written)
}

/// A voxr encoder error as this crate's: an argument it refuses keeps its
/// name and reason.
fn written(e: voxr::Error) -> Error {
    match e {
        voxr::Error::InvalidArgument { arg, reason } => Error::InvalidArgument { arg, reason },
        other => Error::DataError {
            reason: other.to_string(),
        },
    }
}

/// `x` as [`pcm16_wav`] writes it and ffmpeg reads it back: each sample
/// rounded to 16 bits, then divided by 32768.
pub fn pcm16_round_trip(x: &[f32]) -> Vec<f32> {
    pcm16_like_js(x)
        .into_iter()
        .map(|v| f32::from(v) * PCM16_READ)
        .collect()
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// The format code, channels, rate and bits of a `fmt ` chunk body.
fn fmt_fields(body: &[u8]) -> Option<(u16, u16, u32, u16)> {
    let mut code = u16_at(body, 0)?;
    if code == FORMAT_EXTENSIBLE {
        code = u16_at(body, 24)?;
    }
    Some((code, u16_at(body, 2)?, u32_at(body, 4)?, u16_at(body, 14)?))
}

/// The mono take a WAV file's bytes hold: 16-bit samples as ffmpeg decodes
/// them (`v / 32768`), or 32-bit float samples as they are.
///
/// A data chunk that runs past the end of the bytes is read up to the end.
/// Other chunk kinds are skipped.
pub fn read_wav(bytes: &[u8]) -> Result<Take> {
    let bad = |why: &str| Error::DataError {
        reason: format!("{why}. Decode it to a mono 16-bit or 32-bit float WAV first"),
    };
    if !bytes.starts_with(b"RIFF") || bytes.get(8..12) != Some(&b"WAVE"[..]) {
        return Err(bad("the bytes are not a RIFF WAVE file"));
    }
    let mut at: usize = 12;
    let mut fmt = None;
    while let Some(head) = at.checked_add(8).and_then(|end| bytes.get(at..end)) {
        let Some(size) = u32_at(head, 4) else { break };
        let size = size as usize;
        let id = &head[..4];
        let start = at + 8;
        let end = start.saturating_add(size).min(bytes.len());
        let body = &bytes[start..end];
        if id == b"fmt " {
            fmt = Some(fmt_fields(body).ok_or_else(|| bad("the fmt chunk is cut short"))?);
        } else if id == b"data" {
            let (code, channels, rate, bits) =
                fmt.ok_or_else(|| bad("the data chunk comes before any fmt chunk"))?;
            if channels != 1 {
                return Err(bad(
                    format!("the file holds {channels} channels, not 1").as_str()
                ));
            }
            let samples = match (code, bits) {
                (FORMAT_PCM, 16) => body
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| f32::from(i16::from_le_bytes(b)) * PCM16_READ)
                    .collect(),
                (FORMAT_IEEE_FLOAT, 32) => body
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b))
                    .collect(),
                _ => {
                    let why = format!(
                        "the file holds format {code} at {bits} bits, not 16-bit PCM or 32-bit float"
                    );
                    return Err(bad(why.as_str()));
                }
            };
            return Ok(Take { samples, rate });
        }
        // A chunk of odd size carries one pad byte.
        at = start.saturating_add(size).saturating_add(size & 1);
    }
    Err(bad("the file holds no data chunk"))
}

/// [`read_wav`] over the file at `path`. An error names the file.
pub fn read_wav_file(path: &Path) -> Result<Take> {
    let bytes = std::fs::read(path)
        .map_err(|e| file_error("reading", path, &e, "Check the file exists and is readable"))?;
    read_wav(&bytes).map_err(|e| match e {
        Error::DataError { reason } => Error::DataError {
            reason: format!("{}: {reason}", path.display()),
        },
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxr::wav::HEADER_LEN as HEADER;

    #[test]
    fn the_header_matches_the_skill() -> Result<()> {
        let bytes = pcm16_wav(&[0.5, -0.5, 1.5], 48000)?;
        assert_eq!(bytes.len(), 50);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32_at(&bytes, 4), Some(42));
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(u32_at(&bytes, 16), Some(16));
        assert_eq!(u16_at(&bytes, 20), Some(1));
        assert_eq!(u16_at(&bytes, 22), Some(1));
        assert_eq!(u32_at(&bytes, 24), Some(48000));
        assert_eq!(u32_at(&bytes, 28), Some(96000));
        assert_eq!(u16_at(&bytes, 32), Some(2));
        assert_eq!(u16_at(&bytes, 34), Some(16));
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32_at(&bytes, 40), Some(6));
        // 0.5 rounds up to 16384, -0.5 to -16383, 1.5 clamps to 32767.
        assert_eq!(&bytes[44..], &[0x00, 0x40, 0x01, 0xC0, 0xFF, 0x7F]);
        Ok(())
    }

    #[test]
    fn raw_values_are_written_as_they_are() -> Result<()> {
        let bytes = pcm16_wav_raw(&[16384, -16384, i16::MIN], 24000)?;
        assert_eq!(bytes[..HEADER], pcm16_wav(&[0.0; 3], 24000)?[..HEADER]);
        assert_eq!(&bytes[HEADER..], &[0x00, 0x40, 0x00, 0xC0, 0x00, 0x80]);
        Ok(())
    }

    #[test]
    fn probed_durations_round_to_microseconds() {
        assert_eq!(probed_duration(74113, 48000), 1.544021);
        // 1.5345625 s: the half microsecond rounds up.
        assert_eq!(probed_duration(73659, 48000), 1.534563);
        assert_eq!(probed_duration(48000, 48000), 1.0);
        assert_eq!(probed_duration(0, 48000), 0.0);
        assert_eq!(probed_duration(5, 0), 0.0);
    }

    #[test]
    fn a_written_file_reads_back_as_ffmpeg_decodes_it() -> Result<()> {
        let x = [0.5f32, -0.5, 0.25, -1.0];
        let take = read_wav(&pcm16_wav(&x, 44100)?)?;
        assert_eq!(take.rate, 44100);
        assert_eq!(take.samples, pcm16_round_trip(&x));
        assert_eq!(
            take.samples,
            [
                16384.0 / 32768.0,
                -16383.0 / 32768.0,
                8192.0 / 32768.0,
                -32767.0 / 32768.0
            ]
        );
        Ok(())
    }

    #[test]
    fn a_float_file_reads_as_it_is() -> Result<()> {
        let mut bytes = pcm16_wav(&[], 48000)?;
        bytes[20..22].copy_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
        bytes[34..36].copy_from_slice(&32u16.to_le_bytes());
        bytes[40..44].copy_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&0.125f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(read_wav(&bytes)?.samples, [0.125, -2.0]);
        Ok(())
    }

    #[test]
    fn a_float_file_reads_back_exactly() -> Result<()> {
        let x = [0.1f32, -0.7, 1.5, f32::MIN_POSITIVE];
        let bytes = f32_wav(&x, 24000)?;
        assert_eq!(bytes.len(), HEADER + 16);
        assert_eq!(u16_at(&bytes, 20), Some(FORMAT_IEEE_FLOAT));
        assert_eq!(u32_at(&bytes, 28), Some(96000));
        let take = read_wav(&bytes)?;
        assert_eq!((take.samples.as_slice(), take.rate), (&x[..], 24000));
        Ok(())
    }

    #[test]
    fn stereo_and_garbage_are_refused() -> Result<()> {
        let mut stereo = pcm16_wav(&[0.0, 0.0], 48000)?;
        stereo[22..24].copy_from_slice(&2u16.to_le_bytes());
        assert!(matches!(read_wav(&stereo), Err(Error::DataError { .. })));
        assert!(matches!(
            read_wav(b"not a wav"),
            Err(Error::DataError { .. })
        ));
        Ok(())
    }
}
