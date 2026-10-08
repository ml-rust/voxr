//! RIFF/WAVE decoder for PCM and IEEE-float audio, on hound.
//!
//! The counterpart of [`super::encode`]: it turns a WAV byte buffer back into
//! interleaved `f32` samples in `[-1, 1]` plus the format needed to interpret them.
//! Every WAV this crate writes round-trips through [`decode_wav`], and the integer
//! widths real recorders produce (24-bit, 32-bit) decode as well.
//!
//! hound validates the `fmt ` fields and decodes the samples. The chunk walk
//! stays here, for what hound 3.5 does not do: skip the RIFF pad byte after an
//! odd-sized chunk, accept `data` before `fmt `, clamp a `data` chunk left at
//! [`STREAM_SIZE_UNKNOWN`], and resolve a `WAVE_FORMAT_EXTENSIBLE` SubFormat by
//! its leading format tag rather than by the whole GUID. The walk locates both
//! chunks, and hound reads a canonical 44-byte header rebuilt from them,
//! followed by the borrowed `data` bytes.
//!
//! The parser reads untrusted file bytes, so it never panics: every read is bounds
//! checked and every malformed input returns a descriptive [`Error::InvalidArgument`].

use std::io::{Cursor, Read};

use crate::error::{Error, Result};

use super::format::{FmtFields, HEADER_LEN, canonical_header};
use super::layout::{mask_layout, parse_fmt};
use super::stream::STREAM_SIZE_UNKNOWN;

/// A decoded WAV file: interleaved samples plus the format needed to interpret them.
#[derive(Debug, Clone, PartialEq)]
pub struct WavData {
    /// Interleaved samples normalised to `[-1.0, 1.0]`, frame-major.
    pub samples: Vec<f32>,
    /// Sample rate in Hz, as declared by the `fmt ` chunk.
    pub sample_rate: u32,
    /// Channel count, at least 1.
    pub channels: u16,
    /// The channel positions as an ffmpeg channel mask (`AV_CH_*` bits,
    /// equal to `WAVE_FORMAT_EXTENSIBLE`'s for the first 18), the channels
    /// interleaved in ascending bit order. `None` when the file names no
    /// positions: ffmpeg then takes its default layout for the count.
    pub layout: Option<u64>,
}

impl WavData {
    /// Number of frames, i.e. samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }
}

/// What hound supports and this decoder returns, for error messages.
const SUPPORTED: &str = "16/24/32-bit integer PCM or 32-bit IEEE float; re-encode the file \
                         to one of those, or decode it with `decode_audio` (feature `decode`)";

pub(super) fn bad(reason: String) -> Error {
    Error::InvalidArgument {
        arg: "wav bytes",
        reason,
    }
}

pub(super) fn read_u16(bytes: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    let raw: [u8; 2] = bytes.get(off..end)?.try_into().ok()?;
    Some(u16::from_le_bytes(raw))
}

pub(super) fn read_u32(bytes: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let raw: [u8; 4] = bytes.get(off..end)?.try_into().ok()?;
    Some(u32::from_le_bytes(raw))
}

/// Decode a WAV byte buffer.
///
/// Supported formats: 16/24/32-bit signed integer PCM (tag 1), 32-bit IEEE float
/// (tag 3), and `WAVE_FORMAT_EXTENSIBLE` (tag 0xFFFE) wrapping either of those.
/// Integer samples are scaled by the positive full-scale value and clamped to
/// `[-1, 1]`; float samples are returned bit-exact.
///
/// Returns [`Error::InvalidArgument`] for a buffer that is truncated, carries bad
/// RIFF/WAVE magic, declares a chunk size that overruns the buffer, has no `fmt `
/// or `data` chunk, declares zero channels, uses an unsupported format tag or
/// bit width, declares a block alignment too small for its bit width, or holds a
/// `data` chunk that is not a whole number of frames.
pub fn decode_wav(bytes: &[u8]) -> Result<WavData> {
    if bytes.len() < 12 {
        return Err(bad(format!(
            "buffer is {} bytes, too short for the 12-byte RIFF/WAVE header",
            bytes.len()
        )));
    }
    if bytes.get(0..4) != Some(b"RIFF") {
        return Err(bad(
            "missing 'RIFF' magic at offset 0: not a RIFF file".to_string()
        ));
    }
    if bytes.get(8..12) != Some(b"WAVE") {
        return Err(bad(
            "missing 'WAVE' form type at offset 8: RIFF file is not a WAVE".to_string(),
        ));
    }

    let end = bytes.len();
    let mut format: Option<FmtFields> = None;
    let mut data_range: Option<(usize, usize)> = None;
    // Whether the `data` size came from the header (exact) or from the buffer
    // length (streamed, clamped). Only the latter tolerates a partial frame.
    let mut declared_data = false;
    let mut pos = 12usize;

    // Walk the chunk list: every chunk is a 4-byte id, a little-endian u32 size,
    // then that many bytes of body, followed by one pad byte when the size is odd.
    while pos + 8 <= end {
        let id: [u8; 4] = match bytes.get(pos..pos + 4).and_then(|s| s.try_into().ok()) {
            Some(id) => id,
            None => break,
        };
        let size = read_u32(bytes, pos + 4)
            .ok_or_else(|| bad(format!("chunk header at offset {pos} is truncated")))?
            as usize;
        let body = pos + 8;
        let body_end = body.checked_add(size).ok_or_else(|| {
            bad(format!(
                "chunk '{}' at offset {pos} declares size {size}, which overflows",
                tag_name(&id)
            ))
        })?;
        // A writer streaming to a pipe cannot seek back to patch the size, so
        // it leaves `data` at [`STREAM_SIZE_UNKNOWN`] and the chunk runs to
        // end of file. `data` is the last chunk of such a file, so that size
        // is clamped to the bytes present and a trailing partial frame is
        // dropped.
        //
        // The clamp is bound to the sentinel and nothing else. Any OTHER
        // overrunning size is a truncated or corrupt file, and clamping it
        // decodes a short buffer as whole with no error.
        if body_end > end && &id == b"data" && size == STREAM_SIZE_UNKNOWN as usize {
            if data_range.is_none() {
                data_range = Some((body, end));
            }
            break;
        }
        if body_end > end {
            return Err(bad(format!(
                "chunk '{}' at offset {pos} declares size {size}, which overruns the {end}-byte buffer",
                tag_name(&id)
            )));
        }

        match &id {
            b"fmt " => format = Some(parse_fmt(bytes, body, size)?),
            b"data" if data_range.is_none() => {
                data_range = Some((body, body_end));
                declared_data = true;
            }
            _ => {}
        }

        // RIFF pads odd-sized chunk bodies to an even boundary.
        let padded = size + (size & 1);
        pos = match body.checked_add(padded) {
            Some(next) => next,
            None => break,
        };
    }

    let format = format.ok_or_else(|| bad("no 'fmt ' chunk found in file".to_string()))?;
    let (data_start, data_end) =
        data_range.ok_or_else(|| bad("no 'data' chunk found in file".to_string()))?;
    let data = bytes.get(data_start..data_end).ok_or_else(|| {
        bad(format!(
            "'data' chunk range {data_start}..{data_end} is outside the buffer"
        ))
    })?;

    // A zero block alignment is left for hound to reject with the rest of the
    // `fmt ` fields.
    let frame_size = format.block_align.max(1) as usize;
    let whole_frames = data.len() - data.len() % frame_size;
    if whole_frames != data.len() && declared_data {
        return Err(bad(format!(
            "'data' chunk is {} bytes, not a whole number of {frame_size}-byte frames \
             ({} channels x {} bits)",
            data.len(),
            format.channels,
            format.bits_per_sample
        )));
    }
    let data = &data[..whole_frames];
    decode_samples(&format, data)
}

/// Render a chunk id for error messages, replacing non-printable bytes.
fn tag_name(id: &[u8; 4]) -> String {
    id.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '?'
            }
        })
        .collect()
}

/// Hand hound the canonical header for `format` followed by `data`, and
/// convert what it reads to `f32`.
fn decode_samples(f: &FmtFields, data: &[u8]) -> Result<WavData> {
    let data_len = u32::try_from(data.len()).map_err(|_| {
        bad(format!(
            "'data' chunk is {} bytes, past the 4 GiB a WAV size field can declare; split \
             the recording into smaller files",
            data.len()
        ))
    })?;
    let riff_size = data_len.saturating_add((HEADER_LEN - 8) as u32);
    let header = canonical_header(f, riff_size, data_len);
    let rejected = |err: hound::Error| {
        bad(format!(
            "hound rejected the stream (format tag {}, {} channels, {} Hz, {} bits per \
             sample, block alignment {}): {err}; supported are {SUPPORTED}",
            f.format_tag, f.channels, f.sample_rate, f.bits_per_sample, f.block_align
        ))
    };

    let reader = hound::WavReader::new(Cursor::new(header).chain(data)).map_err(rejected)?;
    let spec = reader.spec();
    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(rejected)?,
        (hound::SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            // i16::MAX at 16 bits matches the writer's scale, so the round
            // trip is symmetric.
            let full_scale = ((1i64 << (bits - 1)) - 1) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| scale(v as f64, full_scale)))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(rejected)?
        }
        (hound::SampleFormat::Int, bits) => {
            return Err(bad(format!(
                "integer PCM with {bits} bits per sample is unsupported; supported are \
                 {SUPPORTED}"
            )));
        }
    };
    Ok(WavData {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        layout: mask_layout(f.channel_mask, f.channels),
    })
}

/// Normalise an integer sample against positive full scale, clamped to `[-1, 1]`
/// because the negative extreme is one step past `-1`.
///
/// Shared with the `decode` module (feature `decode`) so the symphonia-backed path and this one
/// return identical values for identical bytes.
pub(crate) fn scale(value: f64, full_scale: f64) -> f32 {
    (value / full_scale).clamp(-1.0, 1.0) as f32
}

/// Downmix interleaved samples to mono by averaging channels.
///
/// Returns the input unchanged when `channels == 1`. Returns
/// [`Error::InvalidArgument`] when `channels` is 0 or `samples.len()` is not
/// divisible by `channels`.
pub fn to_mono(samples: &[f32], channels: u16) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: "channel count is 0".to_string(),
        });
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }
    let channels = channels as usize;
    if !samples.len().is_multiple_of(channels) {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "sample count {} is not divisible by the channel count {channels}",
                samples.len()
            ),
        });
    }
    let scale = 1.0f32 / channels as f32;
    Ok(samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::encode::encode_wav_pcm16;

    use crate::wav::format::{
        DATA_SIZE_OFFSET as DATA_SIZE_OFF, RIFF_SIZE_OFFSET as RIFF_SIZE_OFF,
    };

    fn pcm16_with_sizes(samples: &[f32], riff: u32, data: u32) -> Vec<u8> {
        let mut bytes = encode_wav_pcm16(samples, 16_000).expect("encode");
        assert_eq!(&bytes[36..40], b"data");
        bytes[RIFF_SIZE_OFF..RIFF_SIZE_OFF + 4].copy_from_slice(&riff.to_le_bytes());
        bytes[DATA_SIZE_OFF..DATA_SIZE_OFF + 4].copy_from_slice(&data.to_le_bytes());
        bytes
    }

    #[test]
    fn streamed_sizes_clamp_data_to_the_buffer() {
        let samples = [0.0f32, 0.25, -0.5, 0.75, -1.0, 0.5];
        let exact = decode_wav(&encode_wav_pcm16(&samples, 16_000).expect("encode"))
            .expect("exact header decodes");
        let streamed = decode_wav(&pcm16_with_sizes(&samples, u32::MAX, u32::MAX))
            .expect("0xFFFFFFFF sizes decode");
        assert_eq!(streamed, exact);
        assert_eq!(streamed.frames(), samples.len());
    }

    #[test]
    fn streamed_data_drops_a_trailing_partial_frame() {
        let samples = [0.0f32, 0.25, -0.5, 0.75];
        let mut bytes = pcm16_with_sizes(&samples, u32::MAX, u32::MAX);
        bytes.push(0x7f);
        let decoded = decode_wav(&bytes).expect("odd trailing byte is dropped");
        assert_eq!(decoded.frames(), samples.len());
    }

    #[test]
    fn declared_data_size_must_be_whole_frames() {
        let samples = [0.0f32, 0.25, -0.5, 0.75];
        let mut bytes = pcm16_with_sizes(&samples, 0, 7);
        bytes.truncate(44 + 7);
        let err = decode_wav(&bytes).expect_err("7 bytes of 2-byte frames");
        assert!(err.to_string().contains("not a whole number"), "{err}");
    }

    #[test]
    fn fmt_overrun_is_still_an_error() {
        let mut bytes = encode_wav_pcm16(&[0.0f32; 4], 16_000).expect("encode");
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = decode_wav(&bytes).expect_err("fmt chunk overruns");
        assert!(err.to_string().contains("overruns"), "{err}");
    }
}
