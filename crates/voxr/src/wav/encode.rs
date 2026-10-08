//! RIFF/WAVE encoder for f32 samples in `[-1, 1]`.
//!
//! The TTS path produces `Vec<f32>` waveform samples at the model's native
//! sample rate. 16-bit signed PCM files (small, widely supported) go through
//! hound. 32-bit IEEE float files (lossless) use the canonical 44-byte header
//! from [`super::format`]: hound writes float as `WAVE_FORMAT_EXTENSIBLE` with a
//! 68-byte header, and [`super::stream::wav_stream_header_f32`] must match the
//! finished file byte for byte outside the two size fields.

use std::io::Cursor;

use crate::error::{Error, Result};

pub use super::format::HEADER_LEN;
use super::format::{
    EXTENSIBLE_HEADER_LEN, FORMAT_IEEE_FLOAT, FORMAT_PCM, FmtFields, bad, canonical_header,
    checked_data_size, checked_riff_size, checked_wave_rates,
};

/// Encode mono f32 samples as `audio/wav` with signed 16-bit PCM.
///
/// Samples outside `[-1, 1]` are clipped. Returns a standalone byte buffer
/// suitable as an HTTP response body, or [`Error::InvalidArgument`] when the
/// sample rate is 0 or the sample count is too large for the WAV container's
/// 32-bit size fields.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    encode_wav_pcm16_multichannel(samples, sample_rate, 1)
}

/// Encode interleaved multi-channel f32 samples as 16-bit PCM WAV.
///
/// One or two channels produce the canonical [`HEADER_LEN`]-byte header. More
/// than two produce the 68-byte `WAVE_FORMAT_EXTENSIBLE` header the WAV
/// format requires for them.
///
/// Returns [`Error::InvalidArgument`] when `channels` is 0, when `samples.len()`
/// is not divisible by `channels`, when the sample rate is 0, when the rate and
/// channel count overflow the header's rate fields, or when the sample count is
/// too large for the WAV container's 32-bit size fields.
pub fn encode_wav_pcm16_multichannel(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<Vec<u8>> {
    if channels == 0 {
        return Err(bad("channels", "channel count is 0".to_string()));
    }
    if !samples.len().is_multiple_of(channels as usize) {
        return Err(bad(
            "samples",
            format!(
                "sample count {} is not divisible by the channel count {channels}",
                samples.len()
            ),
        ));
    }
    // hound derives the same rate fields without overflow checks and divides
    // by the sample rate, so both are checked before it runs.
    checked_wave_rates(sample_rate, channels, 16)?;
    let header_len = if channels > 2 {
        EXTENSIBLE_HEADER_LEN
    } else {
        HEADER_LEN
    };
    let data_size = checked_data_size(samples.len(), 2)?; // 2 bytes per i16 sample
    checked_riff_size(samples.len(), data_size, header_len)?;

    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let what = || format!("{} samples of {channels}-channel 16-bit PCM", samples.len());
    let mut cursor = Cursor::new(Vec::with_capacity(header_len + data_size as usize));
    let mut writer =
        hound::WavWriter::new(&mut cursor, spec).map_err(|e| write_error(&what(), e))?;
    for &s in samples {
        writer
            .write_sample(pcm16(s))
            .map_err(|e| write_error(&what(), e))?;
    }
    writer.finalize().map_err(|e| write_error(&what(), e))?;
    Ok(cursor.into_inner())
}

/// A hound write error. The checks before hound runs cover every input it
/// rejects, and the sink is an in-memory buffer, so reaching this is a bug.
fn write_error(what: &str, err: hound::Error) -> Error {
    Error::DataError {
        reason: format!(
            "hound failed to write {what} into memory: {err}; the input passed every \
             size check, so report this as a voxr bug"
        ),
    }
}

/// One sample as clipped signed 16-bit PCM, scaled by `i16::MAX` so `-1.0`
/// and `1.0` map to symmetric codes.
fn pcm16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

/// Append `samples` as clipped little-endian signed 16-bit PCM.
fn write_pcm16(out: &mut Vec<u8>, samples: &[f32]) {
    for &s in samples {
        out.extend_from_slice(&pcm16(s).to_le_bytes());
    }
}

/// Append `samples` as little-endian IEEE f32.
fn write_f32(out: &mut Vec<u8>, samples: &[f32]) {
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
}

/// Encode mono f32 samples as `audio/wav` with 32-bit float PCM (no clipping).
///
/// Returns [`Error::InvalidArgument`] when the sample rate is 0, when it
/// overflows the header's byte rate, or when the sample count is too large for
/// the WAV container's 32-bit size fields.
pub fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let mut out = mono_canonical_header(FORMAT_IEEE_FLOAT, 32, samples.len(), sample_rate)?;
    write_f32(&mut out, samples);
    Ok(out)
}

/// Encode mono signed 16-bit PCM values as `audio/wav`, each written as it is.
///
/// The header is the one [`encode_wav_pcm16`] writes for the same sample
/// count and rate. A caller that rounds samples to 16 bits its own way
/// passes the rounded values here.
///
/// Returns [`Error::InvalidArgument`] when the sample rate is 0, when it
/// overflows the header's byte rate, or when the value count is too large for
/// the WAV container's 32-bit size fields.
pub fn encode_wav_pcm16_values(values: &[i16], sample_rate: u32) -> Result<Vec<u8>> {
    let mut out = mono_canonical_header(FORMAT_PCM, 16, values.len(), sample_rate)?;
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// The canonical header of a mono file of `len` samples of `bits_per_sample`
/// bits in format `format_tag`, with room reserved for the samples after it.
fn mono_canonical_header(
    format_tag: u16,
    bits_per_sample: u16,
    len: usize,
    sample_rate: u32,
) -> Result<Vec<u8>> {
    let (byte_rate, block_align) = checked_wave_rates(sample_rate, 1, bits_per_sample)?;
    let data_size = checked_data_size(len, usize::from(bits_per_sample / 8))?;
    let riff_size = checked_riff_size(len, data_size, HEADER_LEN)?;
    let fmt = FmtFields {
        format_tag,
        channels: 1,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        channel_mask: 0,
    };
    let mut out = canonical_header(&fmt, riff_size, data_size);
    out.reserve(data_size as usize);
    Ok(out)
}

/// Encode mono f32 samples as raw little-endian PCM16 (no WAV header).
///
/// Returned bytes are suitable for `response_format=pcm` streaming, and as
/// the chunks after a [`super::stream::wav_stream_header_pcm16`].
pub fn encode_pcm16_raw(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    write_pcm16(&mut out, samples);
    out
}

/// Encode mono f32 samples as raw little-endian IEEE f32 (no WAV header),
/// the chunks after a [`super::stream::wav_stream_header_f32`].
pub fn encode_f32_raw(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    write_f32(&mut out, samples);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_header_shape() {
        let wav = encode_wav_pcm16(&[0.0; 10], 24_000).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        // fmt chunk size = 16
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        // PCM format = 1
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        // channels = 1
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        // sample rate
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        // bits per sample = 16
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        // data_size = 10 * 2
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 20);
    }

    #[test]
    fn pcm16_clips_out_of_range() {
        let wav = encode_wav_pcm16(&[2.0, -2.0], 8_000).unwrap();
        // Samples live at offset 44, 2 bytes each.
        let s0 = i16::from_le_bytes(wav[44..46].try_into().unwrap());
        let s1 = i16::from_le_bytes(wav[46..48].try_into().unwrap());
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, -i16::MAX);
    }

    #[test]
    fn pcm16_roundtrips_within_quantization() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_pcm16(&input, 16_000).unwrap();
        let mut decoded = Vec::with_capacity(input.len());
        for chunk in wav[44..].as_chunks::<2>().0 {
            let s = i16::from_le_bytes(*chunk);
            decoded.push(s as f32 / i16::MAX as f32);
        }
        for (a, b) in input.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1.0 / i16::MAX as f32 + 1e-6);
        }
    }

    #[test]
    fn f32_header_uses_ieee_format() {
        let wav = encode_wav_f32(&[0.25], 48_000).unwrap();
        // Format code = 3 (IEEE float)
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 3);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 32);
        // The single sample is bit-identical.
        let s = f32::from_le_bytes(wav[44..48].try_into().unwrap());
        assert_eq!(s, 0.25);
    }

    #[test]
    fn pcm16_values_are_written_as_they_are_behind_the_pcm16_header() {
        let wav = encode_wav_pcm16_values(&[16_384, -16_384, i16::MIN], 24_000).unwrap();
        let reference = encode_wav_pcm16(&[0.0; 3], 24_000).unwrap();
        assert_eq!(wav[..HEADER_LEN], reference[..HEADER_LEN]);
        assert_eq!(wav[HEADER_LEN..], [0x00, 0x40, 0x00, 0xC0, 0x00, 0x80]);
    }

    #[test]
    fn raw_pcm16_has_no_header() {
        let raw = encode_pcm16_raw(&[0.0, 1.0]);
        assert_eq!(raw.len(), 4);
        let s1 = i16::from_le_bytes(raw[2..4].try_into().unwrap());
        assert_eq!(s1, i16::MAX);
    }

    #[test]
    fn raw_f32_is_the_wav_body_without_its_header() {
        let samples = [0.25f32, -1.5, 3.0e-3];
        let raw = encode_f32_raw(&samples);
        let wav = encode_wav_f32(&samples, 48_000).unwrap();
        assert_eq!(raw, wav[HEADER_LEN..]);
    }

    #[test]
    fn multichannel_rejects_zero_channels() {
        assert!(encode_wav_pcm16_multichannel(&[0.0, 1.0], 8_000, 0).is_err());
    }

    #[test]
    fn multichannel_rejects_indivisible_sample_count() {
        assert!(encode_wav_pcm16_multichannel(&[0.0, 1.0, 0.5], 8_000, 2).is_err());
    }
    /// `channels * bytes_per_sample` is stored in a `u16`. Above 32767 channels at
    /// 16 bits it wraps, and the header would declare a block alignment smaller
    /// than one frame while the data chunk stays full size.
    #[test]
    fn rejects_a_channel_count_that_overflows_block_align() {
        let err = encode_wav_pcm16_multichannel(&[], 16_000, 40_000)
            .expect_err("40000 channels at 16 bits overflows the u16 block alignment");
        let msg = err.to_string();
        assert!(msg.contains("40000"), "{msg}");
        assert!(msg.contains("block alignment"), "{msg}");
    }

    /// `sample_rate * block_align` is stored in a `u32`, so an extreme sample rate
    /// wraps it and the header would declare a byte rate far below the real one.
    #[test]
    fn rejects_a_sample_rate_that_overflows_byte_rate() {
        let err = encode_wav_pcm16(&[0.0; 4], u32::MAX)
            .expect_err("u32::MAX Hz at 2 bytes per frame overflows the u32 byte rate");
        let msg = err.to_string();
        assert!(msg.contains(&u32::MAX.to_string()), "{msg}");
        assert!(msg.contains("byte rate"), "{msg}");
    }

    /// The largest block alignment that still fits: 32767 channels at 16 bits is
    /// exactly `u16::MAX - 1`, so the guard above must not reject it.
    #[test]
    fn accepts_the_largest_representable_block_align() {
        let wav = encode_wav_pcm16_multichannel(&[], 8_000, 32_767)
            .expect("32767 channels at 16 bits fits a u16 block alignment");
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 65_534);
    }

    /// More than two channels need `WAVE_FORMAT_EXTENSIBLE`; the decoder must
    /// read that header back to the same samples.
    #[test]
    fn three_channel_pcm16_uses_the_extensible_header_and_round_trips() {
        let input = [0.5f32, -0.5, 0.25, 1.0, -1.0, 0.0];
        let wav = encode_wav_pcm16_multichannel(&input, 16_000, 3).unwrap();
        assert_eq!(wav.len(), EXTENSIBLE_HEADER_LEN + input.len() * 2);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 0xFFFE);
        let decoded = crate::wav::decode_wav(&wav).unwrap();
        assert_eq!(decoded.channels, 3);
        for (a, b) in input.iter().zip(decoded.samples.iter()) {
            assert!((a - b).abs() <= 1.0 / 32_768.0, "{a} vs {b}");
        }
    }

    #[test]
    fn rejects_a_zero_sample_rate() {
        let err = encode_wav_pcm16(&[0.0; 4], 0).expect_err("0 Hz");
        assert!(err.to_string().contains("0 Hz"), "{err}");
        assert!(encode_wav_f32(&[0.0; 4], 0).is_err());
    }
}
