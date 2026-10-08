//! RIFF/WAVE format constants, the canonical 44-byte header layout shared
//! by [`super::encode`], [`super::decode`] and [`super::stream`], and the
//! checks that keep a header's rate and size fields in range.

use crate::error::{Error, Result};

/// Uncompressed integer PCM (`WAVE_FORMAT_PCM`).
pub const FORMAT_PCM: u16 = 1;
/// 32-bit IEEE float samples (`WAVE_FORMAT_IEEE_FLOAT`).
pub const FORMAT_IEEE_FLOAT: u16 = 3;
/// Extended header carrying the real format tag in its `SubFormat` GUID.
pub const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Byte length of the canonical header: RIFF (12), `fmt ` with a 16-byte
/// body (24), and the `data` chunk header (8). Every mono and stereo file
/// the encoders write, and every stream header, has this length.
pub const HEADER_LEN: usize = 44;

/// Byte length of the `WAVE_FORMAT_EXTENSIBLE` header hound writes for PCM
/// with more than two channels: the `fmt ` body grows from 16 to 40 bytes.
pub(super) const EXTENSIBLE_HEADER_LEN: usize = 68;

/// Offset of the RIFF chunk size in every header.
pub(super) const RIFF_SIZE_OFFSET: usize = 4;

/// Offset of the `data` chunk size in the canonical header.
pub(super) const DATA_SIZE_OFFSET: usize = HEADER_LEN - 4;

/// The six fields of a 16-byte `fmt ` chunk body.
pub(super) struct FmtFields {
    pub(super) format_tag: u16,
    pub(super) channels: u16,
    pub(super) sample_rate: u32,
    pub(super) byte_rate: u32,
    pub(super) block_align: u16,
    pub(super) bits_per_sample: u16,
    /// `WAVE_FORMAT_EXTENSIBLE`'s `dwChannelMask`, 0 when the chunk has
    /// none. The canonical header does not carry it.
    pub(super) channel_mask: u32,
}

/// The canonical header: `RIFF`, `WAVE`, a 16-byte `fmt ` chunk, then the
/// `data` chunk id and size.
pub(super) fn canonical_header(fmt: &FmtFields, riff_size: u32, data_size: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&fmt.format_tag.to_le_bytes());
    out.extend_from_slice(&fmt.channels.to_le_bytes());
    out.extend_from_slice(&fmt.sample_rate.to_le_bytes());
    out.extend_from_slice(&fmt.byte_rate.to_le_bytes());
    out.extend_from_slice(&fmt.block_align.to_le_bytes());
    out.extend_from_slice(&fmt.bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out
}

/// An [`Error::InvalidArgument`] for the argument `arg`.
pub(super) fn bad(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

/// `fmt ` chunk rate fields: `byte_rate` (bytes per second) and `block_align`
/// (bytes per frame), both checked against the widths the header stores them in.
///
/// A sample rate of 0 describes no playable file. `block_align` is a `u16`, so
/// it overflows above 32767 channels at 16 bits, and `byte_rate` is a `u32`, so
/// a large `sample_rate` and channel count together overflow it. Either would
/// silently emit a header describing a different file.
pub(super) fn checked_wave_rates(
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
) -> Result<(u32, u16)> {
    if sample_rate == 0 {
        return Err(bad(
            "sample_rate",
            "sample rate is 0 Hz; pass the rate the samples were produced at".to_string(),
        ));
    }
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels.checked_mul(bytes_per_sample).ok_or_else(|| {
        bad(
            "channels",
            format!(
                "channel count {channels} at {bits_per_sample} bits per sample overflows \
                 the WAV header's u16 block alignment"
            ),
        )
    })?;
    let byte_rate = sample_rate.checked_mul(block_align as u32).ok_or_else(|| {
        bad(
            "sample_rate",
            format!(
                "sample rate {sample_rate} at {block_align} bytes per frame overflows \
                 the WAV header's u32 byte rate"
            ),
        )
    })?;
    Ok((byte_rate, block_align))
}

/// `data` chunk size in bytes: `sample_count * bytes_per_sample`, checked against
/// the WAV container's 32-bit size field.
pub(super) fn checked_data_size(sample_count: usize, bytes_per_sample: usize) -> Result<u32> {
    sample_count
        .checked_mul(bytes_per_sample)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| {
            bad(
                "samples",
                format!(
                    "sample count {sample_count} is too large to encode as a WAV file \
                     ({sample_count} x {bytes_per_sample} bytes overflows a u32 data size)"
                ),
            )
        })
}

/// `RIFF` chunk size in bytes: the header after the 8-byte RIFF preamble plus
/// `data_size`, checked against the WAV container's 32-bit size field.
pub(super) fn checked_riff_size(
    sample_count: usize,
    data_size: u32,
    header_len: usize,
) -> Result<u32> {
    let fixed = (header_len - 8) as u32;
    fixed.checked_add(data_size).ok_or_else(|| {
        bad(
            "samples",
            format!(
                "sample count {sample_count} produces a {data_size}-byte data chunk, \
                 too large to encode as a WAV file ({fixed} + data_size overflows a u32 \
                 RIFF size)"
            ),
        )
    })
}
