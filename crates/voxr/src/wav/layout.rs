//! The `fmt ` chunk of a RIFF/WAVE file and the channel layout its
//! `WAVE_FORMAT_EXTENSIBLE` `dwChannelMask` names.
//!
//! [`super::decode`] parses the chunk through [`parse_fmt`] and reports the
//! layout through [`mask_layout`]. The `decode` feature reads the layout of a
//! WAV prefix through `wav_layout`.

use crate::error::Result;

use super::decode::{bad, read_u16, read_u32};
use super::format::{FORMAT_EXTENSIBLE, FmtFields};

/// Parse a `fmt ` chunk body, resolving `WAVE_FORMAT_EXTENSIBLE` to the format
/// tag its SubFormat GUID carries.
pub(super) fn parse_fmt(bytes: &[u8], body: usize, size: usize) -> Result<FmtFields> {
    if size < 16 {
        return Err(bad(format!(
            "'fmt ' chunk is {size} bytes, the minimum is 16"
        )));
    }
    let field = |off: usize| -> Result<u16> {
        read_u16(bytes, body + off)
            .ok_or_else(|| bad(format!("'fmt ' chunk is truncated at offset {off}")))
    };
    let mut format_tag = field(0)?;
    let mut channel_mask = 0;
    let channels = field(2)?;
    let sample_rate = read_u32(bytes, body + 4)
        .ok_or_else(|| bad("'fmt ' chunk is truncated at the sample rate".to_string()))?;
    let byte_rate = read_u32(bytes, body + 8)
        .ok_or_else(|| bad("'fmt ' chunk is truncated at the byte rate".to_string()))?;
    let block_align = field(12)?;
    let bits_per_sample = field(14)?;

    if format_tag == FORMAT_EXTENSIBLE {
        // Extensible layout: 16 standard bytes, cbSize (2), validBitsPerSample (2),
        // dwChannelMask (4), then a 16-byte SubFormat GUID whose first two bytes
        // hold the real format tag.
        if size < 40 {
            return Err(bad(format!(
                "WAVE_FORMAT_EXTENSIBLE 'fmt ' chunk is {size} bytes, the minimum is 40"
            )));
        }
        let cb_size = field(16)?;
        if cb_size < 22 {
            return Err(bad(format!(
                "WAVE_FORMAT_EXTENSIBLE cbSize is {cb_size}, the minimum is 22"
            )));
        }
        channel_mask = read_u32(bytes, body + 20)
            .ok_or_else(|| bad("'fmt ' chunk is truncated at the channel mask".to_string()))?;
        format_tag = field(24)?;
        if format_tag == FORMAT_EXTENSIBLE {
            return Err(bad(
                "WAVE_FORMAT_EXTENSIBLE SubFormat GUID is itself 0xFFFE".to_string(),
            ));
        }
    }

    Ok(FmtFields {
        format_tag,
        channels,
        sample_rate,
        // The byte rate is redundant with the other fields. hound rejects a
        // file whose byte rate disagrees with them, so a consistent one is
        // derived here; only an overflowing product keeps the declared value
        // for hound to reject.
        byte_rate: sample_rate
            .checked_mul(block_align as u32)
            .unwrap_or(byte_rate),
        block_align,
        bits_per_sample,
        channel_mask,
    })
}

/// The layout a `dwChannelMask` of `channels` channels names, as ffmpeg's
/// WAV demuxer reads it: the mask when it names exactly that many
/// positions, else none.
pub(crate) fn mask_layout(mask: u32, channels: u16) -> Option<u64> {
    (mask != 0 && mask.count_ones() == u32::from(channels)).then_some(u64::from(mask))
}

/// The layout the `fmt ` chunk of the WAV file `bytes` names, read as
/// [`super::decode::decode_wav`] reads it. `None` when it is no WAV file or
/// names none.
#[cfg(feature = "decode")]
pub(crate) fn wav_layout(bytes: &[u8]) -> Option<u64> {
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return None;
    }
    let mut pos = 12usize;
    while let Some(size) = read_u32(bytes, pos + 4) {
        let size = size as usize;
        let body = pos + 8;
        if bytes.get(pos..pos + 4) == Some(b"fmt ") {
            let f = parse_fmt(bytes, body, size).ok()?;
            return mask_layout(f.channel_mask, f.channels);
        }
        pos = body.checked_add(size + (size & 1))?;
    }
    None
}
