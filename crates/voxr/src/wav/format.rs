//! RIFF/WAVE format constants and the canonical 44-byte header layout shared
//! by [`super::encode`], [`super::decode`] and [`super::stream`].

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
