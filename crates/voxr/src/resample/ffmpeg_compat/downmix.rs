//! The mono mix of ffmpeg's default rematrix: `-ac 1` with float output.
//!
//! libswresample's `swr_build_matrix` mixes every input channel into the
//! one centre output with the default levels: centre and surround at
//! -3 dB, LFE dropped. With float output it does not normalize the
//! matrix, whatever the input sample format. Each channel's weight into
//! the centre is therefore fixed by its position:
//!
//! | Position | Weight |
//! |---|---|
//! | front left and right, front left and right of centre, top front left and right | `M_SQRT1_2` |
//! | front centre | 1 |
//! | back left, right and centre, side left and right | 1/2 |
//! | LFE and every other position | 0 |
//!
//! A 22.2 layout is the exception: its rematrix drops the side and top
//! channels too. A stream with no layout takes ffmpeg's default layout
//! for its channel count. ffmpeg has none for some counts, and refuses
//! to mix those to mono.
//!
//! The mix sums the weighted channels in layout order as `f32`, from 0,
//! skipping a weight of 0, as libswresample's float mixing loop does.

use crate::error::{Error, Result};

/// The ffmpeg channel bits (`AV_CH_*`) the mix names.
mod bit {
    pub const FL: u64 = 1 << 0;
    pub const FR: u64 = 1 << 1;
    pub const FC: u64 = 1 << 2;
    pub const LFE: u64 = 1 << 3;
    pub const BL: u64 = 1 << 4;
    pub const BR: u64 = 1 << 5;
    pub const FLC: u64 = 1 << 6;
    pub const FRC: u64 = 1 << 7;
    pub const BC: u64 = 1 << 8;
    pub const SL: u64 = 1 << 9;
    pub const SR: u64 = 1 << 10;
    pub const TFL: u64 = 1 << 12;
    pub const TFR: u64 = 1 << 14;
    pub const TBL: u64 = 1 << 15;
    pub const TBR: u64 = 1 << 17;
    pub const TSL: u64 = 1 << 36;
    pub const TSR: u64 = 1 << 37;
    /// Every channel of 22.2 (`AV_CH_LAYOUT_22POINT2`).
    pub const LAYOUT_22_2: u64 = 0x3_FFFF | (0x3F << 35);
}

/// ffmpeg's default layout for `channels`: the first layout of that count
/// in its standard list (`av_channel_layout_default`). `None` for a count
/// with no standard layout.
pub fn ffmpeg_default_layout(channels: u16) -> Option<u64> {
    use bit::*;
    let surround_5_1 = FL | FR | FC | LFE | BL | BR;
    let top_four = TFL | TFR | TBL | TBR;
    Some(match channels {
        // mono, stereo, 2.1, 4.0, 5.0, 5.1, 6.1, 7.1
        1 => FC,
        2 => FL | FR,
        3 => FL | FR | LFE,
        4 => FL | FR | FC | BC,
        5 => FL | FR | FC | BL | BR,
        6 => surround_5_1,
        7 => FL | FR | FC | LFE | BC | SL | SR,
        8 => surround_5_1 | SL | SR,
        // 5.1.4, 7.1.4, 9.1.4, 9.1.6, 22.2
        10 => FL | FR | FC | LFE | SL | SR | top_four,
        12 => surround_5_1 | SL | SR | top_four,
        14 => surround_5_1 | FLC | FRC | SL | SR | top_four,
        16 => surround_5_1 | FLC | FRC | SL | SR | top_four | TSL | TSR,
        24 => LAYOUT_22_2,
        _ => return None,
    })
}

/// The weight of the channel at `position` (one `AV_CH_*` bit) in the
/// mono mix of `layout`.
fn weight(position: u64, layout: u64) -> f32 {
    use bit::*;
    let half_power = std::f32::consts::FRAC_1_SQRT_2;
    let fronts = FL | FR | FLC | FRC;
    let rears = BL | BR | BC;
    if layout == LAYOUT_22_2 {
        return match position {
            p if p & fronts != 0 => half_power,
            FC => 1.0,
            p if p & rears != 0 => 0.5,
            _ => 0.0,
        };
    }
    match position {
        p if p & (fronts | TFL | TFR) != 0 => half_power,
        FC => 1.0,
        p if p & (rears | SL | SR) != 0 => 0.5,
        _ => 0.0,
    }
}

/// Interleaved samples of `channels` channels mixed to mono as
/// `ffmpeg -ac 1` mixes them into float output.
///
/// `layout` is the ffmpeg channel mask (`AV_CH_*` bits) of the channels,
/// interleaved in ascending bit order, as [`crate::WavData::layout`]
/// gives it. `None`, or a mask whose bit count is not `channels`, takes
/// [`ffmpeg_default_layout`]. Mono comes back unchanged.
///
/// Returns [`Error::InvalidArgument`] when `channels` is 0, when
/// `samples.len()` is not divisible by `channels`, or when the stream has
/// no layout and ffmpeg has no default one for its channel count.
pub fn to_mono_like_ffmpeg(
    samples: &[f32],
    channels: u16,
    layout: Option<u64>,
) -> Result<Vec<f32>> {
    if channels == 0 {
        return Err(Error::InvalidArgument {
            arg: "channels",
            reason: "channel count is 0".to_string(),
        });
    }
    let n = usize::from(channels);
    if !samples.len().is_multiple_of(n) {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "sample count {} is not divisible by the channel count {n}",
                samples.len()
            ),
        });
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }
    let mask = layout
        .filter(|m| m.count_ones() == u32::from(channels))
        .or_else(|| ffmpeg_default_layout(channels))
        .ok_or_else(|| Error::InvalidArgument {
            arg: "layout",
            reason: format!(
                "{channels} channels carry no channel layout and ffmpeg has no default one for that count, so it does not mix them to mono either. Re-encode the file with a standard layout"
            ),
        })?;
    // Each channel's position, lowest bit first, and its nonzero weight.
    let mixed: Vec<(usize, f32)> = (0..64)
        .map(|b| 1u64 << b)
        .filter(|p| mask & p != 0)
        .enumerate()
        .map(|(c, p)| (c, weight(p, mask)))
        .filter(|&(_, w)| w != 0.0)
        .collect();
    Ok(samples
        .chunks_exact(n)
        .map(|frame| mixed.iter().fold(0.0f32, |v, &(c, w)| v + frame[c] * w))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_mixes_at_minus_three_db_each() -> Result<()> {
        let mixed = to_mono_like_ffmpeg(&[0.5, 0.5, 1.0, -1.0, 0.25, 0.0], 2, None)?;
        let c = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(mixed, vec![0.5 * c + 0.5 * c, 0.0, 0.25 * c]);
        Ok(())
    }

    #[test]
    fn surround_takes_centre_whole_rears_at_half_and_drops_lfe() -> Result<()> {
        let c = std::f32::consts::FRAC_1_SQRT_2;
        // 5.1: FL FR FC LFE BL BR.
        let frame = [0.1, 0.2, 0.3, 0.9, 0.4, 0.6];
        let want = 0.1 * c + 0.2 * c + 0.3 + 0.4 * 0.5 + 0.6 * 0.5;
        assert_eq!(to_mono_like_ffmpeg(&frame, 6, Some(0x3F))?, vec![want]);
        assert_eq!(to_mono_like_ffmpeg(&frame, 6, None)?, vec![want]);
        // 4 channels with no layout are 4.0: FL FR FC BC.
        let quad = [0.1, 0.2, 0.3, 0.4];
        let four = 0.1 * c + 0.2 * c + 0.3 + 0.4 * 0.5;
        assert_eq!(to_mono_like_ffmpeg(&quad, 4, None)?, vec![four]);
        // quad: FL FR BL BR.
        let back = 0.1 * c + 0.2 * c + 0.3 * 0.5 + 0.4 * 0.5;
        assert_eq!(to_mono_like_ffmpeg(&quad, 4, Some(0x33))?, vec![back]);
        Ok(())
    }

    #[test]
    fn a_count_with_no_default_layout_refuses() {
        assert!(to_mono_like_ffmpeg(&[0.0; 9], 9, None).is_err());
        assert!(to_mono_like_ffmpeg(&[0.0; 9], 9, Some(0x7)).is_err());
        assert!(to_mono_like_ffmpeg(&[0.0; 9], 9, Some(0x1FF)).is_ok());
    }

    #[test]
    fn mono_passes_and_bad_frames_refuse() -> Result<()> {
        assert_eq!(to_mono_like_ffmpeg(&[0.1, 0.2], 1, None)?, vec![0.1, 0.2]);
        assert!(to_mono_like_ffmpeg(&[0.1, 0.2, 0.3], 2, None).is_err());
        assert!(to_mono_like_ffmpeg(&[0.1], 0, None).is_err());
        Ok(())
    }
}
