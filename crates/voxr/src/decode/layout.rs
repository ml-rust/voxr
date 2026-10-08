//! The channel layout a decode carries: symphonia's channel positions as
//! ffmpeg channel bits (`AV_CH_*`), and the plane order that interleaves
//! the channels in ffmpeg's order, lowest bit first.

use symphonia::core::audio::Channels;

/// ffmpeg's bit for one symphonia position bit, the bit's index. `None`
/// for a position ffmpeg has no bit for: rear left and right of centre.
fn ffmpeg_bit(index: u32) -> Option<u64> {
    match index {
        // Front left through top back right: the same 18 bits in both.
        0..=17 => Some(1 << index),
        // Front left and right wide: ffmpeg's AAC decoder names 7.1's
        // second front pair left and right of centre.
        20 => Some(1 << 6),
        21 => Some(1 << 7),
        // Front left, centre and right high: top front.
        22 => Some(1 << 12),
        23 => Some(1 << 13),
        24 => Some(1 << 14),
        // LFE 2.
        25 => Some(1 << 35),
        _ => None,
    }
}

/// The ffmpeg layout of decoded planes at `channels`, and the plane each
/// output channel reads, in ascending ffmpeg bit order. symphonia orders
/// planes by its own bits. One channel is ffmpeg's mono, front centre.
/// The layout is `None` when a position has no ffmpeg bit or two share
/// one. The planes then keep their order.
pub(super) fn ffmpeg_layout(channels: Channels, count: usize) -> (Option<u64>, Vec<usize>) {
    let keep = (None, (0..count).collect());
    if count == 1 {
        return (Some(1 << 2), vec![0]);
    }
    let bits: Option<Vec<u64>> = (0..32u32)
        .filter(|b| channels.bits() & (1 << b) != 0)
        .map(ffmpeg_bit)
        .collect();
    let Some(bits) = bits.filter(|b| b.len() == count) else {
        return keep;
    };
    let mask = bits.iter().fold(0u64, |m, b| m | b);
    if mask.count_ones() as usize != count {
        return keep;
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&plane| bits[plane]);
    (Some(mask), order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_map_to_ffmpeg_bits_in_its_order() {
        let surround = Channels::FRONT_LEFT
            | Channels::FRONT_RIGHT
            | Channels::FRONT_CENTRE
            | Channels::LFE1
            | Channels::SIDE_LEFT
            | Channels::SIDE_RIGHT;
        assert_eq!(
            ffmpeg_layout(surround, 6),
            (Some(0x60F), vec![0, 1, 2, 3, 4, 5])
        );
        // AAC 7.1: the wide pair is ffmpeg's left and right of centre,
        // which come before the sides.
        let wide = surround | Channels::FRONT_LEFT_WIDE | Channels::FRONT_RIGHT_WIDE;
        assert_eq!(
            ffmpeg_layout(wide, 8),
            (Some(0x6CF), vec![0, 1, 2, 3, 6, 7, 4, 5])
        );
        assert_eq!(ffmpeg_layout(Channels::FRONT_LEFT, 1), (Some(0x4), vec![0]));
        let unnamed = Channels::FRONT_LEFT | Channels::REAR_LEFT_CENTRE;
        assert_eq!(ffmpeg_layout(unnamed, 2), (None, vec![0, 1]));
    }
}
