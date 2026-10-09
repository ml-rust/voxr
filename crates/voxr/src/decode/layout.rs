//! The channel layout a decode carries: symphonia's channel positions as
//! ffmpeg channel bits (`AV_CH_*`), and the plane order that interleaves
//! the channels in ffmpeg's order, lowest bit first.

use symphonia::core::audio::Channels;

/// ffmpeg's bit for one symphonia position bit, the bit's index. `None`
/// for a position ffmpeg has no bit for.
fn ffmpeg_bit(index: u32) -> Option<u64> {
    match index {
        // Front left through top back right: the same 18 bits in both.
        0..=17 => Some(1 << index),
        // LFE 2, top side left and right, bottom front centre, left and
        // right.
        18..=23 => Some(1 << (index + 17)),
        // Front left and right wide: ffmpeg's AAC decoder names 7.1's
        // second front pair left and right of centre.
        24 => Some(1 << 6),
        25 => Some(1 << 7),
        _ => None,
    }
}

/// The ffmpeg layout of decoded planes at `channels`, and the plane each
/// output channel reads, in ascending ffmpeg bit order. symphonia orders
/// planes by its own bits. One channel is ffmpeg's mono, front centre.
/// The layout is `None` when the channels are not positions, a position
/// has no ffmpeg bit, or two share one. The planes then keep their order.
pub(super) fn ffmpeg_layout(channels: &Channels, count: usize) -> (Option<u64>, Vec<usize>) {
    let keep = (None, (0..count).collect());
    if count == 1 {
        return (Some(1 << 2), vec![0]);
    }
    let Channels::Positioned(positions) = channels else {
        return keep;
    };
    let bits: Option<Vec<u64>> = (0..64u32)
        .filter(|b| positions.bits() & (1 << b) != 0)
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
    use symphonia::core::audio::Position;

    #[test]
    fn positions_map_to_ffmpeg_bits_in_its_order() {
        let surround = Position::FRONT_LEFT
            | Position::FRONT_RIGHT
            | Position::FRONT_CENTER
            | Position::LFE1
            | Position::SIDE_LEFT
            | Position::SIDE_RIGHT;
        assert_eq!(
            ffmpeg_layout(&Channels::Positioned(surround), 6),
            (Some(0x60F), vec![0, 1, 2, 3, 4, 5])
        );
        // AAC 7.1: the wide pair is ffmpeg's left and right of centre,
        // which come before the sides.
        let wide = surround | Position::FRONT_LEFT_WIDE | Position::FRONT_RIGHT_WIDE;
        assert_eq!(
            ffmpeg_layout(&Channels::Positioned(wide), 8),
            (Some(0x6CF), vec![0, 1, 2, 3, 6, 7, 4, 5])
        );
        // LFE 2 is ffmpeg's bit 35, past every bit of the first 18.
        let lfe2 = Position::FRONT_LEFT | Position::FRONT_RIGHT | Position::LFE2;
        assert_eq!(
            ffmpeg_layout(&Channels::Positioned(lfe2), 3),
            (Some(0x8_0000_0003), vec![0, 1, 2])
        );
        let mono = Channels::Positioned(Position::FRONT_LEFT);
        assert_eq!(ffmpeg_layout(&mono, 1), (Some(0x4), vec![0]));
        assert_eq!(ffmpeg_layout(&Channels::Discrete(2), 2), (None, vec![0, 1]));
        // Front left twice over: two planes on one ffmpeg bit.
        let shared = Position::FRONT_LEFT_CENTER | Position::FRONT_LEFT_WIDE;
        assert_eq!(
            ffmpeg_layout(&Channels::Positioned(shared), 2),
            (None, vec![0, 1])
        );
    }
}
