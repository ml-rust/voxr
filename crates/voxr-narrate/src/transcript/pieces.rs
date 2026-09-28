//! Cut points for recognition: a recognizer drops or invents text over long
//! audio, so each piece of at most 20 s, cut in a silence, is read alone.

use super::envelope::{Envelope, HOP};

/// The most frames in one piece: 20 s.
const MAX_FRAMES: usize = 2000;
/// A cut is searched for in this many frames at the end of a piece: 8 s.
const SEARCH_FRAMES: usize = 800;
/// A silence this many frames long or longer can hold a cut: 80 ms.
const MIN_QUIET: usize = 8;

/// Pieces of at most 20 s, as `[start, end)` sample ranges of audio
/// `length` samples long.
///
/// Each piece but the last ends in the middle of the longest silence (80 ms
/// or more) in its last 8 s, the first on a tie. With no such silence, a
/// piece is cut at 20 s. The last piece runs to `length`, past the last
/// whole frame. Audio shorter than one frame has no pieces.
pub fn pieces(env: &Envelope, length: usize) -> Vec<(usize, usize)> {
    let frames = env.env.len();
    let mut out = Vec::new();
    let mut from = 0;
    while from < frames {
        if frames - from <= MAX_FRAMES {
            out.push((from * HOP, length));
            break;
        }
        // The longest quiet run so far and the frame in its middle.
        let mut best: Option<(usize, usize)> = None;
        let mut run = 0;
        for k in from + MAX_FRAMES - SEARCH_FRAMES..from + MAX_FRAMES {
            if env.env[k] <= env.floor {
                run += 1;
            } else {
                run = 0;
            }
            if run >= MIN_QUIET && best.is_none_or(|(longest, _)| run > longest) {
                best = Some((run, k - run / 2));
            }
        }
        let cut = best.map_or(from + MAX_FRAMES, |(_, at)| at);
        out.push((from * HOP, cut * HOP));
        from = cut;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(levels: Vec<f64>) -> Envelope {
        Envelope {
            env: levels,
            floor: 0.01,
        }
    }

    #[test]
    fn short_audio_is_one_piece() {
        let e = env(vec![1.0; 2000]);
        assert_eq!(pieces(&e, 2000 * HOP + 7), [(0, 2000 * HOP + 7)]);
        assert!(pieces(&env(Vec::new()), 100).is_empty());
    }

    #[test]
    fn no_silence_cuts_at_20_s() {
        let e = env(vec![1.0; 4500]);
        assert_eq!(
            pieces(&e, 4500 * HOP),
            [
                (0, 2000 * HOP),
                (2000 * HOP, 4000 * HOP),
                (4000 * HOP, 4500 * HOP)
            ]
        );
    }

    #[test]
    fn cut_in_the_longest_silence() {
        let mut levels = vec![1.0; 3000];
        levels[1300..1310].fill(0.0);
        levels[1500..1520].fill(0.0);
        levels[1700..1720].fill(0.0);
        // Too short to cut in.
        levels[1900..1907].fill(0.0);
        let cuts = pieces(&env(levels), 3000 * HOP);
        assert_eq!(cuts[0], (0, 1509 * HOP));
        assert_eq!(cuts[1], (1509 * HOP, 3000 * HOP));
    }
}
