//! Loudness of 16 kHz narration in 10 ms frames, and the silence floor
//! under it.

use crate::js::js_max;
use crate::js_math::js_pow;

/// Samples per second of the audio a transcript is read from.
pub const TRANSCRIPT_RATE: u32 = 16000;
/// Samples in one loudness frame: 10 ms.
pub const HOP: usize = 160;
/// The silence floor sits this many dB under the loudest frame.
const FLOOR_DB: f64 = -40.0;
/// The loudest frame of audio with no sound.
const MIN_PEAK: f64 = 1e-9;

/// [`TRANSCRIPT_RATE`] as a number of seconds' samples.
pub(crate) const RATE: f64 = TRANSCRIPT_RATE as f64;

/// The RMS level of each 10 ms frame, and the silence floor.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    /// The RMS level of each whole frame of [`HOP`] samples.
    pub env: Vec<f64>,
    /// A frame at or under this level is silent: 40 dB under the loudest
    /// frame.
    pub floor: f64,
}

impl Envelope {
    /// Frames in the envelope.
    pub(crate) fn frames(&self) -> i64 {
        self.env.len() as i64
    }

    /// Frame `k` is over the floor. A frame outside the envelope is not.
    pub(crate) fn loud(&self, k: i64) -> bool {
        usize::try_from(k)
            .ok()
            .and_then(|k| self.env.get(k))
            .is_some_and(|&v| v > self.floor)
    }
}

/// The frame a time in seconds falls in: `floor(t * 16000 / 160)`.
pub(crate) fn frame_at(t: f64) -> i64 {
    ((t * RATE) / HOP as f64).floor() as i64
}

/// `ceil(t * 16000 / 160)`: the first frame starting at or after `t`.
pub(crate) fn frame_ceil(t: f64) -> i64 {
    ((t * RATE) / HOP as f64).ceil() as i64
}

/// [`frame_at`], never before frame 0.
pub(crate) fn frame_from_zero(t: f64) -> i64 {
    frame_at(t).max(0)
}

/// The time frame `k` starts at, in seconds.
pub(crate) fn frame_seconds(k: i64) -> f64 {
    (k * HOP as i64) as f64 / RATE
}

/// Loudness of 16 kHz audio: the RMS level of each whole 10 ms frame, and
/// the silence floor 40 dB under the loudest frame. A trailing part frame
/// is not read. Sums run in `f64` in sample order.
pub fn envelope(pcm: &[f32]) -> Envelope {
    let mut env = Vec::with_capacity(pcm.len() / HOP);
    for frame in pcm.as_chunks::<HOP>().0 {
        let mut e = 0.0f64;
        for &s in frame {
            let s = f64::from(s);
            e += s * s;
        }
        env.push((e / HOP as f64).sqrt());
    }
    let mut peak = MIN_PEAK;
    for &v in &env {
        peak = js_max(peak, v);
    }
    Envelope {
        env,
        floor: peak * js_pow(10.0, FLOOR_DB / 20.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_whole_and_the_floor_follows_the_peak() {
        let mut pcm = vec![0.0f32; 2 * HOP + 10];
        pcm[..HOP].fill(0.5);
        let e = envelope(&pcm);
        assert_eq!(e.env, [0.5, 0.0]);
        assert_eq!(e.floor, 0.005);
        assert!(e.loud(0));
        assert!(!e.loud(1));
        assert!(!e.loud(-1));
        assert!(!e.loud(2));
    }

    #[test]
    fn silence_keeps_a_tiny_floor() {
        let e = envelope(&[0.0; HOP]);
        assert_eq!(e.floor, MIN_PEAK * 0.01);
        assert!(envelope(&[0.1; 10]).env.is_empty());
    }

    #[test]
    fn frame_times() {
        assert_eq!(frame_at(1.0), 100);
        assert_eq!(frame_at(-0.005), -1);
        assert_eq!(frame_from_zero(-0.005), 0);
        assert_eq!(frame_seconds(150), 1.5);
    }
}
