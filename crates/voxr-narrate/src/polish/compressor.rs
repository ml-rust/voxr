//! A gentle feed-forward vocal compressor: it evens a voice's loud and soft
//! syllables with a soft-knee gain curve (Giannoulis, Massberg and Reiss)
//! smoothed per sample.
//!
//! Every step runs in `f64` and each output sample is rounded to `f32`, so
//! the result matches the JavaScript bit for bit.

use crate::js_math::{js_exp, js_log10, js_pow};

/// Added to each sample's magnitude before its level is taken, so silence
/// has a finite level.
const LEVEL_FLOOR: f64 = 1e-9;

/// The compressor's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorOptions {
    /// Level where compression starts, in dB: -22.
    pub threshold: f64,
    /// Level change in over level change out above the knee: 1.5.
    pub ratio: f64,
    /// Width of the soft knee around the threshold, in dB: 30.
    pub knee: f64,
    /// Time for more gain reduction to take hold, in seconds: 0.002.
    pub attack: f64,
    /// Time for gain reduction to ease off, in seconds: 0.45.
    pub release: f64,
    /// Gain added after compression, in dB: 3.6.
    pub makeup: f64,
}

impl Default for CompressorOptions {
    fn default() -> Self {
        Self {
            threshold: -22.0,
            ratio: 1.5,
            knee: 30.0,
            attack: 0.002,
            release: 0.45,
            makeup: 3.6,
        }
    }
}

/// Compresses `x`, sampled at `rate` Hz.
///
/// Each sample's level is `20 * log10(|x| + 1e-9)` dB, and `over` is that
/// level minus the threshold. Above the knee (`2 * over > knee`) the level
/// out is `threshold + over / ratio`. Inside it (`2 * |over| <= knee`) it is
/// `level + (1 / ratio - 1) * (over + knee / 2)^2 / (2 * knee)`. Below it
/// the level passes. The cut, level out minus level, is smoothed from 0 at
/// the first sample: toward a deeper cut with the coefficient
/// `exp(-1 / (attack * rate))`, toward a shallower one with
/// `exp(-1 / (release * rate))`. Each sample is scaled by the smoothed cut
/// plus the make-up gain.
pub fn compress_voice(x: &[f32], rate: u32, options: CompressorOptions) -> Vec<f32> {
    let CompressorOptions {
        threshold,
        ratio,
        knee,
        attack,
        release,
        makeup,
    } = options;
    let rate = f64::from(rate);
    let up = js_exp(-1.0 / (attack * rate));
    let down = js_exp(-1.0 / (release * rate));
    let mut smooth = 0.0f64;
    x.iter()
        .map(|&s| {
            let v = f64::from(s);
            let level = 20.0 * js_log10(v.abs() + LEVEL_FLOOR);
            let over = level - threshold;
            let mut out = level;
            if 2.0 * over > knee {
                out = threshold + over / ratio;
            } else if 2.0 * over.abs() <= knee {
                // `(over + knee / 2) ** 2`: the correctly rounded square.
                let d = over + knee / 2.0;
                out = level + ((1.0 / ratio - 1.0) * (d * d)) / (2.0 * knee);
            }
            let cut = out - level;
            smooth = if cut < smooth {
                up * smooth + (1.0 - up) * cut
            } else {
                down * smooth + (1.0 - down) * cut
            };
            (v * js_pow(10.0, (smooth + makeup) / 20.0)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_stays_silent() {
        let y = compress_voice(&[0.0; 64], 48000, CompressorOptions::default());
        assert_eq!(y, vec![0.0; 64]);
        assert!(compress_voice(&[], 48000, CompressorOptions::default()).is_empty());
    }

    #[test]
    fn a_quiet_voice_gets_only_the_make_up_gain() {
        // 0.001 is -60 dB, under the knee's -37 dB floor: no cut at all.
        let y = compress_voice(&[0.001; 8], 48000, CompressorOptions::default());
        let gained = (f64::from(0.001f32) * js_pow(10.0, 3.6 / 20.0)) as f32;
        assert!(y.iter().all(|&v| v == gained));
    }

    #[test]
    fn a_loud_voice_is_turned_down_at_the_attack_rate() {
        let o = CompressorOptions::default();
        let y = compress_voice(&[0.9; 4800], 48000, o);
        // The cut deepens sample by sample toward its full depth.
        assert!(y.windows(2).all(|w| w[1] <= w[0]));
        let level = 20.0 * js_log10(f64::from(0.9f32) + 1e-9);
        let full = (o.threshold + (level - o.threshold) / o.ratio) - level;
        let settled = (f64::from(0.9f32) * js_pow(10.0, (full + o.makeup) / 20.0)) as f32;
        assert!((y[4799] - settled).abs() < 1e-6);
    }
}
