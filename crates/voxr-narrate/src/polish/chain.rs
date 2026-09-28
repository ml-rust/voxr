//! The voice layer's own processing of a take, sentence by sentence, for
//! any engine's output, and the 16-bit samples it is written as.

use super::compressor::{CompressorOptions, compress_voice};
use super::de_ess::de_ess;
use super::level::{Span, even_levels};
use super::normalize::normalize_peak;
use crate::js::js_round;

/// The peak a polished take is normalized to, in dB.
pub const POLISH_PEAK_DB: f64 = -1.0;
/// Full scale of a 16-bit sample.
const PCM16_SCALE: f64 = 32767.0;

/// What [`polish`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolishReport {
    /// Sentences whose level changed.
    pub leveled: usize,
    /// The largest level change, in dB, as a magnitude.
    pub max_gain_db: f64,
    /// The share of de-essing frames cut by more than 0.5 dB.
    pub ess_share: f64,
    /// The deepest de-essing cut, in dB: 0 or below.
    pub ess_deepest_db: f64,
}

/// Polishes a take in place. It never time-stretches, so `spans` stay
/// where they are.
///
/// In order:
/// - [`even_levels`]: each span's speech toward the spans' median level, by
///   at most 6 dB, with 40 ms ramps.
/// - [`compress_voice`] with the default settings: evens loud and soft
///   syllables. It runs before de-essing, since it lifts the soft parts,
///   an "s" among them.
/// - [`de_ess`].
/// - [`normalize_peak`] to [`POLISH_PEAK_DB`]: the take centered, its peak
///   at -1 dB.
pub fn polish(samples: &mut [f32], spans: &[Span], rate: u32) -> PolishReport {
    let level = even_levels(samples, spans, rate);
    let compressed = compress_voice(samples, rate, CompressorOptions::default());
    let (de_essed, ess) = de_ess(&compressed, rate);
    let out = normalize_peak(&de_essed, POLISH_PEAK_DB);
    samples.copy_from_slice(&out);
    PolishReport {
        leveled: level.leveled,
        max_gain_db: level.max_gain_db,
        ess_share: ess.share,
        ess_deepest_db: ess.deepest,
    }
}

/// `x` as 16-bit samples, as the JavaScript writes a polished take:
/// `clamp(Math.round(v * 32767), -32768, 32767)`, NaN as 0.
///
/// The product is exact in `f64`, and an exact half rounds up, toward
/// positive infinity. So -0.5 gives -16383, a value under -1 can give
/// -32768, and 1.5259255e-5 (0.4999999995 of a step) gives 0.
/// `voxr::encode_wav_pcm16` clamps to ±1 first, multiplies in `f32` and
/// rounds halves away from zero, so it gives -16384, -32767 and 1 on those.
pub fn pcm16_like_js(x: &[f32]) -> Vec<i16> {
    x.iter()
        .map(|&v| {
            let r = js_round(f64::from(v) * PCM16_SCALE);
            if r.is_nan() {
                0
            } else {
                r.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_rounds_and_clamps_as_javascript() {
        let x = [
            1.0,
            -1.0,
            1.5,
            -1.5,
            0.5,
            -0.5,
            f32::from_bits(0x3780_0100),
            -f32::from_bits(0x3780_0100),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -0.0,
        ];
        assert_eq!(
            pcm16_like_js(&x),
            [
                32767, -32767, 32767, -32768, 16384, -16383, 0, 0, 32767, -32768, 0, 0
            ]
        );
    }

    #[test]
    fn polish_keeps_the_length_and_peaks_at_minus_one_db() {
        let mut x: Vec<f32> = (0..48000)
            .map(|i| (0.2 * (i as f64 * 0.05).sin() + 0.01) as f32)
            .collect();
        let spans = [
            Span {
                start: 0.0,
                end: 0.5,
            },
            Span {
                start: 0.5,
                end: 1.0,
            },
        ];
        let report = polish(&mut x, &spans, 48000);
        assert_eq!(x.len(), 48000);
        assert_eq!(report.leveled, 0);
        let peak = x.iter().fold(0.0f32, |p, v| p.max(v.abs()));
        assert!((f64::from(peak) - 10f64.powf(-1.0 / 20.0)).abs() < 1e-6);
    }
}
