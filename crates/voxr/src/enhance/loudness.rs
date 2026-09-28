//! Integrated loudness to ITU-R BS.1770-4 (EBU R128), and normalization to it.
//!
//! Peak and RMS both mislead on speech. Peak is set by one plosive; RMS counts
//! silence between words, so a slow talker measures quieter than a fast one
//! saying the same thing at the same effort. LUFS applies a perceptual
//! weighting and GATES OUT the quiet parts, which is why broadcast and every
//! TTS corpus pipeline specifies it.
//!
//! Mono only, which is all this pipeline produces. Multi-channel BS.1770 sums
//! weighted channel powers with per-channel weights; the single-channel weight
//! is 1.0, so mono is the degenerate case and nothing here would need to change
//! beyond the sum.

use biquad::{Biquad, Coefficients, DirectForm2Transposed};

use crate::error::{Error, Result};

/// BS.1770 block length. The standard fixes this at 400 ms.
const BLOCK_MS: f64 = 400.0;
/// Blocks overlap by 75%, so a new block starts every 100 ms.
const BLOCK_OVERLAP: f64 = 0.75;
/// Absolute gate: blocks quieter than this never count.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;
/// Relative gate: after the absolute gate, drop blocks more than 10 LU below
/// the mean of what survived.
const RELATIVE_GATE_LU: f64 = -10.0;
/// The offset in the BS.1770 loudness formula.
const LOUDNESS_OFFSET: f64 = -0.691;

/// The two K-weighting stages, as BS.1770 defines them for 48 kHz.
///
/// The standard publishes literal coefficients rather than a parametric shape.
/// They are reproduced verbatim at 48 kHz. Other rates use [`k_weighting_at`],
/// the bilinear design those literals come from.
///
/// Returns [`Error::InvalidArgument`] when the 1681.97 Hz shelf corner is at
/// or above the Nyquist limit of `rate`.
fn k_weighting(rate: f64) -> Result<(Coefficients<f64>, Coefficients<f64>)> {
    if (rate - 48_000.0).abs() < 1.0 {
        // Stage 1: high shelf, +4 dB above ~1.68 kHz (head/torso model).
        let stage1 = Coefficients {
            b0: 1.53512485958697,
            b1: -2.69169618940638,
            b2: 1.19839281085285,
            a1: -1.69065929318241,
            a2: 0.73248077421585,
        };
        // Stage 2: high-pass at ~38 Hz (RLB curve).
        let stage2 = Coefficients {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: -1.99004745483398,
            a2: 0.99007225036621,
        };
        return Ok((stage1, stage2));
    }
    k_weighting_at(rate)
}

/// Shelf corner of K-weighting stage 1, in Hz.
const SHELF_HZ: f64 = 1681.974450955533;
/// Shelf gain of stage 1, in dB.
const SHELF_GAIN_DB: f64 = 3.999843853973347;
/// Shelf Q of stage 1.
const SHELF_Q: f64 = 0.7071752369554196;
/// Exponent placing the shelf's mid-band gain `Vb = Vh^x`.
const SHELF_MID_EXP: f64 = 0.4996667741545416;
/// Corner of K-weighting stage 2 (the RLB high-pass), in Hz.
const HIGHPASS_HZ: f64 = 38.13547087602444;
/// Q of stage 2.
const HIGHPASS_Q: f64 = 0.5003270373238773;

/// K-weighting designed for `rate` by the bilinear transform, with the shelf
/// and high-pass parameters fitted to the BS.1770 48 kHz literals (the
/// libebur128 design).
///
/// The shelf is NOT the RBJ cookbook high shelf: its mid-band gain `Vb` sits at
/// `Vh^0.4997`, which the cookbook's `sqrt(A)` form cannot express at this Q.
fn k_weighting_at(rate: f64) -> Result<(Coefficients<f64>, Coefficients<f64>)> {
    let nyquist = rate / 2.0;
    if SHELF_HZ >= nyquist {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: format!(
                "K-weighting needs its {SHELF_HZ:.2} Hz shelf corner below the {nyquist} Hz \
                 Nyquist limit of {rate} Hz audio; resample to at least 8 kHz before \
                 measuring loudness"
            ),
        });
    }

    let k = (std::f64::consts::PI * SHELF_HZ / rate).tan();
    let vh = 10f64.powf(SHELF_GAIN_DB / 20.0);
    let vb = vh.powf(SHELF_MID_EXP);
    let a0 = 1.0 + k / SHELF_Q + k * k;
    let shelf = Coefficients {
        b0: (vh + vb * k / SHELF_Q + k * k) / a0,
        b1: 2.0 * (k * k - vh) / a0,
        b2: (vh - vb * k / SHELF_Q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / SHELF_Q + k * k) / a0,
    };

    let k = (std::f64::consts::PI * HIGHPASS_HZ / rate).tan();
    let a0 = 1.0 + k / HIGHPASS_Q + k * k;
    let highpass = Coefficients {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / HIGHPASS_Q + k * k) / a0,
    };
    Ok((shelf, highpass))
}

/// Mean square of each gating block after K-weighting.
fn block_powers(samples: &[f32], rate: u32) -> Result<Vec<f64>> {
    let rate_f = rate as f64;
    let (c1, c2) = k_weighting(rate_f)?;
    let mut s1 = DirectForm2Transposed::<f64>::new(c1);
    let mut s2 = DirectForm2Transposed::<f64>::new(c2);

    let weighted: Vec<f32> = samples
        .iter()
        .map(|&x| s2.run(s1.run(x as f64)) as f32)
        .collect();

    let block_len = ((BLOCK_MS / 1000.0) * rate_f).round() as usize;
    let step = ((block_len as f64) * (1.0 - BLOCK_OVERLAP))
        .round()
        .max(1.0) as usize;
    if block_len == 0 || weighted.len() < block_len {
        return Ok(Vec::new());
    }

    let mut powers = Vec::new();
    let mut start = 0usize;
    while start + block_len <= weighted.len() {
        let sum_sq: f64 = weighted[start..start + block_len]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum();
        powers.push(sum_sq / block_len as f64);
        start += step;
    }
    Ok(powers)
}

/// Loudness in LUFS from a mean-square power.
fn power_to_lufs(power: f64) -> f64 {
    if power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    LOUDNESS_OFFSET + 10.0 * power.log10()
}

/// Integrated loudness in LUFS, gated per BS.1770-4.
///
/// Returns [`f64::NEG_INFINITY`] for silence, and for input shorter than one
/// 400 ms block — there is nothing to integrate, and reporting a number there
/// would invite normalizing against a measurement that does not exist.
///
/// Returns [`Error::InvalidArgument`] when `rate` is 0, or too low for the
/// K-weighting shelf (3364 Hz and below).
pub fn integrated_lufs(samples: &[f32], rate: u32) -> Result<f64> {
    if rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: "sample rate is 0".to_string(),
        });
    }
    let powers = block_powers(samples, rate)?;
    if powers.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    // Absolute gate.
    let kept: Vec<f64> = powers
        .iter()
        .copied()
        .filter(|&p| power_to_lufs(p) > ABSOLUTE_GATE_LUFS)
        .collect();
    if kept.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    // Relative gate, referenced to the mean of what the absolute gate kept.
    let mean = kept.iter().sum::<f64>() / kept.len() as f64;
    let threshold = power_to_lufs(mean) + RELATIVE_GATE_LU;
    let final_kept: Vec<f64> = kept
        .into_iter()
        .filter(|&p| power_to_lufs(p) > threshold)
        .collect();
    if final_kept.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    let final_mean = final_kept.iter().sum::<f64>() / final_kept.len() as f64;
    Ok(power_to_lufs(final_mean))
}

/// Highest absolute sample, as dBFS. `-inf` for digital silence.
pub fn peak_dbfs(samples: &[f32]) -> f64 {
    let peak = samples.iter().fold(0.0f64, |a, &b| a.max((b as f64).abs()));
    if peak > 0.0 {
        20.0 * peak.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Scale `samples` to `target_lufs`, then pull back if that would exceed
/// `peak_ceiling_dbfs`.
///
/// The ceiling wins. Overshooting it to hit a loudness target is how a
/// normalizer produces clipping, and a clipped reference is unusable for voice
/// cloning at any loudness.
///
/// Returns the samples unchanged when the input is silent or too short to
/// measure, rather than applying an arbitrary gain to something unmeasured.
pub fn normalize_to_lufs(
    samples: &[f32],
    rate: u32,
    target_lufs: f64,
    peak_ceiling_dbfs: f64,
) -> Result<Vec<f32>> {
    let current = integrated_lufs(samples, rate)?;
    if !current.is_finite() {
        return Ok(samples.to_vec());
    }

    let mut gain = 10f64.powf((target_lufs - current) / 20.0);

    let peak = peak_dbfs(samples);
    if peak.is_finite() {
        let ceiling_gain = 10f64.powf((peak_ceiling_dbfs - peak) / 20.0);
        if ceiling_gain < gain {
            gain = ceiling_gain;
        }
    }

    Ok(samples.iter().map(|&s| (s as f64 * gain) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// A 1 kHz sine at the given peak amplitude, `secs` long.
    fn tone(amp: f64, secs: f64) -> Vec<f32> {
        let n = (RATE as f64 * secs) as usize;
        (0..n)
            .map(|i| (amp * (std::f64::consts::TAU * 1000.0 * i as f64 / RATE as f64).sin()) as f32)
            .collect()
    }

    #[test]
    fn silence_has_no_loudness_rather_than_a_number() {
        let lufs = integrated_lufs(&vec![0.0f32; RATE as usize], RATE).unwrap();
        assert!(lufs.is_infinite() && lufs < 0.0, "got {lufs}");
    }

    #[test]
    fn input_shorter_than_one_block_is_unmeasurable() {
        // A 400 ms block is the smallest thing BS.1770 integrates. Returning a
        // number for 100 ms would invite normalizing against nothing.
        let lufs = integrated_lufs(&tone(0.5, 0.1), RATE).unwrap();
        assert!(
            lufs.is_infinite(),
            "100 ms should not produce a reading, got {lufs}"
        );
        assert!(integrated_lufs(&tone(0.5, 1.0), RATE).unwrap().is_finite());
    }

    #[test]
    fn doubling_amplitude_raises_loudness_by_six_db() {
        let quiet = integrated_lufs(&tone(0.25, 3.0), RATE).unwrap();
        let loud = integrated_lufs(&tone(0.5, 3.0), RATE).unwrap();
        assert!(
            (loud - quiet - 6.02).abs() < 0.05,
            "{quiet:.2} -> {loud:.2} should be +6.02 LU"
        );
    }

    #[test]
    fn normalizing_hits_the_target() {
        let out = normalize_to_lufs(&tone(0.1, 3.0), RATE, -23.0, -1.0).unwrap();
        let got = integrated_lufs(&out, RATE).unwrap();
        assert!((got - -23.0).abs() < 0.1, "got {got:.2} LUFS, wanted -23");
    }

    #[test]
    fn the_peak_ceiling_overrides_the_loudness_target() {
        // A very quiet source needs a large gain to reach -16 LUFS; the ceiling
        // must stop it short rather than clip. Clipping a reference destroys it for
        // voice cloning no matter how correct the loudness is.
        let out = normalize_to_lufs(&tone(0.001, 3.0), RATE, -16.0, -1.0).unwrap();
        let peak = peak_dbfs(&out);
        assert!(
            peak <= -1.0 + 1e-6,
            "peak {peak:.2} dBFS breached the ceiling"
        );
        let lufs = integrated_lufs(&out, RATE).unwrap();
        assert!(lufs < -16.0, "must fall short of target, got {lufs:.2}");
    }

    #[test]
    fn the_gate_ignores_silence_between_words() {
        // The point of gating: a signal with long gaps must measure the same as the
        // speech alone. RMS over the whole thing would read far quieter.
        let speech = tone(0.5, 2.0);
        let mut gapped = speech.clone();
        gapped.extend(vec![0.0f32; RATE as usize * 4]);
        gapped.extend(tone(0.5, 2.0));

        let solid = integrated_lufs(&speech, RATE).unwrap();
        let with_gaps = integrated_lufs(&gapped, RATE).unwrap();
        assert!(
            (solid - with_gaps).abs() < 0.5,
            "gated loudness {with_gaps:.2} should match ungapped {solid:.2}"
        );

        // Ungated RMS does NOT have this property, which is why it is not used.
        let rms_db = |s: &[f32]| {
            20.0 * (s.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / s.len() as f64)
                .sqrt()
                .log10()
        };
        assert!(
            rms_db(&speech) - rms_db(&gapped) > 2.0,
            "RMS should be badly skewed by the gaps"
        );
    }

    /// BS.1770-4 calibrates the meter so a 0 dBFS 1 kHz sine in one channel
    /// reads -3.01 LUFS: the -0.691 offset cancels K-weighting's gain at 1 kHz,
    /// leaving the sine's mean square of 1/2 (-3.01 dB). A sine at -20 dBFS
    /// (peak 0.1) has a mean square 20 dB lower, so it reads -23.01 LUFS.
    /// 997 Hz is the test-tone convention: prime, so no sample pattern repeats
    /// at 48 kHz, and within 3 Hz of 1 kHz, where K-weighting differs by far
    /// less than the tolerance.
    #[test]
    fn a_minus_20_dbfs_997_hz_sine_reads_minus_23_01_lufs() {
        let amp = 10f64.powf(-20.0 / 20.0);
        let n = RATE as usize * 10;
        let sine: Vec<f32> = (0..n)
            .map(|i| (amp * (std::f64::consts::TAU * 997.0 * i as f64 / RATE as f64).sin()) as f32)
            .collect();
        let lufs = integrated_lufs(&sine, RATE).unwrap();
        assert!((lufs - -23.01).abs() < 0.1, "got {lufs:.3} LUFS");
    }

    /// The bilinear design must reproduce the literal 48 kHz coefficients, or
    /// every other rate measures a different filter.
    #[test]
    fn the_k_weighting_design_reproduces_the_48k_literals() {
        let (lit1, lit2) = k_weighting(48_000.0).unwrap();
        let (d1, d2) = k_weighting_at(48_000.0).unwrap();
        for (a, b) in [(lit1, d1), (lit2, d2)] {
            for (x, y) in [
                (a.b0, b.b0),
                (a.b1, b.b1),
                (a.b2, b.b2),
                (a.a1, b.a1),
                (a.a2, b.a2),
            ] {
                assert!((x - y).abs() < 1e-8, "{x} vs {y}");
            }
        }
    }

    /// The rebuilt K-weighting at 44.1 kHz must agree with the literal 48 kHz
    /// coefficients on the same tone.
    #[test]
    fn the_rebuilt_k_weighting_matches_the_literal_one() {
        let tone_at = |rate: u32| -> Vec<f32> {
            (0..rate as usize * 5)
                .map(|i| {
                    (0.1 * (std::f64::consts::TAU * 997.0 * i as f64 / rate as f64).sin()) as f32
                })
                .collect()
        };
        let at_48k = integrated_lufs(&tone_at(48_000), 48_000).unwrap();
        let at_44k = integrated_lufs(&tone_at(44_100), 44_100).unwrap();
        assert!((at_48k - at_44k).abs() < 0.05, "{at_48k:.3} vs {at_44k:.3}");
    }

    #[test]
    fn a_zero_sample_rate_is_rejected() {
        assert!(integrated_lufs(&tone(0.5, 1.0), 0).is_err());
        // The 1681.97 Hz shelf has no room below a 1500 Hz Nyquist limit.
        assert!(integrated_lufs(&tone(0.5, 1.0), 3_000).is_err());
    }
}
