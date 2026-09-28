//! The full raw-recording-to-reference chain.

use biquad::{Biquad, Coefficients, DirectForm2Transposed, Q_BUTTERWORTH_F64, Type};

use super::denoise::{DenoiseOptions, denoise, denoise_with_profile, noise_floor_dbfs};
use super::limiter::{LimiterOptions, LimiterReport, limit};
use super::loudness::{integrated_lufs, peak_dbfs};
use crate::error::{Error, Result};

/// Controls for [`enhance`].
#[derive(Debug, Clone, Copy)]
pub struct EnhanceOptions {
    /// Corner of the rumble filter, in Hz. Below roughly 60 Hz a speech
    /// recording holds only desk thumps, HVAC and handling noise — no voice.
    /// Removing it before anything else stops that energy from dominating the
    /// loudness measurement and eating headroom.
    pub highpass_hz: f64,
    /// Spectral gate settings. `None` skips denoising.
    pub denoise: Option<DenoiseOptions>,
    /// Low-shelf gain in dB, and its corner in Hz — the "more bass" control.
    /// Applied AFTER denoising: boosting first would amplify the rumble the
    /// gate then has to fight.
    pub bass_boost_db: f64,
    pub bass_corner_hz: f64,
    /// High-shelf gain in dB and corner in Hz. A small lift here restores the
    /// air a spectral gate always takes off the top. Skipped when the corner
    /// is at or above the Nyquist limit: the audio has no band there to lift.
    pub presence_db: f64,
    pub presence_corner_hz: f64,
    /// Target integrated loudness.
    pub target_lufs: f64,
    /// Sample peak that must not be exceeded.
    pub peak_ceiling_dbfs: f64,
    /// Limiter window. `None` skips limiting, in which case the loudness
    /// target yields to the ceiling and a normal speech take lands roughly
    /// 6 dB short of it.
    pub limiter_window_ms: Option<f64>,
    /// Most *sustained* gain reduction the limiter is allowed to apply, in dB
    /// — [`LimiterReport::sustained_reduction_db`], not the deepest single
    /// sample. Past roughly 6 dB a voice audibly pumps, and a pumping
    /// reference teaches the pumping to the model.
    ///
    /// Capping the deepest sample instead would let one plosive hold the whole
    /// take quiet, since a lone transient can need 15 dB while nothing else
    /// needs any.
    ///
    /// When the target needs more than this, the chain stops at whatever
    /// loudness that much limiting reached and reports `reached_target: false`.
    pub max_limiting_db: f64,
}

impl Default for EnhanceOptions {
    fn default() -> Self {
        Self {
            highpass_hz: 70.0,
            denoise: Some(DenoiseOptions::default()),
            bass_boost_db: 3.0,
            bass_corner_hz: 140.0,
            presence_db: 1.5,
            presence_corner_hz: 7000.0,
            target_lufs: -18.0,
            peak_ceiling_dbfs: -1.0,
            limiter_window_ms: Some(5.0),
            max_limiting_db: 6.0,
        }
    }
}

/// What the chain measured, before and after.
///
/// Kept because "it sounds better" is not a check. A take whose noise floor
/// barely moved was never noisy, and one whose floor dropped 20 dB while
/// loudness stayed put had its voice gated along with the hiss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnhanceReport {
    pub input_lufs: f64,
    pub output_lufs: f64,
    pub input_peak_dbfs: f64,
    pub output_peak_dbfs: f64,
    pub input_noise_floor_dbfs: f64,
    pub output_noise_floor_dbfs: f64,
    /// Gain the loudness stage applied, in dB. `-inf` when the input was
    /// silent and nothing was applied.
    pub applied_gain_db: f64,
    /// Sustained gain reduction the limiter applied, in dB — the 99th
    /// percentile, which is what the cap is measured against. `0.0` when the
    /// limiter never engaged or was disabled.
    pub limiter_reduction_db: f64,
    /// Deepest single-sample gain reduction the limiter applied, in dB.
    pub limiter_peak_reduction_db: f64,
    /// Whether the chain reached [`EnhanceOptions::target_lufs`], within
    /// 0.5 LU. False means the ceiling and the limiting cap stopped it short;
    /// [`Self::output_lufs`] is then what it actually reached.
    pub reached_target: bool,
}

/// Bring a raw recording to reference quality.
///
/// Stage order is fixed:
///
/// | Stage | Why here |
/// |-------|----------|
/// | High-pass | Rumble skews every later measurement |
/// | Denoise | Operates on a signal whose floor is already rumble-free |
/// | Bass shelf | Restores weight the high-pass and gate removed |
/// | Presence shelf | Restores air the gate removed |
/// | Loudness + ceiling | Last: every earlier stage changes the level |
pub fn enhance(
    samples: &[f32],
    rate: u32,
    opts: EnhanceOptions,
) -> Result<(Vec<f32>, EnhanceReport)> {
    enhance_with_noise_profile(samples, None, rate, opts)
}

/// [`enhance`], with the noise profile taken from a supplied silent clip.
///
/// Prefer this whenever a silent stretch of the same chain is available. The
/// automatic path in [`denoise`] has to find the pauses inside the take, and a
/// take recorded without any pause gets no denoising at all.
///
/// `noise_clip` is high-passed alongside the signal, so the profile it yields
/// describes the same filtered floor the mask is compared against.
pub fn enhance_with_noise_profile(
    samples: &[f32],
    noise_clip: Option<&[f32]>,
    rate: u32,
    opts: EnhanceOptions,
) -> Result<(Vec<f32>, EnhanceReport)> {
    if rate == 0 {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: "sample rate is 0".to_string(),
        });
    }
    if samples.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "no samples".to_string(),
        });
    }

    let input_lufs = integrated_lufs(samples, rate)?;
    let input_peak_dbfs = peak_dbfs(samples);
    let input_noise_floor_dbfs = noise_floor_dbfs(samples, rate);

    let rate_f = rate as f64;
    let mut work = samples.to_vec();

    let mut noise = noise_clip.map(|c| c.to_vec());
    if opts.highpass_hz > 0.0 {
        let hp = design(Type::HighPass, rate_f, opts.highpass_hz, "highpass_hz")?;
        filter(hp, &mut work);
        // A fresh section per buffer: carrying filter state across two
        // independent buffers puts a transient at the start of the second.
        if let Some(n) = noise.as_mut() {
            filter(hp, n);
        }
    }

    if let Some(d) = opts.denoise {
        work = match noise.as_deref() {
            Some(n) => denoise_with_profile(&work, n, d)?,
            None => denoise(&work, d)?,
        };
    }

    if opts.bass_boost_db != 0.0 {
        let shelf = design(
            Type::LowShelf(opts.bass_boost_db),
            rate_f,
            opts.bass_corner_hz,
            "bass_corner_hz",
        )?;
        filter(shelf, &mut work);
    }

    if opts.presence_db != 0.0 && opts.presence_corner_hz < rate_f / 2.0 {
        let shelf = design(
            Type::HighShelf(opts.presence_db),
            rate_f,
            opts.presence_corner_hz,
            "presence_corner_hz",
        )?;
        filter(shelf, &mut work);
    }

    let before_gain = integrated_lufs(&work, rate)?;
    let (levelled, limiting) = reach_target(&work, rate, before_gain, &opts)?;
    work = levelled;

    let output_lufs = integrated_lufs(&work, rate)?;
    let applied_gain_db = if before_gain.is_finite() && output_lufs.is_finite() {
        output_lufs - before_gain
    } else {
        f64::NEG_INFINITY
    };

    let report = EnhanceReport {
        input_lufs,
        output_lufs,
        input_peak_dbfs,
        output_peak_dbfs: peak_dbfs(&work),
        input_noise_floor_dbfs,
        output_noise_floor_dbfs: noise_floor_dbfs(&work, rate),
        applied_gain_db,
        limiter_reduction_db: limiting.sustained_reduction_db,
        limiter_peak_reduction_db: limiting.max_reduction_db,
        reached_target: (output_lufs - opts.target_lufs).abs() < 0.5,
    };
    Ok((work, report))
}

/// RBJ cookbook coefficients for `kind` with its corner at `corner_hz` and a
/// Butterworth Q: maximally flat for the high-pass, no overshoot for the
/// shelves.
///
/// Returns [`Error::InvalidArgument`] naming `option` when the corner is
/// negative or at or above the Nyquist limit of `rate`.
fn design(
    kind: Type<f64>,
    rate: f64,
    corner_hz: f64,
    option: &'static str,
) -> Result<Coefficients<f64>> {
    let nyquist = rate / 2.0;
    Coefficients::from_normalized_params(kind, corner_hz / nyquist, Q_BUTTERWORTH_F64).map_err(
        |err| Error::InvalidArgument {
            arg: option,
            reason: format!(
                "corner {corner_hz} Hz is outside 0..{nyquist} Hz, the band of {rate} Hz \
                 audio ({err:?}); set it below {nyquist} Hz or disable the stage"
            ),
        },
    )
}

/// Filter `samples` in place through a fresh Direct Form II transposed
/// section, keeping the state in f64.
fn filter(coeffs: Coefficients<f64>, samples: &mut [f32]) {
    let mut section = DirectForm2Transposed::<f64>::new(coeffs);
    for s in samples.iter_mut() {
        *s = section.run(*s as f64) as f32;
    }
}

/// Gain to the loudness target, holding the ceiling with the limiter.
///
/// Limiting lowers loudness, so one pass undershoots. Each further pass closes
/// the remaining gap; three is enough to converge to well inside 0.1 LU, and a
/// fixed count keeps the stage from looping on a signal it cannot lift.
///
/// Returns the samples and what the limiter did in total.
fn reach_target(
    samples: &[f32],
    rate: u32,
    current_lufs: f64,
    opts: &EnhanceOptions,
) -> Result<(Vec<f32>, LimiterReport)> {
    let idle = LimiterReport {
        max_reduction_db: 0.0,
        sustained_reduction_db: 0.0,
        fraction_reduced: 0.0,
    };
    if !current_lufs.is_finite() {
        // Silence has no loudness to move. Applying a gain to it would only
        // raise whatever numerical dust is in the buffer.
        return Ok((samples.to_vec(), idle));
    }

    let Some(window_ms) = opts.limiter_window_ms else {
        // No limiter: the ceiling caps the gain, and the target is missed by
        // however far the peaks stick out.
        let gain = gain_capped_by_peak(samples, opts.target_lufs - current_lufs, opts);
        return Ok((apply_gain(samples, gain), idle));
    };

    let lim = LimiterOptions {
        ceiling_dbfs: opts.peak_ceiling_dbfs,
        window_ms,
    };

    let mut out = samples.to_vec();
    let mut total = idle;
    for _ in 0..6 {
        let lufs = integrated_lufs(&out, rate)?;
        if !lufs.is_finite() || (lufs - opts.target_lufs).abs() < 0.1 {
            break;
        }
        let wanted_db = opts.target_lufs - lufs;
        let (stepped, report) = largest_step_within_cap(&out, rate, wanted_db, lim, opts);
        // A pass that moves nothing means the cap is already binding; another
        // would compute the same thing again.
        if stepped == out {
            break;
        }
        out = stepped;
        total = LimiterReport {
            max_reduction_db: total.max_reduction_db.max(report.max_reduction_db),
            sustained_reduction_db: total
                .sustained_reduction_db
                .max(report.sustained_reduction_db),
            fraction_reduced: total.fraction_reduced.max(report.fraction_reduced),
        };
    }
    Ok((out, total))
}

/// Apply as much of `wanted_db` as the limiting cap allows.
///
/// The limiter's gain reduction rises monotonically with the gain fed to it,
/// so the largest acceptable gain is found by bisection. Ten halvings settle a
/// 30 dB span to under 0.03 dB, far finer than the 0.1 LU the caller stops at.
///
/// A reduction of gain — `wanted_db` below zero — never engages the limiter,
/// so it is applied whole.
fn largest_step_within_cap(
    samples: &[f32],
    rate: u32,
    wanted_db: f64,
    lim: LimiterOptions,
    opts: &EnhanceOptions,
) -> (Vec<f32>, LimiterReport) {
    let attempt = |db: f64| limit(&apply_gain(samples, 10f64.powf(db / 20.0)), rate, lim);

    if wanted_db <= 0.0 {
        return attempt(wanted_db);
    }

    let (full, full_report) = attempt(wanted_db);
    if full_report.sustained_reduction_db <= opts.max_limiting_db {
        return (full, full_report);
    }

    let (mut lo, mut hi) = (0.0f64, wanted_db);
    let mut best = (
        samples.to_vec(),
        LimiterReport {
            max_reduction_db: 0.0,
            sustained_reduction_db: 0.0,
            fraction_reduced: 0.0,
        },
    );
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let (out, r) = attempt(mid);
        if r.sustained_reduction_db <= opts.max_limiting_db {
            best = (out, r);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    best
}

/// The requested gain, reduced so the sample peak stays under the ceiling.
fn gain_capped_by_peak(samples: &[f32], wanted_db: f64, opts: &EnhanceOptions) -> f64 {
    let mut gain = 10f64.powf(wanted_db / 20.0);
    let peak = peak_dbfs(samples);
    if peak.is_finite() {
        let ceiling_gain = 10f64.powf((opts.peak_ceiling_dbfs - peak) / 20.0);
        if ceiling_gain < gain {
            gain = ceiling_gain;
        }
    }
    gain
}

fn apply_gain(samples: &[f32], gain: f64) -> Vec<f32> {
    samples.iter().map(|&s| (s as f64 * gain) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    /// Magnitude response `|H(e^{jw})|` at `freq`, as a linear gain, evaluated
    /// from the coefficients so a test can check a filter shape without
    /// running a signal through it.
    fn magnitude_at(c: &Coefficients<f64>, rate: f64, freq: f64) -> f64 {
        let w = std::f64::consts::TAU * freq / rate;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let num_re = c.b0 + c.b1 * c1 + c.b2 * c2;
        let num_im = -(c.b1 * s1 + c.b2 * s2);
        let den_re = 1.0 + c.a1 * c1 + c.a2 * c2;
        let den_im = -(c.a1 * s1 + c.a2 * s2);
        ((num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im)).sqrt()
    }

    #[test]
    fn a_highpass_cuts_below_its_corner_and_passes_above() {
        let hp = design(Type::HighPass, RATE, 80.0, "highpass_hz").unwrap();
        // Butterworth is -3 dB at the corner: 10^(-3/20) = 0.708.
        let at_corner = magnitude_at(&hp, RATE, 80.0);
        assert!(
            (at_corner - 0.7079).abs() < 0.01,
            "corner gain {at_corner:.4} should be -3 dB"
        );
        assert!(
            magnitude_at(&hp, RATE, 20.0) < 0.1,
            "20 Hz must be well down"
        );
        assert!(
            magnitude_at(&hp, RATE, 1000.0) > 0.99,
            "passband must be flat"
        );
    }

    #[test]
    fn shelves_apply_their_gain_on_the_correct_side() {
        let low = design(Type::LowShelf(6.0), RATE, 200.0, "bass_corner_hz").unwrap();
        // +6 dB is a linear gain of 2.
        assert!(
            (magnitude_at(&low, RATE, 20.0) - 2.0).abs() < 0.05,
            "low shelf lifts lows"
        );
        assert!(
            (magnitude_at(&low, RATE, 8000.0) - 1.0).abs() < 0.05,
            "and leaves highs"
        );

        let high = design(Type::HighShelf(6.0), RATE, 2000.0, "presence_corner_hz").unwrap();
        assert!(
            (magnitude_at(&high, RATE, 16000.0) - 2.0).abs() < 0.05,
            "high shelf lifts highs"
        );
        assert!(
            (magnitude_at(&high, RATE, 100.0) - 1.0).abs() < 0.05,
            "and leaves lows"
        );
    }

    #[test]
    fn a_shelf_at_zero_gain_is_a_pass_through() {
        let flat = design(Type::LowShelf(0.0), RATE, 200.0, "bass_corner_hz").unwrap();
        for f in [20.0, 100.0, 1000.0, 10000.0] {
            let g = magnitude_at(&flat, RATE, f);
            assert!((g - 1.0).abs() < 1e-9, "{f} Hz gain {g}");
        }
    }

    #[test]
    fn filtering_a_tone_matches_the_computed_magnitude() {
        // The response function and the actual filter must agree, or tests that
        // use `magnitude_at` prove nothing about what `filter` does.
        let freq = 1000.0;
        let hp = design(Type::HighPass, RATE, 300.0, "highpass_hz").unwrap();
        let n = 48_000;
        let mut buf: Vec<f32> = (0..n)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / RATE).sin() as f32)
            .collect();
        filter(hp, &mut buf);

        // Skip the transient; measure amplitude over the steady-state tail.
        let tail = &buf[n / 2..];
        let peak = tail.iter().fold(0.0f32, |a, &b| a.max(b.abs())) as f64;
        let expected = magnitude_at(&hp, RATE, freq);
        assert!(
            (peak - expected).abs() < 0.02,
            "measured {peak:.4} vs computed {expected:.4}"
        );
    }

    #[test]
    fn default_options_skip_a_presence_shelf_above_nyquist() {
        // 8 kHz audio has a 4 kHz Nyquist limit, below the default 7 kHz
        // presence corner; the stage has nothing to lift and is skipped.
        let rate = 8_000u32;
        let tone: Vec<f32> = (0..rate as usize * 2)
            .map(|i| 0.3 * (std::f64::consts::TAU * 440.0 * i as f64 / rate as f64).sin() as f32)
            .collect();
        let (out, _) = enhance(&tone, rate, EnhanceOptions::default())
            .unwrap_or_else(|e| panic!("default options on 8 kHz audio: {e}"));
        assert_eq!(out.len(), tone.len());
    }

    #[test]
    fn a_corner_at_or_above_nyquist_is_rejected() {
        let err = design(Type::HighShelf(1.5), 8_000.0, 7_000.0, "presence_corner_hz")
            .expect_err("7 kHz is above the 4 kHz Nyquist limit of 8 kHz audio");
        let msg = err.to_string();
        assert!(msg.contains("presence_corner_hz"), "{msg}");
        assert!(msg.contains("4000"), "{msg}");
    }
}
