//! The rate plan and the polyphase filter bank.
//!
//! # Plan
//!
//! The rate pair reduces to `up / down`: output sample `m` sits at input
//! time `m * down / up`. The bank holds `up` phases when `up` is at most
//! [`MAX_PHASES`], so every output lands exactly on a phase. A larger `up`
//! falls back to [`MAX_PHASES`] phases, and an output between two phases
//! interpolates linearly between them.
//!
//! # Filter
//!
//! The cutoff factor is `min(1, CUTOFF * to / from)`. Upsampling therefore
//! keeps the band right up to the input Nyquist limit: the factor clamps to 1
//! and the cutoff sits AT the input Nyquist frequency, not below it.
//! Downsampling scales the cutoff to `CUTOFF` times the output Nyquist
//! limit.
//!
//! The filter spans `taps = 2 * ceil(FILTER_SIZE / 2 / factor)` input
//! samples. Phase `k` of `P` has fractional offset `k / P`, and its tap `j`
//! sits at distance `d = j - (taps / 2 - 1) - k / P` from the output time.
//! Its value is `factor * sinc(factor * d)` times a Kaiser window of beta
//! [`KAISER_BETA`] over a half-width of `taps / 2`. The sinc is exactly 0 at
//! every non-zero integer argument. Every phase divides by the tap sum of
//! phase 0, so phase 0 has unit DC gain. At a factor of 1,
//! phase 0 is a unit impulse. The design runs in `f64`, and each coefficient
//! is stored as `f32`.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::resample::math::{bessel_i0, gcd};

/// Filter length, in input samples, at a cutoff factor of 1.
const FILTER_SIZE: f64 = 32.0;

/// Largest phase count: `1 << phase_shift` with `phase_shift` 10.
pub(super) const MAX_PHASES: u64 = 1 << 10;

/// The 6 dB point as a fraction of the lower Nyquist limit.
const CUTOFF: f64 = 0.97;

/// Kaiser window shape.
const KAISER_BETA: f64 = 9.0;

/// Largest filter length accepted, in taps. It bounds the bank at
/// `(MAX_PHASES + 1) * MAX_TAPS` coefficients.
const MAX_TAPS: usize = 1 << 14;

/// The reduced rate ratio and the filter shape it implies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Plan {
    /// Output rate over the rates' greatest common divisor.
    pub up: u64,
    /// Input rate over the rates' greatest common divisor.
    pub down: u64,
    /// Phase count `P`: `up`, or [`MAX_PHASES`] when `up` is larger.
    pub phases: u64,
    /// Filter length in input samples. Always even.
    pub taps: usize,
    /// Cutoff as a fraction of the input Nyquist limit.
    pub factor: f64,
}

impl Plan {
    /// The plan for converting `from` Hz to `to` Hz. Both rates are
    /// non-zero.
    pub fn new(from: u32, to: u32) -> Result<Self> {
        let divisor = gcd(u64::from(from), u64::from(to));
        let up = u64::from(to) / divisor;
        let down = u64::from(from) / divisor;
        let factor = (CUTOFF * f64::from(to) / f64::from(from)).min(1.0);
        let half = (FILTER_SIZE / 2.0 / factor).ceil();
        if half * 2.0 > MAX_TAPS as f64 {
            return Err(Error::InvalidArgument {
                arg: "to",
                reason: format!(
                    "downsampling {from} Hz to {to} Hz needs a {}-tap filter, over the \
                     {MAX_TAPS}-tap limit; resample in two steps through an intermediate rate",
                    half * 2.0
                ),
            });
        }
        Ok(Self {
            up,
            down,
            phases: up.min(MAX_PHASES),
            taps: 2 * half as usize,
            factor,
        })
    }

    /// Whether outputs fall between phases and interpolate.
    pub fn interpolates(&self) -> bool {
        self.phases < self.up
    }
}

/// Phase rows of [`Plan::taps`] coefficients each. An interpolating plan
/// holds one extra row, phase `P`, which is phase 0 one input sample later.
#[derive(Debug, Clone)]
pub(super) struct Bank {
    rows: Vec<Vec<f32>>,
}

impl Bank {
    /// The bank for `plan`.
    pub fn new(plan: &Plan) -> Self {
        let rows = plan.phases + u64::from(plan.interpolates());
        let raw: Vec<Vec<f64>> = (0..rows).map(|k| phase_row(plan, k)).collect();
        let dc: f64 = raw.first().map_or(1.0, |row| row.iter().sum());
        let rows = raw
            .iter()
            .map(|row| row.iter().map(|&v| (v / dc) as f32).collect())
            .collect();
        Self { rows }
    }

    /// Phase `k`'s coefficients, or `None` past the last row.
    pub fn row(&self, k: u64) -> Option<&[f32]> {
        let k = usize::try_from(k).ok()?;
        self.rows.get(k).map(Vec::as_slice)
    }
}

/// Phase `k`'s coefficients before normalization.
fn phase_row(plan: &Plan, k: u64) -> Vec<f64> {
    let frac = k as f64 / plan.phases as f64;
    let center = (plan.taps / 2 - 1) as f64;
    let half = (plan.taps / 2) as f64;
    let i0_beta = bessel_i0(KAISER_BETA);
    (0..plan.taps)
        .map(|j| {
            let d = (j as f64 - center) - frac;
            let x = plan.factor * d;
            let sinc = if x == 0.0 {
                1.0
            } else if x.fract() == 0.0 {
                0.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let r = d / half;
            let w = bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
            sinc * plan.factor * w
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsampling_keeps_the_full_band() -> Result<()> {
        let plan = Plan::new(24_000, 48_000)?;
        assert_eq!((plan.up, plan.down, plan.phases, plan.taps), (2, 1, 2, 32));
        assert_eq!(plan.factor, 1.0);
        assert!(!plan.interpolates());
        Ok(())
    }

    #[test]
    fn downsampling_widens_the_filter() -> Result<()> {
        let plan = Plan::new(48_000, 16_000)?;
        assert_eq!((plan.up, plan.down, plan.taps), (1, 3, 100));
        assert_eq!(Plan::new(48_000, 44_100)?.taps, 36);
        assert_eq!(Plan::new(44_100, 16_000)?.taps, 92);
        Ok(())
    }

    #[test]
    fn a_large_up_term_falls_back_to_interpolation() -> Result<()> {
        let plan = Plan::new(48_000, 44_101)?;
        assert_eq!(plan.phases, MAX_PHASES);
        assert!(plan.interpolates());
        let bank = Bank::new(&plan);
        assert!(bank.row(MAX_PHASES).is_some());
        assert!(bank.row(MAX_PHASES + 1).is_none());
        Ok(())
    }

    #[test]
    fn phase_zero_at_unit_factor_is_an_impulse() -> Result<()> {
        let bank = Bank::new(&Plan::new(16_000, 48_000)?);
        let row = bank.row(0).ok_or_else(|| Error::DataError {
            reason: "no phase 0".to_string(),
        })?;
        for (j, &v) in row.iter().enumerate() {
            assert_eq!(v, if j == 15 { 1.0 } else { 0.0 }, "tap {j}");
        }
        Ok(())
    }

    #[test]
    fn phase_zero_has_unit_dc_gain() -> Result<()> {
        let bank = Bank::new(&Plan::new(44_100, 16_000)?);
        let row = bank.row(0).ok_or_else(|| Error::DataError {
            reason: "no phase 0".to_string(),
        })?;
        let sum: f64 = row.iter().map(|&v| f64::from(v)).sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum {sum}");
        Ok(())
    }

    #[test]
    fn an_extreme_ratio_is_refused() {
        let r = Plan::new(768_000, 8);
        assert!(matches!(r, Err(Error::InvalidArgument { arg: "to", .. })));
    }
}
