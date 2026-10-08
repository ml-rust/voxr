//! The anti-aliasing filter and its polyphase rows.
//!
//! # Filter
//!
//! The filter is a Kaiser-windowed sinc low-pass with linear phase. Its
//! shape is set in units of the lower of the two rates, so one design
//! serves upsampling and downsampling alike:
//!
//! - [`CUTOFF`]: the half-amplitude point, between a passband edge at 0.913
//!   and a stopband edge at 1.0 of the lower Nyquist limit.
//! - [`HALF_WIDTH`]: the window's half-width, in samples of the lower rate.
//! - [`KAISER_BETA`]: the window shape. Kaiser's formula `β = 0.1102 (A −
//!   8.7)` ties this β to a stopband attenuation `A` of about 127 dB, past
//!   20-bit precision.
//!
//! In input samples, with `s = min(1, to / from)`, the filter at distance `d`
//! from the output time is `fc · sinc(fc · d) · w(d / W)`. Here `fc = CUTOFF ·
//! s`, `W = HALF_WIDTH / s`, and `w` is the Kaiser window, zero for `|d| ≥ W`.
//! The gain is unity at DC to within the passband ripple, with no
//! normalization.
//!
//! # Rows
//!
//! The rate pair reduces to `up / down`: output sample `m` sits at input time
//! `m · down / up`, so its phase is `(m · down) mod up`. Phase `p` has
//! `2 · reach + 1` taps, where `reach = ceil(W)`. Tap `j` reads input
//! `floor(m · down / up) − reach + j` at distance `p / up + reach − j`.
//! A bank holds every phase's row when it fits in [`MAX_BANK`] coefficients.
//! Otherwise each output computes its own row.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::resample::math::{bessel_i0, gcd};

/// Half-amplitude cutoff as a fraction of the lower Nyquist limit.
const CUTOFF: f64 = 0.956_872;

/// Kaiser window half-width, in samples of the lower rate.
const HALF_WIDTH: f64 = 96.8336;

/// Kaiser window shape.
const KAISER_BETA: f64 = 13.0401;

/// Largest bank, in `f64` coefficients.
const MAX_BANK: u64 = 1 << 21;

/// Largest `reach`, in input samples. It caps the rate ratio near 338.
const MAX_REACH: f64 = 32_768.0;

/// The reduced rate ratio and the filter it implies.
#[derive(Debug, Clone)]
pub(super) struct Filter {
    /// Output rate over the rates' greatest common divisor.
    pub up: u64,
    /// Input rate over the rates' greatest common divisor.
    pub down: u64,
    /// Taps either side of the output time, in input samples.
    pub reach: usize,
    /// Cutoff as a fraction of the input Nyquist limit.
    fc: f64,
    /// Window half-width, in input samples.
    width: f64,
    /// `I0(KAISER_BETA)`, the window's peak before normalization.
    i0_beta: f64,
}

impl Filter {
    /// The filter for converting `from` Hz to `to` Hz. Both rates are
    /// non-zero.
    pub fn new(from: u32, to: u32) -> Result<Self> {
        let divisor = gcd(u64::from(from), u64::from(to)).max(1);
        let scale = (f64::from(to) / f64::from(from)).min(1.0);
        let width = HALF_WIDTH / scale;
        let reach = width.ceil();
        if reach > MAX_REACH {
            return Err(Error::InvalidArgument {
                arg: "from",
                reason: format!(
                    "resampling {from} Hz to {to} Hz needs a filter reaching {reach} input \
                     samples, over the {MAX_REACH} limit; pass rates whose ratio is at most \
                     338"
                ),
            });
        }
        Ok(Self {
            up: u64::from(to) / divisor,
            down: u64::from(from) / divisor,
            reach: reach as usize,
            fc: CUTOFF * scale,
            width,
            i0_beta: bessel_i0(KAISER_BETA),
        })
    }

    /// Taps in one phase row.
    pub fn taps(&self) -> usize {
        2 * self.reach + 1
    }

    /// The coefficient at `d` input samples from the output time.
    fn coefficient(&self, d: f64) -> f64 {
        if d.abs() >= self.width {
            return 0.0;
        }
        let x = self.fc * d;
        let sinc = if x == 0.0 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        };
        let r = d / self.width;
        let w = bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / self.i0_beta;
        self.fc * sinc * w
    }

    /// Phase `p`'s row into `row`, which holds [`Filter::taps`] values.
    pub fn fill_row(&self, p: u64, row: &mut [f64]) {
        let up = self.up as f64;
        for (j, c) in row.iter_mut().enumerate() {
            let d = (p as f64 + (self.reach as f64 - j as f64) * up) / up;
            *c = self.coefficient(d);
        }
    }
}

/// Phase rows, precomputed or made per output.
pub(super) enum Rows {
    /// Every phase's row, `taps` coefficients each, phase-major.
    Bank { taps: usize, coefs: Vec<f64> },
    /// One scratch row, refilled for each output.
    PerOutput { scratch: Vec<f64> },
}

impl Rows {
    /// A bank when `up · taps` fits in [`MAX_BANK`], else a scratch row.
    pub fn new(filter: &Filter) -> Self {
        let taps = filter.taps();
        if filter.up.saturating_mul(taps as u64) <= MAX_BANK {
            let mut coefs = vec![0.0; filter.up as usize * taps];
            for (p, row) in coefs.chunks_exact_mut(taps).enumerate() {
                filter.fill_row(p as u64, row);
            }
            Self::Bank { taps, coefs }
        } else {
            Self::PerOutput {
                scratch: vec![0.0; taps],
            }
        }
    }

    /// Phase `p`'s row. `p` is below `filter.up`.
    pub fn row(&mut self, filter: &Filter, p: u64) -> &[f64] {
        match self {
            Self::Bank { taps, coefs } => {
                let start = p as usize * *taps;
                &coefs[start..start + *taps]
            }
            Self::PerOutput { scratch } => {
                filter.fill_row(p, scratch);
                scratch
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shape_follows_the_lower_rate() -> Result<()> {
        let down = Filter::new(48_000, 16_000)?;
        assert_eq!((down.up, down.down), (1, 3));
        assert_eq!(down.reach, (HALF_WIDTH * 3.0).ceil() as usize);
        let up = Filter::new(16_000, 48_000)?;
        assert_eq!((up.up, up.down), (3, 1));
        assert_eq!(up.reach, HALF_WIDTH.ceil() as usize);
        assert_eq!(up.fc, CUTOFF);
        Ok(())
    }

    #[test]
    fn every_phase_has_unit_dc_gain() -> Result<()> {
        for (from, to) in [(44_100, 16_000), (16_000, 48_000), (48_000, 16_000)] {
            let filter = Filter::new(from, to)?;
            let mut row = vec![0.0; filter.taps()];
            for p in 0..filter.up {
                filter.fill_row(p, &mut row);
                let sum: f64 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "{from}->{to} phase {p}: {sum}");
            }
        }
        Ok(())
    }

    #[test]
    fn the_filter_is_symmetric_and_bounded() -> Result<()> {
        let filter = Filter::new(48_000, 16_000)?;
        let mut row = vec![0.0; filter.taps()];
        filter.fill_row(0, &mut row);
        for (a, b) in row.iter().zip(row.iter().rev()) {
            assert_eq!(a, b);
        }
        assert_eq!(row[filter.reach], filter.fc);
        assert_eq!(filter.coefficient(filter.width), 0.0);
        Ok(())
    }

    #[test]
    fn a_bank_and_per_output_rows_agree() -> Result<()> {
        let filter = Filter::new(44_100, 16_000)?;
        let mut bank = Rows::new(&filter);
        assert!(matches!(bank, Rows::Bank { .. }));
        let mut scratch = Rows::PerOutput {
            scratch: vec![0.0; filter.taps()],
        };
        for p in [0, 1, 77, 159] {
            assert_eq!(bank.row(&filter, p), scratch.row(&filter, p));
        }
        Ok(())
    }

    #[test]
    fn a_large_up_term_makes_rows_per_output() -> Result<()> {
        let filter = Filter::new(44_101, 16_000)?;
        assert!(matches!(Rows::new(&filter), Rows::PerOutput { .. }));
        Ok(())
    }

    #[test]
    fn an_extreme_ratio_is_refused() {
        let r = Filter::new(768_000, 8);
        assert!(matches!(r, Err(Error::InvalidArgument { arg: "from", .. })));
    }
}
