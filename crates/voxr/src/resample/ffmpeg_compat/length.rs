//! How many samples ffmpeg's resampler emits for a finite input.
//!
//! Output `m` sits at input time `t(m) = m * down / up`, and its window
//! reaches `taps / 2` samples past `t(m)`. With `n` input samples and
//! `h = taps / 2`:
//!
//! 1. Streaming. The resampler starts once it holds more than `taps`
//!    samples. It then emits every output whose window lies inside the
//!    input: `A = ceil((n - h) * up / down)` of them. With `n <= taps` it
//!    emits none here.
//! 2. Flush. `S` counts the input samples from the next output's window
//!    start to the end: `n - (floor(t(A)) - h + 1)` after streaming, or `n`
//!    when nothing streamed. The flush mirrors `(S + 1) / 2` samples past
//!    the end. It emits nothing when `n` plus that padding is at most
//!    `taps`. Otherwise it emits every output with `t(m) < n + (S + 1) / 2 - h`.
//!
//! The total is `ceil(n * up / down)` for every upsampling ratio. A
//! downsampling ratio can end one sample short of that. Very short inputs
//! give no output.

use super::bank::Plan;
use crate::error::{Error, Result};

/// Output length for `in_len` samples at `from` Hz converted to `to` Hz,
/// as ffmpeg's default resampler emits it.
///
/// Equal rates give `in_len`: ffmpeg passes the samples through. Returns
/// [`Error::InvalidArgument`] when either rate is 0 or the ratio needs a
/// filter over the tap limit.
pub fn ffmpeg_resampled_len(in_len: usize, from: u32, to: u32) -> Result<usize> {
    match plan_for(from, to)? {
        Some(plan) => output_len(in_len, &plan),
        None => Ok(in_len),
    }
}

/// The plan for `from` Hz to `to` Hz, or `None` for equal rates. Errors when
/// either rate is 0.
pub(super) fn plan_for(from: u32, to: u32) -> Result<Option<Plan>> {
    if from == 0 || to == 0 {
        return Err(Error::InvalidArgument {
            arg: if from == 0 { "from" } else { "to" },
            reason: format!("sample rate is 0 Hz in {from} Hz to {to} Hz; pass both rates in Hz"),
        });
    }
    if from == to {
        return Ok(None);
    }
    Plan::new(from, to).map(Some)
}

/// Output length for `n` input samples under `plan`.
pub(super) fn output_len(n: usize, plan: &Plan) -> Result<usize> {
    let n = n as u128;
    let h = (plan.taps / 2) as u128;
    let taps = plan.taps as u128;
    let (up, down) = (u128::from(plan.up), u128::from(plan.down));
    let (streamed, span) = if n > taps {
        let streamed = ((n - h) * up).div_ceil(down);
        let index = streamed * down / up;
        (streamed, n + h - 1 - index)
    } else {
        (0, n)
    };
    let pad = span.div_ceil(2);
    if n + pad <= taps {
        return Ok(0);
    }
    let end = n + pad - h;
    let total = streamed.max((end * up).div_ceil(down));
    usize::try_from(total).map_err(|_| Error::InvalidArgument {
        arg: "samples",
        reason: format!(
            "{n} samples converted by {up}/{down} give {total} samples, more than a usize \
             holds; split the input"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(n: usize, from: u32, to: u32) -> Result<usize> {
        ffmpeg_resampled_len(n, from, to)
    }

    #[test]
    fn upsampling_rounds_up() -> Result<()> {
        assert_eq!(len(1000, 24_000, 48_000)?, 2000);
        assert_eq!(len(1000, 22_050, 48_000)?, 2177);
        assert_eq!(len(1013, 44_100, 48_000)?, 1103);
        assert_eq!(len(4097, 16_000, 48_000)?, 12_291);
        Ok(())
    }

    #[test]
    fn downsampling_can_end_one_short() -> Result<()> {
        assert_eq!(len(1000, 48_000, 16_000)?, 333);
        assert_eq!(len(1001, 48_000, 16_000)?, 334);
        assert_eq!(len(5002, 48_000, 44_100)?, 4595);
        assert_eq!(len(5003, 48_000, 44_100)?, 4597);
        Ok(())
    }

    #[test]
    fn short_inputs_flush_only_or_vanish() -> Result<()> {
        assert_eq!(len(21, 24_000, 48_000)?, 0);
        assert_eq!(len(22, 24_000, 48_000)?, 34);
        assert_eq!(len(32, 24_000, 48_000)?, 64);
        assert_eq!(len(33, 24_000, 48_000)?, 66);
        assert_eq!(len(66, 48_000, 16_000)?, 0);
        assert_eq!(len(67, 48_000, 16_000)?, 17);
        assert_eq!(len(0, 44_100, 48_000)?, 0);
        Ok(())
    }

    #[test]
    fn equal_rates_pass_through() -> Result<()> {
        assert_eq!(len(7, 48_000, 48_000)?, 7);
        Ok(())
    }

    #[test]
    fn a_zero_rate_is_refused() {
        assert!(matches!(
            len(10, 0, 48_000),
            Err(Error::InvalidArgument { arg: "from", .. })
        ));
        assert!(matches!(
            len(10, 48_000, 0),
            Err(Error::InvalidArgument { arg: "to", .. })
        ));
    }
}
