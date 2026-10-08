//! A take's seed as the whole number an engine draws its noise from.

// Each adapter that reads a seed sits behind its own feature.
#![cfg_attr(not(any(feature = "vits", feature = "voxcpm")), allow(dead_code))]

use crate::error::{Error, Result};

/// The largest whole number an f64 holds exactly: JavaScript's
/// `Number.MAX_SAFE_INTEGER`.
const MAX_SAFE: f64 = 9_007_199_254_740_991.0;

/// `seed` as a `u64`. A seed that is not a whole number from 0 to 2^53 - 1
/// is refused: an engine would otherwise draw from a seed other than the
/// one the take is keyed by.
pub(crate) fn engine_seed(seed: f64) -> Result<u64> {
    if seed.is_finite() && seed >= 0.0 && seed.fract() == 0.0 && seed <= MAX_SAFE {
        return Ok(seed as u64);
    }
    Err(Error::InvalidArgument {
        arg: "seed",
        reason: format!(
            "seed {seed} is not a whole number from 0 to {MAX_SAFE}. Pass a seed such as 7"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_seeds_pass_and_others_are_refused() {
        assert_eq!(engine_seed(7.0).ok(), Some(7));
        assert_eq!(engine_seed(100_999.0).ok(), Some(100_999));
        assert_eq!(engine_seed(MAX_SAFE).ok(), Some(9_007_199_254_740_991));
        for bad in [-1.0, 7.5, f64::NAN, f64::INFINITY, MAX_SAFE + 2.0] {
            assert!(engine_seed(bad).is_err(), "{bad}");
        }
    }
}
