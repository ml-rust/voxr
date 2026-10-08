//! The rules a pick follows: the fresh seed a re-rolled part draws, the
//! comparison that decides which take is kept, and the request a take is
//! spoken from.

use super::engine::{SpeakRequest, TakeSpec};
use super::pick::PickSetup;

/// The smallest seed a re-rolled part draws.
const REROLL_BASE: f64 = 100000.0;
/// The span re-rolled seeds are drawn from.
const REROLL_SPAN: f64 = 900000.0;

/// `100000 + Math.floor(random * 900000)`: a re-rolled part's fresh seed.
pub(super) fn reroll_seed(random: f64) -> f64 {
    REROLL_BASE + (random * REROLL_SPAN).floor()
}

/// `a < b`, or `a <= b` when `or_equal`, with `None` as JavaScript's
/// `undefined`: every comparison with it is false.
pub(super) fn beats(a: Option<f64>, b: Option<f64>, or_equal: bool) -> bool {
    let (a, b) = (a.unwrap_or(f64::NAN), b.unwrap_or(f64::NAN));
    a < b || (or_equal && a <= b)
}

/// The request for `spec`'s take, spoken as `text`. An empty run style is
/// no style.
pub(super) fn request<'r, H>(
    setup: &'r PickSetup<'_, H>,
    spec: &'r TakeSpec,
    text: &'r str,
) -> SpeakRequest<'r, H> {
    SpeakRequest {
        text,
        delivery: spec.delivery.as_deref(),
        style: setup.style.filter(|s| !s.is_empty()),
        voice: setup.voice,
        seed: spec.seed,
        cfg: setup.cfg,
        steps: setup.steps,
        // VoxCPM2 is never sped up: its pace comes from the reference.
        speed: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undefined_badness_never_compares() {
        assert!(!beats(None, Some(1.0), true));
        assert!(!beats(Some(1.0), None, true));
        assert!(beats(Some(1.0), Some(1.0), true));
        assert!(!beats(Some(1.0), Some(1.0), false));
    }
}
