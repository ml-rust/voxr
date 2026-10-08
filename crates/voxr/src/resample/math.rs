//! Numeric helpers the windowed-sinc resamplers share: the Bessel function
//! of a Kaiser window, and the greatest common divisor that reduces a rate
//! pair.

/// The modified Bessel function of the first kind, order 0, by its power
/// series. The series converges for every finite `x`.
pub(super) fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-18 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// The greatest common divisor of `a` and `b`. `gcd(0, 0)` is 0.
pub(super) fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_i0_matches_known_values() {
        assert_eq!(bessel_i0(0.0), 1.0);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_4).abs() < 1e-15);
        assert!((bessel_i0(9.0) / 1_093.588_354_511_374_6 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn gcd_reduces_rate_pairs() {
        assert_eq!(gcd(44_100, 16_000), 100);
        assert_eq!(gcd(48_000, 16_000), 16_000);
        assert_eq!(gcd(16_001, 16_000), 1);
        assert_eq!(gcd(7, 0), 7);
    }
}
