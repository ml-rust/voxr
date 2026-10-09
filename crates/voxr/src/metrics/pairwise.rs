//! NumPy's pairwise summation, the order `np.sum` and `np.mean` add a
//! contiguous `float64` array in.
//!
//! Below 8 elements the sum runs left to right from zero. Up to 128 it
//! keeps 8 running sums over strides of 8, joins them as a balanced tree,
//! then adds the remainder. Above 128 it splits at half the length, rounded
//! down to a multiple of 8, and adds the two halves' sums.

/// Block length below which NumPy stops splitting.
const BLOCK: usize = 128;

/// Lanes of the unrolled loop.
const LANES: usize = 8;

/// `np.sum(a)` for a contiguous `float64` array, in NumPy's order.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < LANES {
        return a.iter().fold(0.0, |acc, &v| acc + v);
    }
    if n <= BLOCK {
        let mut r = [0.0f64; LANES];
        r.copy_from_slice(&a[..LANES]);
        let whole = n - n % LANES;
        for chunk in a[LANES..whole].as_chunks::<LANES>().0 {
            for (lane, &v) in r.iter_mut().zip(chunk) {
                *lane += v;
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[whole..] {
            res += v;
        }
        return res;
    }
    let half = n / 2;
    let split = half - half % LANES;
    pairwise_sum(&a[..split]) + pairwise_sum(&a[split..])
}

/// `np.mean(a)` for a contiguous `float64` array: [`pairwise_sum`] over the
/// length. `NaN` for an empty slice, as NumPy gives.
pub fn pairwise_mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_sums_run_left_to_right() {
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn long_sums_match_exact_integers() {
        let a: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(pairwise_sum(&a), 499_500.0);
        assert_eq!(pairwise_mean(&a), 499.5);
    }

    #[test]
    fn order_differs_from_a_left_fold() {
        // 1e16 swallows each lone 1.0 of a left fold. The pairwise tree
        // adds 14 of the ones to each other first; the lane that holds
        // 1e16 still swallows one.
        let mut a = vec![1.0f64; 16];
        a[0] = 1e16;
        let left: f64 = a.iter().fold(0.0, |s, &v| s + v);
        assert_eq!(left, 1e16);
        assert_eq!(pairwise_sum(&a), 1e16 + 14.0);
    }
}
