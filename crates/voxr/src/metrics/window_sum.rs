//! Sums of every run of `n` consecutive values: `np.convolve(v, np.ones(n),
//! 'valid')`.
//!
//! Each sum runs in double-double arithmetic and rounds once. When the
//! values are exact squares of 24-bit samples, the exact sum fits in the
//! double-double, so equal runs give equal sums and the first of a tie
//! wins, as NumPy's `argmax` picks it.

/// `a + b` and its rounding error (Knuth's two-sum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

/// A running sum held as an unevaluated `hi + lo` pair.
#[derive(Debug, Clone, Copy, Default)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn add(&mut self, v: f64) {
        let (s, e) = two_sum(self.hi, v);
        let (hi, lo) = two_sum(s, e + self.lo);
        self.hi = hi;
        self.lo = lo;
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// The sum of `values`, rounded once.
pub(crate) fn exact_sum(values: &[f64]) -> f64 {
    let mut acc = DoubleDouble::default();
    for &v in values {
        acc.add(v);
    }
    acc.value()
}

/// The sum of each run of `n` consecutive `values`: `len - n + 1` sums.
/// Empty when `n` is 0 or longer than `values`.
pub(crate) fn window_sums(values: &[f64], n: usize) -> Vec<f64> {
    if n == 0 || n > values.len() {
        return Vec::new();
    }
    let mut acc = DoubleDouble::default();
    for &v in &values[..n] {
        acc.add(v);
    }
    let mut out = Vec::with_capacity(values.len() - n + 1);
    out.push(acc.value());
    for i in n..values.len() {
        acc.add(values[i]);
        acc.add(-values[i - n]);
        out.push(acc.value());
    }
    out
}

/// The index of the first largest value: `np.argmax`. `None` when empty.
pub(crate) fn first_argmax(values: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in values.iter().enumerate() {
        // Only a strictly larger value replaces the best, as NumPy keeps
        // the first of a tie.
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_sums_match_direct_sums() {
        let v: Vec<f64> = (0..50).map(|i| ((i * 7) % 11) as f64 * 0.125).collect();
        let sums = window_sums(&v, 5);
        assert_eq!(sums.len(), 46);
        for (i, s) in sums.iter().enumerate() {
            let direct: f64 = v[i..i + 5].iter().sum();
            assert_eq!(*s, direct, "window {i}");
        }
        assert!(window_sums(&v, 0).is_empty());
        assert!(window_sums(&v, 51).is_empty());
    }

    #[test]
    fn a_plateau_ties_and_the_first_wins() {
        // A burst inside long silence: every window holding all of it sums
        // the same, however far the window has slid.
        let mut v = vec![0.0f64; 100];
        for (k, slot) in v[40..46].iter_mut().enumerate() {
            *slot = 1e-9 * (k as f64 + 1.0) + 3.0;
        }
        let sums = window_sums(&v, 20);
        assert_eq!(first_argmax(&sums), Some(26));
        assert_eq!(sums[26], sums[40]);
    }

    #[test]
    fn exact_sum_keeps_small_terms() {
        // A left fold drops each 1e-16 against 1.0; the exact sum keeps
        // all twenty.
        let mut v = vec![1.0f64];
        v.extend([1e-16f64; 20]);
        let left: f64 = v.iter().sum();
        assert_eq!(left, 1.0);
        assert_eq!(exact_sum(&v), 1.0 + 2e-15);
    }
}
