//! The dot product of an input window with a phase row, in the arithmetic
//! order ffmpeg's float resampler uses on x86-64 with AVX2 and FMA.
//!
//! Eight `f32` lanes run in parallel: lane `l` accumulates taps
//! `l, l + 8, l + 16, ...` in that order, each as one fused multiply-add.
//! The lanes then fold pairwise: lane `i` adds lane `i + 4`, then lane
//! `i + 2`, then lane 1 adds into lane 0. A different order changes the
//! result by float rounding only.

/// Lane count of one accumulator.
const LANES: usize = 8;

/// `window · row` over `min(window.len(), row.len())` taps.
pub(super) fn dot(window: &[f32], row: &[f32]) -> f32 {
    fold(lanes(window, row))
}

/// The output between two phases: each lane's sum for `row0` moved towards
/// its sum for `row1` by `frac`, then folded.
pub(super) fn dot_between(window: &[f32], row0: &[f32], row1: &[f32], frac: f32) -> f32 {
    let a = lanes(window, row0);
    let b = lanes(window, row1);
    let mut mixed = [0.0f32; LANES];
    for ((m, &a), &b) in mixed.iter_mut().zip(&a).zip(&b) {
        *m = (b - a).mul_add(frac, a);
    }
    fold(mixed)
}

/// The per-lane fused sums before folding.
fn lanes(window: &[f32], row: &[f32]) -> [f32; LANES] {
    let mut acc = [0.0f32; LANES];
    let mut x = window.chunks(LANES);
    let mut h = row.chunks(LANES);
    while let (Some(x), Some(h)) = (x.next(), h.next()) {
        for ((a, &x), &h) in acc.iter_mut().zip(x).zip(h) {
            *a = x.mul_add(h, *a);
        }
    }
    acc
}

/// Lanes `0..8` folded to one value: `i + 4`, then `i + 2`, then `1`.
fn fold(a: [f32; LANES]) -> f32 {
    let b = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    let c = [b[0] + b[2], b[1] + b[3]];
    c[0] + c[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_sums_every_tap() {
        let x: Vec<f32> = (1..=20).map(|v| v as f32).collect();
        let h = vec![1.0f32; 20];
        assert_eq!(dot(&x, &h), 210.0);
    }

    #[test]
    fn a_short_row_stops_the_sum() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(dot(&x, &[1.0, 1.0]), 3.0);
    }

    #[test]
    fn lanes_fold_in_ffmpeg_order() {
        // Lanes 1 and 5 each hold 2^-24. Folded together first, they reach
        // 2^-23, which survives the add into 1. A sequential or adjacent-pair
        // sum rounds each 2^-24 away against 1.
        let tiny = 2.0f32.powi(-24);
        let mut x = [0.0f32; 8];
        x[0] = 1.0;
        x[1] = tiny;
        x[5] = tiny;
        assert_eq!(dot(&x, &[1.0; 8]), 1.0 + 2.0 * tiny);
    }

    #[test]
    fn between_rows_interpolates() {
        let x = [2.0f32; 8];
        let lo = [0.0f32; 8];
        let hi = [1.0f32; 8];
        assert_eq!(dot_between(&x, &lo, &hi, 0.25), 4.0);
        assert_eq!(dot_between(&x, &lo, &hi, 0.0), 0.0);
    }
}
