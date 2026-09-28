//! The in-place radix-2 FFT the de-esser runs, step for step as the
//! JavaScript runs it, so each spectrum matches it bit for bit: a
//! bit-reversal permutation, then per stage one `cos` and one `sin` of the
//! same angle and a twiddle advanced by complex multiplication.

use std::f64::consts::PI;

/// Transforms `re` and `im` in place. Their length must be a power of two,
/// and `im` at least as long as `re`. `inverse` runs the inverse transform
/// and scales it by `1 / n`.
pub(super) fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 2.0 } else { -2.0 };
    let mut len = 2;
    while len <= n {
        let ang = (sign * PI) / len as f64;
        let wr = ang.cos();
        let wi = ang.sin();
        let half = len / 2;
        for i in (0..n).step_by(len) {
            let mut cr = 1.0f64;
            let mut ci = 0.0f64;
            for k in 0..half {
                let a = i + k;
                let b = a + half;
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr);
            }
        }
        len <<= 1;
    }
    if inverse {
        let scale = n as f64;
        for v in re.iter_mut().chain(im.iter_mut().take(n)) {
            *v /= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_twiddles_match_javascript() {
        // Math.cos and Math.sin of each stage angle of a 1024-point
        // transform, as V8 computes them, forward then inverse.
        let js: [[u64; 4]; 10] = [
            [
                0xbff0000000000000,
                0xbca1a62633145c07,
                0xbff0000000000000,
                0x3ca1a62633145c07,
            ],
            [
                0x3c91a62633145c07,
                0xbff0000000000000,
                0x3c91a62633145c07,
                0x3ff0000000000000,
            ],
            [
                0x3fe6a09e667f3bcd,
                0xbfe6a09e667f3bcc,
                0x3fe6a09e667f3bcd,
                0x3fe6a09e667f3bcc,
            ],
            [
                0x3fed906bcf328d46,
                0xbfd87de2a6aea963,
                0x3fed906bcf328d46,
                0x3fd87de2a6aea963,
            ],
            [
                0x3fef6297cff75cb0,
                0xbfc8f8b83c69a60a,
                0x3fef6297cff75cb0,
                0x3fc8f8b83c69a60a,
            ],
            [
                0x3fefd88da3d12526,
                0xbfb917a6bc29b42c,
                0x3fefd88da3d12526,
                0x3fb917a6bc29b42c,
            ],
            [
                0x3feff621e3796d7e,
                0xbfa91f65f10dd814,
                0x3feff621e3796d7e,
                0x3fa91f65f10dd814,
            ],
            [
                0x3feffd886084cd0d,
                0xbf992155f7a3667e,
                0x3feffd886084cd0d,
                0x3f992155f7a3667e,
            ],
            [
                0x3fefff62169b92db,
                0xbf8921d1fcdec784,
                0x3fefff62169b92db,
                0x3f8921d1fcdec784,
            ],
            [
                0x3fefffd8858e8a92,
                0xbf7921f0fe670071,
                0x3fefffd8858e8a92,
                0x3f7921f0fe670071,
            ],
        ];
        let mut len = 2;
        for want in js {
            let ahead = (-2.0 * PI) / len as f64;
            let back = (2.0 * PI) / len as f64;
            let got = [ahead.cos(), ahead.sin(), back.cos(), back.sin()].map(f64::to_bits);
            assert_eq!(got, want, "len {len}");
            len <<= 1;
        }
    }

    #[test]
    fn an_impulse_has_a_flat_spectrum_and_comes_back() {
        let mut re = vec![0.0; 8];
        let mut im = vec![0.0; 8];
        re[0] = 1.0;
        fft(&mut re, &mut im, false);
        assert!(re.iter().all(|&v| v == 1.0));
        assert!(im.iter().all(|&v| v == 0.0));
        fft(&mut re, &mut im, true);
        assert_eq!(re[0], 1.0);
        assert!(re[1..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_cosine_lands_in_its_bins() {
        let n = 16;
        let mut re: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).cos())
            .collect();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im, false);
        for (k, &v) in re.iter().enumerate() {
            let want = if k == 2 || k == n - 2 { 8.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-12, "bin {k}: {v}");
        }
    }
}
