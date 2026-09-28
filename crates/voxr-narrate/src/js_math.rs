//! JavaScript's `Math.log10`, `Math.exp` and `Math.pow` as V8 computes
//! them, so a port that takes a logarithm or an exponential per sample
//! gives the same doubles.
//!
//! V8 up to 14.x (Node 26) computes `Math.log` and `Math.log10` with the
//! fdlibm algorithms (`e_log.c`, `e_log10.c`) and `Math.exp` with FreeBSD's
//! `e_exp.c`. The platform C library rounds some results to the other
//! neighbouring double, so [`js_log10`] and [`js_exp`] reproduce the fdlibm
//! steps. `Math.pow` and Rust's `powf` both call the platform `pow`, so
//! [`js_pow`] calls `powf`. Later V8 releases compute these with LLVM's
//! correctly rounded libm, which can differ in the last bit.

// The constants are the fdlibm ones, written as their bit patterns.

/// 2^54, which scales a subnormal input into the normal range.
const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
/// ln 2 split in two: `k * LN2_HI` is exact for every exponent `k`.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
/// The coefficients of the `log(1 + f)` remainder polynomial.
const LG: [f64; 7] = [
    f64::from_bits(0x3fe5_5555_5555_5593),
    f64::from_bits(0x3fd9_9999_9997_fa04),
    f64::from_bits(0x3fd2_4924_9422_9359),
    f64::from_bits(0x3fcc_71c5_1d8e_78af),
    f64::from_bits(0x3fc7_4664_96cb_03de),
    f64::from_bits(0x3fc3_9a09_d078_c69f),
    f64::from_bits(0x3fc2_f112_df3e_5244),
];
/// 1/3, rounded down, in the short `log(1 + f)` series.
const THIRD: f64 = f64::from_bits(0x3fd5_5555_5555_5555);
/// 1 / ln 10.
const IVLN10: f64 = f64::from_bits(0x3fdb_cb7b_1526_e50e);
/// log10(2) split in two.
const LOG10_2HI: f64 = f64::from_bits(0x3fd3_4413_509f_6000);
const LOG10_2LO: f64 = f64::from_bits(0x3d59_fef3_11f1_2b36);
/// 1 / ln 2.
const INVLN2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
/// The coefficients of the `exp` remainder polynomial.
const P: [f64; 5] = [
    f64::from_bits(0x3fc5_5555_5555_553e),
    f64::from_bits(0xbf66_c16c_16be_bd93),
    f64::from_bits(0x3f11_566a_af25_de2c),
    f64::from_bits(0xbebb_bd41_c5d2_6bf1),
    f64::from_bits(0x3e66_3769_72be_a4d0),
];
/// Above this, `exp` overflows: about 709.78.
const EXP_OVER: f64 = f64::from_bits(0x4086_2e42_fefa_39ef);
/// Below this, `exp` underflows to 0: about -745.13.
const EXP_UNDER: f64 = f64::from_bits(0xc087_4910_d52d_3051);
/// 2^-1000, which scales a subnormal `exp` result.
const TWO_M1000: f64 = f64::from_bits(0x0170_0000_0000_0000);
/// 2^1023.
const TWO_1023: f64 = f64::from_bits(0x7fe0_0000_0000_0000);

/// The high 32 bits of `x`, signed.
fn high(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}

/// The low 32 bits of `x`.
fn low(x: f64) -> u32 {
    x.to_bits() as u32
}

/// `x` with its high 32 bits replaced by `h`.
fn with_high(x: f64, h: i32) -> f64 {
    f64::from_bits((u64::from(h as u32) << 32) | u64::from(low(x)))
}

/// `Math.log(x)`: fdlibm's natural logarithm.
fn js_log(mut x: f64) -> f64 {
    let mut hx = high(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        if (hx & 0x7fff_ffff) as u32 | low(x) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = high(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    // Normalizes x into [sqrt(2)/2, sqrt(2)) by halving it when needed.
    let i = (hx + 0x95f64) & 0x0010_0000;
    x = with_high(x, hx | (i ^ 0x3ff0_0000));
    k += i >> 20;
    let f = x - 1.0;
    let dk = f64::from(k);
    if (0x000f_ffff & (2 + hx)) < 3 {
        // |f| < 2^-20.
        if f == 0.0 {
            return if k == 0 {
                0.0
            } else {
                dk * LN2_HI + dk * LN2_LO
            };
        }
        let r = f * f * (0.5 - THIRD * f);
        return if k == 0 {
            f - r
        } else {
            dk * LN2_HI - ((r - dk * LN2_LO) - f)
        };
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG[1] + w * (LG[3] + w * LG[5]));
    let t2 = z * (LG[0] + w * (LG[2] + w * (LG[4] + w * LG[6])));
    let r = t2 + t1;
    if (hx - 0x6147a) | (0x6b851 - hx) > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

/// `Math.log10(x)`: fdlibm's base-10 logarithm, `log(m) / ln 10` plus the
/// exponent times log10(2), each split in two parts.
pub(crate) fn js_log10(mut x: f64) -> f64 {
    let mut hx = high(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        if (hx & 0x7fff_ffff) as u32 | low(x) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = high(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    if hx == 0x3ff0_0000 && low(x) == 0 {
        return 0.0;
    }
    k += (hx >> 20) - 1023;
    // 1 when the exponent is negative: the mantissa is then taken in [0.5, 1).
    let i = ((k as u32) >> 31) as i32;
    hx = (hx & 0x000f_ffff) | ((0x3ff - i) << 20);
    let y = f64::from(k + i);
    x = with_high(x, hx);
    let z = y * LOG10_2LO + IVLN10 * js_log(x);
    z + y * LOG10_2HI
}

/// `Math.exp(x)`: FreeBSD's exponential, `2^k * exp(r)` with `r` within
/// half of ln 2 of 0.
pub(crate) fn js_exp(mut x: f64) -> f64 {
    let bits = high(x) as u32;
    let negative = bits >> 31 == 1;
    let hx = bits & 0x7fff_ffff;
    if hx >= 0x4086_2e42 {
        // |x| >= 709.78.
        if hx >= 0x7ff0_0000 {
            if (hx & 0xf_ffff) | low(x) != 0 {
                return x + x;
            }
            return if negative { 0.0 } else { x };
        }
        if x > EXP_OVER {
            return f64::INFINITY;
        }
        if x < EXP_UNDER {
            return 0.0;
        }
    }
    let (ln2_hi, ln2_lo, half) = if negative {
        (-LN2_HI, -LN2_LO, -0.5)
    } else {
        (LN2_HI, LN2_LO, 0.5)
    };
    let mut hi = 0.0;
    let mut lo = 0.0;
    let mut k: i32 = 0;
    if hx > 0x3fd6_2e42 {
        // |x| > ln(2) / 2.
        if hx < 0x3ff0_a2b2 {
            // |x| < 1.5 ln(2).
            if x == 1.0 {
                return std::f64::consts::E;
            }
            hi = x - ln2_hi;
            lo = ln2_lo;
            k = if negative { -1 } else { 1 };
        } else {
            k = (INVLN2 * x + half) as i32;
            let t = f64::from(k);
            hi = x - t * LN2_HI;
            lo = t * LN2_LO;
        }
        x = hi - lo;
    } else if hx < 0x3e30_0000 {
        // |x| < 2^-28.
        return 1.0 + x;
    }
    let t = x * x;
    let two_pk = if k >= -1021 {
        f64::from_bits(u64::from((0x3ff0_0000 + (k << 20)) as u32) << 32)
    } else {
        f64::from_bits(u64::from((0x3ff0_0000 + ((k + 1000) << 20)) as u32) << 32)
    };
    let c = x - t * (P[0] + t * (P[1] + t * (P[2] + t * (P[3] + t * P[4]))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * TWO_1023;
        }
        y * two_pk
    } else {
        y * two_pk * TWO_M1000
    }
}

/// `Math.pow(x, y)`, the `**` operator: the platform `pow`, which V8 and
/// Rust's `powf` both call.
pub(crate) fn js_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10_matches_javascript() {
        // Math.log10 of each, as V8 prints it.
        assert_eq!(js_log10(1000.0), 3.0);
        assert_eq!(js_log10(0.001), -3.0);
        assert_eq!(js_log10(1e-9), -9.0);
        assert_eq!(js_log10(2.0), std::f64::consts::LOG10_2);
        assert_eq!(js_log10(5e-324), -323.3062153431158);
        // The platform log10 gives the neighbouring double on these.
        assert_eq!(js_log10(0.139), -0.8569851997459048);
        assert_eq!(js_log10(0.146), -0.835647144215563);
        assert_eq!(js_log10(1.0), 0.0);
        assert_eq!(js_log10(0.0), f64::NEG_INFINITY);
        assert_eq!(js_log10(f64::INFINITY), f64::INFINITY);
        assert!(js_log10(-1.0).is_nan());
        assert!(js_log10(f64::NAN).is_nan());
    }

    #[test]
    fn exp_matches_javascript() {
        assert_eq!(js_exp(1.0), std::f64::consts::E);
        assert_eq!(js_exp(-1.0), 0.36787944117144233);
        assert_eq!(js_exp(0.5), 1.6487212707001282);
        assert_eq!(js_exp(-4.1666666666666665e-5), 0.9999583342013768);
        assert_eq!(js_exp(700.0), 1.0142320547350045e304);
        assert_eq!(js_exp(-740.0), 4.2e-322);
        assert_eq!(js_exp(1e-30), 1.0);
        // The platform exp gives the neighbouring double on these.
        assert_eq!(js_exp(-0.052), 0.9493288668428896);
        assert_eq!(js_exp(-0.086), 0.917594231220151);
        assert_eq!(js_exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(js_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(js_exp(710.0), f64::INFINITY);
        assert_eq!(js_exp(-746.0), 0.0);
        assert!(js_exp(f64::NAN).is_nan());
    }

    #[test]
    fn pow_matches_javascript() {
        assert_eq!(js_pow(10.0, -0.4), 0.3981071705534972);
        assert_eq!(js_pow(10.0, 3.6 / 20.0), 1.5135612484362082);
    }
}
