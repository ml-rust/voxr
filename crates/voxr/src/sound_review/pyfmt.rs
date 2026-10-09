//! Python's text forms of the numbers and paths the review prints:
//! `repr(float)`, `round(float, n)`, NumPy's `np.round(x, n)`, `repr(str)`
//! and `str(pathlib.Path)`.

/// `repr(x)` for a Python float: the shortest digits that read back as
/// `x`, positional for decimal exponents from -4 to 15 (always with a
/// fractional part), scientific otherwise (`1e-05`, `1.5e+16`).
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            let (int, frac) = digits.split_at(point as usize);
            format!("{int}.{frac}")
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1.min(digits.len()));
        let frac = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{first}{frac}e{esign}{:02}", exp.abs())
    }
}

/// Python's `round(x, ndigits)` for a float: the decimal value of `x`
/// correctly rounded to `ndigits` places, ties to even, read back.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let text = format!("{x:.ndigits$}");
    text.parse().unwrap_or(x)
}

/// NumPy's `np.round(x, decimals)` for a float64: `rint(x · 10^d) / 10^d`.
pub fn np_round(x: f64, decimals: i32) -> f64 {
    let f = 10f64.powi(decimals);
    (x * f).round_ties_even() / f
}

/// `repr(s)` for a Python str: single quotes, unless `s` holds a single
/// quote and no double quote.
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `str(pathlib.Path(p))` on POSIX: empty and `.` parts dropped, repeated
/// slashes joined, no trailing slash. A leading `//` stays, as POSIX
/// leaves it implementation-defined. The empty path is `.`.
pub fn py_path_str(p: &str) -> String {
    let root = if p.starts_with("//") && !p.starts_with("///") {
        "//"
    } else if p.starts_with('/') {
        "/"
    } else {
        ""
    };
    let parts: Vec<&str> = p
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    let joined = format!("{root}{}", parts.join("/"));
    if joined.is_empty() {
        ".".to_string()
    } else {
        joined
    }
}

/// `str(Path(root) / name)`.
pub fn py_path_join(root: &str, name: &str) -> String {
    let base = py_path_str(root);
    if base == "." {
        py_path_str(name)
    } else if base.ends_with('/') {
        format!("{base}{name}")
    } else {
        format!("{base}/{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.9, "0.9"),
            (2.42, "2.42"),
            (-76.4, "-76.4"),
            (-240.0, "-240.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (1.5e-7, "1.5e-07"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn rounding_follows_python_and_numpy() {
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(0.375, 2), 0.38);
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(np_round(-0.04, 1), -0.0);
        assert!(np_round(-0.04, 1).is_sign_negative());
        assert_eq!(np_round(0.25, 1), 0.2);
        assert_eq!(np_round(-76.44, 1), -76.4);
    }

    #[test]
    fn str_and_path_forms_match_python() {
        assert_eq!(py_str_repr("/a/b.wav"), "'/a/b.wav'");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_path_join("/tmp/x/", "sounds.json"), "/tmp/x/sounds.json");
        assert_eq!(py_path_join("./x//y/.", "a.wav"), "x/y/a.wav");
        assert_eq!(py_path_join(".", "a.wav"), "a.wav");
        assert_eq!(py_path_join("/", "a.wav"), "/a.wav");
        assert_eq!(py_path_str(""), ".");
    }
}
