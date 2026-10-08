//! JavaScript string semantics the module runs under: the `\s` set, `trim`,
//! UTF-16 lengths, `split` and `Number(string)`.
//!
//! `\s` is [`WS`], and a case-insensitive word is an ASCII class per letter
//! ([`ci`]), since JavaScript's `/i` without `u` does not fold `ſ` to `s` the
//! way Unicode folding does.

use regexr::{Captures, Regex};

/// Members of JavaScript's `\s` class, to place inside a character class.
pub(crate) const WS: &str = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

/// A character in JavaScript's `\s` class. Agrees with [`WS`].
pub(crate) fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// `String.prototype.trim`.
pub(crate) fn trim(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `String.prototype.length`: UTF-16 code units.
pub(crate) fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `s.slice(0, n)`: the first `n` UTF-16 code units of `s`. A surrogate
/// pair cut in half leaves U+FFFD where JavaScript keeps a lone surrogate.
pub(crate) fn utf16_prefix(s: &str, n: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().take(n).collect();
    String::from_utf16_lossy(&units)
}

/// A language tag beginning `en`, case-insensitive over ASCII (`/^en/i`):
/// the prefix English speech and narration checks read by.
pub(crate) fn starts_with_en(s: &str) -> bool {
    s.get(..2).is_some_and(|p| p.eq_ignore_ascii_case("en"))
}

/// `word` matched case-insensitively over ASCII letters only, as JavaScript's
/// `/i` does for these words.
pub(crate) fn ci(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        if c.is_ascii_alphabetic() {
            out.push('[');
            out.push(c.to_ascii_lowercase());
            out.push(c.to_ascii_uppercase());
            out.push(']');
        } else {
            out.push_str(&regexr::escape(c.encode_utf8(&mut [0; 4])));
        }
    }
    out
}

/// `text.split(re)` for a pattern without capture groups and without empty
/// matches: the pieces between matches, empty ones included.
pub(crate) fn split<'t>(re: &Regex, text: &'t str) -> Vec<&'t str> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(text) {
        out.push(&text[last..m.start()]);
        last = m.end();
    }
    out.push(&text[last..]);
    out
}

/// `text.replace(re /g, fn)`: every match replaced by what `with` returns.
pub(crate) fn replace_all(
    re: &Regex,
    text: &str,
    mut with: impl FnMut(&Captures<'_>) -> String,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text) {
        let Some(m) = caps.get(0) else { continue };
        out.push_str(&text[last..m.start()]);
        out.push_str(&with(&caps));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

/// `text.replace(re /g, rep)` with a fixed replacement.
pub(crate) fn replace_all_with(re: &Regex, text: &str, rep: &str) -> String {
    replace_all(re, text, |_| rep.to_string())
}

/// Group `i` of a match, `None` when it did not take part.
pub(crate) fn group<'t>(caps: &Captures<'t>, i: usize) -> Option<&'t str> {
    caps.get(i).map(|m| m.as_str())
}

/// `Number(s)` for a string: JavaScript's StringToNumber. Whitespace around
/// is ignored, an empty string is 0, and anything else that is not a decimal,
/// `0x`/`0o`/`0b` or `Infinity` literal is NaN.
pub(crate) fn number(s: &str) -> f64 {
    let t = trim(s);
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = t.strip_prefix(prefix) {
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            // u128 rounds once to the nearest f64, as JavaScript does; a
            // longer literal accumulates in f64.
            return u128::from_str_radix(digits, radix).map_or_else(
                |_| {
                    digits.chars().fold(0.0, |v, c| {
                        v * f64::from(radix) + f64::from(c.to_digit(radix).unwrap_or(0))
                    })
                },
                |v| v as f64,
            );
        }
    }
    if is_decimal_literal(t) {
        t.parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// `[+-]? (digits [. digits?] | . digits) ([eE] [+-]? digits)?`. Rust's
/// float parser also takes `inf` and `nan`, which JavaScript does not.
fn is_decimal_literal(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let digits = |i: &mut usize| {
        let from = *i;
        while b.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i - from
    };
    let mut mantissa = digits(&mut i);
    if b.get(i) == Some(&b'.') {
        i += 1;
        mantissa += digits(&mut i);
    }
    if mantissa == 0 {
        return false;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return false;
        }
    }
    i == b.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_follows_javascript() {
        assert_eq!(number(""), 0.0);
        assert_eq!(number("  12 "), 12.0);
        assert_eq!(number("1.5"), 1.5);
        assert_eq!(number("5."), 5.0);
        assert_eq!(number(".5"), 0.5);
        assert_eq!(number("1e3"), 1000.0);
        assert_eq!(number("0x10"), 16.0);
        assert_eq!(number("-Infinity"), f64::NEG_INFINITY);
        assert!(number("1.2.3").is_nan());
        assert!(number(".").is_nan());
        assert!(number("inf").is_nan());
        assert!(number("nan").is_nan());
        assert!(number("12abc").is_nan());
    }

    #[test]
    fn utf16_prefix_counts_code_units() {
        assert_eq!(utf16_prefix("abc", 5), "abc");
        assert_eq!(utf16_prefix("abcd", 2), "ab");
        assert_eq!(utf16_prefix("a🎙b", 3), "a🎙");
        assert_eq!(utf16_prefix("a🎙b", 2), "a\u{FFFD}");
    }
}
