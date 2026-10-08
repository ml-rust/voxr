//! Text preparation before tokenizing: the whitespace collapse VoxCPM's
//! `generate` applies, and the [`paren`] style prefix.

/// Whether `ch` is whitespace to Python's `re` `\s` on a `str`: every
/// character `str.isspace()` accepts.
///
/// That is Rust's `char::is_whitespace` plus the four ASCII separators
/// U+001C to U+001F, which Python counts as whitespace and Unicode does not.
pub fn is_python_whitespace(ch: char) -> bool {
    ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch)
}

/// Collapse every run of whitespace in `text` to one ASCII space, as
/// VoxCPM's `generate` does: `re.sub(r"\s+", " ", text)`.
///
/// The ends are kept: a leading or trailing run becomes one space, and the
/// tokenizer reads that space. Whitespace is [`is_python_whitespace`], so a
/// tab, a newline or a non-breaking space collapses the same way.
pub fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_run = false;
    for ch in text.chars() {
        if is_python_whitespace(ch) {
            if !in_run {
                out.push(' ');
                in_run = true;
            }
        } else {
            out.push(ch);
            in_run = false;
        }
    }
    out
}

/// Whether `text` holds anything to speak: VoxCPM refuses a text whose
/// `strip()` is empty.
pub fn is_blank(text: &str) -> bool {
    text.chars().all(is_python_whitespace)
}

/// Wrap a style or voice-design prompt in parentheses, the form VoxCPM2
/// reads as a delivery instruction rather than as text to speak.
///
/// Trims `prompt` first. An empty prompt stays empty, and a prompt that
/// already starts with `(` is returned trimmed and otherwise unchanged. A
/// caller prefixes the result to the text: `paren(style) + text`.
pub fn paren(prompt: &str) -> String {
    let prompt = prompt.trim_matches(is_python_whitespace);
    if prompt.is_empty() || prompt.starts_with('(') {
        prompt.to_string()
    } else {
        format!("({prompt})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_interior_runs_and_keeps_one_space_at_each_end() {
        assert_eq!(normalize_whitespace("  a   b  "), " a b ");
        assert_eq!(normalize_whitespace("a b c"), "a b c");
    }

    #[test]
    fn collapses_mixed_whitespace_kinds_to_one_space() {
        assert_eq!(normalize_whitespace("a\t\n\u{00A0}b"), "a b");
        assert_eq!(normalize_whitespace("a\u{1f}\u{3000}b"), "a b");
    }

    #[test]
    fn does_not_alter_punctuation_or_non_latin_text() {
        let text = "Hello, world! \u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}\u{ff01}";
        assert_eq!(normalize_whitespace(text), text);
    }

    #[test]
    fn blank_text_is_whitespace_only() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\n\u{1c}"));
        assert!(!is_blank(" a "));
    }

    #[test]
    fn paren_wraps_a_bare_prompt() {
        assert_eq!(paren("relaxed, explaining"), "(relaxed, explaining)");
    }

    #[test]
    fn paren_trims_before_wrapping() {
        assert_eq!(paren("  A calm male narrator \n"), "(A calm male narrator)");
    }

    #[test]
    fn paren_leaves_a_wrapped_prompt_alone_after_trimming() {
        assert_eq!(paren(" (warm) "), "(warm)");
        // Only the opening parenthesis is checked.
        assert_eq!(paren("(warm"), "(warm");
    }

    #[test]
    fn paren_keeps_an_empty_prompt_empty() {
        assert_eq!(paren(""), "");
        assert_eq!(paren(" \t "), "");
    }
}
