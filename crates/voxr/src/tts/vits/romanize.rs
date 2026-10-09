//! uroman romanization for MMS-TTS checkpoints whose `tokenizer_config.json`
//! sets `is_uroman: true`.
//!
//! The MMS-TTS recipe romanizes the raw text with
//! `uroman.Uroman().romanize_string(text)` before `VitsTokenizer` sees it.
//! The recipe passes no language code, so neither does this module.
//! `VitsTokenizer` then romanizes once more, after lowercasing, when the
//! text still holds a non-ASCII character and the `uroman` package is
//! installed. [`romanize`] is the first step, and `romanize_non_roman` the
//! second.

use uroman::{Uroman, rom_format};

/// Romanizes `text` with uroman and no language code: the MMS-TTS recipe's
/// step before `VitsTokenizer`.
pub fn romanize(text: &str) -> String {
    Uroman::new()
        .romanize_string::<rom_format::Str>(text, None)
        .to_string()
}

/// The tokenizer's own step: romanizes `text` only when it holds a
/// character outside ASCII, as `has_non_roman_characters` decides.
pub(super) fn romanize_non_roman(text: String) -> String {
    if text.is_ascii() {
        text
    } else {
        romanize(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn romanizes_cyrillic_and_keeps_case() {
        assert_eq!(romanize("Привет"), "Privet");
    }

    #[test]
    fn leaves_ascii_text_unchanged() {
        assert_eq!(romanize("hello, world 42"), "hello, world 42");
        assert_eq!(
            romanize_non_roman("hello, world 42".to_string()),
            "hello, world 42"
        );
    }

    #[test]
    fn empty_text_stays_empty() {
        assert_eq!(romanize(""), "");
    }

    #[test]
    fn second_step_runs_only_on_non_ascii_text() {
        assert_eq!(romanize_non_roman("мир".to_string()), "mir");
    }
}
