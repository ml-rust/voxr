//! Recognizer loops collapsed: a recognizer can repeat a phrase ("dengan
//! kembali dengan kembali …") that was said once.

use crate::check::HeardWord;
use crate::check::speech_match::norm;
use crate::error::Result;
use crate::pattern::Patterns;

/// The longest repeated run of words looked for.
const MAX_RUN: usize = 4;
/// A run repeated this many times in a row, or more, is a loop.
const MIN_COPIES: usize = 3;

/// Words with every loop collapsed, and whether any was found.
#[derive(Debug, Clone, PartialEq)]
pub struct Collapsed {
    /// The words, each loop down to its first copy.
    pub words: Vec<HeardWord>,
    /// A loop was collapsed.
    pub found: bool,
}

/// A run of 1 to 4 words repeated three or more times in a row keeps its
/// first copy.
///
/// Words compare by their normalized pieces, so case and punctuation do not
/// matter. Runs of 1 word are collapsed first, then 2, up to 4, each pass
/// reading the words the one before left. Two copies are kept.
pub fn collapse_loops(words: &[HeardWord]) -> Result<Collapsed> {
    let p = Patterns::get()?;
    let mut out = words.to_vec();
    let mut keys: Vec<String> = words.iter().map(|w| norm(p, &w.text).join(" ")).collect();
    let mut found = false;
    for n in 1..=MAX_RUN {
        let same = |keys: &[String], a: usize, b: usize| keys[a..a + n] == keys[b..b + n];
        let mut i = 0;
        while i + MIN_COPIES * n <= out.len() {
            if same(&keys, i, i + n) && same(&keys, i, i + 2 * n) {
                let mut end = i + n;
                while end + n <= out.len() && same(&keys, i, end) {
                    end += n;
                }
                out.drain(i + n..end);
                keys.drain(i + n..end);
                found = true;
            }
            i += 1;
        }
    }
    Ok(Collapsed { words: out, found })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(texts: &[&str]) -> Vec<HeardWord> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| HeardWord {
                text: (*t).to_string(),
                start: i as f64,
                end: i as f64 + 0.5,
            })
            .collect()
    }

    fn texts(c: &Collapsed) -> Vec<&str> {
        c.words.iter().map(|w| w.text.as_str()).collect()
    }

    #[test]
    fn three_copies_collapse() -> Result<()> {
        let c = collapse_loops(&heard(&[
            "a", "dengan", "kembali", "Dengan,", "kembali", "dengan", "kembali.", "b",
        ]))?;
        assert!(c.found);
        assert_eq!(texts(&c), ["a", "dengan", "kembali", "b"]);
        assert_eq!(c.words[3].start, 7.0);
        let c = collapse_loops(&heard(&["x", "x", "x", "x", "y"]))?;
        assert_eq!(texts(&c), ["x", "y"]);
        Ok(())
    }

    #[test]
    fn two_copies_stay() -> Result<()> {
        let words = heard(&["rumah", "rumah", "kita", "ke", "sana", "ke", "sana"]);
        let c = collapse_loops(&words)?;
        assert!(!c.found);
        assert_eq!(c.words, words);
        Ok(())
    }
}
