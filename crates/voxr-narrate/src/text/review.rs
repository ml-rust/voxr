//! A script read the way a voice will read it, before any audio: what a
//! voice is likely to misread or run together.
//!
//! Judgment the rules cannot make (a word that reads as a noun or a verb)
//! stays with the writer. `notes: true` also flags a question with no
//! delivery note: VoxCPM2 voices delivery, so a question without a note of
//! its own reads in the take's delivery and can come out as a statement.

use serde::Serialize;

use super::tags::{is_note, say, without_tags};
use crate::error::Result;
use crate::pattern::{Patterns, group, replace_all_with, split, starts_with_en, trim};

/// One note [`review`] returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReviewNote {
    /// The script line the note is about, from 1.
    pub line: usize,
    /// The words the note is about.
    pub text: String,
    /// What a voice is likely to do with `text`.
    pub note: String,
}

/// [`review`]'s options.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReviewOptions<'a> {
    /// The script's language, such as `"en"` or `"ms"`. `None` reads as
    /// English.
    pub language: Option<&'a str>,
    /// The engine voices delivery notes (VoxCPM2), so a question without
    /// its own note is flagged.
    pub notes: bool,
}

/// Leading characters a token opens with and never says: quotes, brackets.
const OPEN_CHARS: &[char] = &['"', '\'', '“', '‘', '(', '['];

/// Trailing characters a token ends with and never says: quotes, brackets,
/// end punctuation.
const CLOSE_CHARS: &[char] = &[
    '"', '\'', '”', '’', ')', ']', '.', ',', ';', ':', '!', '?', '…',
];

/// Characters a speaker takes as a breath mark: a comma, a semicolon, a
/// colon, a dash or an ellipsis.
const BREATH_CHARS: &[char] = &[',', ';', ':', '—', '–', '…'];

/// The token a voice actually reads: quotes, brackets and end punctuation
/// stripped from both ends.
fn bare_token(token: &str) -> &str {
    token
        .trim_start_matches(OPEN_CHARS)
        .trim_end_matches(CLOSE_CHARS)
}

/// A script's language reads as English: none given, empty, or one
/// starting `en` (case-insensitive).
fn is_english(language: Option<&str>) -> bool {
    language.is_none_or(|l| l.is_empty() || starts_with_en(l))
}

/// The delivery note opening `raw`, and the line with it removed. `raw`
/// holds no note when its leading parentheses hold too short or too long a
/// note: the regex bound is loose (Unicode scalar values), and [`is_note`]
/// applies the exact UTF-16 bound the skill's `LEAD` regex reads by.
fn strip_lead<'a>(p: &Patterns, raw: &'a str) -> (&'a str, bool) {
    match p
        .lead
        .captures(raw)
        .and_then(|caps| Some((group(&caps, 0)?, group(&caps, 1)?)))
    {
        Some((whole, note)) if is_note(note) => (&raw[whole.len()..], true),
        _ => (raw, false),
    }
}

/// The words of one line the voice actually says: pause marks removed,
/// `{written|spoken}` pairs replaced by their spoken form, tags removed.
fn said_words(p: &Patterns, line: &str) -> Result<String> {
    let no_pause = replace_all_with(&p.pause, line, " ");
    without_tags(&say(&no_pause)?)
}

/// One line's sentences, split the way [`review`] reads them: a run of
/// text with no `.`, `!`, `?` or `…`, followed by any run of those.
fn review_sentences<'t>(p: &'t Patterns, text: &'t str) -> Vec<&'t str> {
    p.review_sentence
        .find_iter(text)
        .map(|m| m.as_str())
        .collect()
}

/// A token flagged for its own kind of misread. `None` when none applies.
fn token_note(p: &Patterns, token: &str, english: bool) -> Option<String> {
    let t = bare_token(token);
    if t.is_empty() {
        return None;
    }
    if p.flag_or_address.is_match(t) {
        Some(format!(
            "a flag, file name or address: write how it is said, {{{t}|…}}"
        ))
    } else if p.digit.is_match(t) && p.letter.is_match(t) {
        Some(format!(
            "letters and digits together: write how it is said, {{{t}|…}}"
        ))
    } else if !english && p.plain_digits.is_match(t) {
        Some("digits outside English: write the number as words".to_string())
    } else {
        None
    }
}

/// A sentence's note about a missing breath: no breath after an opening
/// word, or a long run with none at all. `None` when neither applies.
fn sentence_note(p: &Patterns, s: &str) -> Option<(String, String)> {
    let trimmed = trim(s);
    let words: Vec<&str> = split(&p.spaces, trimmed)
        .into_iter()
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return None;
    }
    let opens_after = !words[0]
        .chars()
        .last()
        .is_some_and(|c| BREATH_CHARS.contains(&c));
    if words.len() > 3
        && p.transitions.is_match(words[0])
        && opens_after
        && !p.question_end.is_match(trimmed)
    {
        let clip = words.iter().take(4).copied().collect::<Vec<_>>().join(" ");
        return Some((
            format!("{clip}…"),
            format!("no breath after \"{}\": \"{}, …\"", words[0], words[0]),
        ));
    }
    if words.len() >= 9 && !p.soft_break.is_match(s) {
        let joint = words
            .iter()
            .enumerate()
            .position(|(i, w)| i >= 2 && i < words.len() - 2 && p.joins.is_match(w));
        let clip = words.iter().take(6).copied().collect::<Vec<_>>().join(" ");
        let where_it_sets = match joint {
            Some(i) => format!(", or before \"{}\",", words[i]),
            None => ", or before a joined phrase,".to_string(),
        };
        return Some((
            format!("{clip}…"),
            format!(
                "{} words with no breath: the voice picks its own pauses. A comma after the subject{where_it_sets} sets one",
                words.len()
            ),
        ));
    }
    None
}

/// A question in `said` with no delivery note of its own, or one sharing a
/// line with other sentences. `None` when `notes` review does not apply.
fn notes_mode_issue(p: &Patterns, said: &str, noted: bool) -> Option<(String, &'static str)> {
    let parts: Vec<&str> = review_sentences(p, said)
        .into_iter()
        .map(trim)
        .filter(|s| !s.is_empty())
        .collect();
    let asks: Vec<&str> = parts
        .iter()
        .copied()
        .filter(|s| p.question_end.is_match(s))
        .collect();
    if asks.is_empty() {
        return None;
    }
    if !noted {
        return Some((
            asks[0].to_string(),
            "a question with no delivery note: VoxCPM2 reads it in the take's delivery, and it can come out as a statement. Give it its own line with a note (tts/voxcpm2.md)",
        ));
    }
    if parts.len() > asks.len() {
        return Some((
            asks[0].to_string(),
            "a question shares its line with other sentences: its note covers them too. Give the question a line of its own",
        ));
    }
    None
}

/// A script read the way a voice will read it, before any audio: what a
/// voice is likely to misread or run together. The skill's `review`.
///
/// Flags a token that needs `{written|spoken}` (a flag, a file name or
/// address, letters and digits together, or digits outside English), a
/// sentence with no breath after an opening word or before a joined phrase,
/// a run of 9 or more words with no comma, and, with `options.notes`, a
/// question with no delivery note. Also flags a line with no end
/// punctuation.
pub fn review(script: &str, options: &ReviewOptions<'_>) -> Result<Vec<ReviewNote>> {
    let p = Patterns::get()?;
    let english = is_english(options.language);
    let mut out = Vec::new();
    for (n, raw) in script.replace('\r', "").split('\n').enumerate() {
        let (stripped, noted) = strip_lead(p, raw);
        let line = trim(stripped);
        if line.is_empty() {
            continue;
        }
        let line_no = n + 1;
        let said = said_words(p, line)?;
        let bare = replace_all_with(&p.said, line, " ");
        for token in split(&p.spaces, &bare) {
            if let Some(note) = token_note(p, token, english) {
                out.push(ReviewNote {
                    line: line_no,
                    text: bare_token(token).to_string(),
                    note,
                });
            }
        }
        for s in review_sentences(p, &said) {
            if let Some((text, note)) = sentence_note(p, s) {
                out.push(ReviewNote {
                    line: line_no,
                    text,
                    note,
                });
            }
        }
        if options.notes
            && let Some((text, note)) = notes_mode_issue(p, &said, noted)
        {
            out.push(ReviewNote {
                line: line_no,
                text,
                note: note.to_string(),
            });
        }
        let trimmed_said = trim(&said);
        if p.letter.is_match(&said) && !p.end_punct.is_match(trimmed_said) {
            let words: Vec<&str> = split(&p.spaces, trimmed_said)
                .into_iter()
                .filter(|w| !w.is_empty())
                .collect();
            let tail = words[words.len().saturating_sub(3)..].join(" ");
            out.push(ReviewNote {
                line: line_no,
                text: tail,
                note: "no end punctuation: the voice runs into the next line".to_string(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(language: Option<&str>, notes: bool) -> ReviewOptions<'_> {
        ReviewOptions { language, notes }
    }

    #[test]
    fn flags_addresses_and_files() -> Result<()> {
        let notes = review("Run --out build.tar.gz now.", &opts(None, false))?;
        assert!(
            notes
                .iter()
                .any(|n| n.text == "--out" && n.note.contains("a flag, file name or address"))
        );
        assert!(
            notes.iter().any(
                |n| n.text == "build.tar.gz" && n.note.contains("a flag, file name or address")
            )
        );
        Ok(())
    }

    #[test]
    fn flags_letters_and_digits_together() -> Result<()> {
        let notes = review("It reached v2 today.", &opts(None, false))?;
        assert!(
            notes
                .iter()
                .any(|n| n.text == "v2" && n.note.contains("letters and digits together"))
        );
        Ok(())
    }

    #[test]
    fn flags_bare_digits_outside_english() -> Result<()> {
        let notes = review("Ada 42 orang.", &opts(Some("ms"), false))?;
        assert!(notes.iter().any(
            |n| n.text == "42" && n.note == "digits outside English: write the number as words"
        ));
        let none = review("Ada 42 orang.", &opts(Some("en"), false))?;
        assert!(!none.iter().any(|n| n.text == "42"));
        Ok(())
    }

    #[test]
    fn flags_missing_breath_after_an_opening_word() -> Result<()> {
        let notes = review("Then it gathers facts and moves on.", &opts(None, false))?;
        assert!(
            notes
                .iter()
                .any(|n| n.note.contains("no breath after \"Then\""))
        );
        let fine = review("Then, it gathers facts and moves on.", &opts(None, false))?;
        assert!(!fine.iter().any(|n| n.note.contains("no breath after")));
        Ok(())
    }

    #[test]
    fn flags_a_long_run_with_no_comma() -> Result<()> {
        let notes = review(
            "The team built the whole system from scratch over one long weekend.",
            &opts(None, false),
        )?;
        assert!(
            notes
                .iter()
                .any(|n| n.note.contains("words with no breath"))
        );
        Ok(())
    }

    #[test]
    fn flags_unnoted_questions_only_with_notes_mode() -> Result<()> {
        let with_notes = review("Where does the water go?", &opts(None, true))?;
        assert!(
            with_notes
                .iter()
                .any(|n| n.note.contains("a question with no delivery note"))
        );
        let without_notes = review("Where does the water go?", &opts(None, false))?;
        assert!(!without_notes.iter().any(|n| n.note.contains("question")));
        let noted = review("(curious) Where does the water go?", &opts(None, true))?;
        assert!(!noted.iter().any(|n| n.note.contains("no delivery note")));
        Ok(())
    }

    #[test]
    fn flags_a_question_sharing_its_line() -> Result<()> {
        let notes = review(
            "(curious) Where does it go? It just does.",
            &opts(None, true),
        )?;
        assert!(notes.iter().any(|n| n.note.contains("shares its line")));
        Ok(())
    }

    #[test]
    fn flags_missing_end_punctuation() -> Result<()> {
        let notes = review("The team built the system", &opts(None, false))?;
        assert!(notes.iter().any(|n| n.note.contains("no end punctuation")));
        let fine = review("The team built the system.", &opts(None, false))?;
        assert!(!fine.iter().any(|n| n.note.contains("no end punctuation")));
        Ok(())
    }

    #[test]
    fn blank_lines_are_skipped() -> Result<()> {
        let notes = review("\n\n   \n", &opts(None, false))?;
        assert!(notes.is_empty());
        Ok(())
    }
}
