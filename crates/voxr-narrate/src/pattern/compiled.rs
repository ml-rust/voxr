//! The module's regular expressions, compiled once.
//!
//! The patterns spell out what JavaScript means where regexr differs: `\s` is
//! [`WS`](super::js::WS), `\d` is `[0-9]`, `^`/`$` are `\A`/`\z`, and a
//! case-insensitive word is an ASCII class per letter ([`ci`](super::js::ci)).

use std::sync::OnceLock;

use regexr::Regex;

use super::js::{WS, ci, replace_all_with};
use crate::error::{Error, Result};

/// Every pattern the module runs, compiled on first use.
pub(crate) struct Patterns {
    /// `\s+`
    pub(crate) spaces: Regex,
    /// `\n\s*\n`: a blank line, between beats and between subtitle blocks.
    pub(crate) blank_line: Regex,
    /// `[ \t]+`
    pub(crate) blanks: Regex,
    /// ` ?\n ?`
    pub(crate) padded_newline: Regex,
    /// `TAGS`: VoxCPM2's non-verbal tags.
    pub(crate) tags: Regex,
    /// `LEAD`: a delivery note opening the text. Group 1 is the note; its
    /// 2 to 120 UTF-16 unit bound is checked by the caller.
    pub(crate) lead: Regex,
    /// `LEAD` with the `m` flag: a delivery note opening any line.
    pub(crate) lead_lines: Regex,
    /// `[pause]` or `[pause 0.6]`; group 1 is the value.
    pub(crate) pause: Regex,
    /// `SAID`: `{written|spoken}`.
    pub(crate) said: Regex,
    /// One sentence of whitespace-collapsed text.
    pub(crate) sentence: Regex,
    /// `[^\p{L}\p{N}]+`: what a word key drops.
    pub(crate) not_letters: Regex,
    /// `.srt` or `.vtt` at the end of a file name.
    pub(crate) subtitle_file: Regex,
    /// `<i>` and `{\an8}` subtitle markup.
    pub(crate) markup: Regex,
    /// `STAMP`: a time, an optional range end, a separator, then the text.
    pub(crate) stamp: Regex,
    /// A unit after a time, removed before the time is read.
    pub(crate) unit_suffix: Regex,
    /// A colon or a unit in a time: `/[:s]|sec|saat/i`.
    pub(crate) time_marked: Regex,
    /// `^\s*[[(]`
    pub(crate) opens_bracket: Regex,
    /// `^[\s\d.,]+[|\-–—.):]`: a number then a separator.
    pub(crate) number_then_separator: Regex,
    /// `[^\p{L}\p{N}\s-]`: what a number token drops.
    pub(crate) not_token: Regex,
    /// `[\s-]+`: what separates number tokens.
    pub(crate) token_break: Regex,
    /// `\d`
    pub(crate) digit: Regex,
    /// A number written as digits, commas grouping thousands.
    pub(crate) written_number: Regex,
    /// `(\d)\s+([.,])\s*(\d)`: a decimal a recognizer spaced out.
    pub(crate) spaced_decimal: Regex,
    /// `[^\p{L}\p{N}\s%]`: what a heard or script word drops.
    pub(crate) not_word: Regex,
    /// `(\d)(\p{L}|%)`: a unit written onto its number.
    pub(crate) digit_unit: Regex,
    /// `^\s*(-\p{L}|,\d)`: a recognizer fragment of the word before.
    pub(crate) fragment: Regex,
    /// `^(.*\d)\s*(\p{L}+|%)([.,!?]*)$`: a number, its unit, then end
    /// punctuation.
    pub(crate) number_unit: Regex,
    /// `[\p{L}\p{N}%]`: a script token that holds a word.
    pub(crate) word_char: Regex,
    /// `[,.;:!?…—–-]["'”’)\]]*$`: a word ending in a break.
    pub(crate) break_end: Regex,
    /// `TRANSITIONS`: an opening word a speaker pauses after.
    pub(crate) transitions: Regex,
    /// `JOINS`: a joining word a speaker breathes before.
    pub(crate) joins: Regex,
    /// A token that is a flag, file name or address (`--out`, `a.b_c`,
    /// `https:`, `.txt`).
    pub(crate) flag_or_address: Regex,
    /// `\p{L}`: any letter, anywhere in the text.
    pub(crate) letter: Regex,
    /// `\A[0-9][0-9.,]*\z`: a token that is only digits, dots and commas.
    pub(crate) plain_digits: Regex,
    /// `[^.!?…]+[.!?…]*`: one sentence of a review line, over its whole
    /// text (`find_iter`, not `split`).
    pub(crate) review_sentence: Regex,
    /// `\?["'”’)]*\z`: a sentence ending in a question mark.
    pub(crate) question_end: Regex,
    /// `[,;:—–…]`: a breath mark, anywhere in the text.
    pub(crate) soft_break: Regex,
    /// `[.!?…:;,"'”’)\]]\z`: a sentence ending in punctuation the voice
    /// reads as a stop.
    pub(crate) end_punct: Regex,
    /// `\A[ab][fm]_[a-z]+\z`: a Kokoro voice name.
    pub(crate) kokoro_voice: Regex,
}

static PATTERNS: OnceLock<std::result::Result<Patterns, String>> = OnceLock::new();

fn compile(name: &str, pattern: &str) -> std::result::Result<Regex, String> {
    Regex::new(pattern)
        .map_err(|e| format!("narration pattern `{name}` ({pattern}) does not compile: {e}"))
}

impl Patterns {
    /// The process-wide patterns, compiled on first use.
    pub(crate) fn get() -> Result<&'static Self> {
        PATTERNS
            .get_or_init(Self::build)
            .as_ref()
            .map_err(|reason| Error::Pattern {
                reason: reason.clone(),
            })
    }

    fn build() -> std::result::Result<Self, String> {
        let tags = format!(
            r"\[(?:{}|{}|{}|{}|{}(?:{}|{}|{}|{})|{}(?:{}|{})|{})\]",
            ci("laughing"),
            ci("sigh"),
            ci("uhm"),
            ci("shh"),
            ci("question-"),
            ci("ah"),
            ci("ei"),
            ci("en"),
            ci("oh"),
            ci("surprise-"),
            ci("wa"),
            ci("yo"),
            ci("dissatisfaction-hnn"),
        );
        let note = r"[ \t]*\(([^()\n]+)\)[ \t]*";
        let units = format!(
            "{}|{}|{}|{}?|{}",
            ci("s"),
            ci("sec"),
            ci("secs"),
            ci("seconds"),
            ci("saat")
        );
        let time = format!(r"[0-9]+(?::[0-9]{{1,2}}){{0,2}}(?:[.,][0-9]+)?[{WS}]*(?:{units})?");
        let stamp = format!(
            r"\A[{WS}]*[\[(]?[{WS}]*({time})[{WS}]*(?:(?:-|–|—|{}|{}|{})[{WS}]*({time}))?[{WS}]*[\])]?[{WS}]*(?:[:|\-–—.)][{WS}]*|[{WS}]+)([^\n\r\x{{2028}}\x{{2029}}]*)\z",
            ci("to"),
            ci("hingga"),
            ci("sampai"),
        );
        Ok(Self {
            spaces: compile("spaces", &format!("[{WS}]+"))?,
            blank_line: compile("blank_line", &format!(r"\n[{WS}]*\n"))?,
            blanks: compile("blanks", r"[ \t]+")?,
            padded_newline: compile("padded_newline", r" ?\n ?")?,
            tags: compile("tags", &tags)?,
            lead: compile("lead", &format!(r"\A{note}"))?,
            lead_lines: compile(
                "lead_lines",
                &format!(r"(?:\A|(?<=[\n\r\x{{2028}}\x{{2029}}])){note}"),
            )?,
            pause: compile(
                "pause",
                &format!(r"\[{}(?:[{WS}]+([0-9.]+))?\]", ci("pause")),
            )?,
            said: compile("said", r"\{([^{}|\n]+)\|([^{}\n]+)\}")?,
            sentence: compile(
                "sentence",
                &format!(r#".+?(?:[.!?…]+["')\]」』]*(?=[{WS}]|\z)|[。！？]+[」』]*|\z)"#),
            )?,
            not_letters: compile("not_letters", r"[^\p{L}\p{N}]+")?,
            subtitle_file: compile(
                "subtitle_file",
                &format!(r"\.(?:{}|{})\z", ci("srt"), ci("vtt")),
            )?,
            markup: compile("markup", r"<[^>]+>|\{[^}]*\}")?,
            stamp: compile("stamp", &stamp)?,
            unit_suffix: compile("unit_suffix", &format!(r"[{WS}]*(?:{units})\z"))?,
            time_marked: compile(
                "time_marked",
                &format!("[:sS]|{}|{}", ci("sec"), ci("saat")),
            )?,
            opens_bracket: compile("opens_bracket", &format!(r"\A[{WS}]*[\[(]"))?,
            number_then_separator: compile(
                "number_then_separator",
                &format!(r"\A[{WS}0-9.,]+[|\-–—.):]"),
            )?,
            not_token: compile("not_token", &format!(r"[^\p{{L}}\p{{N}}{WS}\-]"))?,
            token_break: compile("token_break", &format!(r"[{WS}\-]+"))?,
            digit: compile("digit", "[0-9]")?,
            written_number: compile(
                "written_number",
                r"[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
            )?,
            spaced_decimal: compile(
                "spaced_decimal",
                &format!(r"([0-9])[{WS}]+([.,])[{WS}]*([0-9])"),
            )?,
            not_word: compile("not_word", &format!(r"[^\p{{L}}\p{{N}}{WS}%]"))?,
            digit_unit: compile("digit_unit", r"([0-9])(\p{L}|%)")?,
            fragment: compile("fragment", &format!(r"\A[{WS}]*(?:-\p{{L}}|,[0-9])"))?,
            // JavaScript's `.` stops at the four line terminators.
            number_unit: compile(
                "number_unit",
                &format!(r"\A([^\n\r\x{{2028}}\x{{2029}}]*[0-9])[{WS}]*(\p{{L}}+|%)([.,!?]*)\z"),
            )?,
            word_char: compile("word_char", r"[\p{L}\p{N}%]")?,
            break_end: compile("break_end", r#"[,.;:!?…—–\-]["'”’)\]]*\z"#)?,
            transitions: compile(
                "transitions",
                &format!(
                    r"\A(?:{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{})\z",
                    ci("then"),
                    ci("now"),
                    ci("first"),
                    ci("next"),
                    ci("so"),
                    ci("here"),
                    ci("finally"),
                    ci("also"),
                    ci("still"),
                    ci("today"),
                    ci("instead"),
                    ci("meanwhile"),
                    ci("later"),
                    ci("afterwards"),
                    ci("otherwise"),
                    ci("however"),
                    ci("yes"),
                    ci("okay"),
                    ci("well"),
                ),
            )?,
            joins: compile(
                "joins",
                &format!(
                    r"\A(?:{}|{}|{}|{}|{}|{}|{}|{}|{})\z",
                    ci("and"),
                    ci("but"),
                    ci("so"),
                    ci("then"),
                    ci("because"),
                    ci("while"),
                    ci("which"),
                    ci("until"),
                    ci("or"),
                ),
            )?,
            flag_or_address: compile(
                "flag_or_address",
                &format!(
                    r"\A-{{1,2}}[0-9A-Za-z_]|[0-9A-Za-z_][./_@\\][0-9A-Za-z_]|\A{}{}?:|\.[0-9A-Za-z_]{{2,4}}\z",
                    ci("http"),
                    ci("s"),
                ),
            )?,
            letter: compile("letter", r"\p{L}")?,
            plain_digits: compile("plain_digits", r"\A[0-9][0-9.,]*\z")?,
            review_sentence: compile("review_sentence", r"[^.!?…]+[.!?…]*")?,
            question_end: compile("question_end", r#"\?["'”’)]*\z"#)?,
            soft_break: compile("soft_break", "[,;:—–…]")?,
            end_punct: compile("end_punct", r#"[.!?…:;,"'”’)\]]\z"#)?,
            kokoro_voice: compile("kokoro_voice", r"\A[ab][fm]_[a-z]+\z")?,
        })
    }
}

/// `text.replace(/\s+/g, ' ')`
pub(crate) fn collapse(p: &Patterns, text: &str) -> String {
    replace_all_with(&p.spaces, text, " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    use super::super::js::{is_space, split};

    #[test]
    fn space_class_agrees_with_predicate() -> Result<()> {
        let p = Patterns::get()?;
        for cp in 0..=0xFFFFu32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let s = c.to_string();
            assert_eq!(p.spaces.is_match(&s), is_space(c), "U+{cp:04X}");
        }
        Ok(())
    }

    #[test]
    fn ascii_case_only() -> Result<()> {
        let p = Patterns::get()?;
        assert!(p.pause.is_match("[PAUSE]"));
        assert!(p.pause.is_match("[Pause 1.2]"));
        // JavaScript's `/i` without `u` does not fold U+017F to `s`.
        assert!(!p.pause.is_match("[pau\u{17F}e]"));
        assert!(!p.tags.is_match("[\u{17F}igh]"));
        Ok(())
    }

    #[test]
    fn split_keeps_empty_pieces() -> Result<()> {
        let p = Patterns::get()?;
        assert_eq!(split(&p.spaces, " a  b "), vec!["", "a", "b", ""]);
        assert_eq!(split(&p.spaces, ""), vec![""]);
        Ok(())
    }
}
