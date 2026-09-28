//! A take heard back and checked against its text, through one of two
//! speech recognizers.

use crate::check::{HeardWord, StrayPause, Verdict, check_speech, stray_pauses};
use crate::error::Result;
use crate::pattern::starts_with_en;
use crate::transcript::{TranscribeOptions, Transcribed, Transcriber, transcribe_narration};

/// The heard words are clipped to this many UTF-16 code units.
const HEARD_CLIP: usize = 140;

/// Which of the two speech recognizers hears a take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recognizer {
    /// The default recognizer.
    Default,
    /// The small recognizer. It is larger than the default one, and hears
    /// a language other than English.
    Small,
}

impl Recognizer {
    /// The recognizer a second opinion comes from.
    pub fn other(self) -> Self {
        match self {
            Self::Default => Self::Small,
            Self::Small => Self::Default,
        }
    }
}

/// The recognizer for `language` when no config names one:
/// [`Recognizer::Small`] for a language not starting with `en` (any case),
/// else [`Recognizer::Default`].
pub fn asr_model(language: Option<&str>) -> Recognizer {
    match language {
        Some(l) if !l.is_empty() && !starts_with_en(l) => Recognizer::Small,
        _ => Recognizer::Default,
    }
}

/// Reads a take into timed words with one of the two recognizers.
pub trait Hearing {
    /// The words `recognizer` hears in `take`, at `rate` samples per
    /// second.
    fn hear(
        &mut self,
        take: &[f32],
        rate: u32,
        recognizer: Recognizer,
        language: Option<&str>,
    ) -> Result<Transcribed>;
}

/// [`Hearing`] through two [`Transcriber`]s and
/// [`transcribe_narration`]. `to_16k` resamples a take to the 16 kHz the
/// transcribers read.
pub struct TranscriberHearing<'a, F> {
    /// The default recognizer.
    pub default: &'a mut dyn Transcriber,
    /// The small recognizer.
    pub small: &'a mut dyn Transcriber,
    /// Resamples `(samples, rate)` to 16 kHz.
    pub to_16k: F,
}

impl<F> Hearing for TranscriberHearing<'_, F>
where
    F: FnMut(&[f32], u32) -> Result<Vec<f32>>,
{
    fn hear(
        &mut self,
        take: &[f32],
        rate: u32,
        recognizer: Recognizer,
        language: Option<&str>,
    ) -> Result<Transcribed> {
        let pcm = (self.to_16k)(take, rate)?;
        let transcriber: &mut dyn Transcriber = match recognizer {
            Recognizer::Default => &mut *self.default,
            Recognizer::Small => &mut *self.small,
        };
        let opts = TranscribeOptions {
            language: language.map(str::to_string),
        };
        transcribe_narration(&pcm, transcriber, &opts)
    }
}

/// A take heard back and checked against its text.
#[derive(Debug, Clone, PartialEq)]
pub struct HeardCheck {
    /// The speech check.
    pub check: Verdict,
    /// Words both recognizers miss or mishear in a passing take: likely
    /// dropped or said wrong, and marked for a listen.
    pub doubt: Vec<String>,
    /// The words heard.
    pub words: Vec<HeardWord>,
    /// Pauses the text does not ask for.
    pub pauses: Vec<StrayPause>,
    /// The words heard as one line, clipped to 140 UTF-16 code units and
    /// an ellipsis.
    pub heard: String,
}

/// `t.length > n ? t.slice(0, n) + '…' : t`, over UTF-16 code units.
fn clip(text: &str, n: usize) -> String {
    let units: Vec<u16> = text.encode_utf16().collect();
    if units.len() > n {
        let mut out = String::from_utf16_lossy(&units[..n]);
        out.push('…');
        out
    } else {
        text.to_string()
    }
}

/// `[...x.unsure, ...x.missing]`: a check's doubtful words. A check with
/// nothing recognized has no unsure words.
fn off(check: &Verdict) -> Vec<&str> {
    let unsure = check.unsure.iter().flatten();
    unsure.chain(&check.missing).map(String::as_str).collect()
}

/// Hears `take` with `main` and checks it against `text`.
///
/// A failing take the recognizer looped on is heard again by
/// [`Recognizer::Small`], unless `main` is that one. A passing take with
/// words misheard or missing is heard by `main`'s other recognizer, and a
/// word off in both checks goes into `doubt`, once, in the first check's
/// order.
pub fn heard_check<H: Hearing + ?Sized>(
    hearing: &mut H,
    main: Recognizer,
    take: &[f32],
    rate: u32,
    text: &str,
    language: Option<&str>,
) -> Result<HeardCheck> {
    let mut t = hearing.hear(take, rate, main, language)?;
    let mut check = check_speech(&t.words, text)?;
    if !check.ok && t.looped && main != Recognizer::Small {
        t = hearing.hear(take, rate, Recognizer::Small, language)?;
        check = check_speech(&t.words, text)?;
    }
    let mut doubt: Vec<String> = Vec::new();
    let unsure = check.unsure.as_ref().is_some_and(|u| !u.is_empty());
    if check.ok && (unsure || !check.missing.is_empty()) {
        let second = hearing.hear(take, rate, main.other(), language)?;
        let check2 = check_speech(&second.words, text)?;
        let off2 = off(&check2);
        for w in off(&check) {
            if off2.contains(&w) && !doubt.iter().any(|d| d == w) {
                doubt.push(w.to_string());
            }
        }
    }
    let pauses = stray_pauses(&t.words, text)?;
    let line: Vec<&str> = t.words.iter().map(|w| w.text.as_str()).collect();
    Ok(HeardCheck {
        check,
        doubt,
        pauses,
        heard: clip(&line.join(" "), HEARD_CLIP),
        words: t.words,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Hears scripted words per recognizer, and records each call.
    pub(crate) struct ScriptedHearing {
        pub(crate) default: Vec<Transcribed>,
        pub(crate) small: Vec<Transcribed>,
        pub(crate) calls: Vec<Recognizer>,
    }

    impl Hearing for ScriptedHearing {
        fn hear(
            &mut self,
            _take: &[f32],
            _rate: u32,
            recognizer: Recognizer,
            _language: Option<&str>,
        ) -> Result<Transcribed> {
            self.calls.push(recognizer);
            let list = match recognizer {
                Recognizer::Default => &mut self.default,
                Recognizer::Small => &mut self.small,
            };
            Ok(if list.is_empty() {
                heard("", false)
            } else {
                list.remove(0)
            })
        }
    }

    pub(crate) fn heard(line: &str, looped: bool) -> Transcribed {
        let words = line
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| HeardWord {
                text: w.to_string(),
                start: i as f64 * 0.5,
                end: i as f64 * 0.5 + 0.4,
            })
            .collect();
        Transcribed {
            words,
            looped,
            duration: 5.0,
        }
    }

    pub(crate) fn hearing(default: &[(&str, bool)], small: &[(&str, bool)]) -> ScriptedHearing {
        ScriptedHearing {
            default: default.iter().map(|&(l, x)| heard(l, x)).collect(),
            small: small.iter().map(|&(l, x)| heard(l, x)).collect(),
            calls: Vec::new(),
        }
    }

    #[test]
    fn asr_model_picks_small_for_other_languages() {
        assert_eq!(asr_model(None), Recognizer::Default);
        assert_eq!(asr_model(Some("")), Recognizer::Default);
        assert_eq!(asr_model(Some("en")), Recognizer::Default);
        assert_eq!(asr_model(Some("EN-gb")), Recognizer::Default);
        assert_eq!(asr_model(Some("ms")), Recognizer::Small);
        assert_eq!(asr_model(Some("日本")), Recognizer::Small);
    }

    #[test]
    fn clip_counts_utf16_units() {
        assert_eq!(clip("abc", 3), "abc");
        assert_eq!(clip("abcd", 3), "abc…");
        assert_eq!(clip("a🎙b", 2), "a\u{FFFD}…");
    }

    #[test]
    fn a_clean_pass_hears_once() -> Result<()> {
        let mut h = hearing(&[("one two three", false)], &[]);
        let c = heard_check(
            &mut h,
            Recognizer::Default,
            &[0.0; 8],
            48000,
            "one two three",
            None,
        )?;
        assert!(c.check.ok);
        assert!(c.doubt.is_empty());
        assert_eq!(c.heard, "one two three");
        assert_eq!(h.calls, [Recognizer::Default]);
        Ok(())
    }

    #[test]
    fn a_looped_failure_is_heard_again_by_the_small_recognizer() -> Result<()> {
        let mut h = hearing(&[("one", true)], &[("one two three", false)]);
        let c = heard_check(
            &mut h,
            Recognizer::Default,
            &[0.0; 8],
            48000,
            "one two three",
            None,
        )?;
        assert!(c.check.ok);
        assert_eq!(h.calls, [Recognizer::Default, Recognizer::Small]);
        Ok(())
    }

    #[test]
    fn a_word_both_recognizers_mishear_is_doubted() -> Result<()> {
        let text = "the river runs north today";
        let mut h = hearing(
            &[("the river walks north today", false)],
            &[("the river walks north today", false)],
        );
        let c = heard_check(&mut h, Recognizer::Default, &[0.0; 8], 48000, text, None)?;
        assert!(c.check.ok, "{:?}", c.check);
        assert_eq!(h.calls, [Recognizer::Default, Recognizer::Small]);
        assert_eq!(c.doubt, ["runs"]);
        Ok(())
    }

    #[test]
    fn a_word_one_recognizer_hears_right_is_not_doubted() -> Result<()> {
        let text = "the river runs north today";
        let mut h = hearing(
            &[("the river walks north today", false)],
            &[("the river runs north today", false)],
        );
        let c = heard_check(&mut h, Recognizer::Default, &[0.0; 8], 48000, text, None)?;
        assert!(c.doubt.is_empty());
        Ok(())
    }
}
