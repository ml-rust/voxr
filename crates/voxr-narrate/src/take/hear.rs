//! A take heard back and checked against its text, through the default or
//! the small speech recognizer, or the one the user's config names.

use crate::check::{HeardWord, StrayPause, Verdict, check_speech, stray_pauses};
use crate::error::{Error, Result};
use crate::pattern::{starts_with_en, utf16_len, utf16_prefix};
use crate::transcript::{TranscribeOptions, Transcribed, Transcriber, transcribe_narration};

/// The heard words are clipped to this many UTF-16 code units.
const HEARD_CLIP: usize = 140;

/// The skill's default recognizer: `voice.mjs`'s `DEFAULT_MODEL`.
pub const DEFAULT_ASR_MODEL: &str = "onnx-community/whisper-base_timestamped";

/// The skill's small recognizer: `voice.mjs`'s `SMALL_MODEL`.
pub const SMALL_ASR_MODEL: &str = "onnx-community/whisper-small_timestamped";

/// Which speech recognizer hears a take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recognizer {
    /// The default recognizer: [`DEFAULT_ASR_MODEL`].
    Default,
    /// The small recognizer: [`SMALL_ASR_MODEL`]. It is larger than the
    /// default one, and hears a language other than English.
    Small,
    /// The model the user's config names as `asr.model`, when it is neither
    /// of the two. The caller resolves the id and supplies the recognizer.
    Configured,
}

impl Recognizer {
    /// The recognizer a second opinion comes from: the small one for any
    /// recognizer but itself, whose second opinion is the default one.
    pub fn other(self) -> Self {
        match self {
            Self::Default | Self::Configured => Self::Small,
            Self::Small => Self::Default,
        }
    }
}

/// The recognizer heard first: the skill's `asrModel(language)`.
///
/// `configured`, the config's `asr.model`, wins when it is set: the default
/// or the small recognizer when it names one of their ids, else
/// [`Recognizer::Configured`]. An empty id counts as set, as JavaScript's
/// `??` reads it. Without it, [`Recognizer::Small`] hears a language not
/// starting with `en` (any case), and [`Recognizer::Default`] every other.
pub fn asr_model(language: Option<&str>, configured: Option<&str>) -> Recognizer {
    match configured {
        Some(SMALL_ASR_MODEL) => Recognizer::Small,
        Some(DEFAULT_ASR_MODEL) => Recognizer::Default,
        Some(_) => Recognizer::Configured,
        None => match language {
            Some(l) if !l.is_empty() && !starts_with_en(l) => Recognizer::Small,
            _ => Recognizer::Default,
        },
    }
}

/// The error when a take asks for the configured recognizer and none was
/// supplied.
pub(crate) fn no_configured_recognizer() -> Error {
    Error::Recognizer {
        reason: "the config names an asr.model, and no recognizer for it was supplied. Load the configured model and pass it to the hearing".to_string(),
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

impl<H: Hearing + ?Sized> Hearing for &mut H {
    fn hear(
        &mut self,
        take: &[f32],
        rate: u32,
        recognizer: Recognizer,
        language: Option<&str>,
    ) -> Result<Transcribed> {
        (**self).hear(take, rate, recognizer, language)
    }
}

/// `pcm` at 16 kHz read into timed words by `transcriber` through
/// [`transcribe_narration`], in `language`: what every [`Hearing`] over a
/// [`Transcriber`] does once the take is at 16 kHz.
pub fn hear_16k(
    pcm: &[f32],
    transcriber: &mut dyn Transcriber,
    language: Option<&str>,
) -> Result<Transcribed> {
    let opts = TranscribeOptions {
        language: language.map(str::to_string),
    };
    transcribe_narration(pcm, transcriber, &opts)
}

/// [`Hearing`] through [`Transcriber`]s and [`transcribe_narration`].
/// `to_16k` resamples a take to the 16 kHz the transcribers read.
pub struct TranscriberHearing<'a, F> {
    /// The default recognizer.
    pub default: &'a mut dyn Transcriber,
    /// The small recognizer.
    pub small: &'a mut dyn Transcriber,
    /// The recognizer the config's `asr.model` names. A take asking for it
    /// when it is `None` is refused.
    pub configured: Option<&'a mut dyn Transcriber>,
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
            Recognizer::Configured => match self.configured.as_deref_mut() {
                Some(t) => t,
                None => return Err(no_configured_recognizer()),
            },
        };
        hear_16k(&pcm, transcriber, language)
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
    if utf16_len(text) > n {
        let mut out = utf16_prefix(text, n);
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
        pub(crate) configured: Vec<Transcribed>,
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
                Recognizer::Configured => &mut self.configured,
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
            configured: Vec::new(),
            calls: Vec::new(),
        }
    }

    #[test]
    fn asr_model_picks_small_for_other_languages() {
        assert_eq!(asr_model(None, None), Recognizer::Default);
        assert_eq!(asr_model(Some(""), None), Recognizer::Default);
        assert_eq!(asr_model(Some("en"), None), Recognizer::Default);
        assert_eq!(asr_model(Some("EN-gb"), None), Recognizer::Default);
        assert_eq!(asr_model(Some("ms"), None), Recognizer::Small);
        assert_eq!(asr_model(Some("日本"), None), Recognizer::Small);
    }

    #[test]
    fn a_configured_model_wins_over_the_language() {
        let id = Some("openai/whisper-large-v3");
        assert_eq!(asr_model(Some("ms"), id), Recognizer::Configured);
        assert_eq!(asr_model(None, id), Recognizer::Configured);
        assert_eq!(
            asr_model(Some("ms"), Some(DEFAULT_ASR_MODEL)),
            Recognizer::Default
        );
        assert_eq!(
            asr_model(Some("en"), Some(SMALL_ASR_MODEL)),
            Recognizer::Small
        );
        assert_eq!(asr_model(Some("en"), Some("")), Recognizer::Configured);
        assert_eq!(Recognizer::Configured.other(), Recognizer::Small);
    }

    #[test]
    fn a_configured_recognizer_hears_first_and_the_small_one_checks_it() -> Result<()> {
        // A loop retries with the small recognizer, as the skill's
        // `model !== SMALL_MODEL` does.
        let mut h = hearing(&[], &[("one two three", false)]);
        h.configured.push(heard("one", true));
        let c = heard_check(
            &mut h,
            Recognizer::Configured,
            &[0.0; 8],
            48000,
            "one two three",
            None,
        )?;
        assert!(c.check.ok);
        assert_eq!(h.calls, [Recognizer::Configured, Recognizer::Small]);
        Ok(())
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
