//! Whisper as the take loop's [`Hearing`]: the default and the small
//! recognizer, and the one the config's `asr.model` names, each loaded
//! already or loaded the first time a take needs it, and a take resampled
//! to 16 kHz as the skill's ffmpeg decode does.

use super::audio::to_16k;
use crate::error::{Error, Result};
use crate::take::hear::no_configured_recognizer;
use crate::take::{Hearing, Recognizer, hear_16k};
use crate::transcript::{Transcribed, Transcriber};

/// Loads a recognizer the first time a take needs it.
type Loader<'a> = Box<dyn FnMut() -> Result<Box<dyn Transcriber + 'a>> + 'a>;

/// Where one recognizer of a [`WhisperHearing`] comes from.
pub enum RecognizerSource<'a> {
    /// A recognizer the caller loaded, such as a
    /// [`WhisperWords`](crate::WhisperWords) over a bundle it holds.
    Loaded(Box<dyn Transcriber + 'a>),
    /// A loader run the first time a take needs this recognizer. A load
    /// that fails is tried again on the next take.
    Lazy(Loader<'a>),
}

impl<'a> RecognizerSource<'a> {
    /// A recognizer already loaded.
    pub fn loaded(transcriber: impl Transcriber + 'a) -> Self {
        Self::Loaded(Box::new(transcriber))
    }

    /// A recognizer `load` loads the first time a take needs it.
    pub fn lazy<T, F>(mut load: F) -> Self
    where
        T: Transcriber + 'a,
        F: FnMut() -> Result<T> + 'a,
    {
        Self::Lazy(Box::new(move || {
            load().map(|t| Box::new(t) as Box<dyn Transcriber + 'a>)
        }))
    }

    /// The recognizer, loaded now when it is lazy. `name` names it in a
    /// load error.
    fn get(&mut self, name: &str) -> Result<&mut (dyn Transcriber + 'a)> {
        if let Self::Lazy(load) = self {
            let loaded = load().map_err(|e| Error::Recognizer {
                reason: format!(
                    "loading the {name} Whisper recognizer: {e}. Check its checkpoint, or pass it loaded"
                ),
            })?;
            *self = Self::Loaded(loaded);
        }
        match self {
            Self::Loaded(t) => Ok(t.as_mut()),
            Self::Lazy(_) => Err(Error::Recognizer {
                reason: format!("the {name} Whisper recognizer did not load. Pass it loaded"),
            }),
        }
    }
}

/// [`Hearing`] through Whisper recognizers: Whisper base as
/// [`Recognizer::Default`] and Whisper small as [`Recognizer::Small`], as the
/// skill hears with `whisper-base` and `whisper-small`, and the model the
/// config's `asr.model` names as [`Recognizer::Configured`].
///
/// A take is resampled to 16 kHz with [`to_16k`], then read through
/// [`hear_16k`]: what [`TranscriberHearing`](crate::TranscriberHearing)
/// gives over the same recognizers with [`to_16k`] as its resampler.
pub struct WhisperHearing<'a> {
    default: RecognizerSource<'a>,
    small: RecognizerSource<'a>,
    configured: Option<RecognizerSource<'a>>,
}

impl<'a> WhisperHearing<'a> {
    /// Hears with `default` and `small`. A take asking for the configured
    /// recognizer is refused until [`with_configured`](Self::with_configured)
    /// supplies one.
    pub fn new(default: RecognizerSource<'a>, small: RecognizerSource<'a>) -> Self {
        Self {
            default,
            small,
            configured: None,
        }
    }

    /// The same hearing with `configured` as the recognizer the config's
    /// `asr.model` names: the caller resolves the id and loads it, now or
    /// lazily.
    pub fn with_configured(self, configured: RecognizerSource<'a>) -> Self {
        Self {
            configured: Some(configured),
            ..self
        }
    }
}

impl Hearing for WhisperHearing<'_> {
    fn hear(
        &mut self,
        take: &[f32],
        rate: u32,
        recognizer: Recognizer,
        language: Option<&str>,
    ) -> Result<Transcribed> {
        let pcm = to_16k(take, rate)?;
        let transcriber = match recognizer {
            Recognizer::Default => self.default.get("default")?,
            Recognizer::Small => self.small.get("small")?,
            Recognizer::Configured => match self.configured.as_mut() {
                Some(source) => source.get("configured")?,
                None => return Err(no_configured_recognizer()),
            },
        };
        hear_16k(&pcm, transcriber, language)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::check::HeardWord;

    /// Hears one fixed word for the whole piece, and counts its reads.
    struct Fixed<'c> {
        word: &'static str,
        reads: &'c Cell<usize>,
    }

    impl Transcriber for Fixed<'_> {
        fn transcribe_words(
            &mut self,
            samples: &[f32],
            sample_rate: u32,
            _language: Option<&str>,
        ) -> Result<Vec<HeardWord>> {
            self.reads.set(self.reads.get() + 1);
            Ok(vec![HeardWord {
                text: self.word.to_string(),
                start: 0.0,
                end: samples.len() as f64 / f64::from(sample_rate),
            }])
        }
    }

    fn tone() -> Vec<f32> {
        (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.5).collect()
    }

    #[test]
    fn the_small_recognizer_loads_once_and_only_when_asked() -> Result<()> {
        let (reads, loads) = (Cell::new(0), Cell::new(0));
        let mut hearing = WhisperHearing::new(
            RecognizerSource::loaded(Fixed {
                word: "base",
                reads: &reads,
            }),
            RecognizerSource::lazy(|| {
                loads.set(loads.get() + 1);
                Ok(Fixed {
                    word: "small",
                    reads: &reads,
                })
            }),
        );
        let take = tone();
        let heard = hearing.hear(&take, 48_000, Recognizer::Default, None)?;
        assert_eq!(heard.words[0].text, "base");
        assert_eq!(loads.get(), 0);
        for _ in 0..2 {
            let heard = hearing.hear(&take, 48_000, Recognizer::Small, Some("ms"))?;
            assert_eq!(heard.words[0].text, "small");
        }
        assert_eq!(loads.get(), 1);
        assert_eq!(reads.get(), 3);
        Ok(())
    }

    #[test]
    fn the_configured_recognizer_hears_once_supplied() -> Result<()> {
        let reads = Cell::new(0);
        let fixed = |word| Fixed {
            word,
            reads: &reads,
        };
        let mut hearing = WhisperHearing::new(
            RecognizerSource::loaded(fixed("base")),
            RecognizerSource::loaded(fixed("small")),
        );
        let take = tone();
        let none = hearing
            .hear(&take, 48_000, Recognizer::Configured, None)
            .err();
        assert!(none.is_some_and(|e| e.to_string().contains("asr.model")));
        let mut hearing = hearing.with_configured(RecognizerSource::lazy(|| Ok(fixed("large"))));
        let heard = hearing.hear(&take, 48_000, Recognizer::Configured, None)?;
        assert_eq!(heard.words[0].text, "large");
        Ok(())
    }

    #[test]
    fn a_failed_load_names_the_recognizer_and_is_tried_again() {
        let tries = Cell::new(0);
        let reads = Cell::new(0);
        let mut hearing = WhisperHearing::new(
            RecognizerSource::loaded(Fixed {
                word: "base",
                reads: &reads,
            }),
            RecognizerSource::lazy(|| -> Result<Fixed<'_>> {
                tries.set(tries.get() + 1);
                Err(Error::Recognizer {
                    reason: "no checkpoint".to_string(),
                })
            }),
        );
        let take = tone();
        for _ in 0..2 {
            let err = hearing
                .hear(&take, 48_000, Recognizer::Small, None)
                .err()
                .map(|e| e.to_string());
            assert!(err.is_some_and(|m| m.contains("small Whisper recognizer")));
        }
        assert_eq!(tries.get(), 2);
    }
}
