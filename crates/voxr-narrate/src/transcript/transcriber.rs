//! A narration take read into timed words: pieces cut in silences, each
//! read by a recognizer, and the words settled on the audio.

use super::envelope::{RATE, TRANSCRIPT_RATE, envelope};
use super::fit::fit_to_sound;
use super::loops::collapse_loops;
use super::onsets::{hand_back_tails, snap_ends, snap_onsets};
use super::pieces::pieces;
use crate::check::HeardWord;
use crate::error::Result;
use crate::js::{js_max, js_min, js_to_fixed};
use crate::js_math::js_pow;
use crate::pattern::trim;

/// The language a recognizer is told when none is given.
pub const DEFAULT_LANGUAGE: &str = "en";
/// A word whose whole span stays under this level, in dBFS, was never
/// spoken.
const NEVER_LOUD_DB: f64 = -50.0;
/// A word shorter than this, in seconds, has no length.
const NO_LENGTH: f64 = 0.06;
/// Decimals of every time a transcript holds.
const TIME_DECIMALS: usize = 3;

/// A speech recognizer that times each word it hears.
pub trait Transcriber {
    /// The words heard in `samples`, times in seconds from the first
    /// sample. A time the recognizer could not place is NaN: a start then
    /// reads as 0 and an end as the end of `samples`.
    fn transcribe_words(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Vec<HeardWord>>;
}

/// How a take is read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribeOptions {
    /// The language spoken. `None` tells the recognizer
    /// [`DEFAULT_LANGUAGE`].
    pub language: Option<String>,
}

/// A take read into timed words.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcribed {
    /// The words, times in seconds rounded to milliseconds.
    pub words: Vec<HeardWord>,
    /// The recognizer looped on a phrase and the loop was collapsed.
    pub looped: bool,
    /// The take's length in seconds, rounded to milliseconds.
    pub duration: f64,
}

/// Reads a 16 kHz mono take into timed words.
///
/// A recognizer drops or invents text over long audio, so each piece of at
/// most 20 s, cut in a silence ([`pieces`]), is read alone, with the
/// language named. Each word's times are offset by its piece's start, and
/// clamped to the take: no end before its start, nothing past the take's
/// end. Words with no text are skipped.
///
/// A recognizer invents words in digital silence (a "you" after the last
/// line): a word whose whole span stays under -50 dBFS is dropped. A word
/// with no length (under 60 ms) whose next word starts under 60 ms after
/// it is judged over both, the span [`snap_ends`] shares between them.
/// Loops are collapsed ([`collapse_loops`]). Then these settle on the
/// sound, in order:
/// - a start on the previous word's tail ([`hand_back_tails`]),
/// - starts after a pause ([`snap_onsets`]),
/// - ends ([`snap_ends`]),
/// - the rest ([`fit_to_sound`]).
///
/// Times and the length are rounded to milliseconds last.
pub fn transcribe_narration(
    samples_16k: &[f32],
    transcriber: &mut dyn Transcriber,
    opts: &TranscribeOptions,
) -> Result<Transcribed> {
    let pcm = samples_16k;
    let duration = pcm.len() as f64 / RATE;
    let env = envelope(pcm);
    let language = opts.language.as_deref().unwrap_or(DEFAULT_LANGUAGE);
    let mut words = Vec::new();
    for (a, b) in pieces(&env, pcm.len()) {
        let heard = transcriber.transcribe_words(&pcm[a..b], TRANSCRIPT_RATE, Some(language))?;
        let at = a as f64 / RATE;
        for c in heard {
            let text = trim(&c.text);
            if text.is_empty() {
                continue;
            }
            let s = at + if c.start.is_nan() { 0.0 } else { c.start };
            let e = at
                + if c.end.is_nan() {
                    (b - a) as f64 / RATE
                } else {
                    c.end
                };
            words.push(HeardWord {
                text: text.to_string(),
                start: js_min(s, duration),
                end: js_min(js_max(e, s), duration),
            });
        }
    }
    let quiet = js_pow(10.0, NEVER_LOUD_DB / 20.0);
    let spoken: Vec<bool> = (0..words.len())
        .map(|i| peak(pcm, &judged_span(&words, i)) > quiet)
        .collect();
    let words: Vec<HeardWord> = words
        .into_iter()
        .zip(spoken)
        .filter_map(|(w, spoken)| spoken.then_some(w))
        .collect();
    let looped = collapse_loops(&words)?;
    let mut words = looped.words;
    hand_back_tails(&mut words, &env)?;
    snap_onsets(&mut words, &env);
    snap_ends(&mut words, &env, duration);
    fit_to_sound(&mut words, &env, duration)?;
    for w in &mut words {
        w.start = js_to_fixed(w.start, TIME_DECIMALS);
        w.end = js_to_fixed(w.end, TIME_DECIMALS);
    }
    Ok(Transcribed {
        words,
        looped: looped.found,
        duration: js_to_fixed(duration, TIME_DECIMALS),
    })
}

/// The span the level of word `i` is judged over: its own, or with no
/// length and the next word starting under 60 ms after it, through the
/// next word's end.
fn judged_span(words: &[HeardWord], i: usize) -> HeardWord {
    let w = &words[i];
    match words.get(i + 1) {
        Some(n) if w.end - w.start < NO_LENGTH && n.start - w.start < NO_LENGTH => HeardWord {
            text: String::new(),
            start: w.start,
            end: js_max(w.end, n.end),
        },
        _ => w.clone(),
    }
}

/// The largest magnitude of the samples a word spans, from the sample its
/// start falls in up to the sample its end rounds up to. A span reaching
/// before the first sample has no level: NaN.
fn peak(pcm: &[f32], w: &HeardWord) -> f64 {
    let mut peak = 0.0;
    let from = (w.start * RATE).floor() as i64;
    let end = (pcm.len() as i64).min((w.end * RATE).ceil() as i64);
    for i in from..end {
        let v = usize::try_from(i)
            .ok()
            .and_then(|i| pcm.get(i))
            .map_or(f64::NAN, |&s| f64::from(s).abs());
        peak = js_max(peak, v);
    }
    peak
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns one scripted word list per call, and records each call.
    struct Scripted {
        pieces: Vec<Vec<HeardWord>>,
        calls: Vec<(usize, u32, Option<String>)>,
    }

    impl Transcriber for Scripted {
        fn transcribe_words(
            &mut self,
            samples: &[f32],
            sample_rate: u32,
            language: Option<&str>,
        ) -> Result<Vec<HeardWord>> {
            let call = self.calls.len();
            self.calls
                .push((samples.len(), sample_rate, language.map(str::to_string)));
            Ok(self.pieces.get(call).cloned().unwrap_or_default())
        }
    }

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn words_in_silence_are_dropped() -> Result<()> {
        let mut x = vec![0.0f32; 48000];
        x[8000..32000].fill(0.3);
        let mut t = Scripted {
            pieces: vec![vec![
                word(" hello", 0.5, 1.0),
                word(" ", 1.0, 1.1),
                word("world", 1.1, f64::NAN),
                word("you", 2.5, 2.8),
            ]],
            calls: Vec::new(),
        };
        let out = transcribe_narration(&x, &mut t, &TranscribeOptions::default())?;
        assert_eq!(t.calls, [(48000, 16000, Some("en".to_string()))]);
        let texts: Vec<&str> = out.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["hello", "world"]);
        assert_eq!(out.words[1].end, 2.0);
        assert_eq!(out.duration, 3.0);
        assert!(!out.looped);
        Ok(())
    }

    #[test]
    fn word_with_no_length_is_judged_with_the_next() -> Result<()> {
        let mut x = vec![0.0f32; 48000];
        x[8000..32000].fill(0.3);
        let mut t = Scripted {
            pieces: vec![vec![
                word("He", 0.0, 0.0),
                word("laments", 0.0, 1.2),
                word("you", 3.0, 3.0),
            ]],
            calls: Vec::new(),
        };
        let out = transcribe_narration(&x, &mut t, &TranscribeOptions::default())?;
        let texts: Vec<&str> = out.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["He", "laments"]);
        Ok(())
    }

    #[test]
    fn short_audio_is_never_read() -> Result<()> {
        let mut t = Scripted {
            pieces: vec![vec![word("x", 0.0, 0.1)]],
            calls: Vec::new(),
        };
        let out = transcribe_narration(&[0.5; 100], &mut t, &TranscribeOptions::default())?;
        assert!(t.calls.is_empty());
        assert!(out.words.is_empty());
        Ok(())
    }
}
