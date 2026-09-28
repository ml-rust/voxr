//! Word times settled on the audio, after recognition or after a script
//! replaced the recognized words.

use super::envelope::{Envelope, RATE, envelope, frame_from_zero, frame_seconds};
use super::even::even_out;
use crate::check::HeardWord;
use crate::check::speech_match::letters;
use crate::error::Result;
use crate::js::{js_max, js_min};
use crate::pattern::Patterns;
use crate::text::Cue;

/// The least step, in seconds, from one word's start to the next.
const MIN_STEP: f64 = 0.02;
/// A start in silence searches this far ahead for its sound, in seconds.
const SEARCH_AHEAD: f64 = 1.5;
/// The search stops this long before the next word's start, in seconds.
const BEFORE_NEXT: f64 = 0.05;
/// A moved word keeps at most this much of its length, in seconds.
const MAX_KEEP: f64 = 0.5;
/// Frames in a row that make a sound's start.
const LOUD_RUN: i64 = 3;
/// Silent frames in a row that end a word: 150 ms. A shorter dip is a stop
/// consonant inside it ("k" in "Persekutuan").
const END_SILENCE: usize = 15;
/// An end this far past its sound, in seconds, is trimmed.
const LOOSE: f64 = 0.05;
/// A trim never leaves a word shorter than this, in seconds.
const MAX_NEED: f64 = 0.3;
/// Seconds each letter needs at the least.
const LETTER_TIME: f64 = 0.05;
/// A run of sound this many frames long or longer counts: 30 ms.
const MIN_RUN: i64 = 3;
/// A phrase's words last at least this long, in seconds.
const MIN_WORD: f64 = 0.04;
/// Speech after the last word counts once it starts this long after it.
const TRAILING_GAP: f64 = 0.15;

/// Settles word times on the audio.
///
/// 1. Times run in order, 20 ms apart at the least, and no word ends
///    before it starts or after `duration`.
/// 2. A word placed in silence (after a pause, a line break) starts where
///    its sound does: 3 loud frames in a row, searching forward at most
///    1.5 s and stopping 50 ms before the next word. It keeps at most 0.5 s
///    of its length.
/// 3. No word runs into the next one.
/// 4. Crushed runs are spread ([`even_out`]).
/// 5. An end that runs over 50 ms past its sound stops where the sound
///    does, never below what the word's letters need.
/// 6. With no 150 ms silence before the next word, a word lasts until it,
///    and the last word to its sound's end.
///
/// Closing gaps can crush a word again, so crushed runs are spread once
/// more, within `duration`.
pub fn fit_to_sound(words: &mut [HeardWord], env: &Envelope, duration: f64) -> Result<()> {
    let p = Patterns::get()?;
    let len = words.len();
    let frames = env.frames();
    let loud_run = |k: i64| (0..LOUD_RUN).all(|d| env.loud(k + d));
    for i in 0..len {
        if i > 0 {
            words[i].start = js_max(words[i].start, words[i - 1].start + MIN_STEP);
        }
        let w = &mut words[i];
        w.end = js_min(duration, js_max(w.end, w.start));
    }
    for i in 0..len {
        let next_start = words.get(i + 1).map(|n| n.start);
        let w = &mut words[i];
        let f = frame_from_zero(w.start);
        if f >= frames - LOUD_RUN || env.loud(f) {
            continue;
        }
        let stop = (frames - LOUD_RUN)
            .min(frame_from_zero(w.start + SEARCH_AHEAD))
            .min(next_start.map_or(frames - LOUD_RUN, |s| frame_from_zero(s - BEFORE_NEXT)));
        let mut k = f;
        while k < stop && !loud_run(k) {
            k += 1;
        }
        if k >= stop {
            continue;
        }
        let start = frame_seconds(k);
        w.end = js_max(w.end, start + js_min(MAX_KEEP, w.end - w.start));
        w.start = start;
    }
    for i in 0..len.saturating_sub(1) {
        let next_start = words[i + 1].start;
        let w = &mut words[i];
        if w.end > next_start {
            w.end = js_max(w.start, next_start);
        }
    }
    even_out(words, f64::INFINITY)?;
    let weight: Vec<usize> = words.iter().map(|w| letters(p, &w.text)).collect();
    for i in 0..len {
        let next_start = words.get(i + 1).map(|n| n.start);
        let w = &mut words[i];
        let Some(end) = sound_end(env, w.start, next_start.unwrap_or(duration)) else {
            continue;
        };
        if w.end > end + LOOSE && end - w.start >= js_min(MAX_NEED, LETTER_TIME * weight[i] as f64)
        {
            w.end = end;
        }
        match next_start {
            Some(next) if next > w.end && !quiet_run(env, w.end, next) => w.end = next,
            None if end > w.end => w.end = end,
            _ => {}
        }
    }
    even_out(words, duration)
}

/// 150 ms of silence between two times.
fn quiet_run(env: &Envelope, from: f64, to: f64) -> bool {
    let mut run = 0;
    for k in frame_from_zero(from)..env.frames().min(frame_from_zero(to)) {
        if env.loud(k) {
            run = 0;
        } else {
            run += 1;
            if run >= END_SILENCE {
                return true;
            }
        }
    }
    false
}

/// The end of the sound starting in `[from, to)`: the frame after the last
/// loud one before 150 ms of silence, in seconds, or `None` with no loud
/// frame.
fn sound_end(env: &Envelope, from: f64, to: f64) -> Option<f64> {
    let mut last = -1;
    let mut quiet = 0;
    for k in frame_from_zero(from)..env.frames().min(frame_from_zero(to)) {
        if env.loud(k) {
            last = k;
            quiet = 0;
        } else {
            quiet += 1;
            if quiet >= END_SILENCE && last >= 0 {
                break;
            }
        }
    }
    (last >= 0).then(|| frame_seconds(last + 1))
}

/// The last stretch of sound (30 ms or more) between two times, as
/// `(start, end)` in seconds, or `None`. Reading back from `to`, the
/// stretch ends at the first loud frame and starts where 150 ms of silence
/// comes before it.
pub fn last_run(env: &Envelope, from: f64, to: f64) -> Option<(f64, f64)> {
    let mut end = -1;
    let mut start = -1;
    let lo = frame_from_zero(from);
    let mut k = env.frames().min(frame_from_zero(to)) - 1;
    while k >= lo {
        if env.loud(k) {
            if end < 0 {
                end = k;
            }
            start = k;
        } else if end >= 0 && start - k > END_SILENCE as i64 {
            break;
        }
        k -= 1;
    }
    if end < 0 || end - start < MIN_RUN {
        return None;
    }
    Some((frame_seconds(start), frame_seconds(end + 1)))
}

/// The phrase each word belongs to, and each phrase's span.
#[derive(Debug, Clone, Copy)]
pub struct PhraseGroups<'a> {
    /// Each phrase's text and span in seconds.
    pub spans: &'a [Cue],
    /// The index into `spans` of each word's phrase. A word past the end of
    /// this list, or with an index past `spans`, is in no phrase.
    pub phrase: &'a [usize],
}

/// [`fit_to_sound`] for words and the 16 kHz audio they came from: run
/// after a script replaces the recognized words.
///
/// With phrase groups, each phrase settles alone, within the audio and its
/// span's end, and its words stay inside its span, each at least 40 ms
/// long. Speech after a pause at the end of a phrase with no word on it
/// ("statusnya, \[pause\] terancam") belongs to the last word: the
/// recognizer placed it too early, in the tail of the word before. Words
/// crushed against the span's end are spread within it. A word in no
/// phrase keeps its times.
pub fn fit_file(
    words: &mut [HeardWord],
    samples: &[f32],
    phrases: Option<PhraseGroups<'_>>,
) -> Result<()> {
    let env = envelope(samples);
    let duration = samples.len() as f64 / RATE;
    let Some(groups) = phrases else {
        return fit_to_sound(words, &env, duration);
    };
    for (pi, span) in groups.spans.iter().enumerate() {
        let members: Vec<usize> = (0..words.len())
            .filter(|&i| groups.phrase.get(i) == Some(&pi))
            .collect();
        if members.is_empty() {
            continue;
        }
        let mut group: Vec<HeardWord> = members.iter().map(|&i| words[i].clone()).collect();
        fit_to_sound(&mut group, &env, js_min(duration, span.end))?;
        for w in &mut group {
            w.start = js_min(js_max(w.start, span.start), span.end - MIN_WORD);
            w.end = js_min(js_max(w.end, w.start + MIN_WORD), span.end);
        }
        let last = group.len() - 1;
        let after = group[last].end + TRAILING_GAP;
        if let Some((start, end)) = last_run(&env, after, span.end)
            && start > after
        {
            group[last].start = start;
            group[last].end = end;
            if last > 0 {
                group[last - 1].end = js_min(group[last - 1].end, start);
            }
        }
        even_out(&mut group, span.end)?;
        for (&i, w) in members.iter().zip(group) {
            words[i] = w;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::envelope::HOP;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    /// 16 kHz samples, `seconds` long, at 0.3 over each loud span.
    fn audio(seconds: f64, loud: &[(f64, f64)]) -> Vec<f32> {
        let mut x = vec![0.0f32; (seconds * RATE) as usize];
        for &(a, b) in loud {
            x[(a * RATE) as usize..(b * RATE) as usize].fill(0.3);
        }
        x
    }

    #[test]
    fn start_in_silence_moves_to_the_sound() -> Result<()> {
        let x = audio(3.0, &[(1.0, 2.0)]);
        let mut words = [word("sound", 0.6, 0.9)];
        fit_to_sound(&mut words, &envelope(&x), 3.0)?;
        assert_eq!(words[0].start, 1.0);
        // The last word lasts to its sound's end.
        assert_eq!(words[0].end, 2.0);
        Ok(())
    }

    #[test]
    fn speech_runs_on_to_the_next_word() -> Result<()> {
        let x = audio(3.0, &[(0.5, 2.5)]);
        let mut words = [word("hello", 0.5, 1.0), word("there", 1.4, 2.5)];
        fit_to_sound(&mut words, &envelope(&x), 3.0)?;
        assert_eq!(words[0].end, 1.4);
        Ok(())
    }

    #[test]
    fn last_run_finds_trailing_speech() {
        let x = audio(3.0, &[(0.5, 1.0), (2.0, 2.5)]);
        let env = envelope(&x);
        assert_eq!(last_run(&env, 1.2, 3.0), Some((2.0, 2.5)));
        assert_eq!(last_run(&env, 1.2, 1.9), None);
        let blip = audio(1.0, &[(0.5, 0.5 + 2.0 * HOP as f64 / RATE)]);
        assert_eq!(last_run(&envelope(&blip), 0.0, 1.0), None);
    }

    #[test]
    fn phrase_takes_its_trailing_speech() -> Result<()> {
        let x = audio(6.0, &[(3.0, 3.8), (4.6, 5.4)]);
        let spans = [Cue {
            start: 2.8,
            end: 5.8,
            text: "status terancam".into(),
        }];
        let mut words = [word("status", 3.0, 3.5), word("terancam", 3.5, 3.8)];
        let phrase = [0, 0];
        fit_file(
            &mut words,
            &x,
            Some(PhraseGroups {
                spans: &spans,
                phrase: &phrase,
            }),
        )?;
        assert_eq!((words[1].start, words[1].end), (4.6, 5.4));
        assert!(words[0].end <= 4.6);
        Ok(())
    }
}
