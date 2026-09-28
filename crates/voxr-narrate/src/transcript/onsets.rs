//! Recognized word edges moved onto the sound: a late start after a pause
//! walked back to where the sound rises, and a loose end cut where the
//! sound stops.

use super::envelope::{Envelope, frame_at, frame_ceil, frame_seconds};
use crate::check::HeardWord;
use crate::check::speech_match::letters;
use crate::error::Result;
use crate::js::{js_max, js_min, js_to_fixed};
use crate::pattern::{Patterns, utf16_len};

/// A word that starts this long after the previous word's end, in seconds,
/// follows a pause.
const PAUSE: f64 = 0.15;
/// A start walks back at most this many frames: 0.6 s.
const MAX_BACK: i64 = 60;
/// This many silent frames in a row, 80 ms, mark a word's edge.
const EDGE_QUIET: usize = 8;
/// A word shorter than this, in seconds, has no length.
const NO_LENGTH: f64 = 0.06;
/// The least span, in seconds, two words with no length share.
const PAIR_SPAN: f64 = 0.12;
/// An end this far past its sound, in seconds, is loose.
const LOOSE: f64 = 0.05;
/// Decimals of a shared edge.
const TIME_DECIMALS: usize = 3;
/// Seconds per letter a word needs at the least.
const MIN_LETTER: f64 = 0.05;
/// Loud frames in a row, 30 ms, that make a word's own sound.
const MIN_RUN: i64 = 3;

/// Gives the tail of a word's sound back to it when the next word's start
/// lands on that tail.
///
/// A recognizer can start a word on the last frames of the word before it,
/// then a pause, then the word's own sound ("the | furnishing"). A word is
/// moved when:
/// - it starts under 50 ms after the previous word's end,
/// - the sound runs across its start,
/// - that sound stops within less time than the word's letters need
///   (0.05 s per letter), followed by 80 ms of silence,
/// - and 30 ms of sound follows before the word's end.
///
/// The word then starts where that later sound rises, and the previous
/// word ends where the tail stops, when it ended before that.
///
/// # Errors
///
/// [`Error::Pattern`](crate::Error::Pattern) when the word pattern fails to
/// build.
pub fn hand_back_tails(words: &mut [HeardWord], env: &Envelope) -> Result<()> {
    let p = Patterns::get()?;
    for i in 1..words.len() {
        let (head, tail) = words.split_at_mut(i);
        let prev = &mut head[i - 1];
        let w = &mut tail[0];
        let f = frame_at(w.start);
        if w.start - prev.end >= LOOSE || !env.loud(f - 1) || !env.loud(f) {
            continue;
        }
        let bound = frame_at(w.end);
        let mut k = f;
        while k < bound && env.loud(k) {
            k += 1;
        }
        let tail_end = k;
        let need = MIN_LETTER * letters(p, &w.text) as f64;
        if frame_seconds(tail_end - f) >= need {
            continue;
        }
        let mut quiet = 0;
        while k < bound && !env.loud(k) {
            quiet += 1;
            k += 1;
        }
        if quiet < EDGE_QUIET as i64 || !(k..k + MIN_RUN).all(|j| j < bound && env.loud(j)) {
            continue;
        }
        prev.end = js_max(prev.end, frame_seconds(tail_end));
        w.start = frame_seconds(k);
    }
    Ok(())
}

/// Starts a word that follows a pause where its sound rises.
///
/// A recognizer places a word's start late when speech resumes after
/// silence: it can miss the first syllable. The first word, and a word
/// starting 0.15 s or more after the previous word's end, walks back
/// through its sound, across dips inside a syllable, until 80 ms of
/// silence: at most 0.6 s, never before the previous word's end. A start
/// only moves earlier.
pub fn snap_onsets(words: &mut [HeardWord], env: &Envelope) {
    for i in 0..words.len() {
        let prev_end = if i > 0 { words[i - 1].end } else { 0.0 };
        let w = &mut words[i];
        if i > 0 && w.start - prev_end < PAUSE {
            continue;
        }
        let f = frame_at(w.start);
        let stop = frame_ceil(prev_end).max(f - MAX_BACK);
        let mut onset = f;
        let mut quiet = 0;
        let mut k = f - 1;
        while k >= stop {
            if env.loud(k) {
                onset = k;
                quiet = 0;
            } else {
                quiet += 1;
                if quiet >= EDGE_QUIET {
                    break;
                }
            }
            k -= 1;
        }
        w.start = js_min(w.start, frame_seconds(onset));
    }
}

/// Ends each word where its sound stops.
///
/// A recognizer's word ends are loose: an end can run a second into the
/// pause after it, and a word can get no length at all. A word with no
/// length (under 60 ms) whose next word starts under 60 ms after it first
/// starts where the sound rises before the next word's end, when it starts
/// in silence. It then shares the span up to the next word's end (at least
/// 0.12 s) by raw text length, punctuation included, and the shared edge
/// is rounded to milliseconds.
/// Any other word ends at the last loud frame before 80 ms of silence,
/// searched from its start up to the next word's start (or `duration`),
/// when its end runs over 50 ms past that or it has no length. It then
/// keeps at least 60 ms and never passes the next word's start.
pub fn snap_ends(words: &mut [HeardWord], env: &Envelope, duration: f64) {
    for i in 0..words.len() {
        let (head, tail) = words.split_at_mut(i + 1);
        let w = &mut head[i];
        let mut next = tail.first_mut();
        if w.end - w.start < NO_LENGTH
            && let Some(n) = next.as_deref_mut()
            && n.start - w.start < NO_LENGTH
        {
            let stop = frame_at(n.end);
            if let Some(k) = (frame_at(w.start)..stop).find(|&k| env.loud(k)) {
                w.start = js_max(w.start, frame_seconds(k));
            }
            let own = utf16_len(&w.text) as f64;
            let share = own / (own + utf16_len(&n.text) as f64);
            let edge = js_to_fixed(
                w.start + (js_max(n.end, w.start + PAIR_SPAN) - w.start) * share,
                TIME_DECIMALS,
            );
            w.end = edge;
            n.start = edge;
            continue;
        }
        let next_start = next.as_deref().map(|n| n.start);
        let bound = env.frames().min(frame_at(
            next_start.map_or(duration, |s| js_max(s, w.start)),
        ));
        let mut last = -1;
        let mut quiet = 0;
        let mut k = frame_at(w.start);
        while k < bound {
            if env.loud(k) {
                last = k;
                quiet = 0;
            } else {
                quiet += 1;
                if quiet >= EDGE_QUIET && last >= 0 {
                    break;
                }
            }
            k += 1;
        }
        if last < 0 {
            continue;
        }
        let sound_end = frame_seconds(last + 1);
        if w.end > sound_end + LOOSE || w.end - w.start < NO_LENGTH {
            w.end = js_max(
                w.start + NO_LENGTH,
                js_min(sound_end, next_start.unwrap_or(duration)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    /// Frames 100 to 199 loud, the rest silent, 3 s in all.
    fn one_second() -> Envelope {
        let mut env = vec![0.0; 300];
        env[100..200].fill(1.0);
        Envelope { env, floor: 0.01 }
    }

    #[test]
    fn late_start_walks_back_to_the_sound() {
        let mut words = [word("a", 1.3, 1.9)];
        snap_onsets(&mut words, &one_second());
        assert_eq!(words[0].start, 1.0);
        // Following closely, a word keeps its start.
        let mut words = [word("a", 0.0, 1.2), word("b", 1.3, 1.9)];
        snap_onsets(&mut words, &one_second());
        assert_eq!(words[1].start, 1.3);
    }

    #[test]
    fn loose_end_stops_at_the_sound() {
        let mut words = [word("a", 1.0, 2.8)];
        snap_ends(&mut words, &one_second(), 3.0);
        assert_eq!(words[0].end, 2.0);
    }

    /// "the" loud over frames 100 to 167, silence to 195, "furnishing"
    /// loud over 196 to 259, the rest silent, 3 s in all.
    fn tail_then_word() -> Envelope {
        let mut env = vec![0.0; 300];
        env[100..168].fill(1.0);
        env[196..260].fill(1.0);
        Envelope { env, floor: 0.01 }
    }

    #[test]
    fn start_on_the_previous_tail_moves_past_the_pause() -> Result<()> {
        let mut words = [word("the", 1.0, 1.6), word("furnishing", 1.6, 2.5)];
        hand_back_tails(&mut words, &tail_then_word())?;
        assert_eq!(words[0].end, 1.68);
        assert_eq!(words[1].start, 1.96);
        Ok(())
    }

    #[test]
    fn tail_long_enough_for_the_word_stays_its_own() -> Result<()> {
        // 80 ms of sound is all "a" needs: it is the word, not a tail.
        let mut words = [word("the", 1.0, 1.6), word("a", 1.6, 2.5)];
        hand_back_tails(&mut words, &tail_then_word())?;
        assert_eq!((words[0].end, words[1].start), (1.6, 1.6));
        Ok(())
    }

    #[test]
    fn tail_without_later_sound_or_after_a_gap_stays() -> Result<()> {
        // The word ends before the later sound.
        let mut words = [word("the", 1.0, 1.6), word("furnishing", 1.6, 1.9)];
        hand_back_tails(&mut words, &tail_then_word())?;
        assert_eq!((words[0].end, words[1].start), (1.6, 1.6));
        // The word starts 60 ms after the previous end.
        let mut words = [word("the", 1.0, 1.54), word("furnishing", 1.6, 2.5)];
        hand_back_tails(&mut words, &tail_then_word())?;
        assert_eq!((words[0].end, words[1].start), (1.54, 1.6));
        Ok(())
    }

    #[test]
    fn crushed_pair_in_silence_starts_at_the_sound() {
        let mut words = [word("The", 0.0, 0.0), word("end.", 0.0, 1.6)];
        snap_ends(&mut words, &one_second(), 3.0);
        assert_eq!(words[0].start, 1.0);
        assert_eq!(words[0].end, 1.257);
        assert_eq!(words[1].start, 1.257);
    }

    #[test]
    fn crushed_pair_shares_by_raw_length() {
        let mut words = [word("Hi,", 1.0, 1.02), word("there.", 1.03, 1.6)];
        snap_ends(&mut words, &one_second(), 3.0);
        assert_eq!(words[0].end, 1.2);
        assert_eq!(words[1].start, 1.2);
    }
}
