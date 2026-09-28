//! The kept takes shaped and joined into one track: each take's sentences
//! found in its audio, its silences at blank lines and pause marks widened
//! to the script's pause, its heat read ([`part_heat`]), and the takes laid
//! end to end with each part's pause after it.

use crate::check::{HeardWord, align_script};
use crate::error::Result;
use crate::js::{js_max, js_round, js_to_fixed};
use crate::pattern::{Patterns, split};
use crate::text::{Chunk, Phrase, PhraseKind, sayable, spoken};

use super::files::PhrasesFile;
use super::join_heat::{PartHeat, part_heat};

/// Samples per second of every take shaped and of the joined track.
pub const JOIN_RATE: u32 = 48000;

/// Samples in the window whose energy picks where a silence is widened:
/// 10 ms.
const QUIET_WINDOW: i64 = 480;
/// Samples the quiet window steps by.
const QUIET_STEP: i64 = 240;
/// Seconds a sentence span reaches before its first word and past its
/// last.
const SPAN_MARGIN: f64 = 0.05;
/// Decimals of a span's times.
const SPAN_DECIMALS: usize = 3;
/// Decimals of a part's start in the checks.
const START_DECIMALS: usize = 2;

/// A take with its silences set, and where each sentence sits in it.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapedTake {
    /// The take's samples at [`JOIN_RATE`], silence inserted where widened.
    pub audio: Vec<f32>,
    /// One span per sentence, in seconds from the take's start, with the
    /// sentence's script text. A take whose sentences cannot be found holds
    /// one span over the whole take.
    pub spans: Vec<HeardWord>,
}

/// `x.subarray(begin, end)`: a negative index counts from the end, both
/// are clamped to the length, and an end before the begin is empty.
pub(super) fn subarray(x: &[f32], begin: i64, end: i64) -> &[f32] {
    let len = x.len() as i64;
    let at = |i: i64| (if i < 0 { len + i } else { i }).clamp(0, len) as usize;
    let (b, e) = (at(begin), at(end));
    if e > b { &x[b..e] } else { &[] }
}

/// The quietest 10 ms window's midpoint from sample `a`, stepping 5 ms,
/// among windows that end by `b` and inside the audio. The first of equal
/// windows wins. `a` itself when no window fits.
fn quietest(audio: &[f32], a: i64, b: i64) -> i64 {
    let len = audio.len() as i64;
    let mut at = a;
    let mut low = f64::INFINITY;
    let mut x = a;
    while x + QUIET_WINDOW <= b && x + QUIET_WINDOW <= len {
        let mut e = 0.0f64;
        for j in x..x + QUIET_WINDOW {
            // A sample before the start reads as JavaScript's `undefined`.
            let v = usize::try_from(j)
                .ok()
                .and_then(|j| audio.get(j))
                .map_or(f64::NAN, |&v| f64::from(v));
            e += v * v;
        }
        if e < low {
            low = e;
            at = x + QUIET_STEP;
        }
        x += QUIET_STEP;
    }
    at
}

/// Finds each sentence's bounds in `heard` through [`align_script`]: the
/// first word start and the last word end among its script words. `None`
/// when the aligned words do not number the sentences' words, or a
/// sentence has no timed word.
fn sentence_bounds(heard: &[HeardWord], sentences: &[Phrase]) -> Result<Option<Vec<(f64, f64)>>> {
    let p = Patterns::get()?;
    let texts: Vec<String> = sentences
        .iter()
        .map(|s| sayable(&s.text))
        .collect::<Result<_>>()?;
    let counts: Vec<usize> = texts
        .iter()
        .map(|t| {
            split(&p.spaces, t)
                .into_iter()
                .filter(|w| p.word_char.is_match(w))
                .count()
        })
        .collect();
    let aligned = align_script(heard, &texts.join(" "))?;
    if aligned.len() != counts.iter().sum::<usize>() {
        return Ok(None);
    }
    let mut bounds = Vec::with_capacity(counts.len());
    let mut k = 0;
    for c in counts {
        let ws = &aligned[k..k + c];
        k += c;
        let start = ws.iter().map(|w| w.start).find(|t| !t.is_nan());
        let end = ws.iter().rev().map(|w| w.end).find(|t| !t.is_nan());
        match (start, end) {
            (Some(s), Some(e)) => bounds.push((s, e)),
            _ => return Ok(None),
        }
    }
    Ok(Some(bounds))
}

/// Shapes one take: its sentences found in its audio from the heard words,
/// and its silences set.
///
/// At each blank line and `[pause]` mark inside the take, a gap between two
/// sentences shorter than the script's pause is widened to it. The silence
/// goes in at the quietest 10 ms of the gap. An existing silence is never
/// shortened. Each span reaches 50 ms before its first word and past its
/// last, moved by the silence inserted before its first word.
///
/// With no heard words, one sentence, or sentences that cannot be found in
/// the heard words, the take stays as it is, one span over its length.
/// `audio` is at [`JOIN_RATE`].
pub fn shape_take(audio: &[f32], heard: &[HeardWord], sentences: &[Phrase]) -> Result<ShapedTake> {
    let rate = f64::from(JOIN_RATE);
    let whole = || {
        let text: Vec<&str> = sentences.iter().map(|p| p.text.as_str()).collect();
        ShapedTake {
            audio: audio.to_vec(),
            spans: vec![HeardWord {
                text: text.join(" "),
                start: 0.0,
                end: js_to_fixed(audio.len() as f64 / rate, SPAN_DECIMALS),
            }],
        }
    };
    if heard.is_empty() || sentences.len() < 2 {
        return Ok(whole());
    }
    let Some(bounds) = sentence_bounds(heard, sentences)? else {
        return Ok(whole());
    };
    // Each insert: the sample it goes in at, and its length in samples.
    let mut inserts: Vec<(i64, usize)> = Vec::new();
    for (i, p) in sentences.iter().enumerate() {
        if i + 1 >= sentences.len() || !matches!(p.kind, PhraseKind::Beat | PhraseKind::Mark) {
            continue;
        }
        let gap = bounds[i + 1].0 - bounds[i].1;
        if gap >= p.pause {
            continue;
        }
        let a = js_round(bounds[i].1 * rate) as i64;
        let b = (a + QUIET_WINDOW).max(js_round(bounds[i + 1].0 * rate) as i64);
        let n = js_round((p.pause - js_max(0.0, gap)) * rate).max(0.0) as usize;
        inserts.push((quietest(audio, a, b), n));
    }
    let mut out = Vec::with_capacity(audio.len() + inserts.iter().map(|i| i.1).sum::<usize>());
    let mut from = 0;
    for &(at, n) in &inserts {
        out.extend_from_slice(subarray(audio, from, at));
        out.resize(out.len() + n, 0.0);
        from = at;
    }
    out.extend_from_slice(subarray(audio, from, audio.len() as i64));
    let shift_at = |t: f64| {
        let n: usize = inserts
            .iter()
            .filter(|&&(at, _)| at as f64 <= t * rate)
            .map(|&(_, n)| n)
            .sum();
        n as f64 / rate
    };
    let spans = sentences
        .iter()
        .zip(&bounds)
        .map(|(p, &(start, end))| {
            let shift = shift_at(start);
            HeardWord {
                text: p.text.clone(),
                start: js_to_fixed(js_max(0.0, start - SPAN_MARGIN + shift), SPAN_DECIMALS),
                end: js_to_fixed(end + SPAN_MARGIN + shift, SPAN_DECIMALS),
            }
        })
        .collect();
    Ok(ShapedTake { audio: out, spans })
}

/// One part to join: the part as spoken, its kept take and the words heard
/// in it.
#[derive(Debug, Clone, Copy)]
pub struct JoinPart<'a> {
    /// The part, with its sentences and the pause after it.
    pub part: &'a Chunk,
    /// The kept take as its tidied 16-bit file reads back, at
    /// [`JOIN_RATE`].
    pub audio: &'a [f32],
    /// The words heard in the take.
    pub heard: &'a [HeardWord],
}

/// The whole voiceover: every part shaped and laid end to end.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinedTrack {
    /// The samples at [`JOIN_RATE`]: each shaped take, then its part's
    /// pause in silence.
    pub samples: Vec<f32>,
    /// The track's length in seconds.
    pub duration: f64,
    /// One span per sentence on the track's clock, its text as captions
    /// show it.
    pub phrases: Vec<HeardWord>,
    /// Each part's start in seconds.
    pub starts: Vec<f64>,
    /// Each part's hottest sentence.
    pub heat: Vec<PartHeat>,
}

impl JoinedTrack {
    /// The `.phrases.json` of this track, `shown` being the script as
    /// captions show it.
    pub fn phrases_file(&self, shown: &str) -> PhrasesFile {
        PhrasesFile {
            script: shown.to_string(),
            phrases: self.phrases.clone(),
        }
    }

    /// Part `i`'s start, to 2 decimals, as the checks list it.
    pub fn start_at(&self, i: usize) -> Option<f64> {
        self.starts.get(i).map(|&t| js_to_fixed(t, START_DECIMALS))
    }
}

/// Shapes each part's take ([`shape_take`]), reads its heat
/// ([`part_heat`]) against `voice_median`, and joins the takes, each
/// followed by its part's pause in silence.
pub fn join_takes(parts: &[JoinPart<'_>], voice_median: f64) -> Result<JoinedTrack> {
    let rate = f64::from(JOIN_RATE);
    let mut samples = Vec::new();
    let mut phrases = Vec::new();
    let mut starts = Vec::with_capacity(parts.len());
    let mut heats = Vec::with_capacity(parts.len());
    let mut t = 0.0f64;
    for jp in parts {
        let shaped = shape_take(jp.audio, jp.heard, &jp.part.sentences)?;
        heats.push(part_heat(&shaped, voice_median));
        starts.push(t);
        for sp in &shaped.spans {
            phrases.push(HeardWord {
                text: spoken(&sp.text)?,
                start: js_to_fixed(t + sp.start, SPAN_DECIMALS),
                end: js_to_fixed(t + sp.end, SPAN_DECIMALS),
            });
        }
        t += shaped.audio.len() as f64 / rate + jp.part.pause;
        let pause = js_round(jp.part.pause * rate).max(0.0) as usize;
        samples.extend_from_slice(&shaped.audio);
        samples.resize(samples.len() + pause, 0.0);
    }
    Ok(JoinedTrack {
        duration: samples.len() as f64 / rate,
        samples,
        phrases,
        starts,
        heat: heats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::phrases;

    fn word(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    /// Speech (a steady level) from 0 to 0.5 s and from 0.7 to 1.2 s of a
    /// 3 s take, and its words.
    fn two_sentences() -> (Vec<f32>, Vec<HeardWord>) {
        let audio: Vec<f32> = (0..144000)
            .map(|i| {
                if i < 24000 || (33600..57600).contains(&i) {
                    0.1
                } else {
                    0.0
                }
            })
            .collect();
        let heard = vec![
            word("One", 0.0, 0.2),
            word("two.", 0.25, 0.5),
            word("Three", 0.7, 0.9),
            word("four.", 0.95, 1.2),
        ];
        (audio, heard)
    }

    #[test]
    fn a_beat_widens_at_its_quietest_point() -> Result<()> {
        let (audio, heard) = two_sentences();
        let sentences = phrases("One two.\n\nThree four.")?;
        let shaped = shape_take(&audio, &heard, &sentences)?;
        assert_eq!(shaped.audio.len(), 144000 + 28800);
        assert!(shaped.audio[24240..24240 + 28800].iter().all(|&v| v == 0.0));
        assert_eq!(shaped.audio[24239], 0.0);
        assert_eq!(shaped.audio[..24000], audio[..24000]);
        assert_eq!(
            shaped.spans,
            [word("One two.", 0.0, 0.55), word("Three four.", 1.25, 1.85)]
        );
        Ok(())
    }

    #[test]
    fn a_line_break_never_widens() -> Result<()> {
        let (audio, heard) = two_sentences();
        let sentences = phrases("One two.\nThree four.")?;
        let shaped = shape_take(&audio, &heard, &sentences)?;
        assert_eq!(shaped.audio, audio);
        assert_eq!(shaped.spans[1], word("Three four.", 0.65, 1.25));
        Ok(())
    }

    #[test]
    fn no_heard_words_keep_one_span() -> Result<()> {
        let (audio, _) = two_sentences();
        let sentences = phrases("One two.\n\nThree four.")?;
        let shaped = shape_take(&audio, &[], &sentences)?;
        assert_eq!(shaped.audio, audio);
        assert_eq!(shaped.spans, [word("One two. Three four.", 0.0, 3.0)]);
        Ok(())
    }

    #[test]
    fn subarray_clamps_as_javascript() {
        let x = [1.0f32, 2.0, 3.0];
        assert_eq!(subarray(&x, 1, 9), [2.0, 3.0]);
        assert_eq!(subarray(&x, 2, 1), [0.0f32; 0]);
        assert_eq!(subarray(&x, -1, 3), [3.0]);
    }

    #[test]
    fn the_join_lays_each_pause_after_its_part() -> Result<()> {
        let script = phrases("One.\nTwo.")?;
        let part = |p: &Phrase| Chunk {
            text: p.text.clone(),
            pause: p.pause,
            kind: p.kind,
            sentences: vec![p.clone()],
            delivery: None,
        };
        let (a, b) = (part(&script[0]), part(&script[1]));
        let audio = vec![0.0f32; 4800];
        let track = join_takes(
            &[
                JoinPart {
                    part: &a,
                    audio: &audio,
                    heard: &[],
                },
                JoinPart {
                    part: &b,
                    audio: &audio,
                    heard: &[],
                },
            ],
            150.0,
        )?;
        let pause = js_round(a.pause * 48000.0) as usize;
        assert_eq!(track.samples.len(), 9600 + pause);
        assert_eq!(track.starts, [0.0, 0.1 + a.pause]);
        assert_eq!(track.phrases[1].start, js_to_fixed(0.1 + a.pause, 3));
        assert_eq!(
            track.heat[0],
            PartHeat {
                heat: 0.0,
                hot_text: None
            }
        );
        assert_eq!(track.phrases_file("One. Two.").phrases, track.phrases);
        Ok(())
    }
}
