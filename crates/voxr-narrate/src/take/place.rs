//! A timed script's lines placed at their times on the spoken track: each
//! line's phrases, with their own pauses, start at the time the line asks
//! for, or a breath after the line before when that one runs past it. Also
//! the script a timed run speaks, and the problem an engine without phrase
//! spans reports.

use crate::check::HeardWord;
use crate::error::{Error, Result};
use crate::js::{js_max, js_round, js_string, js_to_fixed, js_to_fixed_text};
use crate::pattern::utf16_prefix;
use crate::text::{ChunkOptions, TimedCue, chunks, phrases};

/// The rate a track is placed at: the skill decodes the spoken track at
/// 48 kHz and writes the placed one at it.
pub const PLACE_RATE: u32 = 48000;

/// Seconds of silence between a line and the line before it, when that one
/// runs past the later line's time.
pub const LINE_GAP: f64 = 0.25;

/// Seconds a line can start late, or run past its slot, before it is
/// reported.
pub const TIMING_SLACK: f64 = 0.05;

/// The most samples a placed track holds: a 16-bit WAV's 32-bit data size.
const MAX_SAMPLES: usize = 1 << 31;

/// Characters of a line's text a timing message quotes, in UTF-16 code
/// units.
const QUOTED: usize = 40;

/// The problem a timed run reports when the engine returns no phrase spans.
pub const UNPLACED_PROBLEM: &str = "This engine speaks the script in one piece, so its lines cannot be placed at their times. Run the command without --one-call, or use Kokoro, VoxCPM2 or --model.";

/// How many spoken spans one line of a timed script takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueUnits {
    /// One span per phrase ([`phrases`]): every engine but VoxCPM2.
    Phrases,
    /// One span per generation with every line break a split ([`chunks`]
    /// with `lines` on): VoxCPM2, which speaks each line apart.
    Lines,
}

impl CueUnits {
    /// The spans the line `text` takes.
    pub fn count(self, text: &str) -> Result<usize> {
        Ok(match self {
            Self::Phrases => phrases(text)?.len(),
            Self::Lines => chunks(
                text,
                ChunkOptions {
                    lines: true,
                    ..ChunkOptions::default()
                },
            )?
            .len(),
        })
    }
}

/// A track with every line of a timed script at its time.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedTrack {
    /// The placed track's samples, as joined before the WAV rounds them.
    pub samples: Vec<f32>,
    /// Their rate: [`PLACE_RATE`].
    pub rate: u32,
    /// The placed track's length in seconds.
    pub duration: f64,
    /// Each placed phrase's span on the placed track's clock.
    pub phrases: Vec<HeardWord>,
    /// One line per script line that starts late or runs past its slot.
    pub late: Vec<String>,
}

/// The script a timed run speaks: the lines joined by line breaks, spoken
/// with `lines` on so each line's spans count apart.
pub fn cue_script(cues: &[TimedCue]) -> String {
    cues.iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `x.subarray(a, b)` over `len` items: each end clamped into `0..=len`, a
/// negative end counted from the back, NaN read as 0.
fn sub_range(a: f64, b: f64, len: usize) -> std::ops::Range<usize> {
    let at = |v: f64| -> usize {
        let n = len as f64;
        let v = if v.is_nan() { 0.0 } else { v };
        let i = if v < 0.0 { (n + v).max(0.0) } else { v.min(n) };
        i as usize
    };
    let (start, end) = (at(a), at(b));
    start..end.max(start)
}

/// One script line and the spoken spans it takes.
struct Block<'a> {
    cue: &'a TimedCue,
    list: &'a [HeardWord],
    from: f64,
    to: f64,
}

/// The spans each line takes, in order: line `i` takes the next
/// `units.count(text)` spans.
fn blocks<'a>(
    spoken: &'a [HeardWord],
    cues: &'a [TimedCue],
    units: CueUnits,
) -> Result<Vec<Block<'a>>> {
    let mut out = Vec::with_capacity(cues.len());
    let mut p = 0;
    for (i, cue) in cues.iter().enumerate() {
        let n = units.count(&cue.text)?;
        let list = spoken.get(p..(p + n).min(spoken.len())).unwrap_or(&[]);
        let (Some(first), Some(last)) = (list.first(), list.last()) else {
            return Err(Error::DataError {
                reason: format!(
                    "line {} (\"{}\") takes {n} spans from span {}, but the engine returned {} spans. Speak the lines joined by line breaks with lines on, so each line's phrases are spoken",
                    i + 1,
                    utf16_prefix(&cue.text, QUOTED),
                    p + 1,
                    spoken.len()
                ),
            });
        };
        p += n;
        out.push(Block {
            cue,
            list,
            from: first.start,
            to: last.end,
        });
    }
    Ok(out)
}

/// The skill's `placeOnTimes`: each line of `cues` placed at its time on
/// the spoken track `samples`.
///
/// `samples` is the spoken track as ffmpeg decodes it at [`PLACE_RATE`]:
/// a 16-bit WAV's values over 32768 ([`read_wav`](crate::read_wav)).
/// `spoken` holds each spoken span on that track, in order. Each line takes
/// the next spans `units` counts in its text. A line starts at its time, or
/// [`LINE_GAP`] after the line before ends, whichever is later. Its spans
/// keep their own pauses. Spans past the last line's are dropped.
///
/// A line that starts over [`TIMING_SLACK`] late, or else runs that far
/// past its slot, adds a line to `late`. The slot ends at the line's range
/// end, else at the next line's time.
///
/// Refuses a rate other than [`PLACE_RATE`], a line whose spans the engine
/// did not return, and a line placed past the longest track a WAV holds.
pub fn place_on_times(
    samples: &[f32],
    rate: u32,
    spoken: &[HeardWord],
    cues: &[TimedCue],
    units: CueUnits,
) -> Result<PlacedTrack> {
    if rate != PLACE_RATE {
        return Err(Error::InvalidArgument {
            arg: "rate",
            reason: format!(
                "the spoken track is at {rate} Hz, not {PLACE_RATE} Hz. Resample it to {PLACE_RATE} Hz first"
            ),
        });
    }
    let hz = f64::from(rate);
    let blocks = blocks(spoken, cues, units)?;
    let mut late = Vec::new();
    let mut placed = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    let mut cursor = 0.0;
    for (i, b) in blocks.iter().enumerate() {
        let start = b.cue.start;
        let at = js_max(start, cursor + if i > 0 { LINE_GAP } else { 0.0 });
        let slot_end = b.cue.end.or_else(|| cues.get(i + 1).map(|c| c.start));
        let len = b.to - b.from;
        let quoted = utf16_prefix(&b.cue.text, QUOTED);
        if at > start + TIMING_SLACK {
            late.push(format!(
                "line {} (\"{quoted}\") starts at {}s, {}s after its {}s mark: the line before runs long.",
                i + 1,
                js_to_fixed_text(at, 2),
                js_to_fixed_text(at - start, 2),
                js_string(start)
            ));
        } else if let Some(end) = slot_end.filter(|&end| at + len > end + TIMING_SLACK) {
            late.push(format!(
                "line {} (\"{quoted}\") runs {}s past its slot ({}-{}s): shorten it, or give it more time.",
                i + 1,
                js_to_fixed_text(at + len - end, 2),
                js_string(start),
                js_string(end)
            ));
        }
        let gap = js_round((at - cursor) * hz);
        if gap < 0.0 {
            return Err(Error::DataError {
                reason: format!(
                    "line {} (\"{quoted}\") starts {}s before the track placed so far ends. Give each line a time no earlier than the one before",
                    i + 1,
                    js_string(cursor - at)
                ),
            });
        }
        // `new Float32Array(NaN)` is empty, as a NaN cast to usize is 0.
        let total = out.len().saturating_add(gap as usize);
        if total > MAX_SAMPLES {
            return Err(Error::DataError {
                reason: format!(
                    "line {} (\"{quoted}\") starts at {}s, past the longest track a WAV holds. Check the line's time",
                    i + 1,
                    js_string(at)
                ),
            });
        }
        out.resize(total, 0.0);
        out.extend_from_slice(
            &samples[sub_range(js_round(b.from * hz), js_round(b.to * hz), samples.len())],
        );
        for s in b.list {
            placed.push(HeardWord {
                text: s.text.clone(),
                start: js_to_fixed(at + s.start - b.from, 3),
                end: js_to_fixed(at + s.end - b.from, 3),
            });
        }
        cursor = at + len;
    }
    Ok(PlacedTrack {
        duration: out.len() as f64 / hz,
        samples: out,
        rate,
        phrases: placed,
        late,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(start: f64, end: Option<f64>, text: &str) -> TimedCue {
        TimedCue {
            start,
            end,
            text: text.to_string(),
        }
    }

    fn span(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    /// A track of `secs` seconds whose sample `i` is `i` over its length.
    fn ramp(secs: f64) -> Vec<f32> {
        let n = (secs * f64::from(PLACE_RATE)) as usize;
        (0..n).map(|i| i as f32 / n as f32).collect()
    }

    #[test]
    fn lines_on_time_start_at_their_marks() -> Result<()> {
        let track = ramp(2.0);
        let spoken = [span("One.", 0.1, 0.6), span("Two.", 1.0, 1.5)];
        let cues = [cue(1.0, None, "One."), cue(3.0, None, "Two.")];
        let r = place_on_times(&track, PLACE_RATE, &spoken, &cues, CueUnits::Phrases)?;
        assert_eq!(r.phrases, [span("One.", 1.0, 1.5), span("Two.", 3.0, 3.5)]);
        assert!(r.late.is_empty());
        assert_eq!(r.samples.len(), 3 * 48000 + 24000);
        assert_eq!(r.duration, 3.5);
        // The first line's audio starts right after a second of silence.
        assert_eq!(r.samples[47999], 0.0);
        assert_eq!(r.samples[48000], track[4800]);
        Ok(())
    }

    #[test]
    fn a_long_line_pushes_the_next_one_late() -> Result<()> {
        let track = ramp(3.0);
        let spoken = [span("A long line.", 0.0, 2.0), span("Next.", 2.0, 2.5)];
        let cues = [cue(0.0, None, "A long line."), cue(1.0, None, "Next.")];
        let r = place_on_times(&track, PLACE_RATE, &spoken, &cues, CueUnits::Phrases)?;
        assert_eq!(r.phrases[1], span("Next.", 2.25, 2.75));
        assert_eq!(
            r.late,
            [
                "line 1 (\"A long line.\") runs 1.00s past its slot (0-1s): shorten it, or give it more time.",
                "line 2 (\"Next.\") starts at 2.25s, 1.25s after its 1s mark: the line before runs long.",
            ]
        );
        Ok(())
    }

    #[test]
    fn a_range_end_bounds_the_slot() -> Result<()> {
        let track = ramp(2.0);
        let spoken = [span("Too long.", 0.0, 1.5)];
        let cues = [cue(0.5, Some(1.2), "Too long.")];
        let r = place_on_times(&track, PLACE_RATE, &spoken, &cues, CueUnits::Phrases)?;
        assert_eq!(
            r.late,
            [
                "line 1 (\"Too long.\") runs 0.80s past its slot (0.5-1.2s): shorten it, or give it more time."
            ]
        );
        Ok(())
    }

    #[test]
    fn a_line_takes_each_of_its_phrases() -> Result<()> {
        let track = ramp(3.0);
        let spoken = [
            span("First.", 0.0, 0.5),
            span("Second.", 0.8, 1.2),
            span("Third.", 1.6, 2.0),
        ];
        let cues = [cue(0.0, None, "First. Second."), cue(4.0, None, "Third.")];
        let r = place_on_times(&track, PLACE_RATE, &spoken, &cues, CueUnits::Phrases)?;
        assert_eq!(
            r.phrases,
            [
                span("First.", 0.0, 0.5),
                span("Second.", 0.8, 1.2),
                span("Third.", 4.0, 4.4)
            ]
        );
        Ok(())
    }

    #[test]
    fn missing_spans_are_refused() {
        let spoken = [span("One.", 0.0, 0.5)];
        let cues = [cue(0.0, None, "One."), cue(2.0, None, "Two.")];
        let r = place_on_times(&ramp(1.0), PLACE_RATE, &spoken, &cues, CueUnits::Phrases);
        assert!(matches!(r, Err(Error::DataError { .. })));
    }

    #[test]
    fn another_rate_is_refused() {
        let r = place_on_times(&[0.0; 10], 24000, &[], &[], CueUnits::Phrases);
        assert!(matches!(r, Err(Error::InvalidArgument { arg: "rate", .. })));
    }

    #[test]
    fn lines_units_count_generations() -> Result<()> {
        assert_eq!(CueUnits::Phrases.count("One. Two.")?, 2);
        assert_eq!(CueUnits::Lines.count("One. Two.")?, 1);
        Ok(())
    }

    #[test]
    fn subarray_ends_clamp_like_javascript() {
        assert_eq!(sub_range(2.0, 5.0, 10), 2..5);
        assert_eq!(sub_range(8.0, 20.0, 10), 8..10);
        assert_eq!(sub_range(-3.0, 10.0, 10), 7..10);
        assert_eq!(sub_range(6.0, 2.0, 10), 6..6);
        assert_eq!(sub_range(f64::NAN, 4.0, 10), 0..4);
    }

    #[test]
    fn the_cue_script_joins_lines() {
        let cues = [cue(0.0, None, "One."), cue(2.0, None, "Two.\nThree.")];
        assert_eq!(cue_script(&cues), "One.\nTwo.\nThree.");
    }
}
