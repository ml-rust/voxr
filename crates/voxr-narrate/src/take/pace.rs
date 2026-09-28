//! Speaking pace: syllables per second while a voice speaks, pauses left
//! out, and the verdict against a target pace with the `--speed` that
//! reaches it.

use unicode_normalization::UnicodeNormalization;

use crate::js::{js_max, js_min, js_round, js_to_fixed};

/// The pace a designed reference is brought to, in syllables per second
/// with its pauses: clones speak near it.
pub const REF_PACE: f64 = 3.9;

/// The pace a relaxed explainer sits near, in syllables per second while
/// speaking, pauses left out: 3.5 to 4.7 stays comfortable.
pub const DEFAULT_TARGET_PACE: f64 = 4.1;

/// Half the width of the accepted range around a target pace, in syllables
/// per second.
const PACE_RANGE: f64 = 0.6;

/// One phrase timed in a take: its spoken text and where it sits, in
/// seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct PaceSpan {
    /// The words spoken.
    pub text: String,
    /// Start, in seconds.
    pub start: f64,
    /// End, in seconds.
    pub end: f64,
}

/// Syllables in `text`, from its vowel groups: the unit a listener hears
/// pace in, across languages written in Latin letters ("berlumpur" is 3).
///
/// `text` is lowercased and NFKD-normalized first: an accent splits from
/// its letter and never joins a vowel run to the next ("café" is 2, not 1).
pub fn syllables(text: &str) -> usize {
    let normalized: String = text.to_lowercase().nfkd().collect();
    let mut count = 0;
    let mut in_run = false;
    for c in normalized.chars() {
        let vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
        if vowel && !in_run {
            count += 1;
        }
        in_run = vowel;
    }
    count
}

/// Sample `k` of `audio`, or NaN where JavaScript reads `undefined`: a
/// negative, fractional or past-the-end index.
fn frame_sample(audio: &[f32], k: f64) -> f64 {
    if k >= 0.0 && k.fract() == 0.0 && k < audio.len() as f64 {
        f64::from(audio[k as usize])
    } else {
        f64::NAN
    }
}

/// Seconds from the first to the last 10 ms frame louder than 5% of the
/// span's peak, and above a noise floor of 0.005.
///
/// `audio` is read in `rate / 100` sample hops from `floor(start * rate)`
/// up to `min(audio.len(), ceil(end * rate))`; a hop that runs past that
/// bound is dropped. 0 when no hop is louder than the floor, or when `rate`
/// is 0 or `start` is not finite.
pub fn sounding(audio: &[f32], start: f64, end: f64, rate: u32) -> f64 {
    let rate = f64::from(rate);
    let hop = rate / 100.0;
    let a = (start * rate).floor();
    if hop == 0.0 || !a.is_finite() {
        return 0.0;
    }
    let b = js_min(audio.len() as f64, (end * rate).ceil());
    let mut rms: Vec<f64> = Vec::new();
    let mut i = a;
    while i + hop <= b {
        let mut sum = 0.0f64;
        let mut k = i;
        while k < i + hop {
            let v = frame_sample(audio, k);
            sum += v * v;
            k += 1.0;
        }
        rms.push((sum / hop).sqrt());
        i += hop;
    }
    let peak = rms.iter().copied().fold(0.0, js_max);
    let floor = js_max(0.005, peak * 0.05);
    let Some(first) = rms.iter().position(|&v| v > floor) else {
        return 0.0;
    };
    let mut last = rms.len() - 1;
    while rms[last] <= floor {
        last -= 1;
    }
    ((last - first + 1) as f64) / 100.0
}

/// Speaking pace over `phrases`: syllables per second while speaking,
/// pauses left out.
///
/// `audio` is the take's decoded samples and their rate. Without it, each
/// phrase counts its whole span (`end - start`) as speaking time, pauses
/// included. 0 when no phrase holds any speaking time.
pub fn pace(phrases: &[PaceSpan], audio: Option<(&[f32], u32)>) -> f64 {
    let talk: f64 = phrases
        .iter()
        .map(|p| match audio {
            Some((x, rate)) => sounding(x, p.start, p.end, rate),
            None => p.end - p.start,
        })
        .sum();
    if talk > 0.0 {
        let total: f64 = phrases.iter().map(|p| syllables(&p.text) as f64).sum();
        total / talk
    } else {
        0.0
    }
}

/// How a measured pace compares to its accepted range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceComparison {
    /// Faster than [`PaceVerdict::high`].
    Faster,
    /// Slower than [`PaceVerdict::low`].
    Slower,
    /// Between [`PaceVerdict::low`] and [`PaceVerdict::high`].
    InRange,
}

/// Where a measured pace sits against a target, with the `--speed` that
/// reaches the target from the speed a take was made at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaceVerdict {
    /// The measured pace, rounded to one decimal: syllables per second.
    /// Rounded before it compares to [`Self::low`] and [`Self::high`], so a
    /// pace at the range's edge never prints as outside it.
    pub rate: f64,
    /// The target pace.
    pub target: f64,
    /// Low end of the accepted range: `target - 0.6`.
    pub low: f64,
    /// High end of the accepted range: `target + 0.6`.
    pub high: f64,
    /// The `--speed` that scales [`Self::rate`] to `target` from the speed
    /// this take was made at, held between 0.6 and 1.5.
    pub suggested_speed: f64,
}

impl PaceVerdict {
    /// How [`Self::rate`] compares to the range [`Self::low`] to
    /// [`Self::high`].
    pub fn comparison(&self) -> PaceComparison {
        if self.rate > self.high {
            PaceComparison::Faster
        } else if self.rate < self.low {
            PaceComparison::Slower
        } else {
            PaceComparison::InRange
        }
    }
}

/// The verdict on a raw measured pace against `target`, for a take made at
/// `speed` (the `--speed` already given the take, 1 when none).
pub fn pace_verdict(raw_pace: f64, target: f64, speed: f64) -> PaceVerdict {
    let rate = js_round(raw_pace * 10.0) / 10.0;
    let suggested_speed =
        js_round(js_min(1.5, js_max(0.6, (speed * target) / rate)) * 100.0) / 100.0;
    PaceVerdict {
        rate,
        target,
        low: js_to_fixed(target - PACE_RANGE, 1),
        high: js_to_fixed(target + PACE_RANGE, 1),
        suggested_speed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syllables_counts_vowel_groups() {
        assert_eq!(syllables("Berlumpur"), 3);
        assert_eq!(syllables("café"), 2);
        assert_eq!(syllables("13.5 percent"), 2);
        assert_eq!(syllables("Node.js"), 2);
        assert_eq!(syllables("kanak-kanak"), 4);
        assert_eq!(syllables("well-known"), 2);
        assert_eq!(syllables(""), 0);
        assert_eq!(syllables("123"), 0);
        assert_eq!(syllables("你好"), 0);
    }

    #[test]
    fn sounding_trims_silence_at_the_edges() {
        let mut audio = vec![0.0f32; 4800];
        for s in &mut audio[960..2880] {
            *s = 0.3;
        }
        // Loud from 0.02s to 0.06s: frames 2..6, so 0.04s (4 hops) sound.
        assert_eq!(sounding(&audio, 0.0, 0.1, 48000), 0.04);
        assert_eq!(sounding(&vec![0.0f32; 4800], 0.0, 0.1, 48000), 0.0);
        assert_eq!(sounding(&audio, 0.0, 0.1, 0), 0.0);
        assert_eq!(sounding(&audio, f64::NAN, 0.1, 48000), 0.0);
    }

    #[test]
    fn pace_counts_syllables_over_speaking_time() {
        let phrases = vec![
            PaceSpan {
                text: "Berlumpur".to_string(),
                start: 0.0,
                end: 1.0,
            },
            PaceSpan {
                text: "café".to_string(),
                start: 1.0,
                end: 2.0,
            },
        ];
        // 5 syllables over 2 seconds without audio: pauses count as speech.
        assert_eq!(pace(&phrases, None), 2.5);
        assert_eq!(pace(&[], None), 0.0);
    }

    #[test]
    fn pace_verdict_rounds_before_comparing() {
        let v = pace_verdict(4.75, DEFAULT_TARGET_PACE, 1.0);
        assert_eq!(v.rate, 4.8);
        assert_eq!(v.low, 3.5);
        assert_eq!(v.high, 4.7);
        assert_eq!(v.comparison(), PaceComparison::Faster);
        // A speed-1 take at 4.8 needs 0.85 to reach 4.1.
        assert_eq!(v.suggested_speed, 0.85);
        let slow = pace_verdict(3.0, DEFAULT_TARGET_PACE, 1.0);
        assert_eq!(slow.comparison(), PaceComparison::Slower);
        assert_eq!(slow.suggested_speed, 1.37);
        let in_range = pace_verdict(4.1, DEFAULT_TARGET_PACE, 1.0);
        assert_eq!(in_range.comparison(), PaceComparison::InRange);
        assert_eq!(in_range.suggested_speed, 1.0);
    }
}
