//! A spoken track slowed or quickened without a pitch change, as the skill's
//! `stretch` runs ffmpeg: `atempo` at the track's own rate, then 16-bit
//! mono at 48 kHz. Also the phrase spans and checks rescaled with it.
//!
//! `atempo` runs through [`voxr::atempo_pcm16`], which matches ffmpeg's
//! filter sample for sample. A track at another rate goes to 48 kHz
//! through [`resample_48k`], which matches ffmpeg's `-ar 48000` sample for
//! sample. The stretch therefore matches the skill's at every rate.

use crate::check::HeardWord;
use crate::error::{Error, Result};
use crate::js::js_to_fixed;
use crate::polish::pcm16_like_js;
use crate::take::simple_setup::WholeCheck;
use crate::wav::{pcm16_wav_raw, probed_duration};

/// The rate a stretched track is written at.
pub const STRETCH_RATE: u32 = 48000;

/// A 16-bit value `v` read as `v / 32768`, as ffmpeg converts `s16` to
/// `flt`.
const S16_TO_FLOAT: f32 = 1.0 / 32768.0;

/// A spoken track after [`stretch_track`].
#[derive(Debug, Clone, PartialEq)]
pub struct StretchedTrack {
    /// The 16-bit values the stretched WAV holds.
    pub pcm: Vec<i16>,
    /// Their rate: [`STRETCH_RATE`].
    pub rate: u32,
    /// The stretched track's length in seconds, as ffprobe reports it.
    pub duration: f64,
    /// Each phrase's span on the stretched clock, or `None` when the run
    /// has no spans.
    pub phrases: Option<Vec<HeardWord>>,
}

impl StretchedTrack {
    /// The samples as ffmpeg decodes the stretched WAV: each value over
    /// 32768. [`place_on_times`](super::place_on_times) places these.
    pub fn samples(&self) -> Vec<f32> {
        self.pcm
            .iter()
            .map(|&v| f32::from(v) * S16_TO_FLOAT)
            .collect()
    }

    /// The stretched track as a mono 16-bit WAV behind a 44-byte header.
    /// ffmpeg's own file also holds a LIST chunk naming the muxer, so the
    /// bytes differ there and the samples do not.
    pub fn wav(&self) -> Result<Vec<u8>> {
        pcm16_wav_raw(&self.pcm, self.rate)
    }
}

/// Each span's start and end divided by `speed`, to 3 decimals.
pub fn stretch_phrases(phrases: &[HeardWord], speed: f64) -> Vec<HeardWord> {
    phrases
        .iter()
        .map(|s| HeardWord {
            text: s.text.clone(),
            start: js_to_fixed(s.start / speed, 3),
            end: js_to_fixed(s.end / speed, 3),
        })
        .collect()
}

/// Each check's start divided by `speed`, to 2 decimals.
pub fn stretch_checks(checks: &[WholeCheck], speed: f64) -> Vec<WholeCheck> {
    checks
        .iter()
        .map(|c| WholeCheck {
            at: js_to_fixed(c.at / speed, 2),
            ..c.clone()
        })
        .collect()
}

/// `samples` at `rate` Hz brought to 48 kHz as the skill's ffmpeg decode
/// (`-ac 1 -ar 48000`) brings them, through
/// [`voxr::resample_like_ffmpeg`]: same length, same samples. A 48 kHz
/// track comes back unchanged.
pub fn resample_48k(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    voxr::resample_like_ffmpeg(samples, rate, STRETCH_RATE).map_err(|e| Error::DataError {
        reason: format!(
            "resampling {} samples from {rate} Hz to {STRETCH_RATE} Hz: {e}. Pass audio at a common rate such as 16, 24 or 44.1 kHz",
            samples.len()
        ),
    })
}

/// A float sample as ffmpeg converts `flt` to `s16`: scaled by 32768,
/// rounded to nearest with ties to even, clipped to 16 bits.
fn float_to_s16(v: f32) -> i16 {
    let r = (v * 32768.0).round_ties_even();
    if r.is_nan() {
        0
    } else {
        r.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
    }
}

/// The skill's `stretch`: the spoken track slowed or quickened by `speed`
/// without a pitch change, at [`STRETCH_RATE`], and its phrase spans with
/// it.
///
/// `samples` is the track at `rate` Hz as joined before the skill's WAV
/// rounds it (`SimpleRun::samples`). It is rounded to 16 bits as that WAV
/// holds it, time-stretched at `rate`, then brought to 48 kHz. `speed`
/// above 1 shortens the track. Only `0.5..=2` is accepted.
pub fn stretch_track(
    samples: &[f32],
    rate: u32,
    speed: f64,
    phrases: Option<&[HeardWord]>,
) -> Result<StretchedTrack> {
    let stretched = voxr::atempo_pcm16(&pcm16_like_js(samples), rate, speed).map_err(|e| {
        Error::InvalidArgument {
            arg: "speed",
            reason: format!(
                "time-stretching {} samples at {rate} Hz by {speed}: {e}. Use a speed from 0.5 to 2 and a rate from 48 Hz to 768 kHz",
                samples.len()
            ),
        }
    })?;
    let pcm = if rate == STRETCH_RATE {
        stretched
    } else {
        let float: Vec<f32> = stretched
            .iter()
            .map(|&v| f32::from(v) * S16_TO_FLOAT)
            .collect();
        resample_48k(&float, rate)?
            .into_iter()
            .map(float_to_s16)
            .collect()
    };
    Ok(StretchedTrack {
        duration: probed_duration(pcm.len(), STRETCH_RATE),
        rate: STRETCH_RATE,
        phrases: phrases.map(|p| stretch_phrases(p, speed)),
        pcm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, start: f64, end: f64) -> HeardWord {
        HeardWord {
            text: text.to_string(),
            start,
            end,
        }
    }

    /// A 200 Hz tone of `secs` seconds at `rate` over a faint ramp, so no
    /// segment is digital silence.
    fn tone(secs: f64, rate: u32) -> Vec<f32> {
        let n = (secs * f64::from(rate)) as usize;
        let w = std::f64::consts::TAU * 200.0 / f64::from(rate);
        (0..n)
            .map(|i| (0.3 * (w * i as f64).sin() + 1e-4 * (i % 7) as f64) as f32)
            .collect()
    }

    #[test]
    fn spans_divide_by_the_speed() {
        let got = stretch_phrases(&[span("One.", 1.0, 2.5)], 1.5);
        assert_eq!(got, [span("One.", 0.667, 1.667)]);
    }

    #[test]
    fn checks_divide_by_the_speed() {
        let check = WholeCheck {
            at: 1.0,
            text: "One.".to_string(),
            ok: true,
            heard: "One.".to_string(),
            why: String::new(),
            doubt: Vec::new(),
            pauses: Vec::new(),
        };
        let got = stretch_checks(std::slice::from_ref(&check), 0.8);
        assert_eq!(got, [WholeCheck { at: 1.25, ..check }]);
    }

    #[test]
    fn float_samples_round_like_ffmpeg() {
        assert_eq!(float_to_s16(0.5), 16384);
        assert_eq!(float_to_s16(1.5 / 32768.0), 2);
        assert_eq!(float_to_s16(2.5 / 32768.0), 2);
        assert_eq!(float_to_s16(1.0), i16::MAX);
        assert_eq!(float_to_s16(-2.0), i16::MIN);
        assert_eq!(float_to_s16(f32::NAN), 0);
    }

    #[test]
    fn a_48k_track_is_written_at_48k() -> Result<()> {
        let x = tone(1.0, STRETCH_RATE);
        let spans = [span("One.", 0.2, 0.8)];
        let r = stretch_track(&x, STRETCH_RATE, 1.25, Some(&spans[..]))?;
        assert_eq!(r.rate, STRETCH_RATE);
        assert_eq!(r.duration, probed_duration(r.pcm.len(), STRETCH_RATE));
        assert_eq!(r.phrases, Some(vec![span("One.", 0.16, 0.64)]));
        assert_eq!(r.samples().len(), r.pcm.len());
        Ok(())
    }

    #[test]
    fn a_24k_track_doubles_in_length_after_atempo() -> Result<()> {
        let x = tone(1.0, 24000);
        let at_24k =
            voxr::atempo_pcm16(&pcm16_like_js(&x), 24000, 0.8).map_err(|e| Error::DataError {
                reason: e.to_string(),
            })?;
        let r = stretch_track(&x, 24000, 0.8, None)?;
        assert_eq!(r.pcm.len(), 2 * at_24k.len());
        assert_eq!(r.phrases, None);
        Ok(())
    }

    #[test]
    fn a_speed_out_of_range_is_refused() {
        let r = stretch_track(&tone(0.5, STRETCH_RATE), STRETCH_RATE, 3.0, None);
        assert!(matches!(
            r,
            Err(Error::InvalidArgument { arg: "speed", .. })
        ));
    }
}
