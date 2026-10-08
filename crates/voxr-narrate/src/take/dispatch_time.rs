//! The two branches of a speak run that move its audio in time: the
//! track time-stretched to the resolved speed, and a timed script's lines
//! placed at their times.

use super::anchor_files::{OUT_NEXT, write_whole_with};
#[cfg(feature = "stretch")]
use super::dispatch_run::engine_run;
use super::dispatch_run::{Audio, RunInput};
use super::dispatch_setup::{SpeakRuns, SpokenScript};
use super::place::{PLACE_RATE, UNPLACED_PROBLEM, place_on_times};
use crate::error::Result;
use crate::text::TimedCue;
use crate::wav::pcm16_wav;

/// Each line of `cues` placed at its time on `audio`, the placed track
/// written to `out`. A run with no phrase spans stays as it is, with
/// [`UNPLACED_PROBLEM`] added.
pub(super) fn placed(
    mut spoken: SpokenScript,
    audio: Audio,
    cues: &[TimedCue],
    input: &RunInput<'_>,
) -> Result<SpokenScript> {
    let Some(phrases) = spoken.phrases.as_deref() else {
        spoken.problems.push(UNPLACED_PROBLEM.to_string());
        return Ok(spoken);
    };
    let units = input.resolved.units();
    let p = place_on_times(&audio.decoded()?, PLACE_RATE, phrases, cues, units)?;
    write_whole_with(input.target.out, &pcm16_wav(&p.samples, p.rate)?, OUT_NEXT)?;
    spoken.duration = p.duration;
    spoken.phrases = Some(p.phrases);
    spoken.timing = Some(p.late);
    Ok(spoken)
}

/// The script spoken at speed 1, each line apart for a timed script, then
/// time-stretched to the resolved speed and written to `out`. With cues,
/// the lines are then placed at their times.
#[cfg(feature = "stretch")]
pub(super) fn stretched<R: SpeakRuns + ?Sized>(
    script: &str,
    input: &RunInput<'_>,
    runs: &mut R,
) -> Result<SpokenScript> {
    use super::dispatch_run::Track;
    use super::dispatch_setup::ScriptChecks;
    use super::join_checks::PartCheck;
    use super::stretch::{stretch_checks, stretch_track};
    use crate::js::js_to_fixed;

    let speed = input.resolved.speed;
    let cues = input.opts.cues;
    let Track {
        mut spoken,
        samples,
        rate,
    } = engine_run(script, input, 1.0, cues.is_some(), runs)?;
    let s = stretch_track(&samples, rate, speed, spoken.phrases.as_deref())?;
    write_whole_with(input.target.out, &s.wav()?, OUT_NEXT)?;
    spoken.duration = s.duration;
    spoken.checks = match spoken.checks {
        ScriptChecks::Whole(list) => ScriptChecks::Whole(stretch_checks(&list, speed)),
        ScriptChecks::Parts(list) => ScriptChecks::Parts(
            list.into_iter()
                .map(|c| PartCheck {
                    at: js_to_fixed(c.at / speed, 2),
                    ..c
                })
                .collect(),
        ),
    };
    let audio = Audio::Decoded(s.samples());
    spoken.phrases = s.phrases;
    match cues {
        Some(cues) => placed(spoken, audio, cues, input),
        None => Ok(spoken),
    }
}

/// Without the `stretch` feature, a speed that time-stretches is refused
/// before any audio is made.
#[cfg(not(feature = "stretch"))]
pub(super) fn stretched<R: SpeakRuns + ?Sized>(
    _script: &str,
    input: &RunInput<'_>,
    _runs: &mut R,
) -> Result<SpokenScript> {
    Err(crate::error::Error::InvalidArgument {
        arg: "speed",
        reason: format!(
            "{} time-stretches the {} track, which needs the `stretch` feature. Build voxr-narrate with --features stretch, or pass speed 1",
            input.resolved.speed,
            input.resolved.engine.label()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "stretch"))]
    use crate::error::Error;
    use crate::take::dispatch::tests::{COMMAND, cue, runs, speak};
    use crate::take::speak_opts::{SpeakOpts, TtsConfig};

    #[test]
    fn timed_lines_are_placed_at_their_times() -> Result<()> {
        let mut r = runs(48000);
        let cues = [cue(1.0, "One two."), cue(3.0, "Three four.")];
        let opts = SpeakOpts {
            command: COMMAND,
            cues: Some(&cues),
            ..SpeakOpts::default()
        };
        let s = speak("ignored", &opts, &TtsConfig::default(), false, &mut r)?;
        let starts: Vec<f64> = s.phrases.iter().flatten().map(|p| p.start).collect();
        assert_eq!(starts, [1.0, 3.0]);
        assert_eq!(s.timing, Some(Vec::new()));
        assert_eq!(s.duration, 3.05);
        Ok(())
    }

    #[test]
    fn a_one_call_command_cannot_place_its_lines() -> Result<()> {
        let mut r = runs(48000);
        let cues = [cue(0.0, "One two."), cue(2.0, "Three four.")];
        let tts = TtsConfig {
            one_call: Some(true),
            ..TtsConfig::default()
        };
        let opts = SpeakOpts {
            command: COMMAND,
            cues: Some(&cues),
            ..SpeakOpts::default()
        };
        let s = speak("ignored", &opts, &tts, false, &mut r)?;
        assert_eq!(r.one_call, [true]);
        assert_eq!(
            s.problems.last().map(String::as_str),
            Some(UNPLACED_PROBLEM)
        );
        assert!(s.timing.is_none() && s.phrases.is_none());
        Ok(())
    }

    #[cfg(feature = "stretch")]
    #[test]
    fn another_engine_is_stretched_then_placed() -> Result<()> {
        let mut r = runs(48000);
        let opts = SpeakOpts {
            command: COMMAND,
            speed: Some(1.25),
            ..SpeakOpts::default()
        };
        let s = speak(
            "One two. Three four.",
            &opts,
            &TtsConfig::default(),
            false,
            &mut r,
        )?;
        assert!(r.tone.calls.iter().all(|c| c.1 == 1.0));
        let spans: Vec<(f64, f64)> = s
            .phrases
            .iter()
            .flatten()
            .map(|p| (p.start, p.end))
            .collect();
        assert_eq!(spans, [(0.0, 0.04), (0.28, 0.32)]);
        let cues = [cue(1.0, "One two."), cue(3.0, "Three four.")];
        let timed = SpeakOpts {
            cues: Some(&cues),
            ..opts
        };
        let mut r = runs(48000);
        let s = speak(
            "One two.\nThree four.",
            &timed,
            &TtsConfig::default(),
            false,
            &mut r,
        )?;
        let starts: Vec<f64> = s.phrases.iter().flatten().map(|p| p.start).collect();
        assert_eq!(starts, [1.0, 3.0]);
        assert_eq!(s.timing, Some(Vec::new()));
        Ok(())
    }

    #[cfg(not(feature = "stretch"))]
    #[test]
    fn a_stretch_without_the_feature_is_refused_before_any_audio() {
        let mut r = runs(48000);
        let opts = SpeakOpts {
            command: COMMAND,
            speed: Some(1.25),
            ..SpeakOpts::default()
        };
        let err = speak("One two.", &opts, &TtsConfig::default(), false, &mut r);
        assert!(matches!(
            err,
            Err(Error::InvalidArgument { arg: "speed", ref reason }) if reason.contains("`stretch` feature")
        ));
        assert!(r.tone.calls.is_empty());
    }
}
