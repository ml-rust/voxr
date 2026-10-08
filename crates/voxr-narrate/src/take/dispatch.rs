//! A whole speak run from script to voiceover: the skill's `speak()`. The
//! engine is resolved once, then one branch runs. A speed other than 1 on
//! an engine that is neither VoxCPM2 nor Kokoro speaks at speed 1 and
//! time-stretches the track. A timed script speaks its lines and places
//! each at its time. Every other run is one engine run.

use super::anchor_files::{OUT_NEXT, write_whole_with};
use super::dispatch_run::{Audio, RunInput, engine_run};
use super::dispatch_setup::{ScriptStep, SpeakRuns, SpeakTarget, SpokenScript};
use super::place::{PLACE_RATE, UNPLACED_PROBLEM, cue_script, place_on_times};
use super::speak_opts::{SpeakOpts, TtsConfig};
use super::speak_resolve::resolve_speak;
use crate::error::Result;
use crate::pattern::Patterns;
use crate::text::{TimedCue, without_tags};
use crate::wav::pcm16_wav;

/// `script` without its tags when the engine reads them as words and the
/// script holds one. Any other script stays as it is.
fn untagged(script: &str, voxcpm: bool) -> Result<String> {
    if !voxcpm && Patterns::get()?.tags.is_match(script) {
        without_tags(script)
    } else {
        Ok(script.to_string())
    }
}

/// Speaks `script` into the voiceover at `target.out`: the skill's
/// `speak()`.
///
/// The engine, speed and tag decision come from [`resolve_speak`] over
/// `opts` and the user's config `tts`. An engine other than VoxCPM2 speaks
/// the script without its tags, and the log notes it unless the run is a
/// stretched or per-line one.
///
/// - A speed other than 1, on a run not yet stretched, with an engine that
///   is neither VoxCPM2 nor Kokoro: the engine speaks at speed 1, the track
///   is time-stretched, and each span and check's `at` is divided by the
///   speed. With `opts.cues`, the lines are then placed at their times.
///   This branch needs the `stretch` feature.
/// - `opts.cues`: the lines, joined by line breaks, are spoken with `lines`
///   on and placed at their times ([`place_on_times`]). `timing` holds the
///   lines that start late or overrun. An engine with no phrase spans adds
///   [`UNPLACED_PROBLEM`] instead.
/// - Otherwise one engine run ([`SpeakRuns`]). VoxCPM2 is refused where it
///   is not ready, unless `opts.device` is set.
///
/// Errors name the invalid engine or speed, a missing reference, and what
/// the engine run, the stretch or the placement refuses.
pub fn speak_script<R: SpeakRuns + ?Sized>(
    script: &str,
    opts: &SpeakOpts<'_>,
    tts: &TtsConfig<'_>,
    target: &SpeakTarget<'_>,
    runs: &mut R,
) -> Result<SpokenScript> {
    let resolved = resolve_speak(script, opts, tts, target.voxcpm_ready, target.voxcpm_why)?;
    let mut log = Vec::new();
    if resolved.log_tag_notice {
        log.push(ScriptStep::TagsStripped);
    }
    let script = untagged(script, resolved.voxcpm())?;
    let input = RunInput {
        opts,
        tts,
        target,
        resolved: &resolved,
    };
    let mut spoken = if resolved.stretches(opts.stretched) {
        stretched(&script, &input, runs)?
    } else if let Some(cues) = opts.cues {
        let lines = untagged(&cue_script(cues), resolved.voxcpm())?;
        let (spoken, audio) = engine_run(&lines, &input, resolved.speed, true, runs)?.split();
        placed(spoken, audio, cues, &input)?
    } else {
        engine_run(&script, &input, resolved.speed, opts.lines, runs)?.spoken
    };
    log.append(&mut spoken.log);
    spoken.log = log;
    Ok(spoken)
}

/// Each line of `cues` placed at its time on `audio`, the placed track
/// written to `out`. A run with no phrase spans stays as it is, with
/// [`UNPLACED_PROBLEM`] added.
fn placed(
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
fn stretched<R: SpeakRuns + ?Sized>(
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
fn stretched<R: SpeakRuns + ?Sized>(
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
    use crate::check::HeardWord;
    use crate::error::Error;
    use crate::take::dispatch_setup::ScriptChecks;
    use crate::take::engine::{SpeakRequest, SpeechEngine, Take};
    use crate::take::hear::tests::{ScriptedHearing, hearing};
    use crate::take::join::JoinedTrack;
    use crate::take::join_checks::PartCheck;
    use crate::take::simple::speak_phrases;
    use crate::take::simple_setup::{SimpleRun, SimpleSetup};
    use crate::take::speak_setup::{VoxcpmRun, VoxcpmSetup};
    use crate::text::{ChunkOptions, chunks};

    /// Speaks each text as 2400 samples of 0.25 at `rate`, recording the
    /// text and speed.
    struct Tone {
        rate: u32,
        calls: Vec<(String, f64)>,
    }

    impl SpeechEngine for Tone {
        type Reference = ();

        fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
            self.calls.push((request.text.to_string(), request.speed));
            Ok(Take {
                samples: vec![0.25; 2400],
                rate: self.rate,
            })
        }
    }

    /// Runs every engine but VoxCPM2 through [`speak_phrases`] with
    /// [`Tone`], and stubs VoxCPM2 with one 0.5 s part per chunk.
    struct Runs {
        tone: Tone,
        hearing: ScriptedHearing,
        voxcpm: Vec<(String, bool, Option<String>)>,
        one_call: Vec<bool>,
    }

    impl SpeakRuns for Runs {
        fn voxcpm(&mut self, script: &str, setup: &VoxcpmSetup<'_>) -> Result<VoxcpmRun> {
            self.voxcpm.push((
                script.to_string(),
                setup.lines,
                setup.device.map(str::to_string),
            ));
            let options = ChunkOptions {
                lines: setup.lines,
                ..ChunkOptions::default()
            };
            let parts = chunks(script, options)?;
            let phrases: Vec<HeardWord> = (0..parts.len())
                .map(|i| HeardWord {
                    text: parts[i].text.clone(),
                    start: i as f64,
                    end: i as f64 + 0.5,
                })
                .collect();
            let samples = vec![0.25; parts.len() * 48000];
            write_whole_with(setup.out, &pcm16_wav(&samples, 48000)?, OUT_NEXT)?;
            Ok(VoxcpmRun {
                seed: 7.0,
                track: JoinedTrack {
                    duration: parts.len() as f64,
                    samples,
                    phrases,
                    starts: Vec::new(),
                    heat: Vec::new(),
                },
                checks: vec![PartCheck {
                    at: 0.0,
                    text: script.to_string(),
                    ok: true,
                    heard: String::new(),
                    why: String::new(),
                    seed: 7.0,
                    doubt: Vec::new(),
                    heat: 0.0,
                    hot_text: None,
                    pauses: Vec::new(),
                    clicks: Vec::new(),
                }],
                parts,
                problems: Vec::new(),
                voice_file: None,
                voice_median: 0.0,
                log: Vec::new(),
            })
        }

        fn phrases(&mut self, script: &str, setup: &SimpleSetup<'_>) -> Result<SimpleRun> {
            self.one_call.push(setup.one_call);
            speak_phrases(script, setup, &mut self.tone, &mut self.hearing)
        }
    }

    fn runs(rate: u32) -> Runs {
        Runs {
            tone: Tone {
                rate,
                calls: Vec::new(),
            },
            hearing: hearing(&[("One two. Three four.", false)], &[]),
            voxcpm: Vec::new(),
            one_call: Vec::new(),
        }
    }

    fn speak(
        script: &str,
        opts: &SpeakOpts<'_>,
        tts: &TtsConfig<'_>,
        ready: bool,
        runs: &mut Runs,
    ) -> Result<SpokenScript> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let out = dir.path().join("voiceover.wav");
        let target = SpeakTarget {
            out: &out,
            cache_dir: dir.path(),
            checkpoint: None,
            voxcpm_ready: ready,
            voxcpm_why: "needs an NVIDIA GPU",
        };
        speak_script(script, opts, tts, &target, runs)
    }

    fn cue(start: f64, text: &str) -> TimedCue {
        TimedCue {
            start,
            end: None,
            text: text.to_string(),
        }
    }

    const COMMAND: Option<&str> = Some("tts {text} {out}");

    #[test]
    fn kokoro_by_default_speaks_at_its_own_speed() -> Result<()> {
        let mut r = runs(24000);
        let opts = SpeakOpts {
            speed: Some(1.25),
            ..SpeakOpts::default()
        };
        let s = speak(
            "[sigh] One two.",
            &opts,
            &TtsConfig::default(),
            false,
            &mut r,
        )?;
        assert_eq!(s.engine, "Kokoro (af_heart)");
        assert_eq!(r.tone.calls, [("One two.".to_string(), 1.25)]);
        assert_eq!(s.log, [ScriptStep::TagsStripped]);
        assert!(s.timing.is_none() && s.voice_file.is_none());
        Ok(())
    }

    #[test]
    fn voxcpm_keeps_tags_and_is_never_stretched() -> Result<()> {
        let mut r = runs(48000);
        let opts = SpeakOpts {
            speed: Some(0.8),
            ..SpeakOpts::default()
        };
        let s = speak(
            "[sigh] One two.",
            &opts,
            &TtsConfig::default(),
            true,
            &mut r,
        )?;
        assert_eq!(s.engine, "VoxCPM2");
        assert_eq!(r.voxcpm, [("[sigh] One two.".to_string(), false, None)]);
        assert_eq!(s.duration, 1.0);
        assert!(s.log.is_empty());
        assert_eq!(s.voice_file, Some(None));
        assert!(matches!(s.checks, ScriptChecks::Parts(_)));
        Ok(())
    }

    #[test]
    fn voxcpm_where_it_is_not_ready_needs_a_device() -> Result<()> {
        let mut r = runs(48000);
        let mut opts = SpeakOpts {
            engine: Some("voxcpm"),
            ..SpeakOpts::default()
        };
        let err = speak("One two.", &opts, &TtsConfig::default(), false, &mut r)
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'engine': voxcpm needs an NVIDIA GPU. Use the default Kokoro engine, another TTS with --command, or --device cpu (very slow)."
            )
        );
        opts.device = Some("cpu");
        speak("One two.", &opts, &TtsConfig::default(), false, &mut r)?;
        assert_eq!(r.voxcpm[0].2.as_deref(), Some("cpu"));
        Ok(())
    }

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

    #[test]
    fn an_invalid_engine_or_speed_is_refused() {
        let mut r = runs(48000);
        let bad = |opts: SpeakOpts<'_>, r: &mut Runs| {
            speak("One.", &opts, &TtsConfig::default(), false, r).err()
        };
        let engine = SpeakOpts {
            engine: Some("piper"),
            ..SpeakOpts::default()
        };
        assert!(matches!(
            bad(engine, &mut r),
            Some(Error::InvalidArgument { arg: "engine", .. })
        ));
        let speed = SpeakOpts {
            speed: Some(2.5),
            ..SpeakOpts::default()
        };
        assert!(matches!(
            bad(speed, &mut r),
            Some(Error::InvalidArgument { arg: "speed", .. })
        ));
        assert!(r.tone.calls.is_empty());
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

    #[test]
    fn a_missing_reference_is_refused() {
        let mut r = runs(48000);
        let opts = SpeakOpts {
            reference: Some("/no/such/recording.wav"),
            ..SpeakOpts::default()
        };
        let err = speak("One two.", &opts, &TtsConfig::default(), true, &mut r);
        assert!(matches!(err, Err(Error::File { .. })));
        assert!(r.voxcpm.is_empty());
    }
}
