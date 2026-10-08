//! A whole speak run from script to voiceover: the skill's `speak()`. The
//! engine is resolved once, then one branch runs. A speed other than 1 on
//! an engine that is neither VoxCPM2 nor Kokoro speaks at speed 1 and
//! time-stretches the track. A timed script speaks its lines and places
//! each at its time. Every other run is one engine run.

use super::dispatch_run::{RunInput, engine_run};
use super::dispatch_setup::{ScriptStep, SpeakRuns, SpeakTarget, SpokenScript};
use super::dispatch_time::{placed, stretched};
use super::place::cue_script;
use super::speak_opts::{SpeakOpts, TtsConfig};
use super::speak_resolve::resolve_speak;
use crate::error::Result;
use crate::pattern::Patterns;
use crate::text::without_tags;

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
/// the script without its tags, and `runs` is told so unless the run is a
/// stretched or per-line one.
///
/// - A speed other than 1, on a run not yet stretched, with an engine that
///   is neither VoxCPM2 nor Kokoro: the engine speaks at speed 1, the track
///   is time-stretched, and each span and check's `at` is divided by the
///   speed. With `opts.cues`, the lines are then placed at their times.
///   This branch needs the `stretch` feature.
/// - `opts.cues`: the lines, joined by line breaks, are spoken with `lines`
///   on and placed at their times ([`place_on_times`](super::place::place_on_times)). `timing` holds the
///   lines that start late or overrun. An engine with no phrase spans adds
///   [`UNPLACED_PROBLEM`](super::place::UNPLACED_PROBLEM) instead.
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
    if resolved.log_tag_notice {
        runs.note(ScriptStep::TagsStripped);
    }
    let script = untagged(script, resolved.voxcpm())?;
    let input = RunInput {
        opts,
        tts,
        target,
        resolved: &resolved,
    };
    let spoken = if resolved.stretches(opts.stretched) {
        stretched(&script, &input, runs)?
    } else if let Some(cues) = opts.cues {
        let lines = untagged(&cue_script(cues), resolved.voxcpm())?;
        let (spoken, audio) = engine_run(&lines, &input, resolved.speed, true, runs)?.split();
        placed(spoken, audio, cues, &input)?
    } else {
        engine_run(&script, &input, resolved.speed, opts.lines, runs)?.spoken
    };
    Ok(spoken)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::check::HeardWord;
    use crate::error::Error;
    use crate::take::anchor_files::{OUT_NEXT, write_whole_with};
    use crate::take::dispatch_setup::ScriptChecks;
    use crate::take::engine::{SpeakRequest, SpeechEngine, Take};
    use crate::take::hear::tests::{ScriptedHearing, hearing};
    use crate::take::join::JoinedTrack;
    use crate::take::join_checks::PartCheck;
    use crate::take::simple::speak_phrases;
    use crate::take::simple_setup::{SimpleRun, SimpleSetup, SimpleStep};
    use crate::take::speak_setup::{VoxcpmRun, VoxcpmSetup};
    use crate::text::TimedCue;
    use crate::text::{ChunkOptions, chunks};
    use crate::wav::pcm16_wav;

    /// Speaks each text as 2400 samples of 0.25 at `rate`, recording the
    /// text and speed.
    pub(crate) struct Tone {
        rate: u32,
        pub(crate) calls: Vec<(String, f64)>,
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
    pub(crate) struct Runs {
        pub(crate) tone: Tone,
        hearing: ScriptedHearing,
        pub(crate) voxcpm: Vec<(String, bool, Option<String>)>,
        pub(crate) one_call: Vec<bool>,
        pub(crate) notes: Vec<ScriptStep>,
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
            let notes = &mut self.notes;
            let mut steps = |s: SimpleStep| notes.push(ScriptStep::Phrases(s));
            speak_phrases(script, setup, &mut self.tone, &mut self.hearing, &mut steps)
        }

        fn note(&mut self, step: ScriptStep) {
            self.notes.push(step);
        }
    }

    pub(crate) fn runs(rate: u32) -> Runs {
        Runs {
            tone: Tone {
                rate,
                calls: Vec::new(),
            },
            hearing: hearing(&[("One two. Three four.", false)], &[]),
            voxcpm: Vec::new(),
            one_call: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub(crate) fn speak(
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
            asr_model: None,
            out: &out,
            cache_dir: dir.path(),
            checkpoint: None,
            voxcpm_ready: ready,
            voxcpm_why: "needs an NVIDIA GPU",
        };
        speak_script(script, opts, tts, &target, runs)
    }

    pub(crate) fn cue(start: f64, text: &str) -> TimedCue {
        TimedCue {
            start,
            end: None,
            text: text.to_string(),
        }
    }

    pub(crate) const COMMAND: Option<&str> = Some("tts {text} {out}");

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
        assert_eq!(r.notes, [ScriptStep::TagsStripped]);
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
        assert!(r.notes.is_empty());
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
