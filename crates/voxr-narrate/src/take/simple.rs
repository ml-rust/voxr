//! The speak run of every engine but VoxCPM2: Kokoro, a Hugging Face model
//! and the user's TTS command. Each phrase is spoken apart with its pause
//! after it, the joined audio is written as the voiceover, and the whole
//! file is heard back once. These engines keep no per-phrase takes, so a
//! failed check is reported, never retried.

use super::anchor_files::{OUT_NEXT, write_whole_with};
use super::engine::{SpeakRequest, SpeechEngine, VoiceSource};
use super::files::DEFAULT_SEED;
use super::hear::{Hearing, asr_model, heard_check};
use super::simple_setup::{SimpleRun, SimpleSetup, SimpleStep, WholeCheck};
use super::speak_resolve::EngineChoice;
use crate::check::HeardWord;
use crate::error::{Error, Result};
use crate::js::{js_round, js_to_fixed};
use crate::text::{Phrase, phrases, say, sayable, spoken};
use crate::wav::{pcm16_round_trip, pcm16_wav};

/// Kokoro's sample rate before its first take reports one.
const KOKORO_RATE: u32 = 24000;
/// A Hugging Face model's sample rate before its first take reports one.
const MODEL_RATE: u32 = 16000;
/// The rate the user's command's output is decoded at.
const COMMAND_RATE: u32 = 48000;
/// Decimals of a phrase span's times.
const SPAN_DECIMALS: usize = 3;
/// VoxCPM2's guidance and steps when nothing sets them. The engines here
/// read neither.
pub(super) const CFG: f64 = 1.6;
pub(super) const STEPS: u32 = 16;

/// `new Float32Array(Math.round(pause * rate))`: the silence after a
/// phrase, in samples.
fn pause_samples(phrase: &Phrase, rate: u32) -> Result<usize> {
    let n = js_round(phrase.pause * f64::from(rate));
    if n.is_nan() || n < 0.0 {
        return Err(Error::DataError {
            reason: format!(
                "the phrase \"{}\" asks for a pause of {} s. Give each [pause] mark a positive number of seconds",
                phrase.text, phrase.pause
            ),
        });
    }
    Ok(n as usize)
}

/// Where each phrase sits in the joined audio: the skill's `spans`. Each
/// entry of `lengths` is a phrase's audio and pause, in samples, and
/// `rate` the rate all of them are read at.
fn spans(list: &[Phrase], lengths: &[(usize, usize)], rate: u32) -> Result<Vec<HeardWord>> {
    let rate = f64::from(rate);
    let mut t = 0.0;
    let mut out = Vec::with_capacity(list.len());
    for (p, &(audio, pause)) in list.iter().zip(lengths) {
        let len = audio as f64 / rate;
        out.push(HeardWord {
            text: spoken(&p.text)?,
            start: js_to_fixed(t, SPAN_DECIMALS),
            end: js_to_fixed(t + len, SPAN_DECIMALS),
        });
        t += len + pause as f64 / rate;
    }
    Ok(out)
}

/// Speaks `script` with Kokoro, a Hugging Face model or the user's
/// command, writes the voiceover to `setup.out`, and hears it back: the
/// non-VoxCPM2 tail of the skill's `speak()`.
///
/// Each phrase ([`phrases`]) is one call to `engine` with its text after
/// [`say`], followed by `pause * rate` samples of silence at that take's
/// rate. Kokoro is asked for its voice by name and at `setup.speed`. Every
/// other engine is asked for its own voice at speed 1. With
/// `setup.one_call`, the command speaks the whole [`sayable`] script in one
/// call, and the run has no phrase spans. The command logs its progress
/// after each phrase of a script of more than one.
///
/// The joined audio is written as a 16-bit WAV at the last take's rate,
/// then heard back once through `hearing` ([`heard_check`]) against the
/// [`sayable`] script. `script` is the script after tags are stripped, as
/// [`resolve_speak`](super::resolve_speak) decides.
pub fn speak_phrases<E, H>(
    script: &str,
    setup: &SimpleSetup<'_>,
    engine: &mut E,
    hearing: &mut H,
) -> Result<SimpleRun>
where
    E: SpeechEngine + ?Sized,
    H: Hearing + ?Sized,
{
    let (voice, default_rate, speed, command) = match setup.engine {
        EngineChoice::Voxcpm { .. } => {
            return Err(Error::InvalidArgument {
                arg: "engine",
                reason: "VoxCPM2 keeps a take per part. Speak it through speak_voxcpm".to_string(),
            });
        }
        EngineChoice::Kokoro { voice, .. } => (
            VoiceSource::Named { name: voice },
            KOKORO_RATE,
            setup.speed,
            false,
        ),
        EngineChoice::Model { .. } => (VoiceSource::Own { design: None }, MODEL_RATE, 1.0, false),
        EngineChoice::Command { .. } => {
            (VoiceSource::Own { design: None }, COMMAND_RATE, 1.0, true)
        }
    };
    let mut speak = |text: &str| {
        engine.speak(&SpeakRequest {
            text,
            delivery: None,
            style: None,
            voice,
            seed: DEFAULT_SEED,
            cfg: CFG,
            steps: STEPS,
            speed,
        })
    };
    let mut log = Vec::new();
    let (samples, rate, spans) = if command && setup.one_call {
        let take = speak(&sayable(script)?)?;
        (take.samples, take.rate, None)
    } else {
        let list = phrases(script)?;
        let mut samples = Vec::new();
        let mut lengths = Vec::with_capacity(list.len());
        let mut rate = default_rate;
        for (i, p) in list.iter().enumerate() {
            let take = speak(&say(&p.text)?)?;
            rate = take.rate;
            let pause = pause_samples(p, rate)?;
            lengths.push((take.samples.len(), pause));
            samples.extend_from_slice(&take.samples);
            samples.resize(samples.len() + pause, 0.0);
            if command && list.len() > 1 {
                log.push(SimpleStep::CommandProgress {
                    done: i + 1,
                    total: list.len(),
                });
            }
        }
        let spans = spans(&list, &lengths, rate)?;
        (samples, rate, Some(spans))
    };
    write_whole_with(setup.out, &pcm16_wav(&samples, rate)?, OUT_NEXT)?;
    let duration = samples.len() as f64 / f64::from(rate);
    let heard = heard_check(
        hearing,
        asr_model(setup.language, setup.asr_model),
        &pcm16_round_trip(&samples),
        rate,
        &sayable(script)?,
        setup.language,
    )?;
    let problems = if heard.check.ok {
        Vec::new()
    } else {
        vec![format!("{}. Heard: \"{}\".", heard.check.why, heard.heard)]
    };
    let checks = vec![WholeCheck {
        at: 0.0,
        text: spoken(script)?,
        ok: heard.check.ok,
        heard: heard.heard,
        why: heard.check.why,
        doubt: heard.doubt,
        pauses: heard.pauses,
    }];
    Ok(SimpleRun {
        samples,
        rate,
        duration,
        phrases: spans,
        engine: setup.engine.label(),
        checks,
        problems,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::engine::Take;
    use crate::take::hear::tests::{ScriptedHearing, hearing};

    /// Speaks each text as `n` samples of 0.25 at `rate`, `n` its length
    /// in bytes times 10, and records `(text, voice kind, speed)`.
    struct Tone {
        rate: u32,
        calls: Vec<(String, &'static str, f64)>,
    }

    impl SpeechEngine for Tone {
        type Reference = ();

        fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
            self.calls.push((
                request.text.to_string(),
                request.voice.kind(),
                request.speed,
            ));
            Ok(Take {
                samples: vec![0.25; request.text.len() * 10],
                rate: self.rate,
            })
        }
    }

    fn run(
        script: &str,
        choice: &EngineChoice,
        one_call: bool,
        rate: u32,
        heard: &mut ScriptedHearing,
    ) -> Result<(SimpleRun, Tone)> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let out = dir.path().join("voiceover.wav");
        let setup = SimpleSetup {
            asr_model: None,
            out: &out,
            engine: choice,
            speed: 1.25,
            one_call,
            language: None,
        };
        let mut engine = Tone {
            rate,
            calls: Vec::new(),
        };
        let r = speak_phrases(script, &setup, &mut engine, heard)?;
        Ok((r, engine))
    }

    fn kokoro() -> EngineChoice {
        EngineChoice::Kokoro {
            model: "onnx-community/Kokoro-82M-v1.0-ONNX".to_string(),
            voice: "af_heart".to_string(),
        }
    }

    fn command() -> EngineChoice {
        EngineChoice::Command {
            command: "tts {text} {out}".to_string(),
            voice: String::new(),
            reference: String::new(),
        }
    }

    #[test]
    fn kokoro_speaks_each_phrase_by_name_at_its_speed() -> Result<()> {
        let mut h = hearing(&[("One two. Three four.", false)], &[]);
        let (r, engine) = run("One two. Three four.", &kokoro(), false, 24000, &mut h)?;
        assert_eq!(
            engine.calls,
            [
                ("One two.".to_string(), "named", 1.25),
                ("Three four.".to_string(), "named", 1.25),
            ]
        );
        // 80 samples, 0.3 s of silence, 110 samples, no pause at the end.
        assert_eq!(r.samples.len(), 80 + 7200 + 110);
        let spans = r.phrases.unwrap_or_default();
        assert_eq!(
            spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            [(0.0, 0.003), (0.303, 0.308)]
        );
        assert_eq!(r.engine, "Kokoro (af_heart)");
        assert!(r.log.is_empty());
        assert!(r.problems.is_empty());
        assert_eq!(r.checks[0].text, "One two. Three four.");
        Ok(())
    }

    #[test]
    fn a_command_logs_its_progress_and_ignores_speed() -> Result<()> {
        let mut h = hearing(&[("One two. Three four.", false)], &[]);
        let (r, engine) = run("One two. Three four.", &command(), false, 48000, &mut h)?;
        assert!(engine.calls.iter().all(|c| c.1 == "own" && c.2 == 1.0));
        let log: Vec<String> = r.log.iter().map(SimpleStep::message).collect();
        assert_eq!(
            log,
            ["tts command: 1/2 phrases", "tts command: 2/2 phrases"]
        );
        assert_eq!(r.engine, "command: tts {text} {out}");
        Ok(())
    }

    #[test]
    fn one_call_speaks_the_whole_script_without_spans() -> Result<()> {
        let script = "One {1|one}.\n[pause 1.2] Two.";
        let mut h = hearing(&[("One one. Two.", false)], &[]);
        let (r, engine) = run(script, &command(), true, 48000, &mut h)?;
        assert_eq!(engine.calls.len(), 1);
        assert_eq!(engine.calls[0].0, "One one. Two.");
        assert_eq!(r.phrases, None);
        assert!(r.log.is_empty());
        Ok(())
    }

    #[test]
    fn a_failed_check_is_a_problem() -> Result<()> {
        let mut h = hearing(&[("One", false)], &[]);
        let (r, _) = run("One two three.", &kokoro(), false, 24000, &mut h)?;
        assert!(!r.checks[0].ok);
        assert_eq!(
            r.problems,
            [format!("{}. Heard: \"One\".", r.checks[0].why)]
        );
        Ok(())
    }

    #[test]
    fn voxcpm_is_refused() {
        let mut h = hearing(&[], &[]);
        let voxcpm = EngineChoice::Voxcpm {
            voice: String::new(),
        };
        assert!(matches!(
            run("Hi.", &voxcpm, false, 48000, &mut h),
            Err(Error::InvalidArgument { arg: "engine", .. })
        ));
    }
}
