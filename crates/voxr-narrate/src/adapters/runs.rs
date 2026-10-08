//! [`NarrationRuns`]: the [`SpeakRuns`] a caller builds from loaded models.
//! It holds the engines and the hearing, and runs
//! [`speak_voxcpm`] and [`speak_phrases`] with them.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::path::Path;

use super::audio::decode_48k;
use super::work_dir::WorkDir;
use crate::error::{Error, Result};
use crate::take::{
    CommandEngine, EngineChoice, HeardJudge, Hearing, SimpleRun, SimpleSetup, SpeakRequest,
    SpeakRuns, SpeechEngine, Take, VoxcpmRun, VoxcpmSetup, asr_model, speak_phrases, speak_voxcpm,
};

/// Runs one shell line of the user's TTS command and returns its exit
/// code, `None` when a signal ends it.
pub type Spawn<'a> = Box<dyn FnMut(&str) -> Result<Option<i32>> + 'a>;

/// A speech engine that is not loaded: every take is refused.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoEngine;

impl SpeechEngine for NoEngine {
    type Reference = ();

    fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
        Err(Error::InvalidArgument {
            arg: "engine",
            reason: format!(
                "no engine is loaded to speak \"{}\". Load the engine the run picks and pass it to NarrationRuns",
                request.text
            ),
        })
    }
}

/// `Math.random()`: a value in `[0, 1)` from a hasher the standard library
/// keys at random.
fn random_unit() -> f64 {
    let bits = RandomState::new().build_hasher().finish() >> 11;
    bits as f64 / (1u64 << 53) as f64
}

/// The engines and the hearing a speak run goes through: the
/// [`SpeakRuns`] of [`speak_script`](crate::speak_script) over loaded
/// models.
///
/// - VoxCPM2 runs [`speak_voxcpm`] with the VoxCPM2 engine, a
///   [`HeardJudge`] over the hearing, and recordings decoded by
///   [`decode_48k`]. Its takes are kept in the setup's cache directory.
/// - Kokoro runs [`speak_phrases`] with the Kokoro engine.
/// - A Hugging Face model runs [`speak_phrases`] with the engine loaded
///   under the setup's model id.
/// - The user's command runs [`speak_phrases`] with a [`CommandEngine`]
///   over the spawner, in a fresh work directory removed after the run.
///
/// The recognizer heard first comes from the config's `asr.model`, else
/// the run's language ([`asr_model`]). The caller loads each engine on the device the run
/// asks for: a [`VoxcpmSetup::device`] is not read here. A run whose engine
/// is not loaded is refused before any audio is made.
pub struct NarrationRuns<'a, H, V = NoEngine> {
    voxcpm: Option<V>,
    kokoro: Option<Box<dyn SpeechEngine<Reference = ()> + 'a>>,
    models: Vec<(String, Box<dyn SpeechEngine<Reference = ()> + 'a>)>,
    hearing: H,
    spawn: Spawn<'a>,
    random: Box<dyn FnMut() -> f64 + 'a>,
}

impl<'a, H: Hearing> NarrationRuns<'a, H> {
    /// Runs that hear through `hearing` and run the user's TTS command
    /// through `spawn`, with no engine loaded. A re-rolled part's seed
    /// draws from a random source the standard library keys.
    pub fn new(hearing: H, spawn: impl FnMut(&str) -> Result<Option<i32>> + 'a) -> Self {
        Self {
            voxcpm: None,
            kokoro: None,
            models: Vec::new(),
            hearing,
            spawn: Box::new(spawn),
            random: Box::new(random_unit),
        }
    }
}

impl<'a, H: Hearing, V> NarrationRuns<'a, H, V> {
    /// The same runs with `engine` as VoxCPM2, such as a
    /// [`VoxcpmSpeech`](super::VoxcpmSpeech).
    pub fn with_voxcpm<W: SpeechEngine>(self, engine: W) -> NarrationRuns<'a, H, W> {
        NarrationRuns {
            voxcpm: Some(engine),
            kokoro: self.kokoro,
            models: self.models,
            hearing: self.hearing,
            spawn: self.spawn,
            random: self.random,
        }
    }

    /// The same runs with `engine` as Kokoro, such as a
    /// [`KokoroSpeech`](super::KokoroSpeech).
    pub fn with_kokoro(mut self, engine: impl SpeechEngine<Reference = ()> + 'a) -> Self {
        self.kokoro = Some(Box::new(engine));
        self
    }

    /// The same runs with `engine` as the Hugging Face model `id`, such as
    /// a [`VitsSpeech`](super::VitsSpeech) for `facebook/mms-tts-zlm`. A
    /// model loaded twice under one id keeps the later engine.
    pub fn with_model(
        mut self,
        id: impl Into<String>,
        engine: impl SpeechEngine<Reference = ()> + 'a,
    ) -> Self {
        let id = id.into();
        self.models.retain(|(held, _)| *held != id);
        self.models.push((id, Box::new(engine)));
        self
    }

    /// The same runs with re-rolled seeds drawn from `random`, a value in
    /// `[0, 1)` per call, as `Math.random()` gives.
    pub fn with_random(mut self, random: impl FnMut() -> f64 + 'a) -> Self {
        self.random = Box::new(random);
        self
    }

    /// The hearing every run checks its takes with.
    pub fn hearing(&mut self) -> &mut H {
        &mut self.hearing
    }
}

impl<H: Hearing, V: SpeechEngine> SpeakRuns for NarrationRuns<'_, H, V> {
    fn voxcpm(&mut self, script: &str, setup: &VoxcpmSetup<'_>) -> Result<VoxcpmRun> {
        let Some(engine) = self.voxcpm.as_mut() else {
            return Err(Error::InvalidArgument {
                arg: "engine",
                reason: "the run speaks with VoxCPM2, and no VoxCPM2 engine is loaded. Load one and pass it with NarrationRuns::with_voxcpm, or pick another engine".to_string(),
            });
        };
        let mut judge = HeardJudge {
            hearing: &mut self.hearing,
            main: asr_model(setup.language, setup.asr_model),
            language: setup.language.map(str::to_string),
        };
        let mut decode = |path: &Path| decode_48k(path);
        speak_voxcpm(
            script,
            setup,
            engine,
            &mut judge,
            &mut decode,
            &mut *self.random,
        )
    }

    fn phrases(&mut self, script: &str, setup: &SimpleSetup<'_>) -> Result<SimpleRun> {
        let hearing = &mut self.hearing;
        match setup.engine {
            EngineChoice::Kokoro { .. } => {
                let engine = self.kokoro.as_deref_mut().ok_or_else(|| Error::InvalidArgument {
                    arg: "engine",
                    reason: "the run speaks with Kokoro, and no Kokoro engine is loaded. Load one and pass it with NarrationRuns::with_kokoro, or pick another engine".to_string(),
                })?;
                speak_phrases(script, setup, engine, hearing)
            }
            EngineChoice::Model { model } => {
                let loaded: Vec<&str> = self.models.iter().map(|(id, _)| id.as_str()).collect();
                let listed = if loaded.is_empty() {
                    "none".to_string()
                } else {
                    loaded.join(", ")
                };
                let Some((_, engine)) = self.models.iter_mut().find(|(id, _)| id == model) else {
                    return Err(Error::InvalidArgument {
                        arg: "model",
                        reason: format!(
                            "the run speaks with the model \"{model}\", and it is not loaded (loaded: {listed}). Load it and pass it with NarrationRuns::with_model"
                        ),
                    });
                };
                speak_phrases(script, setup, engine.as_mut(), hearing)
            }
            EngineChoice::Command {
                command,
                voice,
                reference,
            } => {
                let dir = WorkDir::create()?;
                let mut engine = CommandEngine {
                    template: command,
                    voice,
                    reference,
                    dir: dir.path(),
                    run: &mut self.spawn,
                    decode: |path: &Path| decode_48k(path).ok(),
                    windows: cfg!(windows),
                };
                speak_phrases(script, setup, &mut engine, hearing)
            }
            EngineChoice::Voxcpm { .. } => speak_phrases(script, setup, &mut NoEngine, hearing),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::hear::tests::hearing;
    use crate::take::{SpeakOpts, SpeakTarget, TtsConfig, speak_script};

    /// Speaks each text as 2400 samples of 0.25 at 24 kHz, recording each
    /// text.
    #[derive(Default)]
    struct Tone {
        texts: Vec<String>,
    }

    impl SpeechEngine for &mut Tone {
        type Reference = ();

        fn speak(&mut self, request: &SpeakRequest<'_, ()>) -> Result<Take> {
            self.texts.push(request.text.to_string());
            Ok(Take {
                samples: vec![0.25; 2400],
                rate: 24_000,
            })
        }
    }

    fn speak(
        runs: &mut NarrationRuns<'_, impl Hearing>,
        opts: &SpeakOpts<'_>,
    ) -> Result<crate::take::SpokenScript> {
        let dir = tempfile::tempdir().map_err(|e| Error::File {
            reason: format!("creating a temporary directory: {e}"),
        })?;
        let out = dir.path().join("voiceover.wav");
        let target = SpeakTarget {
            asr_model: None,
            out: &out,
            cache_dir: dir.path(),
            checkpoint: None,
            voxcpm_ready: false,
            voxcpm_why: "needs an NVIDIA GPU",
        };
        speak_script("One two.", opts, &TtsConfig::default(), &target, runs)
    }

    #[test]
    fn kokoro_speaks_through_the_loaded_engine() -> Result<()> {
        let mut tone = Tone::default();
        let mut runs =
            NarrationRuns::new(hearing(&[("One two.", false)], &[]), |_: &str| Ok(Some(0)))
                .with_kokoro(&mut tone);
        let spoken = speak(&mut runs, &SpeakOpts::default())?;
        drop(runs);
        assert_eq!(spoken.engine, "Kokoro (af_heart)");
        assert_eq!(tone.texts, ["One two."]);
        Ok(())
    }

    #[test]
    fn an_engine_not_loaded_is_refused_by_name() {
        let mut runs = NarrationRuns::new(hearing(&[], &[]), |_: &str| Ok(Some(0)));
        let kokoro = speak(&mut runs, &SpeakOpts::default())
            .err()
            .map(|e| e.to_string());
        assert!(kokoro.is_some_and(|m| m.contains("no Kokoro engine is loaded")));
        let opts = SpeakOpts {
            model: Some("facebook/mms-tts-zlm"),
            ..SpeakOpts::default()
        };
        let model = speak(&mut runs, &opts).err().map(|e| e.to_string());
        assert!(model.is_some_and(|m| {
            m.contains("\"facebook/mms-tts-zlm\", and it is not loaded (loaded: none)")
        }));
        let opts = SpeakOpts {
            engine: Some("voxcpm"),
            device: Some("cpu"),
            ..SpeakOpts::default()
        };
        let voxcpm = speak(&mut runs, &opts).err().map(|e| e.to_string());
        assert!(voxcpm.is_some_and(|m| m.contains("no VoxCPM2 engine is loaded")));
    }

    #[test]
    fn a_model_speaks_under_its_id() -> Result<()> {
        let (mut a, mut b) = (Tone::default(), Tone::default());
        let mut runs =
            NarrationRuns::new(hearing(&[("One two.", false)], &[]), |_: &str| Ok(Some(0)))
                .with_model("other/model", &mut a)
                .with_model("facebook/mms-tts-zlm", &mut b);
        let opts = SpeakOpts {
            model: Some("facebook/mms-tts-zlm"),
            ..SpeakOpts::default()
        };
        let spoken = speak(&mut runs, &opts)?;
        drop(runs);
        assert_eq!(spoken.engine, "facebook/mms-tts-zlm");
        assert!(a.texts.is_empty());
        assert_eq!(b.texts, ["One two."]);
        Ok(())
    }

    #[test]
    fn the_command_runs_through_the_spawner_in_a_work_directory() {
        let mut lines = Vec::new();
        let mut runs = NarrationRuns::new(hearing(&[], &[]), |line: &str| {
            lines.push(line.to_string());
            Ok(Some(3))
        });
        let opts = SpeakOpts {
            command: Some("tts {text} {out}"),
            ..SpeakOpts::default()
        };
        let err = speak(&mut runs, &opts).err().map(|e| e.to_string());
        drop(runs);
        assert!(err.is_some_and(|m| m.contains("the TTS command exited 3")));
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("tts 'One two.' '"));
        assert!(lines[0].contains("motion-video-tts-"));
    }

    #[test]
    fn random_values_fall_in_the_unit_interval() {
        for _ in 0..64 {
            let r = random_unit();
            assert!((0.0..1.0).contains(&r), "{r}");
        }
    }
}
