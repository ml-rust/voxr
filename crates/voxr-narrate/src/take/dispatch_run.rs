//! One engine run inside [`speak_script`](super::speak_script): VoxCPM2,
//! or Kokoro, a Hugging Face model or the user's command, with every option
//! the run takes from the flags and the config. Also a track read back as
//! the skill's ffmpeg decode reads it.

use std::path::Path;

use super::dispatch_setup::{ScriptChecks, SpeakRuns, SpeakTarget, SpokenScript};
use super::engine_choice::truthy;
use super::join::JOIN_RATE;
use super::place::PLACE_RATE;
use super::simple::{CFG, STEPS};
use super::simple_setup::SimpleSetup;
use super::speak_opts::{SpeakOpts, TtsConfig};
use super::speak_resolve::{EngineChoice, SpeakResolution};
use super::speak_setup::{VoiceReference, VoxcpmSetup};
use crate::error::{Error, Result};
use crate::wav::pcm16_round_trip;

/// The audio at `out`.
pub(super) enum Audio {
    /// The samples as joined before the WAV rounds them, at `rate`.
    Joined {
        /// The joined samples.
        samples: Vec<f32>,
        /// Their rate.
        rate: u32,
    },
    /// Samples as read back from a 16-bit WAV at [`PLACE_RATE`]: a
    /// stretched track.
    #[cfg(feature = "stretch")]
    Decoded(Vec<f32>),
}

impl Audio {
    /// The track as the skill's `samples(out)` decodes it: mono at
    /// [`PLACE_RATE`], each 16-bit value over 32768. A track at another
    /// rate is resampled as ffmpeg's `-ar 48000` does, which needs the
    /// `stretch` feature.
    pub(super) fn decoded(self) -> Result<Vec<f32>> {
        match self {
            #[cfg(feature = "stretch")]
            Self::Decoded(samples) => Ok(samples),
            Self::Joined { samples, rate } if rate == PLACE_RATE => Ok(pcm16_round_trip(&samples)),
            Self::Joined { samples, rate } => resampled(&pcm16_round_trip(&samples), rate),
        }
    }
}

#[cfg(feature = "stretch")]
fn resampled(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    super::stretch::resample_48k(samples, rate)
}

#[cfg(not(feature = "stretch"))]
fn resampled(_samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    Err(Error::InvalidArgument {
        arg: "cues",
        reason: format!(
            "placing timed lines on a {rate} Hz track resamples it to {PLACE_RATE} Hz, which needs the `stretch` feature. Build voxr-narrate with --features stretch, or use an engine that speaks at {PLACE_RATE} Hz"
        ),
    })
}

/// A finished engine run: its result and the samples it wrote.
pub(super) struct Track {
    /// The result as the skill's `speak()` returns it.
    pub(super) spoken: SpokenScript,
    /// The samples as joined before the WAV rounds them.
    pub(super) samples: Vec<f32>,
    /// Their rate.
    pub(super) rate: u32,
}

impl Track {
    /// The result, and the audio at `out`.
    pub(super) fn split(self) -> (SpokenScript, Audio) {
        let audio = Audio::Joined {
            samples: self.samples,
            rate: self.rate,
        };
        (self.spoken, audio)
    }
}

/// One engine run's own inputs inside a speak run.
pub(super) struct RunInput<'a> {
    /// The run's options as the caller passes them.
    pub(super) opts: &'a SpeakOpts<'a>,
    /// The user's config.
    pub(super) tts: &'a TtsConfig<'a>,
    /// Where it writes, and what the machine runs.
    pub(super) target: &'a SpeakTarget<'a>,
    /// The resolved engine and speed.
    pub(super) resolved: &'a SpeakResolution,
}

/// `script` spoken by the resolved engine at `speed`, each line apart
/// for VoxCPM2 when `lines` is set.
pub(super) fn engine_run<R: SpeakRuns + ?Sized>(
    script: &str,
    input: &RunInput<'_>,
    speed: f64,
    lines: bool,
    runs: &mut R,
) -> Result<Track> {
    let RunInput {
        opts,
        tts,
        target,
        resolved,
    } = *input;
    let engine = &resolved.engine;
    if let EngineChoice::Voxcpm { voice } = engine {
        return voxcpm_run(script, input, voice, lines, runs);
    }
    let setup = SimpleSetup {
        out: target.out,
        engine,
        speed,
        one_call: opts.one_call.or(tts.one_call).unwrap_or(false),
        language: opts.language,
        asr_model: target.asr_model,
    };
    let run = runs.phrases(script, &setup)?;
    Ok(Track {
        spoken: SpokenScript {
            duration: run.duration,
            phrases: run.phrases,
            engine: engine.label(),
            checks: ScriptChecks::Whole(run.checks),
            problems: run.problems,
            timing: None,
            voice_file: None,
        },
        samples: run.samples,
        rate: run.rate,
    })
}

/// The VoxCPM2 run: refused where VoxCPM2 is not ready and no device is
/// asked for.
fn voxcpm_run<R: SpeakRuns + ?Sized>(
    script: &str,
    input: &RunInput<'_>,
    voice: &str,
    lines: bool,
    runs: &mut R,
) -> Result<Track> {
    let RunInput {
        opts, tts, target, ..
    } = *input;
    if !target.voxcpm_ready && !truthy(opts.device) {
        return Err(Error::InvalidArgument {
            arg: "engine",
            reason: format!(
                "voxcpm {}. Use the default Kokoro engine, another TTS with --command, or --device cpu (very slow).",
                target.voxcpm_why
            ),
        });
    }
    let reference = opts
        .reference
        .or(tts.reference)
        .filter(|r| !r.is_empty())
        .map(|r| VoiceReference::of(Path::new(r)))
        .transpose()?;
    let setup = VoxcpmSetup {
        cache_dir: target.cache_dir,
        out: target.out,
        voice,
        style: opts.style.or(tts.style).unwrap_or(""),
        reference: reference.as_ref(),
        seed: opts.seed,
        config_seed: tts.seed,
        cfg: opts.cfg.or(tts.cfg).unwrap_or(CFG),
        steps: opts.steps.or(tts.steps).unwrap_or(STEPS),
        hifi: opts.hifi.or(tts.hifi).unwrap_or(false),
        language: opts.language,
        asr_model: target.asr_model,
        rerolls: opts.reroll,
        lines,
        checkpoint: target.checkpoint,
        device: opts.device,
    };
    let run = runs.voxcpm(script, &setup)?;
    Ok(Track {
        spoken: SpokenScript {
            duration: run.track.duration,
            phrases: Some(run.track.phrases),
            engine: input.resolved.engine.label(),
            checks: ScriptChecks::Parts(run.checks),
            problems: run.problems,
            timing: None,
            voice_file: Some(run.voice_file),
        },
        samples: run.track.samples,
        rate: JOIN_RATE,
    })
}
