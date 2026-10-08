//! The engine, model, voice and speed resolution at the top of the
//! skill's `speak()` as a pure function: the config fallbacks, the
//! invalid `--engine` and `--speed` errors, and the tag-strip notice
//! decision, and whether the speed is a time-stretch.
//! [`super::engine_choice::default_engine`] is the piece that picks an
//! engine when none is named. [`EngineChoice`] and [`SpeakResolution`] are
//! what it resolves to.

use super::engine_choice::{
    DEFAULT_VOICE, DefaultEngineOpts, Engine, KOKORO_MODEL, default_engine, truthy,
};
use super::speak_opts::{SpeakOpts, TtsConfig};
pub use super::speak_resolution::{EngineChoice, SpeakResolution};
use crate::error::{Error, Result};
use crate::pattern::Patterns;

/// `opts.engine ?? (opts.command || opts.model ? null : tts.engine)`: the
/// engine a call resolves to before its own branch is picked.
fn effective_engine<'a>(
    engine: Option<&'a str>,
    command: Option<&str>,
    model: Option<&str>,
    tts_engine: Option<&'a str>,
) -> Option<&'a str> {
    engine.or_else(|| {
        if truthy(command) || truthy(model) {
            None
        } else {
            tts_engine
        }
    })
}

/// The engine, speed and tag decision `speak()` resolves before it makes
/// any audio. [`speak_script`](super::speak_script) runs the branch it
/// picks: the engine, the time-stretch, or the timed placement.
pub fn resolve_speak(
    script: &str,
    opts: &SpeakOpts<'_>,
    tts: &TtsConfig<'_>,
    voxcpm_ready: bool,
    voxcpm_why: &str,
) -> Result<SpeakResolution> {
    let p = Patterns::get()?;
    let mut engine: Option<String> = opts.engine.map(str::to_string);
    if !truthy(engine.as_deref())
        && !truthy(opts.command)
        && !truthy(opts.model)
        && !truthy(tts.engine)
        && !truthy(tts.command)
        && !truthy(tts.model)
    {
        let picked = default_engine(
            &DefaultEngineOpts {
                voice: opts.voice,
                reference: opts.reference,
                language: opts.language,
            },
            voxcpm_ready,
            voxcpm_why,
        )?;
        engine = Some(
            match picked {
                Engine::Kokoro => "kokoro",
                Engine::Voxcpm => "voxcpm",
            }
            .to_string(),
        );
    }
    let effective = effective_engine(engine.as_deref(), opts.command, opts.model, tts.engine);

    let strip_tags = effective != Some("voxcpm") && p.tags.is_match(script);
    let log_tag_notice = strip_tags && !opts.stretched && !opts.lines;

    let speed = opts.speed.or(tts.speed).unwrap_or(1.0);
    if !(0.5..=2.0).contains(&speed) {
        return Err(Error::InvalidArgument {
            arg: "speed",
            reason: "use 0.5 to 2 (0.9 is 10% slower).".to_string(),
        });
    }

    let choice = if effective == Some("voxcpm") {
        EngineChoice::Voxcpm {
            voice: opts.voice.or(tts.voice).unwrap_or("").to_string(),
        }
    } else if truthy(effective) && effective != Some("kokoro") {
        return Err(Error::InvalidArgument {
            arg: "engine",
            reason: "use kokoro or voxcpm, or --model / --command for other engines.".to_string(),
        });
    } else {
        let cmd = opts.command.or_else(|| {
            if truthy(opts.model) || truthy(effective) {
                None
            } else {
                tts.command
            }
        });
        let model = opts
            .model
            .unwrap_or_else(|| {
                if effective == Some("kokoro") {
                    KOKORO_MODEL
                } else {
                    tts.model.unwrap_or(KOKORO_MODEL)
                }
            })
            .to_string();
        let voice = opts
            .voice
            .or(tts.voice)
            .unwrap_or(DEFAULT_VOICE)
            .to_string();
        if let Some(cmd) = cmd.filter(|c| !c.is_empty()) {
            EngineChoice::Command {
                command: cmd.to_string(),
                voice: opts.voice.or(tts.voice).unwrap_or("").to_string(),
                reference: opts.reference.or(tts.reference).unwrap_or("").to_string(),
            }
        } else if model.to_lowercase().contains("kokoro") {
            EngineChoice::Kokoro { model, voice }
        } else {
            EngineChoice::Model { model }
        }
    };
    let kokoro_engine =
        effective == Some("kokoro") || matches!(choice, EngineChoice::Kokoro { .. });
    Ok(SpeakResolution {
        engine: choice,
        speed,
        strip_tags,
        log_tag_notice,
        kokoro_engine,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts<'a>(engine: Option<&'a str>, voice: Option<&'a str>) -> SpeakOpts<'a> {
        SpeakOpts {
            engine,
            voice,
            ..SpeakOpts::default()
        }
    }

    #[test]
    fn nothing_set_defaults_the_engine() -> Result<()> {
        let r = resolve_speak(
            "Hello.",
            &SpeakOpts::default(),
            &TtsConfig::default(),
            false,
            "not ready",
        )?;
        assert!(matches!(r.engine, EngineChoice::Kokoro { .. }));
        Ok(())
    }

    #[test]
    fn voxcpm_engine_carries_its_voice() -> Result<()> {
        let r = resolve_speak(
            "Hello.",
            &opts(Some("voxcpm"), Some("design")),
            &TtsConfig::default(),
            false,
            "",
        )?;
        assert_eq!(
            r.engine,
            EngineChoice::Voxcpm {
                voice: "design".to_string()
            }
        );
        Ok(())
    }

    #[test]
    fn an_unknown_engine_errors() {
        let err = resolve_speak(
            "Hello.",
            &opts(Some("piper"), None),
            &TtsConfig::default(),
            false,
            "",
        )
        .err()
        .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "invalid argument 'engine': use kokoro or voxcpm, or --model / --command for other engines."
            )
        );
    }

    #[test]
    fn a_command_wins_over_kokoro() -> Result<()> {
        let o = SpeakOpts {
            command: Some("piper --out {out} --text {text}"),
            ..SpeakOpts::default()
        };
        let r = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")?;
        assert_eq!(
            r.engine,
            EngineChoice::Command {
                command: "piper --out {out} --text {text}".to_string(),
                voice: String::new(),
                reference: String::new(),
            }
        );
        Ok(())
    }

    #[test]
    fn kokoro_named_by_model_id() -> Result<()> {
        let o = SpeakOpts {
            model: Some("onnx-community/Kokoro-82M-v1.0-ONNX"),
            ..SpeakOpts::default()
        };
        let r = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")?;
        assert!(matches!(r.engine, EngineChoice::Kokoro { .. }));
        Ok(())
    }

    #[test]
    fn a_non_kokoro_model_is_a_bare_model_branch() -> Result<()> {
        let o = SpeakOpts {
            model: Some("facebook/mms-tts-eng"),
            ..SpeakOpts::default()
        };
        let r = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")?;
        assert_eq!(
            r.engine,
            EngineChoice::Model {
                model: "facebook/mms-tts-eng".to_string()
            }
        );
        Ok(())
    }

    #[test]
    fn tags_are_stripped_and_noted_off_voxcpm() -> Result<()> {
        let r = resolve_speak(
            "[sigh] Hello.",
            &opts(Some("kokoro"), None),
            &TtsConfig::default(),
            false,
            "",
        )?;
        assert!(r.strip_tags);
        assert!(r.log_tag_notice);
        let voxcpm = resolve_speak(
            "[sigh] Hello.",
            &opts(Some("voxcpm"), None),
            &TtsConfig::default(),
            false,
            "",
        )?;
        assert!(!voxcpm.strip_tags);
        assert!(!voxcpm.log_tag_notice);
        Ok(())
    }

    #[test]
    fn a_stretched_or_lined_call_stays_quiet() -> Result<()> {
        let mut o = opts(Some("kokoro"), None);
        o.stretched = true;
        let r = resolve_speak("[sigh] Hello.", &o, &TtsConfig::default(), false, "")?;
        assert!(r.strip_tags);
        assert!(!r.log_tag_notice);
        Ok(())
    }

    #[test]
    fn speed_out_of_range_errors() {
        let o = SpeakOpts {
            speed: Some(3.0),
            ..SpeakOpts::default()
        };
        let err = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some("invalid argument 'speed': use 0.5 to 2 (0.9 is 10% slower).")
        );
    }

    #[test]
    fn an_empty_string_engine_reads_as_unset() -> Result<()> {
        // JavaScript's `""` is falsy: an explicitly empty engine still lets
        // the default apply, and never trips the "unknown engine" error.
        let r = resolve_speak(
            "Hello.",
            &opts(Some(""), None),
            &TtsConfig::default(),
            false,
            "",
        )?;
        assert!(matches!(r.engine, EngineChoice::Kokoro { .. }));
        Ok(())
    }

    #[test]
    fn an_empty_command_is_no_command_and_engine_kokoro_never_stretches() -> Result<()> {
        let o = SpeakOpts {
            command: Some(""),
            speed: Some(0.8),
            ..SpeakOpts::default()
        };
        let r = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")?;
        assert!(matches!(r.engine, EngineChoice::Kokoro { .. }));
        let o = SpeakOpts {
            engine: Some("kokoro"),
            command: Some("tts {text} {out}"),
            speed: Some(0.8),
            ..SpeakOpts::default()
        };
        let r = resolve_speak("Hello.", &o, &TtsConfig::default(), false, "")?;
        assert!(matches!(r.engine, EngineChoice::Command { .. }));
        assert!(r.kokoro_engine && !r.stretches(false));
        Ok(())
    }
}
