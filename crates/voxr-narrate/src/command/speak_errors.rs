//! A speak run's errors as the skill words them: the flag a refusal is
//! about leads the message, as `speak --speed 3: use 0.5 to 2 …`.

use super::outcome::{CommandError, parts};
use crate::error::Error;
use crate::js::js_string;
use crate::take::SpeakOpts;

/// The skill's message for an error `speak()` raises, a failed run (exit
/// 1).
///
/// The flag an argument error is about leads it: `--language`,
/// `--speed`, `--engine` and `--reroll`. Every other error reads
/// `speak: <reason>`, as the skill's own errors do. A reason that already
/// starts with `speak` is kept as it is.
pub fn speak_failure(e: &Error, opts: &SpeakOpts<'_>) -> CommandError {
    let (arg, reason) = parts(e);
    CommandError::Failed(speak_message(arg, reason, opts))
}

fn speak_message(arg: Option<&str>, reason: &str, opts: &SpeakOpts<'_>) -> String {
    if reason.starts_with("speak") {
        return reason.to_string();
    }
    match arg {
        Some("language") => format!(
            "speak --language {}: {reason}",
            opts.language.unwrap_or_default()
        ),
        Some("speed") if reason.starts_with("use 0.5 to 2") => {
            let speed = opts
                .speed
                .map_or_else(|| "undefined".to_string(), js_string);
            format!("speak --speed {speed}: {reason}")
        }
        Some("engine") if reason.starts_with("use kokoro or voxcpm") => format!(
            "speak --engine \"{}\": {reason}",
            opts.engine.unwrap_or_default()
        ),
        Some("engine") if reason.starts_with("voxcpm ") => format!("speak --engine {reason}"),
        Some("rerolls") => format!("speak {reason}"),
        Some("written") => format!("speak --written: {reason}"),
        _ => format!("speak: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(e: Error, opts: &SpeakOpts<'_>) -> String {
        speak_failure(&e, opts).message().to_string()
    }

    fn invalid(arg: &'static str, reason: &str) -> Error {
        Error::InvalidArgument {
            arg,
            reason: reason.to_string(),
        }
    }

    #[test]
    fn the_flag_leads_its_refusal() {
        let opts = SpeakOpts {
            engine: Some("piper"),
            speed: Some(3.0),
            language: Some("ms"),
            ..SpeakOpts::default()
        };
        assert_eq!(
            message(invalid("speed", "use 0.5 to 2 (0.9 is 10% slower)."), &opts),
            "speak --speed 3: use 0.5 to 2 (0.9 is 10% slower)."
        );
        assert_eq!(
            message(
                invalid(
                    "engine",
                    "use kokoro or voxcpm, or --model / --command for other engines."
                ),
                &opts
            ),
            "speak --engine \"piper\": use kokoro or voxcpm, or --model / --command for other engines."
        );
        assert_eq!(
            message(
                invalid(
                    "engine",
                    "voxcpm needs an NVIDIA GPU. Use the default Kokoro engine."
                ),
                &opts
            ),
            "speak --engine voxcpm needs an NVIDIA GPU. Use the default Kokoro engine."
        );
        assert_eq!(
            message(invalid("language", "Kokoro speaks English only."), &opts),
            "speak --language ms: Kokoro speaks English only."
        );
        assert_eq!(
            message(
                invalid("rerolls", "--reroll 4: the script has parts 1 to 2."),
                &opts
            ),
            "speak --reroll 4: the script has parts 1 to 2."
        );
    }

    #[test]
    fn other_errors_read_speak_and_their_reason() {
        let opts = SpeakOpts::default();
        let engine = Error::Engine {
            reason: "the TTS command exited 3 on \"Hi.\": tts".to_string(),
        };
        assert_eq!(
            message(engine, &opts),
            "speak: the TTS command exited 3 on \"Hi.\": tts"
        );
        assert_eq!(
            message(invalid("voice", "Kokoro has no voice \"x\"."), &opts),
            "speak: Kokoro has no voice \"x\"."
        );
    }
}
