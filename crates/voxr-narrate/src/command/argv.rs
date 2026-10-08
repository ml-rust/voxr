//! The skill's `parseArgs`: every flag the skill's CLI takes, read the
//! skill's way for any command. `speak` and `transcribe` take each flag of
//! every other command, and leave the ones they do not read.
//!
//! - A value flag takes the next argument whatever it is, `--raw`
//!   included. As the last argument it takes no value: the flag is then
//!   left unset, and the default holds.
//! - A number flag is checked against its lower bound when it is read. As
//!   the last argument it reads `undefined`, which fails the check.
//! - `--reroll` as the last argument stops the run, a failed run.
//! - Any other argument that starts with `--` is an unknown option, a bad
//!   input. Every argument that does not is a word of the command.

use std::collections::HashMap;

use super::flags::{flag_number, reroll_parts};
use super::outcome::CommandError;
use super::speak_flags::SpeakFlags;
use super::transcribe_plan::TranscribeFlags;

/// How a flag reads its arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlagKind {
    /// No value: the flag is on.
    Switch,
    /// The next argument as text.
    Text,
    /// The next argument as a number of at least `min`.
    Number {
        /// The lower bound.
        min: f64,
    },
    /// The next argument, added to a list: `--pin`, `--route`, `--layer`.
    List,
    /// `--reroll`: the next argument, split at its commas.
    Parts,
}

/// One flag of the skill's CLI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkillFlag {
    /// The flag, dashes included.
    pub name: &'static str,
    /// How it reads its arguments.
    pub kind: FlagKind,
    /// The skill takes it. `false` for a flag only blazr takes.
    pub skill: bool,
}

const fn flag(name: &'static str, kind: FlagKind) -> SkillFlag {
    SkillFlag {
        name,
        kind,
        skill: true,
    }
}

const fn number(name: &'static str, min: f64) -> SkillFlag {
    flag(name, FlagKind::Number { min })
}

/// Every flag, in the order of the skill's `parseArgs`, then `--json`.
pub const SKILL_FLAGS: &[SkillFlag] = &[
    flag("--manifest", FlagKind::Text),
    flag("--scene", FlagKind::Text),
    flag("--draft", FlagKind::Switch),
    flag("--force", FlagKind::Switch),
    number("--scale", 0.25),
    number("--jobs", 1.0),
    number("--start", 0.0),
    flag("--name", FlagKind::Text),
    flag("--subset", FlagKind::Text),
    flag("--model", FlagKind::Text),
    flag("--language", FlagKind::Text),
    flag("--script", FlagKind::Text),
    flag("--voice", FlagKind::Text),
    flag("--style", FlagKind::Text),
    number("--cfg", 0.5),
    number("--steps", 1.0),
    number("--speed", 0.5),
    number("--pace", 2.0),
    number("--seed", 0.0),
    flag("--reroll", FlagKind::Parts),
    flag("--out", FlagKind::Text),
    flag("--command", FlagKind::Text),
    flag("--one-call", FlagKind::Switch),
    flag("--review", FlagKind::Switch),
    flag("--raw", FlagKind::Switch),
    flag("--hifi", FlagKind::Switch),
    flag("--written", FlagKind::Text),
    flag("--engine", FlagKind::Text),
    flag("--reference", FlagKind::Text),
    flag("--device", FlagKind::Text),
    flag("--fit", FlagKind::Text),
    flag("--bbox", FlagKind::Text),
    flag("--highlight", FlagKind::Text),
    flag("--pin", FlagKind::List),
    flag("--route", FlagKind::List),
    flag("--layer", FlagKind::List),
    flag("--land", FlagKind::Text),
    flag("--detail", FlagKind::Text),
    flag("--size", FlagKind::Text),
    flag("--countries", FlagKind::Switch),
    number("--from", 0.0),
    number("--to", 0.0),
    number("--fps", 1.0),
    number("--width", 16.0),
    flag("-h", FlagKind::Switch),
    flag("--help", FlagKind::Switch),
    flag("--check", FlagKind::Switch),
    SkillFlag {
        name: "--json",
        kind: FlagKind::Switch,
        skill: false,
    },
];

/// The arguments as the skill's `parseArgs` leaves them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkillArgs {
    /// Every argument that is no flag nor a flag's value, in order.
    pub rest: Vec<String>,
    /// Each text flag read: its last value, `None` when it came last with
    /// no value.
    text: HashMap<&'static str, Option<String>>,
    /// Each number flag read: its last value.
    numbers: HashMap<&'static str, f64>,
    /// Each switch given.
    switches: Vec<&'static str>,
    /// `--reroll`'s last value.
    reroll: Option<String>,
}

/// Reads `argv`, the arguments after the command, as the skill's
/// `parseArgs` reads them. `unknown` is the text after an unknown option's
/// line: the command's help.
pub fn parse_skill_args(argv: &[String], unknown: &str) -> Result<SkillArgs, CommandError> {
    let mut out = SkillArgs::default();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let Some(f) = SKILL_FLAGS.iter().find(|f| f.name == a) else {
            if a.starts_with("--") {
                return Err(CommandError::Input(format!(
                    "Unknown option {a}.\n\n{unknown}"
                )));
            }
            out.rest.push(a.to_string());
            i += 1;
            continue;
        };
        if f.kind == FlagKind::Switch {
            out.switches.push(f.name);
            i += 1;
            continue;
        }
        let value = argv.get(i + 1).cloned();
        i += 2;
        match f.kind {
            FlagKind::Text => {
                out.text.insert(f.name, value);
            }
            FlagKind::Number { min } => {
                // `Number(undefined)` is NaN, and prints as "undefined",
                // as `Number("undefined")` reads and prints.
                let n = flag_number(f.name, value.as_deref().unwrap_or("undefined"), min)?;
                out.numbers.insert(f.name, n);
            }
            FlagKind::List => {}
            FlagKind::Parts => match value {
                Some(v) => out.reroll = Some(v),
                // The skill's `argv[++i].split(',')` throws here.
                None => {
                    return Err(CommandError::Failed(
                        "--reroll needs the parts to speak again, such as 3,7. Pass them after --reroll".to_string(),
                    ));
                }
            },
            FlagKind::Switch => {}
        }
    }
    Ok(out)
}

impl SkillArgs {
    /// `-h` or `--help` given.
    pub fn help(&self) -> bool {
        self.on("-h") || self.on("--help")
    }

    /// The switch `name` given.
    pub fn on(&self, name: &str) -> bool {
        self.switches.contains(&name)
    }

    /// The text flag `name`'s value. `None` when it was not given, or came
    /// last with no value.
    pub fn text(&self, name: &str) -> Option<String> {
        self.text.get(name).cloned().flatten()
    }

    /// The number flag `name`'s value.
    pub fn number(&self, name: &str) -> Option<f64> {
        self.numbers.get(name).copied()
    }

    /// The flags `speak` reads. `--steps` rounds to a whole number, and
    /// `--reroll` must name whole parts from 1.
    pub fn speak_flags(&self) -> Result<SpeakFlags, CommandError> {
        Ok(SpeakFlags {
            script: self.rest.clone(),
            engine: self.text("--engine"),
            voice: self.text("--voice"),
            style: self.text("--style"),
            reference: self.text("--reference"),
            speed: self.number("--speed"),
            pace: self.number("--pace"),
            seed: self.number("--seed"),
            reroll: self.reroll.as_deref().map(reroll_parts).transpose()?,
            cfg: self.number("--cfg"),
            steps: self.number("--steps").map(|n| n.round() as u32),
            hifi: self.on("--hifi"),
            written: self.text("--written"),
            review: self.on("--review"),
            raw: self.on("--raw"),
            device: self.text("--device"),
            language: self.text("--language"),
            model: self.text("--model"),
            command: self.text("--command"),
            one_call: self.on("--one-call"),
            out: self.text("--out"),
        })
    }

    /// The flags `transcribe` reads.
    pub fn transcribe_flags(&self) -> TranscribeFlags {
        TranscribeFlags {
            files: self.rest.clone(),
            model: self.text("--model"),
            language: self.text("--language"),
            script: self.text("--script"),
            manifest: self.text("--manifest"),
            manifest_missing: self.text.get("--manifest").is_some_and(Option::is_none),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<SkillArgs, CommandError> {
        let argv: Vec<String> = argv.iter().map(|a| a.to_string()).collect();
        parse_skill_args(&argv, "HELP")
    }

    #[test]
    fn every_flag_of_any_command_is_taken_and_left() -> Result<(), CommandError> {
        let a = parse(&[
            "One",
            "--manifest",
            "v.json",
            "--scene",
            "2",
            "--draft",
            "--scale",
            "2",
            "--pin",
            "A@1,2",
            "--fit",
            "MY",
            "two.",
            "--countries",
            "--width",
            "640",
        ])?;
        assert_eq!(a.rest, ["One", "two."]);
        let f = a.speak_flags()?;
        assert_eq!(f.script, ["One", "two."]);
        assert_eq!(
            f,
            SpeakFlags {
                script: f.script.clone(),
                ..SpeakFlags::default()
            }
        );
        assert_eq!(a.transcribe_flags().manifest.as_deref(), Some("v.json"));
        Ok(())
    }

    #[test]
    fn a_value_flag_takes_the_next_argument_whatever_it_is() -> Result<(), CommandError> {
        let f = parse(&["Hi.", "--voice", "--raw"])?.speak_flags()?;
        assert_eq!(f.voice.as_deref(), Some("--raw"));
        assert!(!f.raw);
        Ok(())
    }

    #[test]
    fn a_value_flag_last_is_left_unset() -> Result<(), CommandError> {
        let f = parse(&["Hi.", "--voice", "af_bella", "--voice"])?.speak_flags()?;
        assert_eq!(f.voice, None);
        let t = parse(&["a.wav", "--manifest"])?.transcribe_flags();
        assert_eq!(t.manifest, None);
        assert!(t.manifest_missing);
        assert!(!parse(&["a.wav"])?.transcribe_flags().manifest_missing);
        Ok(())
    }

    #[test]
    fn a_number_flag_last_reads_undefined() {
        assert_eq!(
            parse(&["Hi.", "--pace"]),
            Err(CommandError::Input(
                "--pace needs a number of at least 2, got \"undefined\".".to_string()
            ))
        );
        // A number another command reads is checked all the same.
        assert_eq!(
            parse(&["Hi.", "--scale", "0.1"]),
            Err(CommandError::Input(
                "--scale needs a number of at least 0.25, got \"0.1\".".to_string()
            ))
        );
    }

    #[test]
    fn reroll_last_fails_the_run_and_unknown_options_are_bad_input() {
        assert!(matches!(
            parse(&["Hi.", "--reroll"]),
            Err(CommandError::Failed(_))
        ));
        assert_eq!(
            parse(&["Hi.", "--nope"]),
            Err(CommandError::Input(
                "Unknown option --nope.\n\nHELP".to_string()
            ))
        );
        assert_eq!(
            parse(&["--", "Hi."]),
            Err(CommandError::Input(
                "Unknown option --.\n\nHELP".to_string()
            ))
        );
        let words = parse(&["-x", "Hi."]);
        assert_eq!(
            words.map(|a| a.rest),
            Ok(vec!["-x".to_string(), "Hi.".to_string()])
        );
    }

    #[test]
    fn help_and_json_are_switches() -> Result<(), CommandError> {
        let a = parse(&["-h", "--json"])?;
        assert!(a.help() && a.on("--json"));
        Ok(())
    }
}
