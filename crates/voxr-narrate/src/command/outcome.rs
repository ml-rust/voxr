//! How a command ends: the lines it prints on each stream, in order, and
//! the error that stops it with the skill's exit code.

use std::fmt;

use crate::error::Error;

/// The stream a line goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// Standard output: what the skill prints with `console.log`.
    Out,
    /// Standard error: what the skill prints with `console.error`.
    Err,
}

/// One printed line, without its line break.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// The stream it goes to.
    pub stream: Stream,
    /// The text.
    pub text: String,
}

impl Line {
    /// A line on standard output.
    pub fn out(text: impl Into<String>) -> Self {
        Self {
            stream: Stream::Out,
            text: text.into(),
        }
    }

    /// A line on standard error.
    pub fn err(text: impl Into<String>) -> Self {
        Self {
            stream: Stream::Err,
            text: text.into(),
        }
    }
}

/// The exit code of a command that finished: 0 when the speech check found
/// nothing, 1 when a problem remains.
pub const EXIT_OK: i32 = 0;
/// The exit code of a run that failed, or of a speech check that found a
/// problem.
pub const EXIT_FAILED: i32 = 1;
/// The exit code of a bad input.
pub const EXIT_BAD_INPUT: i32 = 2;
/// The exit code of a model this runtime does not run, such as a Hugging
/// Face TTS model other than VITS: a caller can run it another way.
pub const EXIT_UNSUPPORTED: i32 = 3;

/// Why a command stopped before it finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandError {
    /// A bad input: a missing file, a flag out of range, a manifest that
    /// does not hold. The skill prints `error: <message>` and exits 2.
    Input(String),
    /// A run that failed on the way: an engine, a recognizer, a file. The
    /// skill prints the error's stack, `Error: <message>` first, and exits
    /// 1.
    Failed(String),
    /// A model this runtime does not run: `error: <message>`, exit 3, so a
    /// caller can hand the model to a runner that does.
    Unsupported(String),
}

impl CommandError {
    /// The exit code: [`EXIT_BAD_INPUT`], [`EXIT_FAILED`] or
    /// [`EXIT_UNSUPPORTED`].
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Input(_) => EXIT_BAD_INPUT,
            Self::Failed(_) => EXIT_FAILED,
            Self::Unsupported(_) => EXIT_UNSUPPORTED,
        }
    }

    /// The message alone.
    pub fn message(&self) -> &str {
        match self {
            Self::Input(m) | Self::Failed(m) | Self::Unsupported(m) => m,
        }
    }

    /// The line printed on standard error.
    pub fn line(&self) -> String {
        match self {
            Self::Input(m) | Self::Unsupported(m) => format!("error: {m}"),
            Self::Failed(m) => format!("Error: {m}"),
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.line())
    }
}

impl std::error::Error for CommandError {}

/// The argument an error names, and its reason without the variant's
/// prefix.
pub(crate) fn parts(e: &Error) -> (Option<&'static str>, &str) {
    match e {
        Error::InvalidArgument { arg, reason } => (Some(arg), reason),
        other => (None, other.reason()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_print_as_the_skill_prints_them() {
        let input = CommandError::Input("speak needs the script.".to_string());
        assert_eq!(input.line(), "error: speak needs the script.");
        assert_eq!(input.exit_code(), 2);
        let failed = CommandError::Failed("speak: the script is empty.".to_string());
        assert_eq!(failed.line(), "Error: speak: the script is empty.");
        assert_eq!(failed.exit_code(), 1);
    }
}
