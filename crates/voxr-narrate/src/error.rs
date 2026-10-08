//! voxr-narrate error types.
//!
//! Argument checks on scripts and takes, data that does not fit the
//! pipeline, the built-in patterns failing to compile, a speech recognizer
//! failing to load or read, a speech engine failing to make audio, and a
//! cache or output file failing to read or write.

/// voxr-narrate result type.
pub type Result<T> = std::result::Result<T, Error>;

/// voxr-narrate errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid argument to an operation.
    #[error("invalid argument '{arg}': {reason}")]
    InvalidArgument {
        /// Argument name.
        arg: &'static str,
        /// Why it is invalid.
        reason: String,
    },

    /// A built-in text pattern failed to compile.
    #[error("pattern error: {reason}")]
    Pattern {
        /// Description of what went wrong.
        reason: String,
    },

    /// Decoded or generated data does not fit the pipeline.
    #[error("data error: {reason}")]
    DataError {
        /// Description of what went wrong.
        reason: String,
    },

    /// A speech recognizer failed to load or to read audio.
    #[error("recognizer error: {reason}")]
    Recognizer {
        /// Description of what went wrong.
        reason: String,
    },

    /// A speech engine failed to make audio: a TTS command exited nonzero
    /// or wrote no audio the decoder reads.
    #[error("speech engine error: {reason}")]
    Engine {
        /// What failed, on which text, and the next action.
        reason: String,
    },

    /// A cache or output file failed to read or write.
    #[error("file error: {reason}")]
    File {
        /// The file, what failed, and the next action.
        reason: String,
    },
}

impl Error {
    /// The reason alone, without the kind of error before it.
    pub fn reason(&self) -> &str {
        match self {
            Self::InvalidArgument { reason, .. }
            | Self::Pattern { reason }
            | Self::DataError { reason }
            | Self::Recognizer { reason }
            | Self::Engine { reason }
            | Self::File { reason } => reason,
        }
    }
}

/// An [`Error::File`] naming what failed on `path`, and the next action.
pub(crate) fn file_error(
    what: &str,
    path: &std::path::Path,
    e: &std::io::Error,
    next: &str,
) -> Error {
    Error::File {
        reason: format!("{what} {}: {e}. {next}", path.display()),
    }
}
