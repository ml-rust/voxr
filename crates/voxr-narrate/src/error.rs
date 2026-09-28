//! voxr-narrate error types.
//!
//! Argument checks on scripts and takes, data that does not fit the
//! pipeline, the built-in patterns failing to compile, and a speech
//! recognizer failing to load or read.

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
}
