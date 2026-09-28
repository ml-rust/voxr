//! Narration scripts: phrases and chunks a voice speaks, tags, subtitle and
//! timed script files, the written script, numbers read back from words and
//! digits, the check of a spoken take against its script, its glitches, and
//! the polish of its voice. Also a spoken take read into timed words, and
//! the take loop's cache keys, file names and take-record files, and its
//! pick of each part's best take through a speech engine and a judge.

mod error;
mod js;
mod js_math;
mod pattern;

pub mod check;
pub mod glitch;
pub mod polish;
pub mod take;
pub mod text;
pub mod transcript;

pub use check::*;
pub use error::{Error, Result};
pub use glitch::*;
pub use polish::*;
pub use take::*;
pub use text::*;
pub use transcript::*;
