//! The speak and transcribe commands around the take loop and the
//! transcriber, as the skill's `speakCmd` and `transcribeCmd` run them:
//! the flags, the script read before any audio, the voice polish and pace
//! after it, the files beside the voiceover, the video manifest's
//! narration, the words files and the video timeline, every printed line,
//! the exit codes, and the JSON objects `--json` prints.
//!
//! A caller parses the flags, reads the config, loads the models and runs
//! the engines. Everything else is here.

mod js_value;
mod manifest_fields;
mod manifest_sound;

pub mod argv;
pub mod flags;
pub mod manifest;
pub mod outcome;
pub mod paths;
pub mod speak_errors;
pub mod speak_finish;
pub mod speak_flags;
pub mod speak_lines;
pub mod speak_prepare;
pub mod speak_rows;
pub mod transcribe_plan;
pub mod transcribe_report;
pub mod transcribe_words;

pub use argv::{FlagKind, SKILL_FLAGS, SkillArgs, SkillFlag, parse_skill_args};
pub use flags::{flag_number, reroll_parts};
pub use manifest::{DEFAULT_MANIFEST, LoadedManifest, load_manifest};
pub use outcome::{
    CommandError, EXIT_BAD_INPUT, EXIT_FAILED, EXIT_OK, EXIT_UNSUPPORTED, Line, Stream,
};
pub use paths::{relative, resolve};
pub use speak_errors::speak_failure;
pub use speak_finish::{FinishInputs, SpeakFiles, SpeakOutcome, finish_speak};
pub use speak_flags::SpeakFlags;
pub use speak_lines::{
    ALL_HEARD, HEARD_BACK, VOXCPM_SPEED_NOTICE, output_lines, pace_line, polish_line, review_lines,
    timed_lines, verdict_lines,
};
pub use speak_prepare::{DEFAULT_OUT, PreparedSpeak, prepare_speak};
pub use speak_rows::heard_back_lines;
pub use transcribe_plan::{
    FILE_LABEL, TranscribeFlags, TranscribePlan, check_narration, plan_transcribe, transcribe_model,
};
pub use transcribe_report::{
    ReadNarration, narration_lines, overrun_line, transcribe_json, write_video_timeline,
};
pub use transcribe_words::{narration_words, write_narration};
