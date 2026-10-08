//! A speak run's own options and the user's `"tts"` config block: what
//! [`resolve_speak`](super::resolve_speak) and
//! [`speak_script`](super::speak_script) read before they fall back to
//! defaults.

use crate::text::TimedCue;

/// A run's own choices, before config fallback: the skill's `opts`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpeakOpts<'a> {
    /// `--engine`.
    pub engine: Option<&'a str>,
    /// `--command`.
    pub command: Option<&'a str>,
    /// `--model`.
    pub model: Option<&'a str>,
    /// `--voice`.
    pub voice: Option<&'a str>,
    /// A recording to clone: VoxCPM2's reference, and the command's
    /// `{reference}`.
    pub reference: Option<&'a str>,
    /// The script's language.
    pub language: Option<&'a str>,
    /// `--speed`.
    pub speed: Option<f64>,
    /// `--style`: VoxCPM2's delivery for every part.
    pub style: Option<&'a str>,
    /// `--hifi`: VoxCPM2 clones with the reference's transcript.
    pub hifi: Option<bool>,
    /// `--cfg`: VoxCPM2's classifier-free guidance.
    pub cfg: Option<f64>,
    /// `--steps`: VoxCPM2's diffusion steps.
    pub steps: Option<u32>,
    /// `--device`. Set, it lets VoxCPM2 run where it is not ready.
    pub device: Option<&'a str>,
    /// `--seed`.
    pub seed: Option<f64>,
    /// `--reroll`: VoxCPM2 parts, from 1, to speak again.
    pub reroll: &'a [usize],
    /// `--one-call`: the command speaks the whole script in one call.
    pub one_call: Option<bool>,
    /// The lines of a timed script. Each line is placed at its time.
    pub cues: Option<&'a [TimedCue]>,
    /// This call is a recursive one already stretched for speed. It only
    /// silences the stripped-tags notice.
    pub stretched: bool,
    /// The script is a timed one, split into lines. It silences the
    /// stripped-tags notice, and VoxCPM2 speaks each line apart.
    pub lines: bool,
}

/// The `"tts"` block of the user's config, read once per run.
#[derive(Debug, Clone, Copy, Default)]
pub struct TtsConfig<'a> {
    /// A configured engine.
    pub engine: Option<&'a str>,
    /// A configured command template.
    pub command: Option<&'a str>,
    /// A configured model.
    pub model: Option<&'a str>,
    /// A configured voice.
    pub voice: Option<&'a str>,
    /// A configured speed.
    pub speed: Option<f64>,
    /// A configured recording to clone.
    pub reference: Option<&'a str>,
    /// A configured VoxCPM2 style.
    pub style: Option<&'a str>,
    /// Hi-Fi cloning by default.
    pub hifi: Option<bool>,
    /// A configured VoxCPM2 guidance.
    pub cfg: Option<f64>,
    /// Configured VoxCPM2 diffusion steps.
    pub steps: Option<u32>,
    /// A configured seed.
    pub seed: Option<f64>,
    /// The command speaks in one call by default.
    pub one_call: Option<bool>,
}
