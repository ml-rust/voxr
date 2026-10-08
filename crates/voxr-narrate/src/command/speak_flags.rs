//! The speak command's flags as the skill's `parseArgs` leaves them, and
//! the run options they make.

use crate::take::SpeakOpts;
use crate::text::TimedCue;

/// Every flag `speak` takes, before any config fallback. A flag left out
/// is `None` (or `false`), as the skill leaves it undefined.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakFlags {
    /// The script: a text file's path, or the words, as separate
    /// arguments the command joins with spaces.
    pub script: Vec<String>,
    /// `--engine`: `kokoro` or `voxcpm`.
    pub engine: Option<String>,
    /// `--voice`: a Kokoro voice, or a VoxCPM2 voice description.
    pub voice: Option<String>,
    /// `--style`: VoxCPM2's delivery for every part.
    pub style: Option<String>,
    /// `--reference`: a recording to clone.
    pub reference: Option<String>,
    /// `--speed`, at least 0.5.
    pub speed: Option<f64>,
    /// `--pace`: the target pace, at least 2.
    pub pace: Option<f64>,
    /// `--seed`, at least 0.
    pub seed: Option<f64>,
    /// `--reroll`: VoxCPM2 parts, from 1, to speak again.
    pub reroll: Option<Vec<usize>>,
    /// `--cfg`: VoxCPM2's guidance, at least 0.5.
    pub cfg: Option<f64>,
    /// `--steps`: VoxCPM2's diffusion steps, at least 1.
    pub steps: Option<u32>,
    /// `--hifi`: VoxCPM2 clones with the reference's transcript.
    pub hifi: bool,
    /// `--written`: the written script, for captions.
    pub written: Option<String>,
    /// `--review`: print the script review and stop.
    pub review: bool,
    /// `--raw`: skip the voice polish.
    pub raw: bool,
    /// `--device`, such as `cpu`.
    pub device: Option<String>,
    /// `--language`.
    pub language: Option<String>,
    /// `--model`: a Hugging Face TTS model.
    pub model: Option<String>,
    /// `--command`: the user's TTS command template.
    pub command: Option<String>,
    /// `--one-call`: the command speaks the whole script in one call.
    pub one_call: bool,
    /// `--out`: the voiceover's path.
    pub out: Option<String>,
}

impl SpeakFlags {
    /// The run options `speak()` receives: each flag as given, the
    /// reference resolved against the working directory, the timed
    /// script's lines, and `--hifi` and `--one-call` left unset when off so
    /// the config can set them.
    pub fn opts<'a>(
        &'a self,
        reference: Option<&'a str>,
        cues: Option<&'a [TimedCue]>,
    ) -> SpeakOpts<'a> {
        SpeakOpts {
            engine: self.engine.as_deref(),
            command: self.command.as_deref(),
            model: self.model.as_deref(),
            voice: self.voice.as_deref(),
            reference,
            language: self.language.as_deref(),
            speed: self.speed,
            style: self.style.as_deref(),
            hifi: self.hifi.then_some(true),
            cfg: self.cfg,
            steps: self.steps,
            device: self.device.as_deref(),
            seed: self.seed,
            reroll: self.reroll.as_deref().unwrap_or(&[]),
            one_call: self.one_call.then_some(true),
            cues,
            stretched: false,
            lines: false,
        }
    }
}
