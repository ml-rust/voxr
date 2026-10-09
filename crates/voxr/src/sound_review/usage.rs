//! The review's usage text: the skill script's docstring, printed when no
//! directory is given, with the command line it runs as.

/// The usage text, `run_line` naming how the review runs (for example
/// `blazr sound-review`).
pub fn usage(run_line: &str) -> String {
    format!(
        "Developer review of the sound catalog: harshness limits and recognizability.

Usage:
  node scripts/dev/render-sounds.mjs <dir>
  {run_line} <dir> [--no-model]

Numeric limits (every rendered form):
  sharpness   DIN 45692-style, loudest 100 ms, acum. Fail above 2.6: the sound reads as harsh.
  hiss        energy above 8 kHz against the whole, dB. Fail above -10: audible hiss or fizz.
  tail        last 5 ms against the loudest 50 ms, dB. Fail above -20: the sound is cut off, not finished.
              Builds that stop on their cue (endsOnCue: riser, reverse, drone) are exempt.
  true peak   4x oversampled, dBTP. Fail above -1.

Recognizability (base variant only, --no-model skips it): the LAION CLAP model scores each sound against every
sound's `says` text. Each text's average score is removed first, so a text the model likes for everything gets no
edge. rank is where the sound's own text lands among all texts. Warn above rank 10.

Exit 1 when any sound fails a numeric limit.
"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_run_line_lands_in_usage() {
        let text = usage("blazr sound-review");
        assert!(text.contains("\n  blazr sound-review <dir> [--no-model]\n"));
        assert!(text.ends_with("numeric limit.\n"));
    }
}
