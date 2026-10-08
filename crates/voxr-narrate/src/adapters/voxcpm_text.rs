//! What VoxCPM2 speaks for one take: the text with its parenthesized
//! prefix, and the transcript a Hi-Fi clone continues from. It follows the
//! skill's runner, `voxcpm_speak.py`.

use voxr::voxcpm::paren;

use crate::take::{SpeakRequest, VoiceSource};

/// The text VoxCPM2 speaks for one take, and its Hi-Fi transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxcpmText<'r> {
    /// The text with its prefix, such as `(calm)Hello.`.
    pub text: String,
    /// The reference's transcript, for a Hi-Fi clone: the take continues
    /// from the reference as a prompt. `None` for every other take.
    pub hifi: Option<&'r str>,
}

/// The runner's `paren(t)` with no strip: `t` as it is when it is empty or
/// opens with `(`, else wrapped in parentheses.
fn runner_paren(note: &str) -> String {
    if note.is_empty() || note.starts_with('(') {
        note.to_string()
    } else {
        format!("({note})")
    }
}

/// The text and Hi-Fi transcript of `request`, as `voxcpm_speak.py` builds
/// them:
///
/// - A clone with a transcript and no delivery note of its own is Hi-Fi:
///   the text alone, the run's style ignored, continued from the reference.
/// - Any other clone, and a named voice, speaks `note + text`. The note is
///   the part's delivery through the runner's `paren`, unstripped, as
///   `paren(s.get('style', ''))` reads it. An empty delivery falls back to
///   the run's style, stripped first, as `paren(a.style.strip())` reads it.
/// - The model's own voice speaks `paren(design.strip()) + text`. Without a
///   design, the note prefixes the text as for a clone.
///
/// An empty delivery or transcript counts as none, as Python's truth test
/// reads it.
pub fn voxcpm_text<'r, H>(request: &SpeakRequest<'r, H>) -> VoxcpmText<'r> {
    let delivery = request.delivery.filter(|d| !d.is_empty());
    let note = match delivery {
        Some(d) => runner_paren(d),
        None => request.style.map_or_else(String::new, paren),
    };
    match request.voice {
        VoiceSource::Clone {
            transcript: Some(t),
            ..
        } if delivery.is_none() && !t.is_empty() => VoxcpmText {
            text: request.text.to_string(),
            hifi: Some(t),
        },
        VoiceSource::Own {
            design: Some(design),
        } if !paren(design).is_empty() => VoxcpmText {
            text: format!("{}{}", paren(design), request.text),
            hifi: None,
        },
        _ => VoxcpmText {
            text: format!("{note}{}", request.text),
            hifi: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::CloneReference;

    const SAMPLES: [f32; 4] = [0.0; 4];

    fn request<'a>(
        voice: VoiceSource<'a, ()>,
        delivery: Option<&'a str>,
        style: Option<&'a str>,
    ) -> SpeakRequest<'a, ()> {
        SpeakRequest {
            text: "Hello there.",
            delivery,
            style,
            voice,
            seed: 7.0,
            cfg: 1.6,
            steps: 16,
            speed: 1.0,
        }
    }

    fn clone(transcript: Option<&str>) -> VoiceSource<'_, ()> {
        VoiceSource::Clone {
            reference: CloneReference::Samples {
                samples: &SAMPLES,
                rate: 48_000,
            },
            transcript,
        }
    }

    fn text(r: &SpeakRequest<'_, ()>) -> (String, Option<String>) {
        let t = voxcpm_text(r);
        (t.text, t.hifi.map(str::to_string))
    }

    #[test]
    fn a_clone_speaks_its_delivery_else_the_run_style() {
        let style = Some("relaxed");
        let r = request(clone(None), None, style);
        assert_eq!(text(&r), ("(relaxed)Hello there.".to_string(), None));
        let r = request(clone(None), Some("(asking)"), style);
        assert_eq!(text(&r), ("(asking)Hello there.".to_string(), None));
        let r = request(clone(None), None, None);
        assert_eq!(text(&r), ("Hello there.".to_string(), None));
    }

    #[test]
    fn hifi_drops_the_style_unless_the_part_has_a_delivery_note() {
        let said = Some("The reference words.");
        let r = request(clone(said), None, Some("(relaxed)"));
        assert_eq!(
            text(&r),
            (
                "Hello there.".to_string(),
                Some("The reference words.".to_string())
            )
        );
        let r = request(clone(said), Some("(asking)"), Some("(relaxed)"));
        assert_eq!(text(&r), ("(asking)Hello there.".to_string(), None));
        // Only an empty transcript is none, as Python's truth test reads it.
        // A blank one goes on, and VoxCPM2 refuses it by name.
        let r = request(clone(Some("")), None, Some("(relaxed)"));
        assert_eq!(text(&r), ("(relaxed)Hello there.".to_string(), None));
        let r = request(clone(Some("  ")), None, Some("(relaxed)"));
        assert_eq!(text(&r).1.as_deref(), Some("  "));
    }

    #[test]
    fn a_delivery_note_is_never_stripped_and_the_style_always_is() {
        let r = request(clone(None), Some(" asking "), Some("calm"));
        assert_eq!(text(&r), ("( asking )Hello there.".to_string(), None));
        let r = request(clone(None), Some("(asking) "), None);
        assert_eq!(text(&r), ("(asking) Hello there.".to_string(), None));
        let r = request(clone(None), Some(""), Some("  calm \n"));
        assert_eq!(text(&r), ("(calm)Hello there.".to_string(), None));
        let said = Some("The reference words.");
        let r = request(clone(said), Some(""), Some("(relaxed)"));
        assert_eq!(text(&r).1.as_deref(), said, "an empty note is no note");
    }

    #[test]
    fn a_design_prefixes_the_text() {
        let own = |design| VoiceSource::Own { design };
        let r = request(own(Some(" young woman, warm ")), None, None);
        assert_eq!(
            text(&r),
            ("(young woman, warm)Hello there.".to_string(), None)
        );
        let r = request(own(None), None, None);
        assert_eq!(text(&r), ("Hello there.".to_string(), None));
        let r = request(VoiceSource::Named { name: "amir" }, None, Some("(calm)"));
        assert_eq!(text(&r), ("(calm)Hello there.".to_string(), None));
    }
}
