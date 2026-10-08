//! A voice designed from a description: the script's opening spoken in the
//! described voice, spoken again from itself calm and slow until its pace
//! reaches [`REF_PACE`], then prepared as the reference every take clones.
//!
//! A clone takes its pace and manner from its reference far more than from
//! a pace word, and a design speaks in whatever manner its description
//! implies. So the design speaks its opening again from itself, calm and
//! natural at a slow pace, and that becomes the reference. If it is still
//! faster than [`REF_PACE`], it speaks once more, very slowly.

use super::engine::{CloneReference, SpeakRequest, SpeechEngine, Take, TakeSpec, VoiceSource};
use super::pace::{PaceSpan, REF_PACE, pace};
use super::progress::Sink;
use crate::error::{Error, Result};
use crate::glitch::{repair_clicks, tidy};
use crate::pattern::is_space;
use crate::polish::prepare_reference;
use crate::text::{phrases, say};
use crate::wav::pcm16_round_trip;

pub use super::anchor_step::AnchorStep;

/// The words a designed voice's opening reaches, at the least: a clip long
/// enough for every part to hold the voice.
pub const OPENING_WORDS: usize = 25;

/// The sample rate a designed voice is built and kept at.
pub const ANCHOR_RATE: u32 = 48000;

/// The delivery each slower take asks for, in order. Without "natural" the
/// voice sounds angry.
pub const SLOWER: [&str; 2] = [
    "(calm, natural tone, speaking slowly)",
    "(calm, natural tone, speaking very slowly)",
];

/// Pieces `text.split(/\s+/)` gives: one more than its runs of whitespace.
pub(crate) fn js_word_count(text: &str) -> usize {
    let mut count = 1;
    let mut in_space = false;
    for c in text.chars() {
        let space = is_space(c);
        if space && !in_space {
            count += 1;
        }
        in_space = space;
    }
    count
}

/// The script's opening a voice is designed on: its phrases joined by
/// spaces, up to the first that brings the text to [`OPENING_WORDS`] words.
/// A shorter script gives all of it.
pub fn opening(script: &str) -> Result<String> {
    let mut words: Vec<String> = Vec::new();
    for p in phrases(script)? {
        words.push(p.text);
        if js_word_count(&words.join(" ")) >= OPENING_WORDS {
            break;
        }
    }
    Ok(words.join(" "))
}

/// What a voice is designed with.
#[derive(Debug, Clone, Copy)]
pub struct DesignSetup<'a> {
    /// The voice description, such as `(young woman, warm, low)`. `None`
    /// lets the model pick its own voice.
    pub voice: Option<&'a str>,
    /// The run's style, parentheses included. Each slower take's own
    /// delivery replaces it.
    pub style: Option<&'a str>,
    /// The run's seed: the design speaks at it, the slower takes at 1 and 2
    /// past it.
    pub seed: f64,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
    /// A Hi-Fi transcript the slower takes clone with: one saved beside an
    /// older design of this voice.
    pub transcript: Option<&'a str>,
}

/// A designed voice.
#[derive(Debug, Clone, PartialEq)]
pub struct DesignedVoice {
    /// The opening, as the script writes it.
    pub opening: String,
    /// The words the reference says: the opening after
    /// [`say`]. Hi-Fi cloning clones with it.
    pub transcript: String,
    /// The design as first spoken, repaired: the `.design.wav` copy.
    pub design: Vec<f32>,
    /// The prepared reference, at [`ANCHOR_RATE`].
    pub anchor: Vec<f32>,
    /// The pace of each reference measured, in syllables per second.
    pub paces: Vec<f64>,
    /// Breath runs turned down in the reference.
    pub breaths: usize,
    /// Each slower take, as the engine spoke it, with what keys it.
    pub retries: Vec<(TakeSpec, Take)>,
}

/// A take's samples at [`ANCHOR_RATE`], or an error naming the rate.
fn at_anchor_rate(take: &Take) -> Result<&[f32]> {
    if take.rate == ANCHOR_RATE {
        Ok(&take.samples)
    } else {
        Err(Error::DataError {
            reason: format!(
                "the engine spoke the designed voice at {} Hz, and a designed voice is built at {ANCHOR_RATE} Hz. Resample the engine's takes to {ANCHOR_RATE} Hz",
                take.rate
            ),
        })
    }
}

/// A reference take repaired like any take: a glitch in it carries into
/// every part cloned from it.
fn repaired(x: &[f32]) -> Vec<f32> {
    repair_clicks(&tidy(x, ANCHOR_RATE).samples, ANCHOR_RATE).samples
}

/// Syllables per second the reference `x` speaks `transcript` at, pauses
/// included between its first and last sound.
fn reference_pace(transcript: &str, x: &[f32]) -> f64 {
    let span = PaceSpan {
        text: transcript.to_string(),
        start: 0.0,
        end: x.len() as f64 / f64::from(ANCHOR_RATE),
    };
    pace(&[span], Some((x, ANCHOR_RATE)))
}

/// Designs a voice from `setup.voice` on `script`'s [`opening`].
///
/// The design speaks the opening at `setup.seed` and is repaired. The
/// reference then speaks the opening again, cloned from itself, at each
/// [`SLOWER`] delivery in turn and the seed 1, then 2, past the design's.
/// The first always follows; the second only when the first is still
/// faster than [`REF_PACE`]. Each repaired reference is read as its 16-bit
/// file reads back. The last is prepared: breaths turned down, a thin
/// voice's bass lifted, compressed and normalized.
///
/// Every step goes to `log` as it happens.
pub fn design_voice<E: SpeechEngine>(
    script: &str,
    setup: &DesignSetup<'_>,
    engine: &mut E,
    log: &mut dyn Sink<AnchorStep>,
) -> Result<DesignedVoice> {
    let opening = opening(script)?;
    if opening.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "script",
            reason: "the script has no words to design a voice on. Pass a script with at least one phrase".to_string(),
        });
    }
    let transcript = say(&opening)?;
    // The runner speaks each text after `say` once more.
    let spoken = say(&transcript)?;
    log.step(AnchorStep::Designing {
        opening: opening.clone(),
    });
    let design_spec = TakeSpec {
        index: 0,
        text: transcript.clone(),
        seed: setup.seed,
        delivery: None,
    };
    log.step(AnchorStep::Speak(design_spec));
    let first = engine.speak(&SpeakRequest {
        text: &spoken,
        delivery: None,
        style: None,
        voice: VoiceSource::Own {
            design: setup.voice.filter(|v| !v.is_empty()),
        },
        seed: setup.seed,
        cfg: setup.cfg,
        steps: setup.steps,
        speed: 1.0,
    })?;
    let design = repaired(at_anchor_rate(&first)?);
    let mut reference = pcm16_round_trip(&design);
    let mut paces = Vec::new();
    let mut retries = Vec::new();
    for (n, &ask) in SLOWER.iter().enumerate() {
        let was = reference_pace(&transcript, &reference);
        paces.push(was);
        if n > 0 && was <= REF_PACE {
            break;
        }
        log.step(AnchorStep::Slower { ask, pace: was });
        let spec = TakeSpec {
            index: 0,
            text: transcript.clone(),
            seed: setup.seed + (n + 1) as f64,
            delivery: Some(ask.to_string()),
        };
        log.step(AnchorStep::Speak(spec.clone()));
        let take = engine.speak(&SpeakRequest {
            text: &spoken,
            delivery: Some(ask),
            style: setup.style.filter(|s| !s.is_empty()),
            voice: VoiceSource::Clone {
                reference: CloneReference::Samples {
                    samples: &reference,
                    rate: ANCHOR_RATE,
                },
                transcript: setup.transcript,
            },
            seed: spec.seed,
            cfg: setup.cfg,
            steps: setup.steps,
            speed: 1.0,
        })?;
        reference = pcm16_round_trip(&repaired(at_anchor_rate(&take)?));
        retries.push((spec, take));
    }
    let (anchor, report) = prepare_reference(&reference, ANCHOR_RATE);
    let breaths = report.runs.len();
    if breaths > 0 {
        log.step(AnchorStep::Breaths { runs: breaths });
    }
    Ok(DesignedVoice {
        opening,
        transcript,
        design,
        anchor,
        paces,
        breaths,
        retries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::take::engine::tests::ScriptedEngine;

    fn setup() -> DesignSetup<'static> {
        DesignSetup {
            voice: Some("(warm, low)"),
            style: Some("(relaxed)"),
            seed: 7.0,
            cfg: 1.6,
            steps: 16,
            transcript: None,
        }
    }

    #[test]
    fn words_count_as_javascript_splits_them() {
        assert_eq!(js_word_count(""), 1);
        assert_eq!(js_word_count("one two"), 2);
        assert_eq!(js_word_count("one \u{a0} two\u{feff}"), 3);
        assert_eq!(js_word_count(" one"), 2);
    }

    #[test]
    fn the_opening_stops_at_the_phrase_reaching_25_words() -> Result<()> {
        let script = "One two three four five six seven eight nine ten. \
            Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty. \
            One two three four five six. Seven eight.";
        assert_eq!(
            opening(script)?,
            "One two three four five six seven eight nine ten. \
             Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty. \
             One two three four five six."
        );
        assert_eq!(opening("Hello there.")?, "Hello there.");
        assert_eq!(opening("")?, "");
        Ok(())
    }

    #[test]
    fn a_silent_design_is_spoken_slower_once() -> Result<()> {
        let mut engine = ScriptedEngine::default();
        let mut log = Vec::new();
        let voice = design_voice("Hello there, friend.", &setup(), &mut engine, &mut log)?;
        assert_eq!(voice.transcript, "Hello there, friend.");
        // An empty take never sounds, so its pace reads 0: under REF_PACE.
        assert_eq!(voice.paces, [0.0, 0.0]);
        assert_eq!(voice.retries.len(), 1);
        assert_eq!(
            engine.calls,
            [
                ("Hello there, friend.".to_string(), None, 7.0, "own"),
                (
                    "Hello there, friend.".to_string(),
                    Some(SLOWER[0].to_string()),
                    8.0,
                    "samples"
                ),
            ]
        );
        let messages: Vec<String> = log.iter().filter_map(AnchorStep::message).collect();
        assert_eq!(
            messages,
            [
                "speak: designing the voice from --voice, speaking the script's opening: \"Hello there, friend.\"",
                "speak: speaking the designed voice's reference again, calm, natural tone, speaking slowly (it spoke at 0.0 syllables per second): clones follow a reference's pace and manner.",
            ]
        );
        Ok(())
    }

    #[test]
    fn an_empty_script_is_refused() {
        let got = design_voice(
            "",
            &setup(),
            &mut ScriptedEngine::default(),
            &mut Vec::new(),
        );
        assert!(matches!(
            got,
            Err(Error::InvalidArgument { arg: "script", .. })
        ));
    }

    /// Speaks every take at 24 kHz.
    struct HalfRate;

    impl SpeechEngine for HalfRate {
        type Reference = ();

        fn speak(&mut self, _request: &SpeakRequest<'_, ()>) -> Result<Take> {
            Ok(Take {
                samples: vec![0.0; 10],
                rate: 24000,
            })
        }
    }

    #[test]
    fn a_take_at_another_rate_is_refused() {
        let got = design_voice("Hello there.", &setup(), &mut HalfRate, &mut Vec::new());
        assert!(matches!(got, Err(Error::DataError { ref reason }) if reason.contains("24000 Hz")));
    }
}
