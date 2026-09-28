//! The seams the take loop speaks and judges through: a speech engine that
//! makes one take, a store that holds takes already made, and a judge that
//! grades a take against its text.

use crate::error::Result;
use crate::text::Chunk;

/// Mono samples at a sample rate: one take as an engine makes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Take {
    /// The samples, in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub rate: u32,
}

/// The reference a cloned voice comes from.
#[derive(Debug)]
pub enum CloneReference<'a, H> {
    /// A reference the engine already holds in its own form, such as one
    /// it encoded once for every part.
    Handle(&'a H),
    /// A recording the engine reads itself.
    Samples {
        /// Mono samples of the recording.
        samples: &'a [f32],
        /// Their sample rate.
        rate: u32,
    },
}

impl<H> Clone for CloneReference<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H> Copy for CloneReference<'_, H> {}

/// The voice a take speaks in.
#[derive(Debug)]
pub enum VoiceSource<'a, H> {
    /// The model's own voice, or one a description designs.
    Own {
        /// The voice description, such as `(young woman, warm, low)`.
        design: Option<&'a str>,
    },
    /// A voice cloned from a reference.
    Clone {
        /// The reference to clone.
        reference: CloneReference<'a, H>,
        /// The reference's exact words, for Hi-Fi cloning. The engine
        /// clones with them only for a part with no delivery note of its
        /// own, as the runner does.
        transcript: Option<&'a str>,
    },
}

impl<H> Clone for VoiceSource<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<H> Copy for VoiceSource<'_, H> {}

impl<H> VoiceSource<'_, H> {
    /// `own`, `handle` or `samples`: which kind of voice this is.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Own { .. } => "own",
            Self::Clone {
                reference: CloneReference::Handle(_),
                ..
            } => "handle",
            Self::Clone {
                reference: CloneReference::Samples { .. },
                ..
            } => "samples",
        }
    }
}

/// One take to speak.
#[derive(Debug)]
pub struct SpeakRequest<'a, H> {
    /// The text as spoken: the part's text after [`say`](crate::text::say).
    pub text: &'a str,
    /// The part's own delivery note, parentheses included. It replaces
    /// `style` for this part.
    pub delivery: Option<&'a str>,
    /// The run's style, parentheses included. `None` when the run has no
    /// style.
    pub style: Option<&'a str>,
    /// The voice to speak in.
    pub voice: VoiceSource<'a, H>,
    /// The seed: the same text and seed give the same take.
    pub seed: f64,
    /// Classifier-free guidance.
    pub cfg: f64,
    /// Diffusion steps.
    pub steps: u32,
}

impl<H> SpeakRequest<'_, H> {
    /// The prefix the text is spoken with: the part's delivery note, else
    /// the run's style.
    pub fn prefix(&self) -> Option<&str> {
        self.delivery.or(self.style)
    }
}

/// A text-to-speech model that makes one take per call.
pub trait SpeechEngine {
    /// The engine's own form of a cloned voice's reference.
    type Reference;

    /// Speaks one take.
    fn speak(&mut self, request: &SpeakRequest<'_, Self::Reference>) -> Result<Take>;
}

/// One take of one part: what makes it, and what keys it in a store.
#[derive(Debug, Clone, PartialEq)]
pub struct TakeSpec {
    /// The part's place in the script, from 0.
    pub index: usize,
    /// The part's text, before [`say`](crate::text::say).
    pub text: String,
    /// The seed.
    pub seed: f64,
    /// The part's own delivery note, when it has one.
    pub delivery: Option<String>,
}

impl TakeSpec {
    /// The take of part `index` of `part` at `seed`.
    pub fn new(index: usize, part: &Chunk, seed: f64) -> Self {
        Self {
            index,
            text: part.text.clone(),
            seed,
            delivery: part.delivery.clone(),
        }
    }
}

/// The takes made so far, by what makes them. A run speaks only the takes
/// its store does not hold.
pub trait TakeStore {
    /// The store holds this take.
    fn has(&mut self, spec: &TakeSpec) -> bool;

    /// Reads a take the store holds.
    fn load(&mut self, spec: &TakeSpec) -> Result<Take>;

    /// Keeps a new take.
    fn save(&mut self, spec: &TakeSpec, take: &Take) -> Result<()>;
}

/// What the take loop reads from a judged take.
pub trait Graded {
    /// The take says its text.
    fn ok(&self) -> bool;

    /// How far the take is from its text. `None` reads as JavaScript's
    /// `undefined`: it never compares less, equal or greater.
    fn badness(&self) -> Option<f64>;

    /// Why the take fails. Empty when it passes.
    fn why(&self) -> &str;

    /// The words heard, as one line.
    fn heard(&self) -> &str;
}

/// Grades one take against its part.
pub trait TakeJudge {
    /// A judged take.
    type Verdict: Graded;

    /// Judges `take`, made from `spec`.
    fn judge(&mut self, spec: &TakeSpec, take: &Take) -> Result<Self::Verdict>;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::Error;
    use crate::js::js_string;

    /// One recorded engine call: `(text, style, seed, voice kind)`.
    pub(crate) type SpeakCall = (String, Option<String>, f64, &'static str);

    /// Records every take asked for, and returns an empty take.
    #[derive(Default)]
    pub(crate) struct ScriptedEngine {
        pub(crate) calls: Vec<SpeakCall>,
    }

    impl SpeechEngine for ScriptedEngine {
        type Reference = u8;

        fn speak(&mut self, request: &SpeakRequest<'_, u8>) -> Result<Take> {
            self.calls.push((
                request.text.to_string(),
                request.prefix().map(str::to_string),
                request.seed,
                request.voice.kind(),
            ));
            Ok(Take {
                samples: Vec::new(),
                rate: 48000,
            })
        }
    }

    /// A scripted grade.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct Scored {
        pub(crate) ok: bool,
        pub(crate) badness: Option<f64>,
        pub(crate) why: String,
        pub(crate) heard: String,
    }

    impl Graded for Scored {
        fn ok(&self) -> bool {
            self.ok
        }

        fn badness(&self) -> Option<f64> {
            self.badness
        }

        fn why(&self) -> &str {
            &self.why
        }

        fn heard(&self) -> &str {
            &self.heard
        }
    }

    /// Grades by `(text, seed)`; a take with no script passes with 0
    /// badness. Records every call as `(text, seed)`.
    #[derive(Default)]
    pub(crate) struct ScriptedJudge {
        pub(crate) scores: Vec<(String, f64, bool, Option<f64>)>,
        pub(crate) calls: Vec<(String, f64)>,
    }

    impl ScriptedJudge {
        pub(crate) fn score(&mut self, text: &str, seed: f64, ok: bool, badness: Option<f64>) {
            self.scores.push((text.to_string(), seed, ok, badness));
        }
    }

    impl TakeJudge for ScriptedJudge {
        type Verdict = Scored;

        fn judge(&mut self, spec: &TakeSpec, _take: &Take) -> Result<Scored> {
            self.calls.push((spec.text.clone(), spec.seed));
            let (ok, badness) = self
                .scores
                .iter()
                .find(|(t, s, _, _)| *t == spec.text && *s == spec.seed)
                .map_or((true, Some(0.0)), |&(_, _, ok, b)| (ok, b));
            Ok(Scored {
                ok,
                badness,
                why: format!("bad {}", badness.map_or("none".to_string(), js_string)),
                heard: format!("heard {}", spec.text),
            })
        }
    }

    /// Holds takes by `(text, seed, delivery)`.
    #[derive(Default)]
    pub(crate) struct MemoryStore {
        pub(crate) held: Vec<(String, f64, Option<String>)>,
    }

    impl MemoryStore {
        fn key(spec: &TakeSpec) -> (String, f64, Option<String>) {
            (spec.text.clone(), spec.seed, spec.delivery.clone())
        }

        pub(crate) fn hold(&mut self, text: &str, seed: f64, delivery: Option<&str>) {
            self.held
                .push((text.to_string(), seed, delivery.map(str::to_string)));
        }
    }

    impl TakeStore for MemoryStore {
        fn has(&mut self, spec: &TakeSpec) -> bool {
            self.held.contains(&Self::key(spec))
        }

        fn load(&mut self, spec: &TakeSpec) -> Result<Take> {
            if self.has(spec) {
                Ok(Take {
                    samples: Vec::new(),
                    rate: 48000,
                })
            } else {
                Err(Error::DataError {
                    reason: format!("no take of {:?} at seed {}", spec.text, spec.seed),
                })
            }
        }

        fn save(&mut self, spec: &TakeSpec, _take: &Take) -> Result<()> {
            self.held.push(Self::key(spec));
            Ok(())
        }
    }

    /// Returns each value of `values` in turn.
    pub(crate) fn sequence(values: Vec<f64>) -> impl FnMut() -> f64 {
        let mut at = 0;
        move || {
            let v = values.get(at).copied().unwrap_or(0.0);
            at += 1;
            v
        }
    }

    pub(crate) const HANDLE: u8 = 1;

    fn request<'a>(
        delivery: Option<&'a str>,
        style: Option<&'a str>,
        voice: VoiceSource<'a, u8>,
    ) -> SpeakRequest<'a, u8> {
        SpeakRequest {
            text: "Hello.",
            delivery,
            style,
            voice,
            seed: 7.0,
            cfg: 1.6,
            steps: 16,
        }
    }

    #[test]
    fn delivery_replaces_the_run_style() {
        let own = VoiceSource::Own { design: None };
        assert_eq!(request(None, None, own).prefix(), None);
        assert_eq!(request(None, Some("(calm)"), own).prefix(), Some("(calm)"));
        assert_eq!(
            request(Some("(asking)"), Some("(calm)"), own).prefix(),
            Some("(asking)")
        );
    }

    #[test]
    fn voice_kinds() {
        let handle = 3u8;
        let samples = [0.0f32; 4];
        assert_eq!(VoiceSource::<u8>::Own { design: None }.kind(), "own");
        let by_handle = VoiceSource::Clone {
            reference: CloneReference::Handle(&handle),
            transcript: None,
        };
        assert_eq!(by_handle.kind(), "handle");
        let by_samples = VoiceSource::<u8>::Clone {
            reference: CloneReference::Samples {
                samples: &samples,
                rate: 48000,
            },
            transcript: Some("Hello."),
        };
        assert_eq!(by_samples.kind(), "samples");
    }
}
