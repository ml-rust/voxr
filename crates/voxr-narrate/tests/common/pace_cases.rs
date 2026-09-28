//! Cases for the speaking-pace oracle: texts `syllables()` reads, signals
//! `sounding()` and `pace()` read with pauses of various lengths, and the
//! verdict/suggestion inputs a pace check computes against.
//!
//! Lists the same cases `tests/tools/narration_oracle/pace_cases.mjs`
//! lists, in the same order.

use super::signals::{Op, noise, tone};

/// A text `syllables()` reads, and its name.
#[derive(Debug, Clone, Copy)]
pub struct TextCase {
    pub name: &'static str,
    pub text: &'static str,
}

/// Every text case, in the oracle's order: digits, hyphens, unicode
/// accents, Malay words.
pub fn text_cases() -> Vec<TextCase> {
    vec![
        TextCase {
            name: "plain word",
            text: "Berlumpur",
        },
        TextCase {
            name: "accented letter",
            text: "café",
        },
        TextCase {
            name: "digits and a decimal point",
            text: "It rose 13.5 percent.",
        },
        TextCase {
            name: "a dotted name",
            text: "Built with Node.js today!",
        },
        TextCase {
            name: "hyphenated word",
            text: "well-known",
        },
        TextCase {
            name: "reduplicated malay word",
            text: "kanak-kanak",
        },
        TextCase {
            name: "malay sentence",
            text: "Kuala Lumpur ialah ibu negara Malaysia.",
        },
        TextCase {
            name: "empty text",
            text: "",
        },
        TextCase {
            name: "digits only",
            text: "123 456",
        },
        TextCase {
            name: "cjk text",
            text: "你好，今天好嗎？",
        },
    ]
}

/// A quiet noise floor under a 48 kHz signal.
fn floor(n: usize) -> Op {
    noise(0, n, 0.00002, 3)
}

/// One phrase's text and span, in seconds.
#[derive(Debug, Clone, Copy)]
pub struct PhraseCase {
    pub text: &'static str,
    pub start: f64,
    pub end: f64,
}

/// A signal `sounding()` (and, where `phrases` holds any, `pace()`) reads.
#[derive(Debug, Clone)]
pub struct SoundingCase {
    pub name: &'static str,
    pub rate: u32,
    pub n: usize,
    pub ops: Vec<Op>,
    pub spans: Vec<(f64, f64)>,
    pub phrases: Vec<PhraseCase>,
}

/// Every sounding/pace signal case, in the oracle's order.
pub fn sounding_cases() -> Vec<SoundingCase> {
    vec![
        SoundingCase {
            name: "a short pause before and after one phrase",
            rate: 48000,
            n: 24000,
            ops: vec![floor(24000), tone("200", 4800, 14400, 0.3, 480, 480)],
            spans: vec![(0.0, 0.5)],
            phrases: vec![PhraseCase {
                text: "Berlumpur itu jauh",
                start: 0.0,
                end: 0.5,
            }],
        },
        SoundingCase {
            name: "a long pause before and after one phrase",
            rate: 48000,
            n: 96000,
            ops: vec![floor(96000), tone("200", 33600, 28800, 0.3, 960, 960)],
            spans: vec![(0.0, 2.0)],
            phrases: vec![PhraseCase {
                text: "Kuala Lumpur ialah ibu negara",
                start: 0.0,
                end: 2.0,
            }],
        },
        SoundingCase {
            name: "two phrases with a pause between them",
            rate: 48000,
            n: 96000,
            ops: vec![
                floor(96000),
                tone("200", 4800, 28800, 0.3, 480, 480),
                tone("180", 57600, 28800, 0.25, 480, 480),
            ],
            spans: vec![(0.0, 1.0), (1.0, 2.0)],
            phrases: vec![
                PhraseCase {
                    text: "well-known café",
                    start: 0.0,
                    end: 1.0,
                },
                PhraseCase {
                    text: "kanak-kanak",
                    start: 1.0,
                    end: 2.0,
                },
            ],
        },
        SoundingCase {
            name: "a span that is only silence",
            rate: 48000,
            n: 4800,
            ops: vec![floor(4800)],
            spans: vec![(0.0, 0.1)],
            phrases: vec![],
        },
        SoundingCase {
            name: "a span past the end of the audio",
            rate: 48000,
            n: 9600,
            ops: vec![floor(9600), tone("200", 0, 9600, 0.3, 480, 480)],
            spans: vec![(0.1, 0.5)],
            phrases: vec![],
        },
        SoundingCase {
            name: "a fractional hop at 22050 Hz",
            rate: 22050,
            n: 22050,
            ops: vec![
                noise(0, 22050, 0.00002, 7),
                tone("200", 4410, 13230, 0.3, 441, 441),
            ],
            spans: vec![(0.0, 1.0)],
            phrases: vec![],
        },
        SoundingCase {
            name: "a span starting before the audio",
            rate: 48000,
            n: 9600,
            ops: vec![floor(9600), tone("200", 0, 9600, 0.3, 480, 480)],
            spans: vec![(-0.1, 0.2)],
            phrases: vec![],
        },
    ]
}

/// A `pace()` case with no audio: each phrase's whole span counts as
/// speech, pauses included.
#[derive(Debug, Clone)]
pub struct PaceNoAudioCase {
    pub name: &'static str,
    pub phrases: Vec<PhraseCase>,
}

/// Every no-audio pace case, in the oracle's order.
pub fn pace_no_audio_cases() -> Vec<PaceNoAudioCase> {
    vec![
        PaceNoAudioCase {
            name: "one phrase",
            phrases: vec![PhraseCase {
                text: "Berlumpur",
                start: 0.0,
                end: 1.0,
            }],
        },
        PaceNoAudioCase {
            name: "phrases with pauses of various lengths between them",
            phrases: vec![
                PhraseCase {
                    text: "well-known café",
                    start: 0.0,
                    end: 0.9,
                },
                PhraseCase {
                    text: "kanak-kanak",
                    start: 1.2,
                    end: 1.9,
                },
                PhraseCase {
                    text: "Kuala Lumpur",
                    start: 2.5,
                    end: 3.1,
                },
            ],
        },
        PaceNoAudioCase {
            name: "no phrases",
            phrases: vec![],
        },
    ]
}

/// A `pace_verdict()` case: a measured pace, a target and the speed a take
/// was made at.
#[derive(Debug, Clone, Copy)]
pub struct VerdictCase {
    pub name: &'static str,
    pub measured_pace: f64,
    pub target: f64,
    pub speed: f64,
}

/// Every verdict case, in the oracle's order.
pub fn verdict_cases() -> Vec<VerdictCase> {
    vec![
        VerdictCase {
            name: "in range at the default target",
            measured_pace: 4.1,
            target: 4.1,
            speed: 1.0,
        },
        VerdictCase {
            name: "faster than the default range",
            measured_pace: 4.75,
            target: 4.1,
            speed: 1.0,
        },
        VerdictCase {
            name: "slower than the default range",
            measured_pace: 3.0,
            target: 4.1,
            speed: 1.0,
        },
        VerdictCase {
            name: "just past the high edge, rounds into it",
            measured_pace: 4.749,
            target: 4.1,
            speed: 1.0,
        },
        VerdictCase {
            name: "a custom target and prior speed",
            measured_pace: 5.2,
            target: 3.9,
            speed: 0.9,
        },
        VerdictCase {
            name: "zero measured pace",
            measured_pace: 0.0,
            target: 4.1,
            speed: 1.0,
        },
    ]
}
