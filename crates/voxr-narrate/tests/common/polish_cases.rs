//! Signals the polish oracle runs the voice polish and the reference
//! preparation over, as `tests/tools/narration_oracle/polish_cases.mjs`
//! lists them, in the same order.

use voxr_narrate::CompressorOptions;

use super::signals::{Op, dc, hiss, noise, tone};

/// A signal of `n` samples at `rate` Hz and what is read on it.
#[derive(Debug, Clone)]
pub struct PolishCase {
    pub name: &'static str,
    pub rate: u32,
    pub n: usize,
    /// The sentences polish levels, in seconds; `None` skips the speech
    /// levels and polish.
    pub spans: Option<Vec<(f64, f64)>>,
    /// Compressor settings run besides the defaults.
    pub compress: Option<CompressorOptions>,
    pub ops: Vec<Op>,
}

/// A quiet noise floor under every 48 kHz case.
fn floor(n: usize) -> Op {
    noise(0, n, 0.00002, 3)
}

fn case(name: &'static str, n: usize, spans: &[(f64, f64)], ops: Vec<Op>) -> PolishCase {
    PolishCase {
        name,
        rate: 48000,
        n,
        spans: Some(spans.to_vec()),
        compress: None,
        ops,
    }
}

/// Every case, in the oracle's order.
pub fn polish_cases() -> Vec<PolishCase> {
    let loud = CompressorOptions {
        threshold: -30.0,
        ratio: 4.0,
        knee: 6.0,
        attack: 0.01,
        release: 0.1,
        makeup: 0.0,
    };
    vec![
        case(
            "sentences within, beyond and under the level limits",
            96000,
            &[(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)],
            vec![
                floor(96000),
                tone("200", 0, 24000, 0.3, 480, 480),
                tone("200", 24000, 24000, 0.15, 480, 480),
                tone("200", 48000, 24000, 0.16, 480, 480),
                tone("200", 72000, 24000, 0.04, 480, 480),
            ],
        ),
        case(
            "one sentence with speech",
            48000,
            &[(0.0, 0.5), (0.5, 0.9), (0.9, 1.5)],
            vec![floor(48000), tone("200", 0, 24000, 0.3, 480, 480)],
        ),
        case(
            "sentences past both ends of the take",
            48000,
            &[(-0.1, 0.4), (0.4, 1.3)],
            vec![
                floor(48000),
                tone("200", 0, 19200, 0.3, 480, 480),
                tone("150", 19200, 28800, 0.08, 480, 480),
            ],
        ),
        case(
            "quiet voice under the knee",
            24000,
            &[],
            vec![floor(24000), tone("200", 0, 24000, 0.004, 480, 480)],
        ),
        case(
            "voice at the knee",
            24000,
            &[],
            vec![floor(24000), tone("200", 0, 24000, 0.05, 480, 480)],
        ),
        PolishCase {
            compress: Some(loud),
            ..case(
                "loud voice over the knee",
                24000,
                &[],
                vec![floor(24000), tone("200", 0, 24000, 0.9, 480, 480)],
            )
        },
        case(
            "sibilant burst over a voiced tone",
            48000,
            &[(0.0, 1.0)],
            vec![
                floor(48000),
                tone("200", 0, 48000, 0.2, 480, 480),
                noise(19200, 4800, 0.5, 21),
            ],
        ),
        case(
            "dull voice",
            48000,
            &[(0.0, 1.0)],
            vec![floor(48000), tone("200", 0, 48000, 0.3, 480, 480)],
        ),
        case("no vowel energy", 24000, &[(0.0, 0.5)], vec![]),
        PolishCase {
            name: "44.1 kHz sibilant burst",
            rate: 44100,
            n: 44100,
            spans: None,
            compress: None,
            ops: vec![
                noise(0, 44100, 0.00002, 5),
                tone("200_44", 0, 44100, 0.2, 441, 441),
                noise(17640, 4410, 0.5, 23),
            ],
        },
        case(
            "voice with a DC offset",
            48000,
            &[(0.0, 1.0)],
            vec![
                floor(48000),
                tone("200", 0, 48000, 0.3, 480, 480),
                dc(0, 48000, 0.1),
            ],
        ),
        case(
            "buffer shorter than a frame",
            700,
            &[(0.0, 0.01)],
            vec![noise(0, 700, 0.1, 9)],
        ),
        case("empty buffer", 0, &[], vec![]),
        case(
            "whole chain: uneven sentences, a sibilant, an offset",
            144000,
            &[(0.0, 0.9), (1.0, 1.9), (2.0, 2.9)],
            vec![
                floor(144000),
                tone("150", 0, 43200, 0.35, 960, 960),
                tone("200", 48000, 43200, 0.12, 960, 960),
                tone("180", 96000, 43200, 0.2, 960, 960),
                noise(110400, 3840, 0.8, 31),
                dc(0, 144000, 0.02),
            ],
        ),
    ]
}

/// Values the 16-bit writing is read on, as `f32`.
pub fn pcm16_values() -> Vec<f32> {
    vec![
        1.0,
        -1.0,
        1.5,
        -1.5,
        2.0,
        -2.0,
        0.5,
        -0.5,
        f32::from_bits(0x3780_0100),
        -f32::from_bits(0x3780_0100),
        0.25,
        -0.75,
        0.0,
    ]
}

/// A 48 kHz signal of `n` samples the reference preparation runs over.
#[derive(Debug, Clone)]
pub struct ReferenceCase {
    pub name: &'static str,
    pub rate: u32,
    pub n: usize,
    pub ops: Vec<Op>,
}

fn reference(name: &'static str, n: usize, ops: Vec<Op>) -> ReferenceCase {
    ReferenceCase {
        name,
        rate: 48000,
        n,
        ops,
    }
}

/// Every reference case, in the oracle's order.
pub fn reference_cases() -> Vec<ReferenceCase> {
    vec![
        reference(
            "breathy take: breaths between phrases",
            72000,
            vec![
                floor(72000),
                tone("200", 0, 14400, 0.3, 480, 480),
                hiss(15360, 7200, 0.009, 41, 480, 480),
                tone("200", 23520, 14400, 0.3, 480, 480),
                tone("200", 38880, 14400, 0.3, 480, 480),
                hiss(54240, 7200, 0.009, 43, 480, 480),
                tone("200", 62400, 9600, 0.3, 480, 480),
            ],
        ),
        reference(
            "clean take: pauses without breath",
            48000,
            vec![
                floor(48000),
                tone("200", 0, 14400, 0.3, 480, 480),
                tone("150", 15360, 14400, 0.25, 480, 480),
                tone("200", 30720, 17280, 0.3, 480, 480),
            ],
        ),
        reference(
            "breath shorter than 8 frames",
            38400,
            vec![
                floor(38400),
                tone("200", 0, 14400, 0.3, 480, 480),
                hiss(15360, 2400, 0.009, 45, 480, 480),
                tone("200", 18720, 19680, 0.3, 480, 480),
            ],
        ),
        reference(
            "thin voice lifted in two passes",
            12000,
            vec![
                floor(12000),
                tone("240", 0, 12000, 0.16, 480, 480),
                tone("1500", 0, 12000, 0.2, 480, 480),
            ],
        ),
        reference(
            "balanced voice left as it is",
            12000,
            vec![
                floor(12000),
                tone("240", 0, 12000, 0.3, 480, 480),
                tone("1500", 0, 12000, 0.1, 480, 480),
            ],
        ),
        reference(
            "thin voice held at the +9 dB cap",
            12000,
            vec![
                floor(12000),
                tone("240", 0, 12000, 0.03, 480, 480),
                tone("1500", 0, 12000, 0.3, 480, 480),
            ],
        ),
        reference(
            "thin voice reaching the cap on the second pass",
            12000,
            vec![
                floor(12000),
                tone("240", 0, 12000, 0.125, 480, 480),
                tone("1500", 0, 12000, 0.2, 480, 480),
            ],
        ),
    ]
}
