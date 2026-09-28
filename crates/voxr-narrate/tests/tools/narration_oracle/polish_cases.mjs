// Signals the polish oracle runs the voice polish over, built from the recipes in signals.mjs.
// tests/common/polish_cases.rs lists the same cases in the same order.
import { dc, noise, tone } from './signals.mjs';

// A quiet noise floor under every 48 kHz case, so no stretch is exact silence.
const floor = (n) => noise(0, n, 0.00002, 3);

// Each case: a signal of `n` samples at `rate` Hz. `spans` (seconds) are the sentences polish levels; a case without
// them skips speechLevel and polish. `compress` names compressor settings run besides the defaults.
export const POLISH_CASES = [
  {
    // Median level is the 0.16 span's: 0.3 is 5.5 dB over it, 0.15 is 0.56 dB under it (left as it is), 0.04 is
    // 12 dB under it (held to 6 dB).
    name: 'sentences within, beyond and under the level limits',
    rate: 48000,
    n: 96000,
    spans: [[0, 0.5], [0.5, 1], [1, 1.5], [1.5, 2]],
    ops: [floor(96000), tone(200, 0, 24000, 0.3, 480, 480), tone(200, 24000, 24000, 0.15, 480, 480), tone(200, 48000, 24000, 0.16, 480, 480), tone(200, 72000, 24000, 0.04, 480, 480)],
  },
  {
    name: 'one sentence with speech',
    rate: 48000,
    n: 48000,
    spans: [[0, 0.5], [0.5, 0.9], [0.9, 1.5]],
    ops: [floor(48000), tone(200, 0, 24000, 0.3, 480, 480)],
  },
  {
    // The first span starts before the take, the second ends after it. The louder is the median of two.
    name: 'sentences past both ends of the take',
    rate: 48000,
    n: 48000,
    spans: [[-0.1, 0.4], [0.4, 1.3]],
    ops: [floor(48000), tone(200, 0, 19200, 0.3, 480, 480), tone(150, 19200, 28800, 0.08, 480, 480)],
  },
  { name: 'quiet voice under the knee', rate: 48000, n: 24000, spans: [], ops: [floor(24000), tone(200, 0, 24000, 0.004, 480, 480)] },
  { name: 'voice at the knee', rate: 48000, n: 24000, spans: [], ops: [floor(24000), tone(200, 0, 24000, 0.05, 480, 480)] },
  {
    name: 'loud voice over the knee',
    rate: 48000,
    n: 24000,
    spans: [],
    compress: { threshold: -30, ratio: 4, knee: 6, attack: 0.01, release: 0.1, makeup: 0 },
    ops: [floor(24000), tone(200, 0, 24000, 0.9, 480, 480)],
  },
  {
    name: 'sibilant burst over a voiced tone',
    rate: 48000,
    n: 48000,
    spans: [[0, 1]],
    ops: [floor(48000), tone(200, 0, 48000, 0.2, 480, 480), noise(19200, 4800, 0.5, 21)],
  },
  { name: 'dull voice', rate: 48000, n: 48000, spans: [[0, 1]], ops: [floor(48000), tone(200, 0, 48000, 0.3, 480, 480)] },
  { name: 'no vowel energy', rate: 48000, n: 24000, spans: [[0, 0.5]], ops: [] },
  {
    name: '44.1 kHz sibilant burst',
    rate: 44100,
    n: 44100,
    ops: [noise(0, 44100, 0.00002, 5), tone('200_44', 0, 44100, 0.2, 441, 441), noise(17640, 4410, 0.5, 23)],
  },
  { name: 'voice with a DC offset', rate: 48000, n: 48000, spans: [[0, 1]], ops: [floor(48000), tone(200, 0, 48000, 0.3, 480, 480), dc(0, 48000, 0.1)] },
  { name: 'buffer shorter than a frame', rate: 48000, n: 700, spans: [[0, 0.01]], ops: [noise(0, 700, 0.1, 9)] },
  { name: 'empty buffer', rate: 48000, n: 0, spans: [], ops: [] },
  {
    name: 'whole chain: uneven sentences, a sibilant, an offset',
    rate: 48000,
    n: 144000,
    spans: [[0, 0.9], [1, 1.9], [2, 2.9]],
    ops: [
      floor(144000),
      tone(150, 0, 43200, 0.35, 960, 960),
      tone(200, 48000, 43200, 0.12, 960, 960),
      tone(180, 96000, 43200, 0.2, 960, 960),
      noise(110400, 3840, 0.8, 31),
      dc(0, 144000, 0.02),
    ],
  },
];

// Values the 16-bit writing is read on: full scale, beyond it, the exact half-step ties at ±0.5, and 1.5259255e-5,
// 0.4999999995 of a step, which a product rounded to f32 turns into a tie.
export const PCM16_VALUES = [1, -1, 1.5, -1.5, 2, -2, 0.5, -0.5, 1.5259254723787308e-5, -1.5259254723787308e-5, 0.25, -0.75, 0];
