// Cases for the speaking-pace oracle (pace_oracle.mjs): texts syllables() reads, signals sounding() and pace() read
// with pauses of various lengths, and the verdict/suggestion inputs that video.mjs's speak command computes pace() against.
// tests/common/pace_cases.rs lists the same cases, in the same order.
import { noise, tone } from './signals.mjs';

// Text syllables() counts: digits, hyphens, unicode accents, Malay words.
export const TEXT_CASES = [
  { name: 'plain word', text: 'Berlumpur' },
  { name: 'accented letter', text: 'café' },
  { name: 'digits and a decimal point', text: 'It rose 13.5 percent.' },
  { name: 'a dotted name', text: 'Built with Node.js today!' },
  { name: 'hyphenated word', text: 'well-known' },
  { name: 'reduplicated malay word', text: 'kanak-kanak' },
  { name: 'malay sentence', text: 'Kuala Lumpur ialah ibu negara Malaysia.' },
  { name: 'empty text', text: '' },
  { name: 'digits only', text: '123 456' },
  { name: 'cjk text', text: '你好，今天好嗎？' },
];

// A quiet noise floor under a 48 kHz signal.
const floor = (n) => noise(0, n, 0.00002, 3);

// Signals sounding() and pace() read, each with named spans (seconds) to probe. `phrases` gives each span text, for
// the full pace() reading; omitted where a case only checks sounding().
export const SOUNDING_CASES = [
  {
    name: 'a short pause before and after one phrase',
    rate: 48000,
    n: 24000,
    ops: [floor(24000), tone('200', 4800, 14400, 0.3, 480, 480)],
    spans: [[0, 0.5]],
    phrases: [{ text: 'Berlumpur itu jauh', start: 0, end: 0.5 }],
  },
  {
    name: 'a long pause before and after one phrase',
    rate: 48000,
    n: 96000,
    ops: [floor(96000), tone('200', 33600, 28800, 0.3, 960, 960)],
    spans: [[0, 2.0]],
    phrases: [{ text: 'Kuala Lumpur ialah ibu negara', start: 0, end: 2.0 }],
  },
  {
    name: 'two phrases with a pause between them',
    rate: 48000,
    n: 96000,
    ops: [
      floor(96000),
      tone('200', 4800, 28800, 0.3, 480, 480),
      tone('180', 57600, 28800, 0.25, 480, 480),
    ],
    spans: [
      [0, 1.0],
      [1.0, 2.0],
    ],
    phrases: [
      { text: 'well-known café', start: 0, end: 1.0 },
      { text: 'kanak-kanak', start: 1.0, end: 2.0 },
    ],
  },
  {
    name: 'a span that is only silence',
    rate: 48000,
    n: 4800,
    ops: [floor(4800)],
    spans: [[0, 0.1]],
  },
  {
    name: 'a span past the end of the audio',
    rate: 48000,
    n: 9600,
    ops: [floor(9600), tone('200', 0, 9600, 0.3, 480, 480)],
    spans: [[0.1, 0.5]],
  },
  {
    name: 'a fractional hop at 22050 Hz',
    rate: 22050,
    n: 22050,
    ops: [noise(0, 22050, 0.00002, 7), tone('200', 4410, 13230, 0.3, 441, 441)],
    spans: [[0, 1.0]],
  },
  {
    name: 'a span starting before the audio',
    rate: 48000,
    n: 9600,
    ops: [floor(9600), tone('200', 0, 9600, 0.3, 480, 480)],
    spans: [[-0.1, 0.2]],
  },
];

// pace() with no audio: each phrase's whole span counts as speech, pauses included.
export const PACE_NO_AUDIO_CASES = [
  {
    name: 'one phrase',
    phrases: [{ text: 'Berlumpur', start: 0, end: 1.0 }],
  },
  {
    name: 'phrases with pauses of various lengths between them',
    phrases: [
      { text: 'well-known café', start: 0, end: 0.9 },
      { text: 'kanak-kanak', start: 1.2, end: 1.9 },
      { text: 'Kuala Lumpur', start: 2.5, end: 3.1 },
    ],
  },
  { name: 'no phrases', phrases: [] },
];

// pace_verdict()'s inputs: a measured pace, a target and the speed a take was made at.
export const VERDICT_CASES = [
  { name: 'in range at the default target', measuredPace: 4.1, target: 4.1, speed: 1 },
  { name: 'faster than the default range', measuredPace: 4.75, target: 4.1, speed: 1 },
  { name: 'slower than the default range', measuredPace: 3.0, target: 4.1, speed: 1 },
  { name: 'just past the high edge, rounds into it', measuredPace: 4.749, target: 4.1, speed: 1 },
  { name: 'a custom target and prior speed', measuredPace: 5.2, target: 3.9, speed: 0.9 },
  { name: 'zero measured pace', measuredPace: 0, target: 4.1, speed: 1 },
];
