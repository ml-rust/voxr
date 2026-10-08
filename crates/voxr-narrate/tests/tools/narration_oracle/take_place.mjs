// Reference outputs for the timed placement and the time-stretch in `voxr_narrate`: runs speak.mjs's own
// `placeOnTimes` and `stretch` on synthetic speech. Prints one JSON document to stdout.
//
// The patch exports both functions and the skill's `wav` writer, and hooks `wav` to record the samples it joins.
// Nothing else is stubbed: `placeOnTimes` decodes its input with ffmpeg, and `stretch` runs ffmpeg's atempo and
// resampler and reads the duration back with ffprobe.
//
// Each input is written by the skill's own `wav`: a tone per spoken span over a noise floor (signals.mjs), at the
// case's rate. A place case records the input's ops, the spans, the timed lines, how a line counts its spans
// (phrases, or VoxCPM2's line chunks), and the result: duration, phrases, late lines, the length of the placed
// samples, and for a 48 kHz input the FNV-1a hash of the samples `wav` joins and of the WAV's bytes. A case whose
// spans run out records that `placeOnTimes` throws. A stretch case records the input, the speed, the spans and the
// result: duration, phrases, the number of samples written, and for a 48 kHz input the hash of the 16-bit data the
// WAV holds. A stretch case with timed lines also places them on the stretched track, as `speak` does.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_place.mjs > ../../fixtures/narration_place_oracle.json
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fnv1a, fnv1aBytes, noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_place: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what follows it.
const PATCHES = [{ at: 'function wav(parts, rate) {\n', after: '  globalThis.oracleWav?.(parts, rate);\n' }];
const EXPORTS = '\nexport { placeOnTimes, stretch, wav as oracleWavFile };\n';

// Synthetic speech: `secs` seconds at `rate` with a noise floor, and a tone over each [start, end] span in seconds.
function speech(rate, secs, spans, seed) {
  const n = Math.round(secs * rate);
  const ops = [noise(0, n, 0.0001, seed)];
  for (const [start, end] of spans) {
    const a = Math.round(start * rate);
    ops.push(tone(200, a, Math.round(end * rate) - a, 0.3, 480, 480));
  }
  return { n, ops };
}

// Spoken spans: each text with its [start, end].
const said = (list) => list.map(([text, start, end]) => ({ text, start, end }));
const cue = (start, end, text) => ({ start, end, text });

const PLACE = [
  {
    name: 'lines on time',
    rate: 48000,
    secs: 3.6,
    spoken: said([['The river runs north.', 0.1, 1.0], ['It turns east.', 1.3, 2.1], ['Herds come back.', 2.4, 3.4]]),
    cues: [cue(0, null, 'The river runs north.'), cue(3, null, 'It turns east.'), cue(6, null, 'Herds come back.')],
    units: 'phrases',
  },
  {
    name: 'a late start',
    rate: 48000,
    secs: 3.4,
    spoken: said([['A long opening line.', 0.05, 2.25], ['Then the next.', 2.6, 3.3]]),
    cues: [cue(0, 2.5, 'A long opening line.'), cue(2, 4, 'Then the next.')],
    units: 'phrases',
  },
  {
    name: 'an overrun of the line end',
    rate: 48000,
    secs: 2.2,
    spoken: said([['Too much to say.', 0.1, 1.6], ['Later.', 1.8, 2.1]]),
    cues: [cue(0.5, 1.5, 'Too much to say.'), cue(4, null, 'Later.')],
    units: 'phrases',
  },
  {
    name: 'an overrun of the next line',
    rate: 48000,
    secs: 2.9,
    spoken: said([['This one runs on.', 0.0, 2.0], ['So this starts late.', 2.3, 2.8]]),
    cues: [cue(0, null, 'This one runs on.'), cue(1.9, null, 'So this starts late.')],
    units: 'phrases',
  },
  {
    name: 'several phrases per line',
    rate: 48000,
    secs: 3.6,
    spoken: said([['First part.', 0.1, 0.7], ['Second part.', 1.0, 1.6], ['Third part.', 2.05, 2.6], ['Done.', 3.0, 3.4]]),
    cues: [cue(1, null, 'First part. Second part.\nThird part.'), cue(5.5, null, 'Done.')],
    units: 'phrases',
  },
  {
    name: 'VoxCPM2 line chunks',
    rate: 48000,
    secs: 2.6,
    spoken: said([['One here. Two there.', 0.1, 1.4], ['Last line.', 1.7, 2.5]]),
    cues: [cue(0.4, null, 'One here. Two there.'), cue(2, null, 'Last line.')],
    units: 'lines',
  },
  {
    name: 'a long line quoted, spans past the last line dropped',
    rate: 48000,
    secs: 3.0,
    spoken: said([['Café 🎙 notes run well past forty characters here.', 0.2, 2.0], ['Unplaced.', 2.3, 2.9]]),
    cues: [cue(0.5, 1.5, 'Café 🎙 notes run well past forty characters here.')],
    units: 'phrases',
  },
  {
    name: 'spans run out',
    rate: 48000,
    secs: 1.0,
    spoken: said([['Only one.', 0.1, 0.9]]),
    cues: [cue(0, null, 'Only one.'), cue(2, null, 'Missing.')],
    units: 'phrases',
  },
  {
    name: 'a 24 kHz take',
    rate: 24000,
    secs: 2.4,
    spoken: said([['Kokoro speaks.', 0.1, 1.1], ['At its own rate.', 1.4, 2.3]]),
    cues: [cue(0, null, 'Kokoro speaks.'), cue(2, null, 'At its own rate.')],
    units: 'phrases',
  },
];

const STRETCH_SPANS = said([['The river runs north.', 0.1, 0.9], ['It turns east.', 1.2, 1.9]]);
const STRETCH = [
  { name: '48 kHz at 0.8', rate: 48000, speed: 0.8 },
  { name: '48 kHz at 1.25', rate: 48000, speed: 1.25 },
  { name: '48 kHz at 1.5', rate: 48000, speed: 1.5 },
  { name: '16 kHz at 1.25', rate: 16000, speed: 1.25 },
  { name: '24 kHz at 0.8', rate: 24000, speed: 0.8 },
  {
    name: '48 kHz at 1.25, then placed',
    rate: 48000,
    speed: 1.25,
    cues: [cue(0.5, null, 'The river runs north.'), cue(1.0, null, 'It turns east.')],
    units: 'phrases',
  },
];

// The body of a WAV's data chunk. ffmpeg's files carry a LIST chunk before it.
function dataChunk(buf) {
  let at = 12;
  while (at + 8 <= buf.length) {
    const size = buf.readUInt32LE(at + 4);
    if (buf.toString('latin1', at, at + 4) === 'data') return buf.subarray(at + 8, at + 8 + size);
    at += 8 + size + (size & 1);
  }
  throw new Error('take_place: the WAV holds no data chunk');
}

const dir = mkdtempSync(join(tmpdir(), 'place-oracle-'));
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_place: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => at + after);
  }
  const base = pathToFileURL(join(jsDir, '/')).href;
  text = text
    .replaceAll("from './", `from '${base}`)
    .replaceAll("import('./", `import('${base}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text + EXPORTS);
  const speak = await import(pathToFileURL(patched).href);
  const unitsOf = { phrases: (t) => speak.phrases(t).length, lines: (t) => speak.chunks(t, { lines: true }).length };

  // The input WAV at `file`, written by the skill's `wav`.
  const writeInput = (file, rate, signal) => {
    globalThis.oracleWav = null;
    writeFileSync(file, speak.oracleWavFile([render(signal.n, signal.ops)], rate).buf);
  };
  // `placeOnTimes` on `file`, with the samples its `wav` joins.
  const place = (file, spoken, cues, units) => {
    let joined = null;
    globalThis.oracleWav = (parts) => {
      const all = new Float32Array(parts.reduce((k, p) => k + p.length, 0));
      let o = 0;
      for (const p of parts) (all.set(p, o), (o += p.length));
      joined = all;
    };
    const r = speak.placeOnTimes(file, spoken, cues, unitsOf[units]);
    return {
      duration: String(r.duration),
      phrases: JSON.stringify(r.phrases, null, 1),
      late: r.late,
      length: joined.length,
      samples: fnv1a(joined),
      wav: fnv1aBytes(readFileSync(file)),
    };
  };

  const placeCases = PLACE.map((c, k) => {
    const signal = speech(c.rate, c.secs, c.spoken.map((s) => [s.start, s.end]), 101 + k);
    const file = join(dir, `place-${k}.wav`);
    writeInput(file, c.rate, signal);
    let result = null;
    let error = false;
    try {
      result = place(file, c.spoken, c.cues, c.units);
      // ffmpeg's resampler decodes a take at another rate: only the length and the timing carry over.
      if (c.rate !== 48000) result = { ...result, samples: null, wav: null };
    } catch {
      error = true;
    }
    return { name: c.name, rate: c.rate, n: signal.n, ops: signal.ops, spoken: c.spoken, cues: c.cues, units: c.units, error, result };
  });

  const stretchCases = STRETCH.map((c, k) => {
    const signal = speech(c.rate, 2.0, STRETCH_SPANS.map((s) => [s.start, s.end]), 201 + k);
    const file = join(dir, `stretch-${k}.wav`);
    writeInput(file, c.rate, signal);
    const s = speak.stretch(file, STRETCH_SPANS, c.speed);
    const data = dataChunk(readFileSync(file));
    const exact = c.rate === 48000;
    const result = {
      duration: String(s.duration),
      phrases: JSON.stringify(s.phrases, null, 1),
      length: data.length / 2,
      pcm: exact ? fnv1aBytes(data) : null,
    };
    const placed = c.cues ? place(file, s.phrases, c.cues, c.units) : null;
    return { name: c.name, rate: c.rate, n: signal.n, ops: signal.ops, speed: c.speed, phrases: STRETCH_SPANS, cues: c.cues ?? null, units: c.units ?? null, result, placed };
  });

  process.stdout.write(`${JSON.stringify({ place: placeCases, stretch: stretchCases }, null, 1)}\n`);
} finally {
  delete globalThis.oracleWav;
  rmSync(dir, { recursive: true, force: true });
}
