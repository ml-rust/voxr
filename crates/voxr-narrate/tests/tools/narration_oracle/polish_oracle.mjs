// Reference outputs for the voice polish in `voxr_narrate`: runs the JavaScript speechLevel, compressVoice,
// deEss, normalizePeak and polish over POLISH_CASES, quietBreaths, breathRuns, bassBalance, liftBass and
// prepareReference over REFERENCE_CASES (both in polish_cases.mjs), and the 16-bit writing over PCM16_VALUES, and
// prints one JSON document to stdout. Sample outputs are written as FNV-1a hashes and numbers as JavaScript's
// String(x), which reads back as the exact double.
//
// speechLevel, breathRuns, the level evening and wav() are not exported, so the oracle imports a copy of speak.mjs
// that exports speechLevel, breathRuns and wav, copies polish's samples at three points (as read, after the level
// evening, and before writing), and records each shortfall liftBass measures and each shelf gain it applies. Each
// probe is inserted at a line that must occur exactly once; nothing else changes. polish reads its take
// through ffmpeg, so the input is written as a 32-bit float WAV, which ffmpeg passes through unchanged (the input
// hash checks it), and the 16-bit file polish writes is hashed from its data bytes.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs> node polish_oracle.mjs > ../../fixtures/narration_polish_oracle.json
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { POLISH_CASES, PCM16_VALUES, REFERENCE_CASES } from './polish_cases.mjs';
import { fnv1a, fnv1aBytes, render } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('polish_oracle: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs.');
  process.exit(2);
}

// Each probe: a line of polish, and the copy inserted after or before it.
const PROBES = [
  { at: '  let x = samples(file);\n', after: '  globalThis.polishProbe.input = Float32Array.from(x);\n' },
  { at: '  x = compressVoice(x, rate);\n', before: '  globalThis.polishProbe.leveled = Float32Array.from(x);\n' },
  { at: '  writeFileSync(file, wav([x], rate).buf);\n', before: '  globalThis.polishProbe.polished = Float32Array.from(x);\n' },
  { at: '    const short = -bassBalance(y);\n', after: '    globalThis.bassProbe?.shorts.push(short);\n' },
  { at: '    gain += step;\n', after: '    globalThis.bassProbe?.steps.push(step);\n' },
];

const dir = mkdtempSync(join(tmpdir(), 'polish-oracle-'));
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', before = '' } of PROBES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`polish_oracle: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => before + at + after);
  }
  text = text.replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`).replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  text += '\nexport { breathRuns, speechLevel, wav };\n';
  const probed = join(dir, 'speak_probed.mjs');
  writeFileSync(probed, text);
  const speak = await import(pathToFileURL(probed).href);

  const num = (v) => (v == null ? null : String(v));
  const wavF32 = (x, rate) => {
    const buf = Buffer.alloc(44 + x.length * 4);
    buf.write('RIFF', 0);
    buf.writeUInt32LE(36 + x.length * 4, 4);
    buf.write('WAVEfmt ', 8);
    buf.writeUInt32LE(16, 16);
    buf.writeUInt16LE(3, 20);
    buf.writeUInt16LE(1, 22);
    buf.writeUInt32LE(rate, 24);
    buf.writeUInt32LE(rate * 4, 28);
    buf.writeUInt16LE(4, 32);
    buf.writeUInt16LE(32, 34);
    buf.write('data', 36);
    buf.writeUInt32LE(x.length * 4, 40);
    Buffer.from(x.buffer, x.byteOffset, x.byteLength).copy(buf, 44);
    return buf;
  };

  const cases = POLISH_CASES.map(({ name, rate, n, spans, compress, ops }, i) => {
    const x = render(n, ops);
    const out = { name, rate, n, input: fnv1a(x) };
    out.compress = fnv1a(speak.compressVoice(x, rate));
    if (compress) out.compressOptions = fnv1a(speak.compressVoice(x, rate, compress));
    const stats = {};
    out.deEss = { samples: fnv1a(speak.deEss(x, rate, stats)), share: num(stats.share), deepest: num(stats.deepest) };
    out.normalize = fnv1a(speak.normalizePeak(x));
    out.normalizeMinus6 = fnv1a(speak.normalizePeak(x, -6));
    if (spans) {
      out.levels = spans.map(([start, end]) => num(speak.speechLevel(x, start, end, rate)));
      const file = join(dir, `case${i}.wav`);
      writeFileSync(file, wavF32(x, rate));
      globalThis.polishProbe = {};
      const done = speak.polish(file, spans.map(([start, end]) => ({ text: '', start, end })));
      const probe = globalThis.polishProbe;
      if (fnv1a(probe.input) !== out.input) throw new Error(`polish_oracle: ${name}: ffmpeg changed the input samples`);
      const s = done.stats;
      out.polish = {
        leveled: fnv1a(probe.leveled),
        samples: fnv1a(probe.polished),
        pcm16: fnv1aBytes(readFileSync(file).subarray(44)),
        stats: { leveled: s.leveled, maxGain: num(s.maxGain), essShare: num(s.essShare), essDeepest: num(s.essDeepest) },
      };
    }
    return out;
  });

  // liftBass run with its probe set: its output, the shortfalls it measured and the shelf gains it applied.
  const withProbe = (run) => {
    globalThis.bassProbe = { shorts: [], steps: [] };
    const y = run();
    const { shorts, steps } = globalThis.bassProbe;
    globalThis.bassProbe = undefined;
    return { y, shorts, steps };
  };
  const reference = REFERENCE_CASES.map(({ name, rate, n, ops }) => {
    const x = render(n, ops);
    const out = { name, rate, n, input: fnv1a(x) };
    const quiet = speak.quietBreaths(x, rate);
    out.breaths = { samples: fnv1a(quiet.samples), runs: speak.breathRuns(x, rate), count: quiet.runs };
    let gain;
    const lift = withProbe(() => {
      const r = speak.liftBass(x, rate);
      gain = r.gain;
      return r.samples;
    });
    out.bass = { balance: num(speak.bassBalance(x)), shorts: lift.shorts.map(num), steps: lift.steps.map(num), gain: num(gain), samples: fnv1a(lift.y) };
    const prepared = withProbe(() => speak.prepareReference(x));
    out.prepare = { samples: fnv1a(prepared.y), shelves: prepared.steps.length };
    return out;
  });

  const values = Float32Array.from(PCM16_VALUES);
  const data = speak.wav([values], 48000).buf.subarray(44);
  const pcm16 = { values: Array.from(values, String), ints: Array.from({ length: values.length }, (_, k) => data.readInt16LE(2 * k)) };

  process.stdout.write(`${JSON.stringify({ cases, pcm16, reference }, null, 1)}\n`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
