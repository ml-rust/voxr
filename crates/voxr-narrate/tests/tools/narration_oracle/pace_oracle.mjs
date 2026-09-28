// Reference outputs for the speaking-pace port in `voxr_narrate`: runs the JavaScript syllables() and sounding()
// (both unexported, so the oracle imports a patched copy of speak.mjs that exports them) over the texts and signals
// in pace_cases.mjs, runs the exported pace() and REF_PACE over the same signals decoded from a written WAV, and
// computes the verdict/suggestion math that video.mjs's speak command runs over pace()'s result (target ± 0.6, the
// rounded rate, the suggested --speed), over VERDICT_CASES. Prints one JSON document to stdout.
//
// pace() takes a file, not raw samples, and decodes it through ffmpeg (samples()); the case signal is written as a
// 32-bit float WAV, which ffmpeg passes through unchanged (the input hash checks it), the same approach
// polish_oracle.mjs uses for compressVoice and friends.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> node pace_oracle.mjs \
//          > ../../fixtures/narration_pace_oracle.json
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { PACE_NO_AUDIO_CASES, SOUNDING_CASES, TEXT_CASES, VERDICT_CASES } from './pace_cases.mjs';
import { fnv1a, render } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('pace_oracle: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

const dir = mkdtempSync(join(tmpdir(), 'pace-oracle-'));
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  text = text.replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`).replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  // sounding() and syllables() are not exported; pace() and REF_PACE already are.
  text += '\nexport { sounding, syllables };\n';
  const probed = join(dir, 'speak_probed.mjs');
  writeFileSync(probed, text);
  const speak = await import(pathToFileURL(probed).href);

  const num = (v) => String(v);

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

  const texts = TEXT_CASES.map(({ name, text: t }) => ({ name, text: t, syllables: speak.syllables(t) }));

  const sounding = SOUNDING_CASES.map(({ name, rate, n, ops, spans, phrases }, i) => {
    const x = render(n, ops);
    const out = { name, rate, n, input: fnv1a(x) };
    out.spans = spans.map(([start, end]) => ({ start, end, value: num(speak.sounding(x, start, end, rate)) }));
    if (phrases?.length) {
      const file = join(dir, `case${i}.wav`);
      writeFileSync(file, wavF32(x, rate));
      out.phrases = phrases;
      out.paceValue = num(speak.pace(phrases, file));
    }
    return out;
  });

  const paceNoAudio = PACE_NO_AUDIO_CASES.map(({ name, phrases }) => ({
    name,
    phrases,
    value: num(speak.pace(phrases)),
  }));

  // The verdict/suggestion math that video.mjs's speak command runs over pace()'s result: the value as printed rounds
  // first, so a pace on the range's edge never reads as outside it; the suggested --speed scales the speed a take
  // was made at and is held between 0.6 and 1.5.
  const verdict = VERDICT_CASES.map(({ name, measuredPace, target, speed }) => {
    const rate = Math.round(measuredPace * 10) / 10;
    const lo = +(target - 0.6).toFixed(1);
    const hi = +(target + 0.6).toFixed(1);
    const suggestedSpeed = Math.round(Math.min(1.5, Math.max(0.6, (speed * target) / rate)) * 100) / 100;
    return { name, measuredPace, target, speed, rate: num(rate), low: num(lo), high: num(hi), suggestedSpeed: num(suggestedSpeed) };
  });

  process.stdout.write(`${JSON.stringify({ refPace: num(speak.REF_PACE), texts, sounding, paceNoAudio, verdict }, null, 1)}\n`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
