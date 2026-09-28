// Reference outputs for the take loop's pick and split in `voxr_narrate`: runs a patched copy of speak.mjs's
// `voxcpm` (unexported) through its `pick` and `breakUp` rounds, with the model call and the judge stubbed. The
// stub model makes an empty take file per take. The stub judge grades each take by a scripted (text, seed) table:
// a take with no entry passes with 0 badness. `Math.random` returns a fixed sequence per case. The run stops once
// the best takes are known, short of shaping or writing audio. Prints one JSON document to stdout.
//
// Each case records every model and judge call in order, the `speak: part` lines the loop prints, the parts as
// finally spoken, the take kept for each, and the `.takes.json` the run writes.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_pick.mjs > ../../fixtures/narration_pick_oracle.json
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_pick: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it. The first
// six fix machine state and skip the branches that run VoxCPM2 or ffmpeg, as take_files.mjs does. The next three
// hand the oracle the take-file closure, then stub the model call and the judge. The last stops the run once the
// best takes are known.
const PATCHES = [
  {
    at: "  const cache = join(cacheRoot(), 'speak', 'voxcpm');\n",
    replace: "  const cache = join(globalThis.oracleCacheDir ?? cacheRoot(), 'speak', 'voxcpm');\n",
  },
  {
    at: '  const checkpoint = voxcpmCheckpoint();\n',
    replace: '  const checkpoint = globalThis.oracleCheckpoint !== undefined ? globalThis.oracleCheckpoint : voxcpmCheckpoint();\n',
  },
  {
    at: '  if (cloneFrom && !existsSync(cloneFrom)) {\n',
    replace: '  if (cloneFrom && !existsSync(cloneFrom) && !globalThis.oracleSkipHeavy) {\n',
  },
  {
    at: '    if (anchorFile && !existsSync(anchorFile)) {\n',
    replace: '    if (anchorFile && !existsSync(anchorFile) && !globalThis.oracleSkipHeavy) {\n',
  },
  {
    at: '    if (voiceFile) copyFileSync(anchorFile, voiceFile);\n',
    replace: '    if (voiceFile && !globalThis.oracleSkipHeavy) copyFileSync(anchorFile, voiceFile);\n',
  },
  {
    at: "    const textSeed = (text) => seed + (parseInt(createHash('sha1').update(text).digest('hex').slice(0, 6), 16) % 1000);\n",
    after: '    if (globalThis.oracleReady) globalThis.oracleReady({ takeFile, textSeed, seed, chosen });\n',
  },
  {
    at: '    const generate = (list, designing = false) => {\n',
    after: '      if (globalThis.oracleGenerate) return globalThis.oracleGenerate(list, designing, { takeFile, say, style });\n',
  },
  {
    at: '    const judge = async (s) => {\n',
    after: '      if (globalThis.oracleJudge) return globalThis.oracleJudge(s);\n',
  },
  {
    at: '    const voiceMedian = medianPitch(samples(reference || anchorFile));\n',
    replace:
      "    if (globalThis.oracleDone) { globalThis.oracleDone({ units, best, seed }); throw new Error('__oracle_stop__'); }\n    const voiceMedian = medianPitch(samples(reference || anchorFile));\n",
  },
];

// A re-rolled part's seed from one draw, as pick computes it.
const rerollSeed = (r) => 100000 + Math.floor(r * 900000);

// The cases. A score's seed is `{ text, plus }` (the text's own seed plus an offset), `{ fresh, plus }` (the seed
// the fresh-th random draw gives, plus an offset) or a number. A cached take is held before the run starts.
const HELLO = 'Hello there, friend.';
const CASES = [
  { name: 'first try passes', script: HELLO },
  {
    name: 'fails then the retry passes',
    script: 'The river runs north today.',
    scores: [
      { text: 'The river runs north today.', seed: { text: 'The river runs north today.', plus: 0 }, ok: false, badness: 2 },
      { text: 'The river runs north today.', seed: { text: 'The river runs north today.', plus: 1000 }, ok: true, badness: 0 },
    ],
  },
  {
    name: 'every take fails whole, split by beats',
    script: 'One two three.\n\nFour five six.',
    scores: [0, 1000, 2000].map((plus, k) => ({ text: 'One two three. Four five six.', seed: { text: 'One two three. Four five six.', plus }, ok: false, badness: [1, 2, 1.5][k] })),
  },
  {
    name: 'the beat split fails, split by sentences',
    script: 'One two three.\n\nFour five six. Seven eight nine.',
    scores: [
      ...[0, 1000, 2000].map((plus) => ({ text: 'One two three. Four five six. Seven eight nine.', seed: { text: 'One two three. Four five six. Seven eight nine.', plus }, ok: false, badness: 1 })),
      ...[0, 1000, 2000].map((plus, k) => ({ text: 'Four five six. Seven eight nine.', seed: { text: 'Four five six. Seven eight nine.', plus }, ok: false, badness: [1, 0.5, 2][k] })),
      { text: 'One two three.', seed: { text: 'One two three.', plus: 0 }, ok: false, badness: 0.7 },
    ],
  },
  {
    name: 'a re-roll that ties replaces the recorded take',
    script: HELLO,
    record: JSON.stringify({ seed: 7, takes: { [HELLO]: 555 } }),
    cached: [{ text: HELLO, seed: 555 }],
    reroll: [1],
    random: [0.25],
    scores: [
      { text: HELLO, seed: 555, ok: true, badness: 0.5 },
      { text: HELLO, seed: { fresh: 0, plus: 0 }, ok: true, badness: 0.5 },
    ],
  },
  {
    name: 'a worse re-roll keeps the recorded take',
    script: HELLO,
    record: JSON.stringify({ seed: 7, takes: { [HELLO]: 555 } }),
    cached: [{ text: HELLO, seed: 555 }],
    reroll: [1],
    random: [0.5],
    scores: [
      { text: HELLO, seed: 555, ok: true, badness: 0.2 },
      ...[0, 1000, 2000].map((plus, k) => ({ text: HELLO, seed: { fresh: 0, plus }, ok: false, badness: [1, 2, 1.5][k] })),
    ],
  },
  {
    name: 'a recorded take is reused from chosen',
    script: HELLO,
    record: JSON.stringify({ seed: 7, takes: { [HELLO]: 555 } }),
    cached: [{ text: HELLO, seed: 555 }],
  },
  {
    name: 'a new asked seed resets the chosen takes',
    script: HELLO,
    record: JSON.stringify({ seed: 9, takes: { [HELLO]: 555 } }),
    cached: [{ text: HELLO, seed: 555 }],
    asked: 99,
  },
  {
    name: 'a delivery note, a style, a part re-rolled twice and a take with no speech',
    script: '(calm tone) First line here.\nSecond line goes here.\n\nThird line ends it.',
    style: '(warm)',
    reroll: [2, 2],
    random: [0.1, 0.7],
    scores: [
      { text: 'First line here.', seed: { text: 'First line here.', plus: 0 }, ok: false, badness: 1 },
      { text: 'Second line goes here. Third line ends it.', seed: { fresh: 1, plus: 0 }, ok: false, badness: null },
      { text: 'Second line goes here. Third line ends it.', seed: { fresh: 1, plus: 1000 }, ok: false, badness: 3 },
      { text: 'Second line goes here. Third line ends it.', seed: { fresh: 1, plus: 2000 }, ok: false, badness: null },
    ],
  },
];

const dir = mkdtempSync(join(tmpdir(), 'pick-oracle-'));
const realRandom = Math.random;
const realError = console.error;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_pick: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  text = text
    .replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  text += '\nexport { voxcpm };\n';
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text);
  const speak = await import(pathToFileURL(patched).href);

  const cases = [];
  for (const [n, c] of CASES.entries()) {
    const caseDir = join(dir, `case-${n}`);
    mkdirSync(caseDir, { recursive: true });
    const out = join(caseDir, 'out.wav');
    if (c.record != null) writeFileSync(join(caseDir, 'out.takes.json'), c.record);
    const random = c.random ?? [];
    let draws = 0;
    Math.random = () => {
      if (draws >= random.length) throw new Error(`take_pick: case "${c.name}" drew more than its ${random.length} random values`);
      return random[draws++];
    };
    const messages = [];
    console.error = (line) => {
      if (/^speak: part/.test(`${line}`)) messages.push(`${line}`);
    };
    const calls = [];
    let scores = [];
    let result = null;
    globalThis.oracleCacheDir = join(caseDir, 'cache');
    globalThis.oracleCheckpoint = null;
    globalThis.oracleSkipHeavy = true;
    globalThis.oracleReady = ({ takeFile, textSeed }) => {
      const resolve = (seed) => {
        if (typeof seed === 'number') return seed;
        if ('text' in seed) return textSeed(seed.text) + seed.plus;
        return rerollSeed(random[seed.fresh]) + seed.plus;
      };
      scores = (c.scores ?? []).map((s) => ({ text: s.text, seed: resolve(s.seed), ok: s.ok, badness: s.badness }));
      for (const t of c.cached ?? []) writeFileSync(takeFile({ text: t.text, seed: t.seed, ...(t.delivery ? { style: t.delivery } : {}) }), '');
    };
    globalThis.oracleGenerate = (list, designing, { takeFile, say, style }) => {
      if (designing) throw new Error('take_pick: the stub model does not design voices');
      for (const s of list) {
        calls.push({ op: 'speak', text: say(s.text), style: s.style ?? (style || null), seed: s.seed });
        writeFileSync(takeFile(s), '');
      }
    };
    globalThis.oracleJudge = async (s) => {
      calls.push({ op: 'judge', text: s.text, seed: s.seed });
      const hit = scores.find((x) => x.text === s.text && x.seed === s.seed);
      const ok = hit ? hit.ok : true;
      const badness = hit ? (hit.badness ?? undefined) : 0;
      return { ok, badness, why: `bad ${badness ?? 'none'}`, heard: `heard ${s.text}`, seed: s.seed };
    };
    globalThis.oracleDone = ({ units, best, seed }) => {
      result = {
        seed,
        parts: units.map((u) => u.text),
        best: best.map((b) => ({ seed: b.seed, ok: b.ok, badness: b.badness ?? null })),
        takes: JSON.stringify({ seed, takes: Object.fromEntries(best.map((b, i) => [units[i].text, b.seed])) }, null, 1),
      };
    };
    try {
      await speak.voxcpm(c.script, { device: 'cpu', voice: 'af_heart', style: c.style ?? '', seed: c.asked, reroll: c.reroll ?? [] }, out);
      throw new Error(`take_pick: case "${c.name}" ran to completion instead of stopping at the oracle`);
    } catch (e) {
      if (e.message !== '__oracle_stop__') throw e;
    } finally {
      Math.random = realRandom;
      console.error = realError;
    }
    cases.push({
      name: c.name,
      script: c.script,
      style: c.style ?? null,
      record: c.record ?? null,
      asked: c.asked ?? null,
      reroll: c.reroll ?? [],
      random,
      cached: (c.cached ?? []).map((t) => ({ text: t.text, seed: t.seed, delivery: t.delivery ?? null })),
      scores: scores.map((s) => ({ ...s, badness: s.badness ?? null })),
      calls,
      messages,
      ...result,
    });
  }
  process.stdout.write(`${JSON.stringify({ cases }, null, 1)}\n`);
} finally {
  Math.random = realRandom;
  console.error = realError;
  for (const k of ['oracleCacheDir', 'oracleCheckpoint', 'oracleSkipHeavy', 'oracleReady', 'oracleGenerate', 'oracleJudge', 'oracleDone']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
