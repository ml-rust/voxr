// Reference outputs for the join of kept takes in `voxr_narrate`: runs a patched copy of speak.mjs's `voxcpm`
// (unexported) to the end, with the model call and the judge stubbed, so its own `shapeTake`, heat reading and join
// shape and lay the kept takes into one track. Prints one JSON document to stdout.
//
// The designed voice sits in the cache before the run, so no voice is designed, and its median pitch is read for real.
// The stub model writes an empty take file per take. The stub judge looks the take up by its text in the case's table:
// a synthetic take from signals.mjs, floor noise under one tone per sentence with the gap the case names after it,
// written as the skill writes a tidied take (16-bit). Its heard words are the sentence's own words spread evenly over
// its tone, times rounded to hundredths, so alignScript runs for real on them. A take marked `deaf` has no heard words.
// The judge returns the take's verdict, doubtful words, stray pauses and clicks from the table.
//
// Each case records every take the judge returned (its signal ops, the FNV-1a hash of the samples the skill reads
// back, its heard words and its verdict), the parts as finally spoken, each shaped take (the hash of its samples, its
// length and its spans), the hash of the joined samples and of the written file, the duration, the phrases, the
// checks and the problems.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_join.mjs > ../../fixtures/narration_join_oracle.json
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fnv1a, fnv1aBytes, noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_join: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it. The first two
// fix machine state: the cache directory and the checkpoint. The next two stub the model call and the judge. The last
// two hand the oracle each shaped take with the parts and the voice's median pitch, and the joined parts.
const PATCHES = [
  {
    at: "  const cache = join(cacheRoot(), 'speak', 'voxcpm');\n",
    replace: "  const cache = globalThis.oracleCacheDir ?? join(cacheRoot(), 'speak', 'voxcpm');\n",
  },
  {
    at: '  const checkpoint = voxcpmCheckpoint();\n',
    replace: '  const checkpoint = globalThis.oracleCheckpoint !== undefined ? globalThis.oracleCheckpoint : voxcpmCheckpoint();\n',
  },
  {
    at: '    const generate = (list, designing = false) => {\n',
    after: '      if (globalThis.oracleGenerate) return globalThis.oracleGenerate(list, designing, { takeFile });\n',
  },
  {
    at: '    const judge = async (s) => {\n',
    after: '      if (globalThis.oracleJudge) return globalThis.oracleJudge(s);\n',
  },
  {
    at: '    const shaped = best.map((b, i) => shapeTake(samples(b.file), b.words, units[i].sentences, alignScript));\n',
    after: '    globalThis.oracleShaped?.(shaped, units, voiceMedian);\n',
  },
  {
    at: '    const r = wav(parts, 48000);\n',
    after: '    globalThis.oracleJoined?.(parts);\n',
  },
];

const RATE = 48000;
// Floor noise before the first tone and after the last.
const LEAD = 4800;
// The run's own checkpoint name, so the cache keys fall back to nothing and no download notice prints.
const CHECKPOINT = 'openbmb/VoxCPM2';
const VOICE = '(young woman, warm, low)';
const SEED = 7;
// The designed voice held in the cache: one second of a 150 Hz tone.
const ANCHOR_OPS = [noise(0, RATE, 0.00002, 5), tone(150, 0, RATE, 0.3, 480, 480)];

// A take of `sentences`, each [text, hz, tone length, gap after] in samples: its signal ops, its length, and the
// words heard in it.
function take(sentences, seed, deaf) {
  const ops = [];
  const heard = [];
  let at = LEAD;
  for (const [text, hz, len, gap] of sentences) {
    ops.push(tone(hz, at, len, 0.3, 480, 480));
    const words = text.split(/\s+/);
    const step = len / words.length;
    words.forEach((w, k) => heard.push({ text: w, start: Math.round(((at + k * step) / RATE) * 100) / 100, end: Math.round(((at + (k + 1) * step) / RATE - 0.02) * 100) / 100 }));
    at += len + gap;
  }
  const n = at + LEAD;
  return { n, ops: [noise(0, n, 0.00002, seed), ...ops], heard: deaf ? [] : heard };
}

// The cases. `takes` maps each part text the judge meets to its sentences and its verdict: `fail` fails it with one
// badness, and `doubt`, `pauses` and `clicks` are the marks the judge hands on.
const S = 48000;
const CASES = [
  {
    name: 'a beat pause widened',
    script: 'The river runs north today.\n\nThe herds come back to graze.',
    takes: {
      'The river runs north today. The herds come back to graze.': {
        sentences: [
          ['The river runs north today.', 150, S, 9600],
          ['The herds come back to graze.', 260, S, 0],
        ],
        doubt: ['river'],
        clicks: [0.42],
      },
    },
  },
  {
    name: 'a mark pause widened',
    script: 'Wait for it now. [pause 1.2] Here it comes at last.',
    takes: {
      'Wait for it now. Here it comes at last.': {
        sentences: [
          ['Wait for it now.', 150, S, 14400],
          ['Here it comes at last.', 180, S, 0],
        ],
      },
    },
  },
  {
    name: 'a line pause and a long beat left as they are',
    script: 'One two three four.\nFive six seven eight.\n\nNine ten eleven twelve.',
    takes: {
      'One two three four. Five six seven eight. Nine ten eleven twelve.': {
        sentences: [
          ['One two three four.', 150, S, 4800],
          ['Five six seven eight.', 150, S, 57600],
          ['Nine ten eleven twelve.', 180, S, 0],
        ],
        pauses: [{ after: 'six', before: 'seven', gap: 0.5 }],
      },
    },
  },
  {
    name: 'a question skips heat',
    script: 'Where does the river go?\n\nIt runs north to the sea.',
    takes: {
      'Where does the river go? It runs north to the sea.': {
        sentences: [
          ['Where does the river go?', 300, S, 9600],
          ['It runs north to the sea.', 180, S, 0],
        ],
      },
    },
  },
  {
    name: 'several chunks',
    script: '(calm tone) First line here now.\nSecond line goes here.\n\nThird line ends it.',
    takes: {
      'First line here now.': { sentences: [['First line here now.', 150, S, 0]] },
      'Second line goes here. Third line ends it.': {
        sentences: [
          ['Second line goes here.', 180, S, 4800],
          ['Third line ends it.', 150, S, 0],
        ],
      },
    },
  },
  {
    name: 'a part split into sentences',
    script: 'One two three four.\n\nFive six seven eight. Nine ten eleven twelve.',
    takes: {
      'One two three four. Five six seven eight. Nine ten eleven twelve.': {
        sentences: [
          ['One two three four.', 150, S, 9600],
          ['Five six seven eight.', 150, S, 4800],
          ['Nine ten eleven twelve.', 150, S, 0],
        ],
        fail: true,
      },
      'One two three four.': { sentences: [['One two three four.', 150, S, 0]] },
      'Five six seven eight. Nine ten eleven twelve.': {
        sentences: [
          ['Five six seven eight.', 150, S, 4800],
          ['Nine ten eleven twelve.', 150, S, 0],
        ],
        fail: true,
      },
      'Five six seven eight.': { sentences: [['Five six seven eight.', 260, S, 0]] },
      'Nine ten eleven twelve.': { sentences: [['Nine ten eleven twelve.', 150, S, 0]], fail: true },
    },
  },
  {
    name: 'no heard words keep one span',
    script: 'Alpha beta gamma delta.\n\nEpsilon zeta eta theta.',
    takes: {
      'Alpha beta gamma delta. Epsilon zeta eta theta.': {
        sentences: [
          ['Alpha beta gamma delta.', 150, S, 9600],
          ['Epsilon zeta eta theta.', 150, S, 0],
        ],
        deaf: true,
      },
    },
  },
];

const dir = mkdtempSync(join(tmpdir(), 'join-oracle-'));
const realError = console.error;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_join: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  const base = pathToFileURL(join(jsDir, '/')).href;
  text = text
    .replaceAll("from './", `from '${base}`)
    .replaceAll("import('./", `import('${base}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  text += '\nexport { voxcpm, wav };\n';
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text);
  const speak = await import(pathToFileURL(patched).href);

  const cases = [];
  for (const [n, c] of CASES.entries()) {
    const caseDir = join(dir, `case-${n}`);
    const cache = join(caseDir, 'cache');
    mkdirSync(cache, { recursive: true });
    const anchorKey = createHash('sha1').update(JSON.stringify([CHECKPOINT, VOICE, '', 'reference', SEED, ''])).digest('hex').slice(0, 20);
    writeFileSync(join(cache, `voice-${anchorKey}.wav`), speak.wav([render(RATE, ANCHOR_OPS)], RATE).buf);
    console.error = () => {};
    const judged = new Map();
    let shaped = null;
    let parts = null;
    let voiceMedian = null;
    let joined = null;
    globalThis.oracleCacheDir = cache;
    globalThis.oracleCheckpoint = CHECKPOINT;
    globalThis.oracleGenerate = (list, designing, { takeFile }) => {
      if (designing) throw new Error('take_join: the stub model does not design voices');
      for (const s of list) writeFileSync(takeFile(s), '');
    };
    globalThis.oracleJudge = async (s) => {
      const t = c.takes[s.text];
      if (!t) throw new Error(`take_join: case "${c.name}" has no take for "${s.text}"`);
      const k = Object.keys(c.takes).indexOf(s.text);
      const signal = take(t.sentences, 11 + k, t.deaf);
      const file = join(caseDir, `take-${k}.wav`);
      writeFileSync(file, speak.wav([render(signal.n, signal.ops)], RATE).buf);
      const ok = !t.fail;
      const verdict = { ok, badness: ok ? 0 : 1, why: ok ? '' : 'missing: twelve', heard: signal.heard.map((w) => w.text).join(' ') };
      judged.set(s.text, { text: s.text, n: signal.n, ops: signal.ops, words: signal.heard, ...verdict, doubt: t.doubt ?? [], pauses: t.pauses ?? [], clicks: t.clicks ?? [] });
      return { ...verdict, seed: s.seed, file, words: signal.heard, doubt: t.doubt ?? [], pauses: t.pauses ?? [], thumps: 0, fixed: 0, clicks: t.clicks ?? [] };
    };
    globalThis.oracleShaped = (list, units, median) => {
      parts = units.map((u) => u.text);
      voiceMedian = String(median);
      shaped = list.map((s) => ({ samples: fnv1a(s.audio), length: s.audio.length, spans: JSON.stringify(s.spans, null, 1) }));
    };
    globalThis.oracleJoined = (parts) => {
      const all = new Float32Array(parts.reduce((k, p) => k + p.length, 0));
      let o = 0;
      for (const p of parts) (all.set(p, o), (o += p.length));
      joined = fnv1a(all);
    };
    const out = join(caseDir, 'out.wav');
    let result;
    try {
      result = await speak.voxcpm(c.script, { device: 'cpu', voice: VOICE, seed: SEED }, out);
    } finally {
      console.error = realError;
    }
    const takes = [...judged.values()].map((t) => {
      const file = join(caseDir, `take-${Object.keys(c.takes).indexOf(t.text)}.wav`);
      const buf = readFileSync(file);
      const read = Float32Array.from({ length: (buf.length - 44) / 2 }, (_, k) => buf.readInt16LE(44 + 2 * k) / 32768);
      return { ...t, input: fnv1a(read) };
    });
    cases.push({
      name: c.name,
      script: c.script,
      voiceMedian,
      takes,
      parts,
      shaped,
      joined,
      file: fnv1aBytes(readFileSync(out)),
      duration: String(result.duration),
      phrases: JSON.stringify(result.phrases, null, 1),
      checks: JSON.stringify(result.checks, null, 1),
      problems: result.problems,
    });
  }
  process.stdout.write(`${JSON.stringify({ checkpoint: CHECKPOINT, voice: VOICE, seed: SEED, anchor: ANCHOR_OPS, cases }, null, 1)}\n`);
} finally {
  console.error = realError;
  for (const k of ['oracleCacheDir', 'oracleCheckpoint', 'oracleGenerate', 'oracleJudge', 'oracleShaped', 'oracleJoined']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
