// Reference outputs for a whole VoxCPM2 run in `voxr_narrate`: runs a patched copy of speak.mjs's `voxcpm`
// (unexported) end to end, with the model call, the judge and the recognizer of a reference recording stubbed. Prints
// one JSON document to stdout.
//
// Each scenario is a series of runs sharing one cache directory and one output, so a later run reuses, re-rolls or
// resets what an earlier one kept. The stub model writes each take as a 32-bit float WAV, which ffmpeg passes through
// unchanged. A designing call writes the runner's anchor.wav and a slower reference take its take file, each a tone
// of the length DESIGN names for the call (signals.mjs `takeSignal`). Every other take is looked up by its spoken
// text in the scenario's table: floor noise seeded from the take's seed under one tone per sentence, with the gap
// the table names after it. The stub judge reads the same table: the sentence words heard evenly over each tone,
// times rounded to hundredths, and a verdict that fails every take of a text (`fail: 'all'`) or only the take at the
// text's own seed (`fail: 'first'`). Math.random returns the run's `random`. Design, preparation, the take loop, the
// split, shapeTake, heat and the join run for real.
//
// Each run records every model call in order (the text as spoken, the prefix, the seed, the voice kind, the design
// description, the Hi-Fi transcript, and the FNV-1a hash of the samples a clone reads from its reference), every
// judge call, every stderr line, the voice's median pitch, the FNV-1a hash of the joined samples, the duration,
// phrases, checks and problems, the `.takes.json` text, and every file under the scenario directory with the
// FNV-1a hash of its bytes.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_speak.mjs > ../../fixtures/narration_speak_oracle.json
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { listing, wavF32, wavSamples } from './oracle_files.mjs';
import { fnv1a, hiss, noise, render, takeSignal, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_speak: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it. The first
// three fix machine state: the cache directory, the checkpoint and a reference's cache id. The next three stub the
// model call (after its notice prints), the judge and the recognizer of a reference recording. The last two hand the
// oracle the voice's median pitch and the joined parts.
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
    at: "  const refId = reference ? `${reference}:${statSync(reference).size}:${statSync(reference).mtimeMs}` : '';\n",
    replace:
      "  const refId = reference ? (globalThis.oracleRefId ?? `${reference}:${statSync(reference).size}:${statSync(reference).mtimeMs}`) : '';\n",
  },
  {
    at: "      console.error(`speak: VoxCPM2 is speaking ${list.length} part(s)${checkpoint ? '' : ' after the download'}, about 3 s per sentence on a GPU.`);\n",
    after: '      if (globalThis.oracleGenerate) return globalThis.oracleGenerate(list, designing, { takeFile, say, style, dir, anchorFile, cloneFrom, hifiText, voice });\n',
  },
  {
    at: '    const judge = async (s) => {\n',
    after: '      if (globalThis.oracleJudge) return globalThis.oracleJudge(s, { takeFile, say });\n',
  },
  {
    at: "      hifiText = (await transcribe(reference, { model: asrModel(language), language })).words.map((w) => w.text).join(' ');\n",
    replace:
      "      hifiText = (await (globalThis.oracleTranscribe ?? transcribe)(reference, { model: asrModel(language), language })).words.map((w) => w.text).join(' ');\n",
  },
  {
    at: '    const voiceMedian = medianPitch(samples(reference || anchorFile));\n',
    after: '    globalThis.oracleMedian?.(voiceMedian);\n',
  },
  {
    at: '    const r = wav(parts, 48000);\n',
    after: '    globalThis.oracleJoined?.(parts);\n',
  },
];

const RATE = 48000;
// Floor noise before the first tone of a take and after the last.
const LEAD = 4800;
// The run's own checkpoint name, so the cache keys fall back to nothing and no download notice prints.
const CHECKPOINT = 'openbmb/VoxCPM2';
// A reference recording's cache id: its path, size and modified time as the skill joins them.
const REF_ID = 'ref.wav:144044:1700000000000';
// The tone length of the design, then of each slower reference take.
const DESIGN = [96000, 168000, 240000];
const SLOWER = ['(calm, natural tone, speaking slowly)', '(calm, natural tone, speaking very slowly)'];
// A user's recording: speech, a breath, then speech.
const RECORDING = { n: 72000, ops: [noise(0, 72000, 0.00002, 3), tone(200, 0, 14400, 0.3, 480, 480), hiss(15360, 7200, 0.009, 43, 480, 480), tone(150, 23520, 48000, 0.3, 480, 480)] };
// The words the stub recognizer hears in the recording.
const REFERENCE_WORDS = ['Hello', 'there,', 'friend.'];

// The noise seed of a take made at `seed`.
const noiseSeed = (seed) => 1 + (seed % 997);
// A part's own text seed, as speak.mjs draws it.
const textSeed = (text, seed) => seed + (parseInt(createHash('sha1').update(text).digest('hex').slice(0, 6), 16) % 1000);

const S = 48000;
const SCENARIOS = [
  {
    name: 'a designed voice, rerun, re-rolled and reset',
    voice: '(young woman, warm, low)',
    script: '(calm tone) First line here now.\nSecond line goes here.\n\nThird line ends it.\n(softly) Go now.',
    takes: {
      'First line here now.': { sentences: [['First line here now.', 150, S, 0]] },
      'Second line goes here. Third line ends it.': {
        sentences: [
          ['Second line goes here.', 180, S, 4800],
          ['Third line ends it.', 260, S, 0],
        ],
        fail: 'first',
        why: 'missing: ends',
      },
      'Go now.': { sentences: [['Go now.', 150, 24000, 0]], thumps: 1 },
    },
    runs: [{ name: 'fresh' }, { name: 'rerun reuses every take' }, { name: 'part 2 re-rolled', reroll: [2], random: 0.25 }, { name: 'a new seed resets', seed: 9 }],
  },
  {
    name: 'a user reference',
    reference: true,
    script: 'The river runs {N|north} today.\n\nThe herds come back to graze.',
    takes: {
      'The river runs north today. The herds come back to graze.': {
        sentences: [
          ['The river runs north today.', 150, S, 9600],
          ['The herds come back to graze.', 260, S, 0],
        ],
      },
    },
    runs: [{ name: 'fresh' }, { name: 'rerun reuses every take' }],
  },
  {
    name: 'Hi-Fi with a user reference and a style',
    reference: true,
    hifi: true,
    style: '(relaxed, explaining)',
    script: 'Wait for it now. [pause 1.2] Here it comes at last.',
    takes: {
      'Wait for it now. Here it comes at last.': {
        sentences: [
          ['Wait for it now.', 150, S, 14400],
          ['Here it comes at last.', 180, S, 0],
        ],
      },
    },
    runs: [{ name: 'fresh' }],
  },
  {
    name: 'Hi-Fi with a designed voice and a style',
    voice: '(older man, deep, calm)',
    hifi: true,
    style: '(relaxed, explaining)',
    script: 'Where does the river go?\n\nIt runs north to the sea.',
    takes: {
      'Where does the river go? It runs north to the sea.': {
        sentences: [
          ['Where does the river go?', 300, S, 9600],
          ['It runs north to the sea.', 180, S, 0],
        ],
      },
    },
    runs: [{ name: 'fresh' }, { name: 'rerun reads the saved transcript' }],
  },
  {
    name: 'a part that fails and splits, with no voice given',
    script: 'One two three four.\n\nFive six seven eight. Nine ten eleven twelve.',
    takes: {
      'One two three four. Five six seven eight. Nine ten eleven twelve.': {
        sentences: [
          ['One two three four.', 150, S, 9600],
          ['Five six seven eight.', 150, S, 4800],
          ['Nine ten eleven twelve.', 150, S, 0],
        ],
        fail: 'all',
        why: 'missing: twelve',
      },
      'One two three four.': { sentences: [['One two three four.', 150, S, 0]] },
      'Five six seven eight. Nine ten eleven twelve.': {
        sentences: [
          ['Five six seven eight.', 150, S, 4800],
          ['Nine ten eleven twelve.', 150, S, 0],
        ],
        fail: 'all',
        why: 'missing: twelve',
      },
      'Five six seven eight.': { sentences: [['Five six seven eight.', 260, S, 0]] },
      'Nine ten eleven twelve.': { sentences: [['Nine ten eleven twelve.', 150, S, 0]], fail: 'all', why: 'missing: twelve' },
    },
    runs: [{ name: 'fresh' }],
  },
];

// A take of `sentences`, each [text, hz, tone length, gap after] in samples: its length, its tone ops, and the words
// heard in it.
function take(sentences) {
  const tones = [];
  const words = [];
  let at = LEAD;
  for (const [text, hz, len, gap] of sentences) {
    tones.push(tone(hz, at, len, 0.3, 480, 480));
    const said = text.split(/\s+/);
    const step = len / said.length;
    said.forEach((w, k) => words.push({ text: w, start: Math.round(((at + k * step) / RATE) * 100) / 100, end: Math.round(((at + (k + 1) * step) / RATE - 0.02) * 100) / 100 }));
    at += len + gap;
  }
  return { n: at + LEAD, tones, words };
}

const dir = mkdtempSync(join(tmpdir(), 'speak-oracle-'));
const realError = console.error;
const realRandom = Math.random;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_speak: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
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

  const scenarios = [];
  for (const [n, sc] of SCENARIOS.entries()) {
    const root = join(dir, `scenario-${n}`);
    const cache = join(root, 'cache');
    mkdirSync(cache, { recursive: true });
    const out = join(root, 'out.wav');
    const takes = Object.fromEntries(Object.entries(sc.takes).map(([said, t]) => [said, { ...take(t.sentences), fail: t.fail ?? null, why: t.why ?? '', thumps: t.thumps ?? 0 }]));
    let reference = '';
    if (sc.reference) {
      reference = join(root, 'ref.wav');
      writeFileSync(reference, speak.wav([render(RECORDING.n, RECORDING.ops)], RATE).buf);
    }
    const runs = [];
    for (const run of sc.runs) {
      const messages = [];
      const calls = [];
      const judged = [];
      let voiceMedian = null;
      let joined = null;
      let runSeed = null;
      globalThis.oracleCacheDir = cache;
      globalThis.oracleCheckpoint = CHECKPOINT;
      globalThis.oracleRefId = REF_ID;
      globalThis.oracleGenerate = (list, designing, ctx) => {
        for (const [done, s] of list.entries()) {
          const from = designing ? null : ctx.cloneFrom ?? ctx.anchorFile;
          calls.push({
            text: ctx.say(s.text),
            prefix: designing ? null : s.style ?? (ctx.style || null),
            seed: s.seed,
            voice: designing ? 'own' : 'samples',
            design: designing && ctx.voice ? ctx.voice : null,
            transcript: !designing && ctx.hifiText ? ctx.hifiText : null,
            reference: from ? fnv1a(wavSamples(readFileSync(from))) : null,
          });
          const slower = SLOWER.indexOf(s.style) + 1;
          let x;
          if (designing || slower > 0) {
            const k = designing ? 0 : slower;
            x = takeSignal(DESIGN[k], false, 101 + k);
          } else {
            const t = takes[ctx.say(s.text)];
            if (!t) throw new Error(`take_speak: scenario "${sc.name}" has no take for "${ctx.say(s.text)}"`);
            x = render(t.n, [noise(0, t.n, 0.00002, noiseSeed(s.seed)), ...t.tones]);
          }
          writeFileSync(designing ? join(ctx.dir, 'anchor.wav') : ctx.takeFile(s), wavF32(x, RATE));
          // The runner's progress line for each take, as generate relays it from the runner's stderr.
          console.error(`  voxcpm: ${done + 1}/${list.length} sentences`);
        }
      };
      globalThis.oracleJudge = async (s, ctx) => {
        const said = ctx.say(s.text);
        const t = takes[said];
        if (!t) throw new Error(`take_speak: scenario "${sc.name}" has no take for "${said}"`);
        judged.push({ text: s.text, seed: s.seed });
        const ok = !(t.fail === 'all' || (t.fail === 'first' && s.seed === textSeed(s.text, runSeed)));
        const heard = t.words.map((w) => w.text).join(' ');
        return { ok, badness: ok ? 0 : 1, why: ok ? '' : t.why, heard, seed: s.seed, file: ctx.takeFile(s), words: t.words, doubt: [], pauses: [], thumps: t.thumps, fixed: 0, clicks: [] };
      };
      globalThis.oracleTranscribe = async () => ({ words: REFERENCE_WORDS.map((w) => ({ text: w })) });
      globalThis.oracleMedian = (median) => {
        voiceMedian = String(median);
      };
      globalThis.oracleJoined = (parts) => {
        const all = new Float32Array(parts.reduce((k, p) => k + p.length, 0));
        let o = 0;
        for (const p of parts) (all.set(p, o), (o += p.length));
        joined = fnv1a(all);
      };
      // The run's seed, as voxcpm resolves it, for the judge's `first` verdicts.
      let saved = {};
      try {
        saved = JSON.parse(readFileSync(join(root, 'out.takes.json'), 'utf8'));
      } catch {}
      runSeed = run.seed ?? saved.seed ?? 7;
      console.error = (line) => messages.push(`${line}`);
      Math.random = () => {
        if (run.random == null) throw new Error(`take_speak: run "${run.name}" draws no random number`);
        return run.random;
      };
      let result;
      try {
        result = await speak.voxcpm(sc.script, { device: 'cpu', voice: sc.voice ?? '', reference, style: sc.style ?? '', hifi: sc.hifi ?? false, seed: run.seed, reroll: run.reroll ?? [] }, out);
      } finally {
        console.error = realError;
        Math.random = realRandom;
      }
      runs.push({
        name: run.name,
        seed: run.seed ?? null,
        reroll: run.reroll ?? [],
        random: run.random ?? null,
        calls,
        judged,
        messages,
        voiceMedian,
        joined,
        duration: String(result.duration),
        phrases: JSON.stringify(result.phrases, null, 1),
        checks: JSON.stringify(result.checks, null, 1),
        problems: result.problems,
        voiceFile: result.voiceFile != null,
        takesJson: readFileSync(join(root, 'out.takes.json'), 'utf8'),
        files: listing(root),
      });
    }
    scenarios.push({
      name: sc.name,
      script: sc.script,
      voice: sc.voice ?? '',
      style: sc.style ?? '',
      hifi: sc.hifi ?? false,
      reference: sc.reference ?? false,
      takes: Object.entries(takes).map(([said, t]) => ({ said, ...t })),
      runs,
    });
  }
  process.stdout.write(`${JSON.stringify({ checkpoint: CHECKPOINT, refId: REF_ID, design: DESIGN, recording: RECORDING, referenceWords: REFERENCE_WORDS, scenarios }, null, 1)}\n`);
} finally {
  console.error = realError;
  Math.random = realRandom;
  for (const k of ['oracleCacheDir', 'oracleCheckpoint', 'oracleRefId', 'oracleGenerate', 'oracleJudge', 'oracleTranscribe', 'oracleMedian', 'oracleJoined']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
