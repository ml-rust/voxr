// Reference outputs for the VoxCPM2 anchor in `voxr_narrate`: runs a patched copy of speak.mjs's `voxcpm`
// (unexported) through the voice it clones from, a designed voice or a user's recording, with the model call
// stubbed, and stops right after the designed voice is copied beside the output. Prints one JSON document to stdout.
//
// The stub model writes a synthetic take per call as a 32-bit float WAV, which ffmpeg passes through unchanged. The
// take is built from signals.mjs: floor noise, then a 200 Hz tone of the length the case names for the call's seed
// offset from the run seed, split by a breath when the case asks for one. The designing call writes the runner's
// anchor.wav, and every other call writes its take file, as the runner's copy does. tidy, repairClicks, pace,
// quietBreaths and prepareReference run for real.
//
// Each case records every model call in order (the text as spoken, the prefix, the seed, the voice kind, the design
// description, the Hi-Fi transcript, and the FNV-1a hash of the samples a clone reads from its reference), each pace
// the design measures, every stderr line, the Hi-Fi transcript the run ends with, and each file under the case
// directory with the FNV-1a hash of its bytes.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_anchor.mjs > ../../fixtures/narration_anchor_oracle.json
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { listing, wavF32, wavSamples } from './oracle_files.mjs';
import { fnv1a, hiss, noise, render, takeSignal, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_anchor: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it. The first
// three fix machine state: the cache directory, the checkpoint and a reference's cache id. The next stubs the model
// call. The next records each pace the design measures. The last stops the run once the voice sits beside the
// output.
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
    at: '    const generate = (list, designing = false) => {\n',
    after: '      if (globalThis.oracleGenerate) return globalThis.oracleGenerate(list, designing, { takeFile, say, style, dir, anchorFile, cloneFrom, hifiText, voice });\n',
  },
  {
    at: '        const was = refPace(anchorFile);\n',
    after: '        globalThis.oraclePace?.(was);\n',
  },
  {
    at: '    if (voiceFile) copyFileSync(anchorFile, voiceFile);\n',
    after: "    if (globalThis.oracleDone) { globalThis.oracleDone({ hifiText }); throw new Error('__oracle_stop__'); }\n",
  },
];

const RATE = 48000;
// The run's own checkpoint name, so the cache keys fall back to nothing and no download notice prints.
const CHECKPOINT = 'openbmb/VoxCPM2';
// A reference recording's cache id: its path, size and modified time as the skill joins them.
const REF_ID = 'ref.wav:144044:1700000000000';

// A designed voice already in the cache: half a second of tone.
const HELD_ANCHOR = () => render(24000, [noise(0, 24000, 0.00002, 5), tone(150, 0, 24000, 0.3, 480, 480)]);
// A user's recording: speech, a breath, then speech.
const RECORDING = () =>
  render(72000, [noise(0, 72000, 0.00002, 3), tone(200, 0, 14400, 0.3, 480, 480), hiss(15360, 7200, 0.009, 43, 480, 480), tone(150, 23520, 48000, 0.3, 480, 480)]);

const SHORT = 'Hello there, friend. The river runs north today.';
const LONG =
  'The river runs north today, past the old mill and the quiet farms. Each spring the water rises over the low meadow and the reeds. By summer the ground dries out again. Then the herds come back to graze.';

// The cases. `takes` gives, for each seed offset from the run seed, the tone length in samples and whether a breath
// splits it. `held` names files placed in the cache before the run: the designed voice, its transcript, a prepared
// reference, or a prepared marker for the recording.
const CASES = [
  {
    name: 'fresh design, the first slow retry is slow enough',
    script: SHORT,
    voice: '(young woman, warm, low)',
    takes: [
      [96000, false],
      [168000, true],
    ],
  },
  {
    name: 'fresh design on a long script, retried twice',
    script: LONG,
    voice: '(older man, deep, calm)',
    seed: 11,
    style: '(relaxed, explaining)',
    takes: [
      [240000, false],
      [264000, false],
      [360000, false],
    ],
  },
  {
    name: 'fresh design with hifi',
    script: SHORT,
    voice: '(young woman, warm, low)',
    hifi: true,
    takes: [
      [96000, true],
      [168000, false],
    ],
  },
  {
    name: 'fresh design with a transcript left from an older design',
    script: SHORT,
    voice: '(young woman, warm, low)',
    hifi: true,
    style: '(relaxed, explaining)',
    held: { transcript: 'An older opening.\n' },
    takes: [
      [96000, false],
      [168000, false],
    ],
  },
  { name: 'cached anchor reused', script: SHORT, voice: '(young woman, warm, low)', held: { anchor: true } },
  {
    name: 'cached anchor reused with its transcript, hifi and a style',
    script: SHORT,
    voice: '(young woman, warm, low)',
    hifi: true,
    style: '(relaxed, explaining)',
    held: { anchor: true, transcript: '  Hello there, friend. The river runs north today.\n' },
  },
  { name: 'cached anchor with no transcript, hifi', script: SHORT, voice: '(young woman, warm, low)', hifi: true, held: { anchor: true } },
  { name: 'user reference prepared fresh', script: SHORT, reference: true },
  { name: 'user reference already prepared', script: SHORT, reference: true, held: { marker: true } },
  { name: 'user reference with its prepared copy held', script: SHORT, reference: true, held: { prepared: true } },
];

const dir = mkdtempSync(join(tmpdir(), 'anchor-oracle-'));
const realError = console.error;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_anchor: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  text = text
    .replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`)
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
    const seed = c.seed ?? 7;
    const voice = c.voice ?? '';
    const anchorKey = createHash('sha1').update(JSON.stringify([CHECKPOINT, voice, '', 'reference', seed, ''])).digest('hex').slice(0, 20);
    const held = c.held ?? {};
    if (held.anchor) writeFileSync(join(cache, `voice-${anchorKey}.wav`), speak.wav([HELD_ANCHOR()], RATE).buf);
    if (held.transcript != null) writeFileSync(join(cache, `voice-${anchorKey}.txt`), held.transcript);
    let reference = '';
    if (c.reference) {
      reference = join(caseDir, 'ref.wav');
      const bytes = speak.wav([RECORDING()], RATE).buf;
      writeFileSync(reference, bytes);
      if (held.marker) {
        mkdirSync(join(cache, 'prepared'), { recursive: true });
        writeFileSync(join(cache, 'prepared', createHash('sha1').update(bytes).digest('hex')), '');
      }
      if (held.prepared) {
        const key = createHash('sha1').update(JSON.stringify([CHECKPOINT, voice, REF_ID, 'breaths', 'bass', 'compressed', 'normalized'])).digest('hex').slice(0, 20);
        writeFileSync(join(cache, `ref-${key}.wav`), speak.wav([HELD_ANCHOR()], RATE).buf);
      }
    }
    const messages = [];
    console.error = (line) => messages.push(`${line}`);
    const calls = [];
    const paces = [];
    let hifiText = null;
    globalThis.oracleCacheDir = cache;
    globalThis.oracleCheckpoint = CHECKPOINT;
    globalThis.oracleRefId = REF_ID;
    globalThis.oraclePace = (was) => paces.push(String(was));
    globalThis.oracleGenerate = (list, designing, ctx) => {
      for (const s of list) {
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
        const k = s.seed - seed;
        if (!c.takes?.[k]) throw new Error(`take_anchor: case "${c.name}" has no take for seed offset ${k}`);
        const [len, breath] = c.takes[k];
        const buf = wavF32(takeSignal(len, breath, 101 + k), RATE);
        writeFileSync(designing ? join(ctx.dir, 'anchor.wav') : ctx.takeFile(s), buf);
      }
    };
    globalThis.oracleDone = (state) => {
      hifiText = state.hifiText ?? null;
    };
    try {
      await speak.voxcpm(c.script, { device: 'cpu', voice, reference, style: c.style ?? '', hifi: c.hifi ?? false, seed: c.seed }, join(caseDir, 'out.wav'));
      throw new Error(`take_anchor: case "${c.name}" ran to completion instead of stopping at the oracle`);
    } catch (e) {
      if (e.message !== '__oracle_stop__') throw e;
    } finally {
      console.error = realError;
    }
    cases.push({
      name: c.name,
      script: c.script,
      voice: c.voice ?? null,
      seed,
      style: c.style ?? null,
      hifi: c.hifi ?? false,
      reference: c.reference ?? false,
      held: { anchor: held.anchor ?? false, transcript: held.transcript ?? null, marker: held.marker ?? false, prepared: held.prepared ?? false },
      takes: (c.takes ?? []).map(([len, breath]) => ({ len, breath })),
      anchorKey,
      calls,
      paces,
      messages,
      hifiText,
      files: listing(caseDir),
    });
  }
  process.stdout.write(`${JSON.stringify({ checkpoint: CHECKPOINT, refId: REF_ID, cases }, null, 1)}\n`);
} finally {
  console.error = realError;
  for (const k of ['oracleCacheDir', 'oracleCheckpoint', 'oracleRefId', 'oraclePace', 'oracleGenerate', 'oracleDone']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
