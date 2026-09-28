// Reference outputs for the take-loop's cache keys, seeds and take files in `voxr_narrate`: runs a patched copy
// of speak.mjs's `voxcpm` (unexported) far enough to build its cache-key closures (hash, anchorKey, anchorFile,
// cloneFrom, prepared, takeFile, textSeed) and its takes.json read state (current, before, chosen), captures them,
// then stops it short of running VoxCPM2 or ffmpeg. Prints one JSON document to stdout.
//
// The patch also lets the oracle fix the cache directory, the VoxCPM2 checkpoint and a reference recording's cache
// id (refId), each of which otherwise reads real machine state (the XDG cache root, a downloaded checkpoint, a real
// file's stat), so every case here is reproducible off this machine.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_files.mjs > ../../fixtures/narration_take_oracle.json
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_files: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it. The first
// three let the oracle override state voxcpm otherwise reads from the real machine; the next two gate the two
// branches that run VoxCPM2 or ffmpeg; the last captures every cache-key closure once textSeed exists (the
// last one voxcpm builds before it speaks) and stops the call there.
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
    at: '  const refId = reference ? `${reference}:${statSync(reference).size}:${statSync(reference).mtimeMs}` : \'\';\n',
    replace: "  const refId = reference ? (globalThis.oracleRefId ?? `${reference}:${statSync(reference).size}:${statSync(reference).mtimeMs}`) : '';\n",
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
    after:
      '    if (globalThis.oracleCapture) { globalThis.oracleCapture({ current: !!current, before: before ?? null, chosen, hash, anchorKey, anchorFile, cloneFrom, prepared, takeFile, textSeed, refId, seed }); throw new Error(\'__oracle_stop__\'); }\n',
  },
];

const dir = mkdtempSync(join(tmpdir(), 'take-oracle-'));
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_files: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  text = text
    .replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  text += '\nexport { voxcpm };\n';
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text);
  const speak = await import(pathToFileURL(patched).href);

  // Runs voxcpm just far enough to capture its cache-key closures, always with `device: 'cpu'` (skips the GPU
  // check) and `oracleSkipHeavy` (skips the branches that run VoxCPM2 or ffmpeg). Returns the captured object.
  const capture = async (opts, out = join(dir, 'out.wav')) => {
    globalThis.oracleCacheDir = opts.cacheDir ?? join(dir, 'cache');
    globalThis.oracleCheckpoint = 'checkpoint' in opts ? opts.checkpoint : null;
    globalThis.oracleRefId = opts.refId;
    globalThis.oracleSkipHeavy = true;
    let captured = null;
    globalThis.oracleCapture = (c) => {
      captured = c;
    };
    try {
      await speak.voxcpm('Hello there.', { device: 'cpu', ...opts.voxcpm }, out);
      throw new Error('take_files: voxcpm ran to completion instead of stopping at the oracle capture');
    } catch (e) {
      if (e.message !== '__oracle_stop__') throw e;
    } finally {
      globalThis.oracleCapture = undefined;
    }
    return captured;
  };
  const base = (name) => (name ? `${name.split('/').pop()}` : name);
  const record = join(dir, 'record.takes.json');

  const cases = [];

  // hash(): JSON.stringify's string escaping (quotes, backslashes, control characters, unicode) and JavaScript's
  // number formatting, over a mix of parts. checkpoint, voice and refId are each resolved once per voxcpm call, so
  // every combination needs its own capture.
  const hashCases = [
    { checkpoint: null, voice: 'af_heart', refId: '', parts: ['reference', 7, ''] },
    { checkpoint: null, voice: 'af_heart', refId: '', parts: ['line one\nline "two"\\three', 1.6, 0.5] },
    { checkpoint: 'openbmb/VoxCPM2', voice: 'suara Melayu 迷路 🎙️', refId: 'ref:123:456.75', parts: ['hifi', 16, 10, null] },
    { checkpoint: null, voice: 'v', refId: '', parts: [900719925474099, -0, 2, 10] },
  ];
  for (const { checkpoint, voice, refId, parts } of hashCases) {
    const cap = await capture({ voxcpm: { voice, seed: 7, reference: refId ? 'ref-placeholder' : '' }, checkpoint, refId });
    const key = cap.hash(...parts);
    cases.push({ case: 'hash', checkpoint, voice, refId, parts, key });
  }

  // textSeed(): the seed a part's own text draws, over unicode and quote-escaping texts and a range of bases.
  // textSeed closes over the run's own seed, so each base needs its own capture.
  const textSeedCases = [
    { text: 'Hello there.', base: 7 },
    { text: 'Ada "kutipan" dan sengkang-lain.', base: 7 },
    { text: 'Perjalanan ke Jepun 🗾 dan 迷路 di sana.', base: 900000 },
    { text: '', base: 0 },
  ];
  for (const { text, base } of textSeedCases) {
    const cap = await capture({ voxcpm: { voice: 'af_heart', seed: base } });
    cases.push({ case: 'textSeed', text, base, seed: cap.textSeed(text) });
  }

  // Anchor and take files: no reference (a designed voice), across languages and seeds.
  for (const { voice, seed, language } of [
    { voice: 'af_heart', seed: 7, language: undefined },
    { voice: 'suara Melayu', seed: 900719925474099, language: 'ms' },
  ]) {
    const cap = await capture({ voxcpm: { voice, seed, reference: '', language } });
    cases.push({
      case: 'anchor',
      checkpoint: null,
      voice,
      seed,
      language: language ?? null,
      anchorKey: cap.anchorKey,
      anchorFile: base(cap.anchorFile),
    });
  }

  // Reference-cloning files: a reference recording's cloneFrom and prepared marker (the sha1 of a file's bytes, in
  // full, unlike hash's 20-character slice), and the takeFile it feeds into.
  const refId = '/rec with spaces.wav:12345:1699999999999.25';
  const refCap = await capture({ voxcpm: { voice: 'af_heart', reference: 'ref-placeholder', seed: 7 }, refId });
  const contentFile = join(dir, 'prepared-content.bin');
  const content = 'a small fixed reference, to hash deterministically';
  writeFileSync(contentFile, content, 'utf8');
  cases.push({
    case: 'reference',
    checkpoint: null,
    voice: 'af_heart',
    refId,
    anchorKey: refCap.anchorKey,
    cloneFrom: base(refCap.cloneFrom),
    content,
    prepared: base(refCap.prepared(contentFile)),
  });

  // Per-take keys: the legacy cfg/steps omission, hifi, and a reference-cloned voice's null anchor.
  const takeCases = [
    { anchor: 'no-reference', voice: 'af_heart', seed: 7, style: '', text: 'Hello there.', cfg: 2, steps: 10, hifi: false },
    { anchor: 'no-reference', voice: 'af_heart', seed: 7, style: '', text: 'Hello there.', cfg: 1.6, steps: 16, hifi: false },
    { anchor: 'no-reference', voice: 'af_heart', seed: 7, style: '(calm tone)', text: 'Hello there.', cfg: 2, steps: 10, hifi: true },
    { anchor: 'reference', voice: 'af_heart', seed: 7, style: '', text: 'Hello there.', cfg: 2, steps: 10, hifi: false },
  ];
  for (const tc of takeCases) {
    const cap =
      tc.anchor === 'reference'
        ? await capture({ voxcpm: { voice: tc.voice, reference: 'ref-placeholder', seed: tc.seed, cfg: tc.cfg, steps: tc.steps, hifi: tc.hifi }, refId })
        : await capture({ voxcpm: { voice: tc.voice, reference: '', seed: tc.seed, cfg: tc.cfg, steps: tc.steps, hifi: tc.hifi } });
    cases.push({
      case: 'take',
      ...tc,
      anchorKey: cap.anchorKey,
      takeFile: base(cap.takeFile({ style: tc.style || undefined, text: tc.text, seed: tc.seed })),
    });
  }

  // takes.json reads: the current format, an old flat file, a corrupt file, and the reset a new asked seed causes.
  const takesReadCases = [
    { name: 'current format', raw: '{"seed":9,"takes":{"Hello there.":812,"unicode 迷路":3}}', asked: null, configSeed: null },
    { name: 'old flat file', raw: '{"Hello there.":812}', asked: null, configSeed: 5 },
    { name: 'corrupt json', raw: '{not json', asked: null, configSeed: null },
    { name: 'takes not an object', raw: '{"seed":9,"takes":"nope"}', asked: null, configSeed: null },
    { name: 'asked seed matches before, keeps chosen', raw: '{"seed":9,"takes":{"a":1}}', asked: 9, configSeed: null },
    { name: 'asked seed resets chosen', raw: '{"seed":9,"takes":{"a":1}}', asked: 99, configSeed: null },
    { name: 'no file, config seed', raw: null, asked: null, configSeed: 42 },
    { name: 'no file, no config seed', raw: null, asked: null, configSeed: null },
  ];
  for (const tc of takesReadCases) {
    if (tc.raw === null) rmSync(record, { force: true });
    else writeFileSync(record, tc.raw);
    const cap = await capture({ voxcpm: { voice: 'af_heart', reference: '', seed: tc.asked ?? undefined, configSeed: tc.configSeed ?? undefined } }, record.replace(/\.takes\.json$/, '.wav'));
    cases.push({ case: 'takesRead', name: tc.name, raw: tc.raw, asked: tc.asked, configSeed: tc.configSeed, current: cap.current, before: cap.before, seed: cap.seed, chosen: cap.chosen });
  }

  process.stdout.write(`${JSON.stringify({ cases }, null, 1)}\n`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
