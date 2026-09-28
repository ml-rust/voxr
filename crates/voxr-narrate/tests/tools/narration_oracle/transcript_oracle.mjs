// Reference outputs for the transcript post-processing in `voxr_narrate`: runs the JavaScript transcribe pipeline,
// collapseLoops, snapOnsets, snapEnds, fitToSound, evenOut, lastRun, fitFile, alignPhrases, findPhrase, wordsFile,
// narrations and spokenWords over the cases below, and prints one JSON document to stdout. Each case carries its
// inputs: word lists as numbers, signals as the signals.mjs ops that build them at 16 kHz, and each signal's FNV-1a
// hash. Output times are written as JavaScript's String(x), which reads back as the exact double, and each file the
// transcribe command writes as its exact text.
//
// envelope, pieces, snapOnsets, snapEnds, evenOut and lastRun are not exported, and transcribe decodes a file with
// ffmpeg and reads it with a Whisper model. The oracle imports a copy of voice.mjs that exports them, whose decode
// returns the generated samples for a file name, and whose runtime hands transcribe a recognizer that returns the
// case's scripted words for each piece it is given, recording each call. Each patch is inserted at a line that must
// occur exactly once; nothing else changes. The words files transcribe writes are composed as the transcribe command
// composes them.
//
// Usage: NARRATION_JS_DIR=<directory holding voice.mjs> node transcript_oracle.mjs > ../../fixtures/narration_transcript_oracle.json
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fnv1a, noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('transcript_oracle: NARRATION_JS_DIR is not set. Set it to the directory that holds voice.mjs.');
  process.exit(2);
}

const PATCHES = [
  { at: 'function decode(file) {\n', after: '  if (globalThis.oraclePcm?.has(file)) return globalThis.oraclePcm.get(file);\n' },
  { at: 'const runtime = () => speechModule(PKG);\n', replace: 'const runtime = async () => ({ pipeline: globalThis.oraclePipeline });\n' },
];

const S = 16000;
const MODEL = 'onnx-community/whisper-base_timestamped';
// Speech from `a` to `b` seconds: a steady tone (the 48 kHz 200 Hz recurrence, 600 Hz at 16 kHz) with 10 ms ramps.
const speech = (a, b) => tone(200, Math.round(a * S), Math.round((b - a) * S), 0.3, 160, 160);
const w = (text, start, end) => ({ text, start, end });
// Words `step` seconds apart from `from`, each lasting `len`.
const spaced = (texts, from, step, len) => texts.map((t, i) => w(t, from + i * step, from + i * step + len));

// A 45 s take: 1.6 s of speech between pauses of 0.3 s, two of them 0.6 s.
const LONG = [];
for (let t = 0.2, i = 0; t + 1.6 < 44.8; i++) {
  LONG.push(speech(t, t + 1.6));
  t += 1.6 + (i === 7 || i === 16 ? 0.6 : 0.3);
}

// Six seconds: speech 0.5-1.2, 1.3-2.0, a pause, 3.0-4.2, 4.3-5.5.
const GAPPED = [speech(0.5, 1.2), speech(1.3, 2.0), speech(3.0, 4.2), speech(4.3, 5.5)];
// Six seconds: two phrases, the second with speech after a pause at its end.
const PHRASED = [speech(0.5, 1.4), speech(1.5, 2.4), speech(3.0, 3.8), speech(4.6, 5.4)];
const PHRASES = [
  { text: 'The river runs north.', start: 0.4, end: 2.6 },
  { text: 'Statusnya, terancam.', start: 2.8, end: 5.8 },
];

// `script` holds each piece's words as the recognizer returns them, times from the piece's start (null: no time).
// `compose` runs the transcribe command's script step: phrase spans when given, else the plain script.
const TRANSCRIBE = [
  {
    name: 'one piece: late onset and loose ends',
    n: 6 * S,
    ops: GAPPED,
    script: [[['Rivers', 0.52, 1.18], [' carry', 1.32, 2.9], ['water', 3.25, 3.8], ['that', 3.85, 4.2], ['runs', 4.35, 4.9], [' downhill.', 4.95, null]]],
    compose: { script: 'Rivers carry the water that runs downhill.', scriptFile: 'narration/one.txt' },
  },
  {
    name: '45 s take cut at silences',
    n: 45 * S,
    ops: LONG,
    language: 'ms',
    script: [
      [['Semuanya', 0.3, 0.9], ['bermula', 1.0, 1.6], ['di', 2.3, 2.6], ['sini', 2.7, 3.4], ['tahun', 8.3, 8.9]],
      [['Raja', 0.4, 0.8], ['Muda', 0.9, 1.5], ['datang', 4.2, 4.9]],
      [['dan', 0.2, 0.5], ['pergi.', 1.2, 1.9]],
    ],
  },
  {
    name: 'no silence: hard cut at 20 s',
    n: 25 * S,
    ops: [tone(200, 0, 25 * S, 0.3, 160, 160)],
    script: [[['one', 1.0, 1.5], ['two', 19.5, 20.0]], [['three', 0.2, 0.8], ['four', 3.0, null]]],
  },
  {
    name: 'words in digital silence',
    n: 4 * S,
    ops: [speech(0.5, 2.0), noise(2.5 * S, 1.5 * S, 0.00001, 9)],
    script: [[['hello', 0.5, 1.0], ['world', 1.1, 1.9], ['Thanks', 2.1, 2.4], ['you', 3.0, 3.3], ['', 3.4, 3.5], ['  ', 3.5, 3.6]]],
  },
  {
    name: 'loops: three copies collapse, two stay',
    n: 8 * S,
    ops: [speech(0.1, 7.9)],
    script: [
      spaced(
        ['dengan', 'kembali', 'dengan', 'kembali', 'Dengan,', 'kembali', 'ke', 'rumah', 'rumah', 'ya', 'ya', 'ya.', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'A', 'b', 'c', 'd.', 'selesai'],
        0.2,
        0.25,
        0.2,
      ).map(({ text, start, end }) => [text, start, end]),
    ],
  },
  {
    name: 'crushed pair split by raw length',
    n: 3 * S,
    ops: [speech(1.0, 2.5)],
    script: [[['Hi,', 1.0, 1.02], ['there.', 1.03, 1.6], ['friend', 1.7, 2.4]]],
  },
  {
    name: 'crushed run at the end of the audio',
    n: 3 * S,
    ops: [speech(0.5, 3.0)],
    script: [[['Absolutely', 0.5, 2.95], ['fine', 2.95, 2.96], ['today.', 2.96, 2.97]]],
  },
  {
    name: 'phrase spans with trailing speech',
    n: 6 * S,
    ops: PHRASED,
    script: [[['The', 0.5, 0.7], ['river', 0.72, 1.1], ['runs', 1.15, 1.4], ['north.', 1.5, 2.3], ['Statusnya', 3.0, 3.5], ['terancam.', 3.5, 3.8]]],
    compose: { phrases: PHRASES, phrasesFile: 'narration/take.srt', scriptFile: 'narration/take.srt' },
  },
];

const SIGNALS = [
  {
    name: 'gapped',
    n: 6 * S,
    ops: GAPPED,
    snapOnsets: [
      [w('late', 3.25, 3.8)],
      [w('first', 0.8, 1.1)],
      [w('close', 0.5, 1.15), w('behind', 1.3, 1.9)],
      [w('far', 4.0, 4.2)],
      [w('prev', 2.9, 3.2), w('bounded', 3.6, 4.1)],
    ],
    snapEnds: [
      [w('loose', 3.0, 4.9), w('next', 5.0, 5.4)],
      [w('Hi,', 3.1, 3.12), w('there.', 3.13, 3.9)],
      [w('a', 3.1, 3.12), w('far', 3.6, 4.1)],
      [w('silent', 2.3, 2.6), w('then', 3.0, 3.5)],
      [w('last', 4.3, 5.95)],
    ],
    fitToSound: [
      [w('early', 2.2, 2.5), w('next', 4.3, 5.0)],
      [w('hello', 0.5, 1.0), w('there', 1.4, 2.0)],
      [w('trimmed', 3.0, 4.28), w('end', 4.4, 5.9)],
      [w('b', 1.0, 1.2), w('a', 0.9, 0.95), w('c', 1.25, 1.1)],
      [w('swallowed', 3.0, 4.1), w('it', 4.1, 4.11), w('all', 4.11, 4.12), w('up', 4.3, 5.0)],
    ],
    lastRun: [[2.0, 6.0], [0.0, 2.9], [2.05, 2.95], [5.45, 6.0], [4.25, 4.35]],
    fitFile: [
      { words: [w('Rivers', 0.52, 1.18), w('carry', 1.32, 2.0), w('water', 2.2, 3.8)] },
      {
        words: [w('Rivers', 0.3, 1.18), w('carry', 1.32, 2.0), w('stray', 2.1, 2.2), w('water', 3.0, 3.8), w('flows', 3.9, 4.0)],
        phrase: [0, 0, 2, 2, 2],
        spans: [{ text: 'Rivers carry', start: 0.4, end: 2.2 }, { text: 'unheard', start: 2.2, end: 2.4 }, { text: 'stray water flows', start: 2.4, end: 6.0 }],
      },
    ],
  },
];

const EVEN_OUT = [
  { words: [w('swallowed', 0.0, 2.0), w('it', 2.0, 2.01), w('all', 2.01, 2.02), w('next', 3.0, 3.4)], limit: null },
  { words: [w('long', 0.0, 0.5), w('a', 0.95, 0.96), w('b', 0.96, 0.97)], limit: 1.0 },
  { words: [w('x', 1.0, 1.01), w('neighbour', 1.05, 1.9), w('after', 2.5, 2.9)], limit: null },
  { words: spaced(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], 1.0, 0.001, 0.0005), limit: 1.2 },
  { words: [w('ok', 0.0, 0.3), w('fine', 0.3, 0.9)], limit: 1.0 },
];

const COLLAPSE = [
  ['x', 'x', 'x', 'y'],
  ['a', 'b', 'a', 'b', 'a', 'b', 'c'],
  ['one', 'two', 'three', 'four', 'One,', 'two', 'three', 'four.', 'one', 'two', 'three', 'four', 'five'],
  ['rumah', 'rumah', 'kita', 'ke', 'sana', 'ke', 'sana'],
  ['go', 'go', 'go', 'go', 'go', 'stop', 'and', 'stop', 'and', 'stop', 'and'],
].map((texts) => spaced(texts, 0, 0.3, 0.25));

const ALIGN_PHRASES = [
  {
    words: [w('The', 0.4, 0.55), w('river', 0.6, 0.9), w('runs', 0.95, 1.5), w('to', 1.55, 1.75), w('the', 1.85, 2.0), w('sea', 2.1, 3.1),
      w('and', 5.0, 5.2), w('back', 5.3, 5.6), w('one', 6.1, 6.3), w('two', 9.0, 9.2), w('three', 6.6, 6.9)],
    phrases: [
      { text: 'The river runs', start: 0.5, end: 1.6 },
      { text: 'to the sea', start: 1.8, end: 3.0 },
      { text: 'and back', start: 3.5, end: 4.5 },
      { text: '—', start: 4.6, end: 4.8 },
      { text: 'one two three', start: 6.0, end: 7.0 },
    ],
  },
  {
    words: [w('Lapan', 1.0, 1.3), w('1857,', 1.3, 2.4), w('Raja', 2.6, 3.0)],
    phrases: [
      { text: 'Lapan belas lima puluh tujuh,', start: 0.9, end: 2.5 },
      { text: 'Raja Muda', start: 2.5, end: 3.6 },
    ],
  },
];

const SENTENCE = spaced(['Semuanya', 'bermula', 'pada', 'tahun', '1857,', 'ketika', 'Raja', 'Muda', 'datang.', 'Raja', 'Muda', 'pergi.'], 1.0, 0.5, 0.4);
const FIND_PHRASE = [
  ['bermula pada tahun', 0],
  ['bermula pada tahun 1857 kemudian', 0],
  ['bermula sini tahun 1985 kemudian', 0],
  ['bermula pada sini 1985 ketika', 0],
  ['Raja Muda', 0],
  ['Raja Muda', 10],
  ['Tahun 1857,', 0],
  ['—', 0],
  ['Semuanya bermula pada tahun 1857 ketika Raja Muda datang Raja Muda pergi lagi', 0],
];

const WORDS_FILES = ['narration/voiceover.wav', 'take.v2.mp3', 'noext', '/abs/path/.hidden', 'dir/file.', 'dir/sub/'];

const dir = mkdtempSync(join(tmpdir(), 'transcript-oracle-'));
try {
  const source = join(jsDir, 'voice.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`transcript_oracle: voice.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  text = text.replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`).replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  text += '\nexport { envelope, evenOut, lastRun, pieces, snapEnds, snapOnsets };\n';
  const patched = join(dir, 'voice_patched.mjs');
  writeFileSync(patched, text);
  const voice = await import(pathToFileURL(patched).href);

  globalThis.oraclePcm = new Map();
  globalThis.oraclePipeline = async () => async (pcm, opts) => {
    const call = globalThis.oracleCalls.length;
    globalThis.oracleCalls.push({ n: pcm.length, language: opts.language, input: fnv1a(pcm) });
    const words = globalThis.oracleScript[call] ?? [];
    return { chunks: words.map(([text, s, e]) => ({ text, timestamp: [s, e] })) };
  };

  const str = (v) => String(v);
  const out = (words) => words.map((x) => ({ text: x.text, start: str(x.start), end: str(x.end) }));
  const copy = (words) => words.map((x) => ({ ...x }));

  const transcribe = [];
  for (const [i, c] of TRANSCRIBE.entries()) {
    const pcm = render(c.n, c.ops);
    const file = `narration/case${i}.wav`;
    globalThis.oraclePcm.set(file, pcm);
    globalThis.oracleCalls = [];
    globalThis.oracleScript = c.script;
    const env = voice.envelope(pcm);
    const t = await voice.transcribe(file, { model: MODEL, language: c.language ?? undefined });
    const row = {
      name: c.name,
      n: c.n,
      ops: c.ops,
      language: c.language ?? null,
      script: c.script,
      file,
      input: fnv1a(pcm),
      floor: str(env.floor),
      pieces: voice.pieces(env, pcm.length),
      calls: globalThis.oracleCalls,
      words: out(t.words),
      looped: t.looped,
      duration: str(t.duration),
      json: JSON.stringify(t, null, 1),
    };
    if (c.compose) {
      // The transcribe command's script step, in its order.
      t.heard = t.words;
      const { phrases, script } = c.compose;
      t.words = phrases
        ? voice.fitFile(voice.alignPhrases(t.words, phrases), file, phrases)
        : voice.fitFile(voice.alignScript(t.words, script), file);
      if (phrases) t.phrases = c.compose.phrasesFile;
      t.script = c.compose.scriptFile;
      row.compose = { ...c.compose, words: out(t.words), json: JSON.stringify(t, null, 1) };
    }
    transcribe.push(row);
  }

  const signals = SIGNALS.map((c) => {
    const pcm = render(c.n, c.ops);
    const file = `signal/${c.name}.wav`;
    globalThis.oraclePcm.set(file, pcm);
    const env = voice.envelope(pcm);
    const duration = pcm.length / S;
    const run = (lists, fn) => lists.map((words) => {
      const got = copy(words);
      fn(got);
      return { words, out: out(got) };
    });
    return {
      name: c.name,
      n: c.n,
      ops: c.ops,
      input: fnv1a(pcm),
      floor: str(env.floor),
      snapOnsets: run(c.snapOnsets, (x) => voice.snapOnsets(x, env)),
      snapEnds: run(c.snapEnds, (x) => voice.snapEnds(x, env, duration)),
      fitToSound: run(c.fitToSound, (x) => voice.fitToSound(x, env, duration)),
      lastRun: c.lastRun.map(([from, to]) => ({ from, to, out: voice.lastRun(env, from, to)?.map(str) ?? null })),
      fitFile: c.fitFile.map(({ words, phrase, spans }) => {
        const got = copy(words);
        if (phrase) got.forEach((x, k) => (x.phrase = phrase[k]));
        voice.fitFile(got, file, spans ?? null);
        return { words, phrase: phrase ?? null, spans: spans ?? null, out: out(got) };
      }),
    };
  });

  const evenOut = EVEN_OUT.map(({ words, limit }) => {
    const got = copy(words);
    voice.evenOut(got, limit ?? undefined);
    return { words, limit, out: out(got) };
  });

  const collapse = COLLAPSE.map((words) => {
    const r = voice.collapseLoops(copy(words));
    return { words, out: out(r.words), found: r.found };
  });

  const alignPhrases = ALIGN_PHRASES.map(({ words, phrases }) => {
    const got = voice.alignPhrases(copy(words), phrases);
    return { words, phrases, out: out(got), phrase: got.map((x) => x.phrase) };
  });

  const findPhrase = {
    words: SENTENCE,
    cases: FIND_PHRASE.map(([phrase, near]) => {
      const hit = voice.findPhrase(SENTENCE, phrase, near);
      return { phrase, near, out: hit && { start: str(hit.start), end: str(hit.end), score: str(hit.score) } };
    }),
  };

  const wordsFile = WORDS_FILES.map((file) => ({ file, out: voice.wordsFile({ outDir: 'out' }, file) }));

  // A voiceover starting 1 s into its audio, a scene with no narration, a narrated scene, and one not transcribed.
  const outDir = join(dir, 'out');
  const manifest = {
    outDir,
    voiceover: { file: 'narration/voiceover.wav', at: -1 },
    scenes: [
      { name: 'intro', start: 0 },
      { name: 'demo', start: 4, audio: 'narration/demo.wav', script: 'narration/demo.txt' },
      { name: 'outro', start: 9.5, audio: 'narration/outro.wav' },
    ],
  };
  const files = {
    'voiceover.words.json': {
      file: 'narration/voiceover.wav', model: MODEL, duration: 3.2, looped: false,
      words: [w('Lead', 0.2, 0.9), w('edge', 0.9, 1.0), w('straddle', 0.8, 1.4), w('after', 1.5, 2.25), w('last', 2.3, 3.1)],
    },
    'demo.words.json': {
      file: 'narration/demo.wav', model: MODEL, duration: 2, looped: true,
      words: [w('Demo', 0.05, 0.4), w('starts', 0.45, 1.05), w('now.', 1.1, 1.5)],
    },
  };
  mkdirSync(join(outDir, 'voice'), { recursive: true });
  const texts = {};
  for (const [name, t] of Object.entries(files)) {
    texts[name] = JSON.stringify(t, null, 1);
    writeFileSync(join(outDir, 'voice', name), texts[name]);
  }
  const spoken = voice.spokenWords(manifest);
  const timeline = JSON.stringify({ words: spoken.words.map((x) => ({ text: x.text, start: +x.start.toFixed(3), end: +x.end.toFixed(3) })) }, null, 1);
  const narrationRows = (list) => list.map((n) => ({ file: n.file, at: str(n.at), where: n.where, script: n.script ?? null }));
  const spokenCase = {
    manifest: { ...manifest, outDir: null },
    files: texts,
    narrations: narrationRows(voice.narrations(manifest)),
    words: out(spoken.words),
    missing: narrationRows(spoken.missing),
    timeline,
  };

  const { AUDIO_LEAD, AUDIO_TAIL } = await import(pathToFileURL(join(jsDir, 'manifest.mjs')).href);
  const constants = { audioLead: str(AUDIO_LEAD), audioTail: str(AUDIO_TAIL) };

  process.stdout.write(
    `${JSON.stringify({ constants, transcribe, signals, evenOut, collapse, alignPhrases, findPhrase, wordsFile, spoken: spokenCase }, null, 1)}\n`,
  );
} finally {
  rmSync(dir, { recursive: true, force: true });
}
