// Reference outputs for the transcribe command in `voxr_narrate` (`plan_transcribe`, `narration_words`, the lines and
// the video timeline): runs the skill's own `transcribeCmd`, taken from video.mjs, and the part of `main` that picks
// its manifest, over patched copies of its modules, and prints one JSON document to stdout.
//
// `transcribe()` is stubbed: it returns the scenario's recognized words for the file's name, on the file's real
// duration. Everything else runs for real: the manifest (with ffprobe), the narration list, the script reading
// (plain, subtitles, and the phrase spans speak recorded), the alignment, the fit on the sound (with ffmpeg), the
// words files, the video timeline and every printed line. Each audio file is a 16 kHz 16-bit WAV, which ffmpeg reads
// back exactly.
//
// Each scenario runs in a fresh directory holding its files, as the working directory. Every path under it prints
// as `<dir>`. Each records every line printed with its stream, the exit code or the error (and whether it is a bad
// input), and every file left with its text, or the FNV-1a hash of its bytes for a WAV.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node command_transcribe.mjs > ../../fixtures/narration_command_transcribe_oracle.json
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { commandModule, normalize, scenarioFiles } from './command_files.mjs';
import { noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('command_transcribe: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

const RATE = 16000;
// Three words of speech over 2 s: a tone each, with gaps.
const SPEECH = { n: 32000, ops: [noise(0, 32000, 0.00002, 5), tone(200, 3200, 6400, 0.3, 160, 160), tone(150, 12800, 4800, 0.25, 160, 160), tone(260, 20800, 6400, 0.3, 160, 160)] };
const HEARD = [
  { text: 'One', start: 0.18, end: 0.62 },
  { text: 'too', start: 0.79, end: 1.12 },
  { text: 'three.', start: 1.3, end: 1.71 },
];
const w = (text, start, end) => ({ text, start, end });

// The part of `main` the oracle replays, as video.mjs holds it.
const MAIN_LINES = [
  "  if (cmd === 'transcribe' && opts.rest.length && !existsSync(opts.manifest)) {\n    await transcribeCmd({ outDir: resolve('out') }, opts);\n    return 0;\n  }\n",
  "  const m = loadManifest(opts.manifest, { voiceoverFits: ['check', 'still', 'render'].includes(cmd) });\n",
  "    await transcribeCmd(m, opts);\n    if (m.voiceoverOverrun > 0) console.log(`The voiceover runs ${m.voiceoverOverrun.toFixed(2)}s past the video's end. Size the scenes to its words above, so the video holds all of it.`);\n",
];

// Manifests the skill refuses for their sound direction: [name, top-level fields, scenes].
const one = (fields) => [{ file: 'a.html', duration: 2, ...fields }, { file: 'b.html', duration: 2 }];
const two = (fields) => [{ file: 'a.html', duration: 2 }, { file: 'b.html', duration: 2, ...fields }];
const SOUND_REFUSALS = [
  ['energy above 1', {}, one({ energy: 1.5 })],
  ['energy of the wrong type', {}, one({ energy: 'high' })],
  ['energy as an empty list', {}, one({ energy: [] })],
  ['an energy point out of range', {}, one({ energy: [[0, 0.5], [1, 2]] })],
  ['an energy point that is no pair', {}, one({ energy: [[0, 0.5], { at: 1 }] })],
  ['an energy point past the scene', {}, one({ energy: [[0, 0.5], [5, 0.2]] })],
  ['energy points out of order', {}, one({ energy: [[1, 0.5], [0.5, 0.2]] })],
  ['an unknown transition', {}, two({ in: 'fade' })],
  ['a transition on the first scene', {}, one({ in: 'whoosh' })],
  ['a transition volume too high', {}, two({ inVol: 5 })],
  ['a transition too short', {}, two({ inDur: 0.1 })],
  ['music with no file', { music: { volume: 0.5 } }],
  ['music as a string', { music: 'music.wav' }],
  ['music too loud', { music: { file: 'music.wav', volume: 2 } }],
  ['music starting before the track', { music: { file: 'music.wav', start: -1 } }],
  ['an unknown space', { sound: { space: 'big' } }],
  ['a fade too long', { sound: { fadeIn: 6 } }],
  ['a fade of the wrong type', { sound: { fadeOut: 'slow' } }],
  ['an unknown palette', { sound: { palette: 'paper' } }],
  ['an unknown key', { sound: { key: 'H' } }],
  ['a key with no root', { sound: { key: 'Cb' } }],
  ['a bed of the wrong type', { sound: { bed: 'yes' } }],
  ['a bed with air', { sound: { bed: { air: 1 } } }],
  ['a bed with unknown fields', { sound: { bed: { zeta: 1, alpha: 2 } } }],
  ['a bed with too many fragments', { sound: { bed: { fragments: 9 } } }],
  ['a bed level of the wrong type', { sound: { bed: { level: 'x' } } }],
  ['a bed with a fractional seed', { sound: { bed: { seed: 1.5 } } }],
  ['a bed of true passes, a bad key after it does not', { sound: { bed: true, key: 'H' } }],
  ['a bed of true is valid, a sound block after a bad voiceover is not read', { voiceover: 'gone.wav', sound: { bed: true, key: 'H' } }],
];

const SCENARIOS = [
  {
    name: 'a manifest: voiceover, a subtitled scene, a plain scene',
    audio: { 'vo.wav': SPEECH, 'scene.wav': SPEECH, 'plain.wav': SPEECH },
    files: {
      'video.json': JSON.stringify({
        fps: 30,
        scenes: [
          { file: 'scenes/01-hook.html', duration: 1 },
          { file: 'scenes/02-sub.html', audio: 'scene.wav', script: 'scene.srt' },
          { file: 'scenes/03-plain.html', audio: 'plain.wav' },
        ],
        voiceover: { file: 'vo.wav', start: -0.25, script: 'vo.txt' },
      }),
      'vo.txt': '[pause] One two, three.',
      'scene.srt': '1\n00:00:00,150 --> 00:00:01,150\nOne two\n\n2\n00:00:01,250 --> 00:00:01,900\nthree.\n',
    },
    heard: { 'vo.wav': HEARD, 'scene.wav': HEARD, 'plain.wav': [w('One', 0.2, 0.6), w('two', 0.8, 1.1), w('three.', 1.3, 1.7), w('you', 1.95, 2.0)] },
    opts: { rest: [], manifest: 'video.json' },
  },
  {
    name: 'a voiceover past the video end',
    audio: { 'vo.wav': SPEECH },
    files: { 'video.json': JSON.stringify({ scenes: [{ file: 'a.html', duration: 1 }], voiceover: 'vo.wav' }) },
    heard: { 'vo.wav': HEARD },
    opts: { rest: [], manifest: 'video.json', language: 'ms' },
    config: { asr: { model: 'openai/whisper-large-v3' } },
  },
  {
    name: 'a file with no manifest, its recorded spans and a script',
    audio: { 'take.wav': SPEECH },
    files: {
      'take.phrases.json': JSON.stringify({ script: 'One two,  three.', phrases: [w('One two,', 0.15, 1.2), w('three.', 1.25, 1.9)] }),
      'take.txt': 'One two, three.\n',
    },
    heard: { 'take.wav': HEARD },
    opts: { rest: ['take.wav'], manifest: 'video.json', script: 'take.txt', model: 'onnx-community/whisper-small_timestamped' },
  },
  {
    name: 'a file beside a manifest writes under its out directory',
    audio: { 'take.wav': SPEECH },
    files: { 'video.json': JSON.stringify({ scenes: [{ file: 'a.html', duration: 3 }] }), 'old.phrases.json': '{}' },
    heard: { 'take.wav': HEARD },
    opts: { rest: ['take.wav'], manifest: 'video.json' },
  },
  {
    // `--manifest` given last, with no value: `opts.manifest` is undefined, and the file is read alone.
    name: 'a file beside a manifest, --manifest given no value',
    audio: { 'take.wav': SPEECH },
    files: { 'video.json': JSON.stringify({ scenes: [{ file: 'a.html', duration: 3 }] }) },
    heard: { 'take.wav': HEARD },
    opts: { rest: ['take.wav'], manifest: undefined },
  },
  { name: 'two files and one script', audio: {}, files: { 's.txt': 'x' }, heard: {}, opts: { rest: ['a.wav', 'b.wav'], manifest: 'video.json', script: 's.txt' } },
  { name: 'a script that does not exist', audio: { 'take.wav': SPEECH }, files: {}, heard: {}, opts: { rest: ['take.wav'], manifest: 'video.json', script: 'gone.txt' } },
  { name: 'a file that does not exist', audio: { 'take.wav': SPEECH }, files: {}, heard: { 'take.wav': HEARD }, opts: { rest: ['take.wav', 'gone.wav'], manifest: 'video.json' } },
  { name: 'no narration', audio: {}, files: { 'video.json': JSON.stringify({ scenes: [{ file: 'a.html', duration: 2 }] }) }, heard: {}, opts: { rest: [], manifest: 'video.json' } },
  { name: 'no manifest', audio: {}, files: {}, heard: {}, opts: { rest: [], manifest: 'video.json' } },
  {
    name: 'every sound field set, and valid',
    audio: { 'vo.wav': SPEECH, 'music.wav': SPEECH },
    files: {
      'video.json': JSON.stringify({
        scenes: [
          { file: 'a.html', duration: 1.5, energy: [[0, 0.2], [0.5, 0.9], [1.5, 0.4]] },
          { file: 'b.html', duration: 1.5, energy: 1, in: 'riser', inVol: 4, inDur: 0.2 },
        ],
        music: { file: 'music.wav', volume: 1, start: 0 },
        voiceover: 'vo.wav',
        sound: { space: 'hall', fadeIn: 0, fadeOut: 5, palette: 'ceramic', key: 'f# minor', bed: { fragments: 8, level: -48, seed: 0 } },
      }),
    },
    heard: { 'vo.wav': HEARD },
    opts: { rest: [], manifest: 'video.json' },
  },
  ...SOUND_REFUSALS.map(([name, extra, scenes]) => ({
    name,
    audio: { 'music.wav': SPEECH },
    files: { 'video.json': JSON.stringify({ scenes: scenes ?? [{ file: 'a.html', duration: 2 }, { file: 'b.html', duration: 2 }], ...extra }) },
    heard: {},
    opts: { rest: [], manifest: 'video.json' },
  })),
  {
    name: 'subtitles with no cues',
    audio: { 'take.wav': SPEECH },
    files: { 'empty.srt': '\n' },
    heard: { 'take.wav': HEARD },
    opts: { rest: ['take.wav'], manifest: 'video.json', script: 'empty.srt' },
  },
];

const PATCHES = {
  'paths.mjs': [{ at: 'export function config() {\n', after: '  if (globalThis.oracleConfig) return globalThis.oracleConfig;\n' }],
  'voice.mjs': [
    {
      at: 'export async function transcribe(file, { model = DEFAULT_MODEL, language } = {}) {\n',
      after: '  if (globalThis.oracleTranscribe) return globalThis.oracleTranscribe(file, { model, language });\n',
    },
  ],
};

// A 16-bit mono WAV of `x`.
function wav16(x, rate) {
  const buf = Buffer.alloc(44 + x.length * 2);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + x.length * 2, 4);
  buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(rate, 24);
  buf.writeUInt32LE(rate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(x.length * 2, 40);
  x.forEach((v, i) => buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32767))), 44 + i * 2));
  return buf;
}

const work = realpathSync(mkdtempSync(join(tmpdir(), 'command-transcribe-oracle-')));
try {
  const cmd = await import(pathToFileURL(commandModule(jsDir, work, PATCHES, MAIN_LINES)).href);
  const home = process.cwd();
  const scenarios = [];
  for (const s of SCENARIOS) {
    const dir = join(work, 'run');
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    for (const [path, text] of Object.entries(s.files)) {
      mkdirSync(join(dir, path, '..'), { recursive: true });
      writeFileSync(join(dir, path), text);
    }
    for (const [path, sig] of Object.entries(s.audio)) writeFileSync(join(dir, path), wav16(render(sig.n, sig.ops), RATE));
    const lines = [];
    const calls = [];
    globalThis.oracleConfig = s.config ?? {};
    globalThis.oracleTranscribe = async (file, { model, language }) => {
      calls.push({ file: normalize(file, dir), model, language: language ?? null });
      const sig = s.audio[basename(file)];
      return { file, model, duration: +(sig.n / RATE).toFixed(3), words: structuredClone(s.heard[basename(file)]), looped: false };
    };
    const log = console.log;
    const error = console.error;
    console.log = (...a) => lines.push({ stream: 'out', text: normalize(a.join(' '), dir) });
    console.error = (...a) => lines.push({ stream: 'err', text: normalize(a.join(' '), dir) });
    process.chdir(dir);
    let code = null;
    let failure = null;
    try {
      const opts = structuredClone(s.opts);
      if (opts.rest.length && !existsSync(opts.manifest)) {
        await cmd.transcribeCmd({ outDir: resolve('out') }, opts);
      } else {
        const m = cmd.loadManifest(opts.manifest, { voiceoverFits: false });
        await cmd.transcribeCmd(m, opts);
        if (m.voiceoverOverrun > 0) console.log(`The voiceover runs ${m.voiceoverOverrun.toFixed(2)}s past the video's end. Size the scenes to its words above, so the video holds all of it.`);
      }
      code = 0;
    } catch (e) {
      failure = { input: e instanceof cmd.ManifestError, message: normalize(e.message, dir) };
    } finally {
      process.chdir(home);
      console.log = log;
      console.error = error;
    }
    scenarios.push({ ...s, output: { calls, lines: lines.flatMap((l) => l.text.split('\n').map((text) => ({ stream: l.stream, text }))), code, failure, files: scenarioFiles(dir) } });
  }
  process.stdout.write(`${JSON.stringify({ rate: RATE, scenarios }, null, 1)}\n`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
