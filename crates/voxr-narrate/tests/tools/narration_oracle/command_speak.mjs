// Reference outputs for the speak command in `voxr_narrate` (`prepare_speak`, `finish_speak`): runs the skill's own
// `speakCmd`, taken from video.mjs, over patched copies of its modules, and prints one JSON document to stdout.
//
// `speak()` is stubbed: it records the script, the output path and the options it receives, writes the scenario's
// signal as a 16-bit WAV at the output path, and returns the scenario's result. The user's config and VoxCPM2's
// readiness are fixed per scenario. Everything else runs for real: the script and timed-script reading, the engine
// the script is reviewed for, the review, the written script, the voice polish, the pace, the script files and every
// printed line.
//
// Each scenario runs in a fresh directory holding its files, as the working directory. Every path under it prints
// as `<dir>`. Each records the speak() calls, every line printed with its stream, the exit code or the error (and
// whether it is a bad input), and every file left with its text, or the FNV-1a hash of its bytes for a WAV.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node command_speak.mjs > ../../fixtures/narration_command_speak_oracle.json
import { cpSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { commandModule, normalize, scenarioFiles } from './command_files.mjs';
import { render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('command_speak: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

const RATE = 48000;
// Two phrases of speech: a tone each, a pause between.
const TWO = { n: 120000, ops: [tone(200, 4800, 43200, 0.3, 480, 480), tone(150, 62400, 48000, 0.12, 480, 480)] };
const SPANS = [
  { text: 'First line here now.', start: 0, end: 1.1 },
  { text: 'Second line ends it.', start: 1.3, end: 2.4 },
];
const NOT_READY = { ok: false, why: 'needs an NVIDIA GPU with 8 GB of memory (none found)' };
const READY = { ok: true, why: 'NVIDIA GeForce, 12 GB' };
const whole = (text, ok = true, extra = {}) => ({ at: 0, text, ok, heard: text, why: ok ? '' : 'missing: it', doubt: [], pauses: [], ...extra });

const SCENARIOS = [
  {
    name: 'kokoro by default, from a file',
    files: { 'script.txt': 'First line here now.\nSecond line ends it\n' },
    args: { rest: ['script.txt'] },
    config: {},
    ready: NOT_READY,
    wav: TWO,
    result: { duration: 2.5, engine: 'Kokoro (af_heart)', phrases: SPANS, checks: [whole('First line here now. Second line ends it')], problems: [] },
  },
  {
    name: 'voxcpm with notes on every part and a problem',
    files: {},
    args: { rest: ['First', 'line', 'here', 'now.', 'Second line ends it?'], voice: '(warm, low)', speed: 0.9, pace: 3.5, seed: 3, reroll: [2] },
    config: {},
    ready: READY,
    wav: TWO,
    result: {
      duration: 2.5,
      engine: 'VoxCPM2 (warm, low)',
      phrases: SPANS,
      voiceFile: null,
      checks: [
        { at: 0, text: 'First line here now.', ok: true, heard: 'First line here now.', why: '', seed: 7, doubt: ['here'], heat: 1.75, hotText: 'First line here now.', pauses: [{ after: 'line', before: 'here', gap: 0.52 }], clicks: [0.31, 0.8] },
        { at: 1.3, text: 'Second line ends it?', ok: false, heard: 'Second line end it', why: 'misheard: ends', seed: 1007, doubt: [], heat: 1.2, hotText: 'Second line ends it?', pauses: [], clicks: [] },
      ],
      problems: ['part 2 "Second line ends it?": misheard: ends (heard "Second line end it"). Three takes failed, so reword it first (a longer line, another word order). --reroll 2 draws new takes.'],
    },
  },
  {
    name: 'a timed script with a written script, raw',
    files: {
      'brief.txt': '0-2s: First line here now.\n2-4s: Second line ends it.\n',
      'written.txt': '0-2s: First line, here now!\n2-4s: Second line ends it.\n',
    },
    args: { rest: ['brief.txt'], written: 'written.txt', raw: true, engine: 'kokoro', out: 'audio/vo.wav' },
    config: { tts: { pace: 4.5 } },
    ready: READY,
    wav: TWO,
    result: { duration: 4.2, engine: 'Kokoro (af_heart)', phrases: SPANS, timing: ['line 2 "Second line ends it." starts 0.40s late: shorten line 1'], checks: [whole('First line here now. Second line ends it.')], problems: [] },
  },
  {
    name: 'a one-call command leaves no spans',
    files: { 'assets/voiceover.phrases.json': '{"script":"old","phrases":[]}' },
    args: { rest: ['Hello there, friend.'], oneCall: true },
    config: { tts: { command: 'tts {text} {out}', speed: 1.2 } },
    ready: READY,
    wav: TWO,
    result: { duration: 2.5, engine: 'command: tts {text} {out}', checks: [whole('Hello there, friend.', false)], problems: ['missing: it. Heard: "Hello there, friend.".'] },
  },
  {
    name: 'the script beside the output keeps its name',
    files: { 'assets/voiceover.txt': '[pause] First line here now.\n\nSecond line ends it.' },
    args: { rest: ['assets/voiceover.txt'], speed: 1.25, engine: 'kokoro' },
    config: { tts: { pace: 4.1 } },
    ready: NOT_READY,
    wav: TWO,
    result: { duration: 2.5, engine: 'Kokoro (af_heart)', phrases: SPANS, checks: [whole('First line here now. Second line ends it.')], problems: [] },
  },
  {
    name: 'a model run is reviewed for no engine',
    files: {},
    args: { rest: ['Then the river runs north past the mill.'], model: 'facebook/mms-tts-zlm', language: 'ms' },
    config: {},
    ready: NOT_READY,
    wav: TWO,
    result: { duration: 2.5, engine: 'facebook/mms-tts-zlm', phrases: SPANS, checks: [whole('Then the river runs north past the mill.')], problems: [] },
  },
  { name: 'review only', files: {}, args: { rest: ['Hello there, friend.'], review: true }, config: {}, ready: NOT_READY },
  { name: 'review with notes, then a missing written script', files: {}, args: { rest: ['Use the --raw flag'], written: 'gone.txt' }, config: {}, ready: NOT_READY },
  { name: 'an empty script', files: {}, args: { rest: ['  '] }, config: {}, ready: NOT_READY },
  {
    name: 'a written script with other words',
    files: { 'w.txt': 'Hello there, enemy.' },
    args: { rest: ['Hello there, friend.'], written: 'w.txt' },
    config: {},
    ready: NOT_READY,
  },
  { name: 'a language with no engine', files: {}, args: { rest: ['Satu dua tiga.'], language: 'ms' }, config: {}, ready: NOT_READY },
  { name: 'a configured engine wins over the default', files: {}, args: { rest: ['Is it here?'], review: true }, config: { tts: { engine: 'voxcpm' } }, ready: NOT_READY },
];

const PATCHES = {
  'paths.mjs': [{ at: 'export function config() {\n', after: '  if (globalThis.oracleConfig) return globalThis.oracleConfig;\n' }],
  'speak.mjs': [
    { at: 'export async function speak(script, out, opts = {}) {\n', after: '  if (globalThis.oracleSpeak) return globalThis.oracleSpeak(script, out, opts);\n' },
    { at: 'export function voxcpmReady() {\n', after: '  if (globalThis.oracleReady) return globalThis.oracleReady;\n' },
  ],
};

// A 16-bit mono WAV of `x`, as speak.mjs writes one.
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

const work = realpathSync(mkdtempSync(join(tmpdir(), 'command-speak-oracle-')));
try {
  const cmd = await import(pathToFileURL(commandModule(jsDir, work, PATCHES)).href);
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
    const calls = [];
    const lines = [];
    globalThis.oracleConfig = s.config;
    globalThis.oracleReady = s.ready;
    globalThis.oracleSpeak = async (script, out, opts) => {
      calls.push({ script, out: normalize(out, dir), opts: JSON.parse(normalize(JSON.stringify(opts), dir)) });
      writeFileSync(out, wav16(render(s.wav.n, s.wav.ops), RATE));
      return structuredClone(s.result);
    };
    const log = console.log;
    const error = console.error;
    console.log = (...a) => lines.push({ stream: 'out', text: normalize(a.join(' '), dir) });
    console.error = (...a) => lines.push({ stream: 'err', text: normalize(a.join(' '), dir) });
    process.chdir(dir);
    let code = null;
    let failure = null;
    try {
      code = await cmd.speakCmd(structuredClone(s.args));
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
