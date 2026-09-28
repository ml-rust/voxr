// Reference outputs for the speak run of every engine but VoxCPM2 in `voxr_narrate`: runs a patched copy of
// speak.mjs's `speak` end to end for Kokoro, a Hugging Face model and the user's TTS command. Prints one JSON document
// to stdout.
//
// The patches stub the config, KokoroTTS, the transformers pipeline, the shell that runs the command, and the
// recognizer and its model choice inside heardCheck. The phrase split, the pauses, the spans, the WAV, ffmpeg's decode
// of the command's output, the speech check, the stray pauses and the result run for real.
//
// Each engine call speaks a tone over floor noise (signals.mjs): 2400 samples plus 1200 per word of its text, divided
// by the speed for Kokoro, with the noise seeded from the call's place in the run. Kokoro speaks at 24 kHz and the
// model at 16 kHz. The command's stub writes the tone to {out} as a 32-bit float WAV at 48 kHz, which ffmpeg passes
// through unchanged, unless the case makes it exit nonzero, end on a signal, or write bytes that are no audio. The
// recognizer hears the case's scripted words, in turn per model, and records the model, the language and the FNV-1a
// hash of the samples the voiceover holds.
//
// Each case records every engine call (the text, the voice, the speed, the rate, the tone's length and ops, and for
// the command its shell line with the work directory as <dir>, the text file's contents and its exit code), every
// stderr line, every recognizer call, the FNV-1a hash of the joined samples and their rate, the hash of the WAV's
// bytes, and the result (duration, phrases, checks, problems, engine), or the error's message.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_simple.mjs > ../../fixtures/narration_simple_oracle.json
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { wavF32, wavSamples } from './oracle_files.mjs';
import { fnv1a, fnv1aBytes, noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_simple: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it.
const PATCHES = [
  {
    at: '  const tts = config().tts ?? {};\n',
    replace: '  const tts = globalThis.oracleConfig ?? config().tts ?? {};\n',
  },
  {
    at: "  const { KokoroTTS } = await speechModule('kokoro-js');\n",
    replace: "  const { KokoroTTS } = globalThis.oracleKokoro ?? (await speechModule('kokoro-js'));\n",
  },
  {
    at: '  const { pipeline } = await speechModule();\n',
    replace: '  const { pipeline } = globalThis.oraclePipeline ?? (await speechModule());\n',
  },
  {
    at: "    const r = spawnSync(cmd, { shell: true, stdio: ['ignore', 'inherit', 'inherit'] });\n",
    replace:
      "    const r = globalThis.oracleShell ? globalThis.oracleShell(cmd, { dir, raw, textFile }) : spawnSync(cmd, { shell: true, stdio: ['ignore', 'inherit', 'inherit'] });\n",
  },
  {
    at: "  const { transcribe, checkSpeech, asrModel, DEFAULT_MODEL, SMALL_MODEL } = await import('./voice.mjs');\n",
    replace:
      "  const { transcribe: realTranscribe, checkSpeech, asrModel: realAsrModel, DEFAULT_MODEL, SMALL_MODEL } = await import('./voice.mjs');\n" +
      '  const transcribe = globalThis.oracleHear ?? realTranscribe;\n' +
      '  const asrModel = globalThis.oracleAsrModel ?? realAsrModel;\n',
  },
  {
    at: 'function wav(parts, rate) {\n',
    after: '  globalThis.oracleWav?.(parts, rate);\n',
  },
];

const COMMAND = 'tts --say {text} --file {text_file} --out {out} --voice {voice} --ref {reference}';
const KOKORO_VOICES = ['af_heart', 'af_bella', 'bm_george'];

// Words heard evenly, one every 0.5 s and 0.4 s long; `gapBefore` pushes that word and every later one 1 s on.
function said(line, { looped = false, gapBefore = -1 } = {}) {
  const words = line.split(/\s+/).map((text, i) => {
    const start = i * 0.5 + (gapBefore >= 0 && i >= gapBefore ? 1 : 0);
    return { text, start, end: start + 0.4 };
  });
  return { words, looped };
}

const CASES = [
  {
    name: 'kokoro, every kind of pause',
    script: '(calm) The river runs north. It turns east at the hill.\nThe herds come back to graze.\n\nThey rest here [pause 1.2] then move on [pause] at dawn.',
    opts: { engine: 'kokoro' },
    hear: { default: [said('The river runs north. It turns east at the hill. The herds come back to graze. They rest here then move on at dawn.')], small: [] },
  },
  {
    name: 'kokoro at 1.25 in another voice, a stray pause heard',
    script: 'Fish swim upstream in spring.\nBirds fly south.',
    opts: { engine: 'kokoro', voice: 'bm_george', speed: 1.25 },
    hear: { default: [said('Fish swim upstream in spring. Birds fly south.', { gapBefore: 2 })], small: [] },
  },
  {
    name: 'a model in Malay, a word both recognizers mishear',
    script: 'The river runs north today.',
    opts: { model: 'facebook/mms-tts-eng', language: 'ms' },
    hear: { default: [said('the river walks north today')], small: [said('the river walks north today')] },
  },
  {
    name: 'a command per phrase, the check fails',
    script: 'Pay $$ now.\nThen $& after it.\n\nDone.',
    opts: { command: COMMAND, voice: "o'neil", reference: 'ref {text}.wav' },
    hear: { default: [said('Pay now.')], small: [] },
  },
  {
    name: 'a command in one call, looped then heard again',
    script: 'One line here.\n\nAnother line [pause 0.5] there.',
    opts: { command: COMMAND, oneCall: true },
    hear: { default: [said('One line', { looped: true })], small: [said('One line here. Another line there.')] },
  },
  {
    name: 'a command that exits nonzero',
    script: 'First part.\nSecond part.\nThird part.',
    opts: { command: COMMAND },
    shell: { at: 1, status: 3 },
    hear: { default: [], small: [] },
  },
  {
    name: 'a command a signal ends',
    script: 'First part.\nSecond part.',
    opts: { command: COMMAND },
    shell: { at: 0, status: null },
    hear: { default: [], small: [] },
  },
  {
    name: 'a command that writes no audio',
    script: 'First part.',
    opts: { command: COMMAND },
    shell: { at: 0, status: 0, garbage: true },
    hear: { default: [], small: [] },
  },
  {
    name: 'kokoro with a voice it lacks',
    script: 'Hello there.',
    opts: { engine: 'kokoro', voice: 'zz_nobody' },
    hear: { default: [], small: [] },
  },
];

// The tone call `k` of a run speaks for `text`, at `speed`.
function signal(text, k, speed) {
  const n = Math.round((2400 + 1200 * text.split(/\s+/).length) / speed);
  const ops = [noise(0, n, 0.00002, 11 + k), tone(200, 240, n - 480, 0.3, 240, 240)];
  return { n, ops, x: render(n, ops) };
}

const dir = mkdtempSync(join(tmpdir(), 'simple-oracle-'));
const realError = console.error;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_simple: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  const base = pathToFileURL(join(jsDir, '/')).href;
  text = text
    .replaceAll("from './", `from '${base}`)
    .replaceAll("import('./", `import('${base}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text);
  const speak = await import(pathToFileURL(patched).href);
  const { DEFAULT_MODEL, SMALL_MODEL } = await import(pathToFileURL(join(jsDir, 'voice.mjs')).href);

  const cases = [];
  for (const [n, c] of CASES.entries()) {
    const root = join(dir, `case-${n}`);
    mkdirSync(root, { recursive: true });
    const out = join(root, 'out.wav');
    const calls = [];
    const heard = [];
    const messages = [];
    const queue = { default: [...c.hear.default], small: [...c.hear.small] };
    let joined = null;
    globalThis.oracleConfig = {};
    globalThis.oracleKokoro = {
      KokoroTTS: {
        from_pretrained: async () => ({
          voices: Object.fromEntries(KOKORO_VOICES.map((v) => [v, {}])),
          generate: async (said, { voice, speed }) => {
            const s = signal(said, calls.length, speed);
            calls.push({ text: said, voice, speed, rate: 24000, n: s.n, ops: s.ops });
            return { audio: s.x, sampling_rate: 24000 };
          },
        }),
      },
    };
    globalThis.oraclePipeline = {
      pipeline: async () => async (said) => {
        const s = signal(said, calls.length, 1);
        calls.push({ text: said, voice: null, speed: null, rate: 16000, n: s.n, ops: s.ops });
        return { audio: s.x, sampling_rate: 16000 };
      },
    };
    globalThis.oracleShell = (cmd, ctx) => {
      const k = calls.length;
      const fails = c.shell && c.shell.at === k;
      const status = fails ? c.shell.status : 0;
      const textFile = readFileSync(ctx.textFile, 'utf8');
      const call = { line: cmd.replaceAll(ctx.dir, '<dir>'), textFile, status, n: null, ops: null, garbage: false };
      if (fails && c.shell.garbage) {
        writeFileSync(ctx.raw, 'not audio');
        call.garbage = true;
      } else if (!fails) {
        const s = signal(textFile.replace(/\n$/, ''), k, 1);
        writeFileSync(ctx.raw, wavF32(s.x, 48000));
        call.n = s.n;
        call.ops = s.ops;
      }
      calls.push(call);
      return { status };
    };
    globalThis.oracleAsrModel = (language) => (language && !/^en/i.test(language) ? SMALL_MODEL : DEFAULT_MODEL);
    globalThis.oracleHear = async (file, { model, language }) => {
      const which = model === SMALL_MODEL ? 'small' : 'default';
      heard.push({ model: which, language: language ?? null, samples: fnv1a(wavSamples(readFileSync(file))) });
      return queue[which].shift() ?? { words: [], looped: false };
    };
    globalThis.oracleWav = (parts, rate) => {
      const all = new Float32Array(parts.reduce((k, p) => k + p.length, 0));
      let o = 0;
      for (const p of parts) (all.set(p, o), (o += p.length));
      joined = { samples: fnv1a(all), rate };
    };
    console.error = (line) => messages.push(`${line}`);
    let result = null;
    let error = null;
    try {
      result = await speak.speak(c.script, out, { ...c.opts });
    } catch (e) {
      error = e.message;
    } finally {
      console.error = realError;
    }
    cases.push({
      name: c.name,
      script: c.script,
      opts: { engine: null, command: null, model: null, voice: null, reference: null, language: null, speed: null, oneCall: false, ...c.opts },
      hear: c.hear,
      calls,
      messages,
      heard,
      joined,
      wav: result ? fnv1aBytes(readFileSync(out)) : null,
      error,
      result: result && {
        duration: String(result.duration),
        phrases: result.phrases === undefined ? null : JSON.stringify(result.phrases, null, 1),
        checks: JSON.stringify(result.checks, null, 1),
        problems: result.problems,
        engine: result.engine,
      },
    });
  }
  process.stdout.write(`${JSON.stringify({ voices: KOKORO_VOICES, cases }, null, 1)}\n`);
} finally {
  console.error = realError;
  for (const k of ['oracleConfig', 'oracleKokoro', 'oraclePipeline', 'oracleShell', 'oracleAsrModel', 'oracleHear', 'oracleWav']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
