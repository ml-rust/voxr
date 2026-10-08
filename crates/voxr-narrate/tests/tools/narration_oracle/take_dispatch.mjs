// Reference outputs for the whole speak run in `voxr_narrate` (`speak_script`): runs a patched copy of speak.mjs's
// `speak` end to end over every branch it picks. Prints one JSON document to stdout.
//
// The patches stub the config, VoxCPM2's readiness, KokoroTTS, the transformers pipeline, the shell that runs the
// command, the recognizer and its model choice inside heardCheck, and the VoxCPM2 run itself right after its
// readiness check. They also record the options `command()` is called with. The engine choice, the tag strip, the
// phrase split, the spans, the WAV, the speech check, ffmpeg's decode, its time-stretch and resampler, and the timed
// placement run for real.
//
// Each Kokoro, model or command call speaks a tone over floor noise (signals.mjs): 2400 samples plus 1200 per word of
// its text, divided by the speed for Kokoro, the noise seeded from the call's place in the run. Kokoro speaks at
// 24 kHz, the model at 16 kHz, and the command writes a 32-bit float WAV at 48 kHz. The VoxCPM2 stub speaks each of
// its parts (one per chunk, or per line for a timed script) as 0.5 s of tone in a 1 s slot at 48 kHz, written by the
// skill's own `wav`, with one passing check per part.
//
// Each case records its script, options, config and readiness, every engine call, the options `command()` gets, the
// VoxCPM2 stub's call and signal, every stderr line, every recognizer call, the FNV-1a hash and length of the 16-bit
// data the final WAV holds, and the result as JSON, or the error's message. A case's directory is written as <dir>.
// A case marked resampled goes through ffmpeg's resampler, whose samples match voxr's bit for bit on its AVX2 and FMA
// path only.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node take_dispatch.mjs > ../../fixtures/narration_dispatch_oracle.json
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { wavF32, wavSamples } from './oracle_files.mjs';
import { fnv1a, fnv1aBytes, noise, render, tone } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('take_dispatch: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs, and what replaces or follows it.
const PATCHES = [
  {
    at: '  const tts = config().tts ?? {};\n',
    replace: '  const tts = globalThis.oracleConfig ?? config().tts ?? {};\n',
  },
  {
    at: 'export function voxcpmReady() {\n',
    after: '  if (globalThis.oracleReady) return globalThis.oracleReady;\n',
  },
  {
    at: '  if (!ready.ok && !device) throw new Error(`speak --engine voxcpm ${ready.why}. Use the default Kokoro engine, another TTS with --command, or --device cpu (very slow).`);\n',
    after:
      '  if (globalThis.oracleVoxcpm) return globalThis.oracleVoxcpm(script, { voice, reference, style, hifi, cfg, steps, device: device ?? null, seed: asked ?? null, configSeed: configSeed ?? null, language: language ?? null, reroll, lines }, out);\n',
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
    at: "export function command(script, { command: template, voice = '', reference = '', oneCall = false }) {\n",
    after: '  globalThis.oracleCommand?.({ template, voice, reference, oneCall });\n',
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
];
const EXPORTS = '\nexport { wav as oracleWavFile };\n';

const COMMAND = 'tts --say {text} --out {out} --voice {voice} --ref {reference}';
const NOT_READY = { ok: false, why: 'needs an NVIDIA GPU with 8 GB of memory (none found)' };
const READY = { ok: true, why: '' };

// Words heard evenly, one every 0.5 s and 0.4 s long.
function said(line) {
  return { words: line.split(/\s+/).map((text, i) => ({ text, start: i * 0.5, end: i * 0.5 + 0.4 })), looped: false };
}
const cue = (start, end, text) => ({ start, end, text });

const CASES = [
  {
    name: 'no engine named, VoxCPM2 not ready: Kokoro, tags stripped',
    script: '[laughing] The river runs north.\nIt turns east [sigh] at the hill.',
    opts: {},
    hear: [said('The river runs north. It turns east at the hill.')],
  },
  {
    name: 'no engine named, VoxCPM2 ready: every VoxCPM2 option, tags kept, speed unused',
    script: '[sigh] The river runs north.\n\nIt turns east.',
    opts: { voice: '(warm, low)', style: '(calm)', speed: 0.8, seed: 11, reroll: [1], cfg: 2, steps: 10, hifi: true, language: 'en' },
    tts: { seed: 5 },
    ready: READY,
  },
  {
    name: 'a reference picks VoxCPM2, a device runs it where it is not ready, config options',
    script: 'Herds come back to graze.',
    opts: { reference: '<dir>/ref.wav', device: 'cpu' },
    tts: { style: '(slow)', cfg: 1.8, steps: 12, hifi: false },
  },
  {
    name: 'VoxCPM2 not ready and no device',
    script: 'Herds come back to graze.',
    opts: { engine: 'voxcpm' },
  },
  {
    name: 'a command at 1.25 is time-stretched',
    script: 'The river runs north.\nIt turns east at the hill.',
    opts: { command: COMMAND, speed: 1.25, voice: 'Ann' },
    hear: [said('The river runs north. It turns east at the hill.')],
  },
  {
    name: 'Kokoro at 0.8 sets its own speed',
    script: 'The river runs north. It turns east.',
    opts: { engine: 'kokoro', speed: 0.8, voice: 'bm_george' },
    hear: [said('The river runs north. It turns east.')],
  },
  {
    name: 'engine kokoro with a command is never stretched',
    script: 'The river runs north.',
    opts: { engine: 'kokoro', command: COMMAND, speed: 0.8 },
    hear: [said('The river runs north.')],
  },
  {
    name: 'a model at the config speed 1.5 is time-stretched',
    script: 'Fish swim upstream.\nBirds fly south.',
    tts: { model: 'facebook/mms-tts-eng', speed: 1.5 },
    hear: [said('Fish swim upstream. Birds fly south.')],
    resampled: true,
  },
  {
    name: 'timed lines placed, one late, one over its slot',
    cues: [cue(0.5, null, 'The river runs north.'), cue(0.6, 0.9, 'It turns east. Then south.'), cue(4, 4.02, 'Done.')],
    opts: { command: COMMAND },
    hear: [said('The river runs north. It turns east. Then south. Done.')],
  },
  {
    name: 'timed lines with a one-call command from the config',
    cues: [cue(0, null, 'The river runs north.'), cue(3, null, 'It turns east.')],
    tts: { command: COMMAND, oneCall: true, reference: 'conf.wav' },
    hear: [said('The river runs north. It turns east.')],
  },
  {
    name: 'timed lines stretched at 0.8',
    cues: [cue(0, null, 'The river runs north.'), cue(2.5, null, 'It turns east.')],
    opts: { command: COMMAND, speed: 0.8 },
    hear: [said('The river runs north. It turns east.')],
  },
  {
    name: 'timed lines with VoxCPM2, a line apiece',
    cues: [cue(0, null, '[sigh] One here. Two there.'), cue(3, null, 'Last line.')],
    opts: { engine: 'voxcpm', voice: '(bright)' },
    ready: READY,
  },
  {
    name: 'timed lines with Kokoro, tags stripped from the lines',
    cues: [cue(0.25, null, '[laughing] The river runs north.'), cue(3, null, 'It turns east.')],
    opts: { engine: 'kokoro' },
    hear: [said('The river runs north. It turns east.')],
    resampled: true,
  },
  {
    name: 'an engine the skill does not run',
    script: 'Hello there.',
    opts: { engine: 'piper' },
  },
  {
    name: 'a speed out of range',
    script: 'Hello there.',
    opts: { command: COMMAND, speed: 2.5 },
  },
];

// The tone call `k` of a run speaks for `text`, at `speed`.
function signal(text, k, speed) {
  const n = Math.round((2400 + 1200 * text.split(/\s+/).length) / speed);
  const ops = [noise(0, n, 0.00002, 11 + k), tone(200, 240, n - 480, 0.3, 240, 240)];
  return { n, ops, x: render(n, ops) };
}

// The body of a WAV's data chunk. ffmpeg's files carry a LIST chunk before it.
function dataChunk(buf) {
  let at = 12;
  while (at + 8 <= buf.length) {
    const size = buf.readUInt32LE(at + 4);
    if (buf.toString('latin1', at, at + 4) === 'data') return buf.subarray(at + 8, at + 8 + size);
    at += 8 + size + (size & 1);
  }
  throw new Error('take_dispatch: the WAV holds no data chunk');
}

const dir = mkdtempSync(join(tmpdir(), 'dispatch-oracle-'));
const realError = console.error;
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, after = '', replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`take_dispatch: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, () => (replace ?? at) + after);
  }
  const base = pathToFileURL(join(jsDir, '/')).href;
  text = text
    .replaceAll("from './", `from '${base}`)
    .replaceAll("import('./", `import('${base}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text + EXPORTS);
  const speak = await import(pathToFileURL(patched).href);
  const { DEFAULT_MODEL, SMALL_MODEL } = await import(pathToFileURL(join(jsDir, 'voice.mjs')).href);

  const cases = [];
  for (const [k, c] of CASES.entries()) {
    const root = join(dir, `case-${k}`);
    mkdirSync(root, { recursive: true });
    writeFileSync(join(root, 'ref.wav'), 'a recording');
    const local = (v) => (typeof v === 'string' ? v.replaceAll('<dir>', root) : v);
    const masked = (v) => v.replaceAll(root, '<dir>');
    const out = join(root, 'out.wav');
    const script = c.script ?? c.cues.map((q) => q.text).join('\n');
    const opts = Object.fromEntries(Object.entries({ ...c.opts, ...(c.cues ? { cues: c.cues } : {}) }).map(([key, v]) => [key, local(v)]));
    const calls = [];
    const heard = [];
    const messages = [];
    const commands = [];
    let voxcpm = null;
    const queue = [...(c.hear ?? [])];
    globalThis.oracleConfig = c.tts ?? {};
    globalThis.oracleReady = c.ready ?? NOT_READY;
    globalThis.oracleKokoro = {
      KokoroTTS: {
        from_pretrained: async () => ({
          voices: { af_heart: {}, bm_george: {} },
          generate: async (words, { voice, speed }) => {
            const s = signal(words, calls.length, speed);
            calls.push({ text: words, voice, speed, rate: 24000, n: s.n, ops: s.ops });
            return { audio: s.x, sampling_rate: 24000 };
          },
        }),
      },
    };
    globalThis.oraclePipeline = {
      pipeline: async () => async (words) => {
        const s = signal(words, calls.length, 1);
        calls.push({ text: words, voice: null, speed: null, rate: 16000, n: s.n, ops: s.ops });
        return { audio: s.x, sampling_rate: 16000 };
      },
    };
    globalThis.oracleCommand = (o) => commands.push(o);
    globalThis.oracleShell = (_cmd, ctx) => {
      const words = readFileSync(ctx.textFile, 'utf8').replace(/\n$/, '');
      const s = signal(words, calls.length, 1);
      writeFileSync(ctx.raw, wavF32(s.x, 48000));
      calls.push({ text: words, voice: null, speed: null, rate: 48000, n: s.n, ops: s.ops });
      return { status: 0 };
    };
    globalThis.oracleVoxcpm = (said, o, file) => {
      const units = speak.chunks(said, o.lines ? { lines: o.lines } : { whole: true, max: 700 });
      const n = units.length * 48000;
      const ops = [noise(0, n, 0.00002, 31), ...units.map((_, i) => tone(200, i * 48000, 24000, 0.3, 240, 240))];
      writeFileSync(file, speak.oracleWavFile([render(n, ops)], 48000).buf);
      const voiceFile = o.reference ? null : file.replace(/\.[^./]+$/, '') + '.voice.wav';
      const result = {
        duration: n / 48000,
        phrases: units.map((u, i) => ({ text: speak.spoken(u.text), start: i, end: i + 0.5 })),
        voiceFile,
        checks: units.map((u, i) => ({ at: i, text: speak.spoken(u.text), ok: true, heard: u.text, why: '', seed: 7, doubt: [], heat: 0, pauses: [], clicks: [] })),
        problems: [],
      };
      voxcpm = { script: said, opts: { ...o, reference: masked(o.reference) }, n, ops, result: masked(JSON.stringify(result)) };
      return result;
    };
    globalThis.oracleAsrModel = (language) => (language && !/^en/i.test(language) ? SMALL_MODEL : DEFAULT_MODEL);
    globalThis.oracleHear = async (file, { model, language }) => {
      heard.push({ model: model === SMALL_MODEL ? 'small' : 'default', language: language ?? null, samples: fnv1a(wavSamples(readFileSync(file))) });
      return queue.shift() ?? { words: [], looped: false };
    };
    console.error = (line) => messages.push(`${line}`);
    let result = null;
    let error = null;
    try {
      result = await speak.speak(script, out, { ...opts });
    } catch (e) {
      error = e.message;
    } finally {
      console.error = realError;
    }
    const data = result ? dataChunk(readFileSync(out)) : null;
    cases.push({
      name: c.name,
      script,
      opts: { reroll: [], ...c.opts, ...(c.cues ? { cues: c.cues } : {}) },
      tts: c.tts ?? {},
      ready: (c.ready ?? NOT_READY).ok,
      why: (c.ready ?? NOT_READY).why,
      hear: c.hear ?? [],
      resampled: !!c.resampled,
      calls,
      commands: commands.map((o) => ({ ...o, reference: masked(o.reference) })),
      voxcpm,
      messages,
      heard,
      out: data && { length: data.length / 2, pcm: fnv1aBytes(data) },
      error,
      result: result && masked(JSON.stringify(result)),
    });
  }
  process.stdout.write(`${JSON.stringify({ cases }, null, 1)}\n`);
} finally {
  console.error = realError;
  for (const k of ['oracleConfig', 'oracleReady', 'oracleKokoro', 'oraclePipeline', 'oracleCommand', 'oracleShell', 'oracleVoxcpm', 'oracleAsrModel', 'oracleHear']) delete globalThis[k];
  rmSync(dir, { recursive: true, force: true });
}
