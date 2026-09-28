// Reference outputs for the take loop's engine choice in `voxr_narrate`: `defaultEngine`, exported as is, and
// the engine/model/voice/speed resolution at the top of `speak()`, run through a patched copy of speak.mjs that
// captures the resolution and stops short of running any engine, ffmpeg or the recursive time-stretch/cues
// branches. Prints one JSON document to stdout.
//
// The patch lets the oracle fix the user's config and VoxCPM2's readiness, each of which otherwise reads real
// machine state (a config file, the GPU, uv), so every case here is reproducible off this machine. Every case
// avoids the speed!==1-with-a-non-Kokoro-non-VoxCPM2-engine and the `cues` branches, which recurse or place timed
// lines rather than choose an engine: they are outside what this oracle checks.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node engine_choice.mjs > ../../fixtures/narration_engine_choice_oracle.json
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('engine_choice: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}

// Each patch names a line that must occur exactly once in speak.mjs. The first two let the oracle override state
// `config()` and `defaultEngine`'s `voxcpmReady()` otherwise read from the real machine. The next two record the
// tag-strip notice and the strip itself into globals a later capture reads. The last three stop the call, with a
// capture, right before it runs VoxCPM2, the user's command, Kokoro or another Hugging Face model.
const PATCHES = [
  {
    at: '  const tts = config().tts ?? {};\n',
    replace: '  const tts = (globalThis.oracleConfig ?? config()).tts ?? {};\n',
  },
  {
    at: '  const vox = voxcpmReady();\n',
    replace: '  const vox = globalThis.oracleVoxcpmReady ?? voxcpmReady();\n',
  },
  {
    at: "    if (!opts.stretched && !opts.lines) console.error('speak: only VoxCPM2 voices tags like [laughing] or [sigh]. This engine speaks the script without them.');\n",
    replace:
      "    if (!opts.stretched && !opts.lines) { globalThis.oracleLogged = true; console.error('speak: only VoxCPM2 voices tags like [laughing] or [sigh]. This engine speaks the script without them.'); }\n",
  },
  {
    at: '    script = withoutTags(script);\n',
    replace: '    globalThis.oracleStripped = true;\n    script = withoutTags(script);\n',
  },
  {
    at: "  const r = await voxcpm(script, { voice, reference: opts.reference ?? tts.reference ?? '', style: opts.style ?? tts.style ?? '', hifi: opts.hifi ?? tts.hifi ?? false, cfg: opts.cfg ?? tts.cfg ?? 1.6, steps: opts.steps ?? tts.steps ?? 16, device: opts.device, seed: opts.seed, configSeed: tts.seed, language: opts.language, reroll: opts.reroll ?? [], lines: !!opts.lines }, out);\n",
    replace:
      "  if (globalThis.oracleCapture) { globalThis.oracleCapture({ branch: 'voxcpm', voice, speed, strippedTags: !!globalThis.oracleStripped, loggedNotice: !!globalThis.oracleLogged }); throw new Error('__oracle_stop__'); }\n" +
      "  const r = await voxcpm(script, { voice, reference: opts.reference ?? tts.reference ?? '', style: opts.style ?? tts.style ?? '', hifi: opts.hifi ?? tts.hifi ?? false, cfg: opts.cfg ?? tts.cfg ?? 1.6, steps: opts.steps ?? tts.steps ?? 16, device: opts.device, seed: opts.seed, configSeed: tts.seed, language: opts.language, reroll: opts.reroll ?? [], lines: !!opts.lines }, out);\n",
  },
  {
    at: '  if (cmd) {\n',
    replace:
      "  if (cmd) {\n    if (globalThis.oracleCapture) { globalThis.oracleCapture({ branch: 'command', command: cmd, speed, strippedTags: !!globalThis.oracleStripped, loggedNotice: !!globalThis.oracleLogged }); throw new Error('__oracle_stop__'); }\n",
  },
  {
    at: '  } else {\n    const r = /kokoro/i.test(model) ? await kokoro(script, { model, voice, speed }) : await transformersModel(script, { model });\n',
    replace:
      "  } else {\n    if (globalThis.oracleCapture) { globalThis.oracleCapture({ branch: 'model', model, voice, kokoroModel: /kokoro/i.test(model), speed, strippedTags: !!globalThis.oracleStripped, loggedNotice: !!globalThis.oracleLogged }); throw new Error('__oracle_stop__'); }\n" +
      '    const r = /kokoro/i.test(model) ? await kokoro(script, { model, voice, speed }) : await transformersModel(script, { model });\n',
  },
];

const dir = mkdtempSync(join(tmpdir(), 'engine-choice-oracle-'));
try {
  const source = join(jsDir, 'speak.mjs');
  let text = readFileSync(source, 'utf8');
  for (const { at, replace } of PATCHES) {
    const count = text.split(at).length - 1;
    if (count !== 1) throw new Error(`engine_choice: speak.mjs holds ${JSON.stringify(at)} ${count} times, not once`);
    text = text.replace(at, replace);
  }
  text = text
    .replaceAll("from './", `from '${pathToFileURL(join(jsDir, '/')).href}`)
    .replaceAll('import.meta.url', JSON.stringify(pathToFileURL(source).href));
  const patched = join(dir, 'speak_patched.mjs');
  writeFileSync(patched, text);
  const speak = await import(pathToFileURL(patched).href);

  const cases = [];
  const add = async (fn, input, run) => {
    let output;
    try {
      output = await run();
    } catch (e) {
      output = { error: e.message };
    }
    cases.push({ fn, input, output });
  };

  // defaultEngine: the Kokoro voice shape, VoxCPM2 readiness and a reference, and the non-English error, over the
  // readiness `voxcpmReady()` otherwise reports for real.
  const readiness = [
    { ok: false, why: 'needs an NVIDIA GPU with 8 GB of memory (none found)' },
    { ok: false, why: 'needs 8 GB of GPU memory (Tiny Card has 4 GB)' },
    { ok: true, why: 'Big Card, 24 GB' },
  ];
  const defaultEngineCases = [
    { voice: 'af_heart' },
    { voice: 'bm_george' },
    { voice: 'AF_HEART' },
    { voice: 'af_' },
    { voice: 'not-a-kokoro-voice' },
    { voice: undefined },
    { reference: '/clip.wav' },
    { reference: '' },
    { language: 'ms' },
    { language: 'ms-MY' },
    { language: 'en' },
    { language: 'EN-us' },
    { language: '' },
    { language: undefined },
    { voice: 'x_x', language: 'ms' },
  ];
  for (const c of defaultEngineCases) {
    for (const ready of readiness) {
      globalThis.oracleVoxcpmReady = ready;
      await add('defaultEngine', { opts: c, readiness: ready }, () => speak.defaultEngine(c));
    }
  }
  globalThis.oracleVoxcpmReady = undefined;

  // The engine/model/voice/speed resolution at the top of speak(): every branch (voxcpm, command, Kokoro, another
  // model), the config fallbacks, the invalid --engine and --speed errors, and the tag-strip notice.
  const capture = async (script, opts, tts, ready) => {
    globalThis.oracleConfig = { tts };
    globalThis.oracleVoxcpmReady = ready;
    globalThis.oracleStripped = false;
    globalThis.oracleLogged = false;
    let captured = null;
    globalThis.oracleCapture = (c) => {
      captured = c;
    };
    try {
      const r = await speak.speak(script, join(dir, 'out.wav'), opts);
      throw new Error(`engine_choice: speak ran to completion instead of stopping at the oracle capture (${JSON.stringify(r)})`);
    } catch (e) {
      if (e.message === '__oracle_stop__') return captured;
      throw e;
    } finally {
      globalThis.oracleCapture = undefined;
    }
  };

  const notReady = { ok: false, why: 'needs an NVIDIA GPU with 8 GB of memory (none found)' };
  const ready = { ok: true, why: 'Big Card, 24 GB' };

  const resolveCases = [
    // Nothing set anywhere: defaultEngine picks Kokoro.
    { script: 'Hello.', opts: {}, tts: {}, ready: notReady },
    // Nothing set anywhere, but VoxCPM2 is ready: defaultEngine picks VoxCPM2.
    { script: 'Hello.', opts: {}, tts: {}, ready },
    // opts.engine set directly to each branch.
    { script: 'Hello.', opts: { engine: 'voxcpm', voice: 'designed' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: 'voxcpm' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro', voice: 'bm_george' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: '' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: 'piper' }, tts: {}, ready: notReady },
    // The config sets the engine instead of opts.
    { script: 'Hello.', opts: {}, tts: { engine: 'voxcpm', voice: 'from-config' }, ready: notReady },
    { script: 'Hello.', opts: {}, tts: { engine: 'kokoro', voice: 'from-config' }, ready: notReady },
    // A command, from opts and from the config, and opts winning over the config.
    { script: 'Hello.', opts: { command: 'piper --text {text} --out {out}' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: {}, tts: { command: 'xtts --text {text} --out {out}' }, ready: notReady },
    { script: 'Hello.', opts: { command: 'a --out {out}' }, tts: { command: 'b --out {out}' }, ready: notReady },
    // A model naming Kokoro, and one that does not, from opts and from the config.
    { script: 'Hello.', opts: { model: 'onnx-community/Kokoro-82M-v1.0-ONNX' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { model: 'facebook/mms-tts-eng' }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: {}, tts: { model: 'KOKORO-mini' }, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro' }, tts: { model: 'some/other-model' }, ready: notReady },
    // opts.model set with opts.engine also set to something other than kokoro/voxcpm: engine wins the invalid check.
    { script: 'Hello.', opts: { engine: 'piper', model: 'x' }, tts: {}, ready: notReady },
    // Model over command when both are set on opts.
    { script: 'Hello.', opts: { command: 'a --out {out}', model: 'facebook/mms-tts-eng' }, tts: {}, ready: notReady },
    // Voice fallback: opts, then config, then DEFAULT_VOICE.
    { script: 'Hello.', opts: { engine: 'kokoro', voice: 'af_bella' }, tts: { voice: 'af_sky' }, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro' }, tts: { voice: 'af_sky' }, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro' }, tts: {}, ready: notReady },
    // Invalid --speed, at each side of the range and with a non-number.
    { script: 'Hello.', opts: { engine: 'kokoro', speed: 0.4 }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro', speed: 2.1 }, tts: {}, ready: notReady },
    // `NaN` cannot round-trip through JSON: the string 'NaN' produces the
    // same coercion (`Number('NaN')` is `NaN`) and the same message.
    { script: 'Hello.', opts: { engine: 'kokoro', speed: 'NaN' }, tts: {}, ready: notReady },
    // Valid --speed at the edges and from the config, kept off the recursive stretch by naming Kokoro or VoxCPM2.
    { script: 'Hello.', opts: { engine: 'kokoro', speed: 0.5 }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: { engine: 'kokoro', speed: 2 }, tts: {}, ready: notReady },
    { script: 'Hello.', opts: {}, tts: { engine: 'kokoro', speed: 1.4 }, ready: notReady },
    { script: 'Hello.', opts: { engine: 'voxcpm', speed: 1.8 }, tts: {}, ready: notReady },
    // The non-English defaultEngine error, surfacing through speak() itself.
    { script: 'Hello.', opts: { language: 'ms' }, tts: {}, ready: notReady },
    // Tags: stripped and logged off VoxCPM2, kept and silent on VoxCPM2, and silent when stretched or timed.
    { script: '[sigh] Hello.', opts: { engine: 'kokoro' }, tts: {}, ready: notReady },
    { script: '[sigh] Hello.', opts: { engine: 'voxcpm' }, tts: {}, ready: notReady },
    { script: '[sigh] Hello.', opts: { engine: 'kokoro', stretched: true }, tts: {}, ready: notReady },
    { script: '[sigh] Hello.', opts: { engine: 'kokoro', lines: true }, tts: {}, ready: notReady },
    { script: 'No tags here.', opts: { engine: 'kokoro' }, tts: {}, ready: notReady },
  ];
  for (const c of resolveCases) {
    await add('resolveSpeak', { script: c.script, opts: c.opts, tts: c.tts, readiness: c.ready }, () => capture(c.script, c.opts, c.tts, c.ready));
  }

  process.stdout.write(`${JSON.stringify({ cases }, null, 1)}\n`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
