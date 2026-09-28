// Reads narration takes into timed words with the narration skill's own transcribe (transformers.js Whisper, q8
// weights, ONNX Runtime), unchanged, and writes each as a words file: the JSON the skill's transcribe command writes,
// without a script. Its counterpart is the transcribe_take example; compare.py compares the two.
//
// The model is --model, else the one the skill picks for the language (asrModel). The skill installs its speech
// runtime on first use and fetches the model into its model folder; both are its own, from its config.
//
// Usage: NARRATION_JS_DIR=<directory holding voice.mjs> node skill_transcribe.mjs [--model ID] [--language xx]
//          [--out DIR] AUDIO...
// With --out, the words file of take.wav is DIR/take.words.json. Without it, each words file goes to stdout.
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const USAGE = 'usage: NARRATION_JS_DIR=<dir> node skill_transcribe.mjs [--model ID] [--language xx] [--out DIR] AUDIO...';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('skill_transcribe: NARRATION_JS_DIR is not set. Set it to the directory that holds voice.mjs.');
  process.exit(2);
}

const opts = { model: null, language: null, out: null, files: [] };
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const arg = argv[i];
  const value = () => {
    if (i + 1 >= argv.length) {
      console.error(`skill_transcribe: ${arg} needs a value. ${USAGE}`);
      process.exit(2);
    }
    return argv[++i];
  };
  if (arg === '--model') opts.model = value();
  else if (arg === '--language') opts.language = value();
  else if (arg === '--out') opts.out = value();
  else if (arg.startsWith('--')) {
    console.error(`skill_transcribe: unknown flag ${arg}. ${USAGE}`);
    process.exit(2);
  } else opts.files.push(arg);
}
if (!opts.files.length) {
  console.error(`skill_transcribe: no audio file given. ${USAGE}`);
  process.exit(2);
}

const { transcribe, asrModel } = await import(pathToFileURL(join(resolve(jsDir), 'voice.mjs')).href);
const model = opts.model ?? asrModel(opts.language);
let failed = 0;
for (const f of opts.files) {
  const file = resolve(f);
  const started = performance.now();
  try {
    const t = await transcribe(file, { model, language: opts.language ?? undefined });
    const text = JSON.stringify(t, null, 1);
    const took = ((performance.now() - started) / 1000).toFixed(1);
    const what = `${file} (${t.duration.toFixed(2)}s, ${t.words.length} words${t.looped ? ', looped' : ''})`;
    if (opts.out) {
      mkdirSync(opts.out, { recursive: true });
      const dest = join(opts.out, `${basename(file).replace(/\.[^.]+$/, '')}.words.json`);
      writeFileSync(dest, text);
      console.error(`${what} -> ${dest} in ${took}s`);
    } else {
      console.log(text);
      console.error(`${what} in ${took}s`);
    }
  } catch (e) {
    console.error(`skill_transcribe: ${file}: ${e.message}`);
    failed++;
  }
}
if (failed) {
  console.error(`skill_transcribe: ${failed} of ${opts.files.length} files failed`);
  process.exit(1);
}
