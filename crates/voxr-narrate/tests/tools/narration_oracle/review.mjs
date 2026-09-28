// Reference outputs for `review` in `voxr_narrate`: runs the skill's exported `review` over the corpus beside
// this file and prints one JSON document to stdout. `review` needs no patch: it is exported as is and reads no
// machine state.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, paths.mjs, voice.mjs> \
//          node review.mjs > ../../fixtures/narration_review_oracle.json
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('review: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, paths.mjs and voice.mjs.');
  process.exit(2);
}
const speak = await import(pathToFileURL(join(jsDir, 'speak.mjs')).href);

const corpus = join(dirname(fileURLToPath(import.meta.url)), 'corpus');
const read = (rel) => readFileSync(join(corpus, rel), 'utf8');
const list = (dir) => readdirSync(join(corpus, dir)).sort().map((f) => `${dir}/${f}`);

// The general narration corpus (English and Malay beats, delivery notes, tags, pauses, CJK, whitespace, a script
// with no end punctuation) plus a corpus of its own for what `review` looks for specifically: flags, file names,
// addresses, URLs, emails and handles; bare digits in and outside English; long runs with and without a joining
// word; a line missing its end punctuation; and questions with and without a delivery note, alone or sharing a
// line.
const files = [...list('scripts'), ...list('review')];

// Every option combination `review` takes: no language (English), Malay, an English region tag (still English),
// a language it does not special-case, and `notes` on and off.
const optionSets = [
  {},
  { notes: true },
  { language: 'ms' },
  { language: 'ms', notes: true },
  { language: 'en-US' },
  { language: 'fr' },
];

const cases = [];
for (const file of files) {
  const script = read(file);
  for (const options of optionSets) {
    let output;
    try {
      output = speak.review(script, options);
    } catch (e) {
      output = { error: e.message };
    }
    cases.push({ file, options, output });
  }
}

process.stdout.write(`${JSON.stringify({ cases }, null, 1)}\n`);
