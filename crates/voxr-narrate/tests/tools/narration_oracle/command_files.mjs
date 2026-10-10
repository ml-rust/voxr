// What the command oracles share: the skill's `speakCmd` and `transcribeCmd`, taken from video.mjs into a module of
// their own over patched copies of the skill's modules, a scenario directory's paths shown as `<dir>`, and every file
// a scenario leaves.
import { cpSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fnv1aBytes } from './signals.mjs';

// The source of the top-level `async function NAME(` in `src`, through its closing brace at the line start.
function functionSource(src, name) {
  const start = src.indexOf(`\nasync function ${name}(`);
  if (start < 0) throw new Error(`video.mjs has no async function ${name}.`);
  const end = src.indexOf('\n}\n', start + 1);
  if (end < 0) throw new Error(`video.mjs: ${name} has no closing brace at a line start.`);
  return src.slice(start + 1, end + 2);
}

// Applies each patch to `text`: the line `at` must occur exactly once, and is replaced by `replace` or followed by
// `after`.
function patch(text, file, patches) {
  for (const p of patches) {
    const n = text.split(p.at).length - 1;
    if (n !== 1) throw new Error(`${file}: the patched line occurs ${n} times, not once: ${p.at.trim()}`);
    text = text.replace(p.at, p.replace ?? `${p.at}${p.after}`);
  }
  return text;
}

const BLAZR_OFF = { at: 'export function runBlazr(cmd, argv) {\n', after: '  return null;\n' };

// Copies the skill's modules (`jsDir`) into `work/lib`, patched per file, and writes `work/cmd.mjs`: video.mjs's
// imports, then its `speakCmd` and `transcribeCmd`, exported with `ManifestError` and `loadManifest`. `mainLines` must
// each occur in video.mjs exactly once: the part of `main` an oracle replays. Returns the module's path.
export function commandModule(jsDir, work, patches, mainLines = []) {
  const lib = join(work, 'lib');
  cpSync(jsDir, lib, { recursive: true });
  // The oracles record the skill's own runner: a blazr on PATH would otherwise take the command.
  const all = { ...patches, 'blazr.mjs': [...(patches['blazr.mjs'] ?? []), BLAZR_OFF] };
  for (const [file, list] of Object.entries(all)) {
    const path = join(lib, file);
    writeFileSync(path, patch(readFileSync(path, 'utf8'), file, list));
  }
  const video = readFileSync(join(jsDir, '..', 'video.mjs'), 'utf8');
  for (const line of mainLines) {
    const n = video.split(line).length - 1;
    if (n !== 1) throw new Error(`video.mjs: a line the oracle replays occurs ${n} times, not once: ${line.trim()}`);
  }
  const imports = video.split('\n').filter((l) => l.startsWith('import ')).join('\n');
  const body = [functionSource(video, 'speakCmd'), functionSource(video, 'transcribeCmd')].join('\n');
  const module = `${imports}\n\n${body}\nexport { speakCmd, transcribeCmd, ManifestError, loadManifest };\n`;
  const path = join(work, 'cmd.mjs');
  writeFileSync(path, module);
  return path;
}

// `text` with every occurrence of `dir` written as `<dir>`.
export const normalize = (text, dir) => text.split(dir).join('<dir>');

// Every file under `dir` by its path from `dir`: a WAV as the FNV-1a hash of its bytes, any other file as its text
// with `dir` written as `<dir>`.
export function scenarioFiles(dir) {
  const out = [];
  const walk = (d) => {
    for (const name of readdirSync(d).sort()) {
      const p = join(d, name);
      if (statSync(p).isDirectory()) walk(p);
      else {
        const bytes = readFileSync(p);
        const path = relative(dir, p);
        out.push(path.endsWith('.wav') ? { path, bytes: fnv1aBytes(bytes) } : { path, text: normalize(bytes.toString('utf8'), dir) });
      }
    }
  };
  walk(dir);
  return out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}
