// WAV files and file listings the take oracles share: the stub model's 32-bit float takes, the samples a WAV holds as
// ffmpeg decodes them, and every file under a directory with the hash of its bytes.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fnv1aBytes } from './signals.mjs';

// A 32-bit float WAV of `x`, as the stub model writes its takes.
export function wavF32(x, rate) {
  const buf = Buffer.alloc(44 + x.length * 4);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + x.length * 4, 4);
  buf.write('WAVEfmt ', 8);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(3, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(rate, 24);
  buf.writeUInt32LE(rate * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(32, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(x.length * 4, 40);
  Buffer.from(x.buffer, x.byteOffset, x.byteLength).copy(buf, 44);
  return buf;
}

// The samples a 16-bit or 32-bit float WAV holds, as ffmpeg decodes them: a 16-bit value v reads as v / 32768.
export function wavSamples(buf) {
  const format = buf.readUInt16LE(20);
  const data = buf.subarray(44);
  if (format === 3) return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.length));
  return Float32Array.from({ length: data.length / 2 }, (_, k) => data.readInt16LE(2 * k) / 32768);
}

// Every file under `root`, by its path from `root`, with the FNV-1a hash of its bytes.
export function listing(root) {
  const out = [];
  const walk = (d) => {
    for (const name of readdirSync(d).sort()) {
      const p = join(d, name);
      if (statSync(p).isDirectory()) walk(p);
      else out.push({ path: relative(root, p), bytes: fnv1aBytes(readFileSync(p)) });
    }
  };
  walk(root);
  return out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}
