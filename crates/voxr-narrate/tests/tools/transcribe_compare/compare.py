#!/usr/bin/env python3
"""Compares two words files of the same take: their text and their word times.

Usage:
    python3 compare.py A.words.json B.words.json
    python3 compare.py A_DIR B_DIR

A is the reference. Given two directories, every *.words.json in both is
compared, then all their matched words are pooled.

Words are matched by text: each word is lowercased and kept to its letters
and digits, and difflib.SequenceMatcher aligns the two token lists. A word
with no letter or digit is left out. The text WER is the word edit
distance from A to B over A's token count. Each time difference is B's time
minus A's, in seconds. Output is plain text on stdout. Standard library
only.
"""

import difflib
import json
import math
import sys
from pathlib import Path

GAP = 0.1
TOP = 10


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(text):
    return "".join(c for c in text.lower() if c.isalnum())


def tokens(words):
    """The normalized tokens and the word each came from."""
    out = []
    for w in words:
        t = norm(w["text"])
        if t:
            out.append((t, w))
    return out


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def time(v):
    """A time as a float; a missing or null time is NaN."""
    return math.nan if v is None else float(v)


def compare(a_file, b_file):
    """Text counts and the matched word pairs of two words files."""
    a, b = load(a_file), load(b_file)
    ta, tb = tokens(a["words"]), tokens(b["words"])
    sa, sb = [t for t, _ in ta], [t for t, _ in tb]
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    pairs = []
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            wa, wb = ta[block.a + k][1], tb[block.b + k][1]
            pairs.append(
                {
                    "file": Path(a_file).name,
                    "text": wa["text"],
                    "a": (time(wa["start"]), time(wa["end"])),
                    "b": (time(wb["start"]), time(wb["end"])),
                }
            )
    return {
        "a_model": a.get("model", "?"),
        "b_model": b.get("model", "?"),
        "a_words": len(a["words"]),
        "b_words": len(b["words"]),
        "a_tokens": sa,
        "b_tokens": sb,
        "edits": edit_distance(sa, sb),
        "a_duration": a.get("duration"),
        "b_duration": b.get("duration"),
        "a_looped": a.get("looped"),
        "b_looped": b.get("looped"),
        "pairs": pairs,
    }


def percentile(sorted_values, q):
    """Nearest-rank percentile of an ascending list."""
    if not sorted_values:
        return math.nan
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def median(sorted_values):
    n = len(sorted_values)
    if n == 0:
        return math.nan
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def diff_stats(label, diffs):
    diffs = [d for d in diffs if not math.isnan(d)]
    absd = sorted(abs(d) for d in diffs)
    n = len(absd)
    if n == 0:
        return f"  {label}: no matched words with times"
    mean_signed = sum(diffs) / n
    mean_abs = sum(absd) / n
    over = sum(1 for d in absd if d > GAP)
    return (
        f"  {label}: |diff| mean {mean_abs:.3f}  median {median(absd):.3f}  "
        f"p90 {percentile(absd, 90):.3f}  max {absd[-1]:.3f}  "
        f"> {GAP}s {over}/{n}  (signed mean B-A {mean_signed:+.3f})"
    )


def timing_report(pairs):
    lines = [
        f"matched words: {len(pairs)}",
        diff_stats("start", [p["b"][0] - p["a"][0] for p in pairs]),
        diff_stats("end  ", [p["b"][1] - p["a"][1] for p in pairs]),
    ]

    def gap(p):
        ds, de = p["b"][0] - p["a"][0], p["b"][1] - p["a"][1]
        worst = max(abs(ds) if not math.isnan(ds) else -1, abs(de) if not math.isnan(de) else -1)
        return worst, ds, de

    ranked = sorted(pairs, key=lambda p: gap(p)[0], reverse=True)[:TOP]
    if ranked:
        lines.append(f"largest {len(ranked)} timing gaps (A start-end | B start-end | dstart dend):")
        for p in ranked:
            _, ds, de = gap(p)
            (a0, a1), (b0, b1) = p["a"], p["b"]
            lines.append(
                f"  {p['file']}  {p['text']!r:<18} A {a0:7.3f}-{a1:7.3f} | "
                f"B {b0:7.3f}-{b1:7.3f} | {ds:+.3f} {de:+.3f}"
            )
    return lines


def file_report(a_file, b_file, r):
    wer = r["edits"] / len(r["a_tokens"]) if r["a_tokens"] else math.nan
    lines = [
        f"== {Path(a_file).name}",
        f"A: {a_file}  model {r['a_model']}  {r['a_words']} words  "
        f"duration {r['a_duration']}  looped {r['a_looped']}",
        f"B: {b_file}  model {r['b_model']}  {r['b_words']} words  "
        f"duration {r['b_duration']}  looped {r['b_looped']}",
        f"text WER (A reference): {wer:.4f}  ({r['edits']} edits over {len(r['a_tokens'])} tokens)",
    ]
    if r["edits"]:
        sm = difflib.SequenceMatcher(None, r["a_tokens"], r["b_tokens"], autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                a_text = " ".join(r["a_tokens"][i1:i2]) or "-"
                b_text = " ".join(r["b_tokens"][j1:j2]) or "-"
                lines.append(f"  {op:<7} A[{i1}:{i2}] {a_text!r} -> B[{j1}:{j2}] {b_text!r}")
    lines += timing_report(r["pairs"])
    return lines


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    a, b = Path(argv[1]), Path(argv[2])
    if a.is_dir() != b.is_dir():
        print("compare.py: pass two words files or two directories, not one of each", file=sys.stderr)
        return 2
    if a.is_dir():
        names = sorted({p.name for p in a.glob("*.words.json")} & {p.name for p in b.glob("*.words.json")})
        if not names:
            print(f"compare.py: {a} and {b} share no *.words.json", file=sys.stderr)
            return 2
        pairs_list = [(a / n, b / n) for n in names]
    else:
        pairs_list = [(a, b)]
    pooled, edits, ref_tokens = [], 0, 0
    for a_file, b_file in pairs_list:
        r = compare(a_file, b_file)
        print("\n".join(file_report(a_file, b_file, r)))
        print()
        pooled += r["pairs"]
        edits += r["edits"]
        ref_tokens += len(r["a_tokens"])
    if len(pairs_list) > 1:
        wer = edits / ref_tokens if ref_tokens else math.nan
        print(f"== all {len(pairs_list)} files")
        print(f"text WER (A reference): {wer:.4f}  ({edits} edits over {ref_tokens} tokens)")
        print("\n".join(timing_report(pooled)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
