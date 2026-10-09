#!/usr/bin/env python3
"""Reference ground truth for VoxCPM2's text tokenizer, for EXACT integer match.

The model never calls `tokenizer.encode()` or `tokenizer.__call__()`. Every
text-tokenization call site in `voxcpm2.py` (`_make_ref_prefix` and friends,
e.g. the `text_token = torch.LongTensor(self.text_tokenizer(text))` calls
mirrored by `make_prefill_fixture.py` and `make_step_fixture.py`) goes
through `self.text_tokenizer`, which is
`mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(ckpt_dir))`
(`voxcpm/model/voxcpm2.py:182`). That wrapper's `__call__` is exactly
`tokenizer.tokenize(text)` -> `convert_tokens_to_ids(tokens)` -- NO BOS, since
BOS injection lives in the tokenizer's `TemplateProcessing` post-processor,
which only `encode()`/`__call__` on the *base* `LlamaTokenizerFast` invoke,
and the wrapper never calls those.

## Version trap this fixture exposes

`LlamaTokenizerFast.from_pretrained()` does NOT hand back the tokenizer as
declared in the checkpoint's `tokenizer.json` unmodified. On disk that file
declares `normalizer: Sequence[Prepend("▁"), Replace(" " -> "▁")]`
and `pre_tokenizer: null`. Under transformers 5.x (verified: 5.15.1),
`from_pretrained` REWRITES the backend tokenizer to
`pre_tokenizer: Metaspace(prepend_scheme="always", split=false)` and
`normalizer: null`. Under transformers 4.57.3 no such rewrite happens.

The two conventions agree everywhere EXCEPT around whitespace runs (a
leading/trailing space, or an interior run of 2+ spaces) -- see the
`raw != transformers` / `raw != transformers (normalized)` summary this
script prints. Any Rust port that reads `tokenizer.json` directly (as
`splintr`'s generic HF reader does, see `boostr/examples/voxcpm_tokenizer_check.rs`)
reproduces the RAW-file convention, not whichever transformers version
happened to be installed when a fixture was captured. So the raw-file path
is the only version-independent ground truth, and it is what `ids_plain_{i}`
now records. `ids_transformers_{i}` records the transformers path actually
executed by the reference pipeline in THIS venv, purely for comparison --
the Rust gate must never be checked against it.

`ids_norm_{i}` records the raw path on whitespace-collapsed
(`re.sub(r"\\s+", " ", text).strip()`) text, and this script asserts in-line
that the transformers path on the SAME normalized text gives identical ids.
That convergence is what lets a Rust caller normalize input before encoding
and be correct under either tokenizer.json-reading convention; if the
assertion ever fails the two conventions have diverged on more than
whitespace and this fixture's premise needs re-examination.

The checkpoint also ships `tokenization_voxcpm2.py` (`auto_map` ->
`VoxCPM2Tokenizer`, used by external consumers such as vLLM through
`AutoTokenizer`, NOT by the reference model itself, which loads a plain
`LlamaTokenizerFast` directly at `voxcpm2.py:1121`). `VoxCPM2Tokenizer`
precomputes a `{token_id: [char_id, ...]}` split map at construction time
for every vocab entry whose stripped form is >= 2 characters and entirely
CJK (`\\u4e00-\\u9fff`, `\\u3400-\\u4dbf`, `\\uf900-\\ufaff`,
`\\U00020000-\\U0002a6df`), and its `_expand_ids` applies that map to a
plain id list. This is algorithmically the same split
`mask_multichar_chinese_tokens` performs (on token strings, pre-conversion,
via a `len(token) >= 2 and all("\\u4e00"<=c<="\\u9fff" for c in token)`
membership set -- narrower, BMP Han block only), so for any text whose
multi-char CJK vocab tokens fall in that block, `ids_expanded_{i}` computed
by `VoxCPM2Tokenizer._expand_ids` on top of `ids_plain_{i}` reproduces what
`self.text_tokenizer` would have produced. `_expand_ids` is reachable
directly as an instance method, no monkeypatching required, so it is
captured for every text below, on top of the raw-file `ids_plain_{i}`. For
Latin-script text the two are identical by construction: no Latin token is
CJK.

Usage:
    python make_tokenizer_fixture.py <ckpt-dir> <outdir>

Writes `<outdir>/tokenizer_fixture.safetensors` (int64 id tensors, four per
text -- `ids_plain_{i}`, `ids_transformers_{i}`, `ids_expanded_{i}`,
`ids_norm_{i}` -- for the Rust gate, which reads `ids_plain_{i}` only) and
`<outdir>/tokenizer_fixture.json` (text, normalized text, label, and every
id list per text, plus library versions and the raw file's declared
normalizer/pre_tokenizer node types, for humans).
"""
import json
import re
import sys
from pathlib import Path

import torch
import tokenizers
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import LlamaTokenizerFast

ckpt_dir, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

# --- Texts, covering every case the Rust tokenizer must match ---------------
TEXTS = [
    # 1. Exact target text from make_prefill_fixture.py / make_step_fixture.py.
    ("prefill_fixture_target_text",
     "Saya dah cuba macam-macam, tapi honestly it doesn't make sense."),
    # 2-3. More Malay+English code-switching sentences.
    ("code_switch_2",
     "Actually aku rasa this project boleh jalan, just kena test properly."),
    ("code_switch_3",
     "Meeting tu at 3pm, can you confirm ke tak?"),
    # 4. Pure English.
    ("pure_english",
     "The quick brown fox jumps over the lazy dog."),
    # 5. Pure Malay.
    ("pure_malay",
     "Selamat pagi, saya harap semua orang sihat dan gembira hari ini."),
    # 6. Punctuation, digits, apostrophe.
    ("punctuation_digits_apostrophe",
     "It's 2026, harga naik 15% lagi!"),
    # 7. CJK, for the multi-char split path (irrelevant to Malay/English text,
    #    but this is the only case where ids_plain and ids_expanded can differ).
    ("cjk_multichar_split",
     "你好世界，今天天气很好。"),
    # 8. Leading/trailing space and an internal double space, to pin the
    #    Prepend("▁") + Replace(" " -> "▁") normalizer sequence, and the
    #    exact case where the raw-file and transformers conventions diverge.
    ("leading_trailing_double_space",
     " double  space and edges "),
]


def normalize(text: str) -> str:
    """Collapse whitespace runs and trim -- the convention-independent form."""
    return re.sub(r"\s+", " ", text).strip()


def describe_node(node):
    """Node-type name for a tokenizer.json `normalizer`/`pre_tokenizer` entry,
    expanding a `Sequence`'s children so composite pipelines are visible."""
    if node is None:
        return None
    node_type = node.get("type")
    if node_type == "Sequence":
        children = [c.get("type") for c in node.get("normalizers", node.get("pre_tokenizers", []))]
        return f"Sequence{children}"
    return node_type


# --- Raw tokenizer.json, exactly as any faithful reader (incl. splintr's HF
# reader used by boostr/examples/voxcpm_tokenizer_check.rs) would load it ----
tokenizer_json_path = ckpt_dir / "tokenizer.json"
raw_tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
raw_json = json.loads(tokenizer_json_path.read_text())
raw_normalizer_desc = describe_node(raw_json.get("normalizer"))
raw_pre_tokenizer_desc = describe_node(raw_json.get("pre_tokenizer"))
print(f"raw tokenizer.json normalizer: {raw_normalizer_desc}", file=sys.stderr)
print(f"raw tokenizer.json pre_tokenizer: {raw_pre_tokenizer_desc}", file=sys.stderr)

# --- transformers path, exactly as voxcpm2.py:1121 loads and calls it -------
tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_dir))
print(f"tokenizer class actually used by the model: {type(tokenizer).__name__}",
      file=sys.stderr)
print(f"vocab size: {tokenizer.vocab_size}", file=sys.stderr)
backend = tokenizer.backend_tokenizer
backend_json = json.loads(backend.to_str())
print(f"transformers backend_tokenizer normalizer: "
      f"{describe_node(backend_json.get('normalizer'))}", file=sys.stderr)
print(f"transformers backend_tokenizer pre_tokenizer: "
      f"{describe_node(backend_json.get('pre_tokenizer'))}", file=sys.stderr)
print(f"transformers.__version__: {transformers.__version__}", file=sys.stderr)
print(f"tokenizers.__version__: {tokenizers.__version__}", file=sys.stderr)

# --- VoxCPM2Tokenizer, for the standalone _expand_ids CJK-split path ---------
sys.path.insert(0, str(ckpt_dir))
from tokenization_voxcpm2 import VoxCPM2Tokenizer  # noqa: E402

vox_tokenizer = VoxCPM2Tokenizer.from_pretrained(str(ckpt_dir))
print(f"VoxCPM2Tokenizer._expand_ids reachable directly: "
      f"{callable(getattr(vox_tokenizer, '_expand_ids', None))}", file=sys.stderr)

tensors = {}
records = {}

for i, (label, text) in enumerate(TEXTS):
    text_norm = normalize(text)

    ids_plain = raw_tokenizer.encode(text, add_special_tokens=False).ids
    ids_norm = raw_tokenizer.encode(text_norm, add_special_tokens=False).ids

    transformers_tokens = tokenizer.tokenize(text)
    ids_transformers = tokenizer.convert_tokens_to_ids(transformers_tokens)
    transformers_tokens_norm = tokenizer.tokenize(text_norm)
    ids_transformers_norm = tokenizer.convert_tokens_to_ids(transformers_tokens_norm)

    ids_expanded = vox_tokenizer._expand_ids(ids_plain)

    tensors[f"ids_plain_{i}"] = torch.tensor(ids_plain, dtype=torch.int64)
    tensors[f"ids_transformers_{i}"] = torch.tensor(ids_transformers, dtype=torch.int64)
    tensors[f"ids_expanded_{i}"] = torch.tensor(ids_expanded, dtype=torch.int64)
    tensors[f"ids_norm_{i}"] = torch.tensor(ids_norm, dtype=torch.int64)
    records[str(i)] = {
        "label": label,
        "text": text,
        "text_norm": text_norm,
        "ids_plain": ids_plain,
        "ids_transformers": ids_transformers,
        "ids_expanded": ids_expanded,
        "ids_norm": ids_norm,
    }

    print(f"[{i}] {label!r}", file=sys.stderr)
    print(f"    text: {text!r}", file=sys.stderr)
    print(f"    text_norm: {text_norm!r}", file=sys.stderr)
    print(f"    ids_plain        ({len(ids_plain)}): {ids_plain}", file=sys.stderr)
    print(f"    ids_transformers ({len(ids_transformers)}): {ids_transformers}", file=sys.stderr)
    print(f"    ids_expanded     ({len(ids_expanded)}): {ids_expanded}", file=sys.stderr)
    print(f"    ids_norm         ({len(ids_norm)}): {ids_norm}", file=sys.stderr)
    print(f"    plain != transformers: {ids_plain != ids_transformers}", file=sys.stderr)

    # The whole point of ids_norm: on whitespace-normalized text, the raw
    # file's convention and transformers' rewritten convention MUST agree.
    # If they don't, the two pipelines diverge on more than whitespace and
    # a Rust caller normalizing input would silently produce wrong ids.
    if ids_norm != ids_transformers_norm:
        print(
            f"FATAL: raw-file and transformers ids diverge on NORMALIZED "
            f"text for [{i}] {label!r} -- this breaks the premise that "
            f"whitespace-normalizing input makes the two conventions agree.\n"
            f"  text_norm: {text_norm!r}\n"
            f"  ids_norm             ({len(ids_norm)}): {ids_norm}\n"
            f"  ids_transformers_norm ({len(ids_transformers_norm)}): {ids_transformers_norm}",
            file=sys.stderr,
        )
        sys.exit(1)

differing_raw_vs_transformers = [
    TEXTS[int(i)][0] for i, r in records.items() if r["ids_plain"] != r["ids_transformers"]
]
differing_expanded = [
    TEXTS[int(i)][0] for i, r in records.items() if r["ids_plain"] != r["ids_expanded"]
]
print(f"texts where raw and transformers ids differ: {differing_raw_vs_transformers}",
      file=sys.stderr)
print(f"texts where plain and expanded ids differ: {differing_expanded}", file=sys.stderr)
print("texts where raw-normalized and transformers-normalized ids differ: "
      "[] (asserted above, per-text, or the script already exited)", file=sys.stderr)

# --- Confirm the encode()-adds-BOS trap on text 0 -----------------------------
# encode() always routes through transformers' TemplateProcessing post-
# processor on top of the transformers tokenize path, regardless of which
# convention tokenizer.json declares on disk.
ids_encode_0 = tokenizer.encode(TEXTS[0][1])
ids_transformers_0 = records["0"]["ids_transformers"]
bos_id = tokenizer.bos_token_id
matches_leading_bos = ids_encode_0 == [bos_id] + ids_transformers_0
print(f"bos_token_id: {bos_id}", file=sys.stderr)
print(f"encode() ids for text 0 ({len(ids_encode_0)}): {ids_encode_0}", file=sys.stderr)
print(f"encode() == [bos_id] + ids_transformers for text 0: {matches_leading_bos}",
      file=sys.stderr)
assert matches_leading_bos, (
    "encode() was expected to differ from the transformers (tokenize + "
    "convert_tokens_to_ids) path by exactly a leading bos_token_id -- it "
    "did not; the Rust side's assumption about the two paths needs "
    "re-checking."
)
print("confirmed: encode() differs from the transformers (tokenize + "
      "convert_tokens_to_ids) path by exactly a leading bos_token_id -- "
      "the Rust tokenizer path must NOT call the encode()-equivalent.",
      file=sys.stderr)

# --- Save ---------------------------------------------------------------------
# `tokenizer_fixture.json` MUST stay a bare `{index: record}` map -- the Rust
# gate (`boostr/examples/voxcpm_tokenizer_check.rs`, not touched here)
# deserializes the whole file straight into `BTreeMap<String, FixtureRecord>`.
# Extra per-record fields are fine (unknown fields are ignored); extra
# top-level keys are not, so version/normalizer metadata goes in a separate
# sibling file instead.
safetensors_path = outdir / "tokenizer_fixture.safetensors"
json_path = outdir / "tokenizer_fixture.json"
meta_path = outdir / "tokenizer_fixture_meta.json"

save_file(tensors, str(safetensors_path), metadata={
    "num_texts": str(len(TEXTS)),
    "vocab_size": str(tokenizer.vocab_size),
    "bos_token_id": str(bos_id),
    "tokenizer_class": type(tokenizer).__name__,
    "transformers_version": transformers.__version__,
    "tokenizers_version": tokenizers.__version__,
    "raw_normalizer": str(raw_normalizer_desc),
    "raw_pre_tokenizer": str(raw_pre_tokenizer_desc),
    "ground_truth": "ids_plain_{i} is the raw tokenizer.json path "
                     "(Tokenizer.from_file(...).encode(text, add_special_tokens=False)); "
                     "this is what the Rust gate checks.",
})
json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
meta_path.write_text(json.dumps({
    "transformers_version": transformers.__version__,
    "tokenizers_version": tokenizers.__version__,
    "raw_tokenizer_json_normalizer": raw_normalizer_desc,
    "raw_tokenizer_json_pre_tokenizer": raw_pre_tokenizer_desc,
    "transformers_backend_normalizer": describe_node(backend_json.get("normalizer")),
    "transformers_backend_pre_tokenizer": describe_node(backend_json.get("pre_tokenizer")),
    "ground_truth_key": "ids_plain",
    "differing_raw_vs_transformers": differing_raw_vs_transformers,
    "differing_plain_vs_expanded": differing_expanded,
}, ensure_ascii=False, indent=2) + "\n")

print(f"wrote {safetensors_path}  ({len(tensors)} tensors)", file=sys.stderr)
print(f"wrote {json_path}  ({len(records)} texts)", file=sys.stderr)
print(f"wrote {meta_path}", file=sys.stderr)
