#!/usr/bin/env python3
"""Fixtures for the uroman romanization step of voxr's VITS tokenizer.

Applies the MMS-TTS recipe the Hugging Face docs give for a checkpoint whose
`tokenizer_config.json` sets `is_uroman: true`: romanize the text with
`uroman.Uroman().romanize_string(text)`, with no language code, then pass the
result to `VitsTokenizer`. The model then runs on those ids.

Writes, under the output directory:

* `romanization.json`: `[{"text", "romanized"}]` over a multi-script sample
  set and the lines of uroman's own `mini-test/` files.
* `<checkpoint>/tokens.json`: `[{"text", "input_ids"}]`, the recipe's ids
  for every sample.
* `<checkpoint>/waveform/index.json` and `sentence_<i>.safetensors`: per
  native sentence, `input_ids`, `attention_mask`, the two Gaussian noise
  draws `VitsModel.forward` makes (`noise_randn_0` for the duration
  predictor, `noise_randn_like_1` for the prior latents), `waveform` and
  `sequence_lengths`.

`crates/voxr/tests/vits_uroman_parity.rs` reads the directory from
`UROMAN_FIXTURE_DIR`.

Checkpoints: `facebook/mms-tts-kor` and `facebook/mms-tts-amh`, read from
`MMS_TTS_KOR_DIR` and `MMS_TTS_AMH_DIR`, else from the Hugging Face cache
(`hf download facebook/mms-tts-kor`).

Usage:
    uv run --with uroman==1.3.1.1 --with transformers --with torch \\
        --with safetensors --with huggingface_hub \\
        crates/voxr/tests/tools/uroman/uroman_fixture.py ~/.cache/voxr-test/uroman
"""

import json
import os
import sys
from pathlib import Path

CHECKPOINTS = {
    "mms-tts-kor": ("facebook/mms-tts-kor", "MMS_TTS_KOR_DIR"),
    "mms-tts-amh": ("facebook/mms-tts-amh", "MMS_TTS_AMH_DIR"),
}

# Native sentences each checkpoint speaks for the waveform check.
SENTENCES = {
    "mms-tts-kor": [
        "이봐 무슨 일이야",
        "안녕하세요, 만나서 반갑습니다.",
    ],
    "mms-tts-amh": [
        "ሰላም ነው?",
        "እንኳን ደህና መጡ።",
    ],
}

SAMPLES = [
    # Cyrillic: Russian and Ukrainian, with capitals.
    "Привет, мир!",
    "МОСКВА — столица России.",
    "Щука і їжак п'ють чай у Києві.",
    # Devanagari.
    "नमस्ते दुनिया",
    "भारत एक विशाल देश है।",
    # Arabic.
    "مرحبا بالعالم",
    "السلام عليكم ورحمة الله",
    # Hangul.
    "이봐 무슨 일이야",
    "안녕하세요, 만나서 반갑습니다.",
    "서울특별시",
    # Han.
    "你好，世界！",
    "北京是中国的首都。",
    # Thai.
    "สวัสดีครับ",
    "ภาษาไทยเป็นภาษาราชการ",
    # Greek, with final sigma and capitals.
    "Καλημέρα κόσμε",
    "ΟΔΥΣΣΕΥΣ και Πηνελόπη",
    # Amharic (Ge'ez).
    "ሰላም ነው?",
    "እንኳን ደህና መጡ።",
    "አዲስ አበባ",
    # Georgian.
    "გამარჯობა",
    "საქართველო",
    # Hebrew.
    "שלום עולם",
    "ירושלים",
    # Digits: ASCII, Arabic-Indic, Devanagari, Han numerals, a fraction.
    "2024",
    "٣٤٥",
    "४२",
    "二十三",
    "½",
    # Punctuation.
    "«Да», — сказал он…",
    "「こんにちは」！？",
    "¿Qué tal? ¡Bien!",
    # Latin with diacritics and ligatures.
    "Café Ærøskøbing Straße",
    # Mixed lines.
    "Hello Привет 안녕 مرحبا 123",
    "The word ሰላም means peace.",
    "Seoul (서울) has 9,700,000 people.",
    # Empty and whitespace-only.
    "",
    " ",
    "   \t  ",
]

MINI_TEST_FILES = ["multi-script.txt", "hin.txt", "fas.txt"]


def checkpoint_dir(repo: str, var: str) -> Path:
    env = os.environ.get(var)
    if env:
        return Path(env)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


def mini_test_lines(uroman_module) -> list[str]:
    root = Path(uroman_module.__file__).parent / "mini-test"
    lines = []
    for name in MINI_TEST_FILES:
        with open(root / name, encoding="utf-8") as f:
            lines.extend(line.rstrip("\n") for line in f)
    return lines


class NoiseCapture:
    """Records every `torch.randn` / `torch.randn_like` draw in call order as
    `noise_<fn>_<index>`, and restores both functions on exit."""

    def __init__(self, torch):
        self.torch = torch
        self.tensors = {}
        self._randn = torch.randn
        self._randn_like = torch.randn_like

    def __enter__(self):
        def randn(*args, **kwargs):
            out = self._randn(*args, **kwargs)
            self.tensors[f"noise_randn_{len(self.tensors)}"] = out.clone()
            return out

        def randn_like(*args, **kwargs):
            out = self._randn_like(*args, **kwargs)
            self.tensors[f"noise_randn_like_{len(self.tensors)}"] = out.clone()
            return out

        self.torch.randn = randn
        self.torch.randn_like = randn_like
        return self

    def __exit__(self, exc_type, exc, tb):
        self.torch.randn = self._randn
        self.torch.randn_like = self._randn_like


def write_waveforms(out_dir: Path, ckpt: Path, uroman, sentences: list[str]):
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import VitsModel, VitsTokenizer

    tokenizer = VitsTokenizer.from_pretrained(str(ckpt))
    model = VitsModel.from_pretrained(str(ckpt), torch_dtype=torch.float32)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for i, text in enumerate(sentences):
        torch.manual_seed(2000 + i)
        inputs = tokenizer(text=uroman.romanize_string(text), return_tensors="pt")
        with NoiseCapture(torch) as capture, torch.no_grad():
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        tensors = {
            "input_ids": inputs["input_ids"].to(torch.int64),
            "attention_mask": inputs["attention_mask"].to(torch.float32),
            "waveform": output.waveform,
            "sequence_lengths": output.sequence_lengths.to(torch.int64),
        }
        tensors.update(capture.tensors)
        name = f"sentence_{i}.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, out_dir / name)
        meta.append({"index": i, "text": text, "file": name})
        print(f"wrote {out_dir / name}: {text!r}")

    index = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sentences": meta,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    import uroman as ur
    from transformers import VitsTokenizer

    uroman = ur.Uroman()
    texts = SAMPLES + mini_test_lines(ur)

    romanization = [{"text": t, "romanized": uroman.romanize_string(t)} for t in texts]
    (out / "romanization.json").write_text(
        json.dumps(romanization, ensure_ascii=False, indent=1) + "\n"
    )
    print(f"wrote {out / 'romanization.json'}")

    for name, (repo, var) in CHECKPOINTS.items():
        ckpt = checkpoint_dir(repo, var)
        tokenizer = VitsTokenizer.from_pretrained(str(ckpt))
        if not tokenizer.is_uroman:
            print(f"error: {repo} does not set is_uroman: true", file=sys.stderr)
            sys.exit(1)
        tokens = [
            {"text": t, "input_ids": tokenizer(text=uroman.romanize_string(t))["input_ids"]}
            for t in texts
        ]
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "tokens.json").write_text(
            json.dumps(tokens, ensure_ascii=False, indent=1) + "\n"
        )
        print(f"wrote {out / name / 'tokens.json'}")
        write_waveforms(out / name / "waveform", ckpt, uroman, SENTENCES[name])


if __name__ == "__main__":
    main()
