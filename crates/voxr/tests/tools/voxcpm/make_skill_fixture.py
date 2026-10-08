#!/usr/bin/env python3
"""Reference ground truth for the skill's VoxCPM2 runner, `voxcpm_speak.py`,
end to end: the model inputs, the noise, the patches and the waveform of one
`VoxCPM.generate` call per mode, through the same library entry point the
runner calls.

Modes, each one branch of the runner:
    design    `generate(text=paren(voice) + text)`, zero-shot with a voice
              description. The runner saves this take as the anchor.
    zeroshot  `generate(text=text)`: no description, no reference.
    clone     `generate(text=paren(style) + text, reference_wav_path=REF)`.
    hifi      `generate(text=text, prompt_wav_path=REF, prompt_text=PT,
              reference_wav_path=REF)`: the runner's Hi-Fi branch.
Every call passes `cfg_value` and `inference_timesteps` and leaves the rest
at the library defaults the runner keeps: `normalize=False`, `denoise=False`
(the runner loads no denoiser), `retry_badcase=True` with 3 takes at ratio
6, `min_len=2`, `max_len=4096`. `torch.manual_seed(seed)` runs right before
each call, as the runner does per sentence.

What differs from the runner, and why:
    - The model computes in float32 on the CPU, not bfloat16 on CUDA, so the
      reference is deterministic and a Rust run at F32 can match it. The
      checkpoint stores bfloat16 weights, which float32 holds exactly.
    - `optimize=False`: torch.compile changes nothing a CPU run computes.
Neither changes the code path: `VoxCPM._generate`, `build_prompt_cache`,
`_generate_with_prompt_cache` and `_inference` all run unmodified.

Captured by wrapping, never by re-implementing:
    - `_encode_wav`: each recording's 16 kHz samples (the same
      `librosa.load(path, sr=16000, mono=True)` call, repeated) and its
      latent patches, with the padding side.
    - `_inference`: each take's inputs (`text_token`, `text_mask`,
      `audio_mask`, `max_len`) and outputs (generated patches,
      `context_len`).
    - `torch.randn`: every noise draw, in order, tagged with its take.
    - the returned waveform.

A token list rides along: the ids `text_tokenizer` gives the runner's text
after `_generate`'s whitespace collapse, for texts the four modes do not
reach (Chinese, mixed scripts, edge whitespace).

Outputs under --out-dir, per mode: `skill_<mode>.safetensors` (tensors) and
`skill_<mode>.json` (texts, ids, settings, per-take patch counts), plus
`skill_tokens.json`.

Paths come from the CLI, with env fallbacks:
    --ckpt-dir    (VOXCPM_CKPT_DIR)     the openbmb/VoxCPM2 snapshot
    --out-dir     (VOXCPM_SKILL_FIXTURE_DIR)
    --ref-wav     (VOXCPM2_REF_WAV)     the recording the clone modes read
    --prompt-text (VOXCPM2_REF_TEXT)    the exact transcript of --ref-wav
    --pkg-dir     (VOXCPM_PKG_DIR)      optional, prepended to sys.path
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_TEXT = "Selamat pagi, today we test the voice that speaks this narration."
DEFAULT_VOICE = "A calm male narrator"
DEFAULT_STYLE = "relaxed, explaining"
TOKEN_TEXTS = [
    "Hello world.",
    "  leading and trailing  ",
    "tab\tand\nnewline  runs",
    "Saya dah cuba macam-macam, tapi honestly it doesn't make sense.",
    "你好世界，今天天气很好。",
    "我们 test 一下 mixed 中文 and English.",
    "(A calm male narrator)The opening line.",
    "Ça coûte 12,50 € — d'accord?",
    "日本語のテキストです。",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=os.environ.get("VOXCPM_CKPT_DIR"))
    p.add_argument("--out-dir", default=os.environ.get("VOXCPM_SKILL_FIXTURE_DIR"))
    p.add_argument("--ref-wav", default=os.environ.get("VOXCPM2_REF_WAV"))
    p.add_argument("--prompt-text", default=os.environ.get("VOXCPM2_REF_TEXT"))
    p.add_argument("--pkg-dir", default=os.environ.get("VOXCPM_PKG_DIR"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--style", default=DEFAULT_STYLE)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--modes", default="design,zeroshot,clone,hifi")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 keeps torch's own")
    a = p.parse_args()
    for flag, value, env in [("--ckpt-dir", a.ckpt_dir, "VOXCPM_CKPT_DIR"),
                             ("--out-dir", a.out_dir, "VOXCPM_SKILL_FIXTURE_DIR")]:
        if not value:
            p.error(f"{flag} is required: pass it or set {env}")
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    unknown = [m for m in a.modes if m not in ("design", "zeroshot", "clone", "hifi")]
    if unknown:
        p.error(f"--modes {unknown}: pick from design, zeroshot, clone, hifi")
    if any(m in ("clone", "hifi") for m in a.modes):
        if not a.ref_wav or not Path(a.ref_wav).is_file():
            p.error(f"--ref-wav {a.ref_wav!r} is not a file: pass a recording or set VOXCPM2_REF_WAV")
    if "hifi" in a.modes and not a.prompt_text:
        p.error("--prompt-text is required for hifi: pass the transcript of --ref-wav or set VOXCPM2_REF_TEXT")
    return a


args = parse_args()
if args.pkg_dir:
    sys.path.insert(0, str(Path(args.pkg_dir)))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import voxcpm.model.voxcpm2 as voxcpm2_module  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402

if args.threads:
    torch.set_num_threads(args.threads)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# float32 compute: the only change to the library, made where it picks the dtype.
voxcpm2_module.pick_runtime_dtype = lambda device, configured: "float32"
model = VoxCPM.from_pretrained(args.ckpt_dir, load_denoiser=False, optimize=False, device="cpu")
tm = model.tts_model
assert next(tm.base_lm.parameters()).dtype == torch.float32, "the model must compute in float32"
SR = tm._encode_sample_rate


def paren(t):
    """The runner's `paren`."""
    return t if not t or t.startswith("(") else f"({t})"


def collapse(text):
    """`VoxCPM._generate`'s whitespace handling, verbatim."""
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text)


class Capture:
    """Wraps the library's own functions on `tm` and `torch.randn`."""

    def __init__(self):
        self.encodes, self.takes, self.noise = [], [], []
        self._encode = tm._encode_wav
        self._inference = tm._inference
        self._randn = torch.randn

    def __enter__(self):
        cap = self

        def encode_wav(wav_path, padding_mode="right", trim_silence_vad=False):
            feat = cap._encode(wav_path, padding_mode=padding_mode, trim_silence_vad=trim_silence_vad)
            audio, _ = librosa.load(wav_path, sr=SR, mono=True)
            cap.encodes.append({"side": padding_mode, "wav": torch.from_numpy(audio).clone(), "feat": feat.clone()})
            return feat

        def inference(text, text_mask, feat, feat_mask, **kw):
            take = {"text_token": text.clone(), "text_mask": text_mask.clone(), "audio_mask": feat_mask.clone(),
                    "feat": feat.clone(), "max_len": kw["max_len"], "min_len": kw["min_len"], "noise_from": len(cap.noise)}
            cap.takes.append(take)
            for out in cap._inference(text, text_mask, feat, feat_mask, **kw):
                take["patches"] = out[1].clone()
                take["context_len"] = int(out[2])
                yield out

        def randn(*a, **kw):
            z = cap._randn(*a, **kw)
            cap.noise.append(z.detach().clone())
            return z

        tm._encode_wav = encode_wav
        tm._inference = inference
        torch.randn = randn
        return self

    def __exit__(self, *exc):
        tm._encode_wav = self._encode
        tm._inference = self._inference
        torch.randn = self._randn


def run_mode(mode):
    text, ref, pt = args.text, args.ref_wav, args.prompt_text
    if mode == "design":
        call = dict(text=f"{paren(args.voice.strip())}{text}")
    elif mode == "zeroshot":
        call = dict(text=text)
    elif mode == "clone":
        call = dict(text=f"{paren(args.style.strip())}{text}", reference_wav_path=ref)
    else:
        call = dict(text=text, prompt_wav_path=ref, prompt_text=pt, reference_wav_path=ref)
    torch.manual_seed(args.seed)
    with Capture() as cap, torch.inference_mode():
        wav = model.generate(**call, cfg_value=args.cfg, inference_timesteps=args.steps)

    target = collapse(call["text"])
    full = (pt + target) if mode == "hifi" else target
    tensors = {"wav": torch.from_numpy(np.asarray(wav, dtype=np.float32)).contiguous()}
    for enc in cap.encodes:
        name = "ref" if enc["side"] == "right" else "prompt"
        tensors[f"{name}_wav16k"] = enc["wav"].float().contiguous()
        tensors[f"{name}_feat"] = enc["feat"].float().contiguous()
    takes = []
    for k, take in enumerate(cap.takes):
        end = cap.takes[k + 1]["noise_from"] if k + 1 < len(cap.takes) else len(cap.noise)
        z = torch.cat(cap.noise[take["noise_from"]:end], dim=0)  # [n, feat_dim, patch_size]
        tensors[f"take{k}_z"] = z.float().contiguous()
        tensors[f"take{k}_patches"] = take["patches"].float().contiguous()
        takes.append({"patches": int(take["patches"].shape[0]), "draws": int(z.shape[0]),
                      "context_len": take["context_len"], "max_len": take["max_len"], "min_len": take["min_len"]})
        if k == 0:
            tensors["text_token"] = take["text_token"][0].to(torch.int64).contiguous()
            tensors["text_mask"] = take["text_mask"][0].to(torch.int64).contiguous()
            tensors["audio_mask"] = take["audio_mask"][0].to(torch.int64).contiguous()
            tensors["prefill_feat"] = take["feat"][0].float().contiguous()
    meta = {
        "mode": mode,
        "generate_text": call["text"],
        "target_text": target,
        "prompt_text": pt if mode == "hifi" else None,
        "prompt_wav": call.get("prompt_wav_path"),
        "reference_wav": call.get("reference_wav_path"),
        "full_ids": tm.text_tokenizer(full),
        "target_ids": tm.text_tokenizer(target),
        "seed": args.seed,
        "cfg_value": args.cfg,
        "inference_timesteps": args.steps,
        "retry_badcase_max_times": 3,
        "retry_badcase_ratio_threshold": 6.0,
        "sample_rate": tm.sample_rate,
        "encode_sample_rate": SR,
        "decode_patch_samples": tm.patch_size * tm._decode_chunk_size,
        "takes": takes,
        "wav_samples": int(tensors["wav"].shape[0]),
    }
    save_file(tensors, str(out_dir / f"skill_{mode}.safetensors"))
    (out_dir / f"skill_{mode}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"{mode}: {len(takes)} take(s), patches {[t['patches'] for t in takes]}, "
          f"{meta['wav_samples']} samples", file=sys.stderr, flush=True)


tokens = [{"text": t, "collapsed": collapse(t), "ids": tm.text_tokenizer(collapse(t))} for t in TOKEN_TEXTS]
(out_dir / "skill_tokens.json").write_text(json.dumps(tokens, ensure_ascii=False, indent=1))
for mode in args.modes:
    run_mode(mode)
