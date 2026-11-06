<div align="center">

# voxr

<h3>Audio and speech in Rust, on boostr models.</h3>

<p>
WAV and compressed-audio decode, signal processing, grapheme-to-phoneme (G2P), and speech synthesis, transcription, and voice-activity pipelines. Model weights load at run time from a directory, and none are compiled in.
</p>

<p>
  <a href="https://docs.rs/voxr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/voxr"><strong>Crate</strong></a>
  ·
  <a href="#crates"><strong>Crates</strong></a>
  ·
  <a href="#what-works"><strong>What works</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="#models"><strong>Models</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/voxr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/voxr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/voxr">
    <img src="https://img.shields.io/crates/v/voxr" alt="crates.io">
  </a>
  <a href="https://docs.rs/voxr">
    <img src="https://img.shields.io/docsrs/voxr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/voxr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/voxr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/voxr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

voxr is an audio and speech library for Rust: text-to-speech, speech-to-text, voice-activity detection, and audio decode and DSP. It is a Cargo workspace on top of [boostr](https://github.com/ml-rust/boostr) models and [numr](https://github.com/ml-rust/numr) tensors. The model networks and their kernels live in boostr.

## No ffmpeg, no Python

The work happens in Rust. voxr links no ffmpeg, no libav and no Python, and a library call starts no process.

- Decode: WAV, FLAC, MP3, AAC-LC, ALAC, Opus and Vorbis decode in Rust. Trims and downmix match ffmpeg. See [Decode formats](#decode-formats).
- Resample and tempo stretch: voxr resamples and stretches tempo in Rust, and its results match ffmpeg's resampler and atempo filter.
- Models: Kokoro, VITS, VoxCPM2, Whisper, CLAP and Silero VAD run on boostr, with no PyTorch.
- G2P: the misaki and spaCy English tokenizer and tagger run in Rust, with their data embedded.

voxr can replace an ffmpeg decode step and a Python speech stack. It does not download model weights.

### Not supported yet

- WebGPU: voxr has no `wgpu` feature.
- Multichannel AAC and HE-AAC: decode refuses them.
- Opus past two channels: decode refuses it.
- Other codecs: `voxr` refuses them. `voxr-narrate` can fall back to an `ffmpeg` on `PATH` for a file voxr refuses. Only that fallback runs a process.
- VAD on CUDA: no test or example in this repository runs Silero VAD on CUDA.

## Crates

| Crate | What it does | Features |
| --- | --- | --- |
| [`voxr`](crates/voxr) | Audio files, DSP, G2P, voices, and speech pipelines | [voxr features](#voxr-features) |
| [`voxr-narrate`](crates/voxr-narrate) | Turns a narration script into a checked voiceover | [voxr-narrate features](#voxr-narrate-features) |

- voxr-narrate runs the speak and transcribe commands of the [motion-video skill](https://github.com/farhan-syah/motion-video-skill).
- voxr provides the sound review of that skill.

## What works

| Task | Engine | Feature | Backend |
| --- | --- | --- | --- |
| Speech synthesis | Kokoro, VITS (MMS-TTS) | `kokoro`, `vits` | CPU, or CUDA with `cuda` |
| Voice cloning | VoxCPM2 | `voxcpm` | CPU, or CUDA with `cuda` |
| Transcription with word timings | Whisper | `whisper` | CPU, or CUDA with `cuda` |
| Audio-text scoring | CLAP | `clap` | CPU, or CUDA with `cuda` |
| Voice activity detection | Silero VAD | `vad` | CPU. No test or example in this repository runs it on CUDA |
| Audio codec | NeuCodec | `neucodec` | CPU, or CUDA in the `noise_probe_decode` example |

- Rust: 1.95 or newer.
- CUDA: needs the setup in [CUDA](#cuda).
- Abbreviations: G2P is grapheme-to-phoneme, TTS is text-to-speech, ASR is automatic speech recognition, and VAD is voice activity detection.

## Quick Start

Add the crate to your project, with the features you need:

```bash
cargo add voxr --features decode
```

Decode a file to 16 kHz mono and write it as a 16-bit WAV file:

```rust
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = voxr::decode_audio_file_mono_at(Path::new("in.m4a"), 16_000)?;
    let wav = voxr::encode_wav_pcm16(&samples, 16_000)?;
    std::fs::write("out.wav", wav)?;
    Ok(())
}
```

To try transcription from a checkout of this repository:

1. Download a Whisper checkpoint in Hugging Face layout, such as `openai/whisper-base`. voxr does not download it.
2. Check that its `generation_config.json` lists `alignment_heads`.
3. Run the example:

```bash
cargo run --release -p voxr-narrate --features whisper --example transcribe_take -- \
  --model WHISPER_DIR AUDIO.wav
```

The example writes the words of each input to standard output.

| Flag | Effect |
| --- | --- |
| `--out DIR` | Writes `NAME.words.json` files to `DIR` |
| `--language xx` | Sets the language. The default is `en` |

## Features

Every feature is off by default. A build with no features includes:

- WAV read and write
- Resamplers and tempo stretch
- Enhancement and single-sound metrics
- Sound review and pitch tracking
- ASR error rates
- TTS quality statistics that need no model

### voxr features

| Feature | Adds |
| --- | --- |
| `decode` | Compressed-audio decode. See [Decode formats](#decode-formats) |
| `spacy-tokenizer` | The spaCy `en_core_web_sm` 3.8.0 tokenizer, with its patterns and tables embedded |
| `spacy-tagger` | The `en_core_web_sm` tagger. Implies `spacy-tokenizer` |
| `misaki` | misaki English G2P with the US and GB lexicons embedded. Implies `spacy-tagger` and `f16` |
| `tts` | `TtsEngine`, `TtsBundle` and `Voice`, the engine-agnostic synthesis surface |
| `kokoro` | Kokoro synthesis. Implies `tts` and `misaki` |
| `vits` | VITS (MMS-TTS) synthesis, with uroman romanization. Implies `tts` |
| `voxcpm` | VoxCPM2 voice cloning. Implies `tts` and `decode` |
| `whisper` | Whisper transcription with word timings. Implies `decode` |
| `vad` | Silero VAD segmentation |
| `clap` | CLAP audio-text scoring, which ranks sounds in the sound review |
| `neucodec` | The NeuCodec model, for its probing examples |
| `corpus` | Speech-corpus preparation. Implies `whisper`, `vad`, `neucodec` and `decode` |
| `f16` | Half-precision tensors |
| `cuda` | The CUDA backend |

### Decode formats

| Codec | Containers | Limit |
| --- | --- | --- |
| WAV, FLAC, MP3 | Their own files | None |
| AAC-LC | MP4, M4A, MOV | Multichannel AAC and HE-AAC are refused |
| ALAC | MP4, M4A, MOV | None |
| Opus | Ogg, WebM, MKV | Past two channels is refused |
| Vorbis | Ogg, WebM, MKV | None |

Trims and downmix match ffmpeg. MP4 edit lists and Matroska codec delay come from containr.

### voxr-narrate features

| Feature | Adds |
| --- | --- |
| `whisper` | Whisper word timings, `WhisperHearing`, and `NarrationRuns` over loaded engines. Implies `voxr/whisper` |
| `stretch` | `stretch_track` and `resample_48k` over voxr's ffmpeg-matched atempo and resampler |
| `kokoro` | `KokoroSpeech`, Kokoro as a speech engine. Implies `voxr/kokoro` |
| `vits` | `VitsSpeech`, VITS as a speech engine. Implies `voxr/vits` |
| `voxcpm` | `VoxcpmSpeech`, VoxCPM2 as a speech engine. Implies `voxr/voxcpm` |
| `cuda` | The CUDA runtime for Whisper, Kokoro, VITS and VoxCPM2 |

## Models

No model weights are compiled in or shipped, and voxr fetches nothing over the network. Download each checkpoint, for example from Hugging Face, and pass its directory at run time.

| Files | Where to get them | Where to put them |
| --- | --- | --- |
| spaCy tagger, for Kokoro | [`fs90/kokoro-g2p`](https://huggingface.co/fs90/kokoro-g2p) | `g2p/tagger/` under the Kokoro model directory |
| Out-of-vocabulary G2P model, for Kokoro | [`fs90/kokoro-g2p`](https://huggingface.co/fs90/kokoro-g2p) | `g2p/oov/` under the Kokoro model directory |

The embedded data is text-processing data only: the misaki lexicons behind `misaki`, and the spaCy tokenizer tables behind `spacy-tokenizer`.

## CUDA

| Need | Value |
| --- | --- |
| Toolkit | CUDA 12.8 or newer, with `nvcc` on `PATH` |
| GPU | Compute capability 7.5 (Turing) or newer |

- CUDA binaries: a build targets the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for a binary that runs on another machine.
- Every value: numr's [docs/backends.md](https://github.com/ml-rust/numr/blob/main/docs/backends.md) lists them.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).

Code ported from other projects and data voxr redistributes are listed with their licenses in [crates/voxr/LICENSE-OTHERS](crates/voxr/LICENSE-OTHERS) and [crates/voxr-narrate/LICENSE-OTHERS](crates/voxr-narrate/LICENSE-OTHERS). Each embedded data set carries its own LICENSE and NOTICE beside it under `crates/voxr/assets/`.
