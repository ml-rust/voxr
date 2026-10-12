<div align="center">

# voxr

<h3>Audio and speech in Rust, on boostr models.</h3>

<p>
WAV and compressed-audio decode, DSP, G2P, and TTS, ASR, and VAD pipelines. Model weights load at run time from a directory, and no weights are compiled in.
</p>

<p>
  <a href="https://docs.rs/voxr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/voxr"><strong>Crate</strong></a>
  ·
  <a href="#crates"><strong>Crates</strong></a>
  ·
  <a href="#models"><strong>Models</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
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

voxr is a Cargo workspace for audio and speech on top of [boostr](https://github.com/ml-rust/boostr) models and [numr](https://github.com/ml-rust/numr) tensors. The `voxr` crate holds the sample-level side: files, codecs, DSP, G2P, voices, and the TTS, ASR, and VAD pipelines. The model networks and their kernels live in boostr. The `voxr-narrate` crate turns a narration script into a checked voiceover.

## Crates

| Crate | What it holds | Features |
| --- | --- | --- |
| [`voxr`](crates/voxr) | WAV and compressed-audio decode, resamplers, tempo stretch, loudness and enhancement, single-sound metrics, misaki and spaCy G2P, voices, and the Kokoro, VITS, VoxCPM2, Whisper, Silero VAD, CLAP and NeuCodec pipelines | [voxr](#voxr) |
| [`voxr-narrate`](crates/voxr-narrate) | Script parsing, the speech check of a take against its script, glitch repair, voice polish, the take loop, engine adapters, and the speak and transcribe commands | [voxr-narrate](#voxr-narrate) |

voxr-narrate implements the speak and transcribe contract of the [motion-video skill](https://github.com/farhan-syah/motion-video-skill). voxr implements its sound review.

## Quick Start

Add the crate. Choose the features you need from the lists below.

```bash
cargo add voxr --features decode
```

Transcribe a take with Whisper. Download a Whisper checkpoint in Hugging Face layout first, such as `openai/whisper-base`. Its `generation_config.json` must list `alignment_heads`. voxr does not download it.

```bash
cargo run --release -p voxr-narrate --features whisper --example transcribe_take -- \
  --model WHISPER_DIR AUDIO.wav
```

The example writes one words file per input. Add `--out DIR` to choose the output directory, and `--language xx` to set the language.

## Features

Every feature is off by default. A build with no features still has WAV, the resamplers, tempo stretch, enhancement, and metrics. It also has the sound review, pitch, ASR error rates, and the signal half of the TTS quality gate.

### voxr

| Feature | Adds |
| --- | --- |
| `decode` | WAV, FLAC, MP3, AAC-LC and ALAC in MP4/M4A/MOV, and Opus and Vorbis in Ogg/WebM/MKV, with ffmpeg-matched trims and downmix. Multichannel AAC, HE-AAC and Opus past two channels are refused |
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
| `neucodec` | The NeuCodec model gate |
| `corpus` | Speech-corpus preparation. Implies `whisper`, `vad`, `neucodec` and `decode` |
| `f16` | Half-precision tensors |
| `cuda` | The CUDA backend |

### voxr-narrate

| Feature | Adds |
| --- | --- |
| `whisper` | Whisper word timings, `WhisperHearing`, and `NarrationRuns` over loaded engines. Implies `voxr/whisper` |
| `stretch` | `stretch_track` and `resample_48k` over voxr's ffmpeg-matched atempo and resampler |
| `kokoro` | `KokoroSpeech`, Kokoro as a speech engine. Implies `voxr/kokoro` |
| `vits` | `VitsSpeech`, VITS as a speech engine. Implies `voxr/vits` |
| `voxcpm` | `VoxcpmSpeech`, VoxCPM2 as a speech engine. Implies `voxr/voxcpm` |
| `cuda` | The CUDA runtime for Whisper, Kokoro, VITS and VoxCPM2 |

## Models

No model weights are compiled in or shipped. Every engine loads its checkpoint at run time from a model directory. voxr fetches nothing over the network. Download a checkpoint, for example from Hugging Face, into a directory first.

The spaCy tagger and the out-of-vocabulary G2P model for Kokoro are in the Hugging Face repo [`fs90/kokoro-g2p`](https://huggingface.co/fs90/kokoro-g2p). Place them under a Kokoro model directory as `g2p/tagger/` and `g2p/oov/`.

The embedded data is text-processing data only. It holds the misaki lexicons behind `misaki` and the spaCy tokenizer tables behind `spacy-tokenizer`.

## CUDA

The `cuda` feature builds numr's CUDA kernels. It needs CUDA 12.8 or newer. CUDA 13.x also works.

`NUMR_CUDA_ARCH` selects the GPU architectures the build compiles native code for. Unset, the build targets the GPUs on the build host. Set `NUMR_CUDA_ARCH=all` for any binary that runs on another machine. numr's README lists every value.

## Minimum Rust version

1.95.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).

Code ported from other projects and data voxr redistributes are listed with their licences in [crates/voxr/LICENSE-OTHERS](crates/voxr/LICENSE-OTHERS) and [crates/voxr-narrate/LICENSE-OTHERS](crates/voxr-narrate/LICENSE-OTHERS). Each embedded data set carries its own LICENSE and NOTICE beside it under `crates/voxr/assets/`.
