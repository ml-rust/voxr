# Contributing to voxr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

## Build and test

Run these commands from the repository root. They match `.github/workflows/test.yml`.

```bash
cargo fmt --all --check
cargo clippy -p voxr --features decode --all-targets -- -D warnings
cargo test -p voxr --features decode,misaki,spacy-tokenizer,spacy-tagger,tts,kokoro,voxcpm,whisper,vits,vad,clap,neucodec,corpus,f16 --no-fail-fast
cargo test -p voxr-narrate --features whisper,stretch,kokoro,vits,voxcpm --no-fail-fast
cargo deny --exclude-dev check
shellcheck scripts/ci/*.sh
```

CI runs clippy for each feature set listed in `.github/workflows/test.yml`. Run clippy with the feature set you change.

Tests that need a model checkpoint skip with a notice when their environment variable is unset. The ffmpeg parity tests run when `ffmpeg` is on `PATH`. CI runs without it.

Model weights never enter the repository. Load them at run time from a model directory.

To build against sibling checkouts of numr, boostr, splintr, and regexr, copy `.cargo/config.toml.example` to `.cargo/config.toml`. The CI builds resolve from crates.io.

## Conventions

- Put tensor code and model networks in [boostr](https://github.com/ml-rust/boostr). Put samples, files, text, voices, and pipelines in voxr.
- Keep the MSRV at the `rust-version` in the manifests. CI checks it.

## Pull requests

- Add tests for every behavior change.
- Add a line to `CHANGELOG.md` for every user-visible change.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.
