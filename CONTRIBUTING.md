# Contributing to voxr

## Before you start

Open an issue before you start anything non-trivial. Discuss design on [Discord](https://discord.gg/jBhFk9kHPg).

| Need | For |
| --- | --- |
| Rust 1.95 or newer | Every build |
| `cargo-deny` | The dependency audit |
| `shellcheck` | The script check |
| `ffmpeg` on `PATH`, optional | The ffmpeg parity tests. CI runs without it |

## Set up

Copy the local config. The build needs it when you work against sibling checkouts.

```bash
cp .cargo/config.toml.example .cargo/config.toml
```

- Sibling checkouts: the file points numr, boostr, splintr, regexr, and containr at `../`. Without it, cargo resolves them from crates.io, and the build fails on APIs the published versions lack.
- Models: set `BOOSTR_MODELS_DIR` in that file to your checkpoint directory.
- Skipped tests: a test that needs a checkpoint or fixture skips with a notice when its variable is unset. Such a run reports success without testing the model.
- Other variables: `WHISPER_BASE_DIR`, `WHISPER_SMALL_DIR`, `KOKORO_DIR`, `MMS_TTS_ZLM_DIR`, `CLAP_DIR`, `CLAP_FIXTURE_DIR`, `SOUND_REVIEW_METRICS_DIR`, `UROMAN_FIXTURE_DIR`, `NARRATION_JS_DIR`, `SPACY_EN_TAGGER_DIR`, and `VOXCPM2_*`. The list is partial. Each test names its variable.

## Build and test

Run these from the repository root.

```bash
cargo fmt --all --check
cargo clippy -p voxr --features decode --all-targets -- -D warnings
cargo test -p voxr --features decode,misaki,spacy-tokenizer,spacy-tagger,tts,kokoro,voxcpm,whisper,vits,vad,clap,neucodec,corpus,f16 --no-fail-fast
cargo test -p voxr-narrate --features whisper,stretch,kokoro,vits,voxcpm --no-fail-fast
cargo deny --exclude-dev check
shellcheck scripts/ci/*.sh
```

- Format, test, audit, and script commands: the same commands run in `.github/workflows/test.yml`.
- Clippy: CI runs it once for each feature set in the matrix of `test.yml`. Run it with the feature set you change.
- Other CI jobs: docs with `-D warnings`, and an MSRV check on the library targets.
- GPU: CI has no GPU and does not build `cuda`.

Model weights never enter the repository. Load them at run time from a model directory.

## Conventions

- Put tensor code and model networks in [boostr](https://github.com/ml-rust/boostr). Put samples, files, text, voices, and pipelines in voxr.
- Keep the MSRV at the `rust-version` in the manifests. CI checks it.

## Pull requests

- Add tests for every behavior change.
- Add a line under `[Unreleased]` in `CHANGELOG.md` for every user-visible change.
- Run the commands in [Build and test](#build-and-test) before you open the pull request.

## Releases

- Changelog: each release needs a non-empty section in `CHANGELOG.md`.
- Tag check: `.github/workflows/release-validate.yml` refuses a tag whose version has no entry.
- Release body: the GitHub Release body is that section.
- Versions: voxr and voxr-narrate release together at one version.
