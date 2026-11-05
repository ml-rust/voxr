#!/usr/bin/env bash
# Run clippy on every feature set of one crate.
#
#   scripts/ci/clippy.sh voxr
#   scripts/ci/clippy.sh voxr-narrate
#
# Each feature gate compiles alone, so a gate that leans on another one's import
# fails here. `--all-targets` also checks the tests and examples each gate
# enables. Add a set for every new feature that builds with no CUDA toolkit.
#
# Every set runs. The run prints each failed set, then exits 1.

set -uo pipefail

crate="${1:-}"

case "$crate" in
    voxr)
        feature_sets=(
            ""
            "decode"
            "spacy-tokenizer"
            "misaki"
            "kokoro"
            "vits"
            "whisper,vad"
            "voxcpm"
            "clap"
            "neucodec"
            "corpus"
            "decode,misaki,spacy-tokenizer,spacy-tagger,tts,kokoro,voxcpm,whisper,vits,vad,clap,neucodec,corpus,f16"
        )
        ;;
    voxr-narrate)
        feature_sets=(
            ""
            "stretch"
            "whisper"
            "kokoro"
            "vits"
            "voxcpm"
            "whisper,stretch,kokoro,vits,voxcpm"
        )
        ;;
    *)
        echo "error: unknown crate '$crate': pass voxr or voxr-narrate" >&2
        exit 2
        ;;
esac

failed=()
for features in "${feature_sets[@]}"; do
    label="${features:-no features}"
    echo "::group::clippy $crate ($label)"
    cargo clippy -p "$crate" --features "$features" --all-targets -- -D warnings
    status=$?
    echo "::endgroup::"
    if [[ $status -ne 0 ]]; then
        failed+=("$label")
        echo "::error title=Clippy ($crate)::clippy failed on '$label'. Run: cargo clippy -p $crate --features \"$features\" --all-targets -- -D warnings"
    fi
done

echo "clippy $crate: $((${#feature_sets[@]} - ${#failed[@]})) of ${#feature_sets[@]} feature sets pass"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf 'failed: %s\n' "${failed[@]}" >&2
    exit 1
fi
