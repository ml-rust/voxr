#!/usr/bin/env bash
# Stamp both crate versions from a release tag.
#
#   scripts/ci/stamp_version.sh <version>       # e.g. 0.1.0, 0.1.0-beta.2
#
# Rewrites `[package] version` in crates/voxr/Cargo.toml and
# crates/voxr-narrate/Cargo.toml, and pins voxr-narrate's `voxr` requirement
# to the same version with `=`, so voxr-narrate never resolves another voxr.
#
# Only a prerelease tag changes anything. release-validate.yml has already
# proved the tag's base version matches both manifests, so for `v0.1.0` this
# is a no-op and for `v0.1.0-beta.2` it appends the suffix. Re-running a
# release stage is idempotent.

set -euo pipefail

VERSION="${1:?usage: stamp_version.sh <version>}"

version_of() {
    cargo metadata --no-deps --format-version=1 \
        | jq -r --arg name "$1" '.packages[] | select(.name == $name) | .version'
}

CURRENT_VOXR=$(version_of voxr)
CURRENT_NARRATE=$(version_of voxr-narrate)

if [[ "$VERSION" == "$CURRENT_VOXR" && "$VERSION" == "$CURRENT_NARRATE" ]]; then
    echo "Versions already $VERSION — nothing to stamp."
    exit 0
fi

# The first `version = "..."` at the start of a line is [package]. Every
# dependency version in these manifests is an inline-table or same-line form.
for manifest in crates/voxr/Cargo.toml crates/voxr-narrate/Cargo.toml; do
    perl -i -pe 'if (!$done && /^version = "/) { s/^version = ".*"/version = "'"$VERSION"'"/; $done=1 }' "$manifest"
done

# A prerelease satisfies no caret requirement on its base version, so the
# path dependency carries the exact version.
perl -i -pe 's/^(voxr = \{ path = "\.\.\/voxr", version = )"[^"]*"/$1"='"$VERSION"'"/' \
    crates/voxr-narrate/Cargo.toml

echo "Stamped voxr: $CURRENT_VOXR -> $VERSION"
echo "Stamped voxr-narrate: $CURRENT_NARRATE -> $VERSION"
