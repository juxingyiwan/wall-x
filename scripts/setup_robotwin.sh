#!/usr/bin/env bash
# Fetch optional source dependencies using Wall-X's ignored third_party layout.
# Does not install packages or change an existing checkout/environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-$ROOT/third_party}"

checkout() {
    local url="$1" path="$2" revision="$3" tag="${4:-}"
    if [[ -e "$path" ]]; then
        if [[ ! -d "$path/.git" ]] || [[ "$(git -C "$path" rev-parse HEAD)" != "$revision" ]]; then
            echo "Existing checkout does not match required revision: $path" >&2
            exit 1
        fi
        return
    fi
    mkdir -p "$(dirname "$path")"
    git init "$path"
    git -C "$path" remote add origin "$url"
    git -C "$path" fetch --depth 1 origin "$revision"
    git -C "$path" checkout --detach FETCH_HEAD
    if [[ -n "$tag" ]]; then
        # cuRobo uses setuptools-scm; retain its version tag in a shallow clone.
        git -C "$path" fetch --depth 1 origin "refs/tags/$tag:refs/tags/$tag"
        [[ "$(git -C "$path" rev-parse "$tag^{commit}")" == "$revision" ]]
    fi
}

checkout https://github.com/RoboTwin-Platform/RoboTwin.git \
    "$THIRD_PARTY_ROOT/RoboTwin" 2eeec322d95799f537cbfe5f291a8220d965ccb8
checkout https://github.com/NVlabs/curobo.git \
    "$THIRD_PARTY_ROOT/RoboTwin/envs/curobo" d64c4b005459db10c5dd867d8b30a87d5bda9bdb v0.7.8
checkout https://github.com/XPolicyLab/XPolicyLab.git \
    "$THIRD_PARTY_ROOT/XPolicyLab" fa431ecd893ee706883e64fe5fe1464ec8cd928d
echo "Optional sources ready. Follow docs/robotwin.md to install into separate environments."
