#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/master}"
source_ref="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-unknown}}"
source_sha="$(git rev-parse HEAD)"
base_sha="$(git rev-parse "${base_ref}")"

git -c user.name='LightOp CI' \
    -c user.email='ci@localhost' \
    merge --no-edit "${base_sha}"

integrated_sha="$(git rev-parse HEAD)"
echo "Source branch: ${source_ref}"
echo "Source commit: ${source_sha}"
echo "Latest master: ${base_sha}"
echo "Integrated commit: ${integrated_sha}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "base_sha=${base_sha}" >> "${GITHUB_OUTPUT}"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Integrated LightOp source"
        echo
        echo "- Source branch: \`${source_ref}\`"
        echo "- Source commit: \`${source_sha}\`"
        echo "- Latest master: \`${base_sha}\`"
        echo "- Integrated commit: \`${integrated_sha}\`"
    } >> "${GITHUB_STEP_SUMMARY}"
fi
