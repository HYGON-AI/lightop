#!/usr/bin/env bash
set -euo pipefail

target_branch="${GITHUB_BASE_REF:-master}"
target_ref="${1:-origin/${target_branch}}"
source_ref="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-unknown}}"
source_sha="$(git rev-parse HEAD)"
target_sha="$(git rev-parse "${target_ref}")"

git -c user.name='LightOp CI' \
    -c user.email='ci@localhost' \
    merge --no-edit "${target_sha}"

integrated_sha="$(git rev-parse HEAD)"
echo "Source branch: ${source_ref}"
echo "Source commit: ${source_sha}"
echo "Target branch: ${target_branch}"
echo "Target commit: ${target_sha}"
echo "Integrated commit: ${integrated_sha}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "target_sha=${target_sha}" >> "${GITHUB_OUTPUT}"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    {
        echo "### Integrated LightOp source"
        echo
        echo "- Source branch: \`${source_ref}\`"
        echo "- Source commit: \`${source_sha}\`"
        echo "- Target branch: \`${target_branch}\`"
        echo "- Target commit: \`${target_sha}\`"
        echo "- Integrated commit: \`${integrated_sha}\`"
    } >> "${GITHUB_STEP_SUMMARY}"
fi
