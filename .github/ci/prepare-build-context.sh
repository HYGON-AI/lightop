#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <context-dir> <source-dir> <dtk-url> <aicc-nightly-dir> <source-branch>" >&2
    exit 2
fi

context_dir="$(realpath -m "$1")"
source_dir="$(realpath "$2")"
dtk_url="$3"
aicc_nightly_dir="$(realpath "$4")"
source_branch="$5"
source_commit="$(git -C "${source_dir}" rev-parse HEAD)"

git check-ref-format --branch "${source_branch}" >/dev/null
if [[ ! "${source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid source commit: ${source_commit}" >&2
    exit 2
fi

: "${RUNNER_TEMP:?RUNNER_TEMP must be set}"
runner_temp="$(realpath -m "${RUNNER_TEMP}")"

case "${context_dir}" in
    "${runner_temp}"/*) ;;
    *)
        echo "Build context must be located under RUNNER_TEMP: ${context_dir}" >&2
        exit 2
        ;;
esac

rm -rf -- "${context_dir}"
mkdir -p "${context_dir}/lightop"

# Archive only the integrated tracked revision. This excludes .git, ignored
# files, untracked files, and outputs left by earlier runner jobs.
git -C "${source_dir}" archive --format=tar "${source_commit}" | \
    tar -C "${context_dir}/lightop" -xf -

dtk_archive="${context_dir}/.dtk-package.tar.gz"
wget --quiet --tries=3 --timeout=30 -O "${dtk_archive}" "${dtk_url}"
tar -tzf "${dtk_archive}" >/dev/null
tar -xzf "${dtk_archive}" -C "${context_dir}"
rm -f "${dtk_archive}"

mapfile -t dtk_dirs < <(find "${context_dir}" -mindepth 1 -maxdepth 1 -type d -name 'dtk-*' -print)
if [[ "${#dtk_dirs[@]}" -ne 1 ]]; then
    echo "Expected exactly one extracted dtk-* directory, found ${#dtk_dirs[@]}" >&2
    exit 1
fi
mv "${dtk_dirs[0]}" "${context_dir}/dtk"

latest_aicc="$(
    find "${aicc_nightly_dir}" -maxdepth 1 -type f -name 'dtk_llvm_*.run' -printf '%f\n' |
        sed 's/\.run$//' |
        sort -t '_' -k4 -nr |
        sed -n '1p'
)"
if [[ -z "${latest_aicc}" ]]; then
    echo "No dtk_llvm_*.run found in ${aicc_nightly_dir}" >&2
    exit 1
fi
latest_aicc="${latest_aicc}.run"
cp "${aicc_nightly_dir}/${latest_aicc}" "${context_dir}/${latest_aicc}"
chmod +x "${context_dir}/${latest_aicc}"

cp "${context_dir}/lightop/.github/ci/Dockerfile" "${context_dir}/Dockerfile"
cp "${context_dir}/lightop/.github/ci/build-lightop-wheel.sh" "${context_dir}/build-lightop-wheel.sh"

# Convert the selected DTK environment into Dockerfile ENV declarations.
dtk_env_tmp="${context_dir}/dtk-env-generator.sh"
cp "${context_dir}/dtk/env.sh" "${dtk_env_tmp}"
sed -i -E \
    "s/^([[:space:]]*)export([[:space:]]+)([^=]+)[[:space:]]*=[[:space:]]*(.+)$/\\1echo 'ENV \\3=\\4'/g" \
    "${dtk_env_tmp}"
HYHAL_PATH=/opt/hyhal bash "${dtk_env_tmp}" |
    grep ENV |
    grep -v DTKROOT > "${context_dir}/dtk.env"
rm -f "${dtk_env_tmp}"
sed -i "/# config dtk env/r ${context_dir}/dtk.env" "${context_dir}/Dockerfile"

dtk_package="$(basename "${dtk_url}")"
dtk_version="$(tr -d '.' < "${context_dir}/dtk/.info/rocm_version")"
dtk_package_stem="${dtk_package%.tar.gz}"
IFS='-' read -r -a dtk_parts <<< "${dtk_package_stem}"
dtk_version_parts=()
for ((i = 1; i < ${#dtk_parts[@]} - 2; i++)); do
    part="${dtk_parts[i]}"
    if [[ "${i}" -eq 1 ]]; then
        part="${part//./}"
    fi
    dtk_version_parts+=("${part,,}")
done
dtk_package_version="dtk$(IFS=-; echo "${dtk_version_parts[*]}")"
{
    printf 'DTK_PACKAGE=%q\n' "${dtk_package}"
    printf 'DTK_PACKAGE_VERSION=%q\n' "${dtk_package_version}"
    printf 'DTK_VERSION=%q\n' "${dtk_version}"
    printf 'AICC_VERSION=%q\n' "${latest_aicc%.run}"
    printf 'LIGHTOP_SOURCE_BRANCH=%q\n' "${source_branch}"
    printf 'LIGHTOP_SOURCE_COMMIT=%q\n' "${source_commit}"
} > "${context_dir}/build-environment.txt"

printf 'DTK package: %s\nDTK version: %s\nAICC version: %s\nSource branch: %s\nSource commit: %s\n' \
    "${dtk_package}" "${dtk_version}" "${latest_aicc%.run}" \
    "${source_branch}" "${source_commit}"
