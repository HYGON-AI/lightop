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

git check-ref-format --branch "${source_branch}" >/dev/null

case "${context_dir}" in
    "${RUNNER_TEMP}"/*) ;;
    *)
        echo "Build context must be located under RUNNER_TEMP: ${context_dir}" >&2
        exit 2
        ;;
esac

mkdir -p "${context_dir}/lightop"

# Preserve .git so a future OpTest LightOp implementation can identify the
# checked-out revision, while excluding outputs from earlier CI attempts.
tar \
    --exclude='./build' \
    --exclude='./dist' \
    --exclude='./test_results' \
    --exclude='./.ci-artifacts' \
    -C "${source_dir}" -cf - . | \
    tar -C "${context_dir}/lightop" -xf -

# Keep the triggering source branch visible to package metadata and OpTest.
# The checked-out contents already include the temporary integration with the
# latest master performed by the workflow.
git -C "${context_dir}/lightop" checkout -B "${source_branch}"

wget -q -O - "${dtk_url}" | tar -xzf - -C "${context_dir}"

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

cp "${source_dir}/.github/ci/Dockerfile" "${context_dir}/Dockerfile"
cp "${source_dir}/.github/ci/build-lightop-wheel.sh" "${context_dir}/build-lightop-wheel.sh"

# Convert the selected DTK environment into Dockerfile ENV declarations using
# the same transformation as CIctl docker/tools/generate_dtk_env.sh.
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
cat > "${context_dir}/build-environment.txt" <<EOF
DTK_PACKAGE=${dtk_package}
DTK_PACKAGE_VERSION=${dtk_package_version}
DTK_VERSION=${dtk_version}
AICC_VERSION=${latest_aicc%.run}
EOF

printf 'DTK package: %s\nDTK version: %s\nAICC version: %s\n' \
    "${dtk_package}" "${dtk_version}" "${latest_aicc%.run}"
