#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <context-dir> <source-dir> <dtk-url> <aicc-dir> <aicc-package>" >&2
    exit 2
fi

context_dir="$(realpath -m "$1")"
source_dir="$(realpath "$2")"
dtk_url="$3"
aicc_dir="$(realpath "$4")"
aicc_package="$5"

if [[ ! "${dtk_url}" =~ ^https?:// ]]; then
    echo "The toolkit package URL must use HTTP or HTTPS." >&2
    exit 2
fi
dtk_package="$(basename "${dtk_url%%\?*}")"
if [[ ! "${dtk_package}" =~ ^[A-Za-z0-9._-]+\.tar\.gz$ ]]; then
    echo "The toolkit URL must identify a fixed .tar.gz package." >&2
    exit 2
fi
if [[ "${aicc_package}" == */* || ! "${aicc_package}" =~ ^dtk_llvm_[A-Za-z0-9._-]+\.run$ ]]; then
    echo "The compiler package name is invalid." >&2
    exit 2
fi

case "${context_dir}" in
    "${RUNNER_TEMP}"/*) ;;
    *)
        echo "Build context must be located under RUNNER_TEMP: ${context_dir}" >&2
        exit 2
        ;;
esac

mkdir -p "${context_dir}/lightop"

# Never copy checkout metadata or credentials into the Docker build context.
tar \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./dist' \
    --exclude='./test_results' \
    --exclude='./.ci-artifacts' \
    -C "${source_dir}" -cf - . | \
    tar -C "${context_dir}/lightop" -xf -

dtk_archive="${context_dir}/${dtk_package}"
wget -q -O "${dtk_archive}" "${dtk_url}"
dtk_sha256="$(sha256sum "${dtk_archive}" | awk '{print $1}')"
tar -xzf "${dtk_archive}" -C "${context_dir}"
rm -f "${dtk_archive}"

mapfile -t dtk_dirs < <(find "${context_dir}" -mindepth 1 -maxdepth 1 -type d -name 'dtk-*' -print)
if [[ "${#dtk_dirs[@]}" -ne 1 ]]; then
    echo "Expected exactly one extracted dtk-* directory, found ${#dtk_dirs[@]}" >&2
    exit 1
fi
mv "${dtk_dirs[0]}" "${context_dir}/dtk"

aicc_source="${aicc_dir}/${aicc_package}"
if [[ ! -f "${aicc_source}" ]]; then
    echo "Configured compiler package was not found." >&2
    exit 1
fi
cp "${aicc_source}" "${context_dir}/${aicc_package}"
aicc_sha256="$(sha256sum "${context_dir}/${aicc_package}" | awk '{print $1}')"
chmod +x "${context_dir}/${aicc_package}"

cp "${source_dir}/.github/ci/Dockerfile" "${context_dir}/Dockerfile"
cp "${source_dir}/.github/ci/build-lightop-wheel.sh" "${context_dir}/build-lightop-wheel.sh"

# Convert the selected DTK environment into Dockerfile ENV declarations using
# a deterministic transformation of the selected package environment.
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
AICC_VERSION=${aicc_package%.run}
DTK_PACKAGE_SHA256=${dtk_sha256}
AICC_PACKAGE_SHA256=${aicc_sha256}
EOF

printf 'DTK package: %s\nDTK version: %s\nDTK SHA-256: %s\nAICC version: %s\nAICC SHA-256: %s\n' \
    "${dtk_package}" "${dtk_version}" "${dtk_sha256}" \
    "${aicc_package%.run}" "${aicc_sha256}"
