#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/hyhal/bin:${PATH}"

# Rocky Linux 8 ships an older libstdc++ in /lib64.  The CI base image also
# carries GCC 13, and binary extensions such as vLLM require its newer C++ ABI.
# Ask the active compiler for the matching runtime instead of hard-coding an
# image-specific GCC installation directory.
gcc_libstdcpp=""
declare -a libstdcpp_candidates=()
for compiler in g++ c++ gcc; do
    if command -v "${compiler}" >/dev/null 2>&1; then
        libstdcpp_candidates+=(
            "$("${compiler}" -print-file-name=libstdc++.so.6 2>/dev/null || true)"
        )
    fi
done

# The base image does not necessarily expose its GCC 13 installation through
# the default compiler command. Include toolkit and non-system prefixes when
# locating the runtime used by prebuilt C++ extensions.
while IFS= read -r candidate; do
    libstdcpp_candidates+=("${candidate}")
done < <(
    find /opt /usr/local /usr/lib64 /usr/lib /lib64 \
        -name 'libstdc++.so.6*' -print 2>/dev/null || true
)

for candidate in "${libstdcpp_candidates[@]}"; do
    [[ -f "${candidate}" ]] || continue
    candidate="$(readlink -f "${candidate}")"
    if grep -aFq 'GLIBCXX_3.4.29' "${candidate}"; then
        gcc_libstdcpp="${candidate}"
        break
    fi
done

if [[ -n "${gcc_libstdcpp}" ]]; then
    gcc_runtime_dir="$(dirname "${gcc_libstdcpp}")"
    export LD_LIBRARY_PATH="${gcc_runtime_dir}:/opt/hyhal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using GCC C++ runtime: ${gcc_libstdcpp}"
else
    # Compiling LightOp itself does not require the vLLM C++ ABI. Keep the
    # build running; the mandatory vLLM import check in the test phase will
    # still fail before pytest if no compatible runtime is available.
    export LD_LIBRARY_PATH="/opt/hyhal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "WARNING: No libstdc++.so.6 providing GLIBCXX_3.4.29 was found; vLLM validation may fail."
fi
