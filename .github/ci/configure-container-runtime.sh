#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/hyhal/bin:${PATH}"

# Rocky Linux 8 ships an older libstdc++ in /lib64.  The CI base image also
# carries GCC 13, and binary extensions such as vLLM require its newer C++ ABI.
# Ask the active compiler for the matching runtime instead of hard-coding an
# image-specific GCC installation directory.
gcc_libstdcpp=""
for compiler in g++ c++ gcc; do
    if ! command -v "${compiler}" >/dev/null 2>&1; then
        continue
    fi
    candidate="$("${compiler}" -print-file-name=libstdc++.so.6 2>/dev/null || true)"
    if [[ -f "${candidate}" ]] && strings "${candidate}" | grep -qx 'GLIBCXX_3\.4\.29'; then
        gcc_libstdcpp="${candidate}"
        break
    fi
done

if [[ -z "${gcc_libstdcpp}" ]]; then
    echo "Compatible GCC runtime not found: libstdc++.so.6 must provide GLIBCXX_3.4.29." >&2
    exit 1
fi

gcc_runtime_dir="$(dirname "${gcc_libstdcpp}")"
export LD_LIBRARY_PATH="${gcc_runtime_dir}:/opt/hyhal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "Using GCC C++ runtime: ${gcc_libstdcpp}"
