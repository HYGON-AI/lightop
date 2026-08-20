#!/usr/bin/env bash
set -euo pipefail

# Rocky Linux 8's system libstdc++ is older than the one used to build vLLM.
# Select the GCC runtime bundled in the compile image by symbol capability
# rather than relying on a base-image-specific installation path.
gcc_runtime_dir="$({
    find /opt/rh /usr/local /opt/dtk \
        -type f -name 'libstdc++.so.6*' -print 2>/dev/null || true
} | while IFS= read -r candidate; do
    if strings "${candidate}" | grep -Fx 'GLIBCXX_3.4.29' >/dev/null; then
        dirname "${candidate}"
        break
    fi
done)"

if [[ -z "${gcc_runtime_dir}" ]]; then
    echo "No libstdc++ runtime providing GLIBCXX_3.4.29 was found." >&2
    return 1
fi

export PATH="/opt/hyhal/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${gcc_runtime_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "Using GCC runtime: ${gcc_runtime_dir}/libstdc++.so.6"
