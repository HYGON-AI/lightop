#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-/lightop}"
test_root="${source_root}/test"

if [[ ! -d "${test_root}" ]]; then
    echo "LightOp test directory was not found: ${test_root}" >&2
    exit 1
fi

relative_cases=(
    test/moe_marlin/test/awq_marlin_repack/test_marlin_utils.py
    test/moe_marlin/test/awq_marlin_repack_w4a8/test_marlin_utils.py
    test/test_gemm.py
    test/test_moe_align_block_size.py
    test/test_rmsnorm.py
    test/test_rotary_positional_embedding.py
    test/test_store_kv_cache.py
    test/test_scaled_masked_softmax.py
)
cases=()
for relative_case in "${relative_cases[@]}"; do
    case_file="${source_root}/${relative_case}"
    if [[ ! -f "${case_file}" ]]; then
        echo "Required fallback case was not found: ${relative_case}" >&2
        exit 1
    fi
    cases+=("${case_file}")
done

if [[ "${#cases[@]}" -eq 0 ]]; then
    echo "No LightOp fallback test cases were discovered." >&2
    exit 1
fi

executed=0
for case_file in "${cases[@]}"; do
    work_dir="$(mktemp -d /tmp/lightop-fallback.XXXXXX)"
    echo "Running fallback case: ${case_file#${source_root}/}"
    set +e
    if grep -Eq '^(async[[:space:]]+)?def[[:space:]]+test_|^class[[:space:]]+Test' "${case_file}"; then
        (cd "${work_dir}" && timeout 3600 python3 -m pytest -v "${case_file}")
    else
        (cd "${work_dir}" && timeout 3600 python3 "${case_file}")
    fi
    status=$?
    set -e
    if [[ "${status}" -ne 0 ]]; then
        rm -rf -- "${work_dir}"
        echo "Fallback case failed: ${case_file#${source_root}/}" >&2
        exit "${status}"
    fi
    rm -rf -- "${work_dir}"
    executed=$((executed + 1))
done

if [[ "${executed}" -eq 0 ]]; then
    echo "No LightOp fallback test cases were executed." >&2
    exit 1
fi

echo "LightOp fallback tests passed: ${executed} case files."
