#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-/lightop}"
test_root="${source_root}/test"

if [[ ! -d "${test_root}" ]]; then
    echo "LightOp test directory was not found: ${test_root}" >&2
    exit 1
fi

mapfile -t cases < <(
    find "${test_root}" -maxdepth 1 -type f -name 'test*' -print | sort
)

extra_cases=(
    "${test_root}/moe_w8a8/release_moe_w8a8.py"
    "${test_root}/moe_marlin/moe_w4a16/benchmark_deepseek_lightop.py"
    "${test_root}/w8a8/w8a8_test.py"
    "${test_root}/moe_marlin/test/w4a16/test_deepseek_lightop.py"
)
for case_file in "${extra_cases[@]}"; do
    if [[ -f "${case_file}" ]]; then
        cases+=("${case_file}")
    fi
done

if [[ "${#cases[@]}" -eq 0 ]]; then
    echo "No LightOp fallback test cases were discovered." >&2
    exit 1
fi

executed=0
for case_file in "${cases[@]}"; do
    if [[ "$(basename "${case_file}")" == "test_bf16_deepgemm_contig.py" ]]; then
        echo "Excluded known long-running case: ${case_file#${source_root}/}"
        continue
    fi

    work_dir="$(mktemp -d /tmp/lightop-fallback.XXXXXX)"
    echo "Running fallback case: ${case_file#${source_root}/}"
    set +e
    (cd "${work_dir}" && timeout 3600 python3 "${case_file}")
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
