"""CI-only pytest policy for temporarily unavailable dependencies."""

from pathlib import Path

import pytest


VLLM_TEST_FILES = {
    "test_fuse_silu_mul_fp8_quant.py",
    "test_reshape_cache_cuda.py",
}
VLLM_SKIP_REASON = (
    "temporarily skipped: the CI image provides GLIBC 2.28, but the "
    "current vLLM native extension requires GLIBC 2.32"
)


def pytest_collection_modifyitems(items):
    """Report vLLM-dependent cases as skipped in the JUnit XML."""
    for item in items:
        if Path(str(item.path)).name in VLLM_TEST_FILES:
            item.add_marker(pytest.mark.skip(reason=VLLM_SKIP_REASON))
