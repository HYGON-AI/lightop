"""JUnit-visible placeholders for tests blocked by the vLLM runtime."""

import pytest

from conftest import VLLM_TEST_FILES


VLLM_SKIP_REASON = (
    "temporarily skipped: the CI image provides GLIBC 2.28, but the "
    "current vLLM native extension requires GLIBC 2.32"
)


@pytest.mark.parametrize("test_file", sorted(VLLM_TEST_FILES), ids=lambda path: path)
@pytest.mark.skip(reason=VLLM_SKIP_REASON)
def test_vllm_dependent_file(test_file):
    """Expose each omitted vLLM-dependent test file in the JUnit report."""
