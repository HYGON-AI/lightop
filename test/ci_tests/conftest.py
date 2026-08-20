"""CI-only pytest policy for temporarily unavailable dependencies."""


VLLM_TEST_FILES = {
    "test_fuse_silu_mul_fp8_quant.py",
    "test_reshape_cache_cuda.py",
}
# Do not import these modules: importing their vLLM baseline attempts to load
# an extension that needs GLIBC 2.32. test_vllm_dependency_skips.py reports one
# explicit skipped JUnit testcase for each ignored module.
collect_ignore = sorted(VLLM_TEST_FILES)
