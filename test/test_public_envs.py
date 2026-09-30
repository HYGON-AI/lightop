# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from lightop import envs as public_envs
from lightop._lmslim_native import envs as legacy_envs


EXPECTED_ENV_NAMES = {
    "LMSLIM_GPU_NAME",
    "LMSLIM_USE_LIGHTOP",
    "LMSLIM_FUSED_MOE_CHUNK_SIZE",
    "LMSLIM_USE_GLOBAL_MOE_CACHE",
    "LMSLIM_USE_FUSED_RMS_QUANT",
    "LMSLIM_USE_FUSE_SILU_AND_MUL",
}


def test_public_envs_exports_runtime_settings() -> None:
    assert EXPECTED_ENV_NAMES <= set(public_envs.__all__)
    for name in EXPECTED_ENV_NAMES:
        assert hasattr(public_envs, name)


def test_private_envs_path_remains_a_compatibility_shim() -> None:
    assert legacy_envs.__all__ == public_envs.__all__
    for name in public_envs.__all__:
        assert getattr(legacy_envs, name) == getattr(public_envs, name)
