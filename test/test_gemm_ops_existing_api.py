"""Compatibility checks for existing LightOp GEMM category exports."""

import pytest

import lightop
import lightop.gemm_ops


@pytest.mark.parametrize("name", lightop.gemm_ops.__all__)
def test_existing_gemm_api_is_direct_reexport(name: str) -> None:
    assert getattr(lightop.gemm_ops, name) is getattr(lightop, name)


def test_importing_gemm_ops_does_not_shadow_top_level_gemm() -> None:
    assert callable(lightop.gemm)
    assert lightop.gemm_ops.gemm is lightop.gemm
