"""Runtime checks for the remaining categorized normalization operators."""

import pytest
import torch

from lightop import op as lightop_native
from lightop.norm import (
    layer_norm_fwd_1pass_opt,
    rms_norm_no_weight,
    rmsnorm_forward_autograd,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device is required")


def test_remaining_norm_bindings_are_direct_reexports() -> None:
    assert layer_norm_fwd_1pass_opt is (
        lightop_native.layer_norm_fwd_1pass_opt
    )
    assert rms_norm_no_weight is lightop_native.rms_norm_no_weight
    assert rmsnorm_forward_autograd is (
        lightop_native.rmsnorm_forward_autograd
    )


def test_layer_norm_fwd_1pass_opt_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(31)
    rows, hidden_size = 4, 128
    inputs = torch.randn(
        (rows, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    weights = torch.randn(
        (hidden_size,), device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty_like(inputs)
    native_output = torch.empty_like(inputs)
    rstd = torch.empty((rows,), device="cuda", dtype=torch.float32)
    native_rstd = torch.empty_like(rstd)

    common = (
        inputs,
        weights,
        None,
        None,
        None,
        hidden_size,
        hidden_size,
        hidden_size,
        rows,
        hidden_size,
        1e-6,
        hidden_size,
        1,
        False,
        False,
        False,
        True,
        "none",
    )
    layer_norm_fwd_1pass_opt(
        common[0], output, *common[1:5], rstd, *common[5:]
    )
    lightop_native.layer_norm_fwd_1pass_opt(
        common[0], native_output, *common[1:5], native_rstd, *common[5:]
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, native_output, rtol=0, atol=0)
    torch.testing.assert_close(rstd, native_rstd, rtol=0, atol=0)


def test_rms_norm_no_weight_inplace_matches_native() -> None:
    _require_cuda()
    torch.manual_seed(32)
    inputs = torch.randn(
        (4, 256), device="cuda", dtype=torch.bfloat16
    )
    native_inputs = inputs.clone()

    rms_norm_no_weight(None, inputs, None, 1e-6)
    lightop_native.rms_norm_no_weight(
        None, native_inputs, None, 1e-6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(inputs, native_inputs, rtol=0, atol=0)


def test_rmsnorm_forward_autograd_matches_native_gradients() -> None:
    _require_cuda()
    torch.manual_seed(33)
    inputs = torch.randn(
        (3, 256),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weights = torch.randn(
        (256,),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    native_inputs = inputs.detach().clone().requires_grad_(True)
    native_weights = weights.detach().clone().requires_grad_(True)

    output = rmsnorm_forward_autograd(inputs, weights, 1e-6, True)
    native_output = lightop_native.rmsnorm_forward_autograd(
        native_inputs, native_weights, 1e-6, True
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    native_output.backward(grad)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, native_output, rtol=0, atol=0)
    torch.testing.assert_close(
        inputs.grad, native_inputs.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        weights.grad, native_weights.grad, rtol=1e-4, atol=1e-4
    )
