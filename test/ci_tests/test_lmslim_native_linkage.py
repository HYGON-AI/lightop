"""Run against the installed wheel on a Linux DTK GPU runner.

LIGHTOP_TEST_PACKAGE can override the package name for an alternate build.
Missing tools/devices fail the gate instead of reporting a skipped success.
"""
import importlib
import os
import subprocess
import sys

import pytest
import torch


PACKAGE = os.environ.get("LIGHTOP_TEST_PACKAGE", "lightop")


@pytest.fixture(scope="module")
def native():
    package = importlib.import_module(PACKAGE)
    print("package:", package.__file__)
    print("extension:", package.op.__file__)
    return package.op


def test_import_in_fresh_process():
    result = subprocess.run(
        [sys.executable, "-c",
         "import importlib,sys; p=importlib.import_module(sys.argv[1]); "
         "print(p.__file__); print(p.op.__file__)", PACKAGE],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout
    print(result.stdout)


def test_half_conversion_is_not_a_runtime_import(native):
    # Inspect .dynsym without assuming the wheel retained its .symtab.
    output = subprocess.check_output(
        ["readelf", "--dyn-syms", "--wide", native.__file__], text=True
    )
    unresolved = [line for line in output.splitlines()
                  if "__extendhfsf2" in line and " UND " in line]
    assert not unresolved, "\n".join(unresolved)


@pytest.mark.parametrize("layout_tn", [False, True])
def test_native_rocblas_fp16_matches_independent_torch(native, layout_tn):
    assert torch.version.hip and torch.cuda.is_available(), "DTK GPU required"
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    a = torch.randn((16, 64), generator=generator, device="cuda", dtype=torch.float16)
    b = torch.randn((64, 32), generator=generator, device="cuda", dtype=torch.float16)
    actual = native._lmslimquant_rocblas_gemm_fp16(a, b, layout_tn)
    # Legacy TN contract: storage is N*K while tensor metadata remains K*N.
    logical_b = b.view(32, 64).T if layout_tn else b
    expected = (a.cpu().float() @ logical_b.cpu().float()).half()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-2)


def test_native_permute_matches_independent_transpose(native):
    assert torch.version.hip and torch.cuda.is_available(), "DTK GPU required"
    source_cpu = torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128)
    source = source_cpu.cuda()
    output = torch.empty_like(source)
    native._lmslimquant_permute_trans(output, source, 4, 128)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.view(128, 4).cpu(), source_cpu.T, rtol=0, atol=0)


def _pack_u4_cpu(values):
    values = values.to(torch.int64).reshape(*values.shape[:-1], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return ((values & 15) << shifts).sum(dim=-1).to(torch.int32)


@pytest.mark.parametrize("k,n,group_size", [(128, 128, 128), (256, 128, 128)])
def test_native_awq_format_and_dequant_against_affine_reference(native, k, n, group_size):
    assert torch.version.hip and torch.cuda.is_available(), "DTK GPU required"
    # Block-constant logical values avoid making the reference depend on the
    # vendor's internal permutation within each 8-element packing tile.
    q = (torch.arange(k)[:, None] // 8 + torch.arange(n)[None, :] // 8) % 16
    groups = k // group_size
    zeros = (torch.arange(groups)[:, None] + torch.arange(n)[None, :] // 8) % 16
    scales = (2.0 ** (-1 - (torch.arange(n)[None, :] // 8) % 4)).expand(groups, n).half()
    qweight = _pack_u4_cpu(q).cuda()
    qzeros = _pack_u4_cpu(zeros).cuda()
    device_scales = scales.contiguous().cuda()
    converted = torch.zeros_like(qweight)
    scale_zero = torch.zeros((groups, n), device="cuda", dtype=torch.int32)
    workspace = torch.empty((groups, n), device="cuda", dtype=torch.float16)
    native._lmslimquant_convert_s4_k_m8(
        converted, scale_zero, workspace, qweight, device_scales, qzeros, n, k, group_size
    )
    transposed = torch.empty_like(scale_zero)
    native._lmslimquant_permute_trans(transposed, scale_zero, groups, n)
    actual = torch.empty((n, k), device="cuda", dtype=torch.float16)
    native._lmslimquant_dequant_w4_gemm_colmajor(
        actual, converted, transposed.view(n, groups), k, n, group_size
    )
    group_index = torch.arange(k) // group_size
    expected = ((q.float() - zeros[group_index].float()) * scales[group_index].float()).T.half()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
