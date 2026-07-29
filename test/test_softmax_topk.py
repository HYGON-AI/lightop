import csv
import glob
import os
import sys
import unittest

import torch

# Running this file from the repository makes the source-only ``lightop``
# package shadow the compiled extension. Prefer the current build output when
# it exists, while retaining the normal installed-package path elsewhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_LIB_DIRS = sorted(glob.glob(os.path.join(REPO_ROOT, "build", "lib.*")))
if BUILD_LIB_DIRS:
    sys.path.insert(0, BUILD_LIB_DIRS[-1])

from lightop import op


NUM_EXPERTS = 256
TOPK = 8
DTYPE = torch.float16
TOKEN_SIZES = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def validate_topk_result(
    testcase: unittest.TestCase,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: torch.Tensor,
) -> float:
    """Validate top-k values without requiring a specific ID for equal logits."""
    num_tokens = gating_output.size(0)

    testcase.assertTrue(torch.all((topk_indices >= 0) & (topk_indices < NUM_EXPERTS)))
    sorted_indices = torch.sort(topk_indices, dim=-1).values
    if TOPK > 1:
        testcase.assertTrue(torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]))

    selected_logits = torch.gather(
        gating_output.float(), 1, topk_indices.to(torch.int64)
    )
    selected_logits_sorted = torch.sort(
        selected_logits, dim=-1, descending=True
    ).values
    reference_logits = torch.topk(
        gating_output.float(), TOPK, dim=-1, sorted=True
    ).values
    testcase.assertTrue(torch.equal(selected_logits_sorted, reference_logits))

    reference_weights = torch.softmax(selected_logits, dim=-1)
    max_weight_error = (topk_weights - reference_weights).abs().max().item()
    testcase.assertTrue(
        torch.allclose(topk_weights, reference_weights, rtol=2e-5, atol=2e-6),
        msg=f"max weight error: {max_weight_error:.6e}",
    )

    rows = torch.arange(num_tokens, device=gating_output.device)[:, None]
    ranks = torch.arange(TOPK, device=gating_output.device)[None, :]
    expected_source_rows = ranks * num_tokens + rows
    testcase.assertTrue(
        torch.equal(source_rows, expected_source_rows.to(source_rows.dtype))
    )
    return max_weight_error


def benchmark_topk(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    source_rows: torch.Tensor,
    warmup_iters: int,
    test_iters: int,
) -> float:
    for _ in range(warmup_iters):
        op.topk_softmax(
            topk_weights, topk_indices, source_rows, gating_output, True
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(test_iters):
        op.topk_softmax(
            topk_weights, topk_indices, source_rows, gating_output, True
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / test_iters


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm device is required")
class TestTopkSoftmax256Fp16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = []
        print("\n=== TopK Softmax: FP16, 256 experts, topk=8, renormalize=True ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"{'Tokens':>8} | {'Latency (us)':>12} | {'Max weight error':>16} | Check")
        print("-" * 62)

    @classmethod
    def tearDownClass(cls):
        output_path = os.environ.get(
            "TOPK_BENCH_OUTPUT", "softmax_topk_256_fp16_results.csv"
        )
        if not cls.results:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=cls.results[0].keys())
            writer.writeheader()
            writer.writerows(cls.results)
        print(f"Results saved to: {os.path.abspath(output_path)}")

    def test_target_sizes(self):
        warmup_iters = int(os.environ.get("TOPK_BENCH_WARMUP", "100"))
        default_iters = int(os.environ.get("TOPK_BENCH_ITERS", "1000"))

        for num_tokens in TOKEN_SIZES:
            with self.subTest(num_tokens=num_tokens):
                torch.manual_seed(2026 + num_tokens)
                gating_output = torch.randn(
                    (num_tokens, NUM_EXPERTS), device="cuda", dtype=DTYPE
                )
                topk_weights = torch.empty(
                    (num_tokens, TOPK), device="cuda", dtype=torch.float32
                )
                topk_indices = torch.empty(
                    (num_tokens, TOPK), device="cuda", dtype=torch.int32
                )
                source_rows = torch.empty_like(topk_indices)

                op.topk_softmax(
                    topk_weights,
                    topk_indices,
                    source_rows,
                    gating_output,
                    True,
                )
                torch.cuda.synchronize()
                max_weight_error = validate_topk_result(
                    self,
                    gating_output,
                    topk_weights,
                    topk_indices,
                    source_rows,
                )

                test_iters = (
                    default_iters
                    if num_tokens <= 4096
                    else max(100, default_iters // 4)
                )
                latency_us = benchmark_topk(
                    gating_output,
                    topk_weights,
                    topk_indices,
                    source_rows,
                    warmup_iters,
                    test_iters,
                )
                print(
                    f"{num_tokens:8d} | {latency_us:12.3f} | "
                    f"{max_weight_error:16.3e} | PASS"
                )
                self.results.append(
                    {
                        "num_tokens": num_tokens,
                        "num_experts": NUM_EXPERTS,
                        "topk": TOPK,
                        "dtype": str(DTYPE),
                        "renormalize": True,
                        "latency_us": f"{latency_us:.6f}",
                        "max_weight_error": f"{max_weight_error:.6e}",
                        "status": "PASS",
                    }
                )


if __name__ == "__main__":
    unittest.main()
