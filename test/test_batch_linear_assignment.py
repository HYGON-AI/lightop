import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from lightop import batch_linear_assignment
import unittest
import time

class TestLightOpCorrectness(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n--- Running test on device: {self.device} ---")
        if self.device == "cuda":
            try:
                # 全局预热 (1)
                dummy_cost = torch.rand(2, 2, device=self.device)
                _ = batch_linear_assignment(dummy_cost)
            except Exception as e:
                print(f"Warning: Global pre-warmup failed (this might be ok): {e}")

    def _run_comparison(self, W, T):
        
        print(f"\n--- Testing Shape: (W={W}, T={T}), Target: W > T ---")

        # 1. 生成数据 (在 CPU 上)
        cost_matrix_2d_cpu = torch.rand(W, T, dtype=torch.float32)

        # ==========================================================
        # 2. 运行 SciPy (获取“标准答案”) - 保持不变
        # ==========================================================
        start_time = time.time()
        print("Running SciPy linear_sum_assignment (CPU)...")
        
        cost_matrix_np = cost_matrix_2d_cpu.numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
        # 将 SciPy 结果移至 GPU，用于断言比较
        matched_row_inds = torch.as_tensor(row_ind, device=self.device)
        matched_col_inds = torch.as_tensor(col_ind, device=self.device)
        scipy_time = time.time() - start_time
        print(f"SciPy matched pairs: {row_ind.size}")
        print(f"SciPy time: {scipy_time:.6f} s")

        # ==========================================================
        # 3. 运行 lightop 算子 (GPU 操作)
        # ==========================================================
        print(f"Running lightop.batch_linear_assignment ({self.device})...")
        cost = cost_matrix_2d_cpu.to(self.device)
        _ = batch_linear_assignment(cost)
        if self.device == "cuda":
            torch.cuda.synchronize() 

        # --- 计时执行 (Timed Execution) ---
        start_time = time.time()
        row_ind_lightop, col_ind_lightop = batch_linear_assignment(cost)
        # w, t = cost.shape 
        # if t < w:
        #     row_ind_lightop, col_ind_lightop = assignment_to_indices(result_tensor, t) 
        # else:
        #     row_ind_lightop, col_ind_lightop = assignment_to_indices(result_tensor, w) 
        if self.device == "cuda":
            torch.cuda.synchronize()    
        op_time = time.time() - start_time
        
        print(f"lightop time ({self.device}): {op_time:.6f} s")
        # ==========================================================
        # 4. 断言 (Assert)
        # ==========================================================
        print("\n--- Comparing SciPy (Ground Truth) vs. lightop (Result) ---")
        print("matched_row_inds", matched_row_inds)
        print("matched_col_inds", matched_col_inds)
        print("row_ind_lightop", row_ind_lightop)
        print("col_ind_lightop", col_ind_lightop)
        
        # 比较行索引 (Worker IDs)
        self.assertTrue(
            torch.all(matched_row_inds == row_ind_lightop),
            "!!! FAILED: Worker (行) 索引不匹配！"
        )
        
        # 比较列索引 (Task IDs)
        self.assertTrue(
            torch.all(matched_col_inds == col_ind_lightop),
            "!!! FAILED: Task (列) 索引不匹配！"
        )
        
        print("\n✅ Test PASSED! 与 SciPy 索引完全一致。")


    def test_W_greater_than_T_case(self):
        """测试 W > T (工人多于任务) 的关键情况：[64, 32]"""
        self._run_comparison(W=64, T=32)
        # self._run_comparison(W=32, T=64)


if __name__ == '__main__':
    unittest.main()
