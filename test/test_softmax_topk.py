import torch
import torch.nn.functional as F
import unittest
import pandas as pd
import os
import ctypes
from vllm import _custom_ops as ops
import pdb
# 尝试导入自定义算子，失败则使用 Mock
try:
    from lightop import op
    print("✅ 成功导入 lightop.op")
except (ImportError, AttributeError) as e:
    print(f"⚠️ 导入自定义算子失败: {e}")
    print("   将使用模拟函数代替。")
    

def torch_softmax_topk_base(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                              token_expert_indices: torch.Tensor,
                              gating_output: torch.Tensor,
                              renormalize: bool) -> tuple[torch.Tensor, ...]:
 
    probs = F.softmax(gating_output, dim=-1)

  
    k = topk_weights.size(-1)
    vals, indices = torch.topk(probs, k=k, dim=-1)


    topk_weights.copy_(vals)
    topk_indices.copy_(indices)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices
    

def vllm_topk_softmax_base(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                      token_expert_indices: torch.Tensor,
                      gating_output: torch.Tensor,
                      renormalize: bool) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


class TestTopkSoftmax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = []
        print("\n=== 开始 TopK Softmax 性能测试 (Custom vs vllm) ===")
        # 更新表头，增加 vllm 耗时和加速比
        print(f"{'Tokens':<10} | {'Custom (us)':<15} | {'vllm (us)':<15} | {'Speedup':<10} | {'Check':<10}")
        print("-" * 75)

    @classmethod
    def tearDownClass(cls):
        if not cls.results:
            return
        
        print("\n--- 测试结束，正在生成报告 ---")
        df = pd.DataFrame(cls.results)
        # 按 token 数量排序
        df = df.sort_values(by='num_tokens').reset_index(drop=True)
        
        output_filename = 'custom_op_performance.xlsx'
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"✅ 结果已保存: {os.path.abspath(output_filename)}\n")

    def _run_test(self, num_tokens: int, num_experts: int, topk: int, dtype: torch.dtype, Is_renormalize: str,
                  rtol=1e-5, atol=1e-5, warmup_iters=10, test_iters=100):
        
        if topk > num_experts:
            return

        # 1. 准备数据
        gating_output = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)
        
        # 准备自定义算子输出
        custom_topk_weights = torch.empty(num_tokens, topk, device="cuda", dtype=torch.float32)
        custom_topk_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)
        custom_token_expert_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)

        # 准备 vllm 基准算子输出 (需要独立的 Tensor 以免混淆)
        ref_topk_weights = torch.empty(num_tokens, topk, device="cuda", dtype=torch.float32)
        ref_topk_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)
        ref_token_expert_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)

        # # 2. 精度验证 (使用 vllm_topk_softmax_base 作为基准)
        # Is_renormalize = (Is_renormalize == "true")
        
        # 运行 vllm 基准
        ref_topk_weights, ref_topk_indices = vllm_topk_softmax_base(
            ref_topk_weights, ref_topk_indices, 
            ref_token_expert_indices, gating_output, 
            Is_renormalize
        )
        
        # 运行自定义算子进行验证
        op.topk_softmax(
            custom_topk_weights, custom_topk_indices,
            custom_token_expert_indices, gating_output, Is_renormalize)
        
        # --- 验证逻辑优化：处理 TopK 顺序不一致的情况 ---
        # 必须按 Expert Index 排序后，同步调整 Weights 的顺序再进行对比

        # 1. 处理 Reference 结果 (按 expert index 排序)
        ref_sorted_indices, ref_sort_idx = torch.sort(ref_topk_indices, dim=-1)
        # 根据索引排序的下标，重新排列权重
        ref_sorted_weights = torch.gather(ref_topk_weights, -1, ref_sort_idx)
        
        # 2. 处理 Custom 结果 (按 expert index 排序)
        custom_sorted_indices, custom_sort_idx = torch.sort(custom_topk_indices, dim=-1)
        # 根据索引排序的下标，重新排列权重
        custom_sorted_weights = torch.gather(custom_topk_weights, -1, custom_sort_idx)
        # pdb.set_trace()
        # 3. 分别验证索引和权重
        indices_match = torch.equal(custom_sorted_indices, ref_sorted_indices)
        weights_match = torch.allclose(custom_sorted_weights, ref_sorted_weights, rtol=rtol, atol=atol)

        status = "PASS" if weights_match and indices_match else "FAIL"
        
        if status == "FAIL":
            print(f"{num_tokens:<10} | {'ERROR':<15} | {'ERROR':<15} | {'N/A':<10} | {status:<10}")
            print(f"   >>> 错误详情: Weights={weights_match}, Indices={indices_match}")
            # 如果失败，打印最大误差
            if not weights_match:
                diff = torch.abs(custom_sorted_weights - ref_sorted_weights)
                print(f"   >>> 最大权重差异: {diff.max().item():.6f}")
            
            # 记录失败结果并抛出异常
            TestTopkSoftmax.results.append({
                'num_tokens': num_tokens,
                'num_experts': num_experts,
                'topk': topk,
                'custom_latency_us': -1,
                'vllm_latency_us': -1,
                'speedup': 0,
                'status': status
            })
            self.fail(f"精度验证失败: num_tokens={num_tokens}")

        # 3. 性能测试：Custom 算子
        # 预热
        for _ in range(warmup_iters):
            op.topk_softmax(
                custom_topk_weights, custom_topk_indices,
                custom_token_expert_indices, gating_output, Is_renormalize)
        torch.cuda.synchronize()

        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(test_iters):
            op.topk_softmax(
                custom_topk_weights, custom_topk_indices,
                custom_token_expert_indices, gating_output, Is_renormalize)
        end_event.record()
        torch.cuda.synchronize()
        
        # 计算耗时 (us)
        custom_time_us = start_event.elapsed_time(end_event) * 1000 / test_iters

        # 4. 性能测试：vllm 基准算子
        # 预热
        for _ in range(warmup_iters):
             vllm_topk_softmax_base(
                ref_topk_weights, ref_topk_indices, 
                ref_token_expert_indices, gating_output, 
                Is_renormalize
            )
        torch.cuda.synchronize()

        # 计时
        start_event.record()
        for _ in range(test_iters):
            vllm_topk_softmax_base(
                ref_topk_weights, ref_topk_indices, 
                ref_token_expert_indices, gating_output, 
                Is_renormalize
            )
        end_event.record()
        torch.cuda.synchronize()
        
        # 计算耗时 (us)
        vllm_time_us = start_event.elapsed_time(end_event) * 1000 / test_iters

        # 计算加速比
        speedup = vllm_time_us / custom_time_us if custom_time_us > 0 else 0

        # 5. 打印精简信息
        print(f"{num_tokens:<10} | {custom_time_us:<15.4f} | {vllm_time_us:<15.4f} | {speedup:<10.2f} | {status:<10}")

        # 6. 记录结果
        TestTopkSoftmax.results.append({
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'topk': topk,
            'custom_latency_us': custom_time_us,
            'vllm_latency_us': vllm_time_us,
            'speedup': speedup,
            'status': status
        })

    def test_specific_token_counts(self):
        num_experts = 160
        topk = 8
        # num_tokens_list = [2048] # 调试用
        num_tokens_list = [1,6,8,32,64,128,256,356,512,1024,2048,4096,8192,16384,16384*2,41000]
        
        Is_renormalize = True

        for num_tokens in num_tokens_list:
            self._run_test(
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                dtype=torch.float32,
                Is_renormalize=Is_renormalize,
                rtol=1e-5,
                atol=1e-5
            )

if __name__ == '__main__':
    unittest.main()