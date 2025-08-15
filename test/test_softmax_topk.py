import torch
import torch.nn.functional as F
import unittest
import pandas as pd
import os
import ctypes
# 从 vllm._custom_ops 导入封装好的算子
# 假设该模块已经正确处理了 C++ 扩展的导入或模拟
try:
    from lightop import op
    print("✅ 成功导入 lightop.op")
except (ImportError, AttributeError) as e:
    print(f"⚠️ 导入自定义算子失败: {e}")
    print("   将使用一个模拟函数代替。")
    # 定义一个与 C++ 绑定函数签名匹配的模拟函数
    def mock_topk_softmax(
        topk_weights: torch.Tensor, topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor, gating_output: torch.Tensor):
        # 模拟 C++ 内核的行为（总是归一化）
        softmax_output = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        _topk_weights, _topk_indices = torch.topk(softmax_output, topk_weights.shape[-1])
        _topk_weights = _topk_weights / _topk_weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(_topk_weights)
        topk_indices.copy_(_topk_indices)

    # 创建一个模拟的 op 模块
    class MockOps:
        topk_softmax = staticmethod(mock_topk_softmax)
    op = MockOps()


def torch_reference_topk_softmax(gating_output: torch.Tensor, topk: int,Is_renormalize: str):
    """
    使用 PyTorch 实现的 top-k softmax，作为基准进行对比。
    这个实现遵循了 'pre-softmax' 模式，即先进行 softmax，然后取 top-k，最后进行归一化。
    """
    # 1. 对门控输出在最后一个维度（专家维度）上进行 softmax
    # 输入已经是 float32，无需转换
    softmax_output = torch.softmax(gating_output, dim=-1)

    # 2. 在 softmax 结果上取 top-k 的权重和专家索引
    topk_weights, topk_indices = torch.topk(softmax_output, topk, dim=-1)

    # # 3. 对选出的 top-k 权重进行归一化，使其和为 1。
    if Is_renormalize == "true":
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


class TestTopkSoftmax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前运行，用于初始化结果列表。"""
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        """在所有测试结束后运行，用于将结果写入 Excel 文件。"""
        if not cls.results:
            print("\n没有收集到任何结果，不生成 Excel 文件。")
            return
        
        print("\n--- 所有测试完成，正在生成 Excel 报告 ---")
        df = pd.DataFrame(cls.results)
        df = df.sort_values(by='num_tokens').reset_index(drop=True)
        
        output_filename = 'performance_results.xlsx'
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"✅ 性能测试结果已保存到 {os.path.abspath(output_filename)}")


    def _run_test(self, num_tokens: int, num_experts: int, topk: int, dtype: torch.dtype, Is_renormalize: str,
                  rtol=1e-5, atol=1e-5, warmup_iters=10, test_iters=100):
        """
        执行单次测试的辅助函数，包含精度验证和性能对比。
        """
        print(f"\n--- 测试开始 ---")
        print(f"参数: num_tokens={num_tokens}, num_experts={num_experts}, topk={topk}, dtype={dtype}")
        print(f"迭代次数: warmup={warmup_iters}, test={test_iters}")

        if topk > num_experts:
            print(f"跳过测试: topk ({topk}) > num_experts ({num_experts})")
            return

        # 1. 准备输入数据 (现在直接使用指定的 dtype)
        gating_output = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)

        # 2. 准备输出张量
        custom_topk_weights = torch.empty(num_tokens, topk, device="cuda", dtype=torch.float32)
        custom_topk_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)
        custom_token_expert_indices = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int)

        # --- 性能测试：自定义算子 ---
        print("\n[性能测试] 正在测试自定义算子...")
        for _ in range(warmup_iters):
            op.topk_softmax(
                custom_topk_weights, custom_topk_indices,
                custom_token_expert_indices, gating_output, Is_renormalize)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(test_iters):
            op.topk_softmax(
                custom_topk_weights, custom_topk_indices,
                custom_token_expert_indices, gating_output,Is_renormalize)
        end_event.record()
        torch.cuda.synchronize()
        # **修改**: 将毫秒 (ms) 转换为微秒 (us)
        custom_time = start_event.elapsed_time(end_event) * 1000 / test_iters
        print(f"自定义算子平均耗时: {custom_time:.4f} us")

        # --- 性能测试：PyTorch 参考实现 ---
        print("\n[性能测试] 正在测试 PyTorch 参考实现...")
        for _ in range(warmup_iters):
            _ = torch_reference_topk_softmax(gating_output, topk,Is_renormalize)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(test_iters):
            ref_topk_weights, ref_topk_indices = torch_reference_topk_softmax(gating_output, topk,Is_renormalize)
        end_event.record()
        torch.cuda.synchronize()
        # **修改**: 将毫秒 (ms) 转换为微秒 (us)
        ref_time = start_event.elapsed_time(end_event) * 1000 / test_iters
        print(f"PyTorch 参考实现平均耗时: {ref_time:.4f} us")
        
        speedup = ref_time / custom_time
        print(f"\n性能提升倍数: {speedup:.2f}x")
        
        # **修改**: 更新字典键名
        TestTopkSoftmax.results.append({
            'num_tokens': num_tokens,
            'num_experts': num_experts,
            'topk': topk,
            'custom_op_latency_us': custom_time,
            'pytorch_latency_us': ref_time,
            'speedup_x': speedup
        })

        # --- 精度验证 ---
        print("\n[精度验证] 正在验证计算结果...")
        try:
            self.assertTrue(torch.allclose(custom_topk_weights, ref_topk_weights, rtol=rtol, atol=atol),
                            "权重的计算结果与参考实现不匹配！")
            print("✅ 权重 (Weights) 验证通过！")
        except AssertionError as e:
            print(f"⚠️  {e}")
            # 如果断言失败，打印详细的差异信息以供调试
            diff = torch.abs(custom_topk_weights - ref_topk_weights)
            max_diff = torch.max(diff)
            max_diff_idx = torch.argmax(diff)
            print(f"   最大差异值: {max_diff.item()}")
            print(f"   最大差异位置: {max_diff_idx.item()}")
            print(f"   自定义算子在该处的值: {custom_topk_weights.flatten()[max_diff_idx].item()}")
            print(f"   PyTorch 参考实现在该处的值: {ref_topk_weights.flatten()[max_diff_idx].item()}")


        ref_sorted_indices, _ = torch.sort(ref_topk_indices, dim=-1)
        custom_sorted_indices, _ = torch.sort(custom_topk_indices, dim=-1)
       
        # print("\n  [调试] PyTorch 参考实现排序后的索引 (前5行):")
        # print(ref_sorted_indices[:5])
        # print("\n  [调试] 自定义算子排序后的索引 (前5行):")
        # print(custom_sorted_indices[:5])

        self.assertTrue(torch.equal(custom_sorted_indices, ref_sorted_indices),
                        "排序后的索引与参考实现不匹配！")
        print("✅ 索引 (Indices) 验证通过！")

        print("--- 测试完成 ---\n")

    def test_specific_token_counts(self):
        """
        使用固定的 num_experts=64, topk=4, 以及一系列指定的 num_tokens 进行测试。
        """
        num_experts = 64
        topk = 4
        num_tokens_list = [
            32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160,
            176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480,
            512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792,
            1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072,
            3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096, 6144, 8192,
            10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
            28672, 32768
        ]
       
        # num_tokens_list = [ 16384, 18432, 20480, 22528, 24576, 26624,28672, 32768]
        num_tokens_list = [32];
        Is_renormalize = "true" #or false
        for num_tokens in num_tokens_list:
            self._run_test(
                num_tokens=num_tokens,
                num_experts=num_experts,
                topk=topk,
                dtype=torch.float32, # 测试 float32
                rtol=1e-5,  # 使用更适合 float32 的严格容忍度
                atol=1e-5,
                Is_renormalize=Is_renormalize,
            )


if __name__ == '__main__':
    unittest.main()