import torch
from lightop.gemm_ops import groupgemm
import random

# ------------------------------
# PyTorch reference GEMM
# ------------------------------
def torch_group_gemm(A, B, batch_sizes, trans_b=False):
    """
    将矩阵A按M维度分成b份进行分组矩阵乘法
    
    Args:
        A: 输入矩阵，形状为 (M, K)
        B: 输入矩阵，形状为 (num_experts, K, N) 或 (num_experts, N, K) 如果 trans_b=True
        batch_sizes: 每份的大小列表,所有元素之和应等于M
        trans_b: 是否转置B矩阵
    
    Returns:
        拼接后的结果矩阵，形状为 (M, N)
    """
    if trans_b:
        B = B.T  # 转置B矩阵
    
    # 验证batch_sizes的总和是否等于M
    # assert sum(batch_sizes) == A.size(0), f"batch_sizes总和 {sum(batch_sizes)} 不等于M {A.size(0)}"
    assert len(batch_sizes) == B.size(0), f"batch_sizes长度 {len(batch_sizes)} 不等于M {B.size(0)}"
    
    results = []
    start_idx = 0
    
    for expert_idx, batch_size in enumerate(batch_sizes):
        
        # 某个专家不需要计算，则跳过
        if batch_size == 0:
            continue
        # 提取当前批次的A矩阵
        A_batch = A[start_idx:start_idx + batch_size, :]  # 形状: (batch_size, K)
        B_batch = B[expert_idx, :, :]  # 形状: (K, N)
        
        # 计算矩阵乘法: A_batch @ B
        result_batch = torch.matmul(A_batch, B_batch)  # 形状: (batch_size, N)
        
        results.append(result_batch)
        start_idx += batch_size
    
    # 拼接所有结果
    final_result = torch.cat(results, dim=0)  # 形状: (M, N)
    
    return final_result

def random_split_num_token(num_token, num_experts):
    """
    随机将m分成num_experts份,每份大小不相等
    """
    total = num_token
    num_parts = num_experts
    
    # 生成7个随机分割点（在1到1023之间）
    split_points = sorted(random.sample(range(1, total), num_parts - 1))
    
    # 添加起始点和结束点
    split_points = [0] + split_points + [total]
    
    # 计算每份的大小
    batch_sizes = [split_points[i+1] - split_points[i] for i in range(num_parts)]
    
    return batch_sizes

# ------------------------------
# 测试函数
# ------------------------------
def test_groupgemm_dynamic():
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = 'cuda'
    num_token = 1024
    num_experts = 8
    # num_experts = 12
    K = 128
    N = 2048
    eps = 1e-3  # bfloat16 精度
    batch_sizes = random_split_num_token(num_token, num_experts)
    # batch_sizes = [0, 0, 0, 0, 0, 0, 0, 1024]
    # batch_sizes = [0, 0, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0]
    print(batch_sizes)
    A = torch.randn((num_token, K), dtype=dtype, device=device)
    B = torch.randn((num_experts, K, N), dtype=dtype, device=device)
    batch_sizes_tensor = torch.tensor(batch_sizes, dtype=torch.int64, device='cuda')
    outs = groupgemm(A, B, batch_sizes_tensor, False)
    ref_d = torch_group_gemm(A, B, batch_sizes, False)
    max_err = (outs - ref_d).abs().max().item()
    print(f"max error = {max_err}")
    if max_err > eps:
        print("outs",outs)
        print("ref_d",ref_d)

# 入口
if __name__ == "__main__":
    test_groupgemm_dynamic()
