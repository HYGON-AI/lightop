import torch
from lightop import groupgemm
import numpy as np
import os

# -------------------------------------------------------
# 工具函数：从 CSV 读入张量
# -------------------------------------------------------
def load_tensor_bf16(fname, shape, dtype=torch.bfloat16, device='cuda'):
    vec = np.loadtxt(fname).astype(np.float32)  # CSV 里默认是 float
    return torch.from_numpy(vec).reshape(shape).to(dtype=dtype, device=device)

def load_tensor_fp32(fname, shape, dtype=torch.float32, device='cuda'):
    vec = np.loadtxt(fname).astype(np.float32)
    return torch.from_numpy(vec).reshape(shape).to(dtype=dtype, device=device)

# -------------------------------------------------------
# 测试函数：完全由文件驱动
# -------------------------------------------------------
def test_groupgemm_from_file(root_dir='.', num_gemms=4):
    # 你需要根据实际 GEMM 形状来调整
    gemm_shapes = [
        (1024, 1024, 1024),
        (1024, 1024, 1024),
        (1024, 1024, 1024),
        (1024, 1024, 1024),
    ][:num_gemms]

    batch_counts = [1] * num_gemms
    alpha_list = [1.0] * num_gemms
    beta_list  = [0.0] * num_gemms
    biastype_list = [0] * num_gemms
    activation_list = [0] * num_gemms

    a_list, b_list, c_list, d_list, bias_list, scale_list = [], [], [], [], [], []

    for i, (M, N, K) in enumerate(gemm_shapes):
        pre = os.path.join(root_dir, f"case{i}")

        # 从 CSV 读取数据
        A    = load_tensor_bf16(f"{pre}_A.csv", M*K)
        B    = load_tensor_bf16(f"{pre}_B.csv", K*N)
        C    = load_tensor_bf16(f"{pre}_C.csv", 1052672) 
        bias = load_tensor_fp32(f"{pre}_bias_gpu.csv", M)
        scale= load_tensor_fp32(f"{pre}_scale.csv", M)

        a_list.append(A)
        b_list.append(B)
        c_list.append(C)
        bias_list.append(bias)
        scale_list.append(scale)

        # D 初始设为全 0，形状跟 C 一样
        d_list.append(torch.zeros((M, N), dtype=torch.bfloat16, device='cuda'))

    # 调用 groupgemm
    outs = groupgemm([
        {
            'a': a_list[i],
            'b': b_list[i],
            'c': c_list[i],
            'd': d_list[i],
            'M': gemm_shapes[i][0],
            'N': gemm_shapes[i][1],
            'K': gemm_shapes[i][2],
            'batchcount': batch_counts[i],
            'alpha': alpha_list[i],
            'beta': beta_list[i],
            'bias': bias_list[i],
            'biastype': biastype_list[i],
            'activationType': activation_list[i],
            'scaleAlphaVec': scale_list[i]
        }
        for i in range(num_gemms)
    ])

    # 保存每个 D 输出
    for i, d_tensor in enumerate(outs):
        flat = d_tensor.float().cpu().numpy().ravel()
        np.savetxt(f"output_D{i}.csv", flat, fmt="%.6f")

    print(f"D 计算完成，已写入 output_D0.csv … output_D{num_gemms-1}.csv")

if __name__ == "__main__":
    test_groupgemm_from_file(".")
