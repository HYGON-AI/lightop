import torch
from lightop import groupgemm

# ------------------------------
# PyTorch reference GEMM
# ------------------------------
def ref_gemm(A, B, C, alpha=1.0, beta=0.0, bias=None):
    D = alpha * torch.matmul(A, B) + beta * C
    # if bias is not None:
    #     D += bias[:, None]  # broadcast along N
    return D

# ------------------------------
# 测试函数
# ------------------------------
def test_groupgemm_dynamic(num_gemms=5):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = 'cuda'

    # GEMM 形状
    gemm_shapes = [
        (64, 128, 32),
        (128, 64, 64),
        (32, 32, 16),
        (64, 64, 64),
        (1024, 2048, 256),
    ][:num_gemms]

    # 构造张量列表
    a_list, b_list, c_list, d_list, bias_list, scaleAlphaVec_list = [], [], [], [], [], []
    for M, N, K in gemm_shapes:
        a_list.append(torch.randn((M, K), dtype=dtype, device=device))
        b_list.append(torch.randn((K, N), dtype=dtype, device=device))
        c_list.append(torch.ones((M, N), dtype=dtype, device=device))
        d_list.append(torch.zeros((M, N), dtype=dtype, device=device))
        bias_list.append(torch.zeros((M,), dtype=torch.float32, device=device))
        scaleAlphaVec_list.append(torch.ones((M,), dtype=torch.float32, device=device))

    batch_counts = [1] * num_gemms
    alpha = 1.0
    beta = 0.0
    biastype_list = [0] * num_gemms
    activation_list = [0] * num_gemms

    # 构造参数字典列表
    args = []
    for i in range(num_gemms):
        args.append({
            'a': a_list[i],
            'b': b_list[i],
            'c': c_list[i],
            'd': d_list[i],
            'M': gemm_shapes[i][0],
            'N': gemm_shapes[i][1],
            'K': gemm_shapes[i][2],
            'batchcount': batch_counts[i],
            'alpha': alpha,
            'beta': beta,
            'bias': bias_list[i],
            'biastype': biastype_list[i],
            'activationType': activation_list[i],
            'scaleAlphaVec': scaleAlphaVec_list[i]
        })

    # 调用 groupgemm
    outs = groupgemm(args)

    # PyTorch reference
    ref_d = [
        ref_gemm(a_list[i], b_list[i], c_list[i],
                 alpha, beta, bias_list[i])
        for i in range(num_gemms)
    ]

    # 比较结果
    eps = 1e-2  # bfloat16 精度
    for i, (d_out, d_ref) in enumerate(zip(outs, ref_d)):
        max_err = (d_out - d_ref).abs().max().item()
        print(f"GEMM {i}: max error = {max_err}")
        if max_err > eps:
            print("d_out",d_out)
            print("d_ref",d_ref)

# 入口
if __name__ == "__main__":
    test_groupgemm_dynamic()
