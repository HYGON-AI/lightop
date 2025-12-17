import torch
import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple,Type,Any
from vllm import _custom_ops as ops
import torch
from lightop import m_grouped_fp8_gemm_nt_masked
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
import argparse
from deepgemm_groupgemm_marlin_quant import weight8bit_nt_kpack2_marlin  # pyright: ignore[reportMissingImports]

from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

print("device_name:", device_name, "num_cus:", num_cus)


FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0
EPS = 1e-10

def dynamic_per_token_scaled_fp8_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None
) -> None:
    """
    纯PyTorch实现逐Token动态FP8量化（ROCm兼容版）
    替换 torch.ops._C.dynamic_per_token_scaled_fp8_quant
    """
    # 1. 提取形状信息（处理padding）
    orig_num_tokens, dim = input.shape
    padded_num_tokens = output.shape[0]

    # 2. Per-Token计算绝对值最大值（核心：逐Token缩放）
    abs_input = torch.abs(input.contiguous())
    absmax_per_token = torch.max(abs_input, dim=-1, keepdim=True)[0]  # [orig_num_tokens, 1]
    absmax_per_token = torch.clamp(absmax_per_token, min=EPS)  # 避免除0

    # 3. 计算缩放因子（映射到FP8最大值448）
    per_token_scale = FP8_E4M3_MAX / absmax_per_token

    # 4. 应用scale_ub限制（对齐原生算子）
    if scale_ub is not None:
        per_token_scale = torch.clamp(per_token_scale, max=scale_ub)

    # 5. 填充缩放因子到padding形状（多余部分填0）
    scale[:orig_num_tokens, :] = per_token_scale
    if padded_num_tokens > orig_num_tokens:
        scale[orig_num_tokens:, :] = 0.0

    # 6. 缩放输入到FP8范围 + 裁剪 + 四舍五入（对齐硬件量化特性）
    input_scaled = input / per_token_scale  # 广播缩放：[orig_num_tokens, dim]
    input_clamped = torch.clamp(input_scaled, min=FP8_E4M3_MIN, max=FP8_E4M3_MAX)
    input_rounded = torch.round(input_clamped)

    # ========== 核心修改2：ROCm 类型兼容 ==========
    # 方案A：强制转为 ROCm 支持的 Float8_e4m3fnuz（优先）
    if output.dtype == torch.float8_e4m3fn:
        output_dtype = torch.float8_e4m3fnuz  # ROCm原生支持的FP8类型
    else:
        output_dtype = output.dtype

    # 7. 填充到输出张量（兼容padding，多余部分填0）
    output[:orig_num_tokens, :] = input_rounded.to(output_dtype)
    if padded_num_tokens > orig_num_tokens:
        output[orig_num_tokens:, :] = 0


def native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype):

    A = input.to(torch.float32)
    B = weight.to(torch.float32)

    print("A shape:", A.shape)
    print("B shape:", B.shape)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"

    C = torch.bmm(A, B.transpose(1,2))  # [E, M, K]
    print("C shape after bmm:", C.shape)
    print("input_scale shape:", input_scale.shape)
    print("weight_scale shape:", weight_scale.shape)
    C = input_scale * C * weight_scale.transpose(1,2)  # Broadcast per-column scale
    C = C.to(output_dtype)

    return C
           
def deepgemm_masked_impl_fp8_ep_correctness(input, input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype):
    
    # w13:[num_experts/ep, N, K // 2] [16, 4096, 3584]
    # w2: [num_experts/ep, K, N//2//2] [16, 7168, 1024]
    # weight_scale:[num_experts/ep, N, 1] [16, 4096, 1]
    # w2_scale: [num_experts/ep, K, 1] [16, 7168, 1]
    
    weight_marlin = weight8bit_nt_kpack2_marlin(weight)


    # print("input:", input.shape, input_scale.shape)
    # torch.set_printoptions(profile="full")
    # input = torch.ones_like(input)
    # input_scale = torch.ones_like(input_scale)
    # input[:, 128:192] = 0
    # print("input:",input.shape,input)
    

    inter_out_gemm1 = native_fp8_perChannel_batch_matmul(input, weight, input_scale, weight_scale, output_dtype=out_dtype)

    # inter_out_gemm1[:, math.floor(expected_m_per_group * 0.7):, :] = 0


    if len(inter_out_gemm1) > 0:
        gemm1_out_cuda = torch.zeros_like(inter_out_gemm1)
        # gemm2_out_cuda = torch.zeros_like(expert_output)
        # act_out_q, act_out_s = per_token_quant_int8(act_out_cat)

    E = len(masked_m)
    M = input.size(1)
    N1 = weight.size(1)
    K1 = input.size(2)//2
    N2 = K1*2
    K2 = N1 // 4

    torch.cuda.synchronize()
    # print("tensor_problem_size: ", tensor_problem_size)
    # print("tensor_expert_offsets: ", tensor_expert_offsets)
    
    # # gemm1
    print("开始 m_grouped_fp8_gemm_nt_masked gemm1...")
    print("input:", input.shape, input.dtype)
    print("weight_marlin:", weight_marlin.shape, weight_marlin.dtype)
    print("gemm1_out_cuda:", gemm1_out_cuda.shape, gemm1_out_cuda.dtype)
    print("input_scale:", input_scale.shape, input_scale.dtype)
    print("weight_scale:", weight_scale.shape, weight_scale.dtype)
    # torch.set_printoptions(profile="full") 
    print("M:", M, "N2:", N2)
    # row_values = torch.arange(-128, 128).reshape(-1, 1).to(input.device)
    # print("row_values:", row_values)
    # # 广播到所有列
    # input = row_values.expand(M, N2).reshape(1, M, N2).to(input.device).to(torch.int8)

    # print("input:", input)
    # torch.set_printoptions(profile="full") 
    # print("weight_marlin:", weight_marlin)
    m_grouped_fp8_gemm_nt_masked((input, input_scale), 
                                  (weight_marlin, weight_scale),
                                    gemm1_out_cuda,
                                    masked_m, 
                                    expected_m_per_group,
                                    )
    torch.cuda.synchronize()
    
    if not (torch.allclose(inter_out_gemm1[:,0:math.floor(expected_m_per_group * 0.5),:], gemm1_out_cuda[:,0:math.floor(expected_m_per_group * 0.5),:], rtol=1e-2, atol=1e-2)):
        print("error deepgemm_groupgemm_fp8 gemm 精度检查不合格！！！")
        # torch.set_printoptions(profile="full") 
        print("rocblas:", inter_out_gemm1)
        print("asm:",gemm1_out_cuda)
        # diff_indices = torch.where(inter_out_gemm1 != gemm1_out_cuda)[0]
        # diff_indices = diff_indices.view(-1, inter_out_gemm1.shape[1])
        # print("不同元素的索引:", diff_indices)
        # torch.set_printoptions(profile="full")  
        diff_mask = inter_out_gemm1 != gemm1_out_cuda
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)
        print("存在差异的坐标与值如下：")
        count = 0
        for idx in diff_indices:
            count += 1
            if count > 33:
                break
            _, i, j = idx.tolist()
            print(f"坐标 ({_}, {i}, {j}): rocblas={inter_out_gemm1[_, i, j]}, asm={gemm1_out_cuda[_, i, j]}")
        # fasm = open('asm_output.txt', 'w')   
        # print(f'asm_output: {gemm1_out_cuda}', file=fasm)
        # fasm.close
        # ftorch = open('torch_output.txt', 'w')   
        # print(f'torch_output: {inter_out_gemm1_out_ref}', file=ftorch)
        # ftorch.close
    else:
        print("success deepgemm_groupgemm_fp8 gemm 精度检查合格")
    return gemm1_out_cuda

def _fused_moe_helper(m: int,
                             n: int,
                             k: int,
                             E:int,
                             out_dtype: Type[torch.dtype] = torch.float16,
                             expected_m_per_group: int = 256,
                             device: str = "cuda",
                             ):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    input = torch.randint(-16, 17, (E, m, k), device=device).to(torch.float8_e4m3fn)

    input_scale = (torch.randn((E,m,1), device=device, dtype=torch.float32))

    weight = torch.randint(-16, 17, (E,n ,k),  device=device).to(torch.float8_e4m3fn)

    weight_scale = (torch.randn((E,n,1), device=device, dtype=torch.float32))
    
    # cuts = torch.randint(1, m + 1, (E - 1,))
    # print("cuts:", cuts)
    # cuts, _ = torch.sort(cuts)
    # print("sorted cuts:", cuts)
    # tokens_per_expert = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([m])]))
    masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
    for j in range(E):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
        # masked_m[j] = int(256)
    return input,input_scale,weight,weight_scale,masked_m




def main():
    # PyTorch CPU随机种子
    seed = 0
    torch.manual_seed(seed)
    
    # PyTorch GPU随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU情况

    
    m_list=[4096]
    E=8
    expected_m_per_group=256
    out_dtype=torch.bfloat16
    
    n_list=[4096] #, 7168] #[256,512]
    k_list=[7168] #, 2048] #[7168,7168]
    
    
    for i in range(0,len(k_list),1):
        for m in m_list:
            print("m:{} expected_m:{} n:{} k:{} ".format(m, expected_m_per_group, n_list[i], k_list))
            input,input_scale,weight,weight_scale,masked_m=_fused_moe_helper(m=m,n=n_list[i],
                    k=k_list[i],E=E,out_dtype=out_dtype,expected_m_per_group=expected_m_per_group)
            
            # weight = torch.ones_like(weight) 
            # rows = torch.arange(-128, 128).unsqueeze(1)  # shape [256, 1]

            # # 扩展成每行重复 128 次，得到 shape [256, 128]
            # tensor = rows.expand(-1, 128).cuda()
            # weight = tensor.to(torch.int8).unsqueeze(0).expand(E, -1, -1)  # shape [1, 256, 128] then expand to [E, 256, 128]
            # print("weight:", weight)

            # weight_scale = torch.ones_like(weight_scale) 
            print("masked_m:", masked_m)
            print("input:",input.shape, input.dtype, input)
            print("weight:",weight.shape,weight.dtype, weight)
            print("weight_scale:",weight_scale.shape,weight_scale.dtype)
            print("masked_m:", masked_m.shape, masked_m.dtype, masked_m)

            
            deepgemm_masked_impl_fp8_ep_correctness(input,input_scale, weight, weight_scale, masked_m, expected_m_per_group, out_dtype)

if __name__ == "__main__":
    main()
