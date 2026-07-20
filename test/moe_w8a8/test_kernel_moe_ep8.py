import torch
import pytest
from typing import Optional
import triton.language as tl
from lightop._lmslim_native import envs as lsenvs
from lightop._lmslim_native.vllm_compat.w8a8_get_cache_json import W8a8GetCacheJSON
from lightop.moe import moe_align_block_size
from release_moe_w8a8_marlin import test_single_gemm_performance_accuracy, moe_kernel_prepare_input, _fused_moe_helper
import argparse
import ast
import os
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

use_lightop = lsenvs.LMSLIM_USE_LIGHTOP
device_name = lsenvs.LMSLIM_GPU_NAME

# 全局变量用于存储失败的配置
gemm1_compare_failed_list = []
gemm2_compare_failed_list = []

# BlockM配置
blockM_configs = {
    0: 16, 1: 32, 2: 64, 3: 16, 4: 32, 5: 64, 6: 16, 7: 32, 8: 64,
    9: 32, 10: 64, 11: 32, 12: 64, 13: 64, 14: 16, 15: 32, 16: 64,
    17: 16, 18: 32, 19: 64, 20: 16, 21: 32, 22: 64,
    1000: 128, 1001: 64
}

blockM_configs_decode = {i: 16 for i in range(44)}

M_THRESHOLD = 128

if use_lightop:
    from lightop.moe import moe_gemm_marlin_w8a8, get_moe_cuda_marlin_config
    try:
        from moe_w8a8_prefill_gemm import backend
        from grouped_gemm import moe_gemm_decode_prof
    except ImportError:
        print("Warning: Could not import moe_w8a8_prefill_gemm or grouped_gemm")
        backend = None
        moe_gemm_decode_prof = None

def get_w8a8moe_json(E: int, m: int, n1: int, n2: int, k: int):
    """获取配置JSON"""
    moe_dict = W8a8GetCacheJSON().triton_moejson_dict if W8a8GetCacheJSON().triton_moejson_dict else {}
    if len(moe_dict) == 0:
        configs = None
    elif f"1_{n1}_{n2}_{k}" in moe_dict:
        if m <= 16:
            m_ = m
        elif m <= 32:
            m_ = 16
        elif m <= 64:
            m_ = 256
        elif m < 1024:
            m_ = 1024
        elif m < 8192:
            m_ = 8192
        else:
            m_ = 16384
        configs = moe_dict[f"{m_}_{n1}_{n2}_{k}"]
        config1 = configs['config1']
        config2 = configs['config2']
    else:
        configs = None
         
    if configs is None:
        if m < E:
            config1 = config2 = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
        else:
            config1 = config2 = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }
    return config1, config2

def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> tuple[int, Optional[torch.Tensor]]:
    """计算专家映射关系（保留原逻辑）"""
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # 创建全局到本地专家的映射，-1表示非当前rank的专家
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    if ep_rank < (ep_size - 1):
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        local_num_experts = (global_num_experts - ep_rank * local_num_experts)
        expert_map[-local_num_experts:] = torch.arange(0, local_num_experts, dtype=torch.int32)
    
    expert_map = expert_map.cuda(0)
    return (local_num_experts, expert_map)


def get_ep_topk_ids_weight(
    M,
    top_k,
    global_num_experts,
    ep_size=8,
    ep_rank=0,
    num_experts_select=1,
    device="cuda:0"
):
    """生成专家并行场景下的topk_ids和权重（保留原逻辑）"""
    assert global_num_experts % ep_size == 0, "global_num_experts must be divisible by ep_size"
    assert num_experts_select <= top_k, "num_experts_select cannot exceed top_k"
    assert ep_rank < ep_size, "ep_rank must be less than ep_size"
    
    local_num_experts = global_num_experts // ep_size
    topk_ids = torch.full((M, top_k), global_num_experts - 1, dtype=torch.int32, device=device)

    # 生成当前rank上的本地专家索引
    local_experts_select = torch.stack([
        torch.randperm(local_num_experts, device=device)[:num_experts_select] 
        for _ in range(M)
    ])

    # 插入本地专家并打乱顺序
    topk_ids[:, :num_experts_select] = local_experts_select
    rows, cols = topk_ids.shape
    rand_indices = torch.argsort(torch.rand(rows, cols, device=device), dim=1)
    shuffled_topk_ids = topk_ids.gather(1, rand_indices)

    # 生成随机权重
    topk_weights = torch.randn((M, top_k), dtype=torch.float32, device=device)

    return shuffled_topk_ids, topk_weights


def test_moe_performance_accuracy(
    m: int, n1: int, k1: int, n2: int, k2: int, topk: int, E: int,
    out_dtype: Type[torch.dtype] = torch.bfloat16,
    per_channel_quant: bool = True
) -> Dict[str, Any]:
    """
    测试完整MOE的性能和精度
    
    Returns:
        Dict包含所有测试结果
    """
    # 基础参数配置
    ep_rank = 0
    num_tokens = m
    top_k = topk
    global_num_experts = E
    ep_size = 8
    num_experts_select = 8


    # 检查CUDA可用性
    assert torch.cuda.is_available(), "CUDA is required for this test"
    device = torch.device("cuda:0")

    # 计算本地专家数量和映射
    local_num_experts, expert_map = determine_expert_map(
        ep_size=ep_size,
        ep_rank=ep_rank,
        global_num_experts=global_num_experts
    )

    # 生成测试数据
    input_data, w1_marlin, w2_marlin, weight1, weight2, weight1_scale, weight2_scale, topk_weights, topk_ids = _fused_moe_helper(
        m, n1, k1, n2, k2, topk, E, out_dtype, per_channel_quant=per_channel_quant
    )

    # 生成topk_ids和权重
    topk_ids, topk_weights = get_ep_topk_ids_weight(
        M=num_tokens,
        top_k=top_k,
        global_num_experts=global_num_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_experts_select=num_experts_select,
        device=device
    )
    
    # 设置compute_type
    if out_dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif out_dtype == torch.float16:
        compute_type = tl.float16
    elif out_dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {out_dtype}")
    
    # 获取CUDA配置
    cuda_config1 = None
    cuda_config2 = None
    status = False
    
    if use_lightop and per_channel_quant:
        # 使用get_moe_cuda_config获取配置
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        try:
            cuda_config1, cuda_config2, status = get_moe_cuda_marlin_config(
                E, m, n1, k1, n2, k2, topk, device_name, num_cus, compute_type
            )
        except Exception as e:
            print(f"Warning: get_moe_cuda_config failed: {e}")
            status = False
    
    # 如果get_moe_cuda_config失败，使用默认配置
    if not status:
        cuda_config1, cuda_config2 = get_w8a8moe_json(E, m, n1, n2, k2)
        
        # 如果没有找到配置，根据数据大小设置默认值
        if cuda_config1 is None:
            if m > M_THRESHOLD:
                cuda_config1 = {'BLOCK_SIZE_M': blockM_configs.get(0, 16), 'MODE': 0, 'DELTA': 1}
                cuda_config2 = {'BLOCK_SIZE_M': blockM_configs.get(0, 16), 'MODE': 0, 'DELTA': 1}
            else:
                cuda_config1 = {'BLOCK_SIZE_M': blockM_configs_decode.get(0, 16), 'MODE': 0, 'DELTA': 1}
                cuda_config2 = {'BLOCK_SIZE_M': blockM_configs_decode.get(0, 16), 'MODE': 0, 'DELTA': 1}
    
    # 为Triton设置不同的配置（使用默认blockM配置）
    triton_config1 = {
        'BLOCK_SIZE_M': blockM_configs.get(0, 64) if m > M_THRESHOLD else blockM_configs_decode.get(0, 16),
        'MODE': 1000,  # 使用不同的MODE
        'DELTA': 1
    }
    triton_config2 = {
        'BLOCK_SIZE_M': blockM_configs.get(1, 64) if m > M_THRESHOLD else blockM_configs_decode.get(1, 16),
        'MODE': 1001,  # 使用不同的MODE
        'DELTA': 1
    }
    
    block_size_m = cuda_config1["BLOCK_SIZE_M"]

    assert expert_map is not None
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size_m, global_num_experts, expert_map)
    
    # triton 算子始终使用 block_m=16 测试
    # sorted_token_ids_triton, expert_ids_triton, num_tokens_post_padded_triton = moe_align_block_size(topk_ids, 16, global_num_experts, expert_map)

    print("cuda_config1: ", cuda_config1)
    print("cuda_config2: ", cuda_config2)
    # 准备第一阶段的输入
    qinput1, qa1_scale = moe_kernel_prepare_input(
        A=input_data,
        B=weight1,
        A_scale=None,
        B_scale=weight1_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=None
    )
    
   

    # 测试GEMM1
    gemm1_time, gemm1_accuracy = test_single_gemm_performance_accuracy(
        input_data, w1_marlin, weight1, weight1_scale, topk_ids, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk, cuda_config1, triton_config1, "gemm1", True, per_channel_quant, compute_type
    )
    
    # 执行CUDA GEMM1获取中间结果
    intermediate_cache1 = torch.zeros((m, topk, n1), device=input_data.device, dtype=input_data.dtype)
    moe_gemm_marlin_w8a8(qinput1, w1_marlin, intermediate_cache1, qa1_scale, weight1_scale,
                  None, sorted_token_ids, expert_ids, num_tokens_post_padded, topk, cuda_config1)
    
    # 执行激活函数
    intermediate_cache2 = torch.empty((m * topk, n1 // 2), device=input_data.device, dtype=input_data.dtype)
    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, n1))
    
    # 准备第二阶段的输入
    qinput2, qa2_scale = moe_kernel_prepare_input(
        A=intermediate_cache2,
        B=weight2,
        A_scale=None,
        B_scale=weight2_scale,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=None
    )
    

    # 测试GEMM2
    gemm2_time, gemm2_accuracy = test_single_gemm_performance_accuracy(
        intermediate_cache2, w2_marlin, weight2, weight2_scale, topk_ids, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded, 1, cuda_config2, triton_config2, "gemm2", False, per_channel_quant, compute_type
    )
    
    # 整体精度验证已在单个GEMM测试中完成
    # 这里我们基于单个GEMM的结果来判断整体精度
    overall_accuracy = gemm1_accuracy and gemm2_accuracy
    
    #GEMM1+GEMM2的总耗时
    total_time = gemm1_time + gemm2_time # 转换为微秒
    
    return {
        'M': m,
        'N1': n1,
        'K1': k1,
        'N2': n2,
        'K2': k2,
        'E': E,
        'TopK': topk,
        'GEMM1_Time_us': gemm1_time,
        'GEMM2_Time_us': gemm2_time,
        'Total_Time_us': total_time,
        'GEMM1_Accuracy_Pass': gemm1_accuracy,
        'GEMM2_Accuracy_Pass': gemm2_accuracy,
        'Overall_Accuracy_Pass': overall_accuracy
    }
    
    
def main():
    parser = argparse.ArgumentParser(description='MOE GEMM性能和精度测试脚本')
    
    # 添加参数
    parser.add_argument('--E', type=str, default='[256]', 
                        help='专家数量,如 "[256]"')
    parser.add_argument('--n1-list', type=str, required=False, default='[4096]',
                       help='N1参数列表，如 "[256]"')
    parser.add_argument('--n2-list', type=str, required=False, default='[7168]',
                       help='N2参数列表，如 "[7168]"')
    parser.add_argument('--k-list', type=str, required=False, default='[2048]',
                       help='K2参数列表，如 "[128]"')
    parser.add_argument('--m-list', type=str, default='[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]',
                       help='M参数列表，如 "[32,128,512,1024]"')
    parser.add_argument('--topk', type=int, default=8, 
                       help='topk参数')
    parser.add_argument('--out-dtype', type=str, default='bfloat16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='输出数据类型')
    parser.add_argument('--use-channel-wise', action='store_true', default=True,
                       help='使用channel-wise量化')
    parser.add_argument('--output-file', type=str, default='moe_w8a8_gemm_channel_marlin_ep8_wise.xlsx',
                       help='输出Excel文件名')
    
    # 解析参数
    args = parser.parse_args()
    
    def parse_list(list_str):
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"无法解析列表参数: {list_str}。错误: {str(e)}")
    
    # 解析参数
    E_list = parse_list(args.E)
    n1_list = parse_list(args.n1_list)
    n2_list = parse_list(args.n2_list)  
    k_list = parse_list(args.k_list)
    m_list = parse_list(args.m_list)
    topk = args.topk
    per_channel_quant = args.use_channel_wise
    
    # 设置数据类型
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    out_dtype = dtype_map[args.out_dtype]
    
    # 初始化配置
    tritonsingleton = W8a8GetCacheJSON()
    triton_json_dir = tritonsingleton.triton_json_dir
    os.makedirs(triton_json_dir, exist_ok=True)
    
    print(f"GPU设备: {device_name}")
    print(f"量化方式: {'channel-wise' if per_channel_quant else 'block-wise'}")
    print(f"输出数据类型: {args.out_dtype}")
    
    # 存储所有测试结果
    all_results = []
    
    # 测试所有配置组合
    for i in range(len(n1_list)):
        E = E_list[i]
        n1, n2, k2 = n1_list[i], n2_list[i], k_list[i]
        k1 = n2  # 根据MOE结构，第一个GEMM的K维度等于第二个GEMM的N维度
        
        print(f"\n测试配置: N1={n1}, N2={n2}, K1={k1}, K2={k2}")
        
        for m in m_list:
            print(f"  测试 M={m}")
            try:
                # 调用测试
                result = test_moe_performance_accuracy(
                    m, n1, k1, n2, k2, topk, E, out_dtype, per_channel_quant
                )
                all_results.append(result)
                
                print(f"    GEMM1耗时: {result['GEMM1_Time_us']:.2f}us, 精度: {'通过' if result['GEMM1_Accuracy_Pass'] else '失败'}")
                print(f"    GEMM2耗时: {result['GEMM2_Time_us']:.2f}us, 精度: {'通过' if result['GEMM2_Accuracy_Pass'] else '失败'}")
                print(f"    总耗时: {result['Total_Time_us']:.2f}us, 整体精度: {'通过' if result['Overall_Accuracy_Pass'] else '失败'}")
                
            except Exception as e:
                print(f"    测试失败: {str(e)}")
                # 添加失败的结果
                all_results.append({
                    'M': m, 'N1': n1, 'K1': k1, 'N2': n2, 'K2': k2, 'E': E, 'TopK': topk,
                    'GEMM1_Time_us': -1, 'GEMM2_Time_us': -1, 'Total_Time_us': -1,
                    'GEMM1_Accuracy_Pass': False, 'GEMM2_Accuracy_Pass': False,
                    'Overall_Accuracy_Pass': False
                })
    
    # 创建DataFrame并保存结果
    df = pd.DataFrame(all_results)
    df.to_excel(args.output_file, index=False)
    
    print(f"\n测试完成！结果已保存到 {args.output_file}")
    print(f"GEMM1精度失败的配置: {gemm1_compare_failed_list}")
    print(f"GEMM2精度失败的配置: {gemm2_compare_failed_list}")
    
    # 打印统计信息
    total_tests = len(all_results)
    gemm1_pass = sum(1 for r in all_results if r['GEMM1_Accuracy_Pass'])
    gemm2_pass = sum(1 for r in all_results if r['GEMM2_Accuracy_Pass'])
    overall_pass = sum(1 for r in all_results if r['Overall_Accuracy_Pass'])
    
    print(f"\n统计信息:")
    print(f"总测试数: {total_tests}")
    print(f"GEMM1精度通过率: {gemm1_pass}/{total_tests} ({100*gemm1_pass/total_tests:.1f}%)")
    print(f"GEMM2精度通过率: {gemm2_pass}/{total_tests} ({100*gemm2_pass/total_tests:.1f}%)")
    print(f"整体精度通过率: {overall_pass}/{total_tests} ({100*overall_pass/total_tests:.1f}%)")


if __name__ == "__main__":
    main()
    # 运行所有shape: HIP_VISIBLE_DEVICES=2 python test_kernel_moe_ep8.py 
    # 单侧某一个
    # HIP_VISIBLE_DEVICES=3 python test_kernel_moe_ep8.py \
    #     --E [256] \
    #     --n1-list '[4096]' \
    #     --n2-list '[7168]' \
    #     --k-list '[2048]' \
    #     --m-list '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64,128,256,512,1024,2048,4096,8192,16384]' \
    #     --topk 8 \
    #     --out-dtype bfloat16 \
    #     --use-channel-wise \
    #     --output-file deepseek_671b_tp16_results.xlsx