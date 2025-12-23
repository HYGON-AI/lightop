import torch
import triton
import triton.language as tl
import time
import numpy as np


# 导入CUDA实现
try:
    from lightop import (
        destindex_copy_quantize_int8_cuda as cuda_quantize,
        destindex_dequantize_int8_cuda as cuda_dequantize
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA扩展加载失败: {e}")
    CUDA_AVAILABLE = False

@triton.jit
def _fwd_kernel_destindex_copy_quantize_init_asym(
    K, Out, Out_scale_zero,
    stride_k_bs, stride_k_h, stride_k_d,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h, stride_os_d,
    head_num, head_dim,
    BLOCK_DMODEL: tl.constexpr,  # 64
    BLOCK_HEAD: tl.constexpr # 16
):
    cur_index = tl.program_id(0)  # 当前block在batch维度上的索引
    offs_h = tl.arange(0, BLOCK_HEAD)  # 头部(head)方向的offset列表
    offs_d = tl.arange(0, BLOCK_DMODEL)  # 特征维度(dim)方向的offset列表

    dest_index = cur_index  # 当前处理的目标batch序号
    m1 = offs_h[:, None] < head_num  # head_num范围内的mask(二维行mask)
    m2 = offs_d[None,:] < head_dim   # head_dim范围内的mask(二维列mask)
    mask = m1&m2   # 仅在head和dim的有效范围内为True，形状为[BLOCK_HEAD, BLOCK_DMODEL]
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :],
                       mask=mask, other=0.0).to(tl.float32)  # 按block方式从输入K加载src数据, 形状为 [BLOCK_HEAD, BLOCK_DMODEL]

    src_data_max = tl.max(src_data, axis=1, keep_dims=True)  # 每个head的最大值, shape [BLOCK_HEAD, 1]
    src_data_min = tl.min(src_data, axis=1, keep_dims=True)  # 每个head的最小值, shape [BLOCK_HEAD, 1]
    data_scale = (src_data_max - src_data_min) / 255.0  # 每个head量化的缩放因子scale, shape [BLOCK_HEAD, 1]
    data_scale = tl.where(data_scale < 1e-8, 1e-8, data_scale)  # 防止除0，保证最小scale
    data_zero = (-1 * src_data_min / data_scale).to(tl.int32)  # 计算zero-point，shape [BLOCK_HEAD, 1]
    q_src_data = (tl.clamp((src_data / data_scale).to(tl.int32).to(tl.float32) + data_zero.to(tl.float32), 0.0, 255.0).to(tl.int32) - 128).to(tl.int8)
                         # 量化到int8, 先转int32取整，再加zero-point，clamp到[0,255]，最后-128到对称int8

    data_scale = data_scale.to(Out_scale_zero.dtype.element_ty)  # 转换scale类型以匹配输出
    data_zero = data_zero.to(Out_scale_zero.dtype.element_ty)    # 转换zero-point类型以匹配输出

    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]  # 计算量化输出的存储位置
    os_ptrs = Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]                   # 计算scale的存储位置
    oz_ptrs = Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None] + 1               # 计算zero-point的存储位置

    tl.store(o_ptrs, q_src_data, mask=mask)     # 存储量化后的数据到Out
    tl.store(os_ptrs, data_scale, mask=m1)      # 存储scale到Out_scale_zero[..., 0]
    tl.store(oz_ptrs, data_zero, mask=m1)       # 存储zero-point到Out_scale_zero[..., 1]

# 前向算子
@torch.no_grad()
def destindex_copy_quantize_int8(K, Out, Out_scale_zero):
    bs_seq = K.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)  # eg: 10->16
    BLOCK_DMODEL = triton.next_power_of_2(head_dim) # eg: 64->64
    grid = (bs_seq,)
    num_warps = 1

    _fwd_kernel_destindex_copy_quantize_init_asym[grid](
        K, Out, Out_scale_zero,
        K.stride(0), K.stride(1), K.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        Out_scale_zero.stride(0), Out_scale_zero.stride(1), Out_scale_zero.stride(2),
        head_num,head_dim,
        BLOCK_DMODEL= BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _bwd_kernel_destindex_dequantize(
    Quantized_Out, Out_scale_zero, Dequantized_Out,
    stride_qo_bs, stride_qo_h, stride_qo_d,
    stride_os_bs, stride_os_h, stride_os_d,
    stride_do_bs, stride_do_h, stride_do_d,
    head_num,head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    scales_dtype = Out_scale_zero.dtype.element_ty

    dest_index = cur_index

    m1 = offs_h[:, None] < head_num
    m2 = offs_d[None,:] < head_dim
    mask = m1&m2

    # Load quantized data
    q_data = tl.load(
        Quantized_Out + dest_index * stride_qo_bs + offs_h[:, None] * stride_qo_h + stride_qo_d * offs_d[None, :],
        mask=mask,
        other=0
    )

    # Load scale and zero point
    data_scale = tl.load(
        Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None],
        mask=m1,
        other=1.0
    )
    data_zero = tl.load(
        Out_scale_zero + dest_index * stride_os_bs + stride_os_h * offs_h[:, None] + 1,
        mask=m1,
        other=0
    )

    # Dequantize
    dequantized_data = (q_data.to(tl.int32) + 128 - data_zero.to(tl.int32)).to(scales_dtype) * data_scale

    # Store dequantized data
    out_ptrs = Dequantized_Out + dest_index * stride_do_bs + stride_do_h * offs_h[:, None] + stride_do_d * offs_d[None, :]
    tl.store(out_ptrs, dequantized_data, mask=mask)

# 反向算子
@torch.no_grad()
def destindex_dequantize_int8(Quantized_Out, Out_scale_zero, Dequantized_Out):
    bs_seq = Quantized_Out.shape[0]
    head_num = Quantized_Out.shape[1]
    head_dim = Quantized_Out.shape[2]
    assert Quantized_Out.shape[1] == Dequantized_Out.shape[1] and Quantized_Out.shape[2] == Dequantized_Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    grid = (bs_seq,)
    num_warps = 1

    _bwd_kernel_destindex_dequantize[grid](
        Quantized_Out, Out_scale_zero, Dequantized_Out,
        Quantized_Out.stride(0), Quantized_Out.stride(1), Quantized_Out.stride(2),
        Out_scale_zero.stride(0), Out_scale_zero.stride(1), Out_scale_zero.stride(2),
        Dequantized_Out.stride(0), Dequantized_Out.stride(1), Dequantized_Out.stride(2),
        head_num,head_dim,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )

def create_strided_tensor(shape, stride, dtype=torch.bfloat16, device='cuda'):
    """创建具有指定stride的张量"""
    # 计算需要的总存储空间
    numel = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
    # 创建存储空间
    storage = torch.empty(numel, dtype=dtype, device=device)
    # 使用as_strided创建指定stride的视图
    tensor = torch.as_strided(storage, size=shape, stride=stride)
    return tensor




def test_correctness():
    """测试CUDA实现的正确性"""
    print("=" * 80)
    print("测试CUDA实现的正确性")
    print("=" * 80)
    
    if not CUDA_AVAILABLE:
        print("CUDA扩展不可用，跳过测试")
        return
    
    # 测试参数
    bs_seq, head_num, head_dim = 128, 10, 64
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 创建输入张量
    K = torch.randn(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    
    # Triton版本
    Out_triton = torch.zeros(bs_seq, head_num, head_dim, dtype=torch.bfloat16, device=device)
    Out_scale_zero_triton = torch.zeros(bs_seq, head_num, 2, dtype=dtype, device=device)
    destindex_copy_quantize_int8(K, Out_triton, Out_scale_zero_triton)
    
    # 转换为int8
    Out_triton_int8 = Out_triton.to(torch.int8)
    
    # CUDA版本
    Out_cuda = torch.zeros(bs_seq, head_num, head_dim, dtype=torch.int8, device=device)
    Out_scale_zero_cuda = torch.zeros(bs_seq, head_num, 2, dtype=dtype, device=device)
    cuda_quantize(K, Out_scale_zero_cuda, Out_cuda)
    
    # import pdb
    # pdb.set_trace()

    # 比较量化结果
    print(f"\n量化输出比较:")
    print(f"Out - 平均差异: {(Out_triton_int8.float() - Out_cuda.float()).abs().mean().item():.6f}")
    print(f"Out - 最大差异: {(Out_triton_int8.float() - Out_cuda.float()).abs().max().item():.6f}")
    print(f"Scale - 平均差异: {(Out_scale_zero_triton[:,:,0] - Out_scale_zero_cuda[:,:,0]).abs().mean().item():.6f}")
    print(f"Scale - 最大差异: {(Out_scale_zero_triton[:,:,0] - Out_scale_zero_cuda[:,:,0]).abs().max().item():.6f}")
    print(f"Zero - 平均差异: {(Out_scale_zero_triton[:,:,1] - Out_scale_zero_cuda[:,:,1]).abs().mean().item():.6f}")
    print(f"Zero - 最大差异: {(Out_scale_zero_triton[:,:,1] - Out_scale_zero_cuda[:,:,1]).abs().max().item():.6f}")
    
    # 测试反量化
    Dequant_triton = torch.zeros(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    destindex_dequantize_int8(Out_triton_int8, Out_scale_zero_triton, Dequant_triton)
    
    Dequant_cuda = torch.zeros(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    cuda_dequantize(Out_cuda, Out_scale_zero_cuda, Dequant_cuda)
    
    print(f"\n反量化输出比较:")
    print(f"平均差异: {(Dequant_triton - Dequant_cuda).abs().mean().item():.6f}")
    print(f"最大差异: {(Dequant_triton - Dequant_cuda).abs().max().item():.6f}")
    
    # 往返误差
    # print(f"\n往返误差 (Triton):")
    # error_triton = (K - Dequant_triton).abs()
    # print(f"平均误差: {error_triton.mean().item():.6f}")
    # print(f"最大误差: {error_triton.max().item():.6f}")
    
    # print(f"\n往返误差 (CUDA):")
    # error_cuda = (K - Dequant_cuda).abs()
    # print(f"平均误差: {error_cuda.mean().item():.6f}")
    # print(f"最大误差: {error_cuda.max().item():.6f}")


def benchmark_quantize():
    """性能测试：量化"""
    print("\n" + "=" * 80)
    print("性能测试：前向量化")
    print("=" * 80)
    
    if not CUDA_AVAILABLE:
        print("CUDA扩展不可用，跳过测试")
        return
    
    # 测试参数
    bs_seq, head_num, head_dim = 19440, 10, 64
    #bs_seq, head_num, head_dim = 1024, 10, 64
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 创建输入
    K = torch.randn(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    
    # Triton版本
    print(f"\n【Triton实现】")
    Out_triton = torch.zeros(bs_seq, head_num, head_dim, dtype=torch.bfloat16, device=device)
    Out_scale_zero_triton = torch.zeros(bs_seq, head_num, 2, dtype=dtype, device=device)
    
    # 预热
    for _ in range(10):
        destindex_copy_quantize_int8(K, Out_triton, Out_scale_zero_triton)
    torch.cuda.synchronize()
    
    # 性能测试
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        destindex_copy_quantize_int8(K, Out_triton, Out_scale_zero_triton)
    torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / num_iterations * 1000
    print(f"平均耗时: {triton_time*1000:.4f} us")
    
    # CUDA版本
    print(f"\n【CUDA实现】")
    Out_cuda = torch.zeros(bs_seq, head_num, head_dim, dtype=torch.int8, device=device)
    Out_scale_zero_cuda = torch.zeros(bs_seq, head_num, 2, dtype=dtype, device=device)
    
    # 预热
    for _ in range(10):
        cuda_quantize(K, Out_scale_zero_cuda, Out_cuda)
    torch.cuda.synchronize()
    
    # 性能测试
    start_time = time.time()
    for _ in range(num_iterations):
        cuda_quantize(K, Out_scale_zero_cuda, Out_cuda)
    torch.cuda.synchronize()
    end_time = time.time()
    
    cuda_time = (end_time - start_time) / num_iterations * 1000
    print(f"平均耗时: {cuda_time*1000:.4f} us")
    
    print(f"\n【性能对比】")
    print(f"加速比: {triton_time/cuda_time:.2f}x")


def benchmark_dequantize():
    """性能测试：反量化"""
    print("\n" + "=" * 80)
    print("性能测试：反向反量化")
    print("=" * 80)
    
    if not CUDA_AVAILABLE:
        print("CUDA扩展不可用，跳过测试")
        return
    
    # 测试参数
    bs_seq, head_num, head_dim = 19440, 10, 64
    #bs_seq, head_num, head_dim = 1024, 10, 64
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 创建输入
    Quantized_Out = torch.randint(-128, 128, (bs_seq, head_num, head_dim), dtype=torch.int8, device=device)
    Out_scale_zero = torch.rand(bs_seq, head_num, 2, dtype=dtype, device=device) * 0.1 + 0.01
    
    # Triton版本
    print(f"\n【Triton实现】")
    Dequant_triton = torch.zeros(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    
    # 预热
    for _ in range(10):
        destindex_dequantize_int8(Quantized_Out, Out_scale_zero, Dequant_triton)
    torch.cuda.synchronize()
    
    # 性能测试
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        destindex_dequantize_int8(Quantized_Out, Out_scale_zero, Dequant_triton)
    torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / num_iterations * 1000
    print(f"平均耗时: {triton_time*1000:.4f} us")
    
    # CUDA版本
    print(f"\n【CUDA实现】")
    Dequant_cuda = torch.zeros(bs_seq, head_num, head_dim, dtype=dtype, device=device)
    
    # 预热
    for _ in range(10):
        cuda_dequantize(Quantized_Out, Out_scale_zero, Dequant_cuda)
    torch.cuda.synchronize()
    
    # 性能测试
    start_time = time.time()
    for _ in range(num_iterations):
        cuda_dequantize(Quantized_Out, Out_scale_zero, Dequant_cuda)
    torch.cuda.synchronize()
    end_time = time.time()
    
    cuda_time = (end_time - start_time) / num_iterations * 1000
    print(f"平均耗时: {cuda_time*1000:.4f} us")
    
    print(f"\n【性能对比】")
    print(f"加速比: {triton_time/cuda_time:.2f}x")


def test_strided_tensors():
    """测试支持strided tensor"""
    print("\n" + "=" * 80)
    print("测试strided tensor支持")
    print("=" * 80)
    
    if not CUDA_AVAILABLE:
        print("CUDA扩展不可用，跳过测试")
        return
    
    bs_seq, head_num, head_dim = 100, 10, 64
    device = 'cuda'
    dtype = torch.bfloat16
    
    # 创建strided输入
    K_shape = (bs_seq, head_num, head_dim)
    K_stride = (640, 64, 1)
    K = create_strided_tensor(K_shape, K_stride, dtype=dtype, device=device)
    K.copy_(torch.randn(K_shape, dtype=dtype, device=device))
    
    # 创建strided输出
    Out_shape = (bs_seq, head_num, head_dim)
    Out_stride = (680, 68, 1)
    Out = create_strided_tensor(Out_shape, Out_stride, dtype=torch.int8, device=device)
    Out.zero_()
    
    Out_scale_zero_shape = (bs_seq, head_num, 2)
    Out_scale_zero_stride = (20, 2, 1)
    Out_scale_zero = create_strided_tensor(Out_scale_zero_shape, Out_scale_zero_stride, 
                                          dtype=dtype, device=device)
    Out_scale_zero.zero_()
    
    print(f"K stride: {K.stride()}")
    print(f"Out stride: {Out.stride()}")
    print(f"Out_scale_zero stride: {Out_scale_zero.stride()}")
    
    # 执行量化
    cuda_quantize(K, Out_scale_zero, Out)
    torch.cuda.synchronize()
    
    print("✓ 成功支持strided tensor")
    
    # 测试反量化
    Dequant_shape = (bs_seq, head_num, head_dim)
    Dequant_stride = (640, 64, 1)
    Dequant = create_strided_tensor(Dequant_shape, Dequant_stride, dtype=dtype, device=device)
    Dequant.zero_()
    
    cuda_dequantize(Out, Out_scale_zero, Dequant)
    torch.cuda.synchronize()
    
    # error = (K - Dequant).abs()
    # print(f"往返误差 - 平均: {error.mean().item():.6f}, 最大: {error.max().item():.6f}")


def main():
    print("CUDA量化算子测试")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print()
    
    # 正确性测试
    test_correctness()
    
    # 性能测试
    benchmark_quantize()
    benchmark_dequantize()
    
    # # Strided tensor测试
    # test_strided_tensors()
    
    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()

