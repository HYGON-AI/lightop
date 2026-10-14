# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.scalar_type import ScalarType

from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights

import numpy as np
import pandas as pd
import pdb

GPTQ_MARLIN_TILE = 16

# 将 NumPy 数组写入 Excel 文件
def write_numpy_to_excel(data, file_path):
    # 将 NumPy 数组转换为 Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 将 DataFrame 写入 Excel 文件
    df.to_excel(file_path, index=False, header=False)


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits

# bw 重排B矩阵的函数
def get_weight_perm(num_bits: int # 4
                    ):
    # 存储 16*32 数据块中 一个线程需要的数据索引 每个线程需要load 8个uint4 数据到 寄存器中 
    perm_list: List[int] = []
    # print("每个线程需要的数据位置：")
    for i in range(64):
        # print("===thread: ", i)
        perm1: List[int] = [] 
        col = i % 16 # 一列放四个线程，因此 i// 4 获得col索引
        row = i // 16
        for block in [0, 1]:
            for j in range(4):  # 之后的数据每个数据跳一行
                perm1.append( row * 128 + 32 * j + col + 16 * block)
        # print("perm1: ", perm1)
        for j in range(4): # 获取k方向 4个16*32块的所有索引
            perm_list.extend([p + 512 * j for p in perm1])

    perm = np.array(perm_list)
    
    # # 完成32bit内的数据交错
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()

    perm = torch.from_numpy(perm)
    return perm

def marlin_permute_weights(
                            q_w, # [7168, 512] [7168, 128]
                            size_k,  # 7168  7168
                            size_n,  # 512  128
                            perm, # 索引重排 4个 16*32个数据块的数据的重排索引  2048个索引
                            tile=GPTQ_MARLIN_TILE # 16
                            ):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
        # [7168, 512] ==> [448, 16, 16, 32]
        # [7168, 128] ==> [448, 16, 4, 32]
    q_w = q_w.reshape((size_k // tile, tile, size_n // (tile *2), (tile*2)))
        # [448, 16, 16, 32] ==> [448, 16, 16, 32]
        # [448, 16, 4, 32]  ==> [448, 4, 16, 32]
    q_w = q_w.permute((0, 2, 1, 3))
        # [448, 16, 16, 32] ==> [448, 8192]
        # [448, 4, 16, 32] ==> [448, 2048]
    q_w = q_w.reshape((size_k // tile, size_n * tile))
        # [-1, 2048]==> 按照给定的索引进行处理
        # [448, 4, 16, 32] ==> [448, 2048]
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w

def marlin_weights(q_w,  # [7168, 512] torch.int32
                    size_k, # 7168
                    size_n, # 512
                    num_bits, # 4
                    perm # 索引重排 4个 16*32个数据块的数据的重排索引  2048个索引
                    ):
    # Permute
    # torch.Size([7168, 512]) torch.int32
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)
    # 处理后的size torch.Size([448, 8192]) torch.int32
    # Pack 32中可放置几个量化后的 数据
    pack_factor = get_pack_factor(num_bits) # 8
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    # [448, 1024] uint32
    # [448, 256]
    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                        dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed

def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(16): # 遍历列方向不同scale的 8个线程
        scale_perm.extend([i + 16 * j for j in range(8)]) # 插入 8 个数据块中 对应位置的索引
    return scale_perm

# 处理 scale矩阵
def marlin_permute_scales(s: torch.Tensor, # [56, 512]  # torch.float16
                            size_k: int,   # 7168
                            size_n: int,   # 512
                            group_size: int # 128
                            ) -> torch.Tensor:
    # 将[128, 128](fp16) B矩阵中 每个[16, 16]计算块中的对应位置的 zero值 放到一起 
    scale_perm = get_scale_perms()
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    s = s.reshape((-1, size_n)).contiguous()
    return s

def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_res = np.zeros((size_k, size_n // pack_factor), dtype=np.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(np.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res

def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    # 和 scale 使用一致的重排逻辑，将[128, 128](fp16) B矩阵中 每个[16, 16]计算块中的对应位置的 zero值 放到一起 
    scale_perm = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # uint4 混排
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))
    # uint4打包成 int32
    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp

def merge_tensor_rows_binary_strict(marlin_s: torch.Tensor, marlin_zp: torch.Tensor,
                                    data_num_0: int, data_num_1: int) -> torch.Tensor:
    """
    合并两个 Tensor, 每行交替取 data_num_0 个 float16 和 data_num_1 个 int32。
    要求：
      - marlin_s 每行长度能被 data_num_0 整除
      - marlin_zp 每行长度能被 data_num_1 整除
      - 合并后的总字节数必为 4 的倍数

    返回:
        [N, M] 的 int32 Tensor(行数一致，列数已对齐)
    """
    assert marlin_s.shape[0] == marlin_zp.shape[0], "Batch size mismatch"
    assert marlin_s.dtype == torch.float16
    assert marlin_zp.dtype == torch.int32

    N, D0 = marlin_s.shape
    _, D1 = marlin_zp.shape

    assert D0 % data_num_0 == 0, "marlin_s 每行必须能被 data_num_0 整除"
    assert D1 % data_num_1 == 0, "marlin_zp 每行必须能被 data_num_1 整除"

    s_block_count = D0 // data_num_0
    zp_block_count = D1 // data_num_1
    assert s_block_count == zp_block_count

    total_blocks = s_block_count

    # 转为字节视图
    s_bytes = marlin_s.view(torch.uint8).reshape(N, -1)
    zp_bytes = marlin_zp.view(torch.uint8).reshape(N, -1)

    # 每行的合并结果
    merged_rows = []

    for i in range(N):
        s_row = s_bytes[i]
        zp_row = zp_bytes[i]
        s_ptr = 0
        zp_ptr = 0
        merged = []

        for _ in range(total_blocks):
            # 如果 s 还有剩余 block，就取
            if s_ptr < s_row.numel():
                chunk_s = s_row[s_ptr: s_ptr + data_num_0 * 2]  # float16 = 2 字节
                merged.append(chunk_s)
                s_ptr += data_num_0 * 2

            # 如果 zp 还有剩余 block，就取
            if zp_ptr < zp_row.numel():
                chunk_zp = zp_row[zp_ptr: zp_ptr + data_num_1 * 4]  # int32 = 4 字节
                merged.append(chunk_zp)
                zp_ptr += data_num_1 * 4

        # 合并所有字节，并直接转换为 int32
        merged_bytes = torch.cat(merged)
        # assert merged_bytes.numel() % 4 == 0, "最终字节长度必须是4的倍数"
        merged_int32 = merged_bytes.view(torch.int32)
        merged_rows.append(merged_int32)

    # 所有合并行长度一致，可以直接堆叠
    result = torch.stack(merged_rows)
    return result

def awq_marlin_quantize(
                        w: torch.Tensor,  # [7168, 512] fp16
                        quant_type: ScalarType, # uint4 
                        group_size: int # 128
                        ):
    size_k, size_n = w.shape # 7168, 512

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size # 56

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w,
                                         quant_type,
                                         group_size,
                                         zero_points=True)
    
    # w_ref[:]=1
    # q_w[:] = 1
    # s[:]=1
    # zp[:]=0
    # pdb.set_trace()
    #  将weight、scale置为全1，zp置为0，



    # 权重初步量化
    #  q_w  [7168, 512] torch.int32
    #  s  [56, 512]  # torch.float16
    #  zp  [56, 512] # torch.int32

    # Reformat to marlin 已经适配 BW
    weight_perm = get_weight_perm(quant_type.size_bits)

    marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                weight_perm)
    # 已适配
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)
    # 已适配
    marlin_zp = marlin_zero_points(zp, num_groups, size_n,
                                   quant_type.size_bits)

    marlin_s_zp = merge_tensor_rows_binary_strict(marlin_s, marlin_zp, 128, 16)
    # print("marlin_s: ", marlin_s.size(), marlin_s.dtype)
    # print("marlin_zp: ", marlin_zp.size(), marlin_zp.dtype)
    # print("marlin_s_zp: ", marlin_s_zp.size(), marlin_s_zp.dtype)
    # print("marlin_zp: ", marlin_zp[0,:16])
    # print("marlin_s_zp: ", marlin_s_zp[0, :80])

    # assert False

    # Create result
    # res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp, marlin_s_zp]
    res_list = [w_ref, marlin_q_w, marlin_s_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

# sacle_idx = get_scale_perms()
# print(sacle_idx)
# print(len(sacle_idx))
