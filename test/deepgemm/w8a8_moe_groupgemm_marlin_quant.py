
import torch
import numpy as np


def w8a8_nt_kpack2_marlin_weight(w8a8_w, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert w8a8_w.dtype == torch.int8, "w8a8_w 必须是 int8 类型"
    size_n, size_k = w8a8_w.shape
    assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

    q = w8a8_w.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
    q = q.permute((0, 2, 1, 3)).contiguous()
    q = q.reshape((size_n // k_tile, size_k * k_tile))
    return q
