def weight8bit_nt_kpack2_marlin(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 1, 3, 2, 4)).contiguous()
        q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q
