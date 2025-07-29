from lightop import op


def miopen_add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon):
    """
    (in) hidden_states
        [num_tokens, hidden_size]
        bfloat16

    (in) residual
        optional
        [num_tokens, hidden_size]
        bfloat16

    (in) weight
        [hidden_size]
        bfloat16 or float32

    (in) smooth_scale
        [hidden_size]
        bfloat16 or float32

    (out) after_res
        [num_tokens, hidden_size]
        bfloat16

    (out) y
        [num_tokens, hidden_size]
        int8

    (out) per_token_scale
        [num_tokens]
        float32

    (attr) epsilon
        float32
    """
    return op.miopen_add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon)


def miopen_rms_norm_dynamic_quant_without_residual(x, weight, smoothScale, epsilon):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 权值张量，形状为 [W]
        smoothScale (Tensor): 形状为 [W]
        epsilon (_tfloatype_): 用于稳定逆方差计算的值

    Returns:
        vector<Tensor>: 返回 addDesc 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            yDesc 形状为 [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
                            perTokenScaleDesc/rstdDesc 形状为 [N, C_in, H] (NCHW) 或 [N, C_in, D, H] (NCDHW)

    """
    return op.miopen_rms_norm_dynamic_quant_without_residual(x, weight, smoothScale, epsilon)


def miopen_head_rms_norm(x, weight, epsilon, head_offset, head_num):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 权重张量，形状为 [W]
        epsilon (float): 用于稳定逆方差计算的值
        head_offset (float): 用于稳定逆方差计算的值
        head_num (float): 用于稳定逆方差计算的值

    Returns:
        Tensor: 返回 yDesc, 形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
    """
    return op.miopen_head_rms_norm(x, weight, epsilon, head_offset, head_num)
