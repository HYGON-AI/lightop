#from lightop import op
from .. import op


def miopen_add_rms_norm_dynamic_quant(x, x2, weight, smoothScale, epsilon):
    """_summary_
    Args:
        x (Tensor): hidden_states张量, 形状为 [num_tokens, hidden_size], bfloat16类型
        x2 (Tensor): residual张量, 形状为 [num_tokens, hidden_size], bfloat16类型
        weight (Tensor): 权值张量, 形状为 [hidden_size], bfloat16 or float32类型
        smoothScale (Tensor): 缩放张量, 形状为 [hidden_size], bfloat16 or float32类型
        
    Returns:
        vector<Tensor>:
            after_res (Tensor): 形状为 [num_tokens, hidden_size], int8类型
            y (Tensor): 形状为 [num_tokens, hidden_size], int8类型
            per_token_scale (Tensor): 形状为 [num_tokens], float32类型
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
        vector<Tensor>: 
            addDesc 形状为  [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
            yDesc 形状为 [N, C_in, H, W] (NCHW) 或 [N, C_in, D, H, W] (NCDHW)
            perTokenScaleDesc/rstdDesc 形状为 [N, C_in, H] (NCHW) 或 [N, C_in, D, H] (NCDHW)

    """
    return op.miopen_rms_norm_dynamic_quant_without_residual(x, weight, smoothScale, epsilon)


def miopen_head_rms_norm(x, weight, epsilon, norm_head_start, norm_head_end, norm_head_num):
    """_summary_

    Args:
        x (Tensor): 输入张量，形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
        weight (Tensor): 权重张量，形状为 [norm_head_num * H * W] 或 [norm_head_num * D * H * W]
        epsilon (float): 用于稳定逆方差计算的值
        norm_head_start (int): 用于稳定逆方差计算的值
        norm_head_end (int): 用于稳定逆方差计算的值
        norm_head_num (int)

    Returns:
        Tensor: 返回 yDesc, 形状为 [N, C_in, H, W] (NCHW) 或  [N, C_in, D, H, W] (NCDHW)
    """
    return op.miopen_head_rms_norm(x, weight, epsilon, norm_head_start, norm_head_end, norm_head_num)
