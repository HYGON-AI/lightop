from lightop import op

def miopen_transform_tensor(input, alpha: float = 1.0, beta: float = 0.0):
    """_summary_

    Args:
        input (Tensor): 输入Tensor, 形状 [N, H, W, C] 或 [N, C, H, W]
        alpha (float): 缩放因子
        beta (float): 缩放因子

    Returns:
        Tensor: 输出Tensor, 形状 [N, C, H, W, c32]
    """
    return op.miopen_transform_tensor(input, alpha, beta)