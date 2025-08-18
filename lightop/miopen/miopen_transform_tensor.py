from lightop import op


def miopen_transform_tensor(input, transpose_mode: int = 0, alpha: float = 1.0, beta: float = 0.0):
    """_summary_

    Args:
        input (Tensor): 输入Tensor, 形状 [N, H, W, C] 或 [N, C, H, W]
        transpose_mode (int): 0: NCHW --> NCHWc32; 1: NHWC ---> NCHWc32; 2: NCHWc32 ---> NCHW; 3:NCHWc32 ---> NHWC
        alpha (float): 缩放因子
        beta (float): 缩放因子

    Returns:
        Tensor: 输出Tensor, 形状 [N, C, H, W, c32] 或 [N, C, H, w] 或 [N, H, W, C]
    """
    return op.miopen_transform_tensor(input, transpose_mode, alpha, beta)
