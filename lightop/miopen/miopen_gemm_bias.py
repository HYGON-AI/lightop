from lightop import op


def miopen_gemm_bias(input_a, input_b, bias):
    """_summary_

    Args:
        input_a (Tensor): 输入张量, 形状如 [batch_size, m, k]
        input_b (Tensor): 输入张量, 形状如 [k, n]
        bias (Tensor): 输入张量, 形状如 [1, n]

    Returns:
        Tensor: 输出张量 y = input_a * input_b + bias, 形状如 [batch_size, m, n]
    """
    return op.miopen_gemm_bias(input_a, input_b, bias)
