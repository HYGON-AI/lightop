import torch
from torch import Tensor

from .. import op
from typing import Optional, Union


def miopen_adamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool = False,
    beta1: Union[float, Tensor] = 0.9,
    beta2: Union[float, Tensor] = 0.999,
    lr: Union[float, Tensor] = 1e-3,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
    maximize: bool = False,
):
    """_summary_

    Args:
        params (list[Tensor]): 要优化的模型参数.
        grads (list[Tensor]): 对应每个参数的梯度.
        exp_avgs (list[Tensor]): 梯度的一阶矩估计, 即梯度的指数移动平均值.
        exp_avg_sqs (list[Tensor]): 梯度的二阶矩估计, 即梯度平方的指数移动平均值.
        max_exp_avg_sqs (list[Tensor]): 仅在使用AMSGrad变体时才会使用,
                                        AMSGrad算法会使用到当前为止的最大二阶矩估计, 而不是指数移动平均值.
        state_steps (list[Tensor]): 每个参数的当前更新步数.
        amsgrad (bool): 是否使用 AMSGrad 变体. 默认值 False.
        beta1 (Union[float, Tensor]): 一阶矩估计的指数衰减率. 默认值 0.9
        beta2 (Union[float, Tensor]): 二阶矩估计的指数衰减率. 默认值 0.999
        lr (Union[float, Tensor]): 学习率, 默认值 1e-3.
                                   如果没有同时指定 fused=True 或 capturable=True, 需使用浮点式的学习率.
        weight_decay (float): 权重衰减系数. 默认值 1e-2.
        eps (float): 添加到分母中的项, 用于提高数值稳定性. 默认值 1e-8.
        maximize (bool): 相对于 params 是否使用梯度上升. 默认值 False.
        foreach (Optional[bool], optional): 是否使用优化器的 foreach 实现. 默认值 None.
        capturable (bool, optional): 此实例是否可以安全地在图中捕获. 默认值 False.
        differentiable (bool, optional): 是否在优化器步骤中允许自动求导. 默认值 False.
        fused (Optional[bool], optional): 是否使用融合实现. 默认值 None.
        grad_scale (Optional[Tensor], optional): 梯度缩放张量, 对于Adamw不支持传递该参数. 默认值 None.
        found_inf (Optional[Tensor], optional): 如果设置并包含一个表示是否发现无穷梯度的值, 则可能会跳过更新. 默认值 None.
        has_complex (bool, optional): 是否存在复数. 默认值 False.

    """
    if grad_scale is not None:
        raise ValueError(f"miopen_adamw does not support grad_scale")
    if found_inf is not None:
        raise ValueError(f"miopen_adamw does not support found_inf")
    if (
        len(params) != len(grads)
        or len(params) != len(exp_avgs)
        or len(params) != len(exp_avg_sqs)
        or len(params) != len(state_steps)
    ):
        raise ValueError(f"All input lists must have the same length.")
    
    if(amsgrad and len(params) != len(max_exp_avg_sqs)):
        raise ValueError(f"max_exp_avg_sqs list and params list must have the same length when use amsgrad")

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        state_steps[i] += 1
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None

        op.miopen_fused_adam(
            param,
            grad,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            state_steps[i],
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            grad_scale,
            found_inf,
        )
