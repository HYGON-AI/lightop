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
        params (list[Tensor]): 输出
        grads (list[Tensor]): 输入
        exp_avgs (list[Tensor]): 输入输出
        exp_avg_sqs (list[Tensor]): 输入输出
        max_exp_avg_sqs (list[Tensor]): 输入输出
        state_steps (list[Tensor]): 输入输出
        amsgrad (bool): 是否使用论文<On the Convergence of Adam and Beyond>中该算法的 AMSGrad 变体. 默认值 False.
        beta1 (Union[float, Tensor]): 计算梯度的滑动平均值的系数. 默认值 0.9
        beta2 (Union[float, Tensor]): 计算梯度平方的滑动平均值的系数. 默认值 0.999
        lr (Union[float, Tensor]): 学习率, 默认值 1e-3.
                                   如果没有同时指定 fused=True 或 capturable=True, 需使用浮点式的学习率.
        weight_decay (float): 权重衰减系数. 默认值 1e-2.
        eps (float): 添加到分母中的项, 用于提高数值稳定性. 默认值 1e-8.
        maximize (bool): 相对于 params 最大化目标, 而非最小化. 默认值 False.
        foreach (Optional[bool], optional): 是否使用优化器的 foreach 实现. 默认值 None.
        capturable (bool, optional): 此实例是否可以安全地在图中捕获. 默认值 False.
        differentiable (bool, optional): 自动求导是否应通过优化器的 step 步骤进行. 默认值 False.
        fused (Optional[bool], optional): 是否使用融合实现. 默认值 None.
        grad_scale (Optional[Tensor], optional): 输入. 梯度缩放张量, 对于Adamw不支持传递该参数. 默认值 None.
        found_inf (Optional[Tensor], optional): 输入. 是否发现无穷梯度值. 默认值 None.
        has_complex (bool, optional): 是否存在复数. 默认值 False.

    """
    if grad_scale is not None:
        raise ValueError(f"miopen_adamw does not support grad_scale")
    if found_inf is not None:
        raise ValueError(f"miopen_adamw does not support found_inf")

    op.miopen_fused_adam(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
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
