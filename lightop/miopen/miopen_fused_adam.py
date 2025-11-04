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
    amsgrad: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    """_summary_

    Args:
        params (list[Tensor]): 输入输出
        grads (list[Tensor]): 输入
        exp_avgs (list[Tensor]): 输入输出
        exp_avg_sqs (list[Tensor]): 输入输出
        max_exp_avg_sqs (list[Tensor]): 输入输出
        state_steps (list[Tensor]): 输入输出
        amsgrad (bool): 输入
        beta1 (Union[float, Tensor]): 输入
        beta2 (Union[float, Tensor]): 输入
        lr (Union[float, Tensor]): 输入
        weight_decay (float): 输入
        eps (float): 输入
        maximize (bool): 输入
        foreach (Optional[bool], optional): _description_. Defaults to None.
        capturable (bool, optional): _description_. Defaults to False.
        differentiable (bool, optional): _description_. Defaults to False.
        fused (Optional[bool], optional): _description_. Defaults to None.
        grad_scale (Optional[Tensor], optional): 输入. Defaults to None.
        found_inf (Optional[Tensor], optional): 输入. Defaults to None.
        has_complex (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    return op.miopen_fused_adam(
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
        found_inf
    )
