import math
import torch
from lightop.scaled_masked_softmax import FusedScaleMaskSoftmax
from lightop.scaled_masked_softmax import AttnMaskType

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def test_fused_upper_triangle_mask_softmax():
    torch.manual_seed(1)
    attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()
    fused_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attn_weights,
        None,
    )

    torch.manual_seed(1)
    torch_attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()
    torch_softmax = (
        FusedScaleMaskSoftmax(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=AttnMaskType.causal,
            scaled_masked_softmax_fusion=False,
        )
        .cuda()
        .half()
    )

    torch_softmax_output = torch_softmax(
        torch_attn_weights,
        None,
    )

    test_result = (fused_softmax_output - torch_softmax_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_upper_triangle_mask_softmax"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_softmax_output[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_softmax_output[-1][-1][-1][:5].tolist()}"
        )
    # backward
    fused_softmax_output.backward(fused_softmax_output)
    torch_softmax_output.backward(torch_softmax_output)
    test_result_backward = (attn_weights.grad - torch_attn_weights.grad).abs()

    while test_result_backward.dim() != 1:
        test_result_backward = test_result_backward.mean(dim=-1)

    diff_backward = test_result_backward.mean(dim=-1)
    if diff_backward <= 1e-3:
        print(
            f"\n[Success] test_fused_upper_triangle_mask_softmax_backward"
            f"\n > mean_difference={diff_backward}"
            f"\n > fused_values={attn_weights.grad[-1][-1][-1][:5].tolist()}"
            f"\n > torch_values={torch_attn_weights.grad[-1][-1][-1][:5].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_upper_triangle_mask_softmax_backward"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={attn_weights.grad[-1][-1][-1][:5].tolist()}, "
            f"\n > torch_values={torch_attn_weights.grad[-1][-1][-1][:5].tolist()}"
        )

if __name__ == "__main__":
    test_fused_upper_triangle_mask_softmax()
