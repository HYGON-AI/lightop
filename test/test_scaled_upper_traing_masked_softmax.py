import math
import torch
from lightop import AttnMaskType
from lightop.attention import FusedScaleMaskSoftmax

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

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
    #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
    print(allclose(fused_softmax_output.mean(dim=-1),torch_softmax_output.mean(dim=-1),rtol=2e-02, atol=2e-02))
    
    # backward
    fused_softmax_output.backward(fused_softmax_output)
    torch_softmax_output.backward(torch_softmax_output)
    test_result_backward = (attn_weights.grad - torch_attn_weights.grad).abs()
    #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
    print(allclose(attn_weights.grad.mean(dim=-1),torch_attn_weights.grad.mean(dim=-1),rtol=2e-02, atol=2e-02))
    

if __name__ == "__main__":
    test_fused_upper_triangle_mask_softmax()
