import math
import torch
import time
from lightop.scaled_masked_softmax import FusedScaleMaskSoftmax
from lightop.scaled_masked_softmax_dropout import FusedScaleMaskSoftmaxDropout
from lightop.scaled_masked_softmax import AttnMaskType
from lightop.scaled_masked_softmax_dropout import FuseAttnMaskType

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def test_time():
    attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()

    fuse_time_start = time_synchronized()
    for i in range(20):
        if (i == 10):
            fuse_time_start = time_synchronized()
        fused_softmax = (
            FusedScaleMaskSoftmaxDropout(
                input_in_fp16=True,
                input_in_bf16=False,
                mask_func=attention_mask_func,
                scale=None,
                softmax_in_fp32=False,
                attn_mask_type=FuseAttnMaskType.causal,
                # attn_mask_type=AttnMaskType.causal,
                scaled_masked_softmax_fusion=True,
                p = 0.5
            )
            .cuda()
            .half()
        )

        fused_softmax_output = fused_softmax(
            attn_weights,
            None,
        )
        # fuse_time_start = time_synchronized()
        fused_softmax_output.backward(fused_softmax_output)
    fuse_time_end = time_synchronized()
    print("===fuse:",  fuse_time_end - fuse_time_start)

    
    torch_attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()
    time_start = time_synchronized()
    for i in range(20):
        if (i == 10):
            time_start = time_synchronized()
        torch_softmax = (
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

        torch_softmax_output = torch_softmax(
            torch_attn_weights,
            None,
        )

        dropout = torch.nn.Dropout(p=0.5).cuda()
        torch_softmax_output = dropout(torch_softmax_output)
        # time_start = time_synchronized()
        torch_softmax_output.backward(torch_softmax_output)
    time_end = time_synchronized()
    print("===time:", time_end - time_start)

def test_fused_upper_triangle_mask_softmax_dropout(pa):
    torch.manual_seed(1)
    attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()
    # attn_weights = torch.randn(4,2,32,32).cuda().half().requires_grad_()
    torch.manual_seed(1)
    fused_softmax = (
        FusedScaleMaskSoftmaxDropout(
            input_in_fp16=True,
            input_in_bf16=False,
            mask_func=attention_mask_func,
            scale=None,
            softmax_in_fp32=False,
            attn_mask_type=FuseAttnMaskType.causal,
            scaled_masked_softmax_fusion=True,
            p = pa
        )
        .cuda()
        .half()
    )

    fused_softmax_output = fused_softmax(
        attn_weights,
        None,
    )
    if (pa != 0):
        fused_softmax_output += 1
    # print(fused_softmax_output)
    torch.manual_seed(1)
    torch_attn_weights = torch.randn(4,16,1024,1024).cuda().half().requires_grad_()
    # torch_attn_weights = torch.randn(4,2,32,32).cuda().half().requires_grad_()
    torch_softmax = (
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

    torch_softmax_output = torch_softmax(
        torch_attn_weights,
        None,
    )

    torch.manual_seed(1)
    dropout = torch.nn.Dropout(p=pa).cuda()
    torch_softmax_output = dropout(torch_softmax_output)
    if (pa != 0):
        torch_softmax_output += 1

    test_result = (fused_softmax_output - torch_softmax_output).abs()
    if (test_result.max() < 10e-6):
        print(("P: {} forward Success! max mise data: {}").format(pa,test_result.max()))
    else:
        print(("P: {} forward Error! max mise data: {}").format(pa,test_result.max()))

    # print(fused_softmax_output)
    torch.autograd.set_detect_anomaly(True)
    fused_softmax_output.backward(fused_softmax_output)
    
    torch_softmax_output.backward(torch_softmax_output)
    test_result_backward = (attn_weights.grad - torch_attn_weights.grad).abs()
    if (test_result_backward.max() < 10e-6):
        print(("P: {} backward Success! max mise data: {}").format(pa,test_result_backward.max()))
    else:
        print(("P: {} backward Error! max mise data: {}").format(pa,test_result_backward.max()))

if __name__ == "__main__":
    # # p:0.5
    # test_fused_upper_triangle_mask_softmax_dropout(0.5)
    # # p:0.8
    # test_fused_upper_triangle_mask_softmax_dropout(0.8)
    # # p:0
    # test_fused_upper_triangle_mask_softmax_dropout(0)
    # # p:1
    # test_fused_upper_triangle_mask_softmax_dropout(1)
    # time
    test_time()
