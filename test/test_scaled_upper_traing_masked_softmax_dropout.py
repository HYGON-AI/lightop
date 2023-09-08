import math
import torch
import time
from lightop.scaled_masked_softmax import FusedScaleMaskSoftmax
from lightop.scaled_masked_softmax_dropout import FusedScaleMaskSoftmaxDropout
from lightop.scaled_masked_softmax import AttnMaskType
from lightop.scaled_masked_softmax_dropout import FuseAttnMaskType


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

def test_time():
    seq = 1024
    attn_weights = torch.randn(4,16,seq,seq).cuda().half().requires_grad_()

    fuse_time_start = time_synchronized()
    for i in range(1000):
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
        fused_softmax_output.backward(fused_softmax_output)
    fuse_time_end = time_synchronized()
    print("===fuse:",  fuse_time_end - fuse_time_start)

    
    torch_attn_weights = torch.randn(4,16,seq,seq).cuda().half().requires_grad_()
    time_start = time_synchronized()
    for i in range(1000):
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
        torch_softmax_output.backward(torch_softmax_output)
    time_end = time_synchronized()
    print("===src:", time_end - time_start)

def test_fused_upper_triangle_mask_softmax_dropout(seq, pa):
    torch.manual_seed(1)

    attn_weights = torch.randn(4,16,seq,seq).cuda().half().requires_grad_()
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

    torch.manual_seed(1)

    torch_attn_weights = torch.randn(4,16,seq,seq).cuda().half().requires_grad_()
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

    #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
    print(allclose(fused_softmax_output,torch_softmax_output,rtol=1e-05, atol=1e-05))

    torch.autograd.set_detect_anomaly(True)
    fused_softmax_output.backward(fused_softmax_output)
    torch_softmax_output.backward(torch_softmax_output)
    #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
    print(allclose(attn_weights.grad,torch_attn_weights.grad,rtol=1e-05, atol=1e-05))
   

p = [0, 0.2, 0.5, 0.8, 1]
seq_len = [64,128,256,512,1024,2048]

if __name__ == "__main__":
    # fp16的处理，小数点第三位就是误差位了，P为0.7或者0.7，(1/p)除不尽的，最大误差在10e-3, p为0.8、0.2、0.5，（1/p)能除尽的，误差为0
    # for seq in seq_len:
    #     for tmp_p in p:
    #         test_fused_upper_triangle_mask_softmax_dropout(int(seq), tmp_p)

    # 测试lm里面的size(4,16,1024,1024)
    for tmp_p in p:
        test_fused_upper_triangle_mask_softmax_dropout(1024, tmp_p)

    # time 测试尽量单独在乌镇环境测试，保证环境的干净
    test_time()
