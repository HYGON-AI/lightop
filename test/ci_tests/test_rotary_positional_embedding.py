# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import time
from lightop.attention import RotaryEmbedding, apply_rotary_pos_emb_index_fused

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q: [sq, b * np, hn] -> [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    sq, b, np = position_id.size(0), position_id.size(1), q.size(1) // position_id.size(1)
    q, k = q.view(sq, b, np, -1), k.view(sq, b, np, -1)
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q.view(sq, b * np, -1), k.view(sq, b * np, -1)
    
if __name__ == '__main__':

    RUNS = 100

    SEQ_LEN = 2048
    HIDDEN = 12288
    NHEADS = 96
    BATCH_SIZE = 1
    HP = HIDDEN // NHEADS
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    q = torch.randn((SEQ_LEN, BATCH_SIZE * NHEADS, HP)).cuda().float()
    k = torch.randn((SEQ_LEN, BATCH_SIZE * NHEADS, HP)).cuda().float()

    q.requires_grad = True
    k.requires_grad = True

    position_id = torch.randint(0, SEQ_LEN, (SEQ_LEN, BATCH_SIZE)).cuda()
    rotary_embedding = RotaryEmbedding(HP).cuda().float()
    cos, sin = rotary_embedding.forward(k, seq_len=SEQ_LEN)

    for _ in range(100):
        apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id)
        apply_rotary_pos_emb_index(q, k, cos, sin, position_id)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(RUNS):
        q.grad = None
        k.grad = None
        q2, k2 = apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id)
        loss = 12 * (q2.mean() + k2.mean()) ** 2
        loss.backward()
    torch.cuda.synchronize()
    end_time = time.time()
    gradq2, gradk2 = q.grad, k.grad

    print(f"custom_kernel time {(end_time - start_time)*1000/RUNS} ms")

    start_time = time.time()
    for _ in range(RUNS):
        q.grad = None
        k.grad = None
        q1, k1 = apply_rotary_pos_emb_index(q, k, cos, sin, position_id)
        loss = 12 * (q1.mean() + k1.mean()) ** 2
        loss.backward()
    torch.cuda.synchronize()
    end_time = time.time()
    gradq1, gradk1 = q.grad, k.grad
    
    print(f"jit time {(end_time - start_time)*1000/RUNS} ms")
    #use_half：rtol=1e-02, atol=1e-02,use_float32:rtol=1e-05, atol=1e-05
    print(allclose(q1,q2,rtol=1e-05, atol=1e-05))
    print(allclose(k1,k2,rtol=1e-05, atol=1e-05))
    print(allclose(gradq1,gradq2,rtol=1e-05, atol=1e-05))
    print(allclose(gradk1,gradk2,rtol=1e-05, atol=1e-05))

