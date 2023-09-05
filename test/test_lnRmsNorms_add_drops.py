import time
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from typing import Optional
from apex.normalization import MixedFusedRMSNorm
from lightop.fuse_norms_add_drops import FusedLnRmsNormsAddDropout


def test_accuracy_ln(is_training=True):
    print('\ntest_accuracy_ln: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fused_ln = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                         bias_type='none', drop_type='none').to(device)
    out1 = torch_ln(x1)
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_ln(x2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()
    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("ln out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    if is_training:
        grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("ln grad_diff: max={}, mean={}".format(grad_diff.max(), grad_diff.mean()))


def test_accuracy_rms(is_training=True):
    print('\ntest_accuracy_rms: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()

    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fused_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                          bias_type='none', drop_type='none').to(device)
    out1 = apex_rms(x1)
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_rms(x2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()
    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("rms out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    if is_training:
        grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("rms grad_diff: max={}, mean={}".format(grad_diff.max(), grad_diff.mean()))


def test_accuracy_add(is_training=True):
    print('\ntest_accuracy_rms: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    fused_ln = FusedLnRmsNormsAddDropout(norm_type='none', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='none', drop_prob=0.5).to(device)
    out1 = (x1 + y1)
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_ln(x2, y2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()

    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("add out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    if is_training:
        x1_grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("add x grad_diff: max={}, mean={}".format(x1_grad_diff.max(), x1_grad_diff.mean()))
        y1_grad_diff = torch.nn.functional.l1_loss(y1.grad, y2.grad, reduction='none')
        print("add y grad_diff: max={}, mean={}".format(y1_grad_diff.max(), y1_grad_diff.mean()))


def test_accuracy_ln_add(is_training=True):
    print('\ntest_accuracy_ln_add: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fused_ln = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='none', drop_prob=0.5).to(device)

    out1 = torch_ln(x1 + y1)
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_ln(x2, y2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()

    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("ln_add out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    if is_training:
        x1_grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("ln_add x grad_diff: max={}, mean={}".format(x1_grad_diff.max(), x1_grad_diff.mean()))
        y1_grad_diff = torch.nn.functional.l1_loss(y1.grad, y2.grad, reduction='none')
        print("ln_add y grad_diff: max={}, mean={}".format(y1_grad_diff.max(), y1_grad_diff.mean()))


def test_accuracy_rms_add(is_training=True):
    print('\ntest_accuracy_rms_add: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    drop_prob = 0.0
    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fused_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                          bias_type='bias', drop_type='none', drop_prob=drop_prob).to(device)

    out1 = apex_rms(x1 + y1)
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_rms(x2, y2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()
    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("Rms_add out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    if is_training:
        x1_grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("Rms_add x grad_diff: max={}, mean={}".format(x1_grad_diff.max(), x1_grad_diff.mean()))
        y1_grad_diff = torch.nn.functional.l1_loss(y1.grad, y2.grad, reduction='none')
        print("Rms_add y grad_diff: max={}, mean={}".format(y1_grad_diff.max(), y1_grad_diff.mean()))


# ##########################################################################################
# dropout test accuracy
# ##########################################################################################

def get_dropout_fraction(dropout_mask, query_padding_mask=None, key_padding_mask=None, causal=False):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, 'b s -> b 1 s 1'), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, 'b s -> b 1 1 s'), False)
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool,
                                            device=dropout_mask.device), 1)
        dropped.masked_fill_(causal_mask, False)
    dropped_total = dropped.sum()
    query_lengths = (query_padding_mask.sum(dim=-1) if query_padding_mask is not None
                     else torch.full((batch_size,), seqlen_q, device=dropout_mask.device))
    key_lengths = (key_padding_mask.sum(dim=-1) if key_padding_mask is not None
                   else torch.full((batch_size,), seqlen_k, device=dropout_mask.device))
    if not causal:
        numel_per_batch = query_lengths * key_lengths
    else:
        numel_per_batch = torch.where(
            query_lengths <= key_lengths,
            query_lengths * (query_lengths + 1) / 2,
            query_lengths * key_lengths - (key_lengths * (key_lengths - 1) / 2)
        )
    return dropped_total / (numel_per_batch.sum() * nheads)


def test_accuracy_add_dropout(is_training=True):
    print('\ntest_accuracy_add_dropout: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    drop_prob = 0.0
    fused_ln = FusedLnRmsNormsAddDropout(norm_type='none', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='dropout', drop_prob=drop_prob).to(device)
    if is_training:
        fused_ln.train()
    else:
        fused_ln.eval()

    out1 = F.dropout(x1, p=drop_prob, training=is_training) + y1
    if is_training:
        out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_ln(x2, y2)
    if is_training:
        out2.backward(grad)
    torch.cuda.synchronize()
    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    print("add_dropout out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))

    if is_training:
        x1_grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
        print("add_dropout x grad_diff: max={}, mean={}".format(x1_grad_diff.max(), x1_grad_diff.mean()))
        # print('x1.grad: ', x1.grad, '\nx2.grad: ', x2.grad, '\nx1_grad_diff: ', x1_grad_diff)
        y1_grad_diff = torch.nn.functional.l1_loss(y1.grad, y2.grad, reduction='none')
        print("add_dropout y grad_diff: max={}, mean={}".format(y1_grad_diff.max(), y1_grad_diff.mean()))
    # print('fused_ln drop_mask: ', fused_ln.drop_mask, torch.max(fused_ln.drop_mask), torch.min(fused_ln.drop_mask))


def test_accuracy_ln_add_dropout(is_training=True):
    print('\ntest_accuracy_ln_add_dropout: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    drop_prob = 0.0

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fused_ln = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='dropout', drop_prob=drop_prob).to(device)
    if is_training:
        fused_ln.train()
    else:
        fused_ln.eval()

    # out1 = torch_ln(F.dropout(x1, p=drop_prob, training=is_training) + y1)
    out1 = torch_ln(x1 + y1)
    # out1 = (F.dropout(x1, p=droprate, training=is_training) + y1)
    out1.backward(grad)
    torch.cuda.synchronize()
    out2 = fused_ln(x2, y2)
    out2.backward(grad)
    torch.cuda.synchronize()

    out_diff = torch.nn.functional.l1_loss(out1, out2, reduction='none')
    # print('out1: ', out1[:3, :3, :3], '\nout2: ', out2[:3, :3, :3])
    print("ln_add_dropout out_diff: max={}, mean={}".format(out_diff.max(), out_diff.mean()))
    x1_grad_diff = torch.nn.functional.l1_loss(x1.grad, x2.grad, reduction='none')
    # print('x1.grad: ', x1.grad[:3, :3, :3], '\nx2.grad: ', x2.grad[:3, :3, :3])
    print("ln_add_dropout x grad_diff: max={}, mean={}".format(x1_grad_diff.max(), x1_grad_diff.mean()))
    y1_grad_diff = torch.nn.functional.l1_loss(y1.grad, y2.grad, reduction='none')
    # print('y1.grad: ', y1.grad[:3, :3, :3], '\ny2.grad: ', y2.grad[:3, :3, :3])
    print("ln_add_dropout y grad_diff: max={}, mean={}".format(y1_grad_diff.max(), y1_grad_diff.mean()))


# ##########################################################################################
# test speed
# ##########################################################################################
def _test_op_speed(op, op_name: str, is_training=True):
    x1 = x0.clone().detach().requires_grad_()

    for i in range(10):
        out1 = op(x1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()

    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op(x1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op_name, (end_time - start_time) / repeat_num * 1000))


def test_speed_ln(is_training=True):
    print('\ntest_speed_ln: x shape ', x0.shape)

    from apex.normalization import FusedLayerNorm
    from lightop.layernorm import LayerNorm

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fuseln = LayerNorm(cols).to(device)
    apex_ln = FusedLayerNorm(normalized_shape=cols).to(device="cuda", dtype=x0.dtype)
    fused_new_ln = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                             bias_type='none', drop_type='none').to(device)
    op1, op1_name = torch_ln, 'torch_ln'
    op2, op2_name = fuseln, 'fuseln'
    op3, op3_name = apex_ln, 'apex_ln'
    op4, op4_name = fused_new_ln, 'fused_new_ln'
    # #################################
    _test_op_speed(op1, op1_name, is_training)
    if cols > 64:
        _test_op_speed(op2, op2_name, is_training)
    _test_op_speed(op3, op3_name, is_training)
    _test_op_speed(op4, op4_name, is_training)


def test_speed_rms(is_training=True):
    print('\ntest_speed_rms: x shape ', x0.shape)

    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fuse_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                         bias_type='none', drop_type='none', drop_prob=0.5).to(device)
    op1, op1_name = apex_rms, 'apex_rms'
    op2, op2_name = fuse_rms, 'fuse_rms'
    # #################################
    _test_op_speed(op1, op1_name, is_training)
    _test_op_speed(op2, op2_name, is_training)


def test_speed_ln_add_dropout(is_training=True):
    print('\ntest_speed_ln_add_dropout: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    droprate = 0.2
    from lightop.fuselnadddropout import LnAddDropout

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fuselndropout = LnAddDropout(cols, droprate=droprate).to(device)

    fused_new_ln = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                             bias_type='bias', drop_type='none', drop_prob=droprate).to(device)
    if is_training:
        fuselndropout.train()
        fused_new_ln.train()
    else:
        fuselndropout.eval()
        fused_new_ln.eval()

    op1, op1_name = torch_ln, 'torch_ln'
    op2, op2_name = fuselndropout, 'fuselndropout'
    op4, op4_name = fused_new_ln, 'fused_new_ln'
    # #################################
    for i in range(10):
        out1 = torch_ln(F.dropout(x1, p=droprate, training=is_training) + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = torch_ln(F.dropout(x1, p=droprate, training=is_training) + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format('torch_ln', (end_time - start_time) / repeat_num * 1000))
    # #################################
    if cols >= 512:
        for i in range(10):
            out1 = fuselndropout(x1, y1)
            if is_training:
                out1.backward(grad)
            torch.cuda.synchronize()
        start_time = time.time()
        for i in range(int(repeat_num)):
            out1 = fuselndropout(x1, y1)
            if is_training:
                out1.backward(grad)
            torch.cuda.synchronize()
        end_time = time.time()
        print("{} time is {:.6f}".format('fuselndropout', (end_time - start_time) / repeat_num * 1000))
    # #################################
    for i in range(10):
        out1 = fused_new_ln(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = fused_new_ln(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format('fused_new_ln', (end_time - start_time) / repeat_num * 1000))
    # #################################


def test_speed_rms_add(is_training=True):
    print('\ntest_speed_rms_add: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    droprate = 0.0
    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fuse_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='none', drop_prob=droprate).to(device)
    if is_training:
        fuse_rms.train()
    else:
        fuse_rms.eval()

    op1, op1_name = apex_rms, 'apex_rms'
    op2, op2_name = fuse_rms, 'fuse_rms'
    # #################################
    for i in range(10):
        out1 = op1(x1 + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op1(x1 + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op1_name, (end_time - start_time) / repeat_num * 1000))

    # #################################
    for i in range(10):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op2_name, (end_time - start_time) / repeat_num * 1000))
    # #################################


def test_speed_rms_add_drop(is_training=True):
    print('\ntest_speed_rms_add_drop: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    droprate = 0.2
    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fuse_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='dropout', drop_prob=droprate).to(device)
    if is_training:
        fuse_rms.train()
    else:
        fuse_rms.eval()

    op1, op1_name = apex_rms, 'apex_rms+add+dropout'
    op2, op2_name = fuse_rms, 'fuse_rms_add_dropout'
    # #################################
    for i in range(10):
        out1 = op1(F.dropout(x1, p=droprate, training=is_training) + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op1(F.dropout(x1, p=droprate, training=is_training) + y1)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op1_name, (end_time - start_time) / repeat_num * 1000))

    # #################################
    for i in range(10):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op2_name, (end_time - start_time) / repeat_num * 1000))
    # #################################


def test_speed_add(is_training=False):
    print('\ntest_speed_add: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    fuse_add = FusedLnRmsNormsAddDropout(norm_type='none', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='none', drop_prob=0.5).to(device)
    if is_training:
        fuse_add.train()
    else:
        fuse_add.eval()

    op2, op2_name = fuse_add, 'fuse_add'
    # #################################
    for i in range(10):
        out1 = x1 + y1
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = x1 + y1
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format('torch add', (end_time - start_time) / repeat_num * 1000))
    # #################################
    for i in range(10):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    start_time = time.time()
    for i in range(int(repeat_num)):
        out1 = op2(x2, y2)
        if is_training:
            out1.backward(grad)
        torch.cuda.synchronize()
    end_time = time.time()
    print("{} time is {:.6f}".format(op2_name, (end_time - start_time) / repeat_num * 1000))
    # #################################


def test_speed_rms_add_drop_by_profiler(is_training=True):
    print('\ntest_speed_rms_add_drop_by_profiler: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    droprate = 0.2
    apex_rms = MixedFusedRMSNorm(cols).to(device)
    fuse_rms = FusedLnRmsNormsAddDropout(norm_type='rms', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='dropout', drop_prob=droprate).to(device)
    if is_training:
        fuse_rms.train()
    else:
        fuse_rms.eval()

    import os
    dst_dir = './profiler_results'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof:
        for _ in range(10):
            out1 = apex_rms(F.dropout(x1, p=droprate, training=is_training) + y1)
            out1.backward(grad)
            time.sleep(0.01)
    prof.export_chrome_trace('{}/rms_add_drop_profile_origin.json'.format(dst_dir))
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof2:
        for _ in range(10):
            out2 = fuse_rms(x2, y2)
            out2.backward(grad)
            time.sleep(0.01)
    prof2.export_chrome_trace('{}/rms_add_drop_profile_fuse.json'.format(dst_dir))


def test_speed_ln_add_drop_by_profiler(is_training=True):
    print('\ntest_speed_ln_add_drop_by_profiler: x shape ', x0.shape)
    x1 = x0.clone().detach().requires_grad_()
    x2 = x0.clone().detach().requires_grad_()
    y1 = y0.clone().detach().requires_grad_()
    y2 = y0.clone().detach().requires_grad_()

    droprate = 0.2
    from lightop.fuselnadddropout import LnAddDropout

    torch_ln = torch.nn.LayerNorm(cols).to(device)
    fuselndropout = LnAddDropout(cols, droprate=droprate).to(device)

    fuse_rms = FusedLnRmsNormsAddDropout(norm_type='ln', normalized_shape=cols, normalized_affine=True,
                                         bias_type='bias', drop_type='dropout', drop_prob=droprate).to(device)
    if is_training:
        fuse_rms.train()
    else:
        fuse_rms.eval()

    import os
    dst_dir = './profiler_results'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof:
        for _ in range(10):
            out1 = torch_ln(F.dropout(x1, p=droprate, training=is_training) + y1)
            out1.backward(grad)
            time.sleep(0.01)
    prof.export_chrome_trace('{}/ln_add_drop_profile_torch.json'.format(dst_dir))
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof2:
        for _ in range(10):
            out2 = fuselndropout(x2, y2)
            out2.backward(grad)
            time.sleep(0.01)
    prof2.export_chrome_trace('{}/ln_add_drop_profile_old_fuse.json'.format(dst_dir))
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                         profile_memory=False) as prof2:
        for _ in range(10):
            out2 = fuse_rms(x2, y2)
            out2.backward(grad)
            time.sleep(0.01)
    prof2.export_chrome_trace('{}/ln_add_drop_profile_new_fuse.json'.format(dst_dir))


def main():
    # #################### test accuracy ####################
    test_accuracy_training = True
    print('=' * 30)
    test_accuracy_ln(test_accuracy_training)
    print('=' * 30)
    test_accuracy_rms(test_accuracy_training)
    print('=' * 30)
    test_accuracy_ln_add(test_accuracy_training)
    print('=' * 30)
    test_accuracy_rms_add(test_accuracy_training)
    print('=' * 30)
    test_accuracy_add(test_accuracy_training)
    print('=' * 30)
    test_accuracy_add_dropout(test_accuracy_training)
    print('=' * 30)
    test_accuracy_ln_add_dropout(test_accuracy_training)
    print('=' * 30)

    # #################### test speed ####################
    test_speed_training = False

    test_speed_ln(is_training=test_speed_training)
    print('=' * 30)
    test_speed_rms(is_training=test_speed_training)
    print('=' * 30)
    test_speed_rms_add(is_training=test_speed_training)
    print('=' * 30)
    test_speed_rms_add_drop(is_training=test_speed_training)
    print('=' * 30)
    test_speed_add(is_training=test_speed_training)
    print('=' * 30)
    test_speed_ln_add_dropout(is_training=test_speed_training)

    # #################### test accuracy ####################
    test_speed_rms_add_drop_by_profiler()
    print('=' * 30)
    test_speed_ln_add_drop_by_profiler()
    print('=' * 30)


if __name__ == "__main__":
    repeat_num = 1e2

    batchsize, seqs, cols = 64, 1000, 512
    # batchsize, seqs, cols = 20, 30, 511
    # batchsize, seqs, cols = 20, 30, 128
    # batchsize, seqs, cols = 2, 3, 12

    device = torch.device('cuda')
    torch.manual_seed(0)
    x0 = torch.rand([batchsize, seqs, cols]).cuda()
    y0 = torch.rand([batchsize, seqs, cols]).cuda()
    # y0 = torch.zeros([batchsize, seqs, cols]).cuda()
    grad = torch.rand([batchsize, seqs, cols]).cuda()

    main()
