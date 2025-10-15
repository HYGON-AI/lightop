from lightop import op as ops
import torch
import os

def benchmark_gpu_timing_and_correctness(num_runs=50):
    """
    GPU前向和后向时间测试，以及与PyTorch参考实现比较
    """
    device = torch.device("cuda")
    B = 2
    H = 8
    C_per_head = int(os.getenv("C_PER_HEAD", 32))
    print(f"当前使用的 C_per_head = {C_per_head}")
    embed_dim = H * C_per_head

    value_spatial_shapes = torch.tensor([[32, 32], [64, 64], [128, 128]], dtype=torch.long)
    level_start_index = torch.cat((
        value_spatial_shapes.new_zeros((1,)),
        (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).cumsum(0)[:-1]
    ))
    S = (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).sum().item()
    L = value_spatial_shapes.shape[0]
    P = 4
    Lq = S
    im2col_step = 64

    # 数据准备
    torch.manual_seed(3)
    value_base = torch.rand(B, S, H, C_per_head) * 0.01
    sampling_locations_base = torch.rand(B, Lq, H, L, P, 2)
    attention_weights_base = torch.rand(B, Lq, H, L, P) + 1e-5
    attention_weights_base = attention_weights_base / (
        attention_weights_base.sum(-1, keepdim=True).sum(-2, keepdim=True)
    )

    value = value_base.clone().to(device).requires_grad_(True)
    sampling_locations = sampling_locations_base.clone().to(device).requires_grad_(True)
    attention_weights = attention_weights_base.clone().to(device).requires_grad_(True)
    shapes_gpu = value_spatial_shapes.to(device)
    level_start_index_gpu = level_start_index.to(device)

    # 前向预热
    for _ in range(5):
        _ = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                       sampling_locations, attention_weights, im2col_step)
    torch.cuda.synchronize()

    # ---------------- ops 前向计时 ----------------
    forward_times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                       sampling_locations, attention_weights, im2col_step)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))
    print(f"[ops] 前向平均时间: {sum(forward_times)/len(forward_times):.3f} ms")

    # ---------------- ops 反向计时 ----------------
    backward_times = []
    for _ in range(num_runs):
        if value.grad is not None:
            value.grad.zero_()
        if sampling_locations.grad is not None:
            sampling_locations.grad.zero_()
        if attention_weights.grad is not None:
            attention_weights.grad.zero_()
        output = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                           sampling_locations, attention_weights, im2col_step)
        grad_output = torch.ones_like(output)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        ops.ms_deform_attn_backward(value, shapes_gpu, level_start_index_gpu,
                                    sampling_locations, attention_weights,
                                    grad_output,
                                    grad_value := torch.zeros_like(value),
                                    grad_sampling_loc := torch.zeros_like(sampling_locations),
                                    grad_attn_weight := torch.zeros_like(attention_weights),
                                    im2col_step)
        end_event.record()
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event))
    print(f"[ops] 反向平均时间: {sum(backward_times)/len(backward_times):.3f} ms")

    # ==================== 前向正确性对比 ====================
    gpu_output = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                            sampling_locations, attention_weights, im2col_step).detach().cpu()
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    pytorch_output = multi_scale_deformable_attn_pytorch(
        value.detach().cpu(), value_spatial_shapes, sampling_locations.detach().cpu(),
        attention_weights.detach().cpu()
    ).detach()
    is_forward_close = torch.allclose(gpu_output, pytorch_output, rtol=1e-2, atol=1e-2)
    print(f"前向输出匹配: {'✓' if is_forward_close else '✗'}")

    # ==================== 反向正确性对比 ====================
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)
    output = ops.ms_deform_attn_forward(value, shapes_gpu, level_start_index_gpu,
                                        sampling_locations, attention_weights, im2col_step)
    grad_output = torch.ones_like(output)
    ops.ms_deform_attn_backward(value, shapes_gpu, level_start_index_gpu,
                                sampling_locations, attention_weights,
                                grad_output,
                                grad_value, grad_sampling_loc, grad_attn_weight,
                                im2col_step)

    # PyTorch reference gradients
    value_ref = value.detach().cpu().clone().requires_grad_(True)
    sampling_ref = sampling_locations.detach().cpu().clone().requires_grad_(True)
    attn_ref = attention_weights.detach().cpu().clone().requires_grad_(True)
    out_ref = multi_scale_deformable_attn_pytorch(value_ref, value_spatial_shapes,
                                                  sampling_ref, attn_ref)
    out_ref.backward(torch.ones_like(out_ref))

    is_value_grad_close = torch.allclose(grad_value.cpu(), value_ref.grad, rtol=1e-2, atol=1e-2)
    is_sampling_grad_close = torch.allclose(grad_sampling_loc.cpu(), sampling_ref.grad, rtol=1e-2, atol=1e-2)
    is_attn_grad_close = torch.allclose(grad_attn_weight.cpu(), attn_ref.grad, rtol=1e-2, atol=1e-2)
    print(f"反向梯度匹配: {'✓' if is_value_grad_close and is_sampling_grad_close and is_attn_grad_close else '✗'}")

    # ==================== PyTorch 前向/反向时间 ====================
    # PyTorch 前向计时
    value_ref = value_base.clone().to(device).requires_grad_(True)
    sampling_ref = sampling_locations_base.clone().to(device).requires_grad_(True)
    attn_ref = attention_weights_base.clone().to(device).requires_grad_(True)
    torch.cuda.synchronize()
    torch_forward_times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out_ref = multi_scale_deformable_attn_pytorch(value_ref, value_spatial_shapes, sampling_ref, attn_ref)
        end_event.record()
        torch.cuda.synchronize()
        torch_forward_times.append(start_event.elapsed_time(end_event))
    print(f"[PyTorch] 前向平均时间: {sum(torch_forward_times)/len(torch_forward_times):.3f} ms")

    # PyTorch 反向计时
    torch_backward_times = []
    for _ in range(num_runs):
        if value_ref.grad is not None:
            value_ref.grad.zero_()
        if sampling_ref.grad is not None:
            sampling_ref.grad.zero_()
        if attn_ref.grad is not None:
            attn_ref.grad.zero_()
        out_ref = multi_scale_deformable_attn_pytorch(value_ref, value_spatial_shapes, sampling_ref, attn_ref)
        grad_output_ref = torch.ones_like(out_ref)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out_ref.backward(grad_output_ref)
        end_event.record()
        torch.cuda.synchronize()
        torch_backward_times.append(start_event.elapsed_time(end_event))
    print(f"[PyTorch] 反向平均时间: {sum(torch_backward_times)/len(torch_backward_times):.3f} ms")


if __name__ == "__main__":
    benchmark_gpu_timing_and_correctness()
