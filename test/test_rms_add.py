import torch
import itertools
from lightop.norm import fused_add_rms_norm_opt

def torch_fused_add_rms_norm(input_tensor, residual_tensor, weight_tensor, eps):
    """
    PyTorch 参考实现。
    为了保证精度，计算过程在 float32 下进行，最后转回原类型。
    """

    input_f32 = input_tensor.float()
    residual_f32 = residual_tensor.float()
    weight_f32 = weight_tensor.float()
    

    new_residual_f32 = input_f32 + residual_f32
    
 
    variance = new_residual_f32.pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    new_input_f32 = new_residual_f32 * rsqrt * weight_f32
    
  
    new_residual = new_residual_f32.to(input_tensor.dtype)
    new_input = new_input_f32.to(input_tensor.dtype)
    
    return new_input, new_residual

def run_test(num_tokens, hidden_size, dtype, eps=1e-5):
    device = 'cuda'
    

    input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight_tensor = torch.randn(hidden_size, dtype=dtype, device=device)
    

    custom_input = input_tensor.clone()
    custom_residual = residual_tensor.clone()
    

    ref_input, ref_residual = torch_fused_add_rms_norm(input_tensor, residual_tensor, weight_tensor, eps)
    

    fused_add_rms_norm_opt(custom_input, custom_residual, weight_tensor, eps)
    
  
    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3
        
    try:
        torch.testing.assert_close(custom_input, ref_input, atol=atol, rtol=rtol)
        torch.testing.assert_close(custom_residual, ref_residual, atol=atol, rtol=rtol)
        precision_pass = True
    except AssertionError as e:
        precision_pass = False
        print(f"Precision failed for shape [num_tokens={num_tokens}, hidden_size={hidden_size}], dtype={dtype}: {e}")
        return
        
    # 性能测试 
    warmup_iters = 10
    test_iters = 100
    
    # 预热 
    for _ in range(warmup_iters):
        fused_add_rms_norm_opt(custom_input, custom_residual, weight_tensor, eps)
    torch.cuda.synchronize()
    

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(test_iters):
        fused_add_rms_norm_opt(custom_input, custom_residual, weight_tensor, eps)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms * 1000) / test_iters
    
    return precision_pass, avg_time_us

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    num_tokens_list = [ 128,288,352]
    hidden_sizes_list = [4096]
    dtypes = [torch.float16, torch.bfloat16]
    
    print(f"{'Tokens':<10} | {'Hidden':<8} | {'Dtype':<12} | {'Precision':<10} | {'Time (us)':<10}")
    print("-" * 60)
    
    for dtype, tokens, hidden in itertools.product(dtypes, num_tokens_list, hidden_sizes_list):
        result = run_test(tokens, hidden, dtype)
        if result is not None:
            precision_pass, avg_time_us = result
            pass_str = "PASS" if precision_pass else "FAIL"
            dtype_str = "FP16" if dtype == torch.float16 else "BF16"
            print(f"{tokens:<10} | {hidden:<8} | {dtype_str:<12} | {pass_str:<10} | {avg_time_us:.2f}")

if __name__ == "__main__":
    main()