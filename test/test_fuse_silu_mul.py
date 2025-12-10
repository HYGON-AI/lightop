import torch
import triton
from lightop import fuse_silu_and_mul
import torch.nn.functional as F
import time

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

if __name__ == "__main__":
    row = 1024
    input_col = 768
    output_col = 384
    input = torch.rand((row, input_col)).cuda().half()
    output = torch.rand((row, output_col)).cuda().half()
    
    repeat_num = 10
    start_time = time.time()
    for i in range(repeat_num):
        fuse_silu_and_mul(input, output)
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    print('size=[',row,',',input_col,',',output_col, ']')
    print('fuse_silu_and_mul', 'averange time is', fc1_espl *1000/ repeat_num, 'ms')

    d = input.shape[-1] // 2
    t_output = F.silu(input[..., :d]) * input[..., d:]

    print(allclose(output,t_output,rtol=1e-02, atol=1e-02))
