import torch
import triton
from lightop import fuse_silu_mul


if __name__ == "__main__":
    row = 8192
    col = 768
    input = torch.rand((row, col)).cuda().half()
    output = torch.rand((row, col / 2)).cuda().half()
    fuse_silu_mul(input, output)
