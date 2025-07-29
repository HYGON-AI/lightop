import lightop
import torch

# 定义张量维度
num_tokens = 256
hidden_size = 16
topk = 2
num_shared_experts_per_rank = 1
expert_start_idx = 16
expert_end_idx = 32
num_experts_per_rank = expert_end_idx - expert_start_idx
num_tokens_offset = 16
num_tokens_per_rank = 64

# 生成随机输入数据 (bfloat16)
hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
selected_experts = torch.randint(0, 2, (num_tokens, topk), dtype=torch.int32).cuda()
smooth = torch.randn(num_shared_experts_per_rank + num_experts_per_rank, hidden_size, dtype=torch.float32).cuda()

# 调用被测函数
scatter_tokens, scatter_per_token_scale, per_tokeexperts_token_countn_scale, token_offset = (
    lightop.miopen_moe_scatter_dynamic_quant(
        hidden,
        selected_experts,
        smooth,
        expert_start_idx,
        expert_end_idx,
        num_shared_experts_per_rank,
        num_tokens_offset,
        num_tokens_per_rank,
    )
)

torch.cuda.synchronize()
print("miopen_moe_scatter_dynamic_quant scatter_tokens is:\n", scatter_tokens)


print("----miopen_moe_scatter_dynamic_quant end -------------\n")

tokens_num = 8192
hidden_size = 4096
num_shared_experts_per_rank = 2
num_tokens_offset = 0
num_experts_per_rank = 6
num_tokens_per_rank = 1024
topk = 6

allocated_tokens = num_tokens_per_rank * topk

scatter_tokens1 = torch.randn(
    num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size * 2, dtype=torch.bfloat16
).cuda()
smooth = torch.randn(num_shared_experts_per_rank + num_experts_per_rank, hidden_size, dtype=torch.float32).cuda()
experts_tokens_count = torch.randint(0, 2, (num_experts_per_rank,), dtype=torch.int32).cuda()
y, per_token_scale = lightop.miopen_moe_swiglu_dynamic_quant(
    scatter_tokens1,
    smooth,
    experts_tokens_count,
    num_shared_experts_per_rank,
    num_tokens_offset,
    num_tokens_per_rank,
    1.0,
)
print("miopen_moe_swiglu_dynamic_quant y is:\n", y)

print("----moe_swiglu_dynamic_quant end-------------\n")


x1 = torch.randn(
    num_shared_experts_per_rank * num_tokens_per_rank + num_tokens_per_rank * topk, hidden_size, dtype=torch.bfloat16
).cuda()
x2 = torch.randn(num_tokens_per_rank * topk, dtype=torch.bfloat16).cuda()
x3 = torch.randn(num_tokens, topk, dtype=torch.float32).cuda()

y = lightop.miopen_moe_gather(x1, x2, x3, num_shared_experts_per_rank, num_tokens_offset, num_tokens_per_rank)

print("miopen_moe_gather y is:\n", y)

print("---------moe_gather end-------------\n")
