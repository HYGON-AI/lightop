import lightop
import torch

# 定义张量维度
num_tokens = 1024
hidden_size = 8192
topk = 128
num_shared_experts_per_rank = 0
num_experts_per_rank = 128

expert_start_idx = 4
expert_end_idx = 9
num_tokens_offset = 4
num_tokens_per_rank = 8

# 生成随机输入数据 (bfloat16)
hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16).cuda()
selected_experts = torch.randn(num_tokens, topk, dtype=torch.bfloat16).cuda()
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
print(scatter_tokens)


print("----moe_swiglu_dynamic_quant-------------\n")

scatter_tokens1 = torch.randn(
    num_shared_experts_per_rank * num_tokens_per_rank + num_tokens_per_rank * topk, hidden_size, dtype=torch.bfloat16
).cuda()
experts_tokens_count = torch.randn(num_experts_per_rank, dtype=torch.bfloat16).cuda()
y, per_token_scale = lightop.miopen_moe_swiglu_dynamic_quant(
    scatter_tokens1,
    smooth,
    experts_tokens_count,
    num_shared_experts_per_rank,
    num_tokens_offset,
    num_tokens_per_rank,
    1.0,
)
print(y)

print("----moe_swiglu_dynamic_quant-------------\n")


x1 = torch.randn(
    num_shared_experts_per_rank * num_tokens_per_rank + num_tokens_per_rank * topk, hidden_size, dtype=torch.bfloat16
).cuda()
x2 = torch.randn(num_tokens_per_rank * topk, dtype=torch.bfloat16).cuda()
x3 = torch.randn(num_tokens, topk, dtype=torch.float32).cuda()

y = lightop.miopen_moe_gather(x1, x2, x3, num_shared_experts_per_rank, num_tokens_offset, num_tokens_per_rank)

print(y)

print("----moe_gather-------------\n")
