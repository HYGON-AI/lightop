import torch
from typing import Optional, Tuple, Union
from lightop import gemma_fused_add_rmsnorm 

def calc_metrics(a: torch.Tensor, b: torch.Tensor, name: str = "tensor"):
    a = a.float().detach().cpu()
    b = b.float().detach().cpu()

    diff = a - b
    abs_diff = diff.abs()

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()

    denom = torch.clamp(a.abs(), min=1e-8)
    rel_err = (abs_diff / denom).mean().item()

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1
    ).item()

    allclose_1e2 = torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    allclose_1e3 = torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    allclose_1e4 = torch.allclose(a, b, atol=1e-4, rtol=1e-4)

    print(f"\n[{name}]")
    print(f"  max_abs_error   : {max_abs:.8e}")
    print(f"  mean_abs_error  : {mean_abs:.8e}")
    print(f"  rmse            : {rmse:.8e}")
    print(f"  mean_rel_error  : {rel_err:.8e}")
    print(f"  cosine_similarity: {cosine:.8f}")
    print(f"  allclose(atol=1e-2, rtol=1e-2): {allclose_1e2}")
    print(f"  allclose(atol=1e-3, rtol=1e-3): {allclose_1e3}")
    print(f"  allclose(atol=1e-4, rtol=1e-4): {allclose_1e4}")


def compare_outputs(
    module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
):
    # 为了避免原地修改影响另一条路径，拷贝一份
    x1 = x.clone()
    x2 = x.clone()

    residual1 = residual.clone() if residual is not None else None
    residual2 = residual.clone() if residual is not None else None

    post1 = (
        post_residual_addition.clone()
        if post_residual_addition is not None
        else None
    )
    post2 = (
        post_residual_addition.clone()
        if post_residual_addition is not None
        else None
    )

    with torch.no_grad():
        out_native = module.forward_native(x1, residual1, post1)
        out_hcu = module.forward_hcu(x2, residual2, post2)

    # 统一处理返回类型
    if isinstance(out_native, tuple):
        native_x, native_residual = out_native
    else:
        native_x, native_residual = out_native, None

    if isinstance(out_hcu, tuple):
        hcu_x, hcu_residual = out_hcu
    else:
        hcu_x, hcu_residual = out_hcu, None

    print("=" * 80)
    print("dtype info:")
    print(f"  input x dtype          : {x.dtype}")
    if residual is not None:
        print(f"  input residual dtype   : {residual.dtype}")
    if post_residual_addition is not None:
        print(f"  post_residual dtype    : {post_residual_addition.dtype}")
    print(f"  native output dtype    : {native_x.dtype}")
    print(f"  hcu output dtype       : {hcu_x.dtype}")

    calc_metrics(native_x, hcu_x, name="output_x")

    if native_residual is not None and hcu_residual is not None:
        calc_metrics(native_residual, hcu_residual, name="output_residual")
    elif native_residual is None and hcu_residual is None:
        print("\n[output_residual]")
        print("  both are None")
    else:
        print("\n[output_residual]")
        print("  mismatch: one side is None, the other is not None")


# ==========================
# 你自己的模块类
# ==========================
#
# 把这里替换成你的真实类
#
class YourRMSNormClass(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward_hcu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 这里请替换成你的真实 forward_hcu
        orig_dtype = x.dtype
        if post_residual_addition is not None:
            residual = (residual + post_residual_addition)
        # residual= residual.float()
        # x = x.float()
        # weight = self.weight.float()

        x, residual = gemma_fused_add_rmsnorm(
            x, residual, self.weight, self.variance_epsilon
        )
        # x = x.to(orig_dtype)
        # residual = residual.to(orig_dtype)
        return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)
def main():
    torch.manual_seed(2026)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    batch = 4
    seq = 128
    hidden = 4096

    module = YourRMSNormClass(hidden_size=hidden, eps=1e-6).to(device)
    # 场景2：有 residual
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    residual = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    print("\n\n###### Case 1: residual is not None ######")
    compare_outputs(module, x, residual=residual)

    # 场景3：有 residual + post_residual_addition
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    residual = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    post = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    print("\n\n###### Case 2: residual + post_residual_addition ######")
    compare_outputs(module, x, residual=residual, post_residual_addition=post)


if __name__ == "__main__":
    main()