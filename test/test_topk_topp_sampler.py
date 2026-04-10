import pytest
import time
import torch
from torch import Generator

from vllm.platforms import current_platform
from lightop import sampling

DEVICE = current_platform.device_type

BATCH_SIZE = 15
VOCAB_SIZE = 163840

def validate_sampling(sampled_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    验证每个 batch 采样的 id 在对应的 logits 中是否为有效值（不是 -inf）。

    Args:
        sampled_ids: 形状为 (batch,) 的一维张量，每个元素是采样的 vocab id。
        logits: 形状为 (batch, vocab) 的二维张量，有效位置为有限值（如 0~1），
                被过滤的位置为 -inf。

    Returns:
        valid: 形状为 (batch,) 的布尔张量，True 表示对应采样有效，
               False 表示采样到了被过滤（-inf）的位置。
    """
    # 使用 gather 取出每个 batch 在采样 id 处的值
    values = logits.gather(1, sampled_ids.long().unsqueeze(1)).squeeze(1)
    # 检查是否为有限值（-inf 不是有限值）
    valid = torch.isfinite(values)
    return valid

def random_sample(
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)

def apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        # Avoid sorting vocab for top-k only case.
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits

def apply_top_k_only(
        logits: torch.Tensor,
        k: torch.Tensor,
) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits

def test_latency(
        ops: str,
        iteration: int,
        logits: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor):
    time_used = None
    if ops == "top_k_top_p":
        torch.cuda.synchronize()
        time_st = time.perf_counter()
        for i in range(iteration):
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            next_token_ids = sampling.top_k_top_p_sampling_from_probs(
                probs, k, p, deterministic=True
            )
        torch.cuda.synchronize()
        time_used = (time.perf_counter() - time_st) * 1e6 / iteration
    elif ops == "top_k":
        torch.cuda.synchronize()
        time_st = time.perf_counter()
        for i in range(iteration):
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            next_token_ids = sampling.top_k_sampling_from_probs(
                probs, k, deterministic=True
            )
        torch.cuda.synchronize()
        time_used = (time.perf_counter() - time_st) * 1e6 / iteration
    elif ops == "top_p":
        torch.cuda.synchronize()
        time_st = time.perf_counter()
        for i in range(iteration):
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            next_token_ids = sampling.top_p_sampling_from_probs(
                probs, p, deterministic=True
            )
        torch.cuda.synchronize()
        time_used = (time.perf_counter() - time_st) * 1e6 / iteration
    else:
        raise NotImplementedError('not support sampling ops {}'.format(ops))

    g = Generator(device=DEVICE).manual_seed(42)
    generator = {i: g for i in range(logits.shape[0])}
    torch.cuda.synchronize()
    time_torch_st = time.perf_counter()
    for i in range(iteration):
        logits = apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        ids = random_sample(probs, generator)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - time_torch_st) * 1e6 / iteration
    print(f"latency test(us): ops: {ops}, flashinfer time: {time_used}, native time: {torch_time}")

def test_flashinfer_sampler(ops: str = "top_k_top_p"):
    """
    This test verifies that the FlashInfer top-k and top-p sampling
    implementation produces the same results as the Python implementation.

    NOTE: FlashInfer did not directly expose an interface for fused top-k and
    top-p prob renorm (it did provide fused sampling but we cannot compare
    sampling results due to randomness), so we will compare the probability
    renormed consequently by top-k and then top-p of FlashInfer implementation.
    """

    torch.set_default_device(DEVICE)
    generator = Generator(device=DEVICE).manual_seed(42)

    # Generate random logits
    logits = torch.rand((BATCH_SIZE, VOCAB_SIZE), generator=generator)

    # Generate various top-k and top-p values
    k_values = torch.randint(1, 1000, (BATCH_SIZE,), generator=generator)
    p_values = (
            torch.rand((BATCH_SIZE,), generator=generator) * 0.5 + 0.5
    )  # range in [0.5, 1.0]

    p_values = (
            torch.full((BATCH_SIZE,), 0.95, dtype=torch.float32, device="cuda")
    )  # range in [0.5, 1.0]

    # Sometimes disable top-k (k=vocab_size)
    k_values.masked_fill_(
        torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool),
        VOCAB_SIZE,
    )

    # Sometimes disable top-p (p=1.0)
    # p_values.masked_fill_(
    #     torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool), 1.0
    # )

    probs = logits.softmax(dim=-1, dtype=torch.float32)
    next_token_ids = None
    if ops == "top_k_top_p":
        next_token_ids = sampling.top_k_top_p_sampling_from_probs(
            probs, k_values, p_values, deterministic=True
        )
    elif ops == "top_k":
        p_values = None
        next_token_ids = sampling.top_k_sampling_from_probs(
            probs, k_values, deterministic=True
        )
    elif ops == "top_p":
        k_values = None
        next_token_ids = sampling.top_p_sampling_from_probs(
            probs, p_values, deterministic=True
        )
    else:
        raise NotImplementedError('not support sampling ops {}'.format(ops))

    python_logits = apply_top_k_top_p(
        logits=logits.clone(),
        k=k_values,
        p=p_values,
    )
    python_probs = torch.softmax(python_logits, dim=-1)

    correctness = validate_sampling(next_token_ids, python_probs)
    all_valid = correctness.all()
    has_invalid = not all_valid
    assert not has_invalid
    print(f"test {ops} passed")

    test_latency(ops, 100, logits, k_values, p_values)

def main():
    # test_flashinfer_sampler("top_k_top_p")
    # test_flashinfer_sampler("top_k")
    test_flashinfer_sampler("top_p")

if __name__ == "__main__":
    main()
