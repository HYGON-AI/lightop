import torch
import functools
from typing import Optional, Tuple, Union
from .. import op

@functools.cache
def get_default_generators(device: torch.device):
    torch.cuda.init()
    return torch.cuda.default_generators[device.index]

def get_seed_and_offset(
        increment: int,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
) -> Tuple[int, int]:
    if generator is None:
        generator = get_default_generators(device)
    # add mutex if multi-trheading needed
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    generator.set_state(
        torch.tensor(
            [seed, offset], dtype=torch.int64, device=torch.device("cpu")
        ).view(torch.uint8)
    )
    return int(seed), int(offset)

def top_k_sampling_from_probs(
        probs: torch.Tensor,
        top_k: Union[torch.Tensor, int],
        indices: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
        check_nan: bool = False,
        seed: Optional[int] = None,
        offset: Optional[int] = None,
) -> torch.Tensor:
    if top_k is None:
        raise ValueError("top_k is None.")

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")

    device = probs.device
    probs = probs.float()
    batch_size = indices.size(0) if indices is not None else probs.size(0)
    out_dtype = indices.dtype if indices is not None else torch.int32
    samples = torch.empty(batch_size, dtype=out_dtype, device=device)
    if seed is None or offset is None:
        seed, offset = get_seed_and_offset(batch_size * 32, generator, device)

    if isinstance(top_k, torch.Tensor):
        op.top_k_sampling_from_probs(
            probs,
            samples,
            indices,
            top_k.int(),
            0,
            deterministic,
            seed,
            offset,
            )
    else:
        op.top_k_sampling_from_probs(
            probs,
            samples,
            indices,
            None,
            top_k,
            deterministic,
            seed,
            offset,
            )

    return samples

def top_p_sampling_from_probs(
        probs: torch.Tensor,
        top_p: Union[torch.Tensor, float],
        indices: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
        check_nan: bool = False,
        seed: Optional[int] = None,
        offset: Optional[int] = None,
) -> torch.Tensor:
    if top_p is None:
        raise ValueError("top_p is None.")

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")

    device = probs.device
    probs = probs.float()
    batch_size = indices.size(0) if indices is not None else probs.size(0)
    out_dtype = indices.dtype if indices is not None else torch.int32
    samples = torch.empty(batch_size, dtype=out_dtype, device=device)
    if seed is None or offset is None:
        seed, offset = get_seed_and_offset(batch_size * 32, generator, device)

    if isinstance(top_p, torch.Tensor):
        op.top_p_sampling_from_probs(
            probs,
            samples,
            indices,
            top_p.float(),
            0,
            deterministic,
            seed,
            offset,
        )
    else:
        op.top_p_sampling_from_probs(
            probs,
            samples,
            indices,
            None,
            top_p,
            deterministic,
            seed,
            offset,
        )
    return samples

def top_k_top_p_sampling_from_probs(
        probs: torch.Tensor,
        top_k: Union[torch.Tensor, int],
        top_p: Union[torch.Tensor, float],
        indices: Optional[torch.Tensor] = None,
        filter_apply_order: str = "joint",
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
        check_nan: bool = False,
        seed: Optional[int] = None,
        offset: Optional[int] = None,
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()

    if top_k is None and top_p is None:
        raise ValueError("top_k and top_p both invalid.")

    if top_k is None:
        return top_p_sampling_from_probs(probs, top_p, indices, deterministic, generator, check_nan, seed, offset)

    if top_p is None:
        return top_k_sampling_from_probs(probs, top_k, indices, deterministic, generator, check_nan, seed, offset)

    batch_size = indices.size(0) if indices is not None else probs.size(0)
    out_dtype = indices.dtype if indices is not None else torch.int32
    samples = torch.empty(batch_size, dtype=out_dtype, device=device)
    if seed is None or offset is None:
        seed, offset = get_seed_and_offset(batch_size * 32, generator, device)
    if filter_apply_order == "top_k_first":
        if check_nan and torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
        if isinstance(top_k, torch.Tensor):
            if isinstance(top_p, torch.Tensor):
                op.top_k_first_top_p_sampling_from_probs(
                    probs, samples, indices, top_k.int(), 0, top_p.float(), 0,
                    deterministic, seed, offset,
                )
            else:
                op.top_k_first_top_p_sampling_from_probs(
                    probs, samples, indices, top_k.int(), 0, None, top_p,
                    deterministic, seed, offset,
                )
        else:
            if isinstance(top_p, torch.Tensor):
                op.top_k_first_top_p_sampling_from_probs(
                    probs, samples, indices, None, top_k, top_p.float(), 0,
                    deterministic, seed, offset,
                )
            else:
                op.top_k_first_top_p_sampling_from_probs(
                    probs, samples, indices, None, top_k, None, top_p,
                    deterministic, seed, offset,
                )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        if isinstance(top_k, torch.Tensor):
            if isinstance(top_p, torch.Tensor):
                op.top_k_top_p_sampling_from_probs(
                    probs,
                    samples,
                    indices,
                    top_k.int(),
                    0,
                    top_p.float(),
                    0,
                    deterministic,
                    seed,
                    offset,
                )
            else:
                op.top_k_top_p_sampling_from_probs(
                    probs,
                    samples,
                    indices,
                    top_k.int(),
                    0,
                    None,
                    top_p,
                    deterministic,
                    seed,
                    offset,
                )
        else:
            if isinstance(top_p, torch.Tensor):
                op.top_k_top_p_sampling_from_probs(
                    probs,
                    samples,
                    indices,
                    None,
                    top_k,
                    top_p.float(),
                    0,
                    deterministic,
                    seed,
                    offset,
                )
            else:
                op.top_k_top_p_sampling_from_probs(
                    probs,
                    samples,
                    indices,
                    None,
                    top_k,
                    None,
                    top_p,
                    deterministic,
                    seed,
                    offset,
                )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")

    return samples