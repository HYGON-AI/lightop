# Fused MoE gate dispatch and unsupported configurations

`moe_fused_gate` contains specialized kernels for the routing layouts that
have been validated on the HCU backend.  The host-side dispatcher validates
the complete layout (expert count, expert-group count, `topk_group`, `topk`,
and fused shared-expert count) before launching one of those kernels.

## Why the dynamic fallback was removed

Older revisions launched `moe_fused_gate_kernel_dynamic` whenever no
specialized case matched.  That kernel was not a general implementation:

* its per-lane input and scratch arrays were statically sized to eight
  elements, while `VPT = (num_experts / num_expert_group) / 4` was computed at
  runtime;
* values such as 256 experts with one group therefore made the loop bounds
  exceed those arrays, while values with `VPT < 8` could read past the end of a
  row;
* arbitrary `topk`/group/shared-expert combinations were not fully checked,
  and the old reduction did not define a deterministic expert-id tie break.

As a result, an unsupported model could silently receive corrupted routing
indices or weights instead of an error.  This is especially dangerous for
MoE models because the output can remain numerically plausible while causing
large downstream accuracy regressions or repeated generations.

The dynamic kernel and its implicit fallback are consequently removed.  The
dispatcher now raises a descriptive `TORCH_CHECK` for an unvalidated layout.
This is an intentional fail-fast behavior: it prevents a model from silently
using an unsafe kernel and makes the required workaround visible in logs.

## Expected behavior for callers

Validated layouts retain their specialized kernels (with deterministic
tie-breaking).  A model that needs a new layout must first add a bounded,
tested specialization.  Until then, vLLM HCU callers should disable the fused
gate (`VLLM_HCU_USE_FUSE_MOE_GATE=0`) and use the standard router rather than
relying on an unvalidated fallback.

The error message reports the requested expert/group/top-k/shared-expert
values and the currently supported layouts, so future model integration does
not require diagnosing a silent precision failure.

## Optional index width

The final API argument, `output_indices_int64=False`, preserves the legacy
`int32` index output by default. Passing `True` requests direct `int64` writes
from a supported group-1 kernel; weights remain `float32`. The index-width
flag does not change expert selection, tie-breaking, normalization, or the
separate `apply_routed_scaling_factor_on_output` option.

LightOp validates support before allocating outputs or launching a kernel.
Other specializations remain available with the default `int32` output and
reject `output_indices_int64=True` with a descriptive error. Callers should
not duplicate the expert-count/top-k capability table.

Both widths use the same deterministic tie-break: the lower expert ID wins
when the FP32 selection scores are exactly equal. Tests cover equal biases
with saturated logits, tied scores with unequal underlying probabilities,
normalization, shared experts, empty batches, and graph replay after changing
the inputs. Index values and weight bits must match between the two widths.
