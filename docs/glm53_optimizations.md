# Explicit GLM 5.3 FP8 optimization APIs

This change packages five previously isolated GLM experiments as explicit
`lightop.op` entrypoints. Standard LightOp operators and beta MQA configuration
remain unchanged. The five new entrypoints advertise `GLM53_OPT_ABI_VERSION=1`;
callers still own model, phase and graph eligibility. No environment variable
is added or automatically enabled by this change.

| Native API | Supported operands and arithmetic |
| --- | --- |
| glm53_prefill_add_rms_align16(x,residual,weight,eps) | BF16 x/residual [M,6144], M704/832/1408/3520/4096, BF16 weight6144, aligned16, in-place writes; FP32 residual addition and normalization |
| glm53_prefill_ep_gather_k8(x,ids,weights,src,out) | BF16 H6144, int64 Top8 IDs, FP32 weights, int32 inverse map, no expert map; FP32 accumulation in K0..7 order |
| glm53_prefill_token_quant_inverse(x,out,scale) | BF16 [4096,16384], E4M3FN output, FP32 [4096,1] scales; scale=absmax/448, zero-row inverse=0 |
| glm53_prefill_activation_inverse(x,out,scale) | BF16 [M,4096], byte [M,2048] FP8 output, FP32 scales, M14336/84224/84992; BF16 SiLU/mul and original scale floor, no clamp/count |
| glm53_decode_token_quant_broadcast(out,x,scale) | BF16 [M,16384], E4M3FN output, FP32 [M,1] scales, 1<=M<=30; disjoint backing storage and 64-byte input/output alignment |

SGLang restricts the prefill APIs to ordinary main-model prefix EXTEND in the
retained TP/CP/EP8 context; graph and speculative paths are excluded. Decode
quant is only eligible within MLA o_proj: M1..5 in DECODE or M6/12/18/24/30 in
TARGET_VERIFY/DRAFT_EXTEND_V2. The kernel wrapper's wider numeric row interval
is not a claim of framework coverage for every M.

Build uses the repository's existing top-level csrc/*.cu glob and native op
module. New symbols use private namespaces to avoid changing common kernels.
No runtime JIT is added. The shared reduction header matches the captured
activation source byte-for-byte (SHA256
071eb3de2cb5688c5b37c322ad45d09cd2ff374f12d435a326f031ff1187d7aa).

The source provenance is the frozen GLM run from November 2026. Numerical and
performance evidence applies to those original diagnostic binaries. The
packaged symbols have not been compiled or run on HCU in this source-only
port. Static comparison of retained kernel bodies, whitespace checks and
project quality checks cannot establish rebuilt-package numerical equivalence.
SGLang's draft documents outstanding installed-package, graph and model checks.
No new capability is advertised for norm+quant or shared-expert quantization.

The existing LightOp #141 Q head stride256 change is already in master and is
not duplicated here. The old prefill branch's norm/shared-quant experiments
are not bulk-submitted.
