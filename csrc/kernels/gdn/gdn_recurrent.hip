// GDN 递推步的手写核（生产版，随 lightop 发布）。
//
// 目标：替换 aiter 的 `fused_sigmoid_gating_delta_rule_update_kernel` 里
// 那个占 20.8% 的 per-token 循环。
//
// 布局是本文件的全部要点。wave64 处理 (slot, hv, 32 个 v)：
//   G = 8   每个 k 组占 8 个 lane
//   KL = 16 每 lane 持 16 个 k
//   VL = 4  每 lane 持 4 个 v
//   lane l: g = l & 7 -> k ∈ [16g, 16g+16)；t1 = l >> 3 -> v ∈ {vb+4*t1+r}
//   状态寄存器 = KL*VL = 64 个 f32（与 Triton 的 64 同量级，125 VGPR / 0 溢出）。
//
// 为什么是 G=8：寄存器数恒等于 KL*VL = (128/G)*(32G/64) = 64，与 G 无关；
// 但每次归约的 shuffle 数 = (v/lane)*log2(G) = (G/2)*log2(G)：
//   G=32 -> 80，G=16 -> 32，G=8 -> 12，G=4 -> 4。
// 原来照 Triton 用 G=32，每 token 170 条 ds_bpermute 串在依赖链上，实测
// 每多一个 token 只加 ~70us（状态搬运量不变），是延迟而不是带宽。
//
// 试过但不可用：DPP 内联汇编。`v_add_f32_dpp` 带 bank_mask 的「shr 写高 bank +
// shl 写低 bank」形式**孤立测试 64/64 正确**，但接在任何一条 DPP 之后结果就错
// （只有 bank2 对），`"+v"` 的读改写形式更是全 0。这套工具链上不可靠，故用
// `__shfl_xor`（宽 8，组内连续 8 lane）。
//
// 构建：setup.py 的独立 lightop._gdn target（aicc + 调优 flags）。

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define BK 128
#define BV 32
// G 是唯一自由参数：一个 k 组占 G 个 lane。
// 其余全部导出，寄存器数恒为 KL*VL = (BK/G)*(BV*G/64) = BK*BV/64 = 64。
#define G 8
#define KL (BK / G)          // 每 lane 的 k 个数
#define VL (BV * G / 64)     // 每 lane 的 v 个数
#define NT1 (64 / G)         // 每 block 的 v 组数

// VL 个 bf16 的打包搬运类型（2 个 -> 4 B，4 个 -> 8 B）
template <int N> struct vpack_of;
template <> struct vpack_of<2> { using type = uint32_t; };
template <> struct vpack_of<4> { using type = uint2; };
template <> struct vpack_of<8> { using type = uint4; };
typedef vpack_of<VL>::type vpack_t;
// 打包类型必须正好覆盖 VL 个 bf16，否则只会搬一半（G=32/VL=16 踩过）。
static_assert(sizeof(vpack_t) == VL * sizeof(hip_bfloat16),
              "vpack_of<VL> 的字节数必须等于 VL*2");

// 组内 G lane 的归约（lane 连续 G 个为一组）
__device__ __forceinline__ float grp_sum(float p) {
#pragma unroll
  for (int m = G / 2; m >= 1; m >>= 1) p = __fadd_rn(p, __shfl_xor(p, m, G));
  return p;
}

// ---- 与 Triton 完全同序的归约树（bitwise 录制树）------------
// aiter 的 tl.sum 在 BK=128 上按 32 个「k 组」（每组 4 个连续 k）归约：
//   组内 d=0..3 顺序累加；32 组的 xor 树是 16,8,4,2,1。
// 本核的 k 归属 k = 4g + 32c 正好把 Triton 线程 tk = g + 8c 的 4 个组放进
// 同一 lane，所以 16/8 两级是 lane 内配对 —— (c0,c2) 与 (c1,c3) —— 只有
// 4/2/1 三级是 8-lane shuffle，shuffle 数与原来的平铺链完全相同。
static_assert(KL == 16, "bitwise 归约树按 KL=16（G=8）展开");
// Triton 的 delta/o 归约里乘积是**独立**的 v_mul_f32，不能被后端折进 v_fma
// （逐位对拍已证实：折了就 1 ULP 漂移，约 3/4 行中招）。
#ifndef BPLUS_NOFOLD_ASM
#define BPLUS_NOFOLD_ASM 1
#endif
__device__ __forceinline__ float mul_nofold(float a, float b) {
#if BPLUS_NOFOLD_ASM
  float d;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(d) : "v"(a), "v"(b));
  return d;
#else
  return __fmul_rn(a, b);
#endif
}
__device__ __forceinline__ float sum4_seq(float x0, float x1, float x2, float x3) {
  return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
}
__device__ __forceinline__ float grp_tree(float g0, float g1, float g2, float g3) {
  float s = __fadd_rn(__fadd_rn(g0, g2), __fadd_rn(g1, g3));
  s = __fadd_rn(s, __shfl_xor(s, 4, 8));
  s = __fadd_rn(s, __shfl_xor(s, 2, 8));
  s = __fadd_rn(s, __shfl_xor(s, 1, 8));
  return s;
}

// l2norm 的归约树：从缓存的 .llir 逐条还原（与 state 的树不同）
//   叶子 q_i = x[2i] + x[2i+1]           —— 同一线程相邻两个 k
//   组内 row_shr 8,4,2,1 的移位链      —— 每 16 个 q 一组，取组末 lane
//   组间 (T0+T1)+(T2+T3)
// 映射到 k = 4g+32c+d：Triton lane = 2g+16c+d/2，所以
//   c ↔ 组，d/2 ↔ 组内 e；移位链在 g 方向是 g-4 / g-2 / g-1，
//   最后一级 row_shr:1 是 e 翻转（同一个 lane 内）。
// 只有 g==7 的 lane 拿到 T_c，再广播给 8-lane 组。
__device__ __forceinline__ float norm_grp(float q0, float q1, int g) {
  float t;
  t = __shfl_up(q0, 4, 8); q0 = g >= 4 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 4, 8); q1 = g >= 4 ? __fadd_rn(q1, t) : q1;
  t = __shfl_up(q0, 2, 8); q0 = g >= 2 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 2, 8); q1 = g >= 2 ? __fadd_rn(q1, t) : q1;
  t = __shfl_up(q0, 1, 8); q0 = g >= 1 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 1, 8); q1 = g >= 1 ? __fadd_rn(q1, t) : q1;
  return __shfl(__fadd_rn(q1, q0), 7, 8);
}

#define NORM_SQ(x, y) __fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y))

// 产线 sglang 核（sglang.kernels.ops.attention.fla）用 1.0/tl.sqrt，编译成
// v_sqrt_f32 + v_div_f32；aiter 的 tl.rsqrt 是 v_rsq_f32 近似，两者差 1 ULP。
__device__ __forceinline__ float norm_inv(float x) {
  return __fdiv_rn(1.0f, sqrtf(x));
}

// BPLUS：只写 store_step[i_n] 那一份快照（-1 = 该槽不写），并让无效槽（h0_idx<0）
//        立即退出，使纠偏那一趟的成本正比于真正要补的槽数。默认 false，
//        原有两个导出符号的行为逐字节不变。
//
//        写哪一行由 store_row[i_n] 决定（空 = 写 store_step[i_n] 那一行）。
//        verify 必须传 (prev_step+1)%D，不能写死第 T-1 行：
//        全接受之后提交指针 prev_step 就是 T-1，下一拍 verify 的 h0 正读第 T-1 行；
//        若这一拍又被截断，纠偏要重放同一行 h0，而 verify 已经把它覆盖成本拍
//        全跑完的状态 → 纠偏从错的 h0 出发，写出垃圾提交态（accept 看不出来）。
//        每槽只有 prev_step 那一行是活的，其余行都是死行，所以 (prev+1)%D 一定安全，
//        且和 store_step 那一行不冲突（store_row 只在全接受时被提交指针采纳）。
// STATE_ONLY：纠偏用。跳过 q 的 l2norm 与 o = h·q 那段，只推 state。
// MULTI: blockIdx.z selects a layer from a meta table and rebinds the
// per-layer pointers below. One launch then covers all GDN layers; the table
// row is 10 pointers + 5 token strides, all int64 (see hcu_sglang/layers/attention/linear/gdn_commit.py).
#define GDN_MULTI_COLS 15

template <bool FMA_OUTER, bool BPLUS = false, bool STATE_ONLY = false,
          bool MULTI = false, bool PRENORM = false, bool GATE = false,
          int WAVES = 1>
// 64 线程/块，最少 2 waves/SIMD。默认的 4 waves 会把 VGPR 压到 64 并溢出 334 个；
// 放到 2 waves 后是 125 VGPR / 0 溢出，与 Triton 的 119 寄存器同量级。
__global__ __launch_bounds__(64 * WAVES, 2) void gdn_recurrent_update_kernel(
    const void* const* layer_ptrs,          // MULTI: [L, GDN_MULTI_COLS]
    const hip_bfloat16* __restrict__ q,     // [tokens, H,  K]
    const hip_bfloat16* __restrict__ k,     // [tokens, H,  K]
    const hip_bfloat16* __restrict__ v,     // [tokens, HV, V]
    const hip_bfloat16* __restrict__ a,     // [tokens, HV]
    const hip_bfloat16* __restrict__ b,     // [tokens, HV]
    const float* __restrict__ A_log,        // [HV]
    const float* __restrict__ dt_bias,      // [HV]
    const int* __restrict__ cu_seqlens,     // [N+1] or null
    float* __restrict__ h0_source,          // [slots, HV, K, V]
    const int* __restrict__ h0_indices,     // [N]
    hip_bfloat16* __restrict__ o,           // [tokens, HV, V]
    // target_verify 的中间态快照：[slots*D, HV, K, V]（per-layer 视图已摊平），
    // 第 step 份在 cache_idx*cache_steps + step。为 null 时不写（普通 decode）。
    float* __restrict__ snap_buf,
    const int* __restrict__ snap_idx,       // [N]
    const int cache_steps,
    const int* __restrict__ store_step,     // [N] B+：只写 step == store_step[i_n]
    // [N] B+：写第几行快照。-1/空 = 写 store_step 那一行本身（纠偏那一趟）。
    const int* __restrict__ store_row,
    const int T, const int HV, const int K, const int V, const int hratio,
    // a 和 b 的行距**不一定相同**：b 常常是某块更宽 buffer 的切片（实测 96），
    // a 是 48。共用一个 stride_b 会让 a[:t] 读到别的 token、越界后读到 NaN。
    int stride_q, int stride_k, int stride_v,
    int stride_a, int stride_b,
    const float scale, const float softplus_beta, const float softplus_threshold,
    const int use_initial_state, const int varlen, const int disable_state_update,
    // PRENORM: l2norm 已由 gdn_verify_norm_pass 算好（fp32, [tokens,H,K] 稠密）；
    // GATE: eg/beta 也已算好（fp32, [tokens,HV] 稠密）。非 PRENORM 传 nullptr。
    const float* __restrict__ qn, const float* __restrict__ kn,
    const float* __restrict__ eg_buf, const float* __restrict__ beta_buf,
    const int stride_qn, const int stride_ab) {
  const int i_v = (WAVES > 1) ? blockIdx.x * WAVES + (threadIdx.x >> 6)
                              : blockIdx.x;
  const int i_nh = blockIdx.y;
  const int lane = threadIdx.x & 63;
  const int g = lane & (G - 1);
  const int t1 = lane / G;

  if (MULTI) {
    const long* m = (const long*)layer_ptrs + (long)blockIdx.z * GDN_MULTI_COLS;
    q = (const hip_bfloat16*)m[0];
    k = (const hip_bfloat16*)m[1];
    v = (const hip_bfloat16*)m[2];
    a = (const hip_bfloat16*)m[3];
    b = (const hip_bfloat16*)m[4];
    A_log = (const float*)m[5];
    dt_bias = (const float*)m[6];
    h0_source = (float*)m[7];
    o = (hip_bfloat16*)m[8];
    snap_buf = (float*)m[9];
    stride_q = (int)m[10];
    stride_k = (int)m[11];
    stride_v = (int)m[12];
    stride_a = (int)m[13];
    stride_b = (int)m[14];
  }

  const int i_n = i_nh / HV;
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / hratio;
  // k 的归属用「步长 32 交错」：k = 4g + 32c（c=0..KL/4-1）。
  // 这样一条 float4 里 8 个 lane 正好覆盖 128 B 连续（4g 步长 16 B），
  // 而每 lane 仍只跨 8 个 lane 归约。用「连续 16 个 k」时同样一条指令
  // 8 个 lane 相隔 64 B，只用到 16 B，状态搬运掉到 800 GB/s。
  const int kbase = 4 * g;
  const int kstep = BK / (KL / 4);   // = 32，块间 k 步长
  const int vbase = i_v * BV + VL * t1;

  int bos, eos;
  if (varlen) {
    bos = cu_seqlens[i_n];
    eos = cu_seqlens[i_n + 1];
  } else {
    bos = i_n * T;
    eos = bos + T;
  }
  // 纠偏只写 step==store_step 那一行，之后的 token 不产生任何写出；终态写回
  // 关闭时截断循环。verify（STATE_ONLY=false）不受影响。
  int eos_eff = eos;
  if (STATE_ONLY && BPLUS && disable_state_update && store_step != nullptr) {
    const int ss = store_step[i_n];
    if (ss >= 0 && ss + bos < eos) eos_eff = bos + ss + 1;
  }

  // state: h[j][r] = h0[k = kbase + j, v = vbase + r]
  float h[KL][VL];
  const int sidx = use_initial_state ? h0_indices[i_n] : -1;
  const int snap_i =
      (snap_buf != nullptr && snap_idx != nullptr) ? snap_idx[i_n] : -1;
  // B+ 纠偏：无效槽（h0_idx < 0）整块退出，让纠偏成本 ∝ 真正要补的槽数。
  if (BPLUS && use_initial_state && sidx < 0) return;
  const long stoff = (long)i_hv * K * V + vbase * K + kbase;
  if (sidx >= 0) {
    const float* p = h0_source + (long)sidx * HV * K * V + stoff;
#pragma unroll
    for (int r = 0; r < VL; ++r)
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        const float4 t = *reinterpret_cast<const float4*>(p + r * K + kstep * c);
        h[4 * c + 0][r] = t.x;
        h[4 * c + 1][r] = t.y;
        h[4 * c + 2][r] = t.z;
        h[4 * c + 3][r] = t.w;
      }
  } else {
#pragma unroll
    for (int j = 0; j < KL; ++j)
#pragma unroll
      for (int r = 0; r < VL; ++r) h[j][r] = 0.0f;
  }

  float bA = 0.0f, dtb = 0.0f, inv_beta = 0.0f;
  if (!GATE) {
    bA = __expf(A_log[i_hv]);
    dtb = dt_bias[i_hv];
    inv_beta = 1.0f / softplus_beta;
  }

  for (int t = bos; t < eos_eff; ++t) {
    // ---- k (pre-l2norm, bf16)，16 个 = 32 B -------------------------------
    float kk[KL];
    if (PRENORM) {
      const float* pk = kn + (long)t * stride_qn + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        const float4 t4 = *reinterpret_cast<const float4*>(pk + kstep * c);
        kk[4 * c + 0] = t4.x;
        kk[4 * c + 1] = t4.y;
        kk[4 * c + 2] = t4.z;
        kk[4 * c + 3] = t4.w;
      }
    } else {
      hip_bfloat16 kb[KL];
      const hip_bfloat16* pk = k + (long)t * stride_k + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c)
        *reinterpret_cast<uint2*>(kb + 4 * c) =
            *reinterpret_cast<const uint2*>(pk + kstep * c);
#pragma unroll
      for (int j = 0; j < KL; ++j) kk[j] = float(kb[j]);
    }

    // ---- gating: x = a + dt_bias; softplus; g = -exp(A_log) * x ----------
    float eg;
    if (GATE) {
      eg = eg_buf[(long)t * stride_ab + i_hv];
    } else {
      const float av = float(a[(long)t * stride_a + i_hv]);
      float x = __fadd_rn(av, dtb);
      const float xs = __fmul_rn(softplus_beta, x);
      if (xs <= softplus_threshold) {
        x = __fmul_rn(inv_beta, __logf(__fadd_rn(1.0f, __expf(xs))));
      }
      const float g_ = __fmul_rn(-bA, x);
      eg = __expf(g_);
    }

    // ---- l2norm(k)：Triton 的相邻配对 + 64-lane 树 -------------------------
    if (!PRENORM) {
      const float t0 = norm_grp(NORM_SQ(kk[0], kk[1]), NORM_SQ(kk[2], kk[3]), g);
      const float t1 = norm_grp(NORM_SQ(kk[4], kk[5]), NORM_SQ(kk[6], kk[7]), g);
      const float t2 = norm_grp(NORM_SQ(kk[8], kk[9]), NORM_SQ(kk[10], kk[11]), g);
      const float t3 = norm_grp(NORM_SQ(kk[12], kk[13]), NORM_SQ(kk[14], kk[15]), g);
      const float ss = __fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3));
      const float invk = norm_inv(__fadd_rn(ss, 1e-6f));
#pragma unroll
      for (int j = 0; j < KL; ++j) kk[j] = __fmul_rn(kk[j], invk);
    }

    // ---- h *= exp(g) ----------------------------------------------------
#pragma unroll
    for (int j = 0; j < KL; ++j)
#pragma unroll
      for (int r = 0; r < VL; ++r) h[j][r] = __fmul_rn(h[j][r], eg);

    // ---- v (bf16 -> fp32)，本 lane 独占 4 个 -------------------------------
    float vv_[VL];
    {
      const hip_bfloat16* pv = v + (long)t * stride_v + i_hv * V + vbase;
      hip_bfloat16 packed[VL];
      *reinterpret_cast<vpack_t*>(packed) = *reinterpret_cast<const vpack_t*>(pv);
#pragma unroll
      for (int r = 0; r < VL; ++r) vv_[r] = float(packed[r]);
    }

    // ---- delta: s[r] = sum_k h[k,r] * k[k] -------------------------------
    float part[VL];
#pragma unroll
    for (int r = 0; r < VL; ++r) {
      part[r] = grp_tree(
          sum4_seq(mul_nofold(h[0][r], kk[0]), mul_nofold(h[1][r], kk[1]),
                   mul_nofold(h[2][r], kk[2]), mul_nofold(h[3][r], kk[3])),
          sum4_seq(mul_nofold(h[4][r], kk[4]), mul_nofold(h[5][r], kk[5]),
                   mul_nofold(h[6][r], kk[6]), mul_nofold(h[7][r], kk[7])),
          sum4_seq(mul_nofold(h[8][r], kk[8]), mul_nofold(h[9][r], kk[9]),
                   mul_nofold(h[10][r], kk[10]), mul_nofold(h[11][r], kk[11])),
          sum4_seq(mul_nofold(h[12][r], kk[12]), mul_nofold(h[13][r], kk[13]),
                   mul_nofold(h[14][r], kk[14]), mul_nofold(h[15][r], kk[15])));
    }

    // ---- v = (v - s) * sigmoid(b) ---------------------------------------
    float beta;
    if (GATE) {
      beta = beta_buf[(long)t * stride_ab + i_hv];
    } else {
      const float bv = float(b[(long)t * stride_b + i_hv]);
      beta = 1.0f / (1.0f + __expf(-bv));
    }
#pragma unroll
    for (int r = 0; r < VL; ++r)
      vv_[r] = __fmul_rn(__fadd_rn(vv_[r], -part[r]), beta);

    // ---- h += k (x) v ----------------------------------------------------
    if (FMA_OUTER) {
#pragma unroll
      for (int j = 0; j < KL; ++j)
#pragma unroll
        for (int r = 0; r < VL; ++r)
          h[j][r] = __fmaf_rn(kk[j], vv_[r], h[j][r]);
    } else {
#pragma unroll
      for (int j = 0; j < KL; ++j)
#pragma unroll
        for (int r = 0; r < VL; ++r)
          h[j][r] = __fadd_rn(h[j][r], __fmul_rn(kk[j], vv_[r]));
    }

    // ---- 中间态快照（target_verify）：每 token 一份完整 state --------------
    // 产线 T=5，即每槽每层写 5 份 3 MB 的 fp32 状态；这一步的带宽不可省，
    // 但可以藏进算的后面（store 无依赖）。
    if (snap_buf != nullptr && snap_i >= 0 &&
        (!BPLUS || (store_step != nullptr && store_step[i_n] == (t - bos)))) {
      // 写哪一行：默认就是「第几步」（纠偏那一趟重放后写提交行）；verify 传
      // store_row=(prev_step+1)%D，故意避开 h0 那一行——见文件头 B+ 注释。
      const int row = (BPLUS && store_row != nullptr) ? store_row[i_n] : (t - bos);
      if (row >= 0) {
      float* ps = snap_buf + ((long)snap_i * cache_steps + row) * HV * K * V
                  + stoff;
#pragma unroll
      for (int r = 0; r < VL; ++r)
#pragma unroll
        for (int c = 0; c < KL / 4; ++c) {
          float4 s4;
          s4.x = h[4 * c + 0][r];
          s4.y = h[4 * c + 1][r];
          s4.z = h[4 * c + 2][r];
          s4.w = h[4 * c + 3][r];
          *reinterpret_cast<float4*>(ps + r * K + kstep * c) = s4;
        }
      }
    }

    if (!STATE_ONLY) {
    // ---- q，同一套归一化 ------------------------------------------------
    float qq[KL];
    if (PRENORM) {
      const float* pq = qn + (long)t * stride_qn + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        const float4 t4 = *reinterpret_cast<const float4*>(pq + kstep * c);
        qq[4 * c + 0] = t4.x;
        qq[4 * c + 1] = t4.y;
        qq[4 * c + 2] = t4.z;
        qq[4 * c + 3] = t4.w;
      }
    } else {
      hip_bfloat16 qb[KL];
      const hip_bfloat16* pq = q + (long)t * stride_q + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c)
        *reinterpret_cast<uint2*>(qb + 4 * c) =
            *reinterpret_cast<const uint2*>(pq + kstep * c);
#pragma unroll
      for (int j = 0; j < KL; ++j) qq[j] = float(qb[j]);
      const float t0 = norm_grp(NORM_SQ(qq[0], qq[1]), NORM_SQ(qq[2], qq[3]), g);
      const float t1 = norm_grp(NORM_SQ(qq[4], qq[5]), NORM_SQ(qq[6], qq[7]), g);
      const float t2 = norm_grp(NORM_SQ(qq[8], qq[9]), NORM_SQ(qq[10], qq[11]), g);
      const float t3 = norm_grp(NORM_SQ(qq[12], qq[13]), NORM_SQ(qq[14], qq[15]), g);
      const float sq = __fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3));
      const float invq = norm_inv(__fadd_rn(sq, 1e-6f));
#pragma unroll
      for (int j = 0; j < KL; ++j) qq[j] = __fmul_rn(__fmul_rn(qq[j], invq), scale);
    }

    // ---- o[r] = sum_k h[k,r] * q[k] --------------------------------------
    float oacc[VL];
#pragma unroll
    for (int r = 0; r < VL; ++r) {
      oacc[r] = grp_tree(
          sum4_seq(mul_nofold(h[0][r], qq[0]), mul_nofold(h[1][r], qq[1]),
                   mul_nofold(h[2][r], qq[2]), mul_nofold(h[3][r], qq[3])),
          sum4_seq(mul_nofold(h[4][r], qq[4]), mul_nofold(h[5][r], qq[5]),
                   mul_nofold(h[6][r], qq[6]), mul_nofold(h[7][r], qq[7])),
          sum4_seq(mul_nofold(h[8][r], qq[8]), mul_nofold(h[9][r], qq[9]),
                   mul_nofold(h[10][r], qq[10]), mul_nofold(h[11][r], qq[11])),
          sum4_seq(mul_nofold(h[12][r], qq[12]), mul_nofold(h[13][r], qq[13]),
                   mul_nofold(h[14][r], qq[14]), mul_nofold(h[15][r], qq[15])));
    }
    {
      hip_bfloat16 packed[VL];
#pragma unroll
      for (int r = 0; r < VL; ++r) packed[r] = hip_bfloat16(oacc[r]);
      hip_bfloat16* po = o + (long)t * HV * V + i_hv * V + vbase;
      *reinterpret_cast<vpack_t*>(po) = *reinterpret_cast<vpack_t*>(packed);
    }
    }  // !STATE_ONLY
  }

  if (!disable_state_update && sidx >= 0) {
    float* p = h0_source + (long)sidx * HV * K * V + stoff;
#pragma unroll
    for (int r = 0; r < VL; ++r)
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        float4 t;
        t.x = h[4 * c + 0][r];
        t.y = h[4 * c + 1][r];
        t.z = h[4 * c + 2][r];
        t.w = h[4 * c + 3][r];
        *reinterpret_cast<float4*>(p + r * K + kstep * c) = t;
      }
  }
}


// PRENORM verify 的 host 入口：q/k 已由 norm pass 归一化成 fp32。
template <bool GATE>
static void launch_prenorm(const void* q, const void* k, const void* v,
                           const void* a, const void* b, const void* A_log,
                           const void* dt_bias, const void* cu_seqlens,
                           void* h0_source, const void* h0_indices, void* o,
                           void* snap_buf, const void* snap_idx, int cache_steps,
                           const void* store_step, const void* store_row,
                           int N, int T, int HV, int K, int V, int hratio,
                           int stride_q, int stride_k, int stride_v,
                           int stride_a, int stride_b, float scale,
                           float softplus_beta, float softplus_threshold,
                           int use_initial_state, int varlen,
                           int disable_state_update, const void* qn,
                           const void* kn, const void* eg_buf,
                           const void* beta_buf, int stride_qn, int stride_ab,
                           hipStream_t stream) {
  dim3 grid(V / BV, N * HV);
  dim3 block(64);
  gdn_recurrent_update_kernel<true, true, false, false, true, GATE, 1>
      <<<grid, block, 0, stream>>>(
      nullptr,
      (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)v,
      (const hip_bfloat16*)a, (const hip_bfloat16*)b, (const float*)A_log,
      (const float*)dt_bias, (const int*)cu_seqlens, (float*)h0_source,
      (const int*)h0_indices, (hip_bfloat16*)o, (float*)snap_buf,
      (const int*)snap_idx, cache_steps, (const int*)store_step,
      (const int*)store_row, T, HV, K, V, hratio, stride_q, stride_k, stride_v,
      stride_a, stride_b, scale, softplus_beta, softplus_threshold,
      use_initial_state, varlen, disable_state_update, (const float*)qn,
      (const float*)kn, (const float*)eg_buf, (const float*)beta_buf,
      stride_qn, stride_ab);
}
// norm pass：每个 wave 一个 (token, head)，算 q/k 的 l2norm（fp32 稠密输出）
// 以及该 head 对应的 hratio 个 hv 的 eg/beta。算术表达式与 verify 核逐条一致。
__global__ __launch_bounds__(256, 2) void gdn_verify_norm_pass_kernel(
    const hip_bfloat16* __restrict__ q, const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ a, const hip_bfloat16* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    float* __restrict__ qn, float* __restrict__ kn,
    float* __restrict__ eg_out, float* __restrict__ beta_out,
    int n_units, int H, int HV, int K, int hratio, int stride_q, int stride_k,
    int stride_a, int stride_b, float softplus_beta,
    float softplus_threshold, float scale) {
  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int unit = blockIdx.x * 4 + wave;
  if (unit >= n_units) return;
  const int h = unit % H;
  const int t = unit / H;
  const int g = lane & (G - 1);
  const int t1 = lane / G;

  hip_bfloat16 kb[KL], qb[KL];
  {
    const hip_bfloat16* pk = k + (long)t * stride_k + h * K + 4 * g;
#pragma unroll
    for (int c = 0; c < KL / 4; ++c)
      *reinterpret_cast<uint2*>(kb + 4 * c) =
          *reinterpret_cast<const uint2*>(pk + (BK / (KL / 4)) * c);
  }
  {
    const hip_bfloat16* pq = q + (long)t * stride_q + h * K + 4 * g;
#pragma unroll
    for (int c = 0; c < KL / 4; ++c)
      *reinterpret_cast<uint2*>(qb + 4 * c) =
          *reinterpret_cast<const uint2*>(pq + (BK / (KL / 4)) * c);
  }
  float kk[KL], qq[KL];
#pragma unroll
  for (int j = 0; j < KL; ++j) {
    kk[j] = float(kb[j]);
    qq[j] = float(qb[j]);
  }
  const float tk0 = norm_grp(NORM_SQ(kk[0], kk[1]), NORM_SQ(kk[2], kk[3]), g);
  const float tk1 = norm_grp(NORM_SQ(kk[4], kk[5]), NORM_SQ(kk[6], kk[7]), g);
  const float tk2 = norm_grp(NORM_SQ(kk[8], kk[9]), NORM_SQ(kk[10], kk[11]), g);
  const float tk3 = norm_grp(NORM_SQ(kk[12], kk[13]), NORM_SQ(kk[14], kk[15]), g);
  const float ss = __fadd_rn(__fadd_rn(tk0, tk1), __fadd_rn(tk2, tk3));
  const float invk = norm_inv(__fadd_rn(ss, 1e-6f));
#pragma unroll
  for (int j = 0; j < KL; ++j) kk[j] = __fmul_rn(kk[j], invk);

  const float tq0 = norm_grp(NORM_SQ(qq[0], qq[1]), NORM_SQ(qq[2], qq[3]), g);
  const float tq1 = norm_grp(NORM_SQ(qq[4], qq[5]), NORM_SQ(qq[6], qq[7]), g);
  const float tq2 = norm_grp(NORM_SQ(qq[8], qq[9]), NORM_SQ(qq[10], qq[11]), g);
  const float tq3 = norm_grp(NORM_SQ(qq[12], qq[13]), NORM_SQ(qq[14], qq[15]), g);
  const float sq = __fadd_rn(__fadd_rn(tq0, tq1), __fadd_rn(tq2, tq3));
  const float invq = norm_inv(__fadd_rn(sq, 1e-6f));
  // 与原核同一顺序：先乘 invq，再乘 scale（调用方传入，与 verify 同一个值）。
#pragma unroll
  for (int j = 0; j < KL; ++j)
    qq[j] = __fmul_rn(__fmul_rn(qq[j], invq), scale);

  if (t1 == 0) {
    const int kbase = 4 * g;
    const int kstep = BK / (KL / 4);
    float* pkn = kn + (long)t * (long)H * K + h * K + kbase;
    float* pqn = qn + (long)t * (long)H * K + h * K + kbase;
#pragma unroll
    for (int c = 0; c < KL / 4; ++c) {
      float4 k4;
      k4.x = kk[4 * c + 0]; k4.y = kk[4 * c + 1];
      k4.z = kk[4 * c + 2]; k4.w = kk[4 * c + 3];
      *reinterpret_cast<float4*>(pkn + kstep * c) = k4;
      float4 q4;
      q4.x = qq[4 * c + 0]; q4.y = qq[4 * c + 1];
      q4.z = qq[4 * c + 2]; q4.w = qq[4 * c + 3];
      *reinterpret_cast<float4*>(pqn + kstep * c) = q4;
    }
    // 每个 k-head 对应 hratio 个 hv：lane 0..hratio-1 各算一个。
    if (lane < hratio) {
      const int hv = h * hratio + lane;
      const float bA = __expf(A_log[hv]);
      const float dtb = dt_bias[hv];
      const float inv_beta = 1.0f / softplus_beta;
      const float av = float(a[(long)t * stride_a + hv]);
      float x = __fadd_rn(av, dtb);
      const float xs = __fmul_rn(softplus_beta, x);
      if (xs <= softplus_threshold) {
        x = __fmul_rn(inv_beta, __logf(__fadd_rn(1.0f, __expf(xs))));
      }
      const float g_ = __fmul_rn(-bA, x);
      eg_out[(long)t * HV + hv] = __expf(g_);
      const float bv = float(b[(long)t * stride_b + hv]);
      beta_out[(long)t * HV + hv] = 1.0f / (1.0f + __expf(-bv));
    }
  }
}
// host 入口（ctypes 直接调）
template <bool BPLUS, bool STATE_ONLY>
static void launch(const void* q, const void* k, const void* v, const void* a,
                   const void* b, const void* A_log, const void* dt_bias,
                   const void* cu_seqlens, void* h0_source, const void* h0_indices,
                   void* o, void* snap_buf, const void* snap_idx,
                   int cache_steps, const void* store_step,
                   const void* store_row,
                   int N, int T, int HV, int K, int V, int hratio,
                   int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
                   float scale, float softplus_beta, float softplus_threshold,
                   int use_initial_state, int varlen, int disable_state_update,
                   int fma_outer, hipStream_t stream) {
  dim3 grid(V / BV, N * HV);
  dim3 block(64);
  if (fma_outer) {
    gdn_recurrent_update_kernel<true, BPLUS, STATE_ONLY><<<grid, block, 0, stream>>>(
        nullptr,
        (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)v,
        (const hip_bfloat16*)a, (const hip_bfloat16*)b, (const float*)A_log,
        (const float*)dt_bias, (const int*)cu_seqlens, (float*)h0_source,
        (const int*)h0_indices, (hip_bfloat16*)o, (float*)snap_buf,
        (const int*)snap_idx, cache_steps, (const int*)store_step,
        (const int*)store_row,
        T, HV, K, V, hratio,
        stride_q, stride_k, stride_v, stride_a, stride_b, scale, softplus_beta,
        softplus_threshold, use_initial_state, varlen, disable_state_update,
        nullptr, nullptr, nullptr, nullptr, 0, 0);
  } else {
    gdn_recurrent_update_kernel<false, BPLUS, STATE_ONLY><<<grid, block, 0, stream>>>(
        nullptr,
        (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)v,
        (const hip_bfloat16*)a, (const hip_bfloat16*)b, (const float*)A_log,
        (const float*)dt_bias, (const int*)cu_seqlens, (float*)h0_source,
        (const int*)h0_indices, (hip_bfloat16*)o, (float*)snap_buf,
        (const int*)snap_idx, cache_steps, (const int*)store_step,
        (const int*)store_row,
        T, HV, K, V, hratio,
        stride_q, stride_k, stride_v, stride_a, stride_b, scale, softplus_beta,
        softplus_threshold, use_initial_state, varlen, disable_state_update,
        nullptr, nullptr, nullptr, nullptr, 0, 0);
  }
}

// MULTI: one launch for L layers, layer chosen by blockIdx.z. Per-layer
// pointers and token strides come from the int64 meta table.
template <bool STATE_ONLY>
static void launch_multi(const void* layer_ptrs, const void* h0_indices,
                         const void* snap_idx, int cache_steps,
                         const void* store_step, int L,
                         int N, int T, int HV, int K, int V, int hratio,
                         float scale, float softplus_beta,
                         float softplus_threshold, int use_initial_state,
                         int varlen, int disable_state_update,
                         hipStream_t stream) {
  // 4 个 v-block 合进一个 256 线程 CTA（4 waves）：CTA 数降到 1/4，纠偏的
  // 派发地板从 1.37 ms 降到 0.39 ms。每 wave 的 lane 映射与原核一致。
  // V/BV 不能被 4 整除的模型退回单 wave，避免 grid.x 取整后漏算。
  const int vb = V / BV;
  const bool merge4 = (vb % 4) == 0;
  const int waves = merge4 ? 4 : 1;
  dim3 grid(vb / waves, N * HV, L);
  dim3 block(64 * waves);
  if (merge4) {
    gdn_recurrent_update_kernel<true, true, STATE_ONLY, true, false, false, 4>
        <<<grid, block, 0, stream>>>(
        (const void* const*)layer_ptrs,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, (const int*)h0_indices, nullptr, nullptr,
        (const int*)snap_idx,
        cache_steps, (const int*)store_step, nullptr,
        T, HV, K, V, hratio, 0, 0, 0, 0, 0,
        scale, softplus_beta, softplus_threshold,
        use_initial_state, varlen, disable_state_update,
        nullptr, nullptr, nullptr, nullptr, 0, 0);
  } else {
    gdn_recurrent_update_kernel<true, true, STATE_ONLY, true>
        <<<grid, block, 0, stream>>>(
        (const void* const*)layer_ptrs,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, (const int*)h0_indices, nullptr, nullptr,
        (const int*)snap_idx,
        cache_steps, (const int*)store_step, nullptr,
        T, HV, K, V, hratio, 0, 0, 0, 0, 0,
        scale, softplus_beta, softplus_threshold,
        use_initial_state, varlen, disable_state_update,
        nullptr, nullptr, nullptr, nullptr, 0, 0);
  }
}

extern "C" void gdn_recurrent_update(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<false, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, nullptr, nullptr,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

// ---- B+：只写 store_step 指定那一份快照 -----------------------------------
// store_step[N] 为 device 指针：第 i 槽只写 step == store_step[i]，-1 = 不写。
// h0_idx[i] < 0 的槽整块退出（纠偏时用它跳过全接受的槽）。
extern "C" void gdn_recurrent_update_bplus(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, const void* store_step,
    const void* store_row,
    int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<true, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, store_step, store_row,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

// 纠偏专用：同样的 store_step 语义，但不算 q 的 l2norm 和 o = h·q。
extern "C" void gdn_recurrent_update_bplus_state(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, const void* store_step,
    const void* store_row,
    int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<true, true>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, store_step, store_row,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

extern "C" void gdn_recurrent_update_nofma(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<false, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, nullptr, nullptr,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 0, (hipStream_t)stream);
}

// B+ 纠偏的多层入口：一次 launch 覆盖全部 GDN 层，层号 = blockIdx.z。
extern "C" void gdn_recurrent_update_bplus_state_multi(
    const void* layer_ptrs, const void* h0_indices, const void* snap_idx,
    int cache_steps, const void* store_step, int L,
    int N, int T, int HV, int K, int V, int hratio,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch_multi<true>(layer_ptrs, h0_indices, snap_idx, cache_steps, store_step,
                     L, N, T, HV, K, V, hratio, scale, softplus_beta,
                     softplus_threshold, use_initial_state, varlen,
                     disable_state_update, (hipStream_t)stream);
}
// norm pass 入口：q/k/eg/beta 一次算完（48 层每层一次 launch）。
extern "C" void gdn_verify_norm_pass(
    const void* q, const void* k, const void* a, const void* b,
    const void* A_log, const void* dt_bias, void* qn, void* kn, void* eg_out,
    void* beta_out, int n_units, int H, int HV, int K, int hratio, int stride_q,
    int stride_k, int stride_a, int stride_b, float softplus_beta,
    float softplus_threshold, float scale, void* stream) {
  const int blocks = (n_units + 3) / 4;
  gdn_verify_norm_pass_kernel<<<blocks, 256, 0, (hipStream_t)stream>>>(
      (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)a,
      (const hip_bfloat16*)b, (const float*)A_log, (const float*)dt_bias,
      (float*)qn, (float*)kn, (float*)eg_out, (float*)beta_out, n_units, H, HV,
      K, hratio, stride_q, stride_k, stride_a, stride_b, softplus_beta,
      softplus_threshold, scale);
}

// PRENORM verify：q/k 用 norm pass 的 fp32 结果；gate=1 时 eg/beta 也直接 load。
extern "C" void gdn_recurrent_update_bplus_prenorm(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, const void* store_step,
    const void* store_row, int N, int T, int HV, int K, int V, int hratio,
    int stride_q, int stride_k, int stride_v, int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, const void* qn,
    const void* kn, const void* eg_buf, const void* beta_buf, int stride_qn,
    int stride_ab, int gate, void* stream) {
  if (gate) {
    launch_prenorm<true>(
        q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source, h0_indices, o,
        snap_buf, snap_idx, cache_steps, store_step, store_row, N, T, HV, K, V,
        hratio, stride_q, stride_k, stride_v, stride_a, stride_b, scale,
        softplus_beta, softplus_threshold, use_initial_state, varlen,
        disable_state_update, qn, kn, eg_buf, beta_buf, stride_qn, stride_ab,
        (hipStream_t)stream);
  } else {
    launch_prenorm<false>(
        q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source, h0_indices, o,
        snap_buf, snap_idx, cache_steps, store_step, store_row, N, T, HV, K, V,
        hratio, stride_q, stride_k, stride_v, stride_a, stride_b, scale,
        softplus_beta, softplus_threshold, use_initial_state, varlen,
        disable_state_update, qn, kn, eg_buf, beta_buf, stride_qn, stride_ab,
        (hipStream_t)stream);
  }
}
// causal conv1d update (one GDN layer per launch), bitwise vs the Triton
// `_causal_conv1d_update_kernel` for KERNEL_WIDTH=4 / bf16.
//
// The Triton kernel's arithmetic, read off its AMDGCN dump:
//   p = cvt_f32_bf16(cvt_bf16_f32(f32(a) * f32(b)))     (product rounded to bf16)
//   acc = fadd(acc, p)                                  (fp32, one tap per step)
//   y = acc / (1 + exp2(-acc * log2e))                  (hw v_exp_f32, IEEE div)
// Tap order differs per path and is part of the contract:
//   no tree : w0*state0 + w1*state1 + w2*state2 + w3*x[t]
//   tree    : w3*x[t] + w2*parent + w1*parent^2 + w0*parent^3
// The conv window columns are raw bf16 moves, never recomputed.
//
// Strides are runtime ints (the Triton wrapper passes them as constexpr), so a
// non-contiguous window / conv_state / x keeps working.
#ifndef GDN_CONV_MAX_T
#define GDN_CONV_MAX_T 16
#endif
// CTA width for both conv kernels; variants are timed against Triton.
#ifndef GDN_CONV_BLOCK
#define GDN_CONV_BLOCK 256
#endif

// 2^v as the Triton kernel gets it: the AMDGPU backend lowers its
// llvm.exp2.f32 to the hardware instruction, not to __ocml_exp2_f32.
__device__ __forceinline__ float gdn_conv_exp2_hw(float v) {
  float r;
  asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(v));
  return r;
}

// One tap: f32 multiply rounded to bf16, like a bf16*bf16 in Triton.
__device__ __forceinline__ float gdn_conv_tap(float a, float b) {
  hip_bfloat16 r(__fmul_rn(a, b));
  return (float)r;
}

__device__ __forceinline__ float gdn_conv_silu(float acc) {
  // fsub 0.0,acc is not the same as -acc at zero; Triton emits the former.
  float neg = __fsub_rn(0.0f, acc);
  float e = gdn_conv_exp2_hw(__fmul_rn(neg, 1.4426950408889634f));
  return acc / __fadd_rn(e, 1.0f);
}

// VEC features per thread; the host only picks VEC>1 when dim % VEC == 0.
// IDT is the element type of retrieve_next_token / _sibling / parent out.
template <int VEC, typename IDT>
__global__ __launch_bounds__(GDN_CONV_BLOCK, 2) void gdn_causal_conv1d_update_kernel(
    const hip_bfloat16* __restrict__ x, const hip_bfloat16* __restrict__ w,
    const hip_bfloat16* __restrict__ bias, hip_bfloat16* __restrict__ conv_state,
    const int* __restrict__ conv_idx, hip_bfloat16* __restrict__ window,
    const int* __restrict__ inter_idx, const IDT* __restrict__ nxt,
    const IDT* __restrict__ sib, IDT* __restrict__ par_out,
    hip_bfloat16* __restrict__ o,
    int dim, int seqlen, int state_len, int num_cache_lines,
    int s_x_seq, int s_x_dim, int s_x_tok,
    int s_w_dim, int s_w_width,
    int s_cs_seq, int s_cs_dim, int s_cs_tok, int s_ci,
    int s_wseq, int s_wstep, int s_wdim, int s_win, int s_ii,
    int s_nseq, int s_ntok, int s_sseq, int s_stok,
    int s_pseq, int s_ptok,
    int s_o_seq, int s_o_dim, int s_o_tok,
    int pad_slot_id, int has_bias, int silu, int cont, int save_inter,
    int tree, int use_pad) {
  const int seq = blockIdx.x;
  const int f0 = (blockIdx.y * GDN_CONV_BLOCK + threadIdx.x) * VEC;
  if (f0 >= dim) return;

  const long long coord = cont ? (long long)conv_idx[seq * s_ci] : (long long)seq;
  if (use_pad && coord == (long long)pad_slot_id) return;

  float wv[VEC][4];
#pragma unroll
  for (int v = 0; v < VEC; ++v)
#pragma unroll
    for (int k = 0; k < 4; ++k)
      wv[v][k] = (float)w[(long long)(f0 + v) * s_w_dim + k * s_w_width];
  float bv[VEC];
  if (has_bias) {
#pragma unroll
    for (int v = 0; v < VEC; ++v) bv[v] = (float)bias[f0 + v];
  }

  const hip_bfloat16* xs = x + (long long)seq * s_x_seq + (long long)f0 * s_x_dim;
  hip_bfloat16* ob = o + (long long)seq * s_o_seq + (long long)f0 * s_o_dim;
  const long long cs_off = coord * s_cs_seq + (long long)f0 * s_cs_dim;
  const hip_bfloat16* cs = conv_state + cs_off;
  hip_bfloat16* cst = conv_state + cs_off;

  // STEP 1: the window columns the conv seeds from (oldest first).
  hip_bfloat16 col0[VEC], col1[VEC], col2[VEC];
#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    col0[v] = cs[0 * s_cs_tok + v * s_cs_dim];
    col1[v] = cs[1 * s_cs_tok + v * s_cs_dim];
    col2[v] = cs[2 * s_cs_tok + v * s_cs_dim];
  }

  // STEP 2: same sliding update as Triton (shift left by seqlen, last tokens
  // come from x; the source index is always the higher one, so in-place is safe).
  {
    const int val = state_len - seqlen;
    const bool slot_ok = coord < num_cache_lines;
    for (int i = 0; i < state_len; ++i) {
      hip_bfloat16* dst = cst + i * s_cs_tok;
      if (slot_ok && (i + seqlen) < state_len) {
        const hip_bfloat16* src = cs + (i + seqlen) * s_cs_tok;
#pragma unroll
        for (int v = 0; v < VEC; ++v) dst[v * s_cs_dim] = src[v * s_cs_dim];
      } else {
        const int xi = i - val;
        const bool in = (xi >= 0) && (xi < seqlen);
#pragma unroll
        for (int v = 0; v < VEC; ++v)
          dst[v * s_cs_dim] =
              in ? xs[(long long)xi * s_x_tok + v * s_x_dim] : hip_bfloat16(0.0f);
      }
    }
  }

  hip_bfloat16* wb = nullptr;
  if (save_inter) {
    const long long row = (long long)inter_idx[seq * s_ii];
    wb = window + row * s_wseq + (long long)f0 * s_wdim;
  }

  if (!tree) {
    for (int t = 0; t < seqlen; ++t) {
      hip_bfloat16 xt[VEC];
#pragma unroll
      for (int v = 0; v < VEC; ++v) xt[v] = xs[(long long)t * s_x_tok + v * s_x_dim];
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        float acc = __fadd_rn(gdn_conv_tap((float)col0[v], wv[v][0]),
                              has_bias ? bv[v] : 0.0f);
        acc = __fadd_rn(acc, gdn_conv_tap((float)col1[v], wv[v][1]));
        acc = __fadd_rn(acc, gdn_conv_tap((float)col2[v], wv[v][2]));
        acc = __fadd_rn(acc, gdn_conv_tap((float)xt[v], wv[v][3]));
        float y = silu ? gdn_conv_silu(acc) : acc;
        ob[(long long)t * s_o_tok + v * s_o_dim] = hip_bfloat16(y);
      }
      if (save_inter) {
        hip_bfloat16* wt = wb + (long long)t * s_wstep;
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          wt[0 * s_win + v * s_wdim] = col1[v];
          wt[1 * s_win + v * s_wdim] = col2[v];
          wt[2 * s_win + v * s_wdim] = xt[v];
        }
      }
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        col0[v] = col1[v];
        col1[v] = col2[v];
        col2[v] = xt[v];
      }
    }
  } else {
    // Same parent map the Triton kernel builds from next/sibling on the fly;
    // each node is assigned once, so the final map equals the incremental one
    // on every entry a chain walk can reach.
    int par[GDN_CONV_MAX_T];
    for (int i = 0; i < seqlen; ++i) par[i] = 0;
    for (int i = 0; i < seqlen; ++i) {
      const IDT nx = nxt[(long long)seq * s_nseq + i * s_ntok];
      if (nx != (IDT)-1 && (int)nx < GDN_CONV_MAX_T) par[(int)nx] = i;
      const IDT sb = sib[(long long)seq * s_sseq + i * s_stok];
      if (sb != (IDT)-1 && (int)sb < GDN_CONV_MAX_T) par[(int)sb] = par[i];
    }

    for (int t = 0; t < seqlen; ++t) {
      hip_bfloat16* wt = save_inter ? wb + (long long)t * s_wstep : nullptr;
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        hip_bfloat16 mx = xs[(long long)t * s_x_tok + v * s_x_dim];
        int idx = t;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (save_inter && (2 - j) >= 0)
            wt[(2 - j) * s_win + v * s_wdim] = mx;
          const float p = gdn_conv_tap((float)mx, wv[v][3 - j]);
          acc = (j == 0) ? __fadd_rn(p, has_bias ? bv[v] : 0.0f)
                         : __fadd_rn(acc, p);
          if (idx > 0) {
            idx = par[idx];
            mx = xs[(long long)idx * s_x_tok + v * s_x_dim];
          } else {
            mx = (idx == 0) ? col2[v] : ((idx == -1) ? col1[v] : col0[v]);
            idx -= 1;
          }
        }
        float y = silu ? gdn_conv_silu(acc) : acc;
        ob[(long long)t * s_o_tok + v * s_o_dim] = hip_bfloat16(y);
      }
    }

    if (par_out) {
      for (int i = 0; i < seqlen; ++i)
        par_out[(long long)seq * s_pseq + i * s_ptok] = (IDT)par[i];
    }
  }
}
// Fast path for the production layout: one feature per thread (the shape that
// measured fastest; every wider per-thread feature block lost parallelism),
// with x/o/weight axes stride 1.
//
// The conv_state row is 3 columns per feature, so a per-feature access lands on
// a 6B-strided pattern and inflates every warp access into 3 L2 sectors. The
// CTA therefore stages that row through shared memory as straight 16B vectors
// (load once at entry, store once at exit). The window is written per feature
// straight from the history registers: staging it through shared memory was
// measured 14us/layer slower, as the L2 already merges the partial-sector
// stores before they reach DRAM.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK, 2) void gdn_conv_update_fast_kernel(
    const hip_bfloat16* __restrict__ x, const hip_bfloat16* __restrict__ w,
    const hip_bfloat16* __restrict__ bias, hip_bfloat16* __restrict__ conv_state,
    const int* __restrict__ conv_idx, hip_bfloat16* __restrict__ window,
    const int* __restrict__ inter_idx, hip_bfloat16* __restrict__ o,
    int seqlen, int num_cache_lines, int s_x_seq, int s_x_tok, int s_cs_seq,
    int s_wseq, int s_wstep, int s_o_seq, int s_o_tok, int s_ci, int s_ii,
    int pad_slot_id, int has_bias, int silu, int save_inter) {
  const int seq = blockIdx.x;
  const int fb = blockIdx.y * BLOCK;
  const int tid = (int)threadIdx.x;
  const long long coord = (long long)conv_idx[seq * s_ci];
  if (coord == (long long)pad_slot_id) return;  // uniform across the CTA

  __shared__ hip_bfloat16 s_st[3 * BLOCK];  // (feature, col), col fastest
  const int nch = 3 * BLOCK / 8;            // 16B chunks in one such row
  const hip_bfloat16* xs = x + (long long)seq * s_x_seq + fb;
  hip_bfloat16* ob = o + (long long)seq * s_o_seq + fb;
  const long long cs_off = coord * s_cs_seq + (long long)fb * 3;
  const hip_bfloat16* cs = conv_state + cs_off;
  hip_bfloat16* cst = conv_state + cs_off;

  if (tid < nch) ((uint4*)s_st)[tid] = ((const uint4*)cs)[tid];
  __syncthreads();
  // (h0,h1,h2) = (x[t-3], x[t-2], x[t-1]); token 0 seeds it from conv_state.
  hip_bfloat16 h0 = s_st[tid * 3 + 0];
  hip_bfloat16 h1 = s_st[tid * 3 + 1];
  hip_bfloat16 h2 = s_st[tid * 3 + 2];

  union {
    uint2 raw;
    hip_bfloat16 v[4];
  } wu;  // one 8B load per thread; w is (dim, 4) contiguous
  wu.raw = *(const uint2*)(w + (long long)(fb + tid) * 4);
  const float w0 = (float)wu.v[0], w1 = (float)wu.v[1];
  const float w2 = (float)wu.v[2], w3 = (float)wu.v[3];
  const float bv = has_bias ? (float)bias[fb + tid] : 0.0f;
  const long long wrow = save_inter ? (long long)inter_idx[seq * s_ii] : 0;

  for (int t = 0; t < seqlen; ++t) {
    const hip_bfloat16 xt = xs[(long long)t * s_x_tok + tid];
    float acc = __fadd_rn(gdn_conv_tap((float)h0, w0), bv);
    acc = __fadd_rn(acc, gdn_conv_tap((float)h1, w1));
    acc = __fadd_rn(acc, gdn_conv_tap((float)h2, w2));
    acc = __fadd_rn(acc, gdn_conv_tap((float)xt, w3));
    ob[(long long)t * s_o_tok + tid] =
        hip_bfloat16(silu ? gdn_conv_silu(acc) : acc);
    if (save_inter) {
      // window[t] = [x(t-2), x(t-1), x(t)] = [h1, h2, xt].
      hip_bfloat16* sw = window + wrow * s_wseq + (long long)t * s_wstep +
                         (long long)fb * 3 + tid * 3;
      sw[0] = h1;
      sw[1] = h2;
      sw[2] = xt;
    }
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }

  // conv_state update: the history after the last token is the new state.
  s_st[tid * 3 + 0] = h0;
  s_st[tid * 3 + 1] = h1;
  s_st[tid * 3 + 2] = h2;
  __syncthreads();
  if (tid < nch) ((uint4*)cst)[tid] = ((const uint4*)s_st)[tid];
}

template <typename IDT>
static void launch_conv(const void* x, const void* w, const void* bias,
                        void* conv_state, const void* conv_idx, void* window,
                        const void* inter_idx, const void* nxt, const void* sib,
                        void* par_out, void* o, int batch, int dim, int seqlen,
                        int state_len, int num_cache_lines, int s_x_seq,
                        int s_x_dim, int s_x_tok, int s_w_dim, int s_w_width,
                        int s_cs_seq, int s_cs_dim, int s_cs_tok, int s_ci,
                        int s_wseq, int s_wstep, int s_wdim, int s_win,
                        int s_ii, int s_nseq, int s_ntok, int s_sseq, int s_stok,
                        int s_pseq, int s_ptok, int s_o_seq, int s_o_dim,
                        int s_o_tok, int pad_slot_id, int has_bias, int silu,
                        int cont, int save_inter, int tree, int use_pad,
                        hipStream_t stream) {
#define GDN_CONV_ARGS                                                          \
  (const hip_bfloat16*)x, (const hip_bfloat16*)w, (const hip_bfloat16*)bias,   \
      (hip_bfloat16*)conv_state, (const int*)conv_idx, (hip_bfloat16*)window,  \
      (const int*)inter_idx, (const IDT*)nxt, (const IDT*)sib, (IDT*)par_out,  \
      (hip_bfloat16*)o, dim, seqlen, state_len, num_cache_lines, s_x_seq,      \
      s_x_dim, s_x_tok, s_w_dim, s_w_width, s_cs_seq, s_cs_dim, s_cs_tok,      \
      s_ci, s_wseq, s_wstep, s_wdim, s_win, s_ii, s_nseq, s_ntok, s_sseq,      \
      s_stok, s_pseq, s_ptok, s_o_seq, s_o_dim, s_o_tok, pad_slot_id,          \
      has_bias, silu, cont, save_inter, tree, use_pad
  const bool aligned = (s_x_tok % 8) == 0 && (s_x_seq % 8) == 0 &&
                       (s_o_tok % 8) == 0 && (s_o_seq % 8) == 0 &&
                       (s_cs_seq % 8) == 0;
  const bool fast = !tree && use_pad && cont && state_len == 3 &&
                    s_x_dim == 1 && s_o_dim == 1 && s_cs_dim == 3 &&
                    s_cs_tok == 1 && s_w_dim == 4 && s_w_width == 1 &&
                    (!save_inter || (s_wdim == 3 && s_win == 1)) &&
                    aligned && dim % GDN_CONV_BLOCK == 0;
  if (fast) {
    dim3 grid(batch, dim / GDN_CONV_BLOCK);
    gdn_conv_update_fast_kernel<GDN_CONV_BLOCK><<<grid, GDN_CONV_BLOCK, 0, stream>>>(
        (const hip_bfloat16*)x, (const hip_bfloat16*)w,
        (const hip_bfloat16*)bias, (hip_bfloat16*)conv_state,
        (const int*)conv_idx, (hip_bfloat16*)window, (const int*)inter_idx,
        (hip_bfloat16*)o, seqlen, num_cache_lines, s_x_seq, s_x_tok, s_cs_seq,
        s_wseq, s_wstep, s_o_seq, s_o_tok, s_ci, s_ii, pad_slot_id, has_bias,
        silu, save_inter);
  } else {
    dim3 grid(batch, (dim + GDN_CONV_BLOCK - 1) / GDN_CONV_BLOCK);
    gdn_causal_conv1d_update_kernel<1, IDT>
        <<<grid, GDN_CONV_BLOCK, 0, stream>>>(GDN_CONV_ARGS);
  }
#undef GDN_CONV_ARGS
}

extern "C" void gdn_causal_conv1d_update(
    const void* x, const void* w, const void* bias, void* conv_state,
    const void* conv_idx, void* window, const void* inter_idx, const void* nxt,
    const void* sib, void* par_out, void* o, int batch, int dim, int seqlen,
    int state_len, int num_cache_lines, int s_x_seq, int s_x_dim, int s_x_tok,
    int s_w_dim, int s_w_width, int s_cs_seq, int s_cs_dim, int s_cs_tok,
    int s_ci, int s_wseq, int s_wstep, int s_wdim, int s_win, int s_ii,
    int s_nseq, int s_ntok, int s_sseq, int s_stok, int s_pseq, int s_ptok,
    int s_o_seq, int s_o_dim, int s_o_tok, int pad_slot_id, int has_bias,
    int silu, int cont, int save_inter, int tree, int use_pad, int idx64,
    void* stream) {
  if (idx64)
    launch_conv<long long>(
        x, w, bias, conv_state, conv_idx, window, inter_idx, nxt, sib, par_out,
        o, batch, dim, seqlen, state_len, num_cache_lines, s_x_seq, s_x_dim,
        s_x_tok, s_w_dim, s_w_width, s_cs_seq, s_cs_dim, s_cs_tok, s_ci,
        s_wseq, s_wstep, s_wdim, s_win, s_ii, s_nseq, s_ntok, s_sseq, s_stok,
        s_pseq, s_ptok, s_o_seq, s_o_dim, s_o_tok, pad_slot_id, has_bias, silu,
        cont, save_inter, tree, use_pad, (hipStream_t)stream);
  else
    launch_conv<int>(
        x, w, bias, conv_state, conv_idx, window, inter_idx, nxt, sib, par_out,
        o, batch, dim, seqlen, state_len, num_cache_lines, s_x_seq, s_x_dim,
        s_x_tok, s_w_dim, s_w_width, s_cs_seq, s_cs_dim, s_cs_tok, s_ci,
        s_wseq, s_wstep, s_wdim, s_win, s_ii, s_nseq, s_ntok, s_sseq, s_stok,
        s_pseq, s_ptok, s_o_seq, s_o_dim, s_o_tok, pad_slot_id, has_bias, silu,
        cont, save_inter, tree, use_pad, (hipStream_t)stream);
}
