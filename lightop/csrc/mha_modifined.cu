#pragma once
#include "layernorm/layernorm.h"
#include <c10/util/MaybeOwned.h>
#include <iostream>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>



namespace at{
namespace native{

inline int get_layernorm_blocksize(int cols) {
    if (cols > 256) return 512;
    if (cols > 128) return 256;
    if (cols >  64) return 128;
    return 64;
}

template <typename scalar_t, int VEC=2>
__global__ void mha_norm_rope_qkv_inplace_kernel(
    scalar_t* __restrict__ qkv,          // [B,S,3,H,D], contiguous
    const scalar_t* __restrict__ gamma_q,// [D] or nullptr
    const scalar_t* __restrict__ beta_q, // [D] or nullptr
    const scalar_t* __restrict__ gamma_k,// [D] or nullptr
    const scalar_t* __restrict__ beta_k, // [D] or nullptr
    const float* __restrict__ cos,       // [S,D], fp32
    const float* __restrict__ sin,       // [S,D], fp32
    int64_t B, int64_t S, int64_t H, int D,
    scalar_t eps)
{
    const int row = blockIdx.x;                 
    const int tid = threadIdx.x;
    const int64_t total_rows = B * S * H;
    if (row >= total_rows) return;

    // 反算 (b,s,h)
    const int64_t h  = row % H;
    const int64_t bs = row / H;
    const int64_t s  = bs % S;
    const int64_t b  = bs / S;

    // [B,S,3,H,D] 的线性偏移（c=0:Q, c=1:K）
    // idx = ((((b*S + s)*3 + c)*H + h)*D + d)
    const size_t base_bs = static_cast<size_t>(b) * static_cast<size_t>(S) + static_cast<size_t>(s);
    const size_t base_3H = base_bs * 3 * static_cast<size_t>(H);
    const size_t q_off   = (base_3H + 0 * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(D);
    const size_t k_off   = (base_3H + 1 * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(D);
    // V 通道是 c=2，kernel内不改

    // 当前 s 的 cos/sin 行
    const float* __restrict__ cos_row = cos + static_cast<size_t>(s) * static_cast<size_t>(D);
    const float* __restrict__ sin_row = sin + static_cast<size_t>(s) * static_cast<size_t>(D);

    using T_ACC = at::acc_type<scalar_t, /*is_cuda=*/true>;
    using WelfordType = at::native::WelfordData<T_ACC, int, T_ACC>;
    using WelfordOp   = at::native::WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
    WelfordOp welford_op;

    __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type sh_q[64];
    __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type sh_k[64];
    __shared__ T_ACC mean_q, mean_k, rstd_q, rstd_k;

    using Vec = at::native::memory::aligned_vector<scalar_t, VEC>;
    const int tcol = D / VEC;

    const Vec* __restrict__ Xq = reinterpret_cast<const Vec*>(qkv + q_off);
    const Vec* __restrict__ Xk = reinterpret_cast<const Vec*>(qkv + k_off);
          Vec* __restrict__ Yq = reinterpret_cast<      Vec*>(qkv + q_off);
          Vec* __restrict__ Yk = reinterpret_cast<      Vec*>(qkv + k_off);

    // 1) Welford 均值方差（Q/K 各自）
    WelfordType vq = {}, vk = {};
    for (int j = tid; j < tcol; j += blockDim.x) {
        const Vec xq = Xq[j];
        const Vec xk = Xk[j];
        const scalar_t* pxq = reinterpret_cast<const scalar_t*>(&xq);
        const scalar_t* pxk = reinterpret_cast<const scalar_t*>(&xk);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            vq = welford_op.reduce(vq, static_cast<T_ACC>(pxq[i]));
            vk = welford_op.reduce(vk, static_cast<T_ACC>(pxk[i]));
        }
    }
    vq = at::native::BlockReduce(vq, welford_op, reinterpret_cast<WelfordType*>(sh_q));
    vk = at::native::BlockReduce(vk, welford_op, reinterpret_cast<WelfordType*>(sh_k));

    if (tid == 0) {
        T_ACC var_q; thrust::tie(var_q, mean_q) = welford_op.project(vq);
        rstd_q = c10::cuda::compat::rsqrt(var_q + static_cast<T_ACC>(eps));
        T_ACC var_k; thrust::tie(var_k, mean_k) = welford_op.project(vk);
        rstd_k = c10::cuda::compat::rsqrt(var_k + static_cast<T_ACC>(eps));
    }
    __syncthreads();

    const T_ACC m_q = mean_q, r_q = rstd_q;
    const T_ACC m_k = mean_k, r_k = rstd_k;

    // 2) LN(+gamma/beta)；3) RoPE 成对旋转（VEC=2），写回 Q/K 行
    for (int j = tid; j < tcol; j += blockDim.x) {
        Vec xq = Xq[j];
        Vec xk = Xk[j];
        Vec yq, yk;

        scalar_t* pyq = reinterpret_cast<scalar_t*>(&yq);
        scalar_t* pyk = reinterpret_cast<scalar_t*>(&yk);
        const scalar_t* pxq = reinterpret_cast<const scalar_t*>(&xq);
        const scalar_t* pxk = reinterpret_cast<const scalar_t*>(&xk);

        // LayerNorm + affine（逐元素）
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = j * VEC + i;
            T_ACC vq_ = (static_cast<T_ACC>(pxq[i]) - m_q) * r_q *
                        (gamma_q ? static_cast<T_ACC>(gamma_q[idx]) : T_ACC(1)) +
                        (beta_q  ? static_cast<T_ACC>(beta_q[idx])  : T_ACC(0));
            T_ACC vk_ = (static_cast<T_ACC>(pxk[i]) - m_k) * r_k *
                        (gamma_k ? static_cast<T_ACC>(gamma_k[idx]) : T_ACC(1)) +
                        (beta_k  ? static_cast<T_ACC>(beta_k[idx])  : T_ACC(0));
            pyq[i] = static_cast<scalar_t>(vq_);
            pyk[i] = static_cast<scalar_t>(vk_);
        }
        // RoPE: 对每个 Vec 内的成对元素 (0,1), (2,3), ...
        #pragma unroll
        for (int i = 0; i < VEC; i += 2) {
            const int idx0 = j * VEC + i;
            const int idx1 = idx0 + 1;
            // safety check (shouldn't trigger because VEC even and pairs inside Vec)
            if (idx1 >= D) break;
            const T_ACC c0 = static_cast<T_ACC>(cos_row[idx0]);
            const T_ACC s0 = static_cast<T_ACC>(sin_row[idx0]);
            const T_ACC c1 = static_cast<T_ACC>(cos_row[idx1]);
            const T_ACC s1 = static_cast<T_ACC>(sin_row[idx1]);

            const T_ACC q0 = static_cast<T_ACC>(pyq[i]);
            const T_ACC q1 = static_cast<T_ACC>(pyq[i+1]);
            const T_ACC k0 = static_cast<T_ACC>(pyk[i]);
            const T_ACC k1 = static_cast<T_ACC>(pyk[i+1]);

            pyq[i+0] = static_cast<scalar_t>(q0 * c0 + (-q1) * s0);
            pyq[i+1] = static_cast<scalar_t>(q1 * c1 + ( q0) * s1);
            pyk[i+0] = static_cast<scalar_t>(k0 * c0 + (-k1) * s0);
            pyk[i+1] = static_cast<scalar_t>(k1 * c1 + ( k0) * s1);
        }
        // vectorized write-back
        Yq[j] = yq;
        Yk[j] = yk;
    } 
}


template <typename scalar_t, int VEC=2>
__global__ void cross_qk_ln_rope_fused_kernel(
    scalar_t* __restrict__ q,               // [B,S1,H,D]
    scalar_t* __restrict__ kv,              // [B,S2,2,H,D] (K=c=0, V=c=1)
    const scalar_t* __restrict__ gamma_q,   // [D] or null
    const scalar_t* __restrict__ beta_q,    // [D] or null
    const scalar_t* __restrict__ gamma_k,   // [D] or null
    const scalar_t* __restrict__ beta_k,    // [D] or null
    const float* __restrict__ cos,          // [S1,D], fp32
    const float* __restrict__ sin,          // [S1,D], fp32
    int64_t B, int64_t S1, int64_t S2, int64_t H, int D,
    scalar_t eps)
{
    const int tid = threadIdx.x;
    const int64_t rows_q = B * S1 * H;
    const int64_t rows_k = B * S2 * H;
    const int64_t total_rows = rows_q + rows_k;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= total_rows) return;

    // 决定 block 处理 Q 行还是 K 行
    const bool is_q = (row < rows_q);
    const int64_t r = is_q ? row : (row - rows_q);

    // 解码索引 (b,s,h)
    int64_t b, s, h;
    if (is_q) {
        h = r % H;
        const int64_t t = r / H;
        s = t % S1;
        b = t / S1;
    } else {
        h = r % H;
        const int64_t t = r / H;
        s = t % S2;     // 这里 S2 的索引用于 K 行
        b = t / S2;
    }

    using T_ACC = at::acc_type<scalar_t, /*is_cuda=*/true>;
    using WelfordType = at::native::WelfordData<T_ACC, int, T_ACC>;
    using WelfordOp   = at::native::WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
    WelfordOp welford_op;

    __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type sh_wf[64];
    __shared__ T_ACC mean_, rstd_;

    using Vec = at::native::memory::aligned_vector<scalar_t, VEC>;
    const int tcol = D / VEC; // 假设 D % VEC == 0

    // 选择 gamma/beta
    const scalar_t* gamma = is_q ? gamma_q : gamma_k;
    const scalar_t* beta  = is_q ? beta_q  : beta_k;

    // 计算行偏移（单位：scalar）
    size_t off = 0;
    if (is_q) {
        // q offset: ((((b*S1 + s)*H + h)*D)
        off = (((static_cast<size_t>(b) * S1 + static_cast<size_t>(s)) * H
                + static_cast<size_t>(h)) * static_cast<size_t>(D));
    } else {
        // kv 的 K 通道（c=0）: ((((b*S2 + s)*2 + 0)*H + h)*D)
        off = (((static_cast<size_t>(b) * S2 + static_cast<size_t>(s)) * 2
                + 0) * static_cast<size_t>(H) + static_cast<size_t>(h)) * static_cast<size_t>(D);
    }

    const Vec* __restrict__ X = reinterpret_cast<const Vec*>((is_q ? (q  + off) : (kv + off)));
          Vec* __restrict__ Y = reinterpret_cast<      Vec*>((is_q ? (q  + off) : (kv + off)));

    // -------- Welford 均值/方差（向量化读取）--------
    WelfordType acc = {};
    for (int j = tid; j < tcol; j += blockDim.x) {
        const Vec vx = X[j];
        const scalar_t* px = reinterpret_cast<const scalar_t*>(&vx);
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            acc = welford_op.reduce(acc, static_cast<T_ACC>(px[i]));
        }
    }
    acc = at::native::BlockReduce(acc, welford_op, reinterpret_cast<WelfordType*>(sh_wf));
    if (tid == 0) {
        T_ACC var_; thrust::tie(var_, mean_) = welford_op.project(acc);
        rstd_ = c10::cuda::compat::rsqrt(var_ + static_cast<T_ACC>(eps));
    }
    __syncthreads();

    const T_ACC m   = mean_;
    const T_ACC rcp = rstd_;

    // cos/sin 行，仅用于 Q
    const float* __restrict__ cos_row = is_q ? (cos + static_cast<size_t>(s) * static_cast<size_t>(D)) : nullptr;
    const float* __restrict__ sin_row = is_q ? (sin + static_cast<size_t>(s) * static_cast<size_t>(D)) : nullptr;

    // VEC in {2,4,8}：对每个 Vec 做 LayerNorm，然后在 Vec 内按 (0,1),(2,3)... 做 RoPE（向量化写回）
    for (int j = tid; j < tcol; j += blockDim.x) {
        const Vec vx = X[j];
        Vec vy;
        scalar_t* py = reinterpret_cast<scalar_t*>(&vy);
        const scalar_t* px = reinterpret_cast<const scalar_t*>(&vx);

        // LayerNorm + affine（Vec 内逐元素）
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const int idx = j * VEC + i;
            T_ACC val = (static_cast<T_ACC>(px[i]) - m) * rcp *
                        (gamma ? static_cast<T_ACC>(gamma[idx]) : T_ACC(1)) +
                        (beta  ? static_cast<T_ACC>(beta[idx])  : T_ACC(0));
            py[i] = static_cast<scalar_t>(val);
        }

        // 如果是 Q 行，在 Vec 内按成对 (2k,2k+1) 做 RoPE
        if (is_q) {
            #pragma unroll
            for (int i = 0; i < VEC; i += 2) {
                const int idx0 = j * VEC + i;
                const int idx1 = idx0 + 1;
                if (idx1 >= D) break; // 安全检查（在假设 D%VEC==0 时一般不会触发）
                const T_ACC c0 = static_cast<T_ACC>(cos_row[idx0]);
                const T_ACC s0 = static_cast<T_ACC>(sin_row[idx0]);
                const T_ACC c1 = static_cast<T_ACC>(cos_row[idx1]);
                const T_ACC s1 = static_cast<T_ACC>(sin_row[idx1]);

                const T_ACC a0 = static_cast<T_ACC>(py[i + 0]);
                const T_ACC a1 = static_cast<T_ACC>(py[i + 1]);

                py[i + 0] = static_cast<scalar_t>(a0 * c0 + (-a1) * s0);
                py[i + 1] = static_cast<scalar_t>(a1 * c1 + ( a0) * s1);
            }
        }
        // 向量化写回
        Y[j] = vy;
    }
}


template <typename scalar_t>
void mha_norm_rope_qkv_inplace_launch(
    at::Tensor qkv,                  // [B,S,3,H,D], contiguous
    const scalar_t* gamma_q,
    const scalar_t* beta_q,
    const scalar_t* gamma_k,
    const scalar_t* beta_k,
    const float* cos,                // [S,D]
    const float* sin,                // [S,D]
    int64_t B, int64_t S, int64_t H, int D,
    scalar_t eps)
{
    TORCH_INTERNAL_ASSERT((D & 1) == 0, "head_dim must be even (RoPE pairing)");
    dim3 grid(B * S * H);
    dim3 block(64);
    // dim3 block(get_layernorm_blocksize(D));
    mha_norm_rope_qkv_inplace_kernel<scalar_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        qkv.data_ptr<scalar_t>(),
        gamma_q, beta_q, gamma_k, beta_k,
        cos, sin,
        B, S, H, D, eps
    );
    AT_CUDA_CHECK(cudaGetLastError());
}

void mha_norm_rope_qkv_inplace_cuda(
    at::Tensor qkv,                             // [B,S,3,H,D]
    c10::optional<at::Tensor> gamma_q,
    c10::optional<at::Tensor> beta_q,
    c10::optional<at::Tensor> gamma_k,
    c10::optional<at::Tensor> beta_k,
    at::Tensor cos,                             // [S,D], float32
    at::Tensor sin,                             // [S,D], float32
    double eps)
{
    TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
    TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
    TORCH_CHECK(qkv.dim() == 5 && qkv.size(2) == 3, "qkv must be [B,S,3,H,D]");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos/sin must be CUDA");
    TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
                "cos/sin must be float32");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous");

    const auto B = qkv.size(0);
    const auto S = qkv.size(1);
    const auto H = qkv.size(3);
    const auto D = qkv.size(4);

    TORCH_CHECK(cos.sizes() == at::IntArrayRef({S, D}), "cos shape must be [S,D]");
    TORCH_CHECK(sin.sizes() == at::IntArrayRef({S, D}), "sin shape must be [S,D]");
    TORCH_CHECK((D & 1) == 0, "head_dim must be even for RoPE");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(qkv.scalar_type(),
        "mha_norm_rope_qkv_inplace_cuda", [&] {
            mha_norm_rope_qkv_inplace_launch<scalar_t>(
                qkv,
                gamma_q.has_value() ? gamma_q->data_ptr<scalar_t>() : nullptr,
                beta_q.has_value()  ? beta_q->data_ptr<scalar_t>()  : nullptr,
                gamma_k.has_value() ? gamma_k->data_ptr<scalar_t>() : nullptr,
                beta_k.has_value()  ? beta_k->data_ptr<scalar_t>()  : nullptr,
                cos.data_ptr<float>(),
                sin.data_ptr<float>(),
                B, S, H, static_cast<int>(D),
                static_cast<scalar_t>(eps)
            );
        });
}

template <typename scalar_t>
void cross_norm_rope_q_kv_inplace_launch(
    at::Tensor q, at::Tensor kv,          // q:[B,S1,H,D], kv:[B,S2,2,H,D]
    const scalar_t* gamma_q, const scalar_t* beta_q,
    const scalar_t* gamma_k, const scalar_t* beta_k,
    const float* cos, const float* sin,   // [S1,D]
    int64_t B, int64_t S1, int64_t S2, int64_t H, int D,
    scalar_t eps)
{
    TORCH_INTERNAL_ASSERT((D & 1) == 0, "head_dim must be even for RoPE pairing (D%2==0).");

    const int64_t rows_q = B * S1 * H;
    const int64_t rows_k = B * S2 * H;
    const int64_t total  = rows_q + rows_k;

    dim3 grid(total);
    dim3 block(64);
    // dim3 block(get_layernorm_blocksize(D));
    cross_qk_ln_rope_fused_kernel<scalar_t, 4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<scalar_t>(),
        kv.data_ptr<scalar_t>(),
        gamma_q, beta_q, gamma_k, beta_k,
        cos, sin,
        B, S1, S2, H, static_cast<int>(D),
        eps
    );
    AT_CUDA_CHECK(cudaGetLastError());
}

// ---------------- Binding（CUDA 入口） ----------------
void cross_norm_rope_q_kv_inplace_cuda(
    at::Tensor q,   // [B,S1,H,D], contiguous
    at::Tensor kv,  // [B,S2,2,H,D], contiguous
    c10::optional<at::Tensor> gamma_q,
    c10::optional<at::Tensor> beta_q,
    c10::optional<at::Tensor> gamma_k,
    c10::optional<at::Tensor> beta_k,
    at::Tensor cos, // [S1,D], float32
    at::Tensor sin, // [S1,D], float32
    double eps)
{
    TORCH_CHECK(q.is_cuda() && kv.is_cuda(), "q/kv must be CUDA.");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous.");
    TORCH_CHECK(q.dim()==4, "q must be [B,S1,H,D].");
    TORCH_CHECK(kv.dim()==5 && kv.size(2)==2, "kv must be [B,S2,2,H,D].");
    TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos/sin must be CUDA.");
    TORCH_CHECK(cos.scalar_type()==at::kFloat && sin.scalar_type()==at::kFloat, "cos/sin must be float32.");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(), "cos/sin must be contiguous.");
    TORCH_CHECK(q.scalar_type()==kv.scalar_type(), "q and kv must have same dtype.");

    const auto B  = q.size(0);
    const auto S1 = q.size(1);
    const auto H  = q.size(2);
    const auto D  = q.size(3);
    const auto B2 = kv.size(0);
    const auto S2 = kv.size(1);
    TORCH_CHECK(B==B2, "q and kv batch dim mismatch.");
    TORCH_CHECK(kv.size(3)==H && kv.size(4)==D, "kv head_dim must match q.");
    TORCH_CHECK(cos.sizes()==at::IntArrayRef({S1, D}), "cos must be [S1,D].");
    TORCH_CHECK(sin.sizes()==at::IntArrayRef({S1, D}), "sin must be [S1,D].");
    TORCH_CHECK((D & 1)==0, "head_dim must be even for RoPE.");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),
        "cross_norm_rope_q_kv_inplace_cuda_single_kernel", [&]{
            cross_norm_rope_q_kv_inplace_launch<scalar_t>(
                q, kv,
                gamma_q.has_value() ? gamma_q->data_ptr<scalar_t>() : nullptr,
                beta_q.has_value()  ? beta_q->data_ptr<scalar_t>()  : nullptr,
                gamma_k.has_value() ? gamma_k->data_ptr<scalar_t>() : nullptr,
                beta_k.has_value()  ? beta_k->data_ptr<scalar_t>()  : nullptr,
                cos.data_ptr<float>(), sin.data_ptr<float>(),
                B, S1, S2, H, static_cast<int>(D),
                static_cast<scalar_t>(eps)
            );
        });
}

} 
}

