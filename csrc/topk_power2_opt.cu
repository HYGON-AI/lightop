// Ported from custom_ops/custom_topk_hip.hip for lightop P1.
// HIP runtime APIs rewritten to CUDA-style for lightop hipify build.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <type_traits>
#include <vector>

template <typename T> struct CmpType { using type = T; };
template <> struct CmpType<at::BFloat16> { using type = float; };

template <typename T> struct VecWidth { static constexpr int value = 1; };
template <> struct VecWidth<float> { static constexpr int value = 4; };
template <> struct VecWidth<at::Half> { static constexpr int value = 8; };
template <> struct VecWidth<at::BFloat16> { static constexpr int value = 8; };

template <typename scalar_t, int VW>
__device__ __forceinline__ void load_vec(const scalar_t* ptr, scalar_t* out) {
    #pragma unroll
    for (int i = 0; i < VW; ++i) out[i] = ptr[i];
}

template <typename scalar_t, int K>
__device__ __forceinline__ void bitonic_sort(scalar_t* vals, int* idxs) {
    #pragma unroll
    for (int size = 2; size <= K; size <<= 1) {
        #pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                int partner = i ^ stride;
                if (partner > i) {
                    bool ascending = ((i & size) == 0);
                    bool do_swap = ascending ? (vals[i] < vals[partner]) : (vals[i] > vals[partner]);
                    if (do_swap) {
                        scalar_t tv = vals[i]; vals[i] = vals[partner]; vals[partner] = tv;
                        int ti = idxs[i]; idxs[i] = idxs[partner]; idxs[partner] = ti;
                    }
                }
            }
        }
    }
}

template <typename scalar_t, int K>
__device__ __forceinline__ void merge_two_sorted(
    const scalar_t* a_vals, const int* a_idxs,
    const scalar_t* b_vals, const int* b_idxs,
    scalar_t* out_vals, int* out_idxs)
{
    int ai = 0, bi = 0;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        bool take_a = (bi >= K) || ((ai < K) && (a_vals[ai] >= b_vals[bi]));
        if (take_a) {
            out_vals[i] = a_vals[ai]; out_idxs[i] = a_idxs[ai]; ++ai;
        } else {
            out_vals[i] = b_vals[bi]; out_idxs[i] = b_idxs[bi]; ++bi;
        }
    }
}

constexpr int WARP_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

constexpr int K2_F32_PEAK_DIM = 8192;
constexpr int K2_F16_PEAK_DIM = 16384;

constexpr int WARPS_PER_ROW_LARGE = 4;
constexpr int BLOCK_SIZE_MW = WARP_SIZE * WARPS_PER_ROW_LARGE;

constexpr int WARPS_PER_ROW_XL = 8;
constexpr int BLOCK_SIZE_MW_XL = WARP_SIZE * WARPS_PER_ROW_XL;

constexpr int K2_F32_LARGE_DIM = 65536;
constexpr int K2_F16_LARGE_DIM = 131072;

constexpr int K4_UNROLL_THRESHOLD = 32768;
constexpr int K8_UNROLL_THRESHOLD = 32768;

template<typename T>
__device__ __forceinline__ T safe_neg_inf() {
    return std::numeric_limits<T>::lowest();
}

template<typename T>
__device__ __forceinline__ T safe_pos_inf() {
    return std::numeric_limits<T>::max();
}

// Replace the current minimum of the running top-K buffer with `val` and return the
// buffer's new minimum, so callers can test candidates against an always-current
// threshold. Only the evicted slot changes, hence new_min = min(second_min, val);
// tracking (min, argmin, second_min) in the same pass costs no extra scan.
// Keeping a stale (smaller) threshold instead would admit values below the true
// minimum, evicting a legitimate top-K element and silently corrupting the result.
template <typename cmp_t, int K>
__device__ __forceinline__ cmp_t buffer_insert_min(cmp_t* vals, int* idxs, cmp_t val, int idx) {
    int mp = 0;
    cmp_t cur_min = vals[0];
    // Start from +inf so the scan yields the smallest slot other than `mp` even when
    // vals[0] itself holds the minimum; a partly filled buffer (all -inf slots) then
    // still reports -inf as its minimum.
    cmp_t second_min = safe_pos_inf<cmp_t>();
    #pragma unroll
    for (int i = 1; i < K; ++i) {
        if (vals[i] < cur_min) {
            second_min = cur_min;
            cur_min = vals[i];
            mp = i;
        } else if (vals[i] < second_min) {
            second_min = vals[i];
        }
    }
    vals[mp] = val;
    idxs[mp] = idx;
    return second_min < val ? second_min : val;
}

template <typename scalar_t>
__global__ void topk_kernel_k2_dedicated(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * 2];
    __shared__ int smem_idxs[BLOCK_SIZE * 2];

    // b is uniform inside a 64-lane wave, so a wave is either fully active or
    // fully out of range. Each wave owns its own smem slice (tid*K) and only
    // merges with lanes of the same wave, so wave-level sync is both sufficient
    // and required here: a block-wide __syncthreads() would be reached by only
    // part of the block when batch_size is not a multiple of WARPS_PER_BLOCK.
    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t v0 = safe_neg_inf<cmp_t>();
    cmp_t v1 = safe_neg_inf<cmp_t>();
    int i0 = -1, i1 = -1;
    cmp_t vmin = safe_neg_inf<cmp_t>();

    int vec_dim = dim_size / VW;

    for (int j = lane; j < vec_dim; j += WARP_SIZE) {
        scalar_t vals[VW];
        load_vec<scalar_t, VW>(&row[j * VW], vals);
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vals[v]);
            int idx = j * VW + v;
            cmp_t old_v0 = v0; int old_i0 = i0;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > old_v0);
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
    }

    for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
        cmp_t val = static_cast<cmp_t>(row[j]);
        cmp_t old_v0 = v0; int old_i0 = i0;
        bool above_min = (val > vmin);
        bool promote = above_min & (val > old_v0);
        v0 = promote ? val : old_v0;
        i0 = promote ? j : old_i0;
        v1 = promote ? old_v0 : (above_min ? val : v1);
        i1 = promote ? old_i0 : (above_min ? j : i1);
        vmin = above_min ? (promote ? old_v0 : val) : vmin;
    }

    cmp_t buf_vals[2] = {v0, v1};
    int buf_idxs[2] = {i0, i1};

    smem_vals[tid_in_block * 2] = buf_vals[0]; smem_idxs[tid_in_block * 2] = buf_idxs[0];
    smem_vals[tid_in_block * 2 + 1] = buf_vals[1]; smem_idxs[tid_in_block * 2 + 1] = buf_idxs[1];
    __syncwarp();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[2] = {smem_vals[neighbor * 2], smem_vals[neighbor * 2 + 1]};
            int ri[2] = {smem_idxs[neighbor * 2], smem_idxs[neighbor * 2 + 1]};
            cmp_t mv[2]; int mi[2];
            merge_two_sorted<cmp_t, 2>(buf_vals, buf_idxs, rv, ri, mv, mi);
            buf_vals[0] = mv[0]; buf_vals[1] = mv[1];
            buf_idxs[0] = mi[0]; buf_idxs[1] = mi[1];
        }
        smem_vals[tid_in_block * 2] = buf_vals[0]; smem_idxs[tid_in_block * 2] = buf_idxs[0];
        smem_vals[tid_in_block * 2 + 1] = buf_vals[1]; smem_idxs[tid_in_block * 2 + 1] = buf_idxs[1];
        __syncwarp();
    }

    if (lane == 0) {
        value[b * 2]     = static_cast<scalar_t>(buf_vals[0]); indice[b * 2]     = static_cast<int64_t>(buf_idxs[0]);
        value[b * 2 + 1] = static_cast<scalar_t>(buf_vals[1]); indice[b * 2 + 1] = static_cast<int64_t>(buf_idxs[1]);
    }
}

template <typename scalar_t, int WARPS_PER_ROW>
__global__ void topk_kernel_k2_multiwarp(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    int b = blockIdx.x;
    if (b >= batch_size) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    const scalar_t* row = src + (int64_t)b * dim_size;

    int vec_dim = dim_size / VW;
    int vecs_per_warp = vec_dim / WARPS_PER_ROW;
    int vec_start = warp_id * vecs_per_warp;
    int vec_end = (warp_id == WARPS_PER_ROW - 1) ? vec_dim : vec_start + vecs_per_warp;
    bool handle_scalar_rem = (warp_id == WARPS_PER_ROW - 1);

    cmp_t v0 = safe_neg_inf<cmp_t>();
    cmp_t v1 = safe_neg_inf<cmp_t>();
    int i0 = -1, i1 = -1;
    cmp_t vmin = safe_neg_inf<cmp_t>();

    int j = vec_start + lane;
    int stride = WARP_SIZE;
    int stride4 = stride * 4;

    for (; j + stride * 3 < vec_end; j += stride4) {
        scalar_t va[VW], vb[VW], vc[VW], vd[VW];
        load_vec<scalar_t, VW>(&row[j * VW], va);
        load_vec<scalar_t, VW>(&row[(j + stride) * VW], vb);
        load_vec<scalar_t, VW>(&row[(j + stride * 2) * VW], vc);
        load_vec<scalar_t, VW>(&row[(j + stride * 3) * VW], vd);

        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(va[v]);
            int idx = j * VW + v;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vb[v]);
            int idx = (j + stride) * VW + v;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vc[v]);
            int idx = (j + stride * 2) * VW + v;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vd[v]);
            int idx = (j + stride * 3) * VW + v;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
    }

    for (; j < vec_end; j += stride) {
        scalar_t vals[VW];
        load_vec<scalar_t, VW>(&row[j * VW], vals);
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vals[v]);
            int idx = j * VW + v;
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? idx : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? idx : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
    }

    if (handle_scalar_rem) {
        int scalar_start = vec_dim * VW;
        for (int s = scalar_start + lane; s < dim_size; s += WARP_SIZE) {
            cmp_t val = static_cast<cmp_t>(row[s]);
            bool above_min = (val > vmin);
            bool promote = above_min & (val > v0);
            cmp_t old_v0 = v0; int old_i0 = i0;
            v0 = promote ? val : old_v0;
            i0 = promote ? s : old_i0;
            v1 = promote ? old_v0 : (above_min ? val : v1);
            i1 = promote ? old_i0 : (above_min ? s : i1);
            vmin = above_min ? (promote ? old_v0 : val) : vmin;
        }
    }

    __shared__ cmp_t smem_vals[WARPS_PER_ROW * WARP_SIZE * 2];
    __shared__ int smem_idxs[WARPS_PER_ROW * WARP_SIZE * 2];

    int tid = threadIdx.x;
    cmp_t buf_vals[2] = {v0, v1};
    int buf_idxs[2] = {i0, i1};

    smem_vals[tid * 2] = buf_vals[0]; smem_idxs[tid * 2] = buf_idxs[0];
    smem_vals[tid * 2 + 1] = buf_vals[1]; smem_idxs[tid * 2 + 1] = buf_idxs[1];
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid + offset;
            cmp_t rv[2] = {smem_vals[neighbor * 2], smem_vals[neighbor * 2 + 1]};
            int ri[2] = {smem_idxs[neighbor * 2], smem_idxs[neighbor * 2 + 1]};
            cmp_t mv[2]; int mi[2];
            merge_two_sorted<cmp_t, 2>(buf_vals, buf_idxs, rv, ri, mv, mi);
            buf_vals[0] = mv[0]; buf_vals[1] = mv[1];
            buf_idxs[0] = mi[0]; buf_idxs[1] = mi[1];
        }
        smem_vals[tid * 2] = buf_vals[0]; smem_idxs[tid * 2] = buf_idxs[0];
        smem_vals[tid * 2 + 1] = buf_vals[1]; smem_idxs[tid * 2 + 1] = buf_idxs[1];
        __syncthreads();
    }

    __shared__ cmp_t warp_vals[WARPS_PER_ROW * 2];
    __shared__ int warp_idxs[WARPS_PER_ROW * 2];

    if (lane == 0) {
        warp_vals[warp_id * 2] = buf_vals[0];
        warp_idxs[warp_id * 2] = buf_idxs[0];
        warp_vals[warp_id * 2 + 1] = buf_vals[1];
        warp_idxs[warp_id * 2 + 1] = buf_idxs[1];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        cmp_t final_vals[2] = {warp_vals[0], warp_vals[1]};
        int final_idxs[2] = {warp_idxs[0], warp_idxs[1]};
        #pragma unroll
        for (int w = 1; w < WARPS_PER_ROW; ++w) {
            cmp_t wv[2] = {warp_vals[w * 2], warp_vals[w * 2 + 1]};
            int wi[2] = {warp_idxs[w * 2], warp_idxs[w * 2 + 1]};
            cmp_t mv[2]; int mi[2];
            merge_two_sorted<cmp_t, 2>(final_vals, final_idxs, wv, wi, mv, mi);
            final_vals[0] = mv[0]; final_vals[1] = mv[1];
            final_idxs[0] = mi[0]; final_idxs[1] = mi[1];
        }
        value[b * 2] = static_cast<scalar_t>(final_vals[0]);
        indice[b * 2] = static_cast<int64_t>(final_idxs[0]);
        value[b * 2 + 1] = static_cast<scalar_t>(final_vals[1]);
        indice[b * 2 + 1] = static_cast<int64_t>(final_idxs[1]);
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k4(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * 4];
    __shared__ int smem_idxs[BLOCK_SIZE * 4];

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t buf_vals[4]; int buf_idxs[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { buf_vals[i] = safe_neg_inf<cmp_t>(); buf_idxs[i] = -1; }
    cmp_t buf_min = safe_neg_inf<cmp_t>();

    int vec_dim = dim_size / VW;
    for (int j = lane; j < vec_dim; j += WARP_SIZE) {
        scalar_t vals[VW];
        load_vec<scalar_t, VW>(&row[j * VW], vals);
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vals[v]);
            if (val > buf_min) {
                int idx = j * VW + v;
                if (val > buf_vals[0]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx; }
                else if (val > buf_vals[1]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx; }
                else if (val > buf_vals[2]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx; }
                else { buf_vals[3]=val;buf_idxs[3]=idx; }
                buf_min = buf_vals[3];
            }
        }
    }
    for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
        cmp_t val = static_cast<cmp_t>(row[j]);
        if (val > buf_min) {
            if (val > buf_vals[0]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=j; }
            else if (val > buf_vals[1]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=j; }
            else if (val > buf_vals[2]) { buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=j; }
            else { buf_vals[3]=val;buf_idxs[3]=j; }
            buf_min = buf_vals[3];
        }
    }

    bitonic_sort<cmp_t, 4>(buf_vals, buf_idxs);

    #pragma unroll
    for (int i = 0; i < 4; ++i) { smem_vals[tid_in_block*4+i]=buf_vals[i]; smem_idxs[tid_in_block*4+i]=buf_idxs[i]; }
    __syncwarp();

    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[4]; int ri[4];
            #pragma unroll
            for (int i=0;i<4;++i){rv[i]=smem_vals[neighbor*4+i];ri[i]=smem_idxs[neighbor*4+i];}
            cmp_t mv[4]; int mi[4];
            merge_two_sorted<cmp_t,4>(buf_vals,buf_idxs,rv,ri,mv,mi);
            #pragma unroll
            for (int i=0;i<4;++i){buf_vals[i]=mv[i];buf_idxs[i]=mi[i];}
        }
        #pragma unroll
        for (int i=0;i<4;++i){smem_vals[tid_in_block*4+i]=buf_vals[i];smem_idxs[tid_in_block*4+i]=buf_idxs[i];}
        __syncwarp();
    }

    if (lane == 0) {
        #pragma unroll
        for (int i=0;i<4;++i){value[b*4+i]=static_cast<scalar_t>(buf_vals[i]);indice[b*4+i]=static_cast<int64_t>(buf_idxs[i]);}
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k4_unroll4(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int K = 4;
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * K];
    __shared__ int smem_idxs[BLOCK_SIZE * K];

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + (int64_t)b * dim_size;

    cmp_t buf_vals[K]; int buf_idxs[K];
    #pragma unroll
    for (int i=0;i<K;++i){buf_vals[i]=safe_neg_inf<cmp_t>();buf_idxs[i]=-1;}
    cmp_t buf_min = safe_neg_inf<cmp_t>();

    int vec_dim = dim_size / VW;
    int stride = WARP_SIZE;
    int stride4 = stride * 4;

    int j = lane;
    for (; j+stride*3<vec_dim; j+=stride4) {
        scalar_t va[VW],vb[VW],vc[VW],vd[VW];
        load_vec<scalar_t,VW>(&row[j*VW],va);
        load_vec<scalar_t,VW>(&row[(j+stride)*VW],vb);
        load_vec<scalar_t,VW>(&row[(j+stride*2)*VW],vc);
        load_vec<scalar_t,VW>(&row[(j+stride*3)*VW],vd);

        #pragma unroll
        for (int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(va[v]);if(val>buf_min){int idx=j*VW+v;
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx;}
            else{buf_vals[3]=val;buf_idxs[3]=idx;}buf_min=buf_vals[3];}}
        #pragma unroll
        for (int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vb[v]);if(val>buf_min){int idx=(j+stride)*VW+v;
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx;}
            else{buf_vals[3]=val;buf_idxs[3]=idx;}buf_min=buf_vals[3];}}
        #pragma unroll
        for (int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vc[v]);if(val>buf_min){int idx=(j+stride*2)*VW+v;
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx;}
            else{buf_vals[3]=val;buf_idxs[3]=idx;}buf_min=buf_vals[3];}}
        #pragma unroll
        for (int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vd[v]);if(val>buf_min){int idx=(j+stride*3)*VW+v;
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx;}
            else{buf_vals[3]=val;buf_idxs[3]=idx;}buf_min=buf_vals[3];}}
    }

    for (;j<vec_dim;j+=stride){
        scalar_t vals[VW];load_vec<scalar_t,VW>(&row[j*VW],vals);
        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vals[v]);if(val>buf_min){int idx=j*VW+v;
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=idx;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=idx;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=idx;}
            else{buf_vals[3]=val;buf_idxs[3]=idx;}buf_min=buf_vals[3];}}}

    for(int s=vec_dim*VW+lane;s<dim_size;s+=WARP_SIZE){
        cmp_t val=static_cast<cmp_t>(row[s]);if(val>buf_min){
            if(val>buf_vals[0]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=buf_vals[0];buf_idxs[1]=buf_idxs[0];buf_vals[0]=val;buf_idxs[0]=s;}
            else if(val>buf_vals[1]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=buf_vals[1];buf_idxs[2]=buf_idxs[1];buf_vals[1]=val;buf_idxs[1]=s;}
            else if(val>buf_vals[2]){buf_vals[3]=buf_vals[2];buf_idxs[3]=buf_idxs[2];buf_vals[2]=val;buf_idxs[2]=s;}
            else{buf_vals[3]=val;buf_idxs[3]=s;}buf_min=buf_vals[3];}}

    bitonic_sort<cmp_t,K>(buf_vals,buf_idxs);

    #pragma unroll
    for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
    __syncwarp();

    for(int offset=WARP_SIZE/2;offset>0;offset>>=1){
        if(lane<offset){
            int neighbor=tid_in_block+offset;
            cmp_t rv[K];int ri[K];
            #pragma unroll
            for(int i=0;i<K;++i){rv[i]=smem_vals[neighbor*K+i];ri[i]=smem_idxs[neighbor*K+i];}
            cmp_t mv[K];int mi[K];
            merge_two_sorted<cmp_t,K>(buf_vals,buf_idxs,rv,ri,mv,mi);
            #pragma unroll
            for(int i=0;i<K;++i){buf_vals[i]=mv[i];buf_idxs[i]=mi[i];}
        }
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
        __syncwarp();
    }

    if(lane==0){
        #pragma unroll
        for(int i=0;i<K;++i){value[b*K+i]=static_cast<scalar_t>(buf_vals[i]);indice[b*K+i]=static_cast<int64_t>(buf_idxs[i]);}
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k8(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int K=8;
    constexpr int VW=VecWidth<scalar_t>::value;
    using cmp_t=typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE*K];
    __shared__ int smem_idxs[BLOCK_SIZE*K];

    int b=(blockIdx.x*WARPS_PER_BLOCK)+(threadIdx.x/WARP_SIZE);
    if(b>=batch_size) return;

    int lane=threadIdx.x%WARP_SIZE;
    int tid_in_block=threadIdx.x;
    const scalar_t* row=src+b*dim_size;

    cmp_t buf_vals[K];int buf_idxs[K];
    #pragma unroll
    for(int i=0;i<K;++i){buf_vals[i]=safe_neg_inf<cmp_t>();buf_idxs[i]=-1;}
    cmp_t buf_min=safe_neg_inf<cmp_t>();

    int vec_dim=dim_size/VW;
    for(int j=lane;j<vec_dim;j+=WARP_SIZE){
        scalar_t vals[VW];load_vec<scalar_t,VW>(&row[j*VW],vals);
        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vals[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,j*VW+v);}}
    }
    for(int j=vec_dim*VW+lane;j<dim_size;j+=WARP_SIZE){
        cmp_t val=static_cast<cmp_t>(row[j]);
        if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,j);}
    }

    bitonic_sort<cmp_t,K>(buf_vals,buf_idxs);

    #pragma unroll
    for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
    __syncwarp();

    for(int offset=WARP_SIZE/2;offset>0;offset>>=1){
        if(lane<offset){
            int neighbor=tid_in_block+offset;
            cmp_t rv[K];int ri[K];
            #pragma unroll
            for(int i=0;i<K;++i){rv[i]=smem_vals[neighbor*K+i];ri[i]=smem_idxs[neighbor*K+i];}
            cmp_t mv[K];int mi[K];
            merge_two_sorted<cmp_t,K>(buf_vals,buf_idxs,rv,ri,mv,mi);
            #pragma unroll
            for(int i=0;i<K;++i){buf_vals[i]=mv[i];buf_idxs[i]=mi[i];}
        }
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
        __syncwarp();
    }

    if(lane==0){
        #pragma unroll
        for(int i=0;i<K;++i){value[b*K+i]=static_cast<scalar_t>(buf_vals[i]);indice[b*K+i]=static_cast<int64_t>(buf_idxs[i]);}
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k8_unroll4(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int K=8;
    constexpr int VW=VecWidth<scalar_t>::value;
    using cmp_t=typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE*K];
    __shared__ int smem_idxs[BLOCK_SIZE*K];

    int b=(blockIdx.x*WARPS_PER_BLOCK)+(threadIdx.x/WARP_SIZE);
    if(b>=batch_size) return;

    int lane=threadIdx.x%WARP_SIZE;
    int tid_in_block=threadIdx.x;
    const scalar_t* row=src+(int64_t)b*dim_size;

    cmp_t buf_vals[K];int buf_idxs[K];
    #pragma unroll
    for(int i=0;i<K;++i){buf_vals[i]=safe_neg_inf<cmp_t>();buf_idxs[i]=-1;}
    cmp_t buf_min=safe_neg_inf<cmp_t>();

    int vec_dim=dim_size/VW;
    int stride=WARP_SIZE;
    int stride4=stride*4;

    int j=lane;
    for(;j+stride*3<vec_dim;j+=stride4){
        scalar_t va[VW],vb[VW],vc[VW],vd[VW];
        load_vec<scalar_t,VW>(&row[j*VW],va);
        load_vec<scalar_t,VW>(&row[(j+stride)*VW],vb);
        load_vec<scalar_t,VW>(&row[(j+stride*2)*VW],vc);
        load_vec<scalar_t,VW>(&row[(j+stride*3)*VW],vd);

        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(va[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,j*VW+v);}}

        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vb[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,(j+stride)*VW+v);}}

        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vc[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,(j+stride*2)*VW+v);}}

        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vd[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,(j+stride*3)*VW+v);}}
    }

    for(;j<vec_dim;j+=stride){
        scalar_t vals[VW];load_vec<scalar_t,VW>(&row[j*VW],vals);
        #pragma unroll
        for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vals[v]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,j*VW+v);}}
    }

    for(int s=vec_dim*VW+lane;s<dim_size;s+=WARP_SIZE){
        cmp_t val=static_cast<cmp_t>(row[s]);
        if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf_vals,buf_idxs,val,s);}
    }

    bitonic_sort<cmp_t,K>(buf_vals,buf_idxs);

    #pragma unroll
    for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
    __syncwarp();

    for(int offset=WARP_SIZE/2;offset>0;offset>>=1){
        if(lane<offset){
            int neighbor=tid_in_block+offset;
            cmp_t rv[K];int ri[K];
            #pragma unroll
            for(int i=0;i<K;++i){rv[i]=smem_vals[neighbor*K+i];ri[i]=smem_idxs[neighbor*K+i];}
            cmp_t mv[K];int mi[K];
            merge_two_sorted<cmp_t,K>(buf_vals,buf_idxs,rv,ri,mv,mi);
            #pragma unroll
            for(int i=0;i<K;++i){buf_vals[i]=mv[i];buf_idxs[i]=mi[i];}
        }
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf_vals[i];smem_idxs[tid_in_block*K+i]=buf_idxs[i];}
        __syncwarp();
    }

    if(lane==0){
        #pragma unroll
        for(int i=0;i<K;++i){value[b*K+i]=static_cast<scalar_t>(buf_vals[i]);indice[b*K+i]=static_cast<int64_t>(buf_idxs[i]);}
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k16(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int K=16;
    constexpr int HALF_K=8;
    constexpr int VW=VecWidth<scalar_t>::value;
    using cmp_t=typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE*K];
    __shared__ int smem_idxs[BLOCK_SIZE*K];

    int b=(blockIdx.x*WARPS_PER_BLOCK)+(threadIdx.x/WARP_SIZE);
    if(b>=batch_size) return;

    int lane=threadIdx.x%WARP_SIZE;
    int tid_in_block=threadIdx.x;
    const scalar_t* row=src+b*dim_size;

    cmp_t neg_inf=safe_neg_inf<cmp_t>();

    if(dim_size<=64){
        cmp_t buf[K];int idx[K];
        #pragma unroll
        for(int i=0;i<K;++i){buf[i]=neg_inf;idx[i]=-1;}
        cmp_t buf_min=neg_inf;

        int vec_dim=dim_size/VW;
        for(int j=lane;j<vec_dim;j+=WARP_SIZE){
            scalar_t vals[VW];load_vec<scalar_t,VW>(&row[j*VW],vals);
            #pragma unroll
            for(int v=0;v<VW;++v){cmp_t val=static_cast<cmp_t>(vals[v]);
                if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf,idx,val,j*VW+v);}}
        }
        for(int j=vec_dim*VW+lane;j<dim_size;j+=WARP_SIZE){
            cmp_t val=static_cast<cmp_t>(row[j]);
            if(val>buf_min){buf_min=buffer_insert_min<cmp_t,K>(buf,idx,val,j);}
        }

        bitonic_sort<cmp_t,K>(buf,idx);
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=buf[i];smem_idxs[tid_in_block*K+i]=idx[i];}
    } else {
        cmp_t buf_a[HALF_K],buf_b[HALF_K];
        int idx_a[HALF_K],idx_b[HALF_K];
        #pragma unroll
        for(int i=0;i<HALF_K;++i){buf_a[i]=neg_inf;idx_a[i]=-1;buf_b[i]=neg_inf;idx_b[i]=-1;}

        cmp_t min_a=neg_inf,min_b=neg_inf;
        int vec_dim=dim_size/VW;

        for(int j=lane;j<vec_dim;j+=WARP_SIZE){
            scalar_t vals[VW];load_vec<scalar_t,VW>(&row[j*VW],vals);
            #pragma unroll
            for(int v=0;v<VW;++v){
                cmp_t val=static_cast<cmp_t>(vals[v]);int idx_val=j*VW+v;
                if(val>min_a){min_a=buffer_insert_min<cmp_t,HALF_K>(buf_a,idx_a,val,idx_val);}
                else if(val>min_b){min_b=buffer_insert_min<cmp_t,HALF_K>(buf_b,idx_b,val,idx_val);}
            }
        }
        for(int j=vec_dim*VW+lane;j<dim_size;j+=WARP_SIZE){
            cmp_t val=static_cast<cmp_t>(row[j]);
            if(val>min_a){min_a=buffer_insert_min<cmp_t,HALF_K>(buf_a,idx_a,val,j);}
            else if(val>min_b){min_b=buffer_insert_min<cmp_t,HALF_K>(buf_b,idx_b,val,j);}
        }

        bitonic_sort<cmp_t,HALF_K>(buf_a,idx_a);
        bitonic_sort<cmp_t,HALF_K>(buf_b,idx_b);

        cmp_t merged[K];int merged_idx[K];
        merge_two_sorted<cmp_t,HALF_K>(buf_a,idx_a,buf_b,idx_b,merged,merged_idx);
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=merged[i];smem_idxs[tid_in_block*K+i]=merged_idx[i];}
    }

    __syncwarp();

    cmp_t final_vals[K];int final_idxs[K];
    #pragma unroll
    for(int i=0;i<K;++i){final_vals[i]=smem_vals[tid_in_block*K+i];final_idxs[i]=smem_idxs[tid_in_block*K+i];}

    for(int offset=WARP_SIZE/2;offset>0;offset>>=1){
        if(lane<offset){
            int neighbor=tid_in_block+offset;
            cmp_t rv[K];int ri[K];
            #pragma unroll
            for(int i=0;i<K;++i){rv[i]=smem_vals[neighbor*K+i];ri[i]=smem_idxs[neighbor*K+i];}
            cmp_t mv[K];int mi[K];
            merge_two_sorted<cmp_t,K>(final_vals,final_idxs,rv,ri,mv,mi);
            #pragma unroll
            for(int i=0;i<K;++i){final_vals[i]=mv[i];final_idxs[i]=mi[i];}
        }
        #pragma unroll
        for(int i=0;i<K;++i){smem_vals[tid_in_block*K+i]=final_vals[i];smem_idxs[tid_in_block*K+i]=final_idxs[i];}
        __syncwarp();
    }

    if(lane==0){
        #pragma unroll
        for(int i=0;i<K;++i){value[b*K+i]=static_cast<scalar_t>(final_vals[i]);indice[b*K+i]=static_cast<int64_t>(final_idxs[i]);}
    }
}

template <typename scalar_t>
void launch_topk_kernel(
    const scalar_t* src,
    scalar_t* value,
    int64_t* indice,
    int batch_size,
    int dim_size,
    int k,
    cudaStream_t stream)
{
    constexpr bool is_f32 = std::is_same<scalar_t, float>::value;
    int num_blocks_4warp = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (k == 2) {
        int threshold = is_f32 ? K2_F32_PEAK_DIM : K2_F16_PEAK_DIM;

        if (dim_size <= threshold) {
            topk_kernel_k2_dedicated<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
        } else {
            int large_threshold = is_f32 ? K2_F32_LARGE_DIM : K2_F16_LARGE_DIM;

            if (dim_size < large_threshold) {
                topk_kernel_k2_multiwarp<scalar_t, WARPS_PER_ROW_LARGE>
                    <<<batch_size, BLOCK_SIZE_MW, 0, stream>>>(
                    src, value, indice, batch_size, dim_size);
            } else {
                topk_kernel_k2_multiwarp<scalar_t, WARPS_PER_ROW_XL>
                    <<<batch_size, BLOCK_SIZE_MW_XL, 0, stream>>>(
                    src, value, indice, batch_size, dim_size);
            }
        }
    } else if (k == 4) {
        if (dim_size <= K4_UNROLL_THRESHOLD) {
            topk_kernel_k4<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
        } else {
            topk_kernel_k4_unroll4<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
        }
    } else if (k == 8) {
        if (is_f32 && dim_size > K8_UNROLL_THRESHOLD) {
            topk_kernel_k8_unroll4<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
        } else {
            topk_kernel_k8<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
        }
    } else {
        topk_kernel_k16<scalar_t><<<num_blocks_4warp, BLOCK_SIZE, 0, stream>>>(
            src, value, indice, batch_size, dim_size);
    }
}

namespace at {
namespace native {

std::vector<torch::Tensor> topk_power2_opt(
    torch::Tensor& value,
    torch::Tensor& indice,
    torch::Tensor& src,
    int64_t k)
{
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(indice.is_contiguous(), "indice must be contiguous");
    // Keep k=16 support for direct callers. Python dispatches this fast path
    // only for k=2/4/8 because k=16 performs better through the fallback.
    TORCH_CHECK(k == 2 || k == 4 || k == 8 || k == 16, "Only k=2,4,8,16 are supported");

    auto dim_size = src.size(-1);
    auto batch_size = src.numel() / dim_size;
    int bs_int = static_cast<int>(batch_size);
    int ds_int = static_cast<int>(dim_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        src.scalar_type(), "topk_power2_opt", [&] {
            launch_topk_kernel<scalar_t>(
                src.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                indice.data_ptr<int64_t>(),
                bs_int, ds_int,
                static_cast<int>(k),
                stream);
        });

    return {value, indice};
}

}  // namespace native
}  // namespace at
