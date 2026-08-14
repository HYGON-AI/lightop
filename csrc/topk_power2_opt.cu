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

template <typename scalar_t, int K, int I>
struct ArgMinUnroll {
    __device__ __forceinline__ static int find(const scalar_t* vals, int cur_min_pos, scalar_t cur_min_val) {
        scalar_t v = vals[I];
        bool is_smaller = (v < cur_min_val);
        cur_min_val = is_smaller ? v : cur_min_val;
        cur_min_pos = is_smaller ? I : cur_min_pos;
        return ArgMinUnroll<scalar_t, K, I + 1>::find(vals, cur_min_pos, cur_min_val);
    }
};

template <typename scalar_t, int K>
struct ArgMinUnroll<scalar_t, K, K> {
    __device__ __forceinline__ static int find(const scalar_t*, int cur_min_pos, scalar_t) {
        return cur_min_pos;
    }
};

template <typename scalar_t, int K>
__device__ __forceinline__ int find_argmin_unrolled(const scalar_t* vals) {
    return ArgMinUnroll<scalar_t, K, 1>::find(vals, 0, vals[0]);
}

constexpr int WARP_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

constexpr int K2_DEDICATED_DIM_THRESHOLD = 65536;

template<typename T>
__device__ __forceinline__ T safe_neg_inf() {
    return std::numeric_limits<T>::lowest();
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

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t v0 = safe_neg_inf<cmp_t>();
    cmp_t v1 = safe_neg_inf<cmp_t>();
    int i0 = -1;
    int i1 = -1;
    cmp_t vmin = safe_neg_inf<cmp_t>();

    int vec_dim = dim_size / VW;

    for (int j = lane; j < vec_dim; j += WARP_SIZE) {
        scalar_t vals[VW];
        load_vec<scalar_t, VW>(&row[j * VW], vals);
        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vals[v]);
            int idx = j * VW + v;

            cmp_t old_v0 = v0;
            int old_i0 = i0;

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

        cmp_t old_v0 = v0;
        int old_i0 = i0;

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
    __syncthreads();

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
        __syncthreads();
    }

    if (lane == 0) {
        value[b * 2]     = static_cast<scalar_t>(buf_vals[0]); indice[b * 2]     = static_cast<int64_t>(buf_idxs[0]);
        value[b * 2 + 1] = static_cast<scalar_t>(buf_vals[1]); indice[b * 2 + 1] = static_cast<int64_t>(buf_idxs[1]);
    }
}

template <typename scalar_t>
__global__ void topk_kernel_k2(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ value,
    int64_t* __restrict__ indice,
    int batch_size,
    int dim_size)
{
    constexpr int K = 2;
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * K];
    __shared__ int smem_idxs[BLOCK_SIZE * K];

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t buf_vals[K];
    int buf_idxs[K];
    buf_vals[0] = safe_neg_inf<cmp_t>();
    buf_vals[1] = safe_neg_inf<cmp_t>();
    buf_idxs[0] = -1;
    buf_idxs[1] = -1;
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
                if (val > buf_vals[0]) {
                    buf_vals[1] = buf_vals[0]; buf_idxs[1] = buf_idxs[0];
                    buf_vals[0] = val;         buf_idxs[0] = idx;
                } else {
                    buf_vals[1] = val;         buf_idxs[1] = idx;
                }
                buf_min = buf_vals[1];
            }
        }
    }

    for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
        cmp_t val = static_cast<cmp_t>(row[j]);
        if (val > buf_min) {
            if (val > buf_vals[0]) {
                buf_vals[1] = buf_vals[0]; buf_idxs[1] = buf_idxs[0];
                buf_vals[0] = val;         buf_idxs[0] = j;
            } else {
                buf_vals[1] = val;         buf_idxs[1] = j;
            }
            buf_min = buf_vals[1];
        }
    }

    bitonic_sort<cmp_t, K>(buf_vals, buf_idxs);

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        smem_vals[tid_in_block * K + i] = buf_vals[i];
        smem_idxs[tid_in_block * K + i] = buf_idxs[i];
    }
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[K]; int ri[K];
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                rv[i] = smem_vals[neighbor * K + i];
                ri[i] = smem_idxs[neighbor * K + i];
            }
            cmp_t mv[K]; int mi[K];
            merge_two_sorted<cmp_t, K>(buf_vals, buf_idxs, rv, ri, mv, mi);
            #pragma unroll
            for (int i = 0; i < K; ++i) { buf_vals[i] = mv[i]; buf_idxs[i] = mi[i]; }
        }
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            smem_vals[tid_in_block * K + i] = buf_vals[i];
            smem_idxs[tid_in_block * K + i] = buf_idxs[i];
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            value[b * K + i]  = static_cast<scalar_t>(buf_vals[i]);
            indice[b * K + i] = static_cast<int64_t>(buf_idxs[i]);
        }
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

    cmp_t buf_vals[4];
    int buf_idxs[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        buf_vals[i] = safe_neg_inf<cmp_t>();
        buf_idxs[i] = -1;
    }
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
                if (val > buf_vals[0]) {
                    buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                    buf_vals[2] = buf_vals[1]; buf_idxs[2] = buf_idxs[1];
                    buf_vals[1] = buf_vals[0]; buf_idxs[1] = buf_idxs[0];
                    buf_vals[0] = val;         buf_idxs[0] = idx;
                } else if (val > buf_vals[1]) {
                    buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                    buf_vals[2] = buf_vals[1]; buf_idxs[2] = buf_idxs[1];
                    buf_vals[1] = val;         buf_idxs[1] = idx;
                } else if (val > buf_vals[2]) {
                    buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                    buf_vals[2] = val;         buf_idxs[2] = idx;
                } else {
                    buf_vals[3] = val;         buf_idxs[3] = idx;
                }
                buf_min = buf_vals[3];
            }
        }
    }
    for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
        cmp_t val = static_cast<cmp_t>(row[j]);
        if (val > buf_min) {
            if (val > buf_vals[0]) {
                buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                buf_vals[2] = buf_vals[1]; buf_idxs[2] = buf_idxs[1];
                buf_vals[1] = buf_vals[0]; buf_idxs[1] = buf_idxs[0];
                buf_vals[0] = val;         buf_idxs[0] = j;
            } else if (val > buf_vals[1]) {
                buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                buf_vals[2] = buf_vals[1]; buf_idxs[2] = buf_idxs[1];
                buf_vals[1] = val;         buf_idxs[1] = j;
            } else if (val > buf_vals[2]) {
                buf_vals[3] = buf_vals[2]; buf_idxs[3] = buf_idxs[2];
                buf_vals[2] = val;         buf_idxs[2] = j;
            } else {
                buf_vals[3] = val;         buf_idxs[3] = j;
            }
            buf_min = buf_vals[3];
        }
    }

    bitonic_sort<cmp_t, 4>(buf_vals, buf_idxs);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        smem_vals[tid_in_block * 4 + i] = buf_vals[i];
        smem_idxs[tid_in_block * 4 + i] = buf_idxs[i];
    }
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[4]; int ri[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                rv[i] = smem_vals[neighbor * 4 + i];
                ri[i] = smem_idxs[neighbor * 4 + i];
            }
            cmp_t mv[4]; int mi[4];
            merge_two_sorted<cmp_t, 4>(buf_vals, buf_idxs, rv, ri, mv, mi);
            #pragma unroll
            for (int i = 0; i < 4; ++i) { buf_vals[i] = mv[i]; buf_idxs[i] = mi[i]; }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            smem_vals[tid_in_block * 4 + i] = buf_vals[i];
            smem_idxs[tid_in_block * 4 + i] = buf_idxs[i];
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            value[b * 4 + i] = static_cast<scalar_t>(buf_vals[i]);
            indice[b * 4 + i] = static_cast<int64_t>(buf_idxs[i]);
        }
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
    constexpr int K = 8;
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * K];
    __shared__ int smem_idxs[BLOCK_SIZE * K];

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t buf_vals[K];
    int buf_idxs[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        buf_vals[i] = safe_neg_inf<cmp_t>();
        buf_idxs[i] = -1;
    }

    cmp_t buf_min = safe_neg_inf<cmp_t>();
    bool buf_dirty = false;

    int vec_dim = dim_size / VW;

    for (int j = lane; j < vec_dim; j += WARP_SIZE) {
        scalar_t vals[VW];
        load_vec<scalar_t, VW>(&row[j * VW], vals);

        #pragma unroll
        for (int v = 0; v < VW; ++v) {
            cmp_t val = static_cast<cmp_t>(vals[v]);
            if (val > buf_min) {
                int min_pos = 0;
                cmp_t cur_min = buf_vals[0];
                #pragma unroll
                for (int i = 1; i < K; ++i) {
                    if (buf_vals[i] < cur_min) {
                        cur_min = buf_vals[i];
                        min_pos = i;
                    }
                }
                buf_vals[min_pos] = val;
                buf_idxs[min_pos] = j * VW + v;
                buf_dirty = true;
            }
        }

        if (buf_dirty) {
            buf_min = buf_vals[0];
            #pragma unroll
            for (int i = 1; i < K; ++i) {
                if (buf_vals[i] < buf_min) buf_min = buf_vals[i];
            }
            buf_dirty = false;
        }
    }

    for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
        cmp_t val = static_cast<cmp_t>(row[j]);
        if (val > buf_min) {
            int min_pos = 0;
            cmp_t cur_min = buf_vals[0];
            #pragma unroll
            for (int i = 1; i < K; ++i) {
                if (buf_vals[i] < cur_min) {
                    cur_min = buf_vals[i];
                    min_pos = i;
                }
            }
            buf_vals[min_pos] = val;
            buf_idxs[min_pos] = j;
            buf_dirty = true;
        }
    }

    if (buf_dirty) {
        buf_min = buf_vals[0];
        #pragma unroll
        for (int i = 1; i < K; ++i) {
            if (buf_vals[i] < buf_min) buf_min = buf_vals[i];
        }
    }

    bitonic_sort<cmp_t, K>(buf_vals, buf_idxs);

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        smem_vals[tid_in_block * K + i] = buf_vals[i];
        smem_idxs[tid_in_block * K + i] = buf_idxs[i];
    }
    __syncthreads();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[K]; int ri[K];
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                rv[i] = smem_vals[neighbor * K + i];
                ri[i] = smem_idxs[neighbor * K + i];
            }
            cmp_t mv[K]; int mi[K];
            merge_two_sorted<cmp_t, K>(buf_vals, buf_idxs, rv, ri, mv, mi);
            #pragma unroll
            for (int i = 0; i < K; ++i) { buf_vals[i] = mv[i]; buf_idxs[i] = mi[i]; }
        }
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            smem_vals[tid_in_block * K + i] = buf_vals[i];
            smem_idxs[tid_in_block * K + i] = buf_idxs[i];
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            value[b * K + i] = static_cast<scalar_t>(buf_vals[i]);
            indice[b * K + i] = static_cast<int64_t>(buf_idxs[i]);
        }
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
    constexpr int K = 16;
    constexpr int HALF_K = 8;
    constexpr int VW = VecWidth<scalar_t>::value;
    using cmp_t = typename CmpType<scalar_t>::type;

    __shared__ cmp_t smem_vals[BLOCK_SIZE * K];
    __shared__ int smem_idxs[BLOCK_SIZE * K];

    int b = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    if (b >= batch_size) return;

    int lane = threadIdx.x % WARP_SIZE;
    int tid_in_block = threadIdx.x;
    const scalar_t* row = src + b * dim_size;

    cmp_t neg_inf = safe_neg_inf<cmp_t>();

    if (dim_size <= 64) {
        cmp_t buf[K];
        int idx[K];
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            buf[i] = neg_inf;
            idx[i] = -1;
        }
        cmp_t buf_min = neg_inf;
        bool dirty = false;

        int vec_dim = dim_size / VW;
        for (int j = lane; j < vec_dim; j += WARP_SIZE) {
            scalar_t vals[VW];
            load_vec<scalar_t, VW>(&row[j * VW], vals);
            #pragma unroll
            for (int v = 0; v < VW; ++v) {
                cmp_t val = static_cast<cmp_t>(vals[v]);
                if (val > buf_min) {
                    int min_pos = find_argmin_unrolled<cmp_t, K>(buf);
                    buf[min_pos] = val;
                    idx[min_pos] = j * VW + v;
                    dirty = true;
                }
            }
            if (dirty) {
                buf_min = buf[0];
                #pragma unroll
                for (int i = 1; i < K; ++i) { if (buf[i] < buf_min) buf_min = buf[i]; }
                dirty = false;
            }
        }
        for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
            cmp_t val = static_cast<cmp_t>(row[j]);
            if (val > buf_min) {
                int min_pos = find_argmin_unrolled<cmp_t, K>(buf);
                buf[min_pos] = val;
                idx[min_pos] = j;
                dirty = true;
            }
        }
        if (dirty) {
            buf_min = buf[0];
            #pragma unroll
            for (int i = 1; i < K; ++i) { if (buf[i] < buf_min) buf_min = buf[i]; }
        }

        bitonic_sort<cmp_t, K>(buf, idx);

        #pragma unroll
        for (int i = 0; i < K; ++i) {
            smem_vals[tid_in_block * K + i] = buf[i];
            smem_idxs[tid_in_block * K + i] = idx[i];
        }
    } else {
        cmp_t buf_a[HALF_K], buf_b[HALF_K];
        int idx_a[HALF_K], idx_b[HALF_K];
        #pragma unroll
        for (int i = 0; i < HALF_K; ++i) {
            buf_a[i] = neg_inf; idx_a[i] = -1;
            buf_b[i] = neg_inf; idx_b[i] = -1;
        }

        cmp_t min_a = neg_inf;
        cmp_t min_b = neg_inf;
        bool dirty_a = false, dirty_b = false;

        int vec_dim = dim_size / VW;

        for (int j = lane; j < vec_dim; j += WARP_SIZE) {
            scalar_t vals[VW];
            load_vec<scalar_t, VW>(&row[j * VW], vals);

            #pragma unroll
            for (int v = 0; v < VW; ++v) {
                cmp_t val = static_cast<cmp_t>(vals[v]);
                int idx_val = j * VW + v;

                if (val > min_a) {
                    int min_pos = find_argmin_unrolled<cmp_t, HALF_K>(buf_a);
                    buf_a[min_pos] = val;
                    idx_a[min_pos] = idx_val;
                    dirty_a = true;
                } else if (val > min_b) {
                    int min_pos = find_argmin_unrolled<cmp_t, HALF_K>(buf_b);
                    buf_b[min_pos] = val;
                    idx_b[min_pos] = idx_val;
                    dirty_b = true;
                }
            }

            if (dirty_a) {
                min_a = buf_a[0];
                #pragma unroll
                for (int i = 1; i < HALF_K; ++i) { if (buf_a[i] < min_a) min_a = buf_a[i]; }
                dirty_a = false;
            }
            if (dirty_b) {
                min_b = buf_b[0];
                #pragma unroll
                for (int i = 1; i < HALF_K; ++i) { if (buf_b[i] < min_b) min_b = buf_b[i]; }
                dirty_b = false;
            }
        }

        for (int j = vec_dim * VW + lane; j < dim_size; j += WARP_SIZE) {
            cmp_t val = static_cast<cmp_t>(row[j]);
            if (val > min_a) {
                int min_pos = find_argmin_unrolled<cmp_t, HALF_K>(buf_a);
                buf_a[min_pos] = val;
                idx_a[min_pos] = j;
                dirty_a = true;
            } else if (val > min_b) {
                int min_pos = find_argmin_unrolled<cmp_t, HALF_K>(buf_b);
                buf_b[min_pos] = val;
                idx_b[min_pos] = j;
                dirty_b = true;
            }
        }

        if (dirty_a) {
            min_a = buf_a[0];
            #pragma unroll
            for (int i = 1; i < HALF_K; ++i) { if (buf_a[i] < min_a) min_a = buf_a[i]; }
        }
        if (dirty_b) {
            min_b = buf_b[0];
            #pragma unroll
            for (int i = 1; i < HALF_K; ++i) { if (buf_b[i] < min_b) min_b = buf_b[i]; }
        }

        bitonic_sort<cmp_t, HALF_K>(buf_a, idx_a);
        bitonic_sort<cmp_t, HALF_K>(buf_b, idx_b);

        cmp_t merged[K];
        int merged_idx[K];
        merge_two_sorted<cmp_t, HALF_K>(buf_a, idx_a, buf_b, idx_b, merged, merged_idx);

        #pragma unroll
        for (int i = 0; i < K; ++i) {
            smem_vals[tid_in_block * K + i] = merged[i];
            smem_idxs[tid_in_block * K + i] = merged_idx[i];
        }
    }

    __syncthreads();

    cmp_t final_vals[K];
    int final_idxs[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        final_vals[i] = smem_vals[tid_in_block * K + i];
        final_idxs[i] = smem_idxs[tid_in_block * K + i];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            int neighbor = tid_in_block + offset;
            cmp_t rv[K]; int ri[K];
            #pragma unroll
            for (int i = 0; i < K; ++i) {
                rv[i] = smem_vals[neighbor * K + i];
                ri[i] = smem_idxs[neighbor * K + i];
            }
            cmp_t mv[K]; int mi[K];
            merge_two_sorted<cmp_t, K>(final_vals, final_idxs, rv, ri, mv, mi);
            #pragma unroll
            for (int i = 0; i < K; ++i) { final_vals[i] = mv[i]; final_idxs[i] = mi[i]; }
        }
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            smem_vals[tid_in_block * K + i] = final_vals[i];
            smem_idxs[tid_in_block * K + i] = final_idxs[i];
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            value[b * K + i] = static_cast<scalar_t>(final_vals[i]);
            indice[b * K + i] = static_cast<int64_t>(final_idxs[i]);
        }
    }
}

template <typename scalar_t>
void launch_topk_kernel_direct(
    const scalar_t* src,
    scalar_t* value,
    int64_t* indice,
    int batch_size,
    int dim_size,
    int k,
    cudaStream_t stream)
{
    int num_blocks = (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    switch (k) {
        case 4:
            topk_kernel_k4<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
            break;
        case 8:
            topk_kernel_k8<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
            break;
        case 16:
            topk_kernel_k16<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
            break;
        default:
            topk_kernel_k16<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                src, value, indice, batch_size, dim_size);
            break;
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
    TORCH_CHECK(k == 2 || k == 4 || k == 8 || k == 16, "Only k=2,4,8,16 are supported");

    auto dim_size = src.size(-1);
    auto batch_size = src.numel() / dim_size;
    int bs_int = static_cast<int>(batch_size);
    int ds_int = static_cast<int>(dim_size);

    cudaStream_t stream = 0;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        src.scalar_type(), "topk_power2_opt", [&] {

            if (k == 2) {
                if (ds_int < K2_DEDICATED_DIM_THRESHOLD) {
                    int num_blocks = (bs_int + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                    topk_kernel_k2_dedicated<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                        src.data_ptr<scalar_t>(),
                        value.data_ptr<scalar_t>(),
                        indice.data_ptr<int64_t>(),
                        bs_int, ds_int);
                } else {
                    int num_blocks = (bs_int + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                    topk_kernel_k2<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    src.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>(),
                    indice.data_ptr<int64_t>(),
                    bs_int, ds_int);
                }
            } else {
                launch_topk_kernel_direct<scalar_t>(
                    src.data_ptr<scalar_t>(),
                    value.data_ptr<scalar_t>(),
                    indice.data_ptr<int64_t>(),
                    bs_int, ds_int,
                    static_cast<int>(k),
                    stream);
            }
        });

    return {value, indice};
}

}  // namespace native
}  // namespace at
