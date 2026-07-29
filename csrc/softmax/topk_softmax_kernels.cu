#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr float LOG2_E = 1.44269504089f; 
#define EXP_FUNC(x) __builtin_amdgcn_exp2f((x) * LOG2_E)

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
 #endif

#ifndef USE_ROCM
    #include <cub/util_type.cuh>
    #include <cub/cub.cuh>
#else
    #include <hipcub/util_type.hpp>
    #include <hipcub/hipcub.hpp>
#endif
#define WARP_SIZE 64
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace at {
namespace native {

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
    T data[N]; 
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB, typename input_t>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const input_t* input, const bool* finished, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData = std::max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData += std::exp((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = std::exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = val;
    }
}

template <int TPB, typename IndType>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    IndType* indices,
    int* source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;

            const int idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            assert(indices[idx] >= 0);
            if (source_rows != nullptr) {
                source_rows[idx] = k_idx * num_rows + block_row;
            }
        }
        __syncthreads();
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT; 
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG; 

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;  

    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;

    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)  
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
    
    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = std::max(thread_max, row_chunk[ii]);
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = std::max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;   
    }

    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;  
     float topk_sum = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        float max_val = row_chunk[0]; 
        int expert = start_col; 
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }
       
        if (thread_group_idx == 0)
        {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert; 
            const bool should_process_row = row_is_active && node_uses_expert; 
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            topk_sum  += max_val;  
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;  
            if (source_rows != nullptr) {
                source_rows[idx] = k_idx * num_rows + thread_row;  
            }
        }

        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;  
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;   

            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }
    if constexpr(Is_renormalize){
         if (thread_group_idx == 0 && row_is_active)
        {
            const float r_topk_sum = 1.0f / (topk_sum + 1e-8f);
            const int idx = k * thread_row;
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                output[idx + k_idx] *= r_topk_sum;
            }
        }
    }
       
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk4GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");


    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    

    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
    const bool row_is_active = finished ? !finished[thread_row] : true;


    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }
    
    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;

    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW);
        int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max_idx, mask, THREADS_PER_ROW);
        if (other_max > thread_max) {
            thread_max = other_max;
            thread_max_idx = other_idx;
        }
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = EXP_FUNC(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / (row_sum);
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    float top_vals[4];
    IndType top_indices[4];

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -1.f; \
        k_expert = -1; \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        _Pragma("unroll") \
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { \
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(k_val, mask, THREADS_PER_ROW); \
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(k_expert, mask, THREADS_PER_ROW); \
            if (other_max > k_val || (other_max == k_val && other_expert < k_expert)) { \
                k_val = other_max; \
                k_expert = other_expert; \
            } \
        } \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert != -1) { \
            int expert_in_thread_col = (k_expert / ELTS_PER_LDG) % THREADS_PER_ROW; \
            if (thread_group_idx == expert_in_thread_col) { \
                int ldg_idx = k_expert / (THREADS_PER_ROW * ELTS_PER_LDG); \
                int elt_idx = k_expert % ELTS_PER_LDG; \
                row_chunk[ldg_idx * ELTS_PER_LDG + elt_idx] = -FLT_MAX; \
            } \
        }

  
    if (thread_group_idx == 0) { 
        top_vals[0] = 1.0 * reciprocal_row_sum; 
        top_indices[0] = thread_max_idx; 
    }
    ERASE_MAX(thread_max_idx);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[1] = max_val; top_indices[1] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[2] = max_val; top_indices[2] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[3] = max_val; top_indices[3] = expert; }

    if (thread_group_idx == 0 && row_is_active)
    {
        float r_topk_sum = 1.0f;
        if constexpr(Is_renormalize){
            const float topk_sum = top_vals[0] + top_vals[1] + top_vals[2] + top_vals[3];
            r_topk_sum = 1.0f / (topk_sum);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr(Is_renormalize){
                top_vals[i] *= r_topk_sum;
            }
            const bool node_uses_expert = top_indices[i] >= start_expert && top_indices[i] < end_expert;
            top_indices[i] = node_uses_expert ? (top_indices[i] - start_expert) : NUM_EXPERTS;
        }

        const int k = 4;
        const int base_idx = k * thread_row;
        *reinterpret_cast<float4*>(&output[base_idx]) = *reinterpret_cast<float4*>(&top_vals[0]);
      
        if (source_rows != nullptr) {
            int source_rows_temp[4];
            source_rows_temp[0] = 0 * num_rows + thread_row;
            source_rows_temp[1] = 1 * num_rows + thread_row;
            source_rows_temp[2] = 2 * num_rows + thread_row;
            source_rows_temp[3] = 3 * num_rows + thread_row;
            *reinterpret_cast<int4*>(&source_rows[base_idx]) = *reinterpret_cast<int4*>(&source_rows_temp[0]);
        }

        for (int i = 0; i < 4; ++i) {
            indices[base_idx + i] = top_indices[i];
        }
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk2GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
    int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows)
    {
        return;
    }

    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    
    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }
    
    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = -FLT_MAX;
    for(int i = 0; i < VPT; ++i) thread_max = std::max(thread_max, row_chunk[i]);

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = std::max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = EXP_FUNC(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / (row_sum);
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    float top_vals[2];
    IndType top_indices[2];
    
    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -1.f; \
        k_expert = -1; \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        _Pragma("unroll") \
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { \
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(k_val, mask, THREADS_PER_ROW); \
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(k_expert, mask, THREADS_PER_ROW); \
            if (other_max > k_val || (other_max == k_val && other_expert < k_expert)) { \
                k_val = other_max; \
                k_expert = other_expert; \
            } \
        } \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert != -1) { \
            int expert_in_thread_col = (k_expert / ELTS_PER_LDG) % THREADS_PER_ROW; \
            if (thread_group_idx == expert_in_thread_col) { \
                int ldg_idx = k_expert / (THREADS_PER_ROW * ELTS_PER_LDG); \
                int elt_idx = k_expert % ELTS_PER_LDG; \
                row_chunk[ldg_idx * ELTS_PER_LDG + elt_idx] = -100000.f; \
            } \
        }

    float max_val;
    int expert;

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[0] = max_val; top_indices[0] = expert; }
    ERASE_MAX(expert);

    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[1] = max_val; top_indices[1] = expert; }

    if (thread_group_idx == 0 && row_is_active)
    {
        float r_topk_sum = 1.0f;
         if constexpr(Is_renormalize){
            const float topk_sum = top_vals[0] + top_vals[1];
            r_topk_sum = 1.0f / (topk_sum);
         }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if constexpr(Is_renormalize){
                 top_vals[i] *= r_topk_sum;
            }
            const bool node_uses_expert = top_indices[i] >= start_expert && top_indices[i] < end_expert;
            top_indices[i] = node_uses_expert ? (top_indices[i] - start_expert) : NUM_EXPERTS;
        }

        const int k = 2;
        const int base_idx = k * thread_row;
        *reinterpret_cast<float2*>(&output[base_idx]) = *reinterpret_cast<float2*>(&top_vals[0]);
        if (source_rows != nullptr) {
            int source_rows_temp[2];
            source_rows_temp[0] = 0 * num_rows + thread_row;
            source_rows_temp[1] = 1 * num_rows + thread_row;
            *reinterpret_cast<int2*>(&source_rows[base_idx]) = *reinterpret_cast<int2*>(&source_rows_temp[0]);
        }

        if constexpr (std::is_same_v<IndType, int>) {
            *reinterpret_cast<int2*>(&indices[base_idx]) = *reinterpret_cast<int2*>(&top_indices[0]);
        } else { 
            for (int i = 0; i < 2; ++i) {
                indices[base_idx + i] = top_indices[i];
            }
        }
    }
}

__device__ __forceinline__ uint32_t topk_half_key(
    uint16_t half_bits, int expert) {
    const uint32_t ordered =
        (half_bits & 0x8000u)
            ? ((~static_cast<uint32_t>(half_bits)) & 0xffffu)
            : (static_cast<uint32_t>(half_bits) ^ 0x8000u);
    return (ordered << 16) | (0xffffu - static_cast<uint32_t>(expert));
}

__device__ __forceinline__ float topk_half_key_value(uint32_t key) {
    const uint16_t ordered = static_cast<uint16_t>(key >> 16);
    const uint16_t half_bits =
        (ordered & 0x8000u)
            ? static_cast<uint16_t>(ordered ^ 0x8000u)
            : static_cast<uint16_t>(~ordered);
    c10::Half value;
    value.x = half_bits;
    return static_cast<float>(value);
}

template <int DppCtrl>
__device__ __forceinline__ uint32_t topk_mov_dpp(uint32_t value) {
    return static_cast<uint32_t>(
        __builtin_amdgcn_mov_dpp(
            value, DppCtrl, 0xf, 0xf, true));
}

template <int kThreadsPerRow>
__launch_bounds__(WARP_SIZE) __global__
void topk8GatingSoftmax_256_half_packed(
    const c10::Half* input,
    const bool* finished,
    float* output,
    const int num_rows,
    int* indices,
    int* source_rows,
    const int start_expert,
    const int end_expert)
{
    constexpr int kExperts = 256;
    constexpr int kTopK = 8;
    constexpr int kValuesPerThread =
        kExperts / kThreadsPerRow;
    constexpr int kRowsPerWave = WARP_SIZE / kThreadsPerRow;

    const int row_in_wave = threadIdx.x / kThreadsPerRow;
    const int lane_in_row = threadIdx.x % kThreadsPerRow;
    const int row = blockIdx.x * kRowsPerWave + row_in_wave;
    if (row >= num_rows || (finished != nullptr && finished[row])) {
        return;
    }

    const int first_expert = lane_in_row * kValuesPerThread;
    const uint16_t* row_ptr = reinterpret_cast<const uint16_t*>(
        input + static_cast<int64_t>(row) * kExperts);
    uint16_t packed_values[kValuesPerThread];
    using LoadType = AlignedArray<uint16_t, kValuesPerThread>;
    *reinterpret_cast<LoadType*>(packed_values) =
        *reinterpret_cast<const LoadType*>(row_ptr + first_expert);

    uint32_t keys[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        keys[i] = topk_half_key(
            packed_values[i], first_expert + i);
    }

    uint32_t stored_key = 0;
    float row_max = -FLT_MAX;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        uint32_t winner_key = keys[0];
#pragma unroll
        for (int i = 1; i < kValuesPerThread; ++i) {
            winner_key = keys[i] > winner_key ? keys[i] : winner_key;
        }
        uint32_t other_key = topk_mov_dpp<0xb1>(winner_key);
        winner_key =
            other_key > winner_key ? other_key : winner_key;
        other_key = topk_mov_dpp<0x4e>(winner_key);
        winner_key =
            other_key > winner_key ? other_key : winner_key;
        other_key = VLLM_SHFL_XOR_SYNC_WIDTH(
            winner_key, 4, kThreadsPerRow);
        winner_key =
            other_key > winner_key ? other_key : winner_key;
        other_key = topk_mov_dpp<0x128>(winner_key);
        winner_key =
            other_key > winner_key ? other_key : winner_key;
        if constexpr (kThreadsPerRow > 16) {
            other_key = VLLM_SHFL_XOR_SYNC_WIDTH(
                winner_key, 16, kThreadsPerRow);
            winner_key =
                other_key > winner_key ? other_key : winner_key;
        }

        if (rank == 0) {
            row_max = topk_half_key_value(winner_key);
        }
        if (lane_in_row == rank) {
            stored_key = winner_key;
        }
        if (rank + 1 < kTopK) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                if (keys[i] == winner_key) {
                    keys[i] = 0;
                }
            }
        }
    }

    float value_exp = 0.0f;
    if (lane_in_row < kTopK) {
        value_exp =
            EXP_FUNC(topk_half_key_value(stored_key) - row_max);
    }
    float topk_sum = value_exp;
#pragma unroll
    for (int mask = kTopK / 2; mask > 0; mask >>= 1) {
        topk_sum +=
            VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum, mask, kThreadsPerRow);
    }

    if (lane_in_row < kTopK) {
        int top_expert =
            0xffff - static_cast<int>(stored_key & 0xffffu);
        const bool node_uses_expert =
            top_expert >= start_expert && top_expert < end_expert;
        top_expert =
            node_uses_expert ? top_expert - start_expert : kExperts;
        const int output_offset = row * kTopK + lane_in_row;
        output[output_offset] =
            value_exp * (1.0f / (topk_sum + 1e-6f));
        indices[output_offset] = top_expert;
        if (source_rows != nullptr) {
            source_rows[output_offset] =
                lane_in_row * num_rows + row;
        }
    }
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk8GatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
 
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
#pragma unroll
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW);
        int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max_idx, mask, THREADS_PER_ROW);
        bool swap = (other_max > thread_max) ||
                    (other_max == thread_max && other_idx < thread_max_idx);
        thread_max = swap ? other_max : thread_max;
        thread_max_idx = swap ? other_idx : thread_max_idx;
    }

 
    float row_sum = 0;
    float reciprocal_row_sum = 0.0f;

   
    if constexpr (!Is_renormalize) {
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii)
        {
            row_chunk[ii] = EXP_FUNC(row_chunk[ii] - thread_max);
            row_sum += row_chunk[ii];
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
        }
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f);
    }

    
    float top_vals = 0.f;
    IndType top_indices = IndType(0);

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        _Pragma("unroll") \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            _Pragma("unroll") \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = first_elt_read_by_thread + ldg * THREADS_PER_ROW * ELTS_PER_LDG + ii; \
                } \
            } \
        } \
        _Pragma("unroll") \
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { \
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(k_val, mask, THREADS_PER_ROW); \
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(k_expert, mask, THREADS_PER_ROW); \
            bool swap = (other_max > k_val) || \
                        (other_max == k_val && other_expert < k_expert); \
            k_val = swap ? other_max : k_val; \
            k_expert = swap ? other_expert : k_expert; \
        } \
    } while(0)

    #define ERASE_MAX(k_expert_to_erase) \
    do { \
        _Pragma("unroll") \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            _Pragma("unroll") \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                int col_idx = first_elt_read_by_thread + ldg * THREADS_PER_ROW * ELTS_PER_LDG + ii; \
                if (col_idx == k_expert_to_erase) { \
                    row_chunk[ldg * ELTS_PER_LDG + ii] = -FLT_MAX; \
                } \
            } \
        } \
    } while(0)

    {
        constexpr int i = 0;
        if (thread_group_idx == i) { 
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx; 
        }
        ERASE_MAX(thread_max_idx); 
    }
    {
        constexpr int i = 1;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 2;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 3;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 4;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 5;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 6;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 7;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
    }

    if(row_is_active){
        float final_scale = 1.0f; 

        if constexpr(Is_renormalize){
            float val_exp = 0.0f;
            if (thread_group_idx < 8) {
                val_exp = EXP_FUNC(top_vals - thread_max);
                top_vals = val_exp; 
            }
            float topk_sum = val_exp;
    #pragma unroll
            for(int i = 4; i > 0; i >>= 1) {
                topk_sum += VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum,i,THREADS_PER_ROW);
            }
            final_scale = 1.0f / (topk_sum + 1e-6f);
        } else {
             final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < 8){
            top_vals *= final_scale;
            
            const bool node_uses_expert = top_indices >= start_expert && top_indices  < end_expert;
            top_indices = node_uses_expert ? (top_indices - start_expert) : NUM_EXPERTS;
            const int base_idx = 8 * thread_row + thread_group_idx;
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals;
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk8GatingSoftmax_no_align(
    const input_t* input, 
    const bool* finished, 
    float* output,   
    const int num_rows, 
    IndType* indices,
    int* source_rows, 
    const int start_expert, 
    const int end_expert, 
    const int model_num_experts) 
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); 
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    
    float max_val;
    int expert;
    int my_search_expert_idx = -1; 
    int my_search_reg_idx = -1;   

    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + thread_row * model_num_experts;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    float row_chunk[VPT];
    
    #pragma unroll
    for (int i=0; i<VPT; ++i) row_chunk[i] = -FLT_MAX;

    #pragma unroll
    for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
        int offset = ldg * THREADS_PER_ROW * ELTS_PER_LDG;
        int col_idx = thread_group_idx * ELTS_PER_LDG + offset;
        if (col_idx < model_num_experts) {
            #pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                if (col_idx + ii < model_num_experts) {
                    row_chunk[ldg * ELTS_PER_LDG + ii] = static_cast<float>(thread_read_ptr[offset + ii]);
                }
            }
        }
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
    
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
            
            my_search_reg_idx = i;
            my_search_expert_idx = thread_max_idx;
        }
    }

    #pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW);
        int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max_idx, mask, THREADS_PER_ROW);
        bool swap = (other_max > thread_max);
        thread_max = swap ? other_max : thread_max;
        thread_max_idx = swap ? other_idx : thread_max_idx;
    }

    float row_sum = 0.0f;
    float reciprocal_row_sum = 0.0f; 

    if constexpr(!Is_renormalize){
        #pragma unroll
        for (int ii = 0; ii < VPT; ++ii)
        {
            row_chunk[ii] = exp2f((row_chunk[ii] - thread_max)* LOG2_E); 
            row_sum += row_chunk[ii];
        }

        #pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
        }
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f); 
    }
   
    float top_vals = 0.f; 
    IndType top_indices = IndType(0);

    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        my_search_reg_idx = -1; \
        int current_col_base = first_elt_read_by_thread; \
        for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                    my_search_reg_idx = ldg * ELTS_PER_LDG + ii; \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        my_search_expert_idx = k_expert; \
        _Pragma("unroll") \
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { \
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(k_val, mask, THREADS_PER_ROW); \
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(k_expert, mask, THREADS_PER_ROW); \
            bool swap = (other_max > k_val); \
            k_val = swap ? other_max : k_val; \
            k_expert = swap ? other_expert : k_expert; \
        } \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert == my_search_expert_idx) { \
            row_chunk[my_search_reg_idx] = -FLT_MAX; \
        }

    {
        constexpr int i = 0;
        if (thread_group_idx == i) { 
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx; 
        }
        ERASE_MAX(thread_max_idx); 
    }
    {
        constexpr int i = 1;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 2;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 3;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 4;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 5;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 6;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
        ERASE_MAX(expert); 
    }
    {
        constexpr int i = 7;
        FIND_AND_REDUCE_MAX(max_val, expert);
        if (thread_group_idx == i) { 
            top_vals = max_val; 
            top_indices = expert; 
        }
    }

    if(row_is_active){
        float final_scale = 1.0f;
        
        if constexpr(Is_renormalize){
            // 此时 top_vals 还是原始 Logits，需要转为 Exp 值
            float val_exp = 0.0f;
            if (thread_group_idx < 8) {
                val_exp = exp2f((top_vals - thread_max) * LOG2_E);
                top_vals = val_exp; 
            }
            
            // 计算局部 TopK Sum
            float topk_sum = val_exp;
            #pragma unroll
            for(int i = 4; i > 0; i >>= 1) {
                topk_sum += VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum, i, THREADS_PER_ROW);
            }
            final_scale = 1.0f / (topk_sum + 1e-6f); 
        } else {
            final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < 8){
            top_vals *= final_scale;

            const bool node_uses_expert = top_indices >= start_expert && top_indices < model_num_experts;
            top_indices = node_uses_expert ? (top_indices - start_expert) : model_num_experts;

            const int base_idx = 8 * thread_row + thread_group_idx;
            
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals; 
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType, typename input_t>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topkGatingSoftmax_opt(const input_t* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert, const int topk)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t); // [修改点]
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    
    if (thread_row >= num_rows)
    {
        return;
    }
    float max_val;
    int expert;
 
    int my_search_expert_idx = -1; 
    int my_search_reg_idx = -1;
    
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const input_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const input_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<input_t, ELTS_PER_LDG>;
    input_t temp_chunk[VPT];
    AccessType* temp_chunk_vec_ptr = reinterpret_cast<AccessType*>(&temp_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        temp_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for(int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = static_cast<float>(temp_chunk[ii]);
    }

    float thread_max = -FLT_MAX;
    int thread_max_idx = -1;
#pragma unroll
    for(int i = 0; i < VPT; ++i) {
        if (row_chunk[i] > thread_max) {
            thread_max = row_chunk[i];
            int ldg_idx = i / ELTS_PER_LDG;
            int elt_idx_in_ldg = i % ELTS_PER_LDG;
            thread_max_idx = first_elt_read_by_thread + ldg_idx * (THREADS_PER_ROW * ELTS_PER_LDG) + elt_idx_in_ldg;
        
            my_search_reg_idx = i;
            my_search_expert_idx = thread_max_idx;
        }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW);
        int other_idx = VLLM_SHFL_XOR_SYNC_WIDTH(thread_max_idx, mask, THREADS_PER_ROW);
        bool swap = (other_max > thread_max);
        thread_max = swap ? other_max : thread_max;
        thread_max_idx = swap ? other_idx : thread_max_idx;
    }

 
    float row_sum = 0;
    float reciprocal_row_sum = 0.0f;

    if constexpr (!Is_renormalize) {
#pragma unroll
        for (int ii = 0; ii < VPT; ++ii)
        {
            row_chunk[ii] = EXP_FUNC(row_chunk[ii] - thread_max);
            row_sum += row_chunk[ii];
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
        }
        reciprocal_row_sum = 1.f / (row_sum + 1e-6f);
    }

    
    float top_vals = 0.f;
    IndType top_indices = IndType(0);

    
    #define FIND_AND_REDUCE_MAX(k_val, k_expert) \
    do { \
        k_val = -FLT_MAX; \
        k_expert = -1; \
        my_search_reg_idx = -1;  \
        int current_col_base = first_elt_read_by_thread; \
       for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) { \
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) { \
                float val = row_chunk[ldg * ELTS_PER_LDG + ii]; \
                if (val > k_val) { \
                    k_val = val; \
                    k_expert = current_col_base + ii; \
                    my_search_reg_idx = ldg * ELTS_PER_LDG + ii;  \
                } \
            } \
            current_col_base += THREADS_PER_ROW * ELTS_PER_LDG; \
        } \
        my_search_expert_idx = k_expert; \
        _Pragma("unroll") \
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) { \
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(k_val, mask, THREADS_PER_ROW); \
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(k_expert, mask, THREADS_PER_ROW); \
            bool swap = (other_max > k_val);  \
            k_val = swap ? other_max : k_val; \
            k_expert = swap ? other_expert : k_expert; \
        } \
    } while(0)

    #define ERASE_MAX(k_expert) \
        if (k_expert == my_search_expert_idx) { \
            row_chunk[my_search_reg_idx] = -FLT_MAX; \
        }

    //top1
    {
       
        if (thread_group_idx == 0) {
            if constexpr (Is_renormalize) {
                top_vals = thread_max; 
            } else {
                top_vals = 1.0f;       
            }
            top_indices = thread_max_idx;
        }
   
        ERASE_MAX(thread_max_idx);
    }

    for (int k = 1; k < topk; ++k) {
        float current_max_val;
        int current_expert;
        FIND_AND_REDUCE_MAX(current_max_val, current_expert);
        if (thread_group_idx == k) {
    
            top_vals = current_max_val;
            top_indices = current_expert;
        }

        ERASE_MAX(current_expert);
    }

    if(row_is_active){
        float final_scale = 1.0f; 

        if constexpr(Is_renormalize){
            float val_exp = 0.0f;
            if (thread_group_idx < topk) {
                val_exp = EXP_FUNC(top_vals - thread_max);
                top_vals = val_exp; 
            }
            float topk_sum = val_exp;
            #pragma unroll
            for(int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
                topk_sum += VLLM_SHFL_XOR_SYNC_WIDTH(topk_sum, mask, THREADS_PER_ROW);
            }
            final_scale = 1.0f / (topk_sum + 1e-6f);
        } else {
             final_scale = reciprocal_row_sum;
        }

        if(thread_group_idx < topk){
            top_vals *= final_scale;
            const bool node_uses_expert = top_indices >= start_expert && top_indices  < end_expert;
            top_indices = node_uses_expert ? (top_indices - start_expert) : NUM_EXPERTS;
            const int base_idx = topk * thread_row + thread_group_idx;
            indices[base_idx] = top_indices;
            output[base_idx] = top_vals;
            
            if (source_rows != nullptr) {
                source_rows[base_idx] = thread_group_idx * num_rows + thread_row;
            }
        }
    }
}
namespace detail
{
template <int EXPERTS, int BYTES_PER_LDG, typename input_t>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(input_t);  
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");

    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;  
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT; 
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, typename IndType, bool Is_renormalize, typename input_t>
void topkGatingSoftmaxLauncherHelper(const input_t* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream,const int num_experts)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 32; 
    static constexpr int TARGET_THREADS_PER_ROW = (EXPERTS >= 8) ? 8 : EXPERTS;
    static constexpr int MAX_BYTES_FOR_THREADS = (EXPERTS * sizeof(input_t)) / TARGET_THREADS_PER_ROW;
    static constexpr int BYTES_PER_LDG = MIN((int)MAX_BYTES_PER_LDG, MAX_BYTES_FOR_THREADS);
    
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, input_t>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB / 1;
    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
 

    if(k == 4){
     topk4GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
     
    }else if(k == 2){
        topk2GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
    }else if(k == 8){
        if(num_experts == 160){
            topk8GatingSoftmax_no_align<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert, num_experts);
        }else if constexpr (
            EXPERTS == 256 &&
            WARPS_PER_TB == 1 &&
            Is_renormalize &&
            std::is_same_v<IndType, int> &&
            std::is_same_v<input_t, c10::Half>) {
            if (num_rows <= 512) {
                topk8GatingSoftmax_256_half_packed<32><<<
                    (num_rows + 1) / 2, WARP_SIZE, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row,
                    start_expert, end_expert);
            } else {
                topk8GatingSoftmax_256_half_packed<16><<<
                    (num_rows + 3) / 4, WARP_SIZE, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row,
                    start_expert, end_expert);
            }
        }else{
            topk8GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
        }
    }
    else{
        topkGatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize, IndType, input_t><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert,k);
    }
}

template <typename IndType, typename input_t>
void topkGatingSoftmaxKernelLauncher(
    const input_t* gating_output,
    float* topk_weights,
    IndType* topk_indicies,
    int* token_expert_indices,
    float* softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    cudaStream_t stream,
    bool Is_renormalize) { 

    auto launch_kernel = [&](auto is_renormalize_flag) {
        static constexpr bool Is_renormalize = decltype(is_renormalize_flag)::value;
        static constexpr int WARPS_PER_TB = 4;
        #define LAUNCH_SOFTMAX(NUM_EXPERTS, WARPS_PER_TB) \
            topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, IndType, Is_renormalize, input_t>( \
                gating_output, nullptr, topk_weights, topk_indicies, \
                token_expert_indices, num_tokens, topk, 0, num_experts, \
                stream,num_experts);

        switch (num_experts) {
            case 1: LAUNCH_SOFTMAX(1, WARPS_PER_TB); break;
            case 2: LAUNCH_SOFTMAX(2, WARPS_PER_TB); break;
            case 4: LAUNCH_SOFTMAX(4, WARPS_PER_TB); break;
            case 8: LAUNCH_SOFTMAX(8, WARPS_PER_TB); break;
            case 16: LAUNCH_SOFTMAX(16, WARPS_PER_TB); break;
            case 32: LAUNCH_SOFTMAX(32, WARPS_PER_TB); break;
            /* 字节 */
            case 64:
                if(num_tokens < 16384) { LAUNCH_SOFTMAX(64, WARPS_PER_TB); }
                else { LAUNCH_SOFTMAX(64, 1); }
                break;
            /* qwen 30b*/
            case 128: {
                LAUNCH_SOFTMAX(128, 2); 
                break;
            }
            case 160:                         //160使用256的模板
            case 256: LAUNCH_SOFTMAX(256, 1); break;
            case 512: LAUNCH_SOFTMAX(512, 2); break;
            default: {
            TORCH_CHECK(softmax_workspace != nullptr,
                "softmax_workspace must be provided for num_experts that are not a power of 2.");
            static constexpr int TPB = 256;
            moeSoftmax<TPB, input_t><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, softmax_workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, nullptr, topk_weights, topk_indicies, token_expert_indices,
                num_experts, topk, 0, num_experts);
            }
        }
        #undef LAUNCH_SOFTMAX 
    };
    if (Is_renormalize) {
        launch_kernel(std::true_type{});
    } else {
        launch_kernel(std::false_type{});
    }
}

void topk_softmax(
    torch::Tensor& topk_weights,                // [num_tokens, topk]
    torch::Tensor& topk_indices,                // [num_tokens, topk]
    torch::optional<torch::Tensor>& token_expert_indices, // [num_tokens, topk]
    torch::Tensor& gating_output,
    bool Is_renormalize = false)               // [num_tokens, num_experts]
{

    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float, 
                "topk_weights must be Float32 (float)");
    TORCH_CHECK(gating_output.scalar_type() == at::ScalarType::Float || 
                gating_output.scalar_type() == at::ScalarType::Half ||
                gating_output.scalar_type() == at::ScalarType::BFloat16,
                "gating_output must be Float32, Float16 or BFloat16");

    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = (!is_pow_2 && num_experts != 160) || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, gating_output.options().dtype(at::kFloat));
   
    int* token_expert_indices_ptr = nullptr;
    if (token_expert_indices.has_value() && token_expert_indices.value().defined()) {
        token_expert_indices_ptr = token_expert_indices.value().data_ptr<int>();
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gating_output.scalar_type(), "topk_softmax_cuda", [&] {
        if(topk_indices.scalar_type() == at::ScalarType::Int)
        {
           topkGatingSoftmaxKernelLauncher<int, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<int>(),
                token_expert_indices_ptr, 
                softmax_workspace.data_ptr<float>(),
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
        else if (topk_indices.scalar_type() == at::ScalarType::UInt32)
        {
           topkGatingSoftmaxKernelLauncher<uint32_t, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<uint32_t>(),
                token_expert_indices_ptr, 
                softmax_workspace.data_ptr<float>(),
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
        else {
            assert(topk_indices.scalar_type() == at::ScalarType::UInt64);
           topkGatingSoftmaxKernelLauncher<int64_t, scalar_t>(
                gating_output.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_indices.data_ptr<int64_t>(),
                token_expert_indices_ptr, 
                softmax_workspace.data_ptr<float>(),
                num_tokens,
                num_experts,
                topk,
                stream,
            Is_renormalize);
        }
    });
}

} // namespace native
} // namespace at
