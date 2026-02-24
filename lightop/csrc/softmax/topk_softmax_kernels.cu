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
    float data[N];
};

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const float* input, const bool* finished, float* output, const int num_cols)
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


template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int k, const int start_expert, const int end_expert)
{
      // 我们首先强制执行编译时断言并设置编译时常量。
      //VPT == (VPT & -VPT) 是一个非常巧妙的位运算技巧，用于在编译时判断一个正整数（这里是 VPT）是否为 2 的幂
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
   // static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

        // 每个线程每次加载的字节数
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT; // 计算处理一行数据需要的线程数。
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG; // 计算每个线程需要执行的加载操作次数。

   // 基于上一节的限制。
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

     // 每行有 NUM_EXPERTS 个元素。我们针对少量 expert 的情况进行特化。
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    // 计算每个 warp 能处理的行数。
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
     // 计算每个block 能处理的行数。
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    //定位到当前block处理的行号
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // 定位到当前warp处理的行号
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    //定位到当前子warp
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    //每个线程子组计算行偏移
    const int thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;  // 判断当前行是否需要处理。

    //每个线程跳转到它将要读取的行的开头。
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // 偏移到要处理的列处
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // 使用我们自己定义的 AlignedArray 作为向量化加载的类型。
    using AccessType = AlignedArray<float, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    float row_chunk[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)  // 循环执行向量化加载。每个线程要load LDG_PER_THREAD次
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // 首先，我们在线程内执行 std::max 规约。我们可以安全地在 fp16 中进行 std::max 操作（我认为），
    // 然后再转换为 float 进行 std::exp + sum 规约。
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = std::max(thread_max, row_chunk[ii]);
    }

//线程组内max规约
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = std::max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

   // 从这一点开始，所有线程中的 thread_max 都拥有行内的最大值。
    // 现在，我们从线程中的每个元素中减去最大值并取指数。我们还计算线程局部和。
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// 现在，我们在每个线程组内执行 sum 规约。与 std::max 规约类似，我们使用蝶式模式。
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

     // 从这一点开始，所有线程分别在 thread_max 和 row_sum 变量中拥有其行的最大值和总和。
    // 最后，我们可以缩放行以进行 softmax。技术上讲，对于 top-k 门控，我们不需要计算整个 softmax 行。
    // 我们可以查看最大值，只计算行中 top-k 的值。
    // 然而，这个内核可能不会是瓶颈，而且在计算 softmax 后再找 argmax 似乎能更好地匹配 torch 的行为。
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;   //计算最终的 softmax 值。
    }

     // 现在，row_chunk 包含了行块的 softmax 结果。现在，我想在每行中找到 topk 元素及其最大索引。
    int start_col = first_elt_read_by_thread;
    //m
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;  //// 计每个加载组覆盖的列数。
    //整体逻辑，先找自己得局部最大值，然后和别得线程对比，获胜得线程将局部最大值设为-1，进行下一轮
     float topk_sum = 0.0f;
    //float output_temp[4];
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        //每个线程局部 Argmax
        float max_val = row_chunk[0]; // 初始化线程内的最大值为第一个元素。
        int expert = start_col; // 初始化对应的 expert 索引。
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }
        //到这里每个线程找到了自己的局部最大值，和局部最大值得索引expert
// 现在，我们执行 argmax 规约。我们使用蝶式模式，以便线程就最大值达成共识。
// 这对于 K > 1 很有用，这样线程可以就“谁”拥有最大值达成一致。那个线程可以
// 然后将其最大值置为-inf，warp 可以运行更多迭代...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            //优化点，减少一次shfl
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // 如果另一个值更大，或者值相等但索引更小。获胜
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }
       
          // Write the std::max for this k iteration to global memory.
        // 将此次 k 迭代的最大值写入全局内存。
        if (thread_group_idx == 0)
        {
              // 添加一个守卫，以忽略此节点不包含的 expert  start_expert = 0 end_expert = num_experts
            const bool node_uses_expert = expert >= start_expert && expert < end_expert; // 检查该 expert 是否由当前节点负责。
            const bool should_process_row = row_is_active && node_uses_expert; // 判断是否应该处理这一行。

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            topk_sum  += max_val;  // 写入 Top-K 的概率值。
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;  // 如果应处理，则写入相对索引；否则写入一个无效索引。
            
            if (source_rows != nullptr) {
                source_rows[idx] = k_idx * num_rows + thread_row;  // 记录这个结果对应的原始行索引。
            }
        }

       // 最后，如果还有下一次迭代要运行，我们清除当前拥有最大值的线程中的该值。
        if (k_idx + 1 < k)
        {
            //expert是专家得列索引， COLS_PER_GROUP_LDG = (单个线程一次加载的元素数) × (小组内的线程总数)，可能不是一次加载
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;  // 计算获胜 expert 所在的加载组。
            //计算持有该 expert 的线程在其小组内的索引。
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;   // 计算持有该 expert 的线程在其小组内的索引。

              // 只有产生最大值的组中的那个线程才会将“获胜”值重置为-inf。
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }
    if constexpr(Is_renormalize){
         if (thread_group_idx == 0 && row_is_active)
        {
            const float r_topk_sum = 1.0f / (topk_sum + 1e-8f);
            const int idx = k * thread_row;
            // 3. Multiply each weight by the reciprocal and write back.
            for (int k_idx = 0; k_idx < k; ++k_idx)
            {
                output[idx + k_idx] *= r_topk_sum;
            }
            
           // *reinterpret_cast<float4 *>(&output[idx]) =  *reinterpret_cast<float4 *>(&output_temp[0]); 
        }
    }
       
}

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk4GatingSoftmax_opt(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");


    static constexpr int ELTS_PER_LDG = 4;//BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    // static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    // static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    // static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

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

    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<float, ELTS_PER_LDG>;
    float row_chunk[VPT];

#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii] = thread_read_ptr[ii];
    }
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii+4] = thread_read_ptr[ii+THREADS_PER_ROW*ELTS_PER_LDG];
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
      //  thread_max = std::max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
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

        // if constexpr (std::is_same_v<IndType, int>) {
        //     *reinterpret_cast<int4*>(&indices[base_idx]) = *reinterpret_cast<int4*>(&top_indices[0]);
        // } else { 
            for (int i = 0; i < 4; ++i) {
                indices[base_idx + i] = top_indices[i];
            }
       // }
    }
}
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk2GatingSoftmax_opt(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
    int* source_rows, const int start_expert, const int end_expert)
{
    // --- Static assertions and initial setup remain unchanged ---
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
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

    // --- Data loading and softmax calculation remain unchanged ---
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<float, ELTS_PER_LDG>;
    float row_chunk[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
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

    // --- Top-K logic modified for k=2 ---
    float top_vals[2];
    IndType top_indices[2];
    
    // The helper macros are general and do not need to be changed.
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

    // Find top-1
    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[0] = max_val; top_indices[0] = expert; }
    ERASE_MAX(expert);

    // Find top-2
    FIND_AND_REDUCE_MAX(max_val, expert);
    if (thread_group_idx == 0) { top_vals[1] = max_val; top_indices[1] = expert; }
    // No need to erase after finding the last one

    // --- Final normalization and write-back modified for k=2 ---
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
        // Use float2 for vectorized write of 2 floats
        *reinterpret_cast<float2*>(&output[base_idx]) = *reinterpret_cast<float2*>(&top_vals[0]);

        if (source_rows != nullptr) {
            int source_rows_temp[2];
            source_rows_temp[0] = 0 * num_rows + thread_row;
            source_rows_temp[1] = 1 * num_rows + thread_row;
            // Use int2 for vectorized write of 2 ints
            *reinterpret_cast<int2*>(&source_rows[base_idx]) = *reinterpret_cast<int2*>(&source_rows_temp[0]);
        }

        if constexpr (std::is_same_v<IndType, int>) {
            // Use int2 for vectorized write if IndType is int
            *reinterpret_cast<int2*>(&indices[base_idx]) = *reinterpret_cast<int2*>(&top_indices[0]);
        } else { 
            for (int i = 0; i < 2; ++i) {
                indices[base_idx + i] = top_indices[i];
            }
        }
    }
}
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk8GatingSoftmax_opt(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = 4;
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

    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<float, ELTS_PER_LDG>;
    float row_chunk[VPT];
    // AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk); // Unused in original snippet
    // const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr); // Unused

    // Load data
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii] = thread_read_ptr[ii];
    }
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii+4] = thread_read_ptr[ii+THREADS_PER_ROW*ELTS_PER_LDG];
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

    //第一次复用softmax
    {
        constexpr int i = 0;
        if (thread_group_idx == i) { 
            if constexpr (Is_renormalize) {
                top_vals = thread_max; // 原始 Logits
            } else {
                top_vals = 1.0f;       // Exp 值
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
    //最后一次不擦除
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

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG, bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topk8GatingSoftmax_no_align(
    const float* input, 
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

    static constexpr int ELTS_PER_LDG = 4;
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

    const float* thread_row_ptr = input + thread_row * model_num_experts;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<float, ELTS_PER_LDG>;
    float row_chunk[VPT];
    
    #pragma unroll
    for (int i=0; i<VPT; ++i) row_chunk[i] = -FLT_MAX;

    int col_idx_1 = thread_group_idx * ELTS_PER_LDG;
    if (col_idx_1 < model_num_experts) {
        #pragma unroll
        for (int ii = 0; ii < 4; ++ii)
        {
            if (col_idx_1 + ii < model_num_experts) 
                row_chunk[ii] = thread_read_ptr[ii];
        }
    }

    int offset = THREADS_PER_ROW * ELTS_PER_LDG;
    int col_idx_2 = col_idx_1 + offset;
    if (col_idx_2 < model_num_experts) {
        #pragma unroll
        for (int ii = 0; ii < 4; ++ii)
        {
            if (col_idx_2 + ii < model_num_experts)
                row_chunk[ii+4] = thread_read_ptr[ii+offset];
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
        // bool swap = (other_max > thread_max) | ((other_max == thread_max) & (other_idx < thread_max_idx));
        thread_max = swap ? other_max : thread_max;
        thread_max_idx = swap ? other_idx : thread_max_idx;
    }

    // Softmax Sum Reduction
    float row_sum = 0.0f;
    float reciprocal_row_sum = 0.0f; 

    // Is_renormalize 时跳过softmax
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
            // 修复后
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

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,bool Is_renormalize, typename IndType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
void topkGatingSoftmax_opt(const float* input, const bool* finished, float* output, const int num_rows, IndType* indices,
        int* source_rows, const int start_expert, const int end_expert, const int topk)
{
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");

    static constexpr int ELTS_PER_LDG = 4;
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

    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // using AccessType = AlignedArray<float, ELTS_PER_LDG>; // Unused
    float row_chunk[VPT];

    // Load data
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii] = thread_read_ptr[ii];
    }
#pragma unroll
    for (int ii = 0; ii < 4; ++ii)
    {
        row_chunk[ii+4] = thread_read_ptr[ii+THREADS_PER_ROW*ELTS_PER_LDG];
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
                top_vals = thread_max; // 原始 Logits
            } else {
                top_vals = 1.0f;       // Exp 值 / 占位
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
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
   
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);  
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");

    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;  
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT; 
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, typename IndType,bool Is_renormalize>
void topkGatingSoftmaxLauncherHelper(const float* input, const bool* finished, float* output, IndType* indices,
    int* source_row, const int num_rows, const int k, const int start_expert, const int end_expert, cudaStream_t stream,const int num_experts)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 32; //每次32字节，8个float
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB / 1;
    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
 

    if(k == 4){
     topk4GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
     
    }else if(k == 2){
        topk2GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
    }else if(k == 8){
        if(num_experts == 160){
            topk8GatingSoftmax_no_align<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert, num_experts);
        }else{
            topk8GatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert);
        }
    }
    else{
        topkGatingSoftmax_opt<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,Is_renormalize><<<num_blocks, block_dim, 0, stream>>>(
                    input, finished, output, num_rows, indices, source_row, start_expert, end_expert,k);
    }
   
}
template <typename IndType>
void topkGatingSoftmaxKernelLauncher(
    const float* gating_output,
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
            topkGatingSoftmaxLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, IndType, Is_renormalize>( \
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
                LAUNCH_SOFTMAX(128, 1); 
                break;
            }
            case 160:                         //160使用256的模板
            case 256: LAUNCH_SOFTMAX(256, 1); break;
            case 512: LAUNCH_SOFTMAX(512, 2); break;
            default: {
            TORCH_CHECK(softmax_workspace != nullptr,
                "softmax_workspace must be provided for num_experts that are not a power of 2.");
            static constexpr int TPB = 256;
            moeSoftmax<TPB><<<num_tokens, TPB, 0, stream>>>(
                gating_output, nullptr, softmax_workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, nullptr, topk_weights, topk_indicies, token_expert_indices,
                num_experts, topk, 0, num_experts);
            }
        }
        #undef LAUNCH_SOFTMAX // 避免宏泄漏
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

    const int num_experts = gating_output.size(-1);
    const int num_tokens = gating_output.numel() / num_experts;
    const int topk = topk_weights.size(-1);

    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    const bool needs_workspace = (!is_pow_2 && num_experts != 160) || num_experts > 256;
    const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor softmax_workspace = torch::empty({workspace_size}, gating_output.options());
   
    
    int* token_expert_indices_ptr = nullptr;
    if (token_expert_indices.has_value() && token_expert_indices.value().defined()) {
        token_expert_indices_ptr = token_expert_indices.value().data_ptr<int>();
    }

    if(topk_indices.scalar_type() == at::ScalarType::Int)
    {
       topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
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
       topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
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
       topkGatingSoftmaxKernelLauncher(
            gating_output.data_ptr<float>(),
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
 
}

} // namespace native
} // namespace at