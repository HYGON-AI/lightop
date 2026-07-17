#include <torch/all.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <vector>

#ifndef USE_ROCM
#define USE_ROCM 
#endif

#ifdef USE_ROCM
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

enum class Fp8KVCacheDataType {
  kAuto,
  kFp8E4M3,
  kFp8E5M2
};

template <typename T, int N> 
struct alignas(sizeof(T) * N) Vector {
  T val[N];
};

#ifndef HIP_FP8_TYPE_OCP
#define HIP_FP8_TYPE_OCP 0
#endif

#if HIP_FP8_TYPE_OCP
using fp8_e4m3_type = __hip_fp8_e4m3;
using fp8_e5m2_type = __hip_fp8_e5m2;
#else
using fp8_e4m3_type = __hip_fp8_e4m3_fnuz;
using fp8_e5m2_type = __hip_fp8_e5m2_fnuz;
#endif

namespace at {
namespace native {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ Tout scaled_convert_fp8(Tin a, float scale) {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        return static_cast<Tout>(a);
    } else {
        float f_val = 0.0f;
        if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
            fp8_e4m3_type f8;
            f8.__x = static_cast<uint8_t>(a);
            f_val = static_cast<float>(f8);
        } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
            fp8_e5m2_type f8;
            f8.__x = static_cast<uint8_t>(a);
            f_val = static_cast<float>(f8);
        }

        f_val *= scale;

        if constexpr (std::is_same_v<Tout, float>) {
            return f_val;
        } else if constexpr (std::is_same_v<Tout, uint16_t>) {
            __half h = __float2half(f_val);
            return reinterpret_cast<const uint16_t&>(h);
        } else if constexpr (std::is_same_v<Tout, __nv_bfloat16>) {
            return __float2bfloat16(f_val);
        } else {
            return static_cast<Tout>(f_val);
        }
    }
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void gather_convert_fp8_kernel(
    const Tin* __restrict__ src_cache,        // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    Tout* __restrict__ dst,                   // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float scale,
    const int32_t* __restrict__ seq_starts) {

 
  const int bid = blockIdx.y;
  const int seq_start = cu_seq_lens[bid];
  const int seq_end = cu_seq_lens[bid + 1];
  const int seq_len = seq_end - seq_start;


  if (seq_len == 0) return;


  constexpr int MAX_VEC_BYTES = 16;
  constexpr int ELEM_SIZE = sizeof(Tout) > sizeof(Tin) ? sizeof(Tout) : sizeof(Tin);
  constexpr int VEC_ELEMS = MAX_VEC_BYTES / ELEM_SIZE;
  using DstVec = Vector<Tout, VEC_ELEMS>;
  using SrcVec = Vector<Tin, VEC_ELEMS>;

  const int total_chunks_per_token = CEIL_DIV(entry_size, VEC_ELEMS);
  
 
  const int total_work = seq_len * total_chunks_per_token;


  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  

  if (global_idx >= total_work) return;


  const int token_idx_in_seq = global_idx / total_chunks_per_token;
  const int chunk_idx = global_idx % total_chunks_per_token;


  const int global_token_idx = seq_start + token_idx_in_seq;
  const int pid = token_idx_in_seq / block_size;
  const int eid = token_idx_in_seq % block_size;

  int offset = 0;
  if (seq_starts != nullptr) {
      offset = seq_starts[bid] / block_size;
  }
  const int block_id = block_table[bid * block_table_stride + offset + pid];


  const int elem_start = chunk_idx * VEC_ELEMS;
  const int valid_elems = entry_size - elem_start;


  const Tin* src_ptr = src_cache + block_id * cache_block_stride + eid * cache_entry_stride + elem_start;
  Tout* dst_ptr = dst + global_token_idx * dst_entry_stride + elem_start;

  if (valid_elems >= VEC_ELEMS) {
      SrcVec src_vec = *reinterpret_cast<const SrcVec*>(src_ptr);
      DstVec dst_vec;
      
      #pragma unroll
      for (int i = 0; i < VEC_ELEMS; ++i) {
          dst_vec.val[i] = scaled_convert_fp8<Tout, Tin, kv_dt>(src_vec.val[i], scale);
      }
      
      *reinterpret_cast<DstVec*>(dst_ptr) = dst_vec;
  } else {
      for (int i = 0; i < valid_elems; ++i) {
          dst_ptr[i] = scaled_convert_fp8<Tout, Tin, kv_dt>(src_ptr[i], scale);
      }
  }
}


#define CALL_GATHER_CONVERT_FP8_OPT(Tout, Tin, KV_DTYPE)                                \
  gather_convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const Tin*>(src_cache.data_ptr()),                               \
      reinterpret_cast<Tout*>(dst.data_ptr()),                                          \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),             \
      block_size, entry_size, block_table_stride, cache_block_stride,               \
      cache_entry_stride, dst_entry_stride, scale, seq_starts_ptr);

void gather_convert_fp8_cache(
    torch::Tensor const& src_cache,    
    torch::Tensor const& dst,          
    torch::Tensor const& block_table,  
    torch::Tensor const& cu_seq_lens,  
    int64_t batch_size,
    double scale,                      
    const std::string& kv_cache_dtype, 
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
    
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  int max_elem_size = std::max((int)dst.element_size(), (int)src_cache.element_size());
  int vec_elems = 16 / max_elem_size;
  int total_chunks_per_token = CEIL_DIV(entry_size, vec_elems);

  int max_seq_len = block_table.size(1) * block_size;
  int total_work_per_batch = max_seq_len * total_chunks_per_token;

  dim3 block(64);
  dim3 grid(CEIL_DIV(total_work_per_batch, block.x), batch_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (kv_cache_dtype == "auto") {
    if (dst.dtype() == at::ScalarType::Float) {
      CALL_GATHER_CONVERT_FP8_OPT(float, float, Fp8KVCacheDataType::kAuto);
    } else if (dst.dtype() == at::ScalarType::Half) {
      CALL_GATHER_CONVERT_FP8_OPT(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);
    } else if (dst.dtype() == at::ScalarType::BFloat16) {
      CALL_GATHER_CONVERT_FP8_OPT(__nv_bfloat16, __nv_bfloat16, Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (dst.dtype() == at::ScalarType::Float) {
      CALL_GATHER_CONVERT_FP8_OPT(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst.dtype() == at::ScalarType::Half) {
      CALL_GATHER_CONVERT_FP8_OPT(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst.dtype() == at::ScalarType::BFloat16) {
      CALL_GATHER_CONVERT_FP8_OPT(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);
    }
  } else if (kv_cache_dtype == "fp8_e5m2") {
    if (dst.dtype() == at::ScalarType::Float) {
      CALL_GATHER_CONVERT_FP8_OPT(float, uint8_t, Fp8KVCacheDataType::kFp8E5M2);
    } else if (dst.dtype() == at::ScalarType::Half) {
      CALL_GATHER_CONVERT_FP8_OPT(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E5M2);
    } else if (dst.dtype() == at::ScalarType::BFloat16) {
      CALL_GATHER_CONVERT_FP8_OPT(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E5M2);
    }
  } else {
    TORCH_CHECK(false, "Unsupported kv_cache_dtype for fused gather_convert: ", kv_cache_dtype);
  }
}
__global__ void cp_gather_and_upconvert_fp8_kv_cache_opt(
    const uint8_t* __restrict__ src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,               // [TOT_TOKENS, 576]
    const int32_t* __restrict__ block_table,       // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ seq_lens,          // [BATCH]
    const int32_t* __restrict__ workspace_starts,  // [BATCH]
    const int32_t block_size, const int32_t head_dim,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t num_splits) 
{
    const int64_t bid = blockIdx.x; // Batch ID
    
    const int32_t task_type = blockIdx.y / num_splits; // 0: FP8 Dequant, 1: RoPE Copy
    const int32_t split = blockIdx.y % num_splits;
    
    const int32_t seq_start = workspace_starts[bid];
    const int32_t seq_len = seq_lens[bid];
    const int32_t tot_slots = seq_len;
    const int32_t split_slots = CEIL_DIV(tot_slots, num_splits);

    const int32_t split_start = split * split_slots;
    const int32_t split_end = std::min((split + 1) * split_slots, tot_slots);

    if (split_start >= tot_slots) return;

    const int32_t batch_offset = bid * block_table_stride;
    const int32_t* batch_block_table = block_table + batch_offset;

    const int tid = threadIdx.x;

    if (task_type == 0) {
        const int threads_per_token = 32;
        const int tokens_per_iter = blockDim.x / threads_per_token;
        const int local_token_id = tid / threads_per_token;
        const int lane_id = tid % threads_per_token;

        for (int pid_base = split_start; pid_base < split_end; pid_base += tokens_per_iter) {
            int pid = pid_base + local_token_id;
            
            if (pid < split_end) {
                int offset_div = pid / block_size;
                int offset_mod = pid % block_size;

                auto block_id = batch_block_table[offset_div];
                const uint8_t* token_ptr = src_cache + block_id * cache_block_stride + offset_mod * cache_entry_stride;
                __nv_bfloat16* dst_ptr = dst + (seq_start + pid) * dst_entry_stride;

                const uint8_t* no_pe_ptr = token_ptr;
                const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
                
                const int tile = lane_id / 8;
                const float scale = scales_ptr[tile];

                using FP8Vec = Vector<uint8_t, 16>;
                const FP8Vec* src_vec_ptr = reinterpret_cast<const FP8Vec*>(no_pe_ptr) + lane_id;
                FP8Vec src_vec = *src_vec_ptr;

                using BF16Vec = Vector<__nv_bfloat16, 16>;
                BF16Vec dst_vec;
                
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    dst_vec.val[i] = scaled_convert_fp8<__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(src_vec.val[i], scale);
                }

                BF16Vec* dst_vec_ptr = reinterpret_cast<BF16Vec*>(dst_ptr) + lane_id;
                *dst_vec_ptr = dst_vec;
            }
        }
    } else {
        const int threads_per_token = 8;
        const int tokens_per_iter = blockDim.x / threads_per_token;
        const int local_token_id = tid / threads_per_token;
        const int lane_id = tid % threads_per_token;

        for (int pid_base = split_start; pid_base < split_end; pid_base += tokens_per_iter) {
            int pid = pid_base + local_token_id;
            
            if (pid < split_end) {
                int offset_div = pid / block_size;
                int offset_mod = pid % block_size;

                auto block_id = batch_block_table[offset_div];
                const uint8_t* token_ptr = src_cache + block_id * cache_block_stride + offset_mod * cache_entry_stride;
                __nv_bfloat16* dst_ptr = dst + (seq_start + pid) * dst_entry_stride;

                const __nv_bfloat16* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(token_ptr + 512 + 16);

                using RoPEVec = Vector<__nv_bfloat16, 8>;
                const RoPEVec* src_rope_vec = reinterpret_cast<const RoPEVec*>(rope_ptr) + lane_id;
                RoPEVec* dst_rope_vec = reinterpret_cast<RoPEVec*>(dst_ptr + 512) + lane_id;
                
                *dst_rope_vec = *src_rope_vec;
            }
        }
    }
}
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size) 
{
    at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int32_t block_size = src_cache.size(1);
    int32_t head_dim = dst.size(1);

    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(workspace_starts.dtype() == torch::kInt32, "workspace_starts must be int32");

    TORCH_CHECK(src_cache.device() == dst.device(), "src_cache and dst must be on the same device");
    TORCH_CHECK(src_cache.device() == block_table.device(), "src_cache and block_table must be on the same device");
    
    TORCH_CHECK(src_cache.dtype() == torch::kUInt8, "src_cache must be uint8");
    TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bfloat16");
    TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

    int64_t block_table_stride = block_table.stride(0);
    int64_t cache_block_stride = src_cache.stride(0);
    int64_t cache_entry_stride = src_cache.stride(1);
    int64_t dst_entry_stride = dst.stride(0);

    constexpr int THREADS_PER_BLOCK = 1024;

   
    int32_t num_splits;
    if (batch_size > 128) {
        num_splits = 2;
    } else if (batch_size > 64) {
        num_splits = 4;
    } else {
        num_splits = 16;
    }

    dim3 grid(batch_size, num_splits * 2); 
    
    dim3 block(THREADS_PER_BLOCK);

    cp_gather_and_upconvert_fp8_kv_cache_opt<<<grid, block, 0, stream>>>(
        src_cache.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
        block_table.data_ptr<int32_t>(), 
        seq_lens.data_ptr<int32_t>(),
        workspace_starts.data_ptr<int32_t>(), 
        block_size, head_dim,
        block_table_stride, cache_block_stride, cache_entry_stride,
        dst_entry_stride,
        num_splits  
    );
}

// Paged prefill counterpart of cp_gather_and_upconvert_fp8_kv_cache_opt.
// It is a drop-in replacement for dequantize_k_cache_paged for packed
// [*, 1, 656] FP8 cache rows: gather by the flattened page table, dequantize
// the 512 no-PE values, and copy the 64 BF16 RoPE values unchanged.
__global__ void prefill_gather_and_upconvert_fp8_kv_cache_kernel(
    const uint8_t* __restrict__ src_cache,
    const int32_t* __restrict__ page_table_flat,
    __nv_bfloat16* __restrict__ kv_dst,
    int64_t num_physical_tokens,
    int64_t num_kv_tokens) {
    constexpr int kThreadsPerItem = 32;
    constexpr int kItemsPerBlock = 1024 / kThreadsPerItem;
    constexpr int kPhysicalTokenStride = 656;
    constexpr int kNopeDim = 512;
    constexpr int kScaleBytes = 16;
    constexpr int kRopeDim = 64;
    constexpr int kHeadDim = kNopeDim + kRopeDim;

    const int local_item = threadIdx.x / kThreadsPerItem;
    const int lane_id = threadIdx.x % kThreadsPerItem;
    const int64_t item_id =
        static_cast<int64_t>(blockIdx.x) * kItemsPerBlock + local_item;
    if (item_id >= num_kv_tokens) {
        return;
    }

    const int64_t physical_token_id =
        static_cast<int64_t>(page_table_flat[item_id]);
    __nv_bfloat16* dst_ptr = kv_dst + item_id * kHeadDim;
    using BF16Vec16 = Vector<__nv_bfloat16, 16>;
    BF16Vec16 nope_dst;

    if (physical_token_id >= 0 &&
        physical_token_id < num_physical_tokens) {
        const uint8_t* token_ptr =
            src_cache + physical_token_id * kPhysicalTokenStride;
        const float* scales_ptr =
            reinterpret_cast<const float*>(token_ptr + kNopeDim);
        const float scale = scales_ptr[lane_id / 8];
        using FP8Vec16 = Vector<uint8_t, 16>;
        const FP8Vec16 src_vec =
            *(reinterpret_cast<const FP8Vec16*>(token_ptr) + lane_id);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            nope_dst.val[i] = scaled_convert_fp8<
                __nv_bfloat16,
                uint8_t,
                Fp8KVCacheDataType::kFp8E4M3>(src_vec.val[i], scale);
        }
        *(reinterpret_cast<BF16Vec16*>(dst_ptr) + lane_id) = nope_dst;

        if (lane_id < 8) {
            using BF16Vec8 = Vector<__nv_bfloat16, 8>;
            const __nv_bfloat16* rope_ptr =
                reinterpret_cast<const __nv_bfloat16*>(
                    token_ptr + kNopeDim + kScaleBytes);
            const BF16Vec8 rope_vec =
                *(reinterpret_cast<const BF16Vec8*>(rope_ptr) + lane_id);
            *(reinterpret_cast<BF16Vec8*>(dst_ptr + kNopeDim) + lane_id) =
                rope_vec;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            nope_dst.val[i] = __float2bfloat16(0.0f);
        }
        *(reinterpret_cast<BF16Vec16*>(dst_ptr) + lane_id) = nope_dst;

        if (lane_id < 8) {
            using BF16Vec8 = Vector<__nv_bfloat16, 8>;
            BF16Vec8 zero_vec;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                zero_vec.val[i] = __float2bfloat16(0.0f);
            }
            *(reinterpret_cast<BF16Vec8*>(dst_ptr + kNopeDim) + lane_id) =
                zero_vec;
        }
    }
}

void prefill_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& page_table_flat,
    torch::Tensor const& kv_dst) {
    constexpr int64_t kPhysicalTokenStride = 656;
    constexpr int64_t kHeadDim = 576;
    constexpr int64_t kItemsPerBlock = 32;

    TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on a GPU");
    TORCH_CHECK(page_table_flat.is_cuda(),
                "page_table_flat must be on a GPU");
    TORCH_CHECK(kv_dst.is_cuda(), "kv_dst must be on a GPU");
    TORCH_CHECK(src_cache.device() == page_table_flat.device() &&
                    src_cache.device() == kv_dst.device(),
                "all inputs and outputs must be on the same device");
    TORCH_CHECK(src_cache.element_size() == 1,
                "src_cache must have one-byte FP8/uint8 elements");
    TORCH_CHECK(src_cache.is_contiguous(), "src_cache must be contiguous");
    TORCH_CHECK((src_cache.dim() == 3 || src_cache.dim() == 4) &&
                    src_cache.size(-2) == 1 &&
                    src_cache.size(-1) == kPhysicalTokenStride,
                "src_cache must have shape [*, 1, 656]");
    TORCH_CHECK(page_table_flat.dim() == 1 &&
                    page_table_flat.scalar_type() == at::ScalarType::Int &&
                    page_table_flat.is_contiguous(),
                "page_table_flat must be contiguous int32 [num_kv_tokens]");

    const int64_t num_kv_tokens = page_table_flat.numel();
    TORCH_CHECK(kv_dst.scalar_type() == at::ScalarType::BFloat16 &&
                    kv_dst.is_contiguous() && kv_dst.dim() == 3 &&
                    kv_dst.size(0) == num_kv_tokens &&
                    kv_dst.size(1) == 1 && kv_dst.size(2) == kHeadDim,
                "kv_dst must be contiguous BF16 [num_kv_tokens, 1, 576]");

    const int64_t block_count = CEIL_DIV(num_kv_tokens, kItemsPerBlock);
    if (block_count == 0) {
        return;
    }

    at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>(block_count));
    const dim3 block(1024);
    prefill_gather_and_upconvert_fp8_kv_cache_kernel
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(src_cache.data_ptr()),
            page_table_flat.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(kv_dst.data_ptr()),
            src_cache.numel() / kPhysicalTokenStride,
            num_kv_tokens);
}

__device__ __forceinline__ int64_t decode_wave_broadcast_i64(
    int64_t value,
    int src_lane) {
    union {
        int64_t value;
        int32_t parts[2];
    } data;
    data.value = value;
    data.parts[0] = __shfl(data.parts[0], src_lane, 64);
    data.parts[1] = __shfl(data.parts[1], src_lane, 64);
    return data.value;
}

template <typename index_t, int kTokensPerBlock>
__launch_bounds__(64 * kTokensPerBlock)
__global__ void decode_gather_and_up_convert_kernel(
    const uint8_t* __restrict__ src_cache,
    const index_t* __restrict__ token_ids,
    const int32_t* __restrict__ valid_lengths,
    __nv_bfloat16* __restrict__ dst,
    int64_t width,
    int64_t num_physical_tokens,
    int64_t token_ids_stride_0,
    int64_t token_ids_stride_1,
    int64_t valid_lengths_stride_0,
    int64_t dst_stride_0,
    int64_t dst_stride_1) {
    constexpr int kThreadsPerToken = 64;
    constexpr int kElementsPerLane = 8;
    constexpr int kPhysicalTokenStride = 656;

    const int lane_id = threadIdx.x & (kThreadsPerToken - 1);
    const int local_token_id = threadIdx.x / kThreadsPerToken;
    const int64_t batch_id = static_cast<int64_t>(blockIdx.y);
    const int64_t token_offset =
        static_cast<int64_t>(blockIdx.x) * kTokensPerBlock + local_token_id;
    if (token_offset >= width) {
        return;
    }

    int32_t valid_width = 0;
    if (lane_id == 0) {
        valid_width = valid_lengths[batch_id * valid_lengths_stride_0];
    }
    valid_width = __shfl(valid_width, 0, kThreadsPerToken);

    int64_t physical_token_id = -1;
    if (lane_id == 0 && token_offset < static_cast<int64_t>(valid_width)) {
        physical_token_id = static_cast<int64_t>(
            token_ids[batch_id * token_ids_stride_0 +
                      token_offset * token_ids_stride_1]);
    }
    physical_token_id = decode_wave_broadcast_i64(physical_token_id, 0);

    __nv_bfloat16* dst_ptr =
        dst + batch_id * dst_stride_0 + token_offset * dst_stride_1;

    using BF16Vec = Vector<__nv_bfloat16, kElementsPerLane>;
    BF16Vec dst_vec;

    if (physical_token_id >= 0 && physical_token_id < num_physical_tokens) {
        const uint8_t* token_ptr =
            src_cache + physical_token_id * kPhysicalTokenStride;
        const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
        float scale = 0.0f;
        if ((lane_id & 15) == 0) {
            scale = scales_ptr[lane_id / 16];
        }
        scale = __shfl(scale, 0, 16);

        using FP8Vec = Vector<uint8_t, kElementsPerLane>;
        const FP8Vec src_vec =
            *(reinterpret_cast<const FP8Vec*>(token_ptr) + lane_id);

        #pragma unroll
        for (int i = 0; i < kElementsPerLane; ++i) {
            dst_vec.val[i] = scaled_convert_fp8<
                __nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(
                    src_vec.val[i], scale);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kElementsPerLane; ++i) {
            dst_vec.val[i] = __float2bfloat16(0.0f);
        }
    }

    *(reinterpret_cast<BF16Vec*>(dst_ptr) + lane_id) = dst_vec;
}

template <typename index_t, int kTokensPerBlock>
void launch_decode_gather_and_up_convert(
    const uint8_t* src_cache,
    const index_t* token_ids,
    const int32_t* valid_lengths,
    __nv_bfloat16* dst,
    int64_t batch_size,
    int64_t width,
    int64_t num_physical_tokens,
    int64_t token_ids_stride_0,
    int64_t token_ids_stride_1,
    int64_t valid_lengths_stride_0,
    int64_t dst_stride_0,
    int64_t dst_stride_1,
    cudaStream_t stream) {
    constexpr int kThreadsPerToken = 64;
    const dim3 grid(CEIL_DIV(width, kTokensPerBlock), batch_size);
    const dim3 block(kThreadsPerToken * kTokensPerBlock);
    decode_gather_and_up_convert_kernel<index_t, kTokensPerBlock>
        <<<grid, block, 0, stream>>>(
            src_cache,
            token_ids,
            valid_lengths,
            dst,
            width,
            num_physical_tokens,
            token_ids_stride_0,
            token_ids_stride_1,
            valid_lengths_stride_0,
            dst_stride_0,
            dst_stride_1);
}

void decode_gather_and_up_convert(
    torch::Tensor const& src_cache,
    torch::Tensor const& token_ids,
    torch::Tensor const& dst,
    torch::Tensor const& valid_lengths) {
    constexpr int64_t kHeadDim = 512;
    constexpr int64_t kPhysicalTokenStride = 656;

    TORCH_CHECK(src_cache.is_cuda(), "src_cache must be on a GPU");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be on a GPU");
    TORCH_CHECK(dst.is_cuda(), "dst must be on a GPU");
    TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be on a GPU");
    TORCH_CHECK(src_cache.device() == token_ids.device(),
                "src_cache and token_ids must be on the same device");
    TORCH_CHECK(src_cache.device() == dst.device(),
                "src_cache and dst must be on the same device");
    TORCH_CHECK(src_cache.device() == valid_lengths.device(),
                "src_cache and valid_lengths must be on the same device");
    TORCH_CHECK(src_cache.dtype() == torch::kUInt8, "src_cache must be uint8");
    TORCH_CHECK(src_cache.is_contiguous(), "src_cache must be contiguous");
    TORCH_CHECK(src_cache.numel() % kPhysicalTokenStride == 0,
                "src_cache.numel() must be divisible by 656");
    TORCH_CHECK(token_ids.dim() == 2, "token_ids must have shape [batch, width]");
    TORCH_CHECK(token_ids.dtype() == torch::kInt32 ||
                    token_ids.dtype() == torch::kInt64,
                "token_ids must be int32 or int64");
    TORCH_CHECK(token_ids.stride(0) > 0 && token_ids.stride(1) > 0,
                "token_ids must have positive strides");
    TORCH_CHECK(valid_lengths.dim() == 1,
                "valid_lengths must have shape [batch]");
    TORCH_CHECK(valid_lengths.size(0) == token_ids.size(0),
                "valid_lengths batch must match token_ids");
    TORCH_CHECK(valid_lengths.dtype() == torch::kInt32,
                "valid_lengths must be int32");
    TORCH_CHECK(valid_lengths.stride(0) > 0,
                "valid_lengths must have a positive stride");
    TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bfloat16");
    TORCH_CHECK(dst.dim() == 3, "dst must have shape [batch, width, 512]");
    TORCH_CHECK(dst.size(0) == token_ids.size(0) &&
                    dst.size(1) == token_ids.size(1) &&
                    dst.size(2) == kHeadDim,
                "dst shape must be [token_ids.size(0), token_ids.size(1), 512]");
    TORCH_CHECK(dst.stride(2) == 1 && dst.stride(1) == kHeadDim,
                "dst must be contiguous in its last two dimensions");
    TORCH_CHECK(dst.stride(0) > 0, "dst batch stride must be positive");

    const int64_t batch_size = token_ids.size(0);
    const int64_t width = token_ids.size(1);
    const int64_t total_tokens = batch_size * width;
    if (total_tokens == 0) {
        return;
    }

    at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t num_physical_tokens = src_cache.numel() / kPhysicalTokenStride;

    int tokens_per_block = 16;
    if (const char* value = std::getenv(
            "LIGHTOP_DECODE_GATHER_TOKENS_PER_BLOCK")) {
        tokens_per_block = std::atoi(value);
    }
    TORCH_CHECK(tokens_per_block == 4 || tokens_per_block == 8 ||
                    tokens_per_block == 16,
                "LIGHTOP_DECODE_GATHER_TOKENS_PER_BLOCK must be 4, 8, or 16");

    AT_DISPATCH_INDEX_TYPES(token_ids.scalar_type(), "decode_gather_and_up_convert", [&] {
        const uint8_t* src_ptr = src_cache.data_ptr<uint8_t>();
        const index_t* token_ids_ptr = token_ids.data_ptr<index_t>();
        const int32_t* valid_lengths_ptr = valid_lengths.data_ptr<int32_t>();
        __nv_bfloat16* dst_ptr =
            reinterpret_cast<__nv_bfloat16*>(dst.data_ptr());
        if (tokens_per_block == 4) {
            launch_decode_gather_and_up_convert<index_t, 4>(
                src_ptr, token_ids_ptr, valid_lengths_ptr, dst_ptr,
                batch_size, width, num_physical_tokens,
                token_ids.stride(0), token_ids.stride(1),
                valid_lengths.stride(0), dst.stride(0), dst.stride(1), stream);
        } else if (tokens_per_block == 8) {
            launch_decode_gather_and_up_convert<index_t, 8>(
                src_ptr, token_ids_ptr, valid_lengths_ptr, dst_ptr,
                batch_size, width, num_physical_tokens,
                token_ids.stride(0), token_ids.stride(1),
                valid_lengths.stride(0), dst.stride(0), dst.stride(1), stream);
        } else {
            launch_decode_gather_and_up_convert<index_t, 16>(
                src_ptr, token_ids_ptr, valid_lengths_ptr, dst_ptr,
                batch_size, width, num_physical_tokens,
                token_ids.stride(0), token_ids.stride(1),
                valid_lengths.stride(0), dst.stride(0), dst.stride(1), stream);
        }
    });
}
}
}
