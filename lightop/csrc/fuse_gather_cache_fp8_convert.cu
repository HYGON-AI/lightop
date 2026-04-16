#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
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
}
}