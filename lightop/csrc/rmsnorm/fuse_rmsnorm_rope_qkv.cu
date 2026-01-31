#include <limits>
#include <stdint.h>
#include <ATen/Dispatch.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/autograd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "reduce.h"
#include <type_traits>
#include <algorithm>
#include <sstream>

// [TODO] 后续改用buffer_load_dwordx{x}
#define BUFFER_LOAD_DWORDX_GLUE(a, b)  a##b
#define BUFFER_LOAD_DWORDX_N(n)        BUFFER_LOAD_DWORDX_GLUE(buffer_load_dwordx, n)
#define BUFFER_LOAD_DWORDX_STRIFY(x)   #x
#define BUFFER_LOAD_DWORDX_STR(n)      BUFFER_LOAD_DWORDX_STRIFY(BUFFER_LOAD_DWORDX_N(n))
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE_CONST, ...) \
  if (interleave) {                                      \
    constexpr bool INTERLEAVE_CONST = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    constexpr bool INTERLEAVE_CONST = false;                       \
    __VA_ARGS__                                          \
  }

#define DISPATCH_HAS_RESIDUAL(has_residual, HAS_RESIDUAL_CONST, ...) \
  if (has_residual) {                                              \
    constexpr bool HAS_RESIDUAL_CONST = true;                                \
    __VA_ARGS__                                                      \
  } else {                                                         \
    constexpr bool HAS_RESIDUAL_CONST = false;                               \
    __VA_ARGS__                                                      \
  }

#define DISPATCH_rotary_dim(rotary_dim, ROTARY_DIM_CONST, ...) \
  switch (rotary_dim) { \
    case 64: { \
      constexpr int ROTARY_DIM_CONST = 64; \
      __VA_ARGS__ \
      break; \
    } \
    case 128: { \
      constexpr int ROTARY_DIM_CONST = 128; \
      __VA_ARGS__ \
      break; \
    } \
    case 256: { \
      constexpr int ROTARY_DIM_CONST = 256; \
      __VA_ARGS__ \
      break; \
    } \
    case 512: { \
      constexpr int ROTARY_DIM_CONST = 512; \
      __VA_ARGS__ \
      break; \
    } \
    default: { \
      std::ostringstream err_msg; \
      err_msg << "Unsupported rotary_dim: " << rotary_dim; \
      printf("%s\n", err_msg.str().c_str()); \
    } \
  }

#define DISPATCH_head_dim(head_dim, HEAD_DIM_CONST,...) \
  switch (head_dim) { \
      case 64: { \
        constexpr int HEAD_DIM_CONST = 64; \
        __VA_ARGS__ \
        break; \
      } \
      case 128: { \
        constexpr int HEAD_DIM_CONST = 128; \
        __VA_ARGS__ \
        break; \
      } \
      case 256: { \
        constexpr int HEAD_DIM_CONST = 256; \
        __VA_ARGS__ \
        break; \
      } \
      case 512: { \
        constexpr int HEAD_DIM_CONST = 512; \
        __VA_ARGS__ \
        break; \
      } \
      default: { \
        std::ostringstream err_msg; \
        err_msg << "Unsupported head_dim: " << head_dim; \
        printf("%s\n", err_msg.str().c_str()); \
      } \
    }

#define DISPATCH_head_dim_v(head_dim_v, HEAD_DIM_V_CONST,...) \
switch (head_dim_v) { \
  case 64: { \
    constexpr int HEAD_DIM_V_CONST = 64; \
    __VA_ARGS__ \
    break; \
  } \
  case 128: { \
    constexpr int HEAD_DIM_V_CONST = 128; \
    __VA_ARGS__ \
    break; \
  } \
  case 256: { \
    constexpr int HEAD_DIM_V_CONST = 256; \
    __VA_ARGS__ \
    break; \
  } \
  case 512: { \
    constexpr int HEAD_DIM_V_CONST = 512; \
    __VA_ARGS__ \
    break; \
  } \
  default: { \
    std::ostringstream err_msg; \
    err_msg << "Unsupported head_dim_v: " << head_dim_v; \
    printf("%s\n", err_msg.str().c_str()); \
  } \
}


#define lgkmcnt_wait_barrier(X)\
__builtin_amdgcn_sched_barrier(0);\
asm volatile(\
  "s_waitcnt lgkmcnt(%0)\n\t"\
  "s_barrier\n"\
  :: "I"(X)\
  :);\
__builtin_amdgcn_sched_barrier(0);

namespace at {
namespace native {

template <typename T,int reducesize=64>
__inline__ __device__ T WarpReduceSum_NEW(T val) {
#pragma unroll
    for (int offset = reducesize/2; offset > 0; offset >>= 1) 
        val += WARP_SHFL_DOWN(val, offset);
    return val;
}

// 每个 warp 独立处理一个 head：tdim = head_dim/VEC，tid 为 lane_id (0~63)
template<typename T_ACC, typename scalar_t, int VEC, int NUM_WARP, bool HAS_RESIDUAL>
__inline__ __device__ void apply_rmsnorm_residual(
    scalar_t* input,
    scalar_t* gamma,
    scalar_t* residual,
    int head_dim,
    T_ACC eps,
    scalar_t* intput_vec
) {
  using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
  __shared__ T_ACC s_rstd[NUM_WARP];
  T_ACC val = 0;
  const int lane_id = threadIdx.x & 63;
  const int warp_id = threadIdx.x >> 6;
  const int tdim = head_dim / VEC;  // 每个 head 仅 head_dim 个元素，参与线程数 = head_dim/VEC
  T_ACC trstd;
  int64_t idx = (int64_t)lane_id * VEC;
  scalar_t residual_vec[VEC];
  if (lane_id < tdim) {
    *(LoadT*)intput_vec = *(LoadT*)(input + idx);
    if constexpr (HAS_RESIDUAL) {
      *(LoadT*)residual_vec = *(LoadT*)(residual + idx);
    }
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      if constexpr (HAS_RESIDUAL) {
        residual_vec[ii] += intput_vec[ii];
        val += static_cast<T_ACC>(residual_vec[ii]) * static_cast<T_ACC>(residual_vec[ii]);
      } else {
        val += static_cast<T_ACC>(intput_vec[ii]) * static_cast<T_ACC>(intput_vec[ii]);
      }
    }
  }
  val = WarpReduceSum_NEW<T_ACC>(val);
  if (lane_id == 0)
    s_rstd[warp_id] = c10::cuda::compat::rsqrt(val / head_dim + eps);
  __syncthreads();
  trstd = s_rstd[warp_id];
  if (lane_id < tdim) {
    #pragma unroll
    for (int ii = 0; ii < VEC; ++ii) {
      int jj = (lane_id * VEC + ii) % head_dim;
      if constexpr (HAS_RESIDUAL) {
        intput_vec[ii] = static_cast<T_ACC>(residual_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
      } else {
        intput_vec[ii] = static_cast<T_ACC>(intput_vec[ii]) * trstd * static_cast<T_ACC>(gamma[jj]);
      }
    }
  }
}




//分 kernel head_dim = rot_dim 与 head_dim < rot_dim

template <typename T_ACC, typename scalar_t, bool INTERLEAVE, bool Residual, int NUM_HEAD_V_WARP, int NUM_HEAD_Q_WARP, int NUM_HEAD_K_WARP, int HEAD_DIM_CONST, int ROTARY_DIM_CONST, bool IS_EQUAL_HEAD_DIM_ROTARY_DIM, int VEC_SIZE, int VEC_SIZE_V>
__global__ void rms_rotary_embedding_kernel_pipeline_with_kv_store(
    const int64_t* __restrict__ pos_ids,
    scalar_t* __restrict__ q,  // [B * S, num_q_head, head_dim]
    scalar_t* __restrict__ k,
    scalar_t* __restrict__ v,  // [B * S, num_kv_head, head_dim]
    //rms arg
    scalar_t* gamm_q,     //[head_dim]
    scalar_t* gamm_k,     //[head_dim]
    scalar_t* Residual_q,  //[B * S, num_q_head, head_dim]
    scalar_t* Residual_k,  //[B* S, num_kv_head, head_dim]
    //[num_blocks, kv_head_num, page_size, head_dim]
    scalar_t* k_buffer,
    scalar_t* v_buffer,
    //如果scalar_t是bf16, 而 Cache是 fp8， 是不是需要量化
    const int64_t* __restrict__ kv_cache_loc,
    scalar_t* __restrict__ cos_sin_cache,   // [MAX_TOKEN, rotary_dim]
    int page_size,
    uint32_t num_seq,
    uint32_t num_q_head,
    uint32_t num_kv_head,
    size_t q_stride_n,
    size_t q_stride_h,
    size_t k_stride_n,
    size_t k_stride_h,
    size_t v_stride_n,
    size_t v_stride_h,
    size_t k_buffer_stride_n,
    size_t k_buffer_stride_h,
    size_t k_buffer_stride_p,
    size_t v_buffer_stride_n,
    size_t v_buffer_stride_h,
    size_t v_buffer_stride_p,
    float eps
) 
{
    using vec_scalar_t_qk = at::native::memory::aligned_vector<scalar_t, VEC_SIZE>;
    using vec_scalar_t_v = at::native::memory::aligned_vector<scalar_t, VEC_SIZE_V>;

    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int lane_id = tid & 63; 
    // constexpr int q_get_lane_id = 64 * NUM_HEAD_Q_WARP - 1;
    // constexpr int k_get_lane_id = 64 * NUM_HEAD_K_WARP - 1;
    // constexpr int v_get_lane_id = 64 * NUM_HEAD_V_WARP - 1;
    const int token_idx_in_cache = kv_cache_loc[token_idx];
    const int cache_block_idx = token_idx_in_cache  / page_size;
    const int cache_block_offset = token_idx_in_cache % page_size;

    if (warp_id >= NUM_HEAD_Q_WARP + NUM_HEAD_K_WARP) {
        const int v_local_warp_id = warp_id - (NUM_HEAD_Q_WARP + NUM_HEAD_K_WARP);
        const int v_offset = token_idx * v_stride_n;
        const int v_cache_offset = cache_block_idx * v_buffer_stride_n + cache_block_offset * v_buffer_stride_p;

        #pragma unroll
        for (int iter = 0; iter < num_kv_head; iter += NUM_HEAD_V_WARP) {
            int v_head_idx = iter + v_local_warp_id;
            if (v_head_idx >= num_kv_head) break;
            vec_scalar_t_v v_vec_data;
            const int v_global_offset = v_offset + v_head_idx * v_stride_h;
            const int v_cache_global_offset = v_cache_offset + v_head_idx * v_buffer_stride_h;
            v_vec_data = *(reinterpret_cast<vec_scalar_t_v*>(v + v_global_offset + lane_id * VEC_SIZE_V));
            *(reinterpret_cast<vec_scalar_t_v*>(v_buffer + v_cache_global_offset + lane_id * VEC_SIZE_V)) = v_vec_data;
        }
          asm volatile("s_barrier");// 其他warp同步
    } else {
        // asm volatile("s_barrier");
        //处理Q, K
        const int pos = pos_ids[token_idx];
        scalar_t cos[VEC_SIZE], sin[VEC_SIZE];
        constexpr int half_rotary_dim = ROTARY_DIM_CONST >> 1;
        if (lane_id * VEC_SIZE < ROTARY_DIM_CONST) {
            constexpr int sin_offset = ROTARY_DIM_CONST >> 1;
            int vec_idx;
            if constexpr (INTERLEAVE)
                vec_idx = (lane_id * VEC_SIZE) >> 1;
            else 
                vec_idx = (lane_id * VEC_SIZE) % half_rotary_dim;
            *(reinterpret_cast<vec_scalar_t_qk*>(cos)) = *(reinterpret_cast<vec_scalar_t_qk*>(cos_sin_cache + pos * ROTARY_DIM_CONST + vec_idx));
            *(reinterpret_cast<vec_scalar_t_qk*>(sin)) = *(reinterpret_cast<vec_scalar_t_qk*>(cos_sin_cache + pos * ROTARY_DIM_CONST + sin_offset + vec_idx));
        }

        if (warp_id < NUM_HEAD_Q_WARP) {
            const int q_offset = token_idx * q_stride_n;
            __shared__ scalar_t q_rms_smem[ROTARY_DIM_CONST * NUM_HEAD_Q_WARP];
            for (int q_head_idx = 0; q_head_idx < num_q_head; q_head_idx += NUM_HEAD_Q_WARP) {
                if (q_head_idx + warp_id >= num_q_head) {
                  asm volatile("s_barrier");
                  break;
                }
                // 每个 warp 处理一个 head：warp_id 对应 head (q_head_idx + warp_id)
                scalar_t* q_ptr = q + q_offset + (q_head_idx + warp_id) * q_stride_h;
                scalar_t q_vec[VEC_SIZE];
                scalar_t q_data[VEC_SIZE];
                scalar_t* residual_q_ptr = nullptr;
                if constexpr (Residual) {
                    residual_q_ptr = Residual_q + q_offset + (q_head_idx + warp_id) * q_stride_h;
                }
                apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE, NUM_HEAD_Q_WARP, Residual>(q_ptr, gamm_q, residual_q_ptr, HEAD_DIM_CONST, eps, q_vec);
                if (lane_id * VEC_SIZE < ROTARY_DIM_CONST) 
                    *(reinterpret_cast<vec_scalar_t_qk*>(q_rms_smem + warp_id * ROTARY_DIM_CONST + lane_id * VEC_SIZE)) = *(reinterpret_cast<vec_scalar_t_qk*>(q_vec));
                lgkmcnt_wait_barrier(0);
                if (lane_id * VEC_SIZE < ROTARY_DIM_CONST) {
                    int sign = lane_id * VEC_SIZE < half_rotary_dim ? -1 : 1;
                    #pragma unroll
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        q_data[i] = q_vec[i] * cos[i] + sign * q_rms_smem[warp_id * ROTARY_DIM_CONST + lane_id * VEC_SIZE + i - sign * half_rotary_dim] * sin[i];
                    }
                } else {
                    *reinterpret_cast<vec_scalar_t_qk*>(q_data) = *reinterpret_cast<vec_scalar_t_qk*>(q_vec);
                }
                *(reinterpret_cast<vec_scalar_t_qk*>(q_ptr + lane_id * VEC_SIZE)) = *(reinterpret_cast<vec_scalar_t_qk*>(q_data));
            }
        } else {
            const int k_local_warp_id = warp_id - NUM_HEAD_Q_WARP;
            const int k_offset = token_idx * k_stride_n;
            const int k_cache_offset = cache_block_idx * k_buffer_stride_n + cache_block_offset * k_buffer_stride_p;
            __shared__ scalar_t k_rms_smem[ROTARY_DIM_CONST * NUM_HEAD_K_WARP];
            for (int k_head_idx = 0; k_head_idx < num_kv_head; k_head_idx += NUM_HEAD_K_WARP) {
              if (k_head_idx + k_local_warp_id >= num_kv_head) {
                asm volatile("s_barrier");
                break;
              }
              scalar_t* k_ptr = k + k_offset + (k_head_idx + k_local_warp_id) * k_stride_h;
              scalar_t k_vec[VEC_SIZE];
              scalar_t k_data[VEC_SIZE];
              scalar_t* residual_k_ptr = nullptr;
              if constexpr (Residual) {
                residual_k_ptr = Residual_k + k_offset + (k_head_idx + k_local_warp_id) * k_stride_h;
              }
              apply_rmsnorm_residual<T_ACC, scalar_t, VEC_SIZE, NUM_HEAD_Q_WARP, Residual>(k_ptr, gamm_k, residual_k_ptr, HEAD_DIM_CONST, eps, k_vec);
              if (lane_id * VEC_SIZE < ROTARY_DIM_CONST) 
                *(reinterpret_cast<vec_scalar_t_qk*>(k_rms_smem + k_local_warp_id * ROTARY_DIM_CONST + lane_id * VEC_SIZE)) = *(reinterpret_cast<vec_scalar_t_qk*>(k_vec));
              lgkmcnt_wait_barrier(0);
              if (lane_id * VEC_SIZE < ROTARY_DIM_CONST) {
                int sign = lane_id * VEC_SIZE < half_rotary_dim ? -1 : 1;
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; ++i) {
                  k_data[i] = k_vec[i] * cos[i] + sign * k_rms_smem[k_local_warp_id * ROTARY_DIM_CONST + lane_id * VEC_SIZE + i - sign * half_rotary_dim] * sin[i];
                }
              } else {
                *reinterpret_cast<vec_scalar_t_qk*>(k_data) = *reinterpret_cast<vec_scalar_t_qk*>(k_vec);
              }
              *(reinterpret_cast<vec_scalar_t_qk*>(k_ptr + lane_id * VEC_SIZE)) = *(reinterpret_cast<vec_scalar_t_qk*>(k_data));
              const int k_cache_global_offset = k_cache_offset + (k_head_idx + k_local_warp_id) * k_buffer_stride_h;
              *(reinterpret_cast<vec_scalar_t_qk*>(k_buffer + k_cache_global_offset + lane_id * VEC_SIZE)) = *(reinterpret_cast<vec_scalar_t_qk*>(k_data));
            }
        }
    }
    return;
}

template<typename T_ACC, typename scalar_t>
void FuseRmsRopeWithKvStore(
    scalar_t* q,
    scalar_t* k,
    scalar_t* v,
    scalar_t* k_buffer,
    scalar_t* v_buffer,
    scalar_t* cos_sin_cache,
    int64_t* positions,
    int64_t* kv_cache_loc,
    scalar_t* weight_q,
    scalar_t* weight_k,
    scalar_t* residual_q,
    scalar_t* residual_k,
    int bs,
    int num_qo_heads,
    int num_kv_heads,
    int rotary_dim,
    int head_dim,
    int head_dim_v,
    int page_size,
    int q_stride_n,
    int q_stride_h,
    int k_stride_n,
    int k_stride_h,
    int v_stride_n,
    int v_stride_h,
    int k_buffer_stride_n,
    int k_buffer_stride_h,
    int k_buffer_stride_p,
    int v_buffer_stride_n,
    int v_buffer_stride_h,
    int v_buffer_stride_p,
    bool interleave,
    float eps
) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // bool has_residual = residual_q.has_value() && residual_k.has_value();
    bool has_residual = residual_q != nullptr && residual_k != nullptr;
    DISPATCH_HAS_RESIDUAL(has_residual, HAS_RESIDUAL_CONST, {
        DISPATCH_INTERLEAVE(interleave, INTERLEAVE_CONST, {
            DISPATCH_rotary_dim(rotary_dim, ROTARY_DIM_CONST, {
                DISPATCH_head_dim_v(head_dim_v, HEAD_DIM_V_CONST, {
                    DISPATCH_head_dim(head_dim, HEAD_DIM_CONST, {
                constexpr bool IS_EQUAL_HEAD_DIM_ROTARY_DIM = HEAD_DIM_CONST == ROTARY_DIM_CONST;
                constexpr int bytes_per_vec = 16 / sizeof(scalar_t);
                constexpr int vec_size_qk = HEAD_DIM_CONST / 64;
                constexpr uint32_t vec_size = (bytes_per_vec < vec_size_qk) ? bytes_per_vec : vec_size_qk;
                constexpr int vec_size_v_dim = HEAD_DIM_V_CONST / 64;
                constexpr uint32_t vec_size_v = (bytes_per_vec < vec_size_v_dim) ? bytes_per_vec : vec_size_v_dim;
                printf("vec_size: %d, vec_size_v: %d\n", vec_size, vec_size_v);

                //[TODO] 根据model 中的shape进行config调整
                constexpr uint32_t NUM_HEAD_V_WARP = 4;
                constexpr uint32_t NUM_HEAD_K_WARP = 1;
                constexpr uint32_t NUM_HEAD_Q_WARP = 3;
                
                dim3 nblks(bs);
                dim3 nthrs((NUM_HEAD_V_WARP + NUM_HEAD_Q_WARP + NUM_HEAD_K_WARP) * 64);

                rms_rotary_embedding_kernel_pipeline_with_kv_store<
                    T_ACC,
                    scalar_t,
                    INTERLEAVE_CONST, 
                    HAS_RESIDUAL_CONST,
                    NUM_HEAD_V_WARP,
                    NUM_HEAD_Q_WARP,
                    NUM_HEAD_K_WARP,
                    HEAD_DIM_CONST, 
                    ROTARY_DIM_CONST, 
                    IS_EQUAL_HEAD_DIM_ROTARY_DIM,
                    vec_size, vec_size_v><<<nblks, nthrs, 0, stream>>>(positions, q, k, v, weight_q, weight_k, residual_q, residual_k, 
                                    k_buffer, v_buffer, kv_cache_loc, cos_sin_cache, page_size, bs, num_qo_heads, num_kv_heads, 
                                    q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h, 
                                    k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p, v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p, eps);
                    });
                });
            });
        });
    });
}



void rms_rotary_embedding_fuse_with_kv_store (
    Tensor& positions,       //[MAX_TOKEN]
    Tensor& query,  // [B * S, H, D]
    Tensor& key,    // [B * S, K, D]
    Tensor& v,      // [B * S, V, D]
    Tensor& cos_sin_cache,   //[MAX_TOKEN, ROT_DIM]
    int page_size,
    Tensor& k_buffer,        //[NUM_BLOCKS, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM]
    Tensor& v_buffer,        //[NUM_BLOCKS, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM]
    Tensor& kv_cache_loc,    //[B * S]
    bool interleave,
    Tensor& weight_q,         //[HEAD_DIM]
    Tensor& weight_k,         //[HEAD_DIM]
    std::optional<Tensor>& residual_q,       //[B*S, H, D]
    std::optional<Tensor>& residual_k,       //[B*S, K, D]
    float epsilon) {
        CHECK_CONTIGUOUS(query);
        CHECK_CONTIGUOUS(key);
        CHECK_CONTIGUOUS(v);
        CHECK_CONTIGUOUS(cos_sin_cache);
        CHECK_CONTIGUOUS(positions);
        CHECK_CONTIGUOUS(k_buffer);
        CHECK_CONTIGUOUS(v_buffer);
        CHECK_CONTIGUOUS(kv_cache_loc);
        CHECK_CONTIGUOUS(weight_q);
        CHECK_CONTIGUOUS(weight_k);
        TORCH_CHECK(query.dim() == 3, "query must have shape [B * S, H, D]");
        TORCH_CHECK(key.dim() == 3, "key must have shape [B * S, K, D]");
        TORCH_CHECK(v.dim() == 3, "v must have shape [B * S, V, D]");
        TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must have shape [MAX_TOKEN, ROT_DIM]");
        TORCH_CHECK(positions.dim() == 1, "positions must have shape [MAX_TOKEN]");
        TORCH_CHECK(k_buffer.dim() == 4, "k_buffer must have shape [NUM_BLOCKS, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM]");
        TORCH_CHECK(v_buffer.dim() == 4, "v_buffer must have shape [NUM_BLOCKS, KV_HEAD_NUM, PAGE_SIZE, HEAD_DIM]");
        TORCH_CHECK(kv_cache_loc.dim() == 1, "kv_cache_loc must have shape [MAX_TOKEN]");
        //TORCH_CHECK(key.shape[-1] == v.shape[-1], "key and v head_dim must have the same shape");
        int head_dim_v = v.size(2);
        size_t k_buffer_stride_n = k_buffer.stride(0);
        size_t k_buffer_stride_h = k_buffer.stride(1);
        size_t k_buffer_stride_p = k_buffer.stride(2);
        size_t v_buffer_stride_n = v_buffer.stride(0);
        size_t v_buffer_stride_h = v_buffer.stride(1);
        size_t v_buffer_stride_p = v_buffer.stride(2);

        size_t v_stride_n = v.stride(0);
        size_t v_stride_h = v.stride(1);

        CHECK_EQ(query.size(0), key.size(0));
        CHECK_EQ(query.size(2), key.size(2));
        unsigned int rotary_dim = cos_sin_cache.size(1);
        unsigned int num_qo_heads = query.size(1);
        unsigned int num_kv_heads = key.size(1);
        unsigned int head_dim = query.size(2);
        unsigned int bs = query.size(0);

        size_t q_stride_n = query.stride(0);
        size_t q_stride_h = query.stride(1);
        size_t k_stride_n = key.stride(0);
        size_t k_stride_h = key.stride(1);

        // printf("bs: %d, num_qo_heads: %d, num_kv_heads: %d, head_dim: %d, rotary_dim: %d, head_dim_v: %d, page_size: %d, q_stride_n: %zu, q_stride_h: %zu, k_stride_n: %zu, k_stride_h: %zu, v_stride_n: %zu, v_stride_h: %zu, k_buffer_stride_n: %zu, k_buffer_stride_h: %zu, k_buffer_stride_p: %zu, v_buffer_stride_n: %zu, v_buffer_stride_h: %zu, v_buffer_stride_p: %zu\n", 
        //     bs, num_qo_heads, num_kv_heads, head_dim, rotary_dim, head_dim_v, page_size, q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h, k_buffer_stride_n, k_buffer_stride_h, k_buffer_stride_p, v_buffer_stride_n, v_buffer_stride_h, v_buffer_stride_p);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            query.scalar_type(), "", [&] {
                using T_ACC = at::acc_type<scalar_t, true>; // float;
                FuseRmsRopeWithKvStore<T_ACC, scalar_t>(
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    k_buffer.data_ptr<scalar_t>(),
                    v_buffer.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    positions.data_ptr<int64_t>(),
                    kv_cache_loc.data_ptr<int64_t>(),
                    weight_q.data_ptr<scalar_t>(),
                    weight_k.data_ptr<scalar_t>(),
                    residual_q.has_value() ? residual_q->data_ptr<scalar_t>() : nullptr,
                    residual_k.has_value() ? residual_k->data_ptr<scalar_t>() : nullptr,
                    bs, 
                    num_qo_heads,
                    num_kv_heads,
                    rotary_dim,
                    head_dim,
                    head_dim_v,
                    page_size,
                    q_stride_n,
                    q_stride_h,
                    k_stride_n,
                    k_stride_h,
                    v_stride_n,
                    v_stride_h,
                    k_buffer_stride_n,
                    k_buffer_stride_h,
                    k_buffer_stride_p,
                    v_buffer_stride_n,
                    v_buffer_stride_h,
                    v_buffer_stride_p,
                    interleave,
                    epsilon
                );
        });
    }

}
}