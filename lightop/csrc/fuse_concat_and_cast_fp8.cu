#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace at {
namespace native {

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

using vec_bf16_8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) unsigned short;

inline __device__ uint8_t float_to_fp8e4m3(float f) {
    constexpr uint32_t fp8_max = UINT32_C(1087) << 20;
    constexpr uint32_t denorm_mask = UINT32_C(141) << 23;
    uint32_t f_bits = c10::detail::fp32_to_bits(f);
    uint8_t result = 0u;
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    f_bits ^= sign;
    if (f_bits >= fp8_max) {
      result = 0x7f;  // 01111111 (NaN/最大值)
    } else {
      if (f_bits < (UINT32_C(121) << 23)) {
        f_bits = c10::detail::fp32_to_bits(
          c10::detail::fp32_from_bits(f_bits) + 
           c10::detail::fp32_from_bits(denorm_mask)
        );
        result = static_cast<uint8_t>(f_bits - denorm_mask);
      } else {
        uint8_t mant_odd = (f_bits >> 20) & 1;
        f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
        f_bits += mant_odd;
        result = static_cast<uint8_t>(f_bits >> 20);
      }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

inline __device__ uint16_t float_to_fp8e4m3_x2(float a, float b) {
#if defined(__gfx938__)
    int32_t temp = 0; 
    temp = __builtin_hcu_cvt_pk_fp8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
#else
    uint16_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    return out;
#endif
}

inline __device__ uint32_t float_to_fp8e4m3_x4(float a, float b, float c, float d) {
#if defined(__gfx938__)
    int32_t res = 0; 
    res = __builtin_hcu_cvt_pk_fp8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_fp8_f32(c, d, res, true);
    return res;
#else
    int32_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e4m3(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e4m3(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e4m3(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e4m3(d);
    return out;
#endif
}

inline __device__ uint8_t float_to_fp8e5m2(float f) {
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;
  uint32_t f_bits = c10::detail::fp32_to_bits(f);
  uint8_t result = 0u;
  const uint32_t sign = f_bits & UINT32_C(0x80000000);
  f_bits ^= sign;
  if (f_bits >= fp8_max) {
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      f_bits = c10::detail::fp32_to_bits(c10::detail::fp32_from_bits(f_bits)
               + c10::detail::fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      uint32_t mant_odd = (f_bits >> 21) & 1;
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;
      f_bits += mant_odd;
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }
  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

inline __device__ uint32_t float_to_fp8e5m2_x4(float a, float b, float c, float d) {
#if defined(__gfx938__)
    int32_t res = 0;
    res = __builtin_hcu_cvt_pk_bf8_f32(a, b, res, false);
    res = __builtin_hcu_cvt_pk_bf8_f32(c, d, res, true);
    return res;
#else
    int32_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    reinterpret_cast<uint8_t*>(&out)[2] = float_to_fp8e5m2(c);
    reinterpret_cast<uint8_t*>(&out)[3] = float_to_fp8e5m2(d);
    return out;
#endif
}

inline __device__ uint16_t float_to_fp8e5m2_x2(float a, float b) {
#if defined(__gfx938__)
    int32_t temp = 0; 
    temp = __builtin_hcu_cvt_pk_fp8_f32(a, b, temp, false);
    return static_cast<uint16_t>(temp & 0xFFFF);
#else
    uint16_t out = 0;
    reinterpret_cast<uint8_t*>(&out)[0] = float_to_fp8e5m2(a);
    reinterpret_cast<uint8_t*>(&out)[1] = float_to_fp8e5m2(b);
    return out;
#endif
}

template<int VEC_SIZE, int CACHE_T_CONST>
__inline__ __device__ uint64_t convert_vec_to_uint8_t_8(const float* v_vec_data_float) {
    uint32_t w0 = 0, w1 = 0;
    if constexpr (CACHE_T_CONST == 1) {
        w0 = float_to_fp8e4m3_x4(v_vec_data_float[0], v_vec_data_float[1], v_vec_data_float[2], v_vec_data_float[3]);
        w1 = float_to_fp8e4m3_x4(v_vec_data_float[4], v_vec_data_float[5], v_vec_data_float[6], v_vec_data_float[7]);
    } else {
        w0 = float_to_fp8e5m2_x4(v_vec_data_float[0], v_vec_data_float[1], v_vec_data_float[2], v_vec_data_float[3]);
        w1 = float_to_fp8e5m2_x4(v_vec_data_float[4], v_vec_data_float[5], v_vec_data_float[6], v_vec_data_float[7]);
    }
    return ((uint64_t)w1 << 32) | w0;
}

template<typename Element, int WARP_NUM, int per_block>
__device__ __forceinline__ void direct_copy_cast_fp8(
    const Element* __restrict__ input,
    uint8_t* __restrict__ output,
    int bidx,                   int tid,                    
    uint32_t M,                 uint32_t N,                
    uint32_t in_section_num,     uint32_t out_section_num,
    uint32_t stride_m,    uint32_t stride_n,
    uint32_t block_num     ) {
    
    int ele_num_perthread = 16 / sizeof(Element);
    int threadnum_need_by_one_section = in_section_num / ele_num_perthread;
    int num_warp_need_one_section = 0;
    int num_section_perwarp = 0;
    int total_num_section_perblock = 16;
    
    if(threadnum_need_by_one_section >= 64){
        num_warp_need_one_section = threadnum_need_by_one_section / 64;
    } else {
        num_section_perwarp = 64 / threadnum_need_by_one_section;
        total_num_section_perblock = num_section_perwarp * WARP_NUM;
    }
        
    if(num_section_perwarp > 0 ){
        int tid_insection = tid % threadnum_need_by_one_section;
        int tid_section_num = tid / threadnum_need_by_one_section;
        
        if (tid_section_num < total_num_section_perblock) {
            int itr = (per_block / total_num_section_perblock) == 0 ? 1 : (per_block / total_num_section_perblock);
                    
            for(int init=0; init< itr ; init++){
                int sectiond = init * total_num_section_perblock + tid_section_num;
                int sub_offset = bidx * per_block + sectiond;
                
                if(sub_offset < block_num){
                    auto output_ptr_block_start_base = output + sub_offset * out_section_num;
                    int M_idx =  sub_offset / N;
                    int N_idx =  sub_offset % N;
                    auto input_ptr_block_start = input + M_idx * stride_m + N_idx * stride_n;
                    int threadindex_offset = tid_insection * ele_num_perthread;
                                    
                    if( threadindex_offset < in_section_num){
                        vec_bf16_8 val  =  *(vec_bf16_8*)(input_ptr_block_start + threadindex_offset);
                        float f_arr[8];
                        unsigned short* us_ptr = (unsigned short*)&val;
                        #pragma unroll
                        for(int i=0; i<8; i++) {
                            f_arr[i] = static_cast<float>(*(reinterpret_cast<const Element*>(&us_ptr[i])));
                        }
                        
                        uint64_t fp8_val = convert_vec_to_uint8_t_8<8, 1>(f_arr);
                        *reinterpret_cast<uint64_t*>(output_ptr_block_start_base + threadindex_offset) = fp8_val;
                    }
                }
            }
        }
    }
}

template<
  typename Element, 
  uint16_t WARP_NUM,
  uint16_t per_block_A, 
  uint16_t per_block_B, 
  uint16_t Blcoksize> 
__global__ void __launch_bounds__(1024) fused_qkv_prefill_cast_fp8_kernel(
  const Element* __restrict__ input_Q, 
  const Element* __restrict__ input_A, 
  const Element* __restrict__ input_B, 
  const Element* __restrict__ input_V, 
  uint8_t* __restrict__ output_Q,        
  uint8_t* __restrict__ output_K,       
  uint8_t* __restrict__ output_V,      
  uint32_t M_q, 
  uint32_t M_kv, 
  uint32_t N_q,                    
  uint32_t N_k,                       
  uint32_t N_v,                         
  uint32_t q_section_num,              
  uint32_t a_section_num,   
  uint32_t b_section_num,   
  uint32_t c_section_num,             
  uint32_t v_section_num,              
  uint32_t stride_qm, uint32_t stride_qn,   
  uint32_t stride_am, uint32_t stride_an,   
  uint32_t stride_bm, uint32_t stride_bn,   
  uint32_t stride_vm, uint32_t stride_vn,   
  uint32_t block_num_A,  
  uint32_t block_num_B,  
  uint32_t block_num_Q,  
  uint32_t block_num_V) 
{
  const int bidx = blockIdx.z;
   int tid = threadIdx.x;
  int warp_id = tid / 64;
   int ele_num_perthread = 16 / sizeof(Element);
   
  int total_blockidx_num_A = DIVIDE(block_num_A, per_block_A);
    int total_blockidx_num_B = DIVIDE(block_num_B, per_block_B);
    int total_blockidx_num_Q = DIVIDE(block_num_Q, per_block_A);
    int total_blockidx_num_V = DIVIDE(block_num_V, per_block_A);

    if(bidx < total_blockidx_num_A){
       direct_copy_cast_fp8<Element, WARP_NUM, per_block_A>(
          input_A, output_K, bidx, tid, M_kv, N_k, a_section_num, c_section_num, stride_am, stride_an, block_num_A 
      );
  }
  else if (bidx < total_blockidx_num_A + total_blockidx_num_B) {
       if(warp_id == 0){
        int bidx_b = bidx - total_blockidx_num_A;
         int theradnum_need_by_one_B_section = b_section_num / ele_num_perthread;
         int tid_insection_B = tid % theradnum_need_by_one_B_section;
         int tid_section_num_B = tid / theradnum_need_by_one_B_section;
         int total_num_section_perblock =  64 / (b_section_num / ele_num_perthread)  * 1;
        int itr = (per_block_B / total_num_section_perblock) ==0 ? 1 : (per_block_B / total_num_section_perblock);
                  
        for(int init=0; init< itr ; init++){
          int sectiond = init * total_num_section_perblock + tid_section_num_B ;
          int sub_offset_B = bidx_b * per_block_B + sectiond;
                      
          if(sub_offset_B < block_num_B){
              int M_idx, N_idx;
              const Element* input_B_ptr_block_start;
              uint8_t* output_ptr_block_start_base;
              int loop_count;
              int output_stride;
              if (stride_bn == 0) {
                  M_idx = sub_offset_B;
                  N_idx = 0;
                  input_B_ptr_block_start = input_B + M_idx * stride_bm;
                  output_ptr_block_start_base = output_K + M_idx * (N_k * c_section_num) + a_section_num;
                  loop_count = N_k;
                  output_stride = c_section_num;
              } else {
                  M_idx = sub_offset_B / N_k;
                  N_idx = sub_offset_B % N_k;
                   input_B_ptr_block_start = input_B + M_idx * stride_bm + N_idx * stride_bn;
                  output_ptr_block_start_base = output_K + sub_offset_B * c_section_num + a_section_num;
                  loop_count = 1;
                  output_stride = 0;
               }
              int threadindex_offset = tid_insection_B * ele_num_perthread;
                            
              if( threadindex_offset < b_section_num){
                vec_bf16_8 val_B  =  *(vec_bf16_8*)(input_B_ptr_block_start + threadindex_offset);
                float f_arr[8];
                unsigned short* us_ptr = (unsigned short*)&val_B;
                #pragma unroll
                for(int i=0; i<8; i++) {
                    f_arr[i] = static_cast<float>(*(reinterpret_cast<const Element*>(&us_ptr[i])));
                }
                
                uint64_t fp8_val = convert_vec_to_uint8_t_8<8, 1>(f_arr);
                      
                for(int i=0; i<loop_count; i++){
                  *reinterpret_cast<uint64_t*>(output_ptr_block_start_base + i * output_stride + threadindex_offset) = fp8_val;
                }
              }
            }
        }
      }
  }
  else if (bidx < total_blockidx_num_A + total_blockidx_num_B + total_blockidx_num_Q) {
      int bidx_q = bidx - total_blockidx_num_A - total_blockidx_num_B;
       direct_copy_cast_fp8<Element, WARP_NUM, per_block_A>(
          input_Q, output_Q, bidx_q, tid, M_q, N_q, q_section_num, q_section_num, stride_qm, stride_qn, block_num_Q 
      );
  }
  else {
      int bidx_v = bidx - total_blockidx_num_A - total_blockidx_num_B - total_blockidx_num_Q;
       direct_copy_cast_fp8<Element, WARP_NUM, per_block_A>(
          input_V, output_V, bidx_v, tid, M_kv, N_v, v_section_num, v_section_num, stride_vm, stride_vn, block_num_V 
      );
  }
}

template<int per_block_A, int per_block_B, int Blcoksize, int WARP_NUM, typename T>
void launch_fused_qkv_prefill_cast_fp8(
    const T* input_Q_ptr, const T* input_A_ptr, const T* input_B_ptr, const T* input_V_ptr,
    uint8_t* output_Q_ptr, uint8_t* output_K_ptr, uint8_t* output_V_ptr,
    uint32_t M_q, uint32_t M_kv, 
    uint32_t N_q, uint32_t N_k, uint32_t N_v,
    uint32_t q_section_num, uint32_t a_section_num, uint32_t b_section_num, uint32_t c_section_num, uint32_t v_section_num,
    uint32_t stride_qm, uint32_t stride_qn,
    uint32_t stride_am, uint32_t stride_an,
    uint32_t stride_bm, uint32_t stride_bn,
    uint32_t stride_vm, uint32_t stride_vn,
    bool skip_q = false) { 
    
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64; 
    blockDim.y = 1;
    blockDim.z = 1;
        
    int block_num_A = M_kv * N_k; 
    int block_num_B = M_kv * N_k; 
    int block_num_Q = skip_q ? 0 : (M_q * N_q);  
    int block_num_V = M_kv * N_v; 
    if(stride_bn == 0){
      block_num_B = M_kv; 
    }
    
    int total_blockidx_num_A = DIVIDE(block_num_A, per_block_A);
    int total_blockidx_num_B = DIVIDE(block_num_B, per_block_B);
    int total_blockidx_num_Q = DIVIDE(block_num_Q, per_block_A);
    int total_blockidx_num_V = DIVIDE(block_num_V, per_block_A);
    
    gridDim.z = total_blockidx_num_A + total_blockidx_num_B + total_blockidx_num_Q + total_blockidx_num_V;
    gridDim.y = 1; 
    gridDim.x = 1;
         
    const hipStream_t stream = at::cuda::getCurrentHIPStream();
        
    fused_qkv_prefill_cast_fp8_kernel<T, WARP_NUM, per_block_A, per_block_B, Blcoksize><<<gridDim, blockDim, 0, stream>>>(
      input_Q_ptr, input_A_ptr, input_B_ptr, input_V_ptr,
      output_Q_ptr, output_K_ptr, output_V_ptr,
      M_q, M_kv, 
      N_q, N_k, N_v,
      q_section_num, a_section_num, b_section_num, c_section_num, v_section_num,
      stride_qm, stride_qn, stride_am, stride_an, stride_bm, stride_bn, stride_vm, stride_vn,
      block_num_A, block_num_B, block_num_Q, block_num_V
    );
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fused qkv fp8 cast: %s\n", cudaGetErrorString(err));
    }
}

//fuse split_kv，cat(kv_nope,k_pe),qkv_to_fp8
std::vector<torch::Tensor> ds_fused_qkv_cast_fp8(
  torch::Tensor q,
  torch::Tensor kv_nope, 
  torch::Tensor k_pe,     
  int64_t qk_nope_head_dim, 
  int64_t v_head_dim)
{
    TORCH_CHECK(q.dim() == 3 && kv_nope.dim() == 3 && k_pe.dim() == 3, "All inputs must be 3D");
    
    bool skip_q = (q.scalar_type() == torch::kFloat8_e4m3fn);
    
    int64_t M_kv = kv_nope.size(0); 
    int64_t M_q = q.size(0);        
            
    int64_t N_q = q.size(1);
    int64_t N_k = kv_nope.size(1);
    int64_t N_v = kv_nope.size(1);
    
    int q_section_num = q.size(2);
    int a_section_num = qk_nope_head_dim;
    int b_section_num = k_pe.size(2);
    int c_section_num = a_section_num + b_section_num;
    int v_section_num = v_head_dim;
    
    auto output_options = kv_nope.options().dtype(torch::kFloat8_e4m3fn);
    
    torch::Tensor out_q;
    if (skip_q) {
        out_q = q;
    } else {
        out_q = torch::empty({M_q, N_q, q_section_num}, output_options);
    }
    
    torch::Tensor out_k = torch::empty({M_kv, N_k, c_section_num}, output_options); 
    torch::Tensor out_v = torch::empty({M_kv, N_v, v_section_num}, output_options); 
        
    const int stride_qm = q.stride(0); const int stride_qn = q.stride(1);
    const int stride_am = kv_nope.stride(0); const int stride_an = kv_nope.stride(1); 
    const int stride_bm = k_pe.stride(0); const int stride_bn = k_pe.stride(1);
    const int stride_vm = kv_nope.stride(0); const int stride_vn = kv_nope.stride(1); 
    
    if (kv_nope.scalar_type() == at::ScalarType::BFloat16) {
        c10::BFloat16* kv_nope_ptr = (c10::BFloat16*)kv_nope.data_ptr();
        launch_fused_qkv_prefill_cast_fp8<32, 8, 1024, 1, c10::BFloat16>(
            skip_q ? nullptr : (c10::BFloat16*)q.data_ptr(), 
             kv_nope_ptr,                                               
             (c10::BFloat16*)k_pe.data_ptr(), 
             kv_nope_ptr + qk_nope_head_dim,                             
             skip_q ? nullptr : (uint8_t*)out_q.data_ptr(), 
             (uint8_t*)out_k.data_ptr(), 
             (uint8_t*)out_v.data_ptr(),
            M_q, M_kv, 
            N_q, N_k, N_v,
            q_section_num, a_section_num, b_section_num, c_section_num, v_section_num,
            stride_qm, stride_qn, stride_am, stride_an, stride_bm, stride_bn, stride_vm, stride_vn,
            skip_q
        );
    } else if (kv_nope.scalar_type() == at::ScalarType::Half) {
        c10::Half* kv_nope_ptr = (c10::Half*)kv_nope.data_ptr();
        launch_fused_qkv_prefill_cast_fp8<32, 8, 1024, 1, c10::Half>(
            skip_q ? nullptr : (c10::Half*)q.data_ptr(), 
             kv_nope_ptr, 
             (c10::Half*)k_pe.data_ptr(), 
             kv_nope_ptr + qk_nope_head_dim,                            
             skip_q ? nullptr : (uint8_t*)out_q.data_ptr(), 
             (uint8_t*)out_k.data_ptr(), 
             (uint8_t*)out_v.data_ptr(),
            M_q, M_kv, 
            N_q, N_k, N_v,
            q_section_num, a_section_num, b_section_num, c_section_num, v_section_num,
            stride_qm, stride_qn, stride_am, stride_an, stride_bm, stride_bn, stride_vm, stride_vn,
            skip_q
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Only BF16 and FP16 are supported.");
    }
             
    return {out_q, out_k, out_v};
}
} // namespace native
} // namespace at