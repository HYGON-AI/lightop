#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>
#include <vector>
#include <algorithm>  // 记得加上这个头文件

#define GET_BOOL_FROM_ENV(env_var, default_val) \
    ([]() { \
        const char* val = std::getenv(env_var); \
        if (val == nullptr) return default_val; \
        std::string s(val); \
        return s == "1" || s == "true" || s == "yes" || s == "on"; \
    })()

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK_1024 1024
#define THREADS_PER_BLOCK_256 256
#define THREADS_PER_BLOCK 512

namespace at {
namespace native {

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
    int optimal_block_num = (N + num_threads - 1) / num_threads;
    int max_block_num = 8192;
    return std::min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  const int h_low = __float2int_rd(h);
  const int w_low = __float2int_rd(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ __forceinline__ float4 ms_deform_attn_im2col_bilinear_v4(
    const float4* __restrict__ bottom_data, 
    const int height, const int width,
    const int nheads, const int channels_v4, // channels_v4 = channels / 4
    const float h, const float w, 
    const int m, const int c_v4) {

    const int h_low = __float2int_rd(h);
    const int w_low = __float2int_rd(w);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const float lh = h - h_low;
    const float lw = w - w_low;
    const float hh = 1.0f - lh;
    const float hw = 1.0f - lw;

    const int w_stride = nheads * channels_v4;
    const int h_stride = width * w_stride;
    const int base_ptr = m * channels_v4 + c_v4;

    float4 res = {0, 0, 0, 0};

    if (h_low >= 0 && w_low >= 0) {
        float4 v = bottom_data[h_low * h_stride + w_low * w_stride + base_ptr];
        float w_interp = hh * hw;
        res.x += v.x * w_interp; res.y += v.y * w_interp; res.z += v.z * w_interp; res.w += v.w * w_interp;
    }
    if (h_low >= 0 && w_high < width) {
        float4 v = bottom_data[h_low * h_stride + w_high * w_stride + base_ptr];
        float w_interp = hh * lw;
        res.x += v.x * w_interp; res.y += v.y * w_interp; res.z += v.z * w_interp; res.w += v.w * w_interp;
    }
    if (h_high < height && w_low >= 0) {
        float4 v = bottom_data[h_high * h_stride + w_low * w_stride + base_ptr];
        float w_interp = lh * hw;
        res.x += v.x * w_interp; res.y += v.y * w_interp; res.z += v.z * w_interp; res.w += v.w * w_interp;
    }
    if (h_high < height && w_high < width) {
        float4 v = bottom_data[h_high * h_stride + w_high * w_stride + base_ptr];
        float w_interp = lh * lw;
        res.x += v.x * w_interp; res.y += v.y * w_interp; res.z += v.z * w_interp; res.w += v.w * w_interp;
    }

    return res;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_vec4(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c_base, 
    const scalar_t top_grad[4],
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t grad_sampling_loc[2],  // 改为数组，返回局部贡献
    scalar_t *grad_attn_weight) {   // 返回局部贡献
  
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  
  scalar_t top_grad_value[4];
  for (int i = 0; i < 4; i++) {
      top_grad_value[i] = top_grad[i] * attn_weight;
  }

  // 初始化输出
  grad_sampling_loc[0] = 0;
  grad_sampling_loc[1] = 0;
  *grad_attn_weight = 0;

#if defined(__gfx936__)
  if constexpr (std::is_same_v<scalar_t, float>) {
    typedef float float4_vec __attribute__((vector_size(16)));
    
    const bool valid1 = (h_low >= 0 && w_low >= 0);
    const bool valid2 = (h_low >= 0 && w_high <= width - 1);
    const bool valid3 = (h_high <= height - 1 && w_low >= 0);
    const bool valid4 = (h_high <= height - 1 && w_high <= width - 1);
    
    const int base_ptr = m * channels + c_base;
    
    // ===== 使用x4原子加写grad_value =====
    if (valid1) {
      const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
      float4_vec grad1 = {w1 * top_grad_value[0], w1 * top_grad_value[1],
                          w1 * top_grad_value[2], w1 * top_grad_value[3]};
      __builtin_hcu_global_atomic_fadd_f32_x4(grad_value + ptr1, grad1);
    }
    
    if (valid2) {
      const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
      float4_vec grad2 = {w2 * top_grad_value[0], w2 * top_grad_value[1],
                          w2 * top_grad_value[2], w2 * top_grad_value[3]};
      __builtin_hcu_global_atomic_fadd_f32_x4(grad_value + ptr2, grad2);
    }
    
    if (valid3) {
      const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
      float4_vec grad3 = {w3 * top_grad_value[0], w3 * top_grad_value[1],
                          w3 * top_grad_value[2], w3 * top_grad_value[3]};
      __builtin_hcu_global_atomic_fadd_f32_x4(grad_value + ptr3, grad3);
    }
    
    if (valid4) {
      const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
      float4_vec grad4 = {w4 * top_grad_value[0], w4 * top_grad_value[1],
                          w4 * top_grad_value[2], w4 * top_grad_value[3]};
      __builtin_hcu_global_atomic_fadd_f32_x4(grad_value + ptr4, grad4);
    }
    
    for (int i = 0; i < 4; i++) {
      const int ptr_offset = base_ptr + i;
      const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + ptr_offset;
      const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + ptr_offset;
      const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + ptr_offset;
      const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + ptr_offset;
      
      scalar_t v1 = valid1 ? bottom_data[ptr1] : scalar_t(0);
      scalar_t v2 = valid2 ? bottom_data[ptr2] : scalar_t(0);
      scalar_t v3 = valid3 ? bottom_data[ptr3] : scalar_t(0);
      scalar_t v4 = valid4 ? bottom_data[ptr4] : scalar_t(0);
      
      scalar_t grad_h_weight = -(hw * v1) - (lw * v2) + (hw * v3) + (lw * v4);
      scalar_t grad_w_weight = -(hh * v1) + (hh * v2) - (lh * v3) + (lh * v4);
      
      grad_sampling_loc[0] += width * grad_w_weight * top_grad_value[i];
      grad_sampling_loc[1] += height * grad_h_weight * top_grad_value[i];
      
      const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
      *grad_attn_weight += top_grad[i] * val;
    }
    
    return;
  }
#endif

  // ===== 标量路径 =====
  for (int i = 0; i < 4; i++) {
    const int base_ptr = m * channels + c_base + i;
    const scalar_t top_grad_val = top_grad_value[i];
    
    const bool valid1 = (h_low >= 0 && w_low >= 0);
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    scalar_t v1 = valid1 ? bottom_data[ptr1] : scalar_t(0);
    if (valid1) {
      atomicAdd(grad_value + ptr1, w1 * top_grad_val);
    }

    const bool valid2 = (h_low >= 0 && w_high <= width - 1);
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    scalar_t v2 = valid2 ? bottom_data[ptr2] : scalar_t(0);
    if (valid2) {
      atomicAdd(grad_value + ptr2, w2 * top_grad_val);
    }

    const bool valid3 = (h_high <= height - 1 && w_low >= 0);
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    scalar_t v3 = valid3 ? bottom_data[ptr3] : scalar_t(0);
    if (valid3) {
      atomicAdd(grad_value + ptr3, w3 * top_grad_val);
    }

    const bool valid4 = (h_high <= height - 1 && w_high <= width - 1);
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    scalar_t v4 = valid4 ? bottom_data[ptr4] : scalar_t(0);
    if (valid4) {
      atomicAdd(grad_value + ptr4, w4 * top_grad_val);
    }

    scalar_t grad_h_weight = -(hw * v1) - (lw * v2) + (hw * v3) + (lw * v4);
    scalar_t grad_w_weight = -(hh * v1) + (hh * v2) - (lh * v3) + (lh * v4);

    grad_sampling_loc[0] += width * grad_w_weight * top_grad_val;
    grad_sampling_loc[1] += height * grad_h_weight * top_grad_val;

    const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *grad_attn_weight += top_grad[i] * val;
  }
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c, const scalar_t &top_grad,
    const scalar_t &attn_weight, scalar_t *&grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}


// Optimized im2col kernel (replaces branching with masks + inline bilinear)
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n,
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index,
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    scalar_t* __restrict__ data_col) {

  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;

    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    // iterate levels
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w, num_heads, channels,
                                                h_im, w_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_im2col_gpu_kernel_3d_grid(
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes,
    const int64_t* __restrict__ data_level_start_index,
    const scalar_t* __restrict__ data_sampling_loc,
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size, const int spatial_size,
    const int num_heads, const int channels,
    const int num_levels, const int num_query,
    const int num_point, scalar_t* __restrict__ data_col) {

    const int CH_V4 = 8;
    
    extern __shared__ char smem[];
    int* s_level_start = (int*)smem;
    int* s_shapes = (int*)(s_level_start + num_levels);
    
    if (threadIdx.x < num_levels) {
        s_level_start[threadIdx.x] = (int)data_level_start_index[threadIdx.x];
        s_shapes[threadIdx.x * 2] = (int)data_spatial_shapes[threadIdx.x * 2];
        s_shapes[threadIdx.x * 2 + 1] = (int)data_spatial_shapes[threadIdx.x * 2 + 1];
    }

    const int b = blockIdx.z;
    const int m = blockIdx.y;

    const int q_in_block = threadIdx.x / CH_V4;
    const int c_v4 = threadIdx.x % CH_V4; // 0~7
    const int queries_per_block = blockSize / CH_V4;
    const int q = blockIdx.x * queries_per_block + q_in_block;

    const int pts_per_query = num_levels * num_point;

    // 计算当前 query 采样数据的 SMEM 偏移
    // 偏移量避开前面的 shapes 空间
    scalar_t* my_smem_base = reinterpret_cast<scalar_t*>(s_shapes + num_levels * 2) + q_in_block * pts_per_query * 3;
    scalar_t* lds_wei = my_smem_base;
    scalar_t* lds_loc = my_smem_base + pts_per_query;

    const int sampling_base_ptr = ((b * num_query + q) * num_heads + m) * pts_per_query;

    if (q < num_query) {
        for (int i = c_v4; i < pts_per_query; i += CH_V4) {
            lds_wei[i] = data_attn_weight[sampling_base_ptr + i];
            reinterpret_cast<float2*>(lds_loc)[i] = reinterpret_cast<const float2*>(data_sampling_loc)[sampling_base_ptr + i];
        }
    }
    __syncthreads();

    if (q >= num_query) return;

    const int qid_stride_v4 = num_heads * CH_V4;
    const int data_value_ptr_init_offset_v4 = b * spatial_size * qid_stride_v4;
    const float4* v_ptr_base = reinterpret_cast<const float4*>(data_value);
    
    float4 col_v4 = {0, 0, 0, 0};

    #pragma unroll
    for (int l = 0; l < num_levels; ++l) {
        const int level_start_id = s_level_start[l];
        const int spatial_h = s_shapes[l * 2];
        const int spatial_w = s_shapes[l * 2 + 1];
        
        const float4* v_ptr = v_ptr_base + (data_value_ptr_init_offset_v4 + level_start_id * qid_stride_v4);

        #pragma unroll
        for (int p = 0; p < num_point; ++p) {
            const int pts_idx = l * num_point + p;
            
            const float weight = lds_wei[pts_idx];
            const float loc_w  = lds_loc[pts_idx * 2];
            const float loc_h  = lds_loc[pts_idx * 2 + 1];

            const float h_im = loc_h * (float)spatial_h - 0.5f;
            const float w_im = loc_w * (float)spatial_w - 0.5f;

            if (h_im > -1.0f && w_im > -1.0f && h_im < (float)spatial_h && w_im < (float)spatial_w) {
                float4 val = ms_deform_attn_im2col_bilinear_v4(
                    v_ptr, spatial_h, spatial_w, num_heads, CH_V4,
                    h_im, w_im, m, c_v4);
                
                col_v4.x += val.x * weight;
                col_v4.y += val.y * weight;
                col_v4.z += val.z * weight;
                col_v4.w += val.w * weight;
            }
        }
    }
    float4* data_col_ptr_v4 = reinterpret_cast<float4*>(data_col);
    data_col_ptr_v4[((b * num_query + q) * num_heads + m) * CH_V4 + c_v4] = col_v4;
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {

    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  const int qid_stride = num_heads * channels;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int _tid = 1; _tid < blockSize; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 2;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }

}


template <typename scalar_t, unsigned int blockSize>
__launch_bounds__(1024, 1) __global__
void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt(
    const int n, const scalar_t* grad_col, const scalar_t* data_value,
    const int64_t* data_spatial_shapes, const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc, const scalar_t* data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t* grad_value, scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {
    

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 64;
    const unsigned int lane_id = tid % 64;
    const unsigned int tid_in_pos = tid % channels;
    const unsigned int reduce_pos_id = tid / channels;
    const int qid_stride = num_heads * channels;
    
    CUDA_1D_KERNEL_LOOP(index, n) {
        int _temp = index;
        
        // 解码 index -> c_col, sampling_index, m_col, b_col
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        _temp /= num_query;
        const int b_col = _temp;
        
        const scalar_t top_grad = grad_col[index];
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;
        const int grad_sampling_ptr = data_weight_ptr;
        scalar_t* grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr << 1);
        scalar_t* grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
        
        // 遍历 level
        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = int(data_level_start_index[l_col]);
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = int(data_spatial_shapes[spatial_h_ptr]);
            const int spatial_w = int(data_spatial_shapes[spatial_h_ptr + 1]);
            
            const int value_ptr_offset = data_value_ptr_init_offset + 
                                        level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;
            
            // 遍历 point
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
                const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
                const scalar_t weight = data_attn_weight[data_weight_ptr];
                
                const scalar_t h_im = loc_h * (scalar_t)spatial_h - (scalar_t)0.5;
                const scalar_t w_im = loc_w * (scalar_t)spatial_w - (scalar_t)0.5;
                
                // 线程本地容器
                scalar_t grad_sh[2];
                grad_sh[0] = (scalar_t)0;
                grad_sh[1] = (scalar_t)0;
                scalar_t grad_a = (scalar_t)0;
                
                if (h_im > -1 && w_im > -1 && 
                    h_im < (scalar_t)spatial_h && 
                    w_im < (scalar_t)spatial_w) {
                    ms_deform_attn_col2im_bilinear<scalar_t>(
                        data_value_ptr, spatial_h, spatial_w, num_heads, 
                        channels, h_im, w_im, m_col, c_col, top_grad, 
                        weight, grad_value_ptr, grad_sh, &grad_a);
                }
                
                if constexpr (std::is_same_v<scalar_t, float>) {
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad_sh[0] += __shfl_down(grad_sh[0], offset, channels);
                        grad_sh[1] += __shfl_down(grad_sh[1], offset, channels);
                        grad_a += __shfl_down(grad_a, offset, channels);
                    }
                } else if constexpr (std::is_same_v<scalar_t, double>) {
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad_sh[0] += __shfl_down(grad_sh[0], offset, channels);
                        grad_sh[1] += __shfl_down(grad_sh[1], offset, channels);
                        grad_a += __shfl_down(grad_a, offset, channels);
                    }
                } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                    // c10::Half 转 float 做 shuffle
                    float grad0 = __half2float(grad_sh[0]);
                    float grad1 = __half2float(grad_sh[1]);
                    float grada = __half2float(grad_a);
                    
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad0 += __shfl_down(grad0, offset, channels);
                        grad1 += __shfl_down(grad1, offset, channels);
                        grada += __shfl_down(grada, offset, channels);
                    }
                    
                    grad_sh[0] = __float2half(grad0);
                    grad_sh[1] = __float2half(grad1);
                    grad_a = __float2half(grada);
                } else {
                    static_assert(sizeof(scalar_t) == 0, 
                                 "Unsupported scalar type for shuffle reduce");
                }
                
                // 每个 reduce_pos_id 的第一个线程写回结果
                if (tid_in_pos == 0) {
                    *grad_sampling_loc_out = grad_sh[0];
                    *(grad_sampling_loc_out + 1) = grad_sh[1];
                    *grad_attn_weight_out = grad_a;
                }
                
                __syncthreads();
                
                data_weight_ptr += 1;
                data_loc_w_ptr += 2;
                grad_attn_weight_out += grad_weight_stride;
                grad_sampling_loc_out += grad_loc_stride;
            }  // p_col
        }  // l_col
    }  // CUDA_1D_KERNEL_LOOP
}

typedef float float4_vec __attribute__((vector_size(16)));

template <typename scalar_t, int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_3d_grid(
    const scalar_t* __restrict__ grad_col, 
    const scalar_t* __restrict__ data_value,
    const int64_t* __restrict__ data_spatial_shapes, 
    const int64_t* __restrict__ data_level_start_index,
    const scalar_t* __restrict__ data_sampling_loc, 
    const scalar_t* __restrict__ data_attn_weight,
    const int batch_size, 
    const int spatial_size, 
    const int num_heads,
    const int channels, 
    const int num_levels, 
    const int num_query,
    const int num_point, 
    scalar_t* __restrict__ grad_value, 
    scalar_t* __restrict__ grad_sampling_loc,
    scalar_t* __restrict__ grad_attn_weight) {
#if defined(__gfx936__)
    const int CH_V4 = 8; // channels=32 / 4
    extern __shared__ char smem[];
    int* s_level_start = (int*)smem;
    int* s_shapes = (int*)(s_level_start + num_levels);
    
    if (threadIdx.x < num_levels) {
        s_level_start[threadIdx.x] = (int)data_level_start_index[threadIdx.x];
        s_shapes[threadIdx.x * 2] = (int)data_spatial_shapes[threadIdx.x * 2];
        s_shapes[threadIdx.x * 2 + 1] = (int)data_spatial_shapes[threadIdx.x * 2 + 1];
    }

    const int b = blockIdx.z, m = blockIdx.y;
    const int q_in_blk = threadIdx.x / CH_V4;
    const int c_v4 = threadIdx.x % CH_V4;
    const int q = blockIdx.x * (blockSize / CH_V4) + q_in_blk;
    const int pts_per_q = num_levels * num_point;

    scalar_t* my_smem = reinterpret_cast<scalar_t*>(s_shapes + num_levels * 2) + q_in_blk * pts_per_q * 3;
    scalar_t* lds_wei = my_smem;
    scalar_t* lds_loc = my_smem + pts_per_q;

    const int sampling_base = ((b * num_query + q) * num_heads + m) * pts_per_q;

    if (q < num_query) {
        for (int i = c_v4; i < pts_per_q; i += CH_V4) {
            lds_wei[i] = data_attn_weight[sampling_base + i];
            reinterpret_cast<float2*>(lds_loc)[i] = reinterpret_cast<const float2*>(data_sampling_loc)[sampling_base + i];
        }
    }
    __syncthreads();

    if (q >= num_query) return;

    const int grad_col_idx = ((b * num_query + q) * num_heads + m) * CH_V4 + c_v4;
    const float4 raw_top_g = reinterpret_cast<const float4*>(grad_col)[grad_col_idx];

    const int qid_str_v4 = num_heads * CH_V4;
    const int val_init_off_v4 = b * spatial_size * qid_str_v4;
    const float4* v_ptr_base = reinterpret_cast<const float4*>(data_value);
    float* g_v_ptr = reinterpret_cast<float*>(grad_value);

    for (int l = 0; l < num_levels; ++l) {
        const int s_h = s_shapes[l * 2], s_w = s_shapes[l * 2 + 1];
        const int level_offset_v4 = val_init_off_v4 + s_level_start[l] * qid_str_v4;
        const float4* v_ptr = v_ptr_base + level_offset_v4;
        float* g_v_level_ptr = g_v_ptr + level_offset_v4 * 4;

        for (int p = 0; p < num_point; ++p) {
            const int pts_idx = l * num_point + p;
            const float weight = lds_wei[pts_idx];
            const float loc_w  = lds_loc[pts_idx * 2];
            const float loc_h  = lds_loc[pts_idx * 2 + 1];

            const float h_im = loc_h * (float)s_h - 0.5f;
            const float w_im = loc_w * (float)s_w - 0.5f;

            float g_h = 0, g_w = 0, g_a = 0;

            if (h_im > -1.0f && w_im > -1.0f && h_im < (float)s_h && w_im < (float)s_w) {
                const int h_l = __float2int_rd(h_im), w_l = __float2int_rd(w_im);
                const float lh = h_im - h_l, lw = w_im - w_l;
                const float hh = 1.f - lh, hw = 1.f - lw;
                const int w_str = num_heads * CH_V4, h_str = s_w * w_str, base = m * CH_V4 + c_v4;

                const float w_arr[4] = {hh * hw, hh * lw, lh * hw, lh * lw};
                const int r_arr[4] = {h_l, h_l, h_l + 1, h_l + 1};
                const int c_arr[4] = {w_l, w_l + 1, w_l, w_l + 1};

                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (r_arr[i] >= 0 && r_arr[i] < s_h && c_arr[i] >= 0 && c_arr[i] < s_w) {
                        int ptr_idx = r_arr[i] * h_str + c_arr[i] * w_str + base;
                        float4 v = v_ptr[ptr_idx];
                        
                        float4_vec g_vec = { 
                            raw_top_g.x * weight * w_arr[i], raw_top_g.y * weight * w_arr[i],
                            raw_top_g.z * weight * w_arr[i], raw_top_g.w * weight * w_arr[i] 
                        };
                        __builtin_hcu_global_atomic_fadd_f32_x4(g_v_level_ptr + ptr_idx * 4, g_vec);
                        float gh_w = (i < 2 ? -1.f : 1.f) * (i % 2 == 0 ? hw : lw);
                        float gw_w = (i % 2 == 0 ? -1.f : 1.f) * (i < 2 ? hh : lh);
                        
                        float dot_v = (v.x * raw_top_g.x + v.y * raw_top_g.y + v.z * raw_top_g.z + v.w * raw_top_g.w);
                        g_h += dot_v * gh_w * weight * (float)s_h;
                        g_w += dot_v * gw_w * weight * (float)s_w;
                        g_a += dot_v * w_arr[i];
                    }
                }
            }

            // Warp 归约
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                g_h += __shfl_down(g_h, offset, CH_V4);
                g_w += __shfl_down(g_w, offset, CH_V4);
                g_a += __shfl_down(g_a, offset, CH_V4);
            }

            if (c_v4 == 0) {
                const int out_idx = sampling_base + pts_idx;
                grad_sampling_loc[out_idx * 2] = g_w;
                grad_sampling_loc[out_idx * 2 + 1] = g_h;
                grad_attn_weight[out_idx] = g_a;
            }
        }
    }
#endif
}

template <typename scalar_t, unsigned int blockSize>
__launch_bounds__(1024, 1) __global__
void ms_deformable_col2im_gpu_kernel_shm_g2s(
    const int n, const scalar_t* grad_col, const scalar_t* data_value,
    const int64_t* data_spatial_shapes, const int64_t* data_level_start_index,
    const scalar_t* data_sampling_loc, const scalar_t* data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t* grad_value, scalar_t* grad_sampling_loc,
    scalar_t* grad_attn_weight) {

    const int block_dim = blockSize;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 64;
    const unsigned int lane_id = tid % 64;
    const unsigned int tid_in_pos = tid % channels;
    const unsigned int reduce_pos_id = tid / channels;
    const int qid_stride = num_heads * channels;

    extern __shared__ unsigned char lds_base[];

    scalar_t* lds_wei_loc = reinterpret_cast<scalar_t*>(lds_base);

    // extern __shared__ scalar_t lds_wei_loc[];

    const int kgroup = block_dim / channels;
    const int group_idx = tid / channels;

    const int wei_length = kgroup * num_levels * num_point;
    const int loc_length = wei_length << 1;
    const int kAccessPerThread = (wei_length + block_dim - 1) / block_dim;

    scalar_t* lds_wei = lds_wei_loc;
    scalar_t* lds_loc = (lds_wei_loc + wei_length);
    const int attn_wei_valid_size = batch_size * num_query * num_heads * num_levels * num_point;

    CUDA_1D_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int c_col = _temp % channels;
        _temp /= channels;
        const int sampling_index = _temp;
        const int m_col = _temp % num_heads;
        _temp /= num_heads;
        _temp /= num_query;
        const int b_col = _temp;
        
        const scalar_t top_grad = grad_col[index];
        int data_weight_ptr = sampling_index * num_levels * num_point;
        int data_loc_w_ptr = data_weight_ptr << 1;

        const int grad_sampling_ptr = data_weight_ptr;
        scalar_t* grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr << 1);
        scalar_t* grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
        const int grad_weight_stride = 1;
        const int grad_loc_stride = 2;
        const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

        const int gmem_wei_init_offset = 
          (index / blockDim.x) * wei_length;

        const int gmem_loc_init_offset = 
          gmem_wei_init_offset << 1;

        int wei_g2s_idx = tid;
        int loc_g2s_idx = wei_g2s_idx << 1;

        #pragma unroll
        for (int k = 0; k < kAccessPerThread; ++k) {
          //* 不要跳到下一个块的点 && 尾块访存不要越界
          if  ((wei_g2s_idx < wei_length) && 
               (wei_g2s_idx + gmem_wei_init_offset) < attn_wei_valid_size) {
            *(lds_wei + wei_g2s_idx) = data_attn_weight[gmem_wei_init_offset + wei_g2s_idx];
            *(lds_loc + loc_g2s_idx) = data_sampling_loc[gmem_loc_init_offset + loc_g2s_idx];
            *(lds_loc + loc_g2s_idx + 1) = data_sampling_loc[gmem_loc_init_offset + loc_g2s_idx + 1];
          }
          wei_g2s_idx += (blockDim.x);
          loc_g2s_idx += (blockDim.x * 2);
       }

        int ds_wei_ptr = group_idx * num_levels * num_point;
        int ds_loc_ptr = ds_wei_ptr << 1;

        __syncthreads();

        for (int l_col = 0; l_col < num_levels; ++l_col) {
            const int level_start_id = int(data_level_start_index[l_col]);
            const int spatial_h_ptr = l_col << 1;
            const int spatial_h = int(data_spatial_shapes[spatial_h_ptr]);
            const int spatial_w = int(data_spatial_shapes[spatial_h_ptr + 1]);
            
            const int value_ptr_offset = data_value_ptr_init_offset + 
                                        level_start_id * qid_stride;
            const scalar_t* data_value_ptr = data_value + value_ptr_offset;
            scalar_t* grad_value_ptr = grad_value + value_ptr_offset;
            
            // 遍历 point
            for (int p_col = 0; p_col < num_point; ++p_col) {
                const scalar_t loc_w = *(lds_loc + ds_loc_ptr);
                const scalar_t loc_h = *(lds_loc + ds_loc_ptr + 1);
                const scalar_t weight = *(lds_wei + ds_wei_ptr);

                const scalar_t h_im = loc_h * (scalar_t)spatial_h - (scalar_t)0.5;
                const scalar_t w_im = loc_w * (scalar_t)spatial_w - (scalar_t)0.5;
                
                // 线程本地容器
                scalar_t grad_sh[2];
                grad_sh[0] = (scalar_t)0;
                grad_sh[1] = (scalar_t)0;
                scalar_t grad_a = (scalar_t)0;
                
                if (h_im > -1 && w_im > -1 && 
                    h_im < (scalar_t)spatial_h && 
                    w_im < (scalar_t)spatial_w) {
                    ms_deform_attn_col2im_bilinear<scalar_t>(
                        data_value_ptr, spatial_h, spatial_w, num_heads, 
                        channels, h_im, w_im, m_col, c_col, top_grad, 
                        weight, grad_value_ptr, grad_sh, &grad_a);
                }
                
                if constexpr (std::is_same_v<scalar_t, float>) {
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad_sh[0] += __shfl_down(grad_sh[0], offset, channels);
                        grad_sh[1] += __shfl_down(grad_sh[1], offset, channels);
                        grad_a += __shfl_down(grad_a, offset, channels);
                    }
                } else if constexpr (std::is_same_v<scalar_t, double>) {
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad_sh[0] += __shfl_down(grad_sh[0], offset, channels);
                        grad_sh[1] += __shfl_down(grad_sh[1], offset, channels);
                        grad_a += __shfl_down(grad_a, offset, channels);
                    }
                } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                    // c10::Half 转 float 做 shuffle
                    float grad0 = __half2float(grad_sh[0]);
                    float grad1 = __half2float(grad_sh[1]);
                    float grada = __half2float(grad_a);
                    
                    for (int offset = (channels/2); offset > 0; offset /= 2) {
                        grad0 += __shfl_down(grad0, offset, channels);
                        grad1 += __shfl_down(grad1, offset, channels);
                        grada += __shfl_down(grada, offset, channels);
                    }
                    
                    grad_sh[0] = __float2half(grad0);
                    grad_sh[1] = __float2half(grad1);
                    grad_a = __float2half(grada);
                } else {
                    static_assert(sizeof(scalar_t) == 0, 
                                 "Unsupported scalar type for shuffle reduce");
                }
                
                // 每个 reduce_pos_id 的第一个线程写回结果
                if (tid_in_pos == 0) {
                    *grad_sampling_loc_out = grad_sh[0];
                    *(grad_sampling_loc_out + 1) = grad_sh[1];
                    *grad_attn_weight_out = grad_a;
                }
                
                __syncthreads();
                
                ds_wei_ptr += 1;
                ds_loc_ptr += 2;
                grad_attn_weight_out += grad_weight_stride;
                grad_sampling_loc_out += grad_loc_stride;
            }  // p_col
        }  // l_col
        __syncthreads();
    }  // CUDA_1D_KERNEL_LOOP
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
  __shared__ scalar_t cache_grad_attn_weight[blockSize];
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__launch_bounds__(1024, 1)
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2_opt(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {

  __shared__ scalar_t shm_grad_sampling_loc[blockSize * 2/64];
  __shared__ scalar_t shm_grad_attn_weight[blockSize/64];

  const unsigned int tid = threadIdx.x;
  const unsigned int lane_id = tid % 64;   // warp 内 lane
  const unsigned int warp_id = tid / 64;   // warp id
  const unsigned int tid_in_pos = tid % channels;

  const int qid_stride = num_heads * channels;

  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        // 线程本地容器
        scalar_t grad_sh[2];
        grad_sh[0] = (scalar_t)0;
        grad_sh[1] = (scalar_t)0;
        scalar_t grad_a = (scalar_t)0;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels,
              h_im, w_im, m_col, c_col, top_grad, weight,grad_value_ptr,
              grad_sh, &grad_a); // 注意改成传寄存器地址
        }

        // ---------------------------
        // Stage 1: warp 内规约（寄存器 shuffle）
        // ---------------------------
        if constexpr (std::is_same_v<scalar_t, float>) {
            for (int offset = 32; offset > 0; offset /= 2) {
                grad_sh[0] += __shfl_down(grad_sh[0], offset, 64);
                grad_sh[1] += __shfl_down(grad_sh[1], offset, 64);
                grad_a += __shfl_down(grad_a, offset, 64);
            }
        } else if constexpr (std::is_same_v<scalar_t, double>) {
            for (int offset = 32; offset > 0; offset /= 2) {
                grad_sh[0] += __shfl_down(grad_sh[0], offset, 64);
                grad_sh[1] += __shfl_down(grad_sh[1], offset, 64);
                grad_a += __shfl_down(grad_a, offset, 64);
            }
        } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            // c10::Half 转 float 做 shuffle
            float grad0 = __half2float(grad_sh[0]);
            float grad1 = __half2float(grad_sh[1]);
            float grada = __half2float(grad_a);
            
            for (int offset = 32; offset > 0; offset /= 2) {
                grad0 += __shfl_down(grad0, offset, 64);
                grad1 += __shfl_down(grad1, offset, 64);
                grada += __shfl_down(grada, offset, 64);
            }
            
            grad_sh[0] = __float2half(grad0);
            grad_sh[1] = __float2half(grad1);
            grad_a = __float2half(grada);
        } else {
            static_assert(sizeof(scalar_t) == 0, 
                         "Unsupported scalar type for shuffle reduce");
        }
        
        // 每个 reduce_pos_id 的第一个线程写回结果
        if (lane_id == 0) {
          shm_grad_sampling_loc[warp_id * 2] = grad_sh[0];
          shm_grad_sampling_loc[warp_id * 2 + 1] = grad_sh[1];
          shm_grad_attn_weight[warp_id] = grad_a;
        }

        __syncthreads();

        // ---------------------------
        // Stage 2: 共享内存规约，tid_in_pos==0 写回全局
        // ---------------------------

        if (tid_in_pos == 0) {
          scalar_t sum0 = 0, sum1 = 0, suma = 0;
          const int num_warps_for_channel = (channels + 63) / 64;
          for (int w = 0; w < num_warps_for_channel; ++w) {
              sum0 += shm_grad_sampling_loc[warp_id * 2+w * 2];
              sum1 += shm_grad_sampling_loc[warp_id * 2+w * 2 + 1];
              suma += shm_grad_attn_weight[warp_id+w];
          }

          *grad_sampling_loc_out = sum0;
          *(grad_sampling_loc_out + 1) = sum1;
          *grad_attn_weight_out = suma;
        }

        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int _tid = 1; _tid < blockDim.x; ++_tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[_tid];
            sid += 2;
          }

          *grad_sampling_loc_out = _grad_w;
          *(grad_sampling_loc_out + 1) = _grad_h;
          *grad_attn_weight_out = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc_out = cache_grad_sampling_loc[0];
          *(grad_sampling_loc_out + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight_out = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  extern __shared__ int _s[];
  scalar_t *cache_grad_sampling_loc = reinterpret_cast<scalar_t *>(_s);
  scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
  unsigned int tid = threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc_out, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc_out + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight_out, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(
    const int n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    scalar_t *grad_sampling_loc_out =
        grad_sampling_loc + (grad_sampling_ptr << 1);
    scalar_t *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          ms_deform_attn_col2im_bilinear_gm(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              grad_sampling_loc_out, grad_attn_weight_out);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  // 判断是否使用优化的kernel（channels == 32 且类型为 float）
  constexpr bool is_float = std::is_same<scalar_t, float>::value;
  
  if (channels == 32 && is_float) {
      // 使用优化的 3D grid kernel
    constexpr int vec_size = 4;                 // float4 向量大小
    const int threads_per_pixel = channels / vec_size; // 每个 Query 需要的线程数：32 / 4 = 8
    const int threads_per_block = 256;          // 256 线程
    const int queries_per_block = threads_per_block / threads_per_pixel; // 现在每个 Block 能处理 32 个 Query

    dim3 grid_size;
    grid_size.x = (num_query + queries_per_block - 1) / queries_per_block; // 修正后的分块数
    grid_size.y = num_heads;
    grid_size.z = batch_size;

    const int metadata_smem = num_levels * 3 * sizeof(int); 
    const int data_smem = queries_per_block * num_levels * num_point * 3 * sizeof(scalar_t);
    const int smem_size = metadata_smem + data_smem;

    // 4. 发起调用
    ms_deformable_im2col_gpu_kernel_3d_grid<scalar_t, threads_per_block>
        <<<grid_size, threads_per_block, smem_size, stream>>>(
            data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_point, 
            data_col);
  } else {
    const int num_kernels = batch_size * num_query * num_heads * channels;
    const int num_actual_kernels = batch_size * num_query * num_heads * channels;
    const int num_threads = 1024;
    ms_deformable_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_point, data_col);
  }
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  constexpr bool is_float = std::is_same<scalar_t, float>::value;
  const int num_threads =
      (channels > THREADS_PER_BLOCK) ? THREADS_PER_BLOCK : channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > THREADS_PER_BLOCK) {
    if ((channels & THREADS_PER_BLOCK - 1) == 0) {
      if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
      printf("进入channel > 512\n");
      printf("调用default kernel\n");}
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
             num_threads * 3 * sizeof(scalar_t), stream>>>(
              num_kernels, grad_col, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight,
              batch_size, spatial_size, num_heads, channels, num_levels,
              num_query, num_point, grad_value, grad_sampling_loc,
              grad_attn_weight);
    } else {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
    }
  } else {
    switch (channels) {
      case 1:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 1\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 2:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 2\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      2>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 4:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 4\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      4>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 8:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 8\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      8>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 16:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 16\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 32: 
            if (GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH", false)) {
                printf("进入case 32\n");
                printf("调用优化后kernel\n");
            }

            if (channels == 32 && is_float) {
                constexpr int vec_size = 4;
                const int threads_per_pixel = channels / vec_size;
                const int threads_per_block = 256;
                const int queries_per_block = threads_per_block / threads_per_pixel;
                const int metadata_size = num_levels * 3 * sizeof(int);
                const int sampling_data_size = queries_per_block * num_levels * num_point * 3 * sizeof(float);
                const int smem_size = metadata_size + sampling_data_size;
                dim3 grid_size((num_query + queries_per_block - 1) / queries_per_block, num_heads, batch_size);
                ms_deformable_col2im_gpu_kernel_3d_grid<scalar_t, 256>
                    <<<grid_size, threads_per_block, smem_size, stream>>>(
                        grad_col, data_value, data_spatial_shapes, data_level_start_index,
                        data_sampling_loc, data_attn_weight, batch_size, spatial_size,
                        num_heads, channels, num_levels, num_query, num_point,
                        grad_value, grad_sampling_loc, grad_attn_weight);
            } else {
                const int shm_required = (1024 / channels) * num_levels * num_point * sizeof(scalar_t) * 3;
                if (shm_required >= 16384) {
                    ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t, 1024>
                        <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0, stream>>>(
                            num_kernels, grad_col, data_value, data_spatial_shapes,
                            data_level_start_index, data_sampling_loc, data_attn_weight,
                            batch_size, spatial_size, num_heads, channels, num_levels,
                            num_query, num_point, grad_value,
                            grad_sampling_loc, grad_attn_weight);
                } else {
                    ms_deformable_col2im_gpu_kernel_shm_g2s<scalar_t, 1024>
                        <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, shm_required, stream>>>(
                            num_kernels, grad_col, data_value, data_spatial_shapes,
                            data_level_start_index, data_sampling_loc, data_attn_weight,
                            batch_size, spatial_size, num_heads, channels, num_levels,
                            num_query, num_point, grad_value,
                            grad_sampling_loc, grad_attn_weight);
                }
            }
        break;
      case 64:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 64\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 128:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 128\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 256:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 256\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 512:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入case 512\n");printf("调用优化后kernel\n");}
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2_opt<scalar_t,
                                                                      1024>
            <<<GET_BLOCKS(num_actual_kernels, 1024), 1024, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      default:
        if(GET_BOOL_FROM_ENV("CHECK_KERNEL_LAUNCH",false)){
        printf("进入default\n");printf("调用default kernel\n");}
        if (channels < 64) {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        } else {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step) {

  // ===============================
  // Checks
  // ===============================
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  auto output =
      at::zeros({batch, num_query, num_heads, channels}, value.options());

  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
        ms_deformable_im2col_cuda(
            at::cuda::getCurrentCUDAStream(),
            value.data_ptr<scalar_t>(),
            spatial_shapes.data_ptr<int64_t>(),
            level_start_index.data_ptr<int64_t>(),
            sampling_loc.data_ptr<scalar_t>(),
            attn_weight.data_ptr<scalar_t>(),
            batch,            // ← 全 batch
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            output.data_ptr<scalar_t>());
      }));

  output = output.view({batch, num_query, num_heads * channels});

  return output;
}

void ms_deform_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  AT_ASSERTM(value.is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
  AT_ASSERTM(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
  AT_ASSERTM(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  auto grad_output_ =
      grad_output.view({batch, num_query, num_heads, channels});

  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "ms_deform_attn_backward_cuda_fused", ([&] {
        ms_deformable_col2im_cuda(
            at::cuda::getCurrentCUDAStream(),
            grad_output_.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            spatial_shapes.data_ptr<int64_t>(),
            level_start_index.data_ptr<int64_t>(),
            sampling_loc.data_ptr<scalar_t>(),
            attn_weight.data_ptr<scalar_t>(),
            batch,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            grad_value.data_ptr<scalar_t>(),
            grad_sampling_loc.data_ptr<scalar_t>(),
            grad_attn_weight.data_ptr<scalar_t>());
      }));
}
}
}




