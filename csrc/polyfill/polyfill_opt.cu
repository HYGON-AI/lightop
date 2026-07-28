#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Loops.cuh>
#include <cmath>

constexpr float LOG2_E = 1.44269504089f; 
#define EXP_FUNC(x) __expf(x)


namespace at {
namespace native {



const int MODE_BOUNDARY  = 0;
const int MODE_MASK      = 1;
const int MODE_HARD_MASK = 2;
const int TILE_SIZE = 256; 

template <typename scalar_t>
__device__ __forceinline__ scalar_t fast_clamp(scalar_t v) {
    return fmaxf(scalar_t(0), fminf(scalar_t(1), v));
}

template <typename scalar_t>
__global__ void forward_polyfill_hard_mask_kernel(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int height,
    int width
) {
    extern __shared__ char shared_mem[];
    scalar_t* verts_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vertex_base = bi * nv * 2;

    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        verts_shared[v] = vertices[vertex_base + v];
    }
    __syncthreads();

    if (local_idx >= pixels_per_batch) return;

    const int idx = bi * pixels_per_batch + local_idx;
    const int yp = local_idx / width;
    const int xp = local_idx % width;

    int c = 0;
    for (int vn = 0, j = nv - 1; vn < nv; j = vn++) {
        scalar_t from_x = verts_shared[vn * 2];
        scalar_t from_y = verts_shared[vn * 2 + 1];
        scalar_t to_x   = verts_shared[j * 2];
        scalar_t to_y   = verts_shared[j * 2 + 1];

        if (((from_y > yp) != (to_y > yp)) &&
            (xp < (to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x)) {
            c = !c;
        }
    }
    
    output[idx] = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
}


template <typename scalar_t>
__global__ void forward_polyfill_hard_mask_kernel_large(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int height,
    int width
) {
    extern __shared__ char shared_mem[];
    scalar_t* verts_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vertex_base = bi * number_vertices * 2;

    if (local_idx >= pixels_per_batch) return;


    const int yp = local_idx / width;
    const int xp = local_idx % width;

    const int idx = bi * pixels_per_batch + local_idx;

    scalar_t to_x = vertices[vertex_base + (number_vertices - 1) * 2];
    scalar_t to_y = vertices[vertex_base + (number_vertices - 1) * 2 + 1];
    
    int to_gt = (to_y > yp); 

    int c = 0; 

    for (int chunk_base = 0; chunk_base < number_vertices; chunk_base += TILE_SIZE) {
        int limit = std::min(TILE_SIZE, number_vertices - chunk_base);
        for (int t = threadIdx.x; t < limit * 2; t += blockDim.x) {
            verts_shared[t] = vertices[vertex_base + (chunk_base * 2) + t];
        }
        __syncthreads();
        #pragma unroll 4
        for (int t = 0; t < limit; ++t) {
            scalar_t from_x = verts_shared[t * 2];
            scalar_t from_y = verts_shared[t * 2 + 1];
            int from_gt = (from_y > yp);

            if (from_gt != to_gt) {
                scalar_t min_x = (from_x < to_x) ? from_x : to_x;
                scalar_t max_x = (from_x < to_x) ? to_x : from_x;

                if (xp < min_x) {
                    c ^= 1;
                } else if (xp < max_x) {
                    if (xp < (to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x) {
                        c ^= 1;
                    }
                }
            }
            to_x = from_x;
            to_y = from_y;
            to_gt = from_gt;
        }
        __syncthreads();
    }

    output[idx] = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
}

template <typename scalar_t, int MODE, int THREADS_PER_PIXEL>
__global__ void forward_polyfill_general_kernel_large_opt_mutli_threads(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int* __restrict__ nearest_edge_map,
    int height,
    int width,
    float inv_smoothness
) {
    extern __shared__ char shared_mem[];
    scalar_t* s_mem = reinterpret_cast<scalar_t*>(shared_mem);
    
    scalar_t* s_v        = s_mem; 
    scalar_t* s_edge_dx = s_v + (TILE_SIZE + 1) * 2;
    scalar_t* s_edge_dy = s_edge_dx + TILE_SIZE;
    scalar_t* s_inv      = s_edge_dy + TILE_SIZE;

    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int vertex_base = bi * number_vertices * 2;

    const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int pixel_idx = global_thread_idx / THREADS_PER_PIXEL;
    const int lane_idx  = threadIdx.x % THREADS_PER_PIXEL;
    
    bool active = pixel_idx < pixels_per_batch;

    scalar_t xp = 0, yp = 0;
    if (active) {
        int yp_int = pixel_idx / width;
        int xp_int = pixel_idx % width;
        xp = static_cast<scalar_t>(xp_int);
        yp = static_cast<scalar_t>(yp_int);
    }

    scalar_t min_dist_sq = scalar_t(1e10);
    int closest_edge = 0;
    int c = 0;

    for (int chunk_base = 0; chunk_base < number_vertices; chunk_base += TILE_SIZE) {
        int limit = std::min(TILE_SIZE, number_vertices - chunk_base);

        for (int t = threadIdx.x; t < limit; t += blockDim.x) {
            s_v[t * 2]     = vertices[vertex_base + (chunk_base + t) * 2];
            s_v[t * 2 + 1] = vertices[vertex_base + (chunk_base + t) * 2 + 1];
        }

        if (threadIdx.x == 0) {
            int next_idx = (chunk_base + limit) % number_vertices;
            s_v[limit * 2]     = vertices[vertex_base + next_idx * 2];
            s_v[limit * 2 + 1] = vertices[vertex_base + next_idx * 2 + 1];
        }
        __syncthreads();

        for (int t = threadIdx.x; t < limit; t += blockDim.x) {
            scalar_t dx = s_v[(t + 1) * 2] - s_v[t * 2];
            scalar_t dy = s_v[(t + 1) * 2 + 1] - s_v[t * 2 + 1];
            scalar_t len_sq = fmaf(dx, dx, fmaf(dy, dy, scalar_t(1e-6)));
            
            s_edge_dx[t] = dx;
            s_edge_dy[t] = dy;
            s_inv[t] = scalar_t(1.0) / len_sq;
        }
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int j = lane_idx; j < limit; j += THREADS_PER_PIXEL) {
                scalar_t edge_dx = s_edge_dx[j];
                scalar_t edge_dy = s_edge_dy[j];
                scalar_t inv_sq  = s_inv[j];
                scalar_t curr_vx = s_v[j * 2];
                scalar_t curr_vy = s_v[j * 2 + 1];
                
                scalar_t dx_v = xp - curr_vx;
                scalar_t dy_v = yp - curr_vy;

                scalar_t dot = fmaf(dx_v, edge_dx, dy_v * edge_dy);
                scalar_t t = dot * inv_sq;
                t = fmaxf(scalar_t(0), fminf(scalar_t(1), t));

                scalar_t diff_x = fmaf(-t, edge_dx, dx_v);
                scalar_t diff_y = fmaf(-t, edge_dy, dy_v);
                scalar_t dist_sq_segment = fmaf(diff_x, diff_x, diff_y * diff_y);

                if (dist_sq_segment < min_dist_sq) {
                    min_dist_sq = dist_sq_segment;
                    closest_edge = chunk_base + j;
                }

                if constexpr (MODE == MODE_MASK) {
                    scalar_t next_vy = s_v[(j + 1) * 2 + 1];
                    bool c1 = curr_vy > yp;
                    bool c2 = next_vy > yp;
                    if (c1 != c2) { 
                        scalar_t lhs = dx_v * edge_dy;
                        scalar_t rhs = edge_dx * dy_v;
                        if (edge_dy > scalar_t(0)) {
                            if (lhs < rhs) c = !c;
                        } else {
                            if (lhs > rhs) c = !c;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        #pragma unroll
        for (int offset = THREADS_PER_PIXEL / 2; offset > 0; offset /= 2) {
            scalar_t remote_dist = __shfl_down( min_dist_sq, offset);
            int remote_edge = __shfl_down( closest_edge, offset);
            int remote_c = __shfl_down( c, offset);

            if (remote_dist < min_dist_sq) {
                min_dist_sq = remote_dist;
                closest_edge = remote_edge;
            }
            if constexpr (MODE == MODE_MASK) {
                c ^= remote_c;
            }
        }
        if (lane_idx == 0) {
            int idx = bi * pixels_per_batch + pixel_idx;
            if constexpr (MODE == MODE_MASK) {
                scalar_t in_out_sign = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
                output[idx] = scalar_t(1.0) / (scalar_t(1.0) + EXP_FUNC(-in_out_sign * min_dist_sq / inv_smoothness));
            } else {
                output[idx] = EXP_FUNC(-min_dist_sq / inv_smoothness);
            }
            nearest_edge_map[idx] = closest_edge;
        }
    }
}

template <typename scalar_t, int MODE>
__global__ void forward_polyfill_general_kernel_opt(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int* __restrict__ nearest_edge_map,
    int height,
    int width,
    float inv_smoothness
) {
    extern __shared__ char shared_mem[];
    scalar_t* s_mem = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* verts_shared = s_mem;
    scalar_t* inv_len_sq_shared = s_mem + 2 * number_vertices;

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int vertex_base = bi * nv * 2;

    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        verts_shared[v] = vertices[vertex_base + v];
    }
    __syncthreads(); 

    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        int next_i = (i + 1 < nv) ? i + 1 : 0;
        scalar_t dx = verts_shared[next_i * 2] - verts_shared[i * 2];
        scalar_t dy = verts_shared[next_i * 2 + 1] - verts_shared[i * 2 + 1];
        scalar_t len_sq = fmaf(dx, dx, fmaf(dy, dy, scalar_t(1e-6)));
        inv_len_sq_shared[i] = scalar_t(1.0) / len_sq; 
    }
    __syncthreads(); 

    if (local_idx >= pixels_per_batch) return;

    const int idx = bi * pixels_per_batch + local_idx;
    const int yp_int = local_idx / width;
    const int xp_int = local_idx % width;
    const scalar_t xp = static_cast<scalar_t>(xp_int);
    const scalar_t yp = static_cast<scalar_t>(yp_int);

    scalar_t min_dist_sq = scalar_t(1e10);
    int closest_edge = 0;
    int c = 0; 

    scalar_t curr_vx = verts_shared[0];
    scalar_t curr_vy = verts_shared[1];
    scalar_t dx_v = xp - curr_vx;
    scalar_t dy_v = yp - curr_vy;

    #pragma unroll 4
    for (int j = 0; j < nv; ++j) {
        int next_j = (j + 1 < nv) ? j + 1 : 0;
        
        scalar_t next_vx = verts_shared[next_j * 2];
        scalar_t next_vy = verts_shared[next_j * 2 + 1];

        scalar_t edge_dx = next_vx - curr_vx;
        scalar_t edge_dy = next_vy - curr_vy;

        scalar_t ndx_v = xp - next_vx;
        scalar_t ndy_v = yp - next_vy;

        scalar_t dot = fmaf(dx_v, edge_dx, dy_v * edge_dy);
        scalar_t t = dot * inv_len_sq_shared[j];

        t = fmaxf(scalar_t(0), fminf(scalar_t(1), t));

        scalar_t diff_x = fmaf(-t, edge_dx, dx_v);
        scalar_t diff_y = fmaf(-t, edge_dy, dy_v);
        
        scalar_t dist_sq_segment = fmaf(diff_x, diff_x, diff_y * diff_y);

        if (dist_sq_segment < min_dist_sq) {
            min_dist_sq = dist_sq_segment;
            closest_edge = j;
        }

        if constexpr (MODE == MODE_MASK) {
            bool c1 = curr_vy > yp;
            bool c2 = next_vy > yp;
            if (c1 != c2) { 
                scalar_t lhs = dx_v * edge_dy;
                scalar_t rhs = edge_dx * dy_v;
                if (edge_dy > scalar_t(0)) {
                    if (lhs < rhs) c = !c;
                } else {
                    if (lhs > rhs) c = !c;
                }
            }
        }

        curr_vx = next_vx;
        curr_vy = next_vy;
        dx_v = ndx_v;
        dy_v = ndy_v;
    }

    if constexpr (MODE == MODE_MASK) {
        scalar_t in_out_sign = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
        output[idx] = scalar_t(1.0) / (scalar_t(1.0) + EXP_FUNC(-in_out_sign * min_dist_sq / inv_smoothness));
    } else {
        output[idx] = EXP_FUNC(-min_dist_sq / inv_smoothness);
    }
    
    nearest_edge_map[idx] = closest_edge;
}
template <typename scalar_t, int MODE>
__global__ void forward_polyfill_general_kernel_large_opt(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int* __restrict__ nearest_edge_map,
    int height,
    int width,
    float inv_smoothness
) {
    extern __shared__ char shared_mem[];
    scalar_t* s_mem = reinterpret_cast<scalar_t*>(shared_mem);
    
    scalar_t* s_v       = s_mem; 
    scalar_t* s_edge_dx = s_v + (TILE_SIZE + 1) * 2;
    scalar_t* s_edge_dy = s_edge_dx + TILE_SIZE;
    scalar_t* s_inv     = s_edge_dy + TILE_SIZE;

    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vertex_base = bi * number_vertices * 2;

    if (local_idx >= pixels_per_batch) return;

    const int idx = bi * pixels_per_batch + local_idx;
    const int yp_int = local_idx / width;
    const int xp_int = local_idx % width;
    const scalar_t xp = static_cast<scalar_t>(xp_int);
    const scalar_t yp = static_cast<scalar_t>(yp_int);

  
    scalar_t min_dist_sq = scalar_t(1e10);
    int closest_edge = 0;
    int c = 0;
    for (int chunk_base = 0; chunk_base < number_vertices; chunk_base += TILE_SIZE) {
        int limit = std::min(TILE_SIZE, number_vertices - chunk_base);
        using Vec2 = typename std::conditional<sizeof(scalar_t) == 4, float2, double2>::type;
        const Vec2* v_ptr = reinterpret_cast<const Vec2*>(vertices + vertex_base);
      
    for (int t = threadIdx.x; t < limit; t += blockDim.x) {
            int curr_idx = chunk_base + t;
            int next_idx;
            
            if (t == limit - 1) {
                next_idx = (chunk_base + limit) % number_vertices;
            } else {
                next_idx = curr_idx + 1;
            }
    
            Vec2 v1 = v_ptr[curr_idx]; 
            Vec2 v2 = v_ptr[next_idx]; 

            scalar_t v1_x = v1.x;
            scalar_t v1_y = v1.y;
            scalar_t v2_x = v2.x;
            scalar_t v2_y = v2.y;

            s_v[t * 2]     = v1_x;
            s_v[t * 2 + 1] = v1_y;
            
            scalar_t dx = v2_x - v1_x;
            scalar_t dy = v2_y - v1_y;
            
            s_edge_dx[t] = dx;
            s_edge_dy[t] = dy;

            scalar_t len_sq = fmaf(dx, dx, fmaf(dy, dy, scalar_t(1e-6)));
            s_inv[t] = scalar_t(1.0) / len_sq;
        }

        __syncthreads();

        #pragma unroll 4
        for (int j = 0; j < limit; ++j) {
         
            scalar_t edge_dx = s_edge_dx[j];
            scalar_t edge_dy = s_edge_dy[j];
            scalar_t inv_sq  = s_inv[j];

            scalar_t curr_vx = s_v[j * 2];
            scalar_t curr_vy = s_v[j * 2 + 1];
            
            scalar_t dx_v = xp - curr_vx;
            scalar_t dy_v = yp - curr_vy;
            scalar_t dot = fmaf(dx_v, edge_dx, dy_v * edge_dy);
            scalar_t t = dot * inv_sq;

            t = fast_clamp(t);

            scalar_t diff_x = fmaf(-t, edge_dx, dx_v);
            scalar_t diff_y = fmaf(-t, edge_dy, dy_v);
            scalar_t dist_sq_segment = fmaf(diff_x, diff_x, diff_y * diff_y);

            if (dist_sq_segment < min_dist_sq) {
                min_dist_sq = dist_sq_segment;
                closest_edge = chunk_base + j;
            }

            if constexpr (MODE == MODE_MASK) {

                bool c1 = (-dy_v) > scalar_t(0); 
                
                scalar_t next_dy_v = dy_v - edge_dy;
                bool c2 = next_dy_v < scalar_t(0); 
                if (c1 != c2) { 
                    scalar_t lhs = dx_v * edge_dy;
                    scalar_t rhs = edge_dx * dy_v;
                    if (edge_dy > scalar_t(0)) {
                        if (lhs < rhs) c = !c;
                    } else {
                        if (lhs > rhs) c = !c;
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (MODE == MODE_MASK) {
        scalar_t in_out_sign = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
        output[idx] = scalar_t(1.0) / (scalar_t(1.0) + EXP_FUNC(-in_out_sign * min_dist_sq / inv_smoothness));
    } else {
        output[idx] = EXP_FUNC(-min_dist_sq / inv_smoothness);
    }
    
    nearest_edge_map[idx] = closest_edge;
}


template <typename scalar_t, int MODE>
__global__ void forward_polyfill_general_kernel_large(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int* __restrict__ nearest_edge_map,
    int height,
    int width,
    float inv_smoothness
) {
    extern __shared__ char shared_mem[];
    scalar_t* s_mem = reinterpret_cast<scalar_t*>(shared_mem);
    scalar_t* s_v = s_mem; 
    scalar_t* s_inv = s_mem + (TILE_SIZE + 1) * 2;

    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int vertex_base = bi * number_vertices * 2;

    if (local_idx >= pixels_per_batch) return;

    const int idx = bi * pixels_per_batch + local_idx;

    const int yp_int = local_idx / width;
    const int xp_int = local_idx % width;
    const scalar_t xp = static_cast<scalar_t>(xp_int);
    const scalar_t yp = static_cast<scalar_t>(yp_int);

    scalar_t min_dist_sq = scalar_t(1e10);
    int closest_edge = 0;
    int c = 0;

    scalar_t curr_vx = vertices[vertex_base + 0];
    scalar_t curr_vy = vertices[vertex_base + 1];
    scalar_t dx_v = xp - curr_vx;
    scalar_t dy_v = yp - curr_vy;
    //分块加载,共享内存不然放不下
    for (int chunk_base = 0; chunk_base < number_vertices; chunk_base += TILE_SIZE) {
        int limit = std::min(TILE_SIZE, number_vertices - chunk_base);

        for (int t = threadIdx.x; t < limit; t += blockDim.x) {
            s_v[t * 2]     = vertices[vertex_base + (chunk_base + t) * 2];
            s_v[t * 2 + 1] = vertices[vertex_base + (chunk_base + t) * 2 + 1];
        }

        if (threadIdx.x == 0) {
            int next_idx = (chunk_base + limit) % number_vertices;
            s_v[limit * 2]     = vertices[vertex_base + next_idx * 2];
            s_v[limit * 2 + 1] = vertices[vertex_base + next_idx * 2 + 1];
        }
        __syncthreads();

        for (int t = threadIdx.x; t < limit; t += blockDim.x) {
            scalar_t dx = s_v[(t + 1) * 2] - s_v[t * 2];
            scalar_t dy = s_v[(t + 1) * 2 + 1] - s_v[t * 2 + 1];
            scalar_t len_sq = fmaf(dx, dx, fmaf(dy, dy, scalar_t(1e-6)));
            s_inv[t] = scalar_t(1.0) / len_sq;
        }
        __syncthreads();

        for (int j = 0; j < limit; ++j) {
            scalar_t next_vx = s_v[(j + 1) * 2];
            scalar_t next_vy = s_v[(j + 1) * 2 + 1];

            scalar_t edge_dx = next_vx - curr_vx;
            scalar_t edge_dy = next_vy - curr_vy;

            scalar_t ndx_v = xp - next_vx;
            scalar_t ndy_v = yp - next_vy;

            scalar_t dot = fmaf(dx_v, edge_dx, dy_v * edge_dy);
            scalar_t t = dot * s_inv[j];
            t = fmaxf(scalar_t(0), fminf(scalar_t(1), t));

            scalar_t diff_x = fmaf(-t, edge_dx, dx_v);
            scalar_t diff_y = fmaf(-t, edge_dy, dy_v);
            scalar_t dist_sq_segment = fmaf(diff_x, diff_x, diff_y * diff_y);

            if (dist_sq_segment < min_dist_sq) {
                min_dist_sq = dist_sq_segment;
                closest_edge = chunk_base + j;
            }
            //无除法交叉相乘检测
            if constexpr (MODE == MODE_MASK) {
                bool c1 = curr_vy > yp;
                bool c2 = next_vy > yp;
                if (c1 != c2) { 
                    scalar_t lhs = dx_v * edge_dy;
                    scalar_t rhs = edge_dx * dy_v;
                    if (edge_dy > scalar_t(0)) {
                        if (lhs < rhs) c = !c;
                    } else {
                        if (lhs > rhs) c = !c;
                    }
                }
            }

            curr_vx = next_vx;
            curr_vy = next_vy;
            dx_v = ndx_v;
            dy_v = ndy_v;
        }
        __syncthreads();
    }

    if constexpr (MODE == MODE_MASK) {
        scalar_t in_out_sign = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
        output[idx] = scalar_t(1.0) / (scalar_t(1.0) + EXP_FUNC(-in_out_sign * min_dist_sq / inv_smoothness));
    } else {
        output[idx] = EXP_FUNC(-min_dist_sq / inv_smoothness);
    }
    
    nearest_edge_map[idx] = closest_edge;
}


std::vector<torch::Tensor> forward_polyfill_lightop(
    torch::Tensor vertices,
    torch::Tensor output,
    torch::Tensor nearest_edge_map,
    int width,
    int height,
    float inv_smoothness,
    int mode
) {
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);


    if (batch_size == 0) {
        return { output, nearest_edge_map };
    }

    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "forward_polyfill_fused", ([&] {
        
        // HARD_MASK
        if (mode == MODE_HARD_MASK) {
            int threads = 512;
            const int pixels_per_batch = width * height;
            const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;
            dim3 grid(blocks_per_batch, batch_size);
            size_t shared_size = TILE_SIZE * 2 * sizeof(scalar_t);

            forward_polyfill_hard_mask_kernel_large<scalar_t>
                <<<grid, threads, shared_size>>>(
                    vertices.data_ptr<scalar_t>(),
                    (int)number_vertices,
                    output.data_ptr<scalar_t>(),
                    height,
                    width
                );
            return; 
        }

        // 特定优化 [256x256]
        if (width <= 256 && number_vertices > 5000) {
            const int threads = 256;
            const int THREADS_PER_PIXEL = 4;
            const int pixels_per_batch = width * height;
            const int total_threads = pixels_per_batch * THREADS_PER_PIXEL;
            const int blocks_per_batch = (total_threads + threads - 1) / threads;
            dim3 grid(blocks_per_batch, batch_size);
            size_t shared_size = (5 * TILE_SIZE + 2) * sizeof(scalar_t);

            switch (mode) {
                case MODE_MASK:
                    forward_polyfill_general_kernel_large_opt_mutli_threads<scalar_t, MODE_MASK, THREADS_PER_PIXEL>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
                case MODE_BOUNDARY:
                default:
                    forward_polyfill_general_kernel_large_opt_mutli_threads<scalar_t, MODE_BOUNDARY, THREADS_PER_PIXEL>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
            }
            return; 
        }

        // 大顶点数通用
        if (number_vertices > 5000) {
            int threads = 256;
            const int pixels_per_batch = width * height;
            const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;
            dim3 grid(blocks_per_batch, batch_size);
            size_t shared_size = (5 * TILE_SIZE + 2) * sizeof(scalar_t);

            switch (mode) {
                case MODE_MASK:
                    forward_polyfill_general_kernel_large_opt<scalar_t, MODE_MASK>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
                case MODE_BOUNDARY:
                default:
                    forward_polyfill_general_kernel_large_opt<scalar_t, MODE_BOUNDARY>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
            }
            return; 
        }

        //  默认情况 
        {
            int threads = 256;
            const int pixels_per_batch = width * height;
            const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;
            dim3 grid(blocks_per_batch, batch_size);
            size_t shared_size = number_vertices * 3 * sizeof(scalar_t);

            switch (mode) {
                case MODE_MASK:
                    forward_polyfill_general_kernel_opt<scalar_t, MODE_MASK>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
                case MODE_BOUNDARY:
                default:
                    forward_polyfill_general_kernel_opt<scalar_t, MODE_BOUNDARY>
                        <<<grid, threads, shared_size>>>(
                            vertices.data_ptr<scalar_t>(), (int)number_vertices, output.data_ptr<scalar_t>(),
                            nearest_edge_map.data_ptr<int>(), height, width, inv_smoothness);
                    break;
            }
        }
    }));

    return { output, nearest_edge_map };
}

template <typename scalar_t, int K, bool VALID_VECTOR_LOAD>
__global__ void backward_polyfill_optimized_kernel(
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ output,
    const int* __restrict__ nearest_edge_map,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_vertices,
    int number_vertices,
    int width,
    int height,
    float inv_smoothness_param
) {
    
    extern __shared__ char shared_mem[];
    scalar_t* grad_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;

    const scalar_t inv_smoothness = static_cast<scalar_t>(inv_smoothness_param);

    const int start_local_idx = (blockIdx.x * blockDim.x + threadIdx.x) * K;
    const int vertex_base = bi * nv * 2;

    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        grad_shared[v] = scalar_t(0);
    }
    __syncthreads();

    int prev_vn = -1;
    
    scalar_t cache_from_x = 0, cache_from_y = 0;
    scalar_t cache_to_x = 0,   cache_to_y = 0;
    scalar_t cache_x2_sub_x1 = 0, cache_y2_sub_y1 = 0;
    scalar_t cache_sq_len_inv = 0;


    scalar_t acc_grad_x1 = 0;
    scalar_t acc_grad_y1 = 0;
    scalar_t acc_grad_x2 = 0;
    scalar_t acc_grad_y2 = 0;


    auto commit_gradients_to_shared = [&](int vn) {
        if (vn == -1) return;
        const int from_idx = vn;
        const int to_idx = (vn + 1 < nv) ? vn + 1 : 0;
        
        if (acc_grad_x1 != scalar_t(0)) atomicAdd(&grad_shared[from_idx * 2],     acc_grad_x1);
        if (acc_grad_y1 != scalar_t(0)) atomicAdd(&grad_shared[from_idx * 2 + 1], acc_grad_y1);
        if (acc_grad_x2 != scalar_t(0)) atomicAdd(&grad_shared[to_idx * 2],       acc_grad_x2);
        if (acc_grad_y2 != scalar_t(0)) atomicAdd(&grad_shared[to_idx * 2 + 1],   acc_grad_y2);
    };


    scalar_t r_output[K];
    int r_edge[K];
    scalar_t r_grad_output[K];

    const int global_offset = bi * pixels_per_batch + start_local_idx;

    if constexpr (VALID_VECTOR_LOAD) {
        const float4* out_ptr = reinterpret_cast<const float4*>(output + global_offset);
        const float4* grad_ptr = reinterpret_cast<const float4*>(grad_output + global_offset);
        const int4* edge_ptr = reinterpret_cast<const int4*>(nearest_edge_map + global_offset);
        #pragma unroll
        for (int k = 0; k < K / 4; ++k) {
            float4 o = out_ptr[k];
            float4 g = grad_ptr[k];
            int4 e = edge_ptr[k];

            r_output[k*4+0] = o.x; r_output[k*4+1] = o.y; r_output[k*4+2] = o.z; r_output[k*4+3] = o.w;
            r_grad_output[k*4+0] = g.x; r_grad_output[k*4+1] = g.y; r_grad_output[k*4+2] = g.z; r_grad_output[k*4+3] = g.w;
            r_edge[k*4+0] = e.x; r_edge[k*4+1] = e.y; r_edge[k*4+2] = e.z; r_edge[k*4+3] = e.w;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            int local_idx = start_local_idx + k;
            if (local_idx < pixels_per_batch) {
                int idx = bi * pixels_per_batch + local_idx;
                r_output[k] = output[idx];
                r_edge[k] = nearest_edge_map[idx];
                r_grad_output[k] = grad_output[idx];
            } else {
                r_edge[k] = -1; 
                r_grad_output[k] = 0;
            }
        }
    }

    #pragma unroll K
    for (int k = 0; k < K; ++k) {
        int local_idx = start_local_idx + k;

        if (local_idx < pixels_per_batch) {
            const int vn = r_edge[k];
            if (vn != prev_vn) {
                commit_gradients_to_shared(prev_vn);

                acc_grad_x1 = 0; acc_grad_y1 = 0;
                acc_grad_x2 = 0; acc_grad_y2 = 0;

    
                const int from_index = vn;
                const int to_index = (vn + 1 < nv) ? vn + 1 : 0;

                cache_from_x = vertices[vertex_base + from_index * 2];
                cache_from_y = vertices[vertex_base + from_index * 2 + 1];
                cache_to_x   = vertices[vertex_base + to_index * 2];
                cache_to_y   = vertices[vertex_base + to_index * 2 + 1];

                cache_x2_sub_x1 = cache_to_x - cache_from_x;
                cache_y2_sub_y1 = cache_to_y - cache_from_y;
                
                scalar_t sq_len = fmaf(cache_x2_sub_x1, cache_x2_sub_x1, 
                                     fmaf(cache_y2_sub_y1, cache_y2_sub_y1, scalar_t(1e-6)));
                cache_sq_len_inv = scalar_t(1.0) / sq_len;

                prev_vn = vn;
            }

            const int yp_int = local_idx / width;
            const int xp_int = local_idx % width;
            scalar_t xp = static_cast<scalar_t>(xp_int);
            scalar_t yp = static_cast<scalar_t>(yp_int);

            scalar_t x_sub_x1 = xp - cache_from_x;
            scalar_t y_sub_y1 = yp - cache_from_y;
            scalar_t x_sub_x2 = xp - cache_to_x;
            scalar_t y_sub_y2 = yp - cache_to_y;

            scalar_t dot_num = fmaf(x_sub_x1, cache_x2_sub_x1, y_sub_y1 * cache_y2_sub_y1);
            scalar_t dot = dot_num * cache_sq_len_inv;

            scalar_t r_i = r_output[k];
            scalar_t sigmoid_deriv = r_i * (scalar_t(1.0) - r_i);
            scalar_t in_out = (r_i >= scalar_t(0.5)) ? scalar_t(1.0) : scalar_t(-1.0);
            scalar_t go = r_grad_output[k];
            scalar_t base = in_out * sigmoid_deriv * inv_smoothness;

            scalar_t grad_x1 = 0, grad_y1 = 0, grad_x2 = 0, grad_y2 = 0;

            if (dot < 0) {
                 scalar_t factor = base * scalar_t(-2.0);
                 grad_x1 = factor * x_sub_x1;
                 grad_y1 = factor * y_sub_y1;
            } else if (dot > 1) {
                 scalar_t factor = base * scalar_t(-2.0);
                 grad_x2 = factor * x_sub_x2;
                 grad_y2 = factor * y_sub_y2;
            } else {
                 scalar_t x_proj = xp - fmaf(dot, cache_x2_sub_x1, cache_from_x);
                 scalar_t y_proj = yp - fmaf(dot, cache_y2_sub_y1, cache_from_y);

                 scalar_t ss_x1 = scalar_t(-2.0) * cache_x2_sub_x1;
                 scalar_t ss_x2 = scalar_t( 2.0) * cache_x2_sub_x1;
                 scalar_t ss_y1 = scalar_t(-2.0) * cache_y2_sub_y1;
                 scalar_t ss_y2 = scalar_t( 2.0) * cache_y2_sub_y1;

                 scalar_t dot_x1 = fmaf(-dot, ss_x1, -cache_x2_sub_x1 - x_sub_x1) * cache_sq_len_inv;
                 scalar_t dot_x2 = fmaf(-dot, ss_x2,  x_sub_x1) * cache_sq_len_inv;
                 scalar_t dot_y1 = fmaf(-dot, ss_y1, -cache_y2_sub_y1 - y_sub_y1) * cache_sq_len_inv;
                 scalar_t dot_y2 = fmaf(-dot, ss_y2,  y_sub_y1) * cache_sq_len_inv;

                 scalar_t x_proj_x1 = fmaf(-dot_x1, cache_x2_sub_x1, dot - scalar_t(1.0));
                 scalar_t x_proj_x2 = -fmaf(dot_x2, cache_x2_sub_x1, dot);
                 scalar_t y_proj_y1 = fmaf(-dot_y1, cache_y2_sub_y1, dot - scalar_t(1.0));
                 scalar_t y_proj_y2 = -fmaf(dot_y2, cache_y2_sub_y1, dot);

                 scalar_t y_proj_x1 = -dot_x1 * cache_y2_sub_y1;
                 scalar_t y_proj_x2 = -dot_x2 * cache_y2_sub_y1;
                 scalar_t x_proj_y1 = -dot_y1 * cache_x2_sub_x1;
                 scalar_t x_proj_y2 = -dot_y2 * cache_x2_sub_x1;

                 scalar_t factor = base * scalar_t(2.0);
                 grad_x1 = factor * fmaf(x_proj, x_proj_x1, y_proj * y_proj_x1);
                 grad_x2 = factor * fmaf(x_proj, x_proj_x2, y_proj * y_proj_x2);
                 grad_y1 = factor * fmaf(x_proj, x_proj_y1, y_proj * y_proj_y1);
                 grad_y2 = factor * fmaf(x_proj, x_proj_y2, y_proj * y_proj_y2);
            }
            
            acc_grad_x1 += grad_x1 * go;
            acc_grad_y1 += grad_y1 * go;
            acc_grad_x2 += grad_x2 * go;
            acc_grad_y2 += grad_y2 * go;
        }
    }
    
    commit_gradients_to_shared(prev_vn);
    __syncthreads();

    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        scalar_t val = grad_shared[v];
        if (val != scalar_t(0)) {
            atomicAdd(&grad_vertices[vertex_base + v], val);
        }
    }
}

/** 去掉了lds **/
template <typename scalar_t, int K, bool VALID_VECTOR_LOAD>
__global__ void backward_polyfill_large_kernel(
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ output,
    const int* __restrict__ nearest_edge_map,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_vertices,
    int number_vertices,
    int width,
    int height,
    float inv_smoothness_float 
) {
    const int nv = number_vertices;
    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    const scalar_t inv_smoothness = static_cast<scalar_t>(inv_smoothness_float);
    const int start_local_idx = (blockIdx.x * blockDim.x + threadIdx.x) * K;
    const int vertex_base = bi * nv * 2;

    int prev_vn = -1;
    scalar_t cache_from_x = 0, cache_from_y = 0, cache_to_x = 0, cache_to_y = 0;
    scalar_t cache_x2_sub_x1 = 0, cache_y2_sub_y1 = 0, cache_sq_len_inv = 0;
    scalar_t acc_grad_x1 = 0, acc_grad_y1 = 0, acc_grad_x2 = 0, acc_grad_y2 = 0;

    auto commit_gradients = [&](int vn) {
        if (vn == -1) return;
        const int from_idx = vn;
        const int to_idx = (vn + 1 < nv) ? vn + 1 : 0;
        if (acc_grad_x1 != 0) atomicAdd(&grad_vertices[vertex_base + from_idx * 2],     acc_grad_x1);
        if (acc_grad_y1 != 0) atomicAdd(&grad_vertices[vertex_base + from_idx * 2 + 1], acc_grad_y1);
        if (acc_grad_x2 != 0) atomicAdd(&grad_vertices[vertex_base + to_idx * 2],       acc_grad_x2);
        if (acc_grad_y2 != 0) atomicAdd(&grad_vertices[vertex_base + to_idx * 2 + 1],   acc_grad_y2);
    };

    scalar_t r_output[K], r_grad_output[K];
    int r_edge[K];
    const int global_offset = bi * pixels_per_batch + start_local_idx;

    // 向量化读取
    if constexpr (VALID_VECTOR_LOAD) {
        const float4* out_ptr = reinterpret_cast<const float4*>(output + global_offset);
        const float4* grad_ptr = reinterpret_cast<const float4*>(grad_output + global_offset);
        const int4* edge_ptr = reinterpret_cast<const int4*>(nearest_edge_map + global_offset);
        #pragma unroll
        for (int k = 0; k < K / 4; ++k) {
            float4 o = out_ptr[k]; float4 g = grad_ptr[k]; int4 e = edge_ptr[k];
            r_output[k*4+0] = o.x; r_output[k*4+1] = o.y; r_output[k*4+2] = o.z; r_output[k*4+3] = o.w;
            r_grad_output[k*4+0] = g.x; r_grad_output[k*4+1] = g.y; r_grad_output[k*4+2] = g.z; r_grad_output[k*4+3] = g.w;
            r_edge[k*4+0] = e.x; r_edge[k*4+1] = e.y; r_edge[k*4+2] = e.z; r_edge[k*4+3] = e.w;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            int local_idx = start_local_idx + k;
            if (local_idx < pixels_per_batch) {
                int idx = bi * pixels_per_batch + local_idx;
                r_output[k] = output[idx]; r_edge[k] = nearest_edge_map[idx]; r_grad_output[k] = grad_output[idx];
            } else { r_edge[k] = -1; r_grad_output[k] = 0; }
        }
    }

    #pragma unroll K
    for (int k = 0; k < K; ++k) {
        int local_idx = start_local_idx + k;
        if (local_idx < pixels_per_batch) {
            const int vn = r_edge[k];
            if (vn != prev_vn) {
                commit_gradients(prev_vn);
                acc_grad_x1 = 0; acc_grad_y1 = 0; acc_grad_x2 = 0; acc_grad_y2 = 0;
                cache_from_x = vertices[vertex_base + vn * 2];
                cache_from_y = vertices[vertex_base + vn * 2 + 1];
                int to_index = (vn + 1 < nv) ? vn + 1 : 0;
                cache_to_x = vertices[vertex_base + to_index * 2];
                cache_to_y = vertices[vertex_base + to_index * 2 + 1];
                cache_x2_sub_x1 = cache_to_x - cache_from_x;
                cache_y2_sub_y1 = cache_to_y - cache_from_y;
                scalar_t sq_len = fmaf(cache_x2_sub_x1, cache_x2_sub_x1, fmaf(cache_y2_sub_y1, cache_y2_sub_y1, scalar_t(1e-6)));
                cache_sq_len_inv = scalar_t(1.0) / sq_len;
                prev_vn = vn;
            }

            scalar_t xp = static_cast<scalar_t>(local_idx % width);
            scalar_t yp = static_cast<scalar_t>(local_idx / width);
            scalar_t x_sub_x1 = xp - cache_from_x, y_sub_y1 = yp - cache_from_y;
            scalar_t dot = fmaf(x_sub_x1, cache_x2_sub_x1, y_sub_y1 * cache_y2_sub_y1) * cache_sq_len_inv;
            scalar_t r_i = r_output[k], go = r_grad_output[k];
            scalar_t base = ((r_i >= 0.5f) ? 1.0f : -1.0f) * r_i * (1.0f - r_i) * inv_smoothness;
            scalar_t g_x1 = 0, g_y1 = 0, g_x2 = 0, g_y2 = 0;

            if (dot < 0) {
                g_x1 = base * -2.0f * x_sub_x1; g_y1 = base * -2.0f * y_sub_y1;
            } else if (dot > 1) {
                g_x2 = base * -2.0f * (xp - cache_to_x); g_y2 = base * -2.0f * (yp - cache_to_y);
            } else {
                scalar_t x_proj = xp - fmaf(dot, cache_x2_sub_x1, cache_from_x);
                scalar_t y_proj = yp - fmaf(dot, cache_y2_sub_y1, cache_from_y);
                scalar_t dot_x1 = fmaf(-dot, -2.0f * cache_x2_sub_x1, -cache_x2_sub_x1 - x_sub_x1) * cache_sq_len_inv;
                scalar_t dot_x2 = fmaf(-dot,  2.0f * cache_x2_sub_x1,  x_sub_x1) * cache_sq_len_inv;
                scalar_t dot_y1 = fmaf(-dot, -2.0f * cache_y2_sub_y1, -cache_y2_sub_y1 - y_sub_y1) * cache_sq_len_inv;
                scalar_t dot_y2 = fmaf(-dot,  2.0f * cache_y2_sub_y1,  y_sub_y1) * cache_sq_len_inv;
                g_x1 = base * 2.0f * (x_proj * (fmaf(-dot_x1, cache_x2_sub_x1, dot - 1.0f)) + y_proj * (-dot_x1 * cache_y2_sub_y1));
                g_x2 = base * 2.0f * (x_proj * (-fmaf(dot_x2, cache_x2_sub_x1, dot)) + y_proj * (-dot_x2 * cache_y2_sub_y1));
                g_y1 = base * 2.0f * (x_proj * (-dot_y1 * cache_x2_sub_x1) + y_proj * (fmaf(-dot_y1, cache_y2_sub_y1, dot - 1.0f)));
                g_y2 = base * 2.0f * (x_proj * (-dot_y2 * cache_x2_sub_x1) + y_proj * (-fmaf(dot_y2, cache_y2_sub_y1, dot)));
            }
            acc_grad_x1 += g_x1 * go; acc_grad_y1 += g_y1 * go; acc_grad_x2 += g_x2 * go; acc_grad_y2 += g_y2 * go;
        }
    }
    commit_gradients(prev_vn);
}

torch::Tensor backward_polyfill_lightop(
    torch::Tensor vertices,
    torch::Tensor output,
    torch::Tensor nearest_edge_map,
    torch::Tensor grad_output,
    torch::Tensor grad_vertices,
    int width,
    int height,
    float inv_smoothness,
    int mode
) {
    const auto batch_size = vertices.size(0);
    const auto number_vertices = vertices.size(1);

    if (batch_size == 0) {
        return grad_vertices;
    }

    int threads = 1024;
    const int K_ELEM = 4;
    if(width <= 256){
        threads = 64;
    }

    bool use_seme = number_vertices <= 8000 && width > 256 && height > 256;

    const int pixels_per_batch = width * height;
    
    const int items_per_block = threads * K_ELEM;
    const int blocks_per_batch = (pixels_per_batch + items_per_block - 1) / items_per_block;

    dim3 grid(blocks_per_batch, batch_size);

     AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "backward_polyfill", ([&] {
        float smoothness = 1.0 / inv_smoothness;
        bool host_valid_vector = false;
        if (std::is_same<scalar_t, float>::value && (K_ELEM % 4 == 0)) {
            bool aligned = ((uintptr_t)output.data_ptr<float>() % 16 == 0) &&
                           ((uintptr_t)grad_output.data_ptr<float>() % 16 == 0) &&
                           ((uintptr_t)nearest_edge_map.data_ptr<int>() % 16 == 0);
            if (aligned && (pixels_per_batch % K_ELEM == 0)) host_valid_vector = true;
        }

        if (!use_seme) {
            if(host_valid_vector) {
                backward_polyfill_large_kernel<scalar_t, K_ELEM, true> <<<grid, threads>>>(
                    vertices.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), nearest_edge_map.data_ptr<int>(), 
                    grad_output.data_ptr<scalar_t>(), grad_vertices.data_ptr<scalar_t>(), (int)number_vertices, 
                    width, height, smoothness
                );
            } else {
                backward_polyfill_large_kernel<scalar_t, K_ELEM, false> <<<grid, threads>>>(
                    vertices.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), nearest_edge_map.data_ptr<int>(), 
                    grad_output.data_ptr<scalar_t>(), grad_vertices.data_ptr<scalar_t>(), (int)number_vertices, 
                    width, height, smoothness
                );
            }
        } else {
            size_t shared_size = number_vertices * 2 * sizeof(scalar_t);
            if(host_valid_vector){
                backward_polyfill_optimized_kernel<scalar_t, K_ELEM, true> <<<grid, threads, shared_size>>>(
                    vertices.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), nearest_edge_map.data_ptr<int>(), 
                    grad_output.data_ptr<scalar_t>(), grad_vertices.data_ptr<scalar_t>(), (int)number_vertices, 
                    width, height, smoothness
                );
            } else {
                backward_polyfill_optimized_kernel<scalar_t, K_ELEM, false> <<<grid, threads, shared_size>>>(
                    vertices.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), nearest_edge_map.data_ptr<int>(), grad_output.data_ptr<scalar_t>(), 
                    grad_vertices.data_ptr<scalar_t>(), (int)number_vertices, width, height, 
                    smoothness
                );
            }
        }
    }));
    return grad_vertices;
}

}
}
