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
#define EXP_FUNC(x) __builtin_amdgcn_exp2f((x) * LOG2_E)


namespace at {
namespace native {



const int MODE_BOUNDARY  = 0;
const int MODE_MASK      = 1;
const int MODE_HARD_MASK = 2;
const int TILE_SIZE = 512; 


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

    const int idx = bi * pixels_per_batch + local_idx;
    const int yp = local_idx / width;
    const int xp = local_idx % width;

    scalar_t to_x = vertices[vertex_base + (number_vertices - 1) * 2];
    scalar_t to_y = vertices[vertex_base + (number_vertices - 1) * 2 + 1];

    int c = 0;


    for (int chunk_base = 0; chunk_base < number_vertices; chunk_base += TILE_SIZE) {
        int limit = std::min(TILE_SIZE, number_vertices - chunk_base);
        
        for (int t = threadIdx.x; t < limit * 2; t += blockDim.x) {
            verts_shared[t] = vertices[vertex_base + (chunk_base * 2) + t];
        }
        __syncthreads();

        for (int t = 0; t < limit; ++t) {
            scalar_t from_x = verts_shared[t * 2];
            scalar_t from_y = verts_shared[t * 2 + 1];
            if (((from_y > yp) != (to_y > yp)) &&
                (xp < (to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x)) {
                c = !c;
            }
            to_x = from_x;
            to_y = from_y;
        }
        __syncthreads();
    }

    output[idx] = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
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

    int threads = 256;
    if(number_vertices < 1000){
        threads = 256;
    }else if (number_vertices < 3000){
        threads = 512;
    }else{
        threads = 1024;
    }
    if(width == 256) threads = 256;
    const int pixels_per_batch = width * height;
    const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;

    dim3 grid(blocks_per_batch, batch_size);

    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "forward_polyfill_fused", ([&] {
        
        if (mode == MODE_HARD_MASK) {
         size_t shared_size = TILE_SIZE * 2 * sizeof(scalar_t);
        forward_polyfill_hard_mask_kernel_large<scalar_t>
                    <<<grid, threads, shared_size>>>(
                        vertices.data_ptr<scalar_t>(),
                        (int)number_vertices,
                        output.data_ptr<scalar_t>(),
                        height,
                        width
                    );
        } else {
            if (number_vertices > 5000) {
                size_t shared_size = ((TILE_SIZE + 1) * 2 + TILE_SIZE) * sizeof(scalar_t);

                switch (mode) {
                    case MODE_MASK:
                        forward_polyfill_general_kernel_large<scalar_t, MODE_MASK>
                            <<<grid, threads, shared_size>>>(
                                vertices.data_ptr<scalar_t>(),
                                (int)number_vertices,
                                output.data_ptr<scalar_t>(),
                                nearest_edge_map.data_ptr<int>(),
                                height,
                                width,
                                inv_smoothness
                            );
                        break;
                    case MODE_BOUNDARY:
                    default:
                        forward_polyfill_general_kernel_large<scalar_t, MODE_BOUNDARY>
                            <<<grid, threads, shared_size>>>(
                                vertices.data_ptr<scalar_t>(),
                                (int)number_vertices,
                                output.data_ptr<scalar_t>(),
                                nearest_edge_map.data_ptr<int>(),
                                height,
                                width,
                                inv_smoothness
                            );
                        break;
                }
            } else {
                size_t shared_size = number_vertices * 3 * sizeof(scalar_t);

                switch (mode) {
                    case MODE_MASK:
                        forward_polyfill_general_kernel_opt<scalar_t, MODE_MASK>
                            <<<grid, threads, shared_size>>>(
                                vertices.data_ptr<scalar_t>(),
                                (int)number_vertices,
                                output.data_ptr<scalar_t>(),
                                nearest_edge_map.data_ptr<int>(),
                                height,
                                width,
                                inv_smoothness
                            );
                        break;
                    case MODE_BOUNDARY:
                    default:
                        forward_polyfill_general_kernel_opt<scalar_t, MODE_BOUNDARY>
                            <<<grid, threads, shared_size>>>(
                                vertices.data_ptr<scalar_t>(),
                                (int)number_vertices,
                                output.data_ptr<scalar_t>(),
                                nearest_edge_map.data_ptr<int>(),
                                height,
                                width,
                                inv_smoothness
                            );
                        break;
                }
            }
        }
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in forward_polyfill: %s\n", cudaGetErrorString(err));
    }

    return { output, nearest_edge_map };
}

template <typename scalar_t, int K>
__global__ void backward_polyfill_optimized_kernel(
    const scalar_t* __restrict__ vertices,
    const scalar_t* __restrict__ output,
    const int* __restrict__ nearest_edge_map,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_vertices,
    int number_vertices,
    int width,
    int height,
    float inv_smoothness
) {
   
    extern __shared__ char shared_mem[];
    scalar_t* grad_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;
    const int bi = blockIdx.y;
    

    const int start_local_idx = (blockIdx.x * blockDim.x + threadIdx.x) * K;
    const int vertex_base = bi * nv * 2;


    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        grad_shared[v] = scalar_t(0);
    }
    __syncthreads();

    #pragma unroll K
    for (int k = 0; k < K; ++k) {
        int local_idx = start_local_idx + k;

        if (local_idx < pixels_per_batch) {
            const int i  = bi * pixels_per_batch + local_idx;
            const int yp = local_idx / width;
            const int xp = local_idx % width;

    
            const int vn = nearest_edge_map[i];
            const int from_index = vn;
            const int to_index = (vn + 1 < nv) ? vn + 1 : 0;

          
            scalar_t from_x = vertices[vertex_base + from_index * 2];
            scalar_t from_y = vertices[vertex_base + from_index * 2 + 1];
            scalar_t to_x   = vertices[vertex_base + to_index * 2];
            scalar_t to_y   = vertices[vertex_base + to_index * 2 + 1];

            scalar_t x2_sub_x1 = to_x - from_x;
            scalar_t y2_sub_y1 = to_y - from_y;
            scalar_t sq_len =
                fmaf(x2_sub_x1, x2_sub_x1, fmaf(y2_sub_y1, y2_sub_y1, scalar_t(1e-6)));
            scalar_t sq_len_inv = scalar_t(1.0) / sq_len;

            scalar_t x_sub_x1 = scalar_t(xp) - from_x;
            scalar_t y_sub_y1 = scalar_t(yp) - from_y;
            scalar_t x_sub_x2 = scalar_t(xp) - to_x;
            scalar_t y_sub_y2 = scalar_t(yp) - to_y;

            scalar_t dot_num = fmaf(x_sub_x1, x2_sub_x1, y_sub_y1 * y2_sub_y1);
            scalar_t dot = dot_num * sq_len_inv;

            scalar_t r_i = output[i];
            scalar_t sigmoid_deriv = r_i * (scalar_t(1.0) - r_i);
            scalar_t in_out = (r_i >= scalar_t(0.5)) ? scalar_t(1.0) : scalar_t(-1.0);
            scalar_t go = grad_output[i];
            scalar_t base = in_out * sigmoid_deriv / inv_smoothness;

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
                scalar_t x_proj = scalar_t(xp) - fmaf(dot, x2_sub_x1, from_x);
                scalar_t y_proj = scalar_t(yp) - fmaf(dot, y2_sub_y1, from_y);

                scalar_t ss_x1 = scalar_t(-2.0) * x2_sub_x1;
                scalar_t ss_x2 = scalar_t( 2.0) * x2_sub_x1;
                scalar_t ss_y1 = scalar_t(-2.0) * y2_sub_y1;
                scalar_t ss_y2 = scalar_t( 2.0) * y2_sub_y1;

                scalar_t dot_x1 =
                    fmaf(-dot, ss_x1, -x2_sub_x1 - x_sub_x1) * sq_len_inv;
                scalar_t dot_x2 =
                    fmaf(-dot, ss_x2,  x_sub_x1) * sq_len_inv;
                scalar_t dot_y1 =
                    fmaf(-dot, ss_y1, -y2_sub_y1 - y_sub_y1) * sq_len_inv;
                scalar_t dot_y2 =
                    fmaf(-dot, ss_y2,  y_sub_y1) * sq_len_inv;

                scalar_t x_proj_x1 =
                    fmaf(-dot_x1, x2_sub_x1, dot - scalar_t(1.0));
                scalar_t x_proj_x2 =
                    -fmaf(dot_x2, x2_sub_x1, dot);
                scalar_t y_proj_y1 =
                    fmaf(-dot_y1, y2_sub_y1, dot - scalar_t(1.0));
                scalar_t y_proj_y2 =
                    -fmaf(dot_y2, y2_sub_y1, dot);

                scalar_t y_proj_x1 = -dot_x1 * y2_sub_y1;
                scalar_t y_proj_x2 = -dot_x2 * y2_sub_y1;
                scalar_t x_proj_y1 = -dot_y1 * x2_sub_x1;
                scalar_t x_proj_y2 = -dot_y2 * x2_sub_x1;

                scalar_t factor = base * scalar_t(2.0);
                grad_x1 = factor * fmaf(x_proj, x_proj_x1, y_proj * y_proj_x1);
                grad_x2 = factor * fmaf(x_proj, x_proj_x2, y_proj * y_proj_x2);
                grad_y1 = factor * fmaf(x_proj, x_proj_y1, y_proj * y_proj_y1);
                grad_y2 = factor * fmaf(x_proj, x_proj_y2, y_proj * y_proj_y2);
            }
            grad_x1 *= go;
            grad_y1 *= go;
            grad_x2 *= go;
            grad_y2 *= go;
            
            atomicAdd(&grad_shared[from_index * 2],     grad_x1);
            atomicAdd(&grad_shared[from_index * 2 + 1], grad_y1);
            atomicAdd(&grad_shared[to_index * 2],       grad_x2);
            atomicAdd(&grad_shared[to_index * 2 + 1],   grad_y2);
        }
    }

    __syncthreads();

    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        scalar_t val = grad_shared[v];
        if (val != scalar_t(0)) {
            atomicAdd(&grad_vertices[vertex_base + v], val);
        }
    }
}
/** 去掉了lds **/
template <typename scalar_t, int K>
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

    scalar_t cache_from_x = 0, cache_from_y = 0;
    scalar_t cache_to_x = 0,   cache_to_y = 0;
    scalar_t cache_x2_sub_x1 = 0, cache_y2_sub_y1 = 0;
    scalar_t cache_sq_len_inv = 0;

    scalar_t acc_grad_x1 = 0;
    scalar_t acc_grad_y1 = 0;
    scalar_t acc_grad_x2 = 0;
    scalar_t acc_grad_y2 = 0;


    auto commit_gradients = [&](int vn) {
        if (vn == -1) return;
        const int from_idx = vn;
        const int to_idx = (vn + 1 < nv) ? vn + 1 : 0;
        
        if (acc_grad_x1 != 0) atomicAdd(&grad_vertices[vertex_base + from_idx * 2],     acc_grad_x1);
        if (acc_grad_y1 != 0) atomicAdd(&grad_vertices[vertex_base + from_idx * 2 + 1], acc_grad_y1);
        if (acc_grad_x2 != 0) atomicAdd(&grad_vertices[vertex_base + to_idx * 2],       acc_grad_x2);
        if (acc_grad_y2 != 0) atomicAdd(&grad_vertices[vertex_base + to_idx * 2 + 1],   acc_grad_y2);
    };

    #pragma unroll K
    for (int k = 0; k < K; ++k) {
        int local_idx = start_local_idx + k;

        if (local_idx < pixels_per_batch) {
            const int i  = bi * pixels_per_batch + local_idx;
         
            scalar_t go = grad_output[i];
            
            const int vn = nearest_edge_map[i];
            
            if (vn != prev_vn) {
                //边变了,把上一条的数据加载进去
                commit_gradients(prev_vn);
                
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

 
            const int yp = local_idx / width;
            const int xp = local_idx % width;
            
            scalar_t x_sub_x1 = scalar_t(xp) - cache_from_x;
            scalar_t y_sub_y1 = scalar_t(yp) - cache_from_y;
            scalar_t x_sub_x2 = scalar_t(xp) - cache_to_x;
            scalar_t y_sub_y2 = scalar_t(yp) - cache_to_y;

            scalar_t dot_num = fmaf(x_sub_x1, cache_x2_sub_x1, y_sub_y1 * cache_y2_sub_y1);
            scalar_t dot = dot_num * cache_sq_len_inv;

            scalar_t r_i = output[i];
            scalar_t sigmoid_deriv = r_i * (scalar_t(1.0) - r_i);
            scalar_t in_out = (r_i >= scalar_t(0.5)) ? scalar_t(1.0) : scalar_t(-1.0);
            scalar_t base = in_out * sigmoid_deriv / inv_smoothness;

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
                scalar_t x_proj = scalar_t(xp) - fmaf(dot, cache_x2_sub_x1, cache_from_x);
                scalar_t y_proj = scalar_t(yp) - fmaf(dot, cache_y2_sub_y1, cache_from_y);

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
        threads = 256;
    }else if(width <= 512){
        threads = 512;
    }else{
        threads = 1024;
    }
 
    const int pixels_per_batch = width * height;
    
    const int items_per_block = threads * K_ELEM;
    const int blocks_per_batch = (pixels_per_batch + items_per_block - 1) / items_per_block;

    dim3 grid(blocks_per_batch, batch_size);

    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "backward_polyfill_optimized", ([&] {
        if (number_vertices > 5000) {
            backward_polyfill_large_kernel<scalar_t, K_ELEM>
                <<<grid, threads, 0>>>(
                    vertices.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    nearest_edge_map.data_ptr<int>(),
                    grad_output.data_ptr<scalar_t>(),
                    grad_vertices.data_ptr<scalar_t>(),
                    (int)number_vertices,
                    width,
                    height,
                    inv_smoothness
                );
        } else {
            size_t shared_size = number_vertices * 2 * sizeof(scalar_t);
            backward_polyfill_optimized_kernel<scalar_t, K_ELEM>
                <<<grid, threads, shared_size>>>(
                    vertices.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    nearest_edge_map.data_ptr<int>(),
                    grad_output.data_ptr<scalar_t>(),
                    grad_vertices.data_ptr<scalar_t>(),
                    (int)number_vertices,
                    width,
                    height,
                    inv_smoothness
                );
        }
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in backward_polyfill_optimized: %s\n", cudaGetErrorString(err));
    }

    return grad_vertices;
}

}
}