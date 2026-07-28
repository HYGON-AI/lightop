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

namespace at {
namespace native {
// ============================================================
// Constants and Macros
// ============================================================

const int MODE_BOUNDARY  = 0;
const int MODE_MASK      = 1;
const int MODE_HARD_MASK = 2;



// ============================================================
// Fused Forward Kernel (inside_outside + distance + output 融合)
// ============================================================

template <typename scalar_t, int MODE>
__global__ void forward_polyfill_fused_kernel(
    const scalar_t* __restrict__ vertices,
    int number_vertices,
    scalar_t* __restrict__ output,
    int* __restrict__ nearest_edge_map,
    int height,
    int width,
    float inv_smoothness
) {
    // 使用共享内存存放顶点
    extern __shared__ char shared_mem[];
    scalar_t* verts_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;

    // 2D Grid: blockIdx.y = batch, blockIdx.x = block within batch
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vertex_base = bi * nv * 2;

    // 协作加载顶点到共享内存
    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        verts_shared[v] = vertices[vertex_base + v];
    }
    __syncthreads();

    if (local_idx >= pixels_per_batch) return;

    const int idx = bi * pixels_per_batch + local_idx;
    const int yp = local_idx / width;
    const int xp = local_idx % width;

    // ================= Step 1: Inside/Outside 判断（仅 MASK 和 HARD_MASK 需要） =================
    scalar_t in_out_sign = scalar_t(1.0);  // 默认外部
    if constexpr (MODE == MODE_MASK || MODE == MODE_HARD_MASK) {
        // Ray casting algorithm (William Randolph Franklin)
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
        in_out_sign = (c == 0) ? scalar_t(-1.0) : scalar_t(1.0);
    }

    // HARD_MASK 模式：只用 inside_outside, 输出 -1.0 或 1.0
    if constexpr (MODE == MODE_HARD_MASK) {
        output[idx] = in_out_sign;  // -1.0 (outside) 或 1.0 (inside)
        return;
    }

    // ================= Step 2: 计算到每条边的最短距离 =================
    scalar_t min_dist = scalar_t(1e10);
    int closest_vertex = 0;

#pragma unroll 4
    for (int j = 0; j < nv; ++j) {
        int next_j = (j + 1 < nv) ? j + 1 : 0;

        scalar_t p1_x = verts_shared[j * 2];
        scalar_t p1_y = verts_shared[j * 2 + 1];
        scalar_t p2_x = verts_shared[next_j * 2];
        scalar_t p2_y = verts_shared[next_j * 2 + 1];

        scalar_t dx = p2_x - p1_x;
        scalar_t dy = p2_y - p1_y;
        scalar_t len_sq = fmaf(dx, dx, fmaf(dy, dy, scalar_t(1e-6)));

        scalar_t t =
            fmaf(scalar_t(xp) - p1_x, dx, (scalar_t(yp) - p1_y) * dy) / len_sq;
        t = fmaxf(scalar_t(0), fminf(scalar_t(1), t));

        scalar_t proj_x = fmaf(t, dx, p1_x);
        scalar_t proj_y = fmaf(t, dy, p1_y);

        scalar_t diff_x = scalar_t(xp) - proj_x;
        scalar_t diff_y = scalar_t(yp) - proj_y;
        scalar_t dist = fmaf(diff_x, diff_x, diff_y * diff_y);

        if (dist < min_dist) {
            min_dist = dist;
            closest_vertex = j;
        }
    }

    // ================= Step 3: 根据模式计算输出 =================
    scalar_t contribution;
    if constexpr (MODE == MODE_MASK) {
        // 使用 __expf 进行快速 sigmoid
        contribution =
            scalar_t(1.0) /
            (scalar_t(1.0) + __expf(-in_out_sign * min_dist / inv_smoothness));
    } else {  // MODE_BOUNDARY
        contribution = __expf(-min_dist / inv_smoothness);
    }

    output[idx] = contribution;
    nearest_edge_map[idx] = closest_vertex;
}

// ============================================================
// Optimized Backward Kernel
// ============================================================

template <typename scalar_t>
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
    // 共享内存用于块内梯度归约
    extern __shared__ char shared_mem[];
    scalar_t* grad_shared = reinterpret_cast<scalar_t*>(shared_mem);

    const int nv = number_vertices;
    const int pixels_per_batch = width * height;

    // 2D Grid: blockIdx.y = batch, blockIdx.x = block within batch
    const int bi = blockIdx.y;
    const int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vertex_base = bi * nv * 2;

    // 初始化共享内存
    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        grad_shared[v] = scalar_t(0);
    }
    __syncthreads();

    if (local_idx < pixels_per_batch) {
        const int i  = bi * pixels_per_batch + local_idx;
        const int yp = local_idx / width;
        const int xp = local_idx % width;

        // 获取最近边的索引
        const int vn = nearest_edge_map[i];
        const int from_index = vn;
        const int to_index = (vn + 1 < nv) ? vn + 1 : 0;

        // 读取顶点坐标
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

        // 提取公共因子
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

        // 乘以上游梯度
        grad_x1 *= go;
        grad_y1 *= go;
        grad_x2 *= go;
        grad_y2 *= go;

        // 累加到共享内存（块内冲突比全局少很多）
        atomicAdd(&grad_shared[from_index * 2],     grad_x1);
        atomicAdd(&grad_shared[from_index * 2 + 1], grad_y1);
        atomicAdd(&grad_shared[to_index * 2],       grad_x2);
        atomicAdd(&grad_shared[to_index * 2 + 1],   grad_y2);
    }

    __syncthreads();

    // 块内归约完成，写回全局内存
    for (int v = threadIdx.x; v < nv * 2; v += blockDim.x) {
        scalar_t val = grad_shared[v];
        if (val != scalar_t(0)) {
            atomicAdd(&grad_vertices[vertex_base + v], val);
        }
    }
}

// ============================================================
// Launch Functions
// ============================================================

std::vector<torch::Tensor> forward_polyfill_cuda(
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

    const int threads = 256;
    const int pixels_per_batch = width * height;
    const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;

    // 2D grid: (blocks_per_batch, batch_size)
    dim3 grid(blocks_per_batch, batch_size);

    // 使用融合 kernel，一次完成 inside_outside + distance + output 计算
    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "forward_polyfill_fused", ([&] {
        size_t shared_size = number_vertices * 2 * sizeof(scalar_t);

        switch (mode) {
            case MODE_MASK:
                forward_polyfill_fused_kernel<scalar_t, MODE_MASK>
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

            case MODE_HARD_MASK:
                forward_polyfill_fused_kernel<scalar_t, MODE_HARD_MASK>
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
                forward_polyfill_fused_kernel<scalar_t, MODE_BOUNDARY>
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
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in forward_polyfill_fused: %s\n", cudaGetErrorString(err));
    }

    return { output, nearest_edge_map };
}

torch::Tensor backward_polyfill_cuda(
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

    const int threads = 256;
    const int pixels_per_batch = width * height;
    const int blocks_per_batch = (pixels_per_batch + threads - 1) / threads;

    // 2D grid: (blocks_per_batch, batch_size)
    dim3 grid(blocks_per_batch, batch_size);

    AT_DISPATCH_FLOATING_TYPES(vertices.scalar_type(), "backward_polyfill_optimized", ([&] {
        size_t shared_size = number_vertices * 2 * sizeof(scalar_t);
        backward_polyfill_optimized_kernel<scalar_t>
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
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in backward_polyfill_optimized: %s\n", cudaGetErrorString(err));
    }

    return grad_vertices;
}
}
}