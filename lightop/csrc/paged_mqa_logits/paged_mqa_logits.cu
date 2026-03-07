#include "paged_mqa_logits.h"
#include "paged_mqa_logits_impl.hpp"

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "ATen/Dispatch.h"

#include "device_runtime.h"
#include "exception.h"
#include "utils.h"

namespace lightop {
namespace deep_gemm {

#define USE_CLEAN_LOGITS 0
#define DISPATCH_ALIGNED_BATCH(ALIGNED_BATCH, SMS) \
    case ALIGNED_BATCH: \
        paged_mqa_logits_metadata<ALIGNED_BATCH, split_kv, SMS><<<grid, block, smem_size, 0>>>( \
            batch_size, context_lens.data_ptr<int>(), schedule_metadata.data_ptr<int>()); \
        break;

#define DISPATCH_SMS(SMS_VAL) \
    switch (aligned_batch_size) { \
        DISPATCH_ALIGNED_BATCH(64, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(128, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(192, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(256, SMS_VAL) \
        default: LIGHTOP_HOST_UNREACHABLE("unsupported aligned batch size\n"); \
    }

#define DISPATCH_NEXT_N(next_n, NEXT_N, ...)                        \
    [&]() {                                                         \
        if (next_n == 1) {                                          \
            constexpr int NEXT_N = 1;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 2) {                                   \
            constexpr int NEXT_N = 2;                               \
            grid.x *= next_n;                                       \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        LIGHTOP_HOST_UNREACHABLE("unsupported next_n size\n");      \
    }()

#define DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, ...)               \
    [&]() {                                                         \
        if (num_heads == 32) {                                      \
            constexpr int NUM_HEADS = 32;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 64) {                               \
            constexpr int NUM_HEADS = 64;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        LIGHTOP_HOST_UNREACHABLE("unsupported head size\n");        \
    }()

#define DISPATCH_FP8_TYPES(scalar_type, FP8_TYPE, ...)              \
    [&]() {                                                         \
        if (scalar_type == at::ScalarType::Float8_e4m3fn) {         \
            using FP8_TYPE = at::Float8_e4m3fn;                     \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (scalar_type == at::ScalarType::Float8_e5m2) {    \
            using FP8_TYPE = at::Float8_e5m2;                       \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        LIGHTOP_HOST_UNREACHABLE("unsupported fp8 type\n");         \
    }()

int get_sms_info() {
    int device_id = 0;
    if (hipGetDevice(&device_id) != hipSuccess) {
        throw std::runtime_error("Failed to get current HIP device");
    }
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_id) != hipSuccess) {
        throw std::runtime_error("Failed to get HIP device properties");
    }

    int num_cus = device_runtime->get_num_sms();
    int sms = 0;

    std::string arch(prop.gcnArchName);  

    if (arch.rfind("gfx936", 0) == 0 && num_cus == 80) {
        sms = 320;
    } 
    else if (arch.rfind("gfx938", 0) == 0 && num_cus == 64) {
        sms = 128;
    } 
    else if (arch.rfind("gfx938", 0) == 0 && num_cus == 72) {
        sms = 144;
    } 
    else {
        std::cerr << "Unsupported device config: " << arch 
                  << ", CU=" << num_cus << std::endl;
        throw std::runtime_error("Unsupported device configuration");
    }
    return sms;
}

torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    const auto& [batch_size] = get_shape<1>(context_lens);
    LIGHTOP_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    LIGHTOP_HOST_ASSERT(context_lens.is_contiguous());
    LIGHTOP_HOST_ASSERT(block_kv == 64);
    LIGHTOP_HOST_ASSERT(device_runtime->get_num_sms() == num_sms);

    constexpr int num_math_warpgroups = 4;
    const int aligned_batch_size = align(batch_size, 64);
    constexpr int split_kv = 64 * num_math_warpgroups;

    int sms = get_sms_info();
    auto schedule_metadata = torch::empty({sms + 1, 2}, context_lens.options());

    // Calculate shared memory size
    const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));

    // Dispatch implementation
    dim3 grid = {1, 1, 1};
    dim3 block = {WARP_SIZE_GPU, 1, 1};

    switch (sms) {
        case 128:  DISPATCH_SMS(128)  break;
        case 144:  DISPATCH_SMS(144)  break;
        case 320: DISPATCH_SMS(320) break;
        default: LIGHTOP_HOST_UNREACHABLE("unsupported SMS value\n");
    }

    return schedule_metadata;
}

torch::Tensor paged_mqa_logits(const torch::Tensor& q,
                               const torch::Tensor& fused_kv_cache,
                               const torch::Tensor& weights,
                               const torch::Tensor& context_lens,
                               const torch::Tensor& block_table,
                               const torch::Tensor& schedule_meta,
                               const int& max_context_len,
                               const bool& clean_logits) {
    // Declare these variables explicitly to avoid mis-binding in kernel launch lambdas
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf;
    std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q);
    std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_] = get_shape<1>(context_lens);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size__, max_block_len] = get_shape<2>(block_table);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const auto& num_sms = device_runtime->get_num_sms();
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);

    unsigned int BW_SMS = get_sms_info();

    LIGHTOP_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    LIGHTOP_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    LIGHTOP_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    LIGHTOP_HOST_ASSERT(schedule_meta_size == BW_SMS + 1 and meta_info_size == 2);

    LIGHTOP_HOST_ASSERT(next_n == 1 or next_n == 2);
    LIGHTOP_HOST_ASSERT(block_kv == 64);

    LIGHTOP_HOST_ASSERT(q.is_contiguous());
    LIGHTOP_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    LIGHTOP_HOST_ASSERT(weights.is_contiguous());
    LIGHTOP_HOST_ASSERT(context_lens.is_contiguous());
    LIGHTOP_HOST_ASSERT(block_table.stride(1) == 1);
    LIGHTOP_HOST_ASSERT(schedule_meta.is_contiguous());

    LIGHTOP_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    LIGHTOP_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    LIGHTOP_HOST_ASSERT(block_table.scalar_type() == torch::kInt32);
    LIGHTOP_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt32);

    torch::Tensor kv_cache_scales;
    torch::Tensor kv_cache;
    int smem_kv_size_per_stage = 0;
    bool is_fp8 = (q.scalar_type() == torch::kFloat8_e4m3fn || q.scalar_type() == torch::kFloat8_e5m2);
    bool is_fp16 = (q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16);

    if (is_fp8) {
        LIGHTOP_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
        LIGHTOP_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);
        // Derive FP8 values and SF tensor from KV cache
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
        );

        kv_cache_scales = torch::from_blob(
            fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size()) + block_kv * sizeof(float);
    } else {
        LIGHTOP_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kFloat16 || fused_kv_cache.scalar_type() == torch::kBFloat16);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat16)
        );
        smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv_cache.element_size());
    }

    // Allocate output
    constexpr int num_math_warp_groups = 4;
    const auto& aligned_max_context_len = align(max_context_len, block_kv);

    torch::Tensor logits;
    if (clean_logits) {
        #if USE_CLEAN_LOGITS
        logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
        #else
        constexpr float neg_inf = -std::numeric_limits<float>::infinity();
        logits = torch::full({batch_size * next_n, aligned_max_context_len}, neg_inf, q.options().dtype(torch::kFloat));
        #endif
    } else {
        logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
    }
    logits = logits.slice(-1, 0, max_context_len);

    // constexpr int split_kv = num_math_warp_groups * block_kv;
    constexpr int split_kv = 4 * 64;

    // Calculate shared memory size
    constexpr int num_q_stages = 1, num_kv_stages = 1;

    constexpr int smem_q_size_per_stage = 0;
    constexpr int aligned_smem_weight_size_per_stage = 0;
    constexpr int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage);

    const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage);

    const int smem_size = smem_q_pipe_size + smem_kv_pipe_size;

    // Dispatch implementation
    dim3 grid = {BW_SMS, 1, 1};
    dim3 block = {num_math_warp_groups * WARP_SIZE_GPU, 1, 1};

    const bool use_fp8_kv = (kv_cache_scales.defined() && kv_cache_scales.numel() > 0);

    if (!use_fp8_kv) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                q.scalar_type(),
                "paged_mqa_float16",
                ([&] {
                    DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                        DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                            paged_mqa_logits<scalar_t, NEXT_N, NUM_HEADS, 128, 64, split_kv, 4><<<grid, block, smem_size
                            , 0>>>(
                                reinterpret_cast<scalar_t*>(q.data_ptr()),
                                reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                weights.data_ptr<float>(), batch_size,
                                num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>(),
                                schedule_meta.data_ptr<int32_t>());
                        });
                    });
                })
            );
    } else {
        LIGHTOP_HOST_ASSERT(num_heads == 64);
        DISPATCH_FP8_TYPES(q.scalar_type(), FP8_TYPE, [&] {
            DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    paged_mqa_logits_fp8<FP8_TYPE, NEXT_N, NUM_HEADS, 128, 64, split_kv, 4><<<grid, block,
                    smem_size, 0>>>(
                        reinterpret_cast<FP8_TYPE*>(q.data_ptr()),
                        reinterpret_cast<FP8_TYPE*>(kv_cache.data_ptr()),
                        kv_cache_scales.data_ptr<float>(), weights.data_ptr<float>(),
                        batch_size, num_kv_blocks, kv_cache_stride_bytes,
                        aligned_max_context_len, block_table_stride,
                        context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                        block_table.data_ptr<int32_t>(),
                        schedule_meta.data_ptr<int32_t>());
                });
            });
        });
      }

    hipError_t result = hipGetLastError();
    if (result != hipSuccess) {
        printf("launch compute logits kernel error: %d\n", (int)result);
    }

    if (clean_logits) {
        #if USE_CLEAN_LOGITS
        if (next_n == 1) {
            internal_clean_logits<1, 4><<<ceil_div(batch_size * next_n, 4), 256>>>(
            batch_size * next_n, max_context_len, aligned_max_context_len, nullptr, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>());
        } else if (next_n == 2) {
            internal_clean_logits<2, 4><<<ceil_div(batch_size * next_n, 4), 256>>>(
            batch_size * next_n, max_context_len, aligned_max_context_len, nullptr, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>());
        } else {
            LIGHTOP_HOST_UNREACHABLE("unsupported next n\n");
        }
        #endif
    }

    return logits;
}

}
}