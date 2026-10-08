#include "paged_mqa_logits.h"
#include "paged_mqa_logits_prefill.h"
#include "paged_mqa_logits_impl.hpp"
#include "paged_mqa_logits_fp8_impl.hpp"
#include "paged_mqa_logits_int8_impl.hpp"

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "ATen/Dispatch.h"

#include "device_runtime.h"
#include "exception.h"
#include "utils.h"

#include <mutex>
#include <unordered_map>

namespace lightop {
namespace deep_gemm {

#define USE_CLEAN_LOGITS 1

#define DISPATCH_ALIGNED_BATCH(ALIGNED_BATCH, SMS) \
    case ALIGNED_BATCH: \
        paged_mqa_logits_metadata<ALIGNED_BATCH, split_kv, SMS><<<grid, block, smem_size, stream>>>( \
            batch_size, context_lens.data_ptr<int>(), schedule_metadata.data_ptr<int>()); \
        break;

#define DISPATCH_SMS(SMS_VAL) \
    switch (aligned_batch_size) { \
        DISPATCH_ALIGNED_BATCH(64, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(128, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(192, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(256, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(320, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(384, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(448, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(512, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(576, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(640, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(704, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(768, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(832, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(896, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(960, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(1024, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(2048, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(4096, SMS_VAL) \
        DISPATCH_ALIGNED_BATCH(8192, SMS_VAL) \
        default: LIGHTOP_HOST_UNREACHABLE("unsupported aligned batch size\n"); \
    }

#define DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, ...)                  \
    [&]() {                                                         \
        if (next_n == 1) {                                          \
            constexpr int NEXT_N = 1;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 2) {                                   \
            constexpr int NEXT_N = 2;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 3) {                                   \
            constexpr int NEXT_N = 3;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 4) {                                   \
            constexpr int NEXT_N = 4;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 5) {                                   \
            constexpr int NEXT_N = 5;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        LIGHTOP_HOST_UNREACHABLE("unsupported next_n size\n");      \
    }()

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
        if (num_heads == 8) {                                       \
            constexpr int NUM_HEADS = 8;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 16) {                               \
            constexpr int NUM_HEADS = 16;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 32) {                               \
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

#define DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, ...)                  \
    [&]() {                                                         \
        if (block_kv == 4) {                                        \
            constexpr int BLOCK_KV = 4;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 8) {                                 \
            constexpr int BLOCK_KV = 8;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 16) {                                \
            constexpr int BLOCK_KV = 16;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 32) {                                \
            constexpr int BLOCK_KV = 32;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 64) {                                \
            constexpr int BLOCK_KV = 64;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        LIGHTOP_HOST_UNREACHABLE("unsupported block_kv size\n");    \
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

    // The scheduler calls this for every metadata plan.  Device properties are
    // immutable for the lifetime of a worker, so do the HIP property query once
    // per device instead of paying it on every C1/C2/publish invocation.
    static std::mutex cache_mutex;
    static std::unordered_map<int, int> sms_cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto cached = sms_cache.find(device_id);
        if (cached != sms_cache.end()) {
            return cached->second;
        }
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
    else if (arch.rfind("gfx928", 0) == 0 && num_cus == 120) {
        sms = 240;
    } else {
        std::cerr << "Unsupported device config: " << arch
                  << ", CU=" << num_cus << std::endl;
        throw std::runtime_error("Unsupported device configuration");
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        // Another worker thread may have filled the entry while this thread
        // queried the properties; both values are deterministic.
        sms_cache.emplace(device_id, sms);
    }
    return sms;
}

torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    const auto& [batch_size] = get_shape<1>(context_lens);
    LIGHTOP_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    LIGHTOP_HOST_ASSERT(context_lens.is_contiguous());
    LIGHTOP_HOST_ASSERT(device_runtime->get_num_sms() == num_sms);

    constexpr int num_math_warpgroups = 4;
    const int aligned_batch_size = align(batch_size, 64);

    int sms = get_sms_info();
    auto schedule_metadata = torch::empty({sms + 1, 2}, context_lens.options());

    // Calculate shared memory size
    const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));

    // Dispatch implementation
    dim3 grid = {1, 1, 1};
    dim3 block = {WARP_SIZE_GPU, 1, 1};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
        constexpr int split_kv = BLOCK_KV * num_math_warpgroups;
        switch (sms) {
            case 128:  DISPATCH_SMS(128)  break;
            case 144:  DISPATCH_SMS(144)  break;
            case 320: DISPATCH_SMS(320) break;
            case 240: DISPATCH_SMS(240) break;
            default: LIGHTOP_HOST_UNREACHABLE("unsupported SMS value\n");
        }
    });

    return schedule_metadata;
}

torch::Tensor paged_mqa_logits(const torch::Tensor& q,
                               const torch::Tensor& fused_kv_cache,
                               const torch::Tensor& weights,
                               const torch::Tensor& context_lens,
                               const torch::Tensor& block_table,
                               std::optional<torch::Tensor>& schedule_meta,
                               const int& max_context_len,
                               const bool& clean_logits,
                               const bool& is_prefill) {
    if (is_prefill) {
        return paged_mqa_logits_prefill(q, fused_kv_cache, weights, context_lens, block_table,
                                        schedule_meta, max_context_len, clean_logits);
    }

    // Declare these variables explicitly to avoid mis-binding in kernel launch lambdas
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf;
    bool use_meta = schedule_meta.has_value();

    std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q);
    std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_] = get_shape<1>(context_lens);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size__, max_block_len] = get_shape<2>(block_table);
    const auto& num_sms = device_runtime->get_num_sms();
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);

    if (use_meta) {
        unsigned int BW_SMS = get_sms_info();
        auto& schedule_meta_ = schedule_meta.value();
        const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta_);
        LIGHTOP_HOST_ASSERT(schedule_meta_.is_contiguous());
        LIGHTOP_HOST_ASSERT(schedule_meta_size == BW_SMS + 1 and meta_info_size == 2);
        LIGHTOP_HOST_ASSERT(schedule_meta_.scalar_type() == torch::kInt32);
    }

    LIGHTOP_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    LIGHTOP_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    LIGHTOP_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);

    LIGHTOP_HOST_ASSERT(next_n >= 1 and next_n <= 5);
    LIGHTOP_HOST_ASSERT(block_kv == 4 or block_kv == 8 or block_kv == 16 or block_kv == 32 or block_kv == 64);

    LIGHTOP_HOST_ASSERT(q.is_contiguous());
    LIGHTOP_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    LIGHTOP_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    LIGHTOP_HOST_ASSERT(weights.is_contiguous());
    LIGHTOP_HOST_ASSERT(context_lens.is_contiguous());
    LIGHTOP_HOST_ASSERT(block_table.stride(1) == 1);

    LIGHTOP_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    LIGHTOP_HOST_ASSERT(context_lens.scalar_type() == torch::kInt32);
    LIGHTOP_HOST_ASSERT(block_table.scalar_type() == torch::kInt32);

    torch::Tensor kv_cache_scales;
    torch::Tensor kv_cache;
    const bool q_is_fp8 = q.scalar_type() == torch::kFloat8_e4m3fn || q.scalar_type() == torch::kFloat8_e5m2;
    const bool kv_is_fp8 = fused_kv_cache.scalar_type() == torch::kByte;
    const bool q_is_int8 = q.scalar_type() == torch::kInt8;
    const bool kv_is_int8 = fused_kv_cache.scalar_type() == torch::kInt8;
    const bool kv_is_quantized = kv_is_fp8 || kv_is_int8;

    if (kv_is_quantized) {
        LIGHTOP_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
        LIGHTOP_HOST_ASSERT(kv_is_int8 ? q_is_int8 : (q_is_fp8 || q.scalar_type() == torch::kBFloat16));
        const auto kv_dtype = kv_is_int8 ? torch::kInt8 : torch::kFloat8_e4m3fn;

        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(kv_dtype)
        );

        kv_cache_scales = torch::from_blob(
            reinterpret_cast<uint8_t*>(fused_kv_cache.data_ptr()) + block_kv * head_dim,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
    } else {
        LIGHTOP_HOST_ASSERT(!q_is_fp8 && !q_is_int8);
        LIGHTOP_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kFloat16 ||
                            fused_kv_cache.scalar_type() == torch::kBFloat16);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat16)
        );
    }

    constexpr int num_math_warp_groups = 4;
    constexpr int mmac_row_tile = 16;
    constexpr int kTokensPerWarp = 64;
    constexpr int kTokensPerCta = kTokensPerWarp * num_math_warp_groups;

    const bool use_bf16_q_fp8_kv = kv_is_fp8 && q.scalar_type() == torch::kBFloat16;
    const int aligned_max_context_len = block_kv < mmac_row_tile ?
        align(max_context_len, kTokensPerCta) : align(max_context_len, block_kv);

    // Allocate output
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

    // Calculate shared memory size
    constexpr int num_q_stages = 1, num_kv_stages = 1;

    constexpr int smem_q_size_per_stage = 0;
    constexpr int aligned_smem_weight_size_per_stage = 0;
    constexpr int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage);

    // for fp8: prefetch_seq_kv_stage = 2, k_tile = 64;
    // for fp16: prefetch_seq_kv_stage = 1, k_tile = 32;
    // smem_per_warp = prefetch_seq_kv_stage * head_dim * seq_kv_tile(16) * sizeof(T) = 32 * head_dim
    const int smem_kv_pipe_size = num_kv_stages * num_math_warp_groups * 32 * head_dim;
    const int smem_q_cache_size = num_heads * head_dim * static_cast<int>(q.element_size());
    // The BF16/FP16 kernels reuse the Q cache for weights when a full four-wave H64 Q tile fills LDS. Quantized-Q
    // kernels retain Q, weights, and scales together until all three have been consumed.
    const bool weights_reuse_q_cache = !q_is_fp8 && !q_is_int8 && num_heads >= mmac_row_tile * num_math_warp_groups;
    const int smem_weight_cache_size =
        weights_reuse_q_cache ? 0 : num_heads * static_cast<int>(sizeof(float));
    const int smem_scale_cache_size =
        (q_is_fp8 || q_is_int8) ? num_math_warp_groups * 64 * static_cast<int>(sizeof(float)) : 0;
    const int smem_cache_size = smem_q_cache_size + smem_weight_cache_size + smem_scale_cache_size;

    const int smem_size = smem_q_pipe_size +
        (block_kv < 16 ? (smem_kv_pipe_size > smem_cache_size ? smem_kv_pipe_size : smem_cache_size)
                       : smem_kv_pipe_size);
    // The regular native-FP8 kernel reuses the first 8 KiB for Q, metadata, scales, and wave-private KV tiles. Its
    // 1 KiB result scratch starts at offset 8 KiB, so no LDS above 9 KiB is addressable by this kernel.
    constexpr int fp8_regular_tile_smem_size = num_math_warp_groups * mmac_row_tile * 128 * sizeof(uint8_t);
    constexpr int fp8_regular_result_smem_size = kTokensPerCta * static_cast<int>(sizeof(float));
    constexpr int fp8_regular_smem_size = fp8_regular_tile_smem_size + fp8_regular_result_smem_size;
    const int max_chunk_size = ceil_div(max_context_len, kTokensPerCta);

    // Dispatch implementation
    dim3 grid = {(uint32_t)max_chunk_size, (uint32_t)batch_size, 1};
    const dim3 multi_next_grid = {(uint32_t)max_chunk_size, (uint32_t)batch_size, 1};
    const auto make_multi_next_legacy_grid = [&](int dispatched_next_n) {
        return dim3{(uint32_t)(max_chunk_size * dispatched_next_n), (uint32_t)batch_size, 1};
    };
    dim3 block = {(uint32_t)(num_math_warp_groups * WARP_SIZE_GPU), 1, 1};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (!kv_is_quantized) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                q.scalar_type(),
                "paged_mqa_float16",
                ([&] {
                    DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
                        DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, [&] {
                            const dim3 legacy_grid = make_multi_next_legacy_grid(NEXT_N);
                            DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                                constexpr bool kPageMajor = BLOCK_KV >= 16;
                                constexpr bool kSmallHead = NUM_HEADS <= 16;
                                constexpr bool kUseRegKv =
                                    kPageMajor && kSmallHead && (NEXT_N == 3 || NEXT_N == 4);
                                constexpr bool kUseLdsKv =
                                    kPageMajor && NEXT_N >= 2 && (NEXT_N <= 3 || kSmallHead);
                                // The one-CTA multi-next launch regresses stably for
                                // N2/H64, so this new optimization keeps that shape
                                // on the pre-existing legacy path.
                                constexpr bool kN2WideHeadNeedsLegacy =
                                    NEXT_N == 2 && NUM_HEADS == 64;
                                constexpr bool kUseLdsKvMultiNext =
                                    kUseLdsKv && !kN2WideHeadNeedsLegacy;

                                if constexpr (kUseRegKv) {
                                    dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                                    paged_mqa_logits_multi_next_reg_qw<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, true>
                                        <<<multi_next_grid, multi_next_block, 0, stream>>>(
                                            reinterpret_cast<scalar_t*>(q.data_ptr()),
                                            reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                            weights.data_ptr<float>(), batch_size,
                                            num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                            block_table_stride, context_lens.data_ptr<int32_t>(),
                                            logits.data_ptr<float>(),
                                            block_table.data_ptr<int32_t>(),
                                            use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                                } else if constexpr (kUseLdsKvMultiNext) {
                                    dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                                    constexpr int multi_next_smem_size = 4 * 16 * 128 * sizeof(scalar_t);
                                    paged_mqa_logits_multi_next_reg_qw<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV>
                                        <<<multi_next_grid, multi_next_block, multi_next_smem_size, stream>>>(
                                            reinterpret_cast<scalar_t*>(q.data_ptr()),
                                            reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                            weights.data_ptr<float>(), batch_size,
                                            num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                            block_table_stride, context_lens.data_ptr<int32_t>(),
                                            logits.data_ptr<float>(),
                                            block_table.data_ptr<int32_t>(),
                                            use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                                } else if constexpr (BLOCK_KV < 16) {
                                    paged_mqa_logits_small_block<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV,
                                        kTokensPerWarp, num_math_warp_groups><<<legacy_grid, block, smem_size, stream>>>(
                                        reinterpret_cast<scalar_t*>(q.data_ptr()),
                                        reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                        weights.data_ptr<float>(), batch_size,
                                        num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                        block_table_stride, context_lens.data_ptr<int32_t>(),
                                        logits.data_ptr<float>(),
                                        block_table.data_ptr<int32_t>(),
                                        use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                                } else {
                                    constexpr int split_kv = BLOCK_KV * 4;
                                    paged_mqa_logits<
                                        scalar_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                                        num_math_warp_groups, kTokensPerWarp><<<legacy_grid, block, smem_size, stream>>>(
                                        reinterpret_cast<scalar_t*>(q.data_ptr()),
                                        reinterpret_cast<scalar_t*>(kv_cache.data_ptr()),
                                        weights.data_ptr<float>(), batch_size,
                                        num_kv_blocks, kv_cache_stride_bytes, aligned_max_context_len,
                                        block_table_stride, context_lens.data_ptr<int32_t>(),
                                        logits.data_ptr<float>(),
                                        block_table.data_ptr<int32_t>(),
                                        use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                                }
                            });
                        });
                    });
                })
        );
    } else if (use_bf16_q_fp8_kv) {
        const int bf16_fp8_num_kv_stages = num_heads == 64 ? 2 : 1;
        const int bf16_fp8_raw_kv_smem_size = bf16_fp8_num_kv_stages * num_math_warp_groups * mmac_row_tile * head_dim;
        const int bf16_fp8_q_smem_size = num_heads * head_dim * static_cast<int>(sizeof(at::BFloat16));
        const int bf16_fp8_cache_smem_size = num_heads >= mmac_row_tile * num_math_warp_groups
                                                 ? bf16_fp8_q_smem_size
                                                 : bf16_fp8_q_smem_size + smem_weight_cache_size;
        const int bf16_fp8_smem_size =
            bf16_fp8_raw_kv_smem_size > bf16_fp8_cache_smem_size ? bf16_fp8_raw_kv_smem_size : bf16_fp8_cache_smem_size;

        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    if constexpr (BLOCK_KV < 16) {
                        paged_mqa_logits_bf16_fp8_kv_small_block<NEXT_N, NUM_HEADS, BLOCK_KV,
                                                                 kTokensPerWarp, num_math_warp_groups>
                            <<<grid, block, bf16_fp8_smem_size, stream>>>(
                                reinterpret_cast<at::BFloat16*>(q.data_ptr()),
                                reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), kv_cache_stride_bytes, aligned_max_context_len,
                                block_table_stride, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>());
                    } else {
                        paged_mqa_logits_bf16_fp8_kv<NEXT_N, NUM_HEADS, BLOCK_KV, num_math_warp_groups,
                                                     kTokensPerWarp>
                            <<<grid, block, bf16_fp8_smem_size, stream>>>(
                                reinterpret_cast<at::BFloat16*>(q.data_ptr()),
                                reinterpret_cast<uint8_t*>(kv_cache.data_ptr()), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), kv_cache_stride_bytes, aligned_max_context_len,
                                block_table_stride, context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                block_table.data_ptr<int32_t>());
                    }
                });
            });
        });
    } else if (kv_is_int8) {
        DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
            DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, [&] {
                const dim3 legacy_grid = make_multi_next_legacy_grid(NEXT_N);
                DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                    // N1 remains on the unchanged legacy kernel.  Page-major H8/H16
                    // multi-next cases use Register-QW: it loads every KV tile once
                    // and reuses it for every next query without LDS Q/weight reads.
                    // B4/B8 and H32/H64 retain their established kernels.
                    constexpr bool kUseRegisterQwMultiNextKernel =
                        NEXT_N >= 2 && BLOCK_KV >= 16 && NUM_HEADS <= 16;
                    if constexpr (kUseRegisterQwMultiNextKernel) {
                        dim3 multi_next_block = {(uint32_t)(4 * WARP_SIZE_GPU), 1, 1};
                        paged_mqa_logits_int8_multi_next_reg_qw<int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV>
                            <<<multi_next_grid, multi_next_block, 0, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), nullptr);
                    } else if constexpr (BLOCK_KV < 16) {
                        paged_mqa_logits_int8_small_block<
                            int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, kTokensPerWarp, num_math_warp_groups>
                            <<<legacy_grid, block, smem_size, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), nullptr);
                    } else {
                        constexpr int split_kv = BLOCK_KV * num_math_warp_groups;
                        paged_mqa_logits_int8<
                            int8_t, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv, num_math_warp_groups, kTokensPerWarp>
                            <<<legacy_grid, block, smem_size, stream>>>(
                                q.data_ptr<int8_t>(), kv_cache.data_ptr<int8_t>(), kv_cache_scales.data_ptr<float>(),
                                weights.data_ptr<float>(), batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                aligned_max_context_len, block_table_stride, context_lens.data_ptr<int32_t>(),
                                logits.data_ptr<float>(), block_table.data_ptr<int32_t>(), nullptr);
                    }
                });
            });
        });
    } else {
        DISPATCH_FP8_TYPES(q.scalar_type(), FP8_TYPE, [&] {
               DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, [&] {
                   DISPATCH_NEXT_N(next_n, NEXT_N, [&] {
                        DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, [&] {
                           if constexpr (BLOCK_KV < 16) {
                               paged_mqa_logits_fp8_small_block<
                                   FP8_TYPE, NEXT_N, NUM_HEADS, 128, BLOCK_KV,
                                   kTokensPerWarp, num_math_warp_groups><<<grid, block,smem_size, stream>>>(
                                   reinterpret_cast<FP8_TYPE*>(q.data_ptr()),
                                   reinterpret_cast<FP8_TYPE*>(kv_cache.data_ptr()),
                                   kv_cache_scales.data_ptr<float>(), weights.data_ptr<float>(),
                                   batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                   aligned_max_context_len, block_table_stride,
                                   context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                   block_table.data_ptr<int32_t>(),
                                   use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                           } else {
                               constexpr int split_kv = BLOCK_KV * 4;
                               paged_mqa_logits_fp8<
                                   FP8_TYPE, NEXT_N, NUM_HEADS, 128, BLOCK_KV, split_kv,
                                   num_math_warp_groups, kTokensPerWarp>
                                   <<<grid, block, fp8_regular_smem_size, stream>>>(
                                   reinterpret_cast<FP8_TYPE*>(q.data_ptr()),
                                   reinterpret_cast<FP8_TYPE*>(kv_cache.data_ptr()),
                                   kv_cache_scales.data_ptr<float>(), weights.data_ptr<float>(),
                                   batch_size, num_kv_blocks, kv_cache_stride_bytes,
                                   aligned_max_context_len, block_table_stride,
                                   context_lens.data_ptr<int32_t>(), logits.data_ptr<float>(),
                                   block_table.data_ptr<int32_t>(),
                                   use_meta ? schedule_meta.value().data_ptr<int32_t>() : nullptr);
                           }
                       });
                   });
               });
        });
    }

    hipError_t result = hipGetLastError();
    if (result != hipSuccess) {
        printf("launch compute logits kernel error: %d\n", (int)result);
    }

    return logits;
}

}
}
