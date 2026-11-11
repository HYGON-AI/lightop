// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "lmslimquant_native/quant_kernels/torch_bridge/call_gemm.h"
#include <hip/hip_runtime.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>


torch::Tensor run_gemm_w4a16_v2(torch::Tensor input, torch::Tensor weight,
    torch::Tensor zeros_and_scales, int m, int n, int k, int StrideA,
    int StrideB, int StrideB_padded, int StrideC, int group_size,
    torch::Tensor splitK_padA_workspace, int splitK_padA_workspace_size,
    string modelsize_key_ptr) {
  TORCH_CHECK(m >= 0 && n >= 0 && k >= 0 && StrideA >= k &&
              StrideB >= k && StrideB_padded == StrideB && StrideC == n,
              "invalid AWQ dimensions or strides");
  TORCH_CHECK(group_size > 0 && group_size % 8 == 0 &&
              k % group_size == 0 && StrideB % group_size == 0,
              "AWQ K and weight stride must contain complete quantization groups");
  const auto device = input.device();
  lightop::awq::check_tensor(input, at::kHalf, device,
      m == 0 ? 0 : int64_t(m - 1) * StrideA + k, "input");
  lightop::awq::check_tensor(weight, at::kInt, device,
      int64_t(n) * (StrideB / 8), "weight");
  lightop::awq::check_tensor(zeros_and_scales, at::kInt, device,
      int64_t(n) * (StrideB / group_size), "scale_zero");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(input));
  auto output = at::empty({m, n}, input.options());
  if (m == 0 || n == 0) return output;
  if (k == 0) return output.zero_();
  // Legacy workspace/config arguments remain in the ABI. Source kernels own
  // their temporary storage; vendor template names do not select this backend.
  auto a = input.as_strided({m, k}, {StrideA, 1});
  if (m <= 64 && n % 16 == 0 && k % 128 == 0 && group_size % 32 == 0) {
    a = a.contiguous();
    auto packed = weight.view({-1}).narrow(0, 0, int64_t(n)*(StrideB/8))
                      .view({n, StrideB / 8}).narrow(1, 0, k / 8).contiguous();
    auto sz = zeros_and_scales.view({-1}).narrow(0, 0, int64_t(n)*(StrideB/group_size))
                  .view({n, StrideB / group_size})
                  .narrow(1, 0, k / group_size).contiguous();
    at::native::gemm_awq_w4a16(output, a, packed, sz, 1, 1, 16, 128, 1, false);
  } else {
    auto dequant = at::empty({n, k}, input.options());
    lightop::awq::dequant_colmajor(dequant, weight, zeros_and_scales, n, k, StrideB, group_size);
    at::mm_out(output, a, dequant.transpose(0, 1));
  }
  return output;
}

void run_awq_input_padding(torch::Tensor input, int m, int k, int padded_k,
    torch::Tensor workspace, int workspace_size) {
  TORCH_CHECK(m >= 0 && k >= 0 && padded_k >= k, "invalid AWQ padding dimensions");
  const int64_t count = int64_t(m) * padded_k;
  lightop::awq::check_tensor(input, at::kHalf, input.device(), int64_t(m) * k, "input");
  lightop::awq::check_tensor(workspace, at::kHalf, input.device(), count, "workspace");
  TORCH_CHECK(workspace_size >= 0 && int64_t(workspace_size) >= count * 2,
              "AWQ padding workspace_size is insufficient");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(input));
  auto output = workspace.view({-1}).narrow(0, 0, count).view({m, padded_k});
  auto source = input.view({-1}).narrow(0, 0, int64_t(m) * k).view({m, k});
  at::assert_no_overlap(output, source);
  output.zero_();
  output.narrow(1, 0, k).copy_(source);
}

void run_dequant_gemm_rocblas_gpu(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor deweight,
        torch::Tensor zeros_and_scales,
        torch::Tensor output,
        int m,
        int n,
        int k,
        int group_size)
    {

        if (!input.is_cuda()||!weight.is_cuda()||!deweight.is_cuda()||!zeros_and_scales.is_cuda()||!output.is_cuda()) {
            throw std::runtime_error("all Tensor must be on the CUDA device.");
        }

        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

        half* _input = reinterpret_cast<half*>(input.data_ptr<at::Half>());
        half* _output = reinterpret_cast<half*>(output.data_ptr<at::Half>());
        half* _deweight = reinterpret_cast<half*>(deweight.data_ptr<at::Half>());

        uint32_t* _weight = reinterpret_cast<uint32_t*>(weight.data_ptr<int32_t>());
        uint32_t* _zeros_and_scales = reinterpret_cast<uint32_t*>(zeros_and_scales.data_ptr<int32_t>());
        launch_dequant_gemm_rocblas_gpu(stream,
                        reinterpret_cast<void*> (_output),
                        reinterpret_cast<const void*> (_input),
                        reinterpret_cast<const void*> (_zeros_and_scales),
                        reinterpret_cast<const void*> (_weight),
                        m,
                        n,
                        k,
                        reinterpret_cast<void*> (_deweight),
                        group_size
                        );
    }


void run_convert_s4_k_m8(torch::Tensor _qw,
                            torch::Tensor _sz,
                            torch::Tensor _ws,
                            torch::Tensor qw,
                            torch::Tensor scales,
                            torch::Tensor qzeros,
                            int   m,
                            int   k,
                            int   group_size)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(qw));
        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

        return convert_s4_k_m8_(
                                (uint32_t*) _qw.data_ptr(),
                                (half2*) _sz.data_ptr(),
                                (half*) _ws.data_ptr(),
                                (const uint32_t*) qw.data_ptr(),
                                (const half*) scales.data_ptr(),
                                (const uint32_t*) qzeros.data_ptr(),
                                m,
                                k,
                                group_size,
                                stream
                                );
    }

void run_permute_trans(torch::Tensor _sz,
                        torch::Tensor sz,
                        int64_t dim0,
                        int64_t n)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(sz));
        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        return permute_trans((uint32_t*) _sz.data_ptr(),
                                (uint32_t*) sz.data_ptr(),
                                dim0,
                                n,
                                stream
                                );

    }

void run_awq_dequant_colmajor(torch::Tensor output, torch::Tensor weight,
    torch::Tensor scale_zero, int64_t k, int64_t padded_k, int64_t n,
    int64_t group_size, int64_t workspace_size) {
  TORCH_CHECK(n >= 0 && k >= 0 && n <= INT32_MAX && k <= INT32_MAX,
              "invalid AWQ output dimensions");
  TORCH_CHECK(workspace_size >= 0 && workspace_size >= n * k * 2,
              "AWQ dequant workspace_size is insufficient");
  lightop::awq::dequant_colmajor(output, weight, scale_zero, n, k, padded_k, group_size);
}

void run_dequant_w4_gemm_colmajor(torch::Tensor dqweight,
                                torch::Tensor qweight,
                                torch::Tensor zeros_and_scales,
                                int64_t k,
                                int64_t n,
                                int64_t group_size)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        return dequant_w4_gemm_colmajor(stream,
                                        (half*) dqweight.data_ptr(),
                                        (const uint32_t*) qweight.data_ptr(),
                                        (const half2*) zeros_and_scales.data_ptr(),
                                        k,
                                        n,
                                        group_size);
    }

torch::Tensor rocblas_gemm_int8(torch::Tensor input,torch::Tensor weight,bool layout_tn) { //,hipStream_t stream_id

        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
        //at::Tensor output = torch::empty({m,n}, options);
        //gemm layout为NN
        int m=input.size(0);
        int k=input.size(1);
        int n=weight.size(1);

        at::Tensor output = torch::zeros({m,n}, options);

        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

        //at::cuda::getCurrentCUDABlasHandle()
        gemm_rocblas_int8_gpu(stream,
                                (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
                                (int32_t*)output.data_ptr(),
                                (const int8_t*)input.data_ptr(),
                                (const int8_t*)weight.data_ptr(),
                                m,
                                n,
                                k,
                                layout_tn);
        return output;
    }

// torch::Tensor cutlass_gemm_w8a8(torch::Tensor input,
//                                     torch::Tensor weight,
//                                     torch::Tensor scale_x,
//                                     torch::Tensor scale_y,
//                                     torch::Tensor bias,
//                                     int m,
//                                     int n,
//                                     int k
//                                     )
//     {
//         const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//         auto options = torch::TensorOptions().dtype(torch::kHalf).device(input.device());
//         //at::Tensor output = torch::empty({m,n}, options);
//         at::Tensor output = torch::zeros({m,n}, options);

//         const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
//         if (!bias.numel() == 0)
//             w8a8_gemm(reinterpret_cast<const void*>(input.data_ptr()),
//                         reinterpret_cast<const void*>(weight.data_ptr()),
//                         reinterpret_cast<const void*>(scale_x.data_ptr()),
//                         reinterpret_cast<const void*>(scale_y.data_ptr()),
//                         reinterpret_cast<const void*>(bias.data_ptr()),
//                         reinterpret_cast<void*>(output.data_ptr()),
//                         m,
//                         n,
//                         k,
//                         stream
//             );
//         else
//             w8a8_gemm(reinterpret_cast<const void*>(input.data_ptr()),
//                         reinterpret_cast<const void *>(weight.data_ptr()),
//                         reinterpret_cast<const void*>(scale_x.data_ptr()),
//                         reinterpret_cast<const void*>(scale_y.data_ptr()),
//                         nullptr,
//                         reinterpret_cast<void*>(output.data_ptr()),
//                         m,
//                         n,
//                         k,
//                         stream
//             );
//         return output;
//     }

torch::Tensor rocblas_gemm_fp16(torch::Tensor input,torch::Tensor weight,bool layout_tn)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    //at::Tensor output = torch::empty({m,n}, options);
    //gemm layout为NN
    int m=input.size(0);
    int k=input.size(1);
    int n=weight.size(1);

    at::Tensor output = torch::zeros({m,n}, options);

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    sgemm_rocblas_fp16_gpu(stream,
                            (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
                            (half*)output.data_ptr(),
                            (const half*)input.data_ptr(),
                            (const half*)weight.data_ptr(),
                            m,
                            n,
                            k,
                            layout_tn);
    return output;
}
