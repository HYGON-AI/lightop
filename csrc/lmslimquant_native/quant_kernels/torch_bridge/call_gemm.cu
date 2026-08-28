#include "lmslimquant_native/quant_kernels/torch_bridge/call_gemm.h"
#include <hip/hip_runtime.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>


torch::Tensor run_gemm_w4a16_v2(torch::Tensor input,torch::Tensor weight,torch::Tensor zeros_and_scales,//torch::Tensor output,
                    int m,int n,int k,int StrideA,int StrideB,int StrideB_padded,int StrideC,int group_size,
                    torch::Tensor splitK_padA_workspace,int splitK_padA_workspace_size,string modelsize_key_ptr) { //,hipStream_t stream_id
    // 检查tensor是否在GPU上
    if (!input.is_cuda()||!weight.is_cuda()||!zeros_and_scales.is_cuda()) {
        throw std::runtime_error("all Tensor must be on the CUDA device.");
    }

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    at::Tensor output = torch::empty({m,n}, options);

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    gemm_w4a16(reinterpret_cast<const void*>(input.data_ptr()),
                            reinterpret_cast<const void*>(weight.data_ptr()),
                            reinterpret_cast<const void*>(zeros_and_scales.data_ptr()),
                            reinterpret_cast<void*> (output.data_ptr()),
                            m,  //bs
                            n,
                            k,
                            StrideA,
                            StrideB,
                            StrideC,
                            group_size,
                            reinterpret_cast<void*> (splitK_padA_workspace.data_ptr()),
                            splitK_padA_workspace_size,
                            stream,
                            (void*)modelsize_key_ptr.c_str()
                        );
    return output;
}

void run_awq_input_padding(torch::Tensor input, int m,int k,int StrideB_padded, torch::Tensor splitK_padA_workspace,int splitK_padA_workspace_size) {
    //,hipStream_t stream_id
    // 检查tensor是否在GPU上
    if (!input.is_cuda()) {
        throw std::runtime_error("all Tensor must be on the CUDA device.");
    }
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    input_padding(reinterpret_cast<const void*>(input.data_ptr()),
                    reinterpret_cast<void*> (splitK_padA_workspace.data_ptr()),
                    splitK_padA_workspace_size,
                    m,
                    k,
                    StrideB_padded,
                    stream);
    return;
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

void run_awq_dequant_colmajor(torch::Tensor dqweight,
                                torch::Tensor qweight,
                                torch::Tensor zeros_and_scales,
                                int64_t k,
                                int64_t k_pad,
                                int64_t n,
                                int64_t group_size,
                                int64_t workspace_size)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));
        // int workspace_size = k * n * 2 + 1;
        const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
        dequant_w4_gemm_colmajor(
                                (const uint32_t*) qweight.data_ptr(),
                                (const half2*) zeros_and_scales.data_ptr(),
                                (half*) dqweight.data_ptr(),
                                workspace_size,
                                n,
                                k,
                                k_pad,
                                group_size,
                                stream);
        return;
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