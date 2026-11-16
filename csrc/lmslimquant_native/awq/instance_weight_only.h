// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <unordered_map>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device_wrapper/device_gemm_multiple_b_col_int4_xdl_cshuffle.hpp"
#include "gemv_batch.h"
#include "gemm_utils.h"


namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ALayout = Row;
using BLayout = Col;
using ELayout = Row;
using SpkLayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = weight_only_tools::Dequation;
using CDEElementOp = PassThrough;

using SpkDataType       = F16;
using EDataType         = F16;
using ADataType         = ck::half_t;
using BDataType0        = uint32_t;
using BDataType1        = uint32_t;
using AccDataType       = float;
using CShuffleDataType  = ck::half_t;
using COutDataType      = ck::half_t;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace weightonly {

using GemmMultipleB_col_V1 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, ck::Tuple<>, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0,BDataType1>, AccDataType, CShuffleDataType, ck::Tuple<>, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        512,    64,   128,    64,   16,  16,   16,   16,   1,    4,        1,           2>;

using GemmMultipleB_col_V2 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, ck::Tuple<>, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0,BDataType1>, AccDataType, CShuffleDataType, ck::Tuple<>, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        64,     16,    64,    64,   16,  16,   16,   16,   1,    4,        1,           1>;

using GemmMultipleB_col_V3 = ck::tensor_operation::device_wrapper::DeviceGemmMultipleB_col_Xdl_CShuffle<
    ck::Tuple<ALayout>, ck::Tuple<BLayout, BLayout>, ck::Tuple<>, ELayout,
    ck::Tuple<ADataType>, ck::Tuple<BDataType0,BDataType1>, AccDataType, CShuffleDataType, ck::Tuple<>, EDataType,
    AElementOp, BElementOp, CDEElementOp,
//    | Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|     CShuffle|    CShuffle|
//    | Size | Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|  MXdlPerWave| NXdlPerWave|
//    |      |      |      |      |    |    |     |     | Wave| Wave|   PerShuffle|  PerShuffle|
        64,     16,    16,   256,   16,  16,   16,   16,   1,    1,        1,           1>;

}   // namespace weightonly


template <typename DeviceOpStu>
__host__ void profiler_gemm_xdl_int4(const void* A,
                                     const void* B0,
                                     const void* B1,
                                     void* C,
                                     int M,
                                     int N,
                                     int K,
                                     int StrideA,
                                     int StrideB,
                                     int StrideC,
                                     int SplitK_Batch,
                                     int GroupSize,
                                     void* splitK_workspace, // 用于SplitK添加pad的显存空间
                                     hipStream_t stream_id) {
    //// 1.设置模板，运行kernel
    auto device_op = DeviceOpStu{};

    auto invoker = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(std::array<const void*, 1>{A},
                                           std::array<const void*, 2>{B0, B1},
                                           {},
                                           C,
                                           M,
                                           N,
                                           K,
                                           std::array<ck::index_t, 1>{StrideA},
                                           std::array<ck::index_t, 2>{StrideB, StrideB},
                                           {},
                                           StrideC,
                                           AElementOp{},
                                           BElementOp{},
                                           CDEElementOp{});
    if(!device_op.IsSupportedArgument(argument)) {
        throw std::runtime_error("wrong! device_gemm with the specified compilation parameters does "
                                 "not support this GEMM problem");
    }
    invoker.Run(argument, StreamConfig{stream_id, false});
}

__host__ void profiler_gemv_int4(
    const void* A,
    const void* B0,
    const void* B1,
    void* C,
    int M,
    int N,
    int K,
    int StrideA,
    int StrideB,
    int StrideC,
    int SplitK_Batch,
    int GroupSize,
    void* splitK_workspace, // 用于SplitK添加pad的显存空间
    hipStream_t stream_id
) {
    run_gemv_batch_wrapper(A, B0, B1, C,
        M, N, K, StrideA, StrideB, StrideC, GroupSize, stream_id);
}

}   // namespace instance
