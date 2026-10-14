// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device_wrapper/device_global_padding.hpp"

#include "gemm_utils.h"

#pragma once

namespace global_padding {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ALayout                         = Row;
using AElementOp                      = PassThrough;
static constexpr int PaddingKPerBlock = 256;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// M小于512时最快模板
template <typename ADataType>
using DeviceOpInstance_Half_GlobalPadding_M_S512 =
    ck::tensor_operation::device_wrapper::DeviceGloabalPadding<ALayout,
                                                               ALayout,
                                                               ADataType,
                                                               ADataType,
                                                               AElementOp,
                                                               //  | BlockSize | MPerBlock |      KPerBlock     | AK1 |
                                                               512,
                                                               32,
                                                               PaddingKPerBlock,
                                                               8>;

template <typename ADataType>
using DeviceOpInstance_Half_GlobalPadding_M_512_1024 =
    ck::tensor_operation::device_wrapper::DeviceGloabalPadding<ALayout, ALayout, ADataType, ADataType, AElementOp, 512, 256, PaddingKPerBlock, 8>;
// M大于1024时最快模板
template <typename ADataType>
using DeviceOpInstance_Half_GlobalPadding_M_L1024 =
    ck::tensor_operation::device_wrapper::DeviceGloabalPadding<ALayout, ALayout, ADataType, ADataType, AElementOp, 512, 512, PaddingKPerBlock, 8>;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DeviceOpStu>
__host__ void launch_global_padding(
    const void* A, void* C, int M, int K, int K_Padded, int StrideA, int StrideA_Padded, int SplitK_Batch, bool fill_zero_const, hipStream_t stream_id) {
    //// 1.设置模板，运行kernel
    auto device_op = DeviceOpStu{};

    auto invoker = device_op.MakeInvoker();

    auto argument = device_op.MakeArgument(A, C, M, K, K_Padded, StrideA, StrideA_Padded, AElementOp{}, SplitK_Batch, fill_zero_const);
    if(!device_op.IsSupportedArgument(argument)) {
        throw std::runtime_error("wrong! device_gemm with the specified compilation parameters does "
                                 "not support this GEMM problem");
    }
    invoker.Run(argument, StreamConfig{stream_id, false});
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__host__ int get_half_mat_global_padding_workspace_size(int M, int K, int K_Padded) {
    using ADataType = ck::half_t;

    const int matA_padded_element_cnt = M * ck::math::max(K, K_Padded);
    const int padA_workspace_size     = matA_padded_element_cnt * sizeof(ADataType);
    return padA_workspace_size;
}

__host__ void launch_half_mat_global_padding(
    const void* A, void* C, int M, int K, int K_Padded, int StrideA, int StrideA_Padded, bool fill_zero_const, hipStream_t stream_id) {
    int padding_SplitK_Batch = K / PaddingKPerBlock; // global padding: K/KPerBlock
    using ADataType          = ck::half_t;
    // do padding
    using DeviceOpStu_GlobalPadding_M_S512     = DeviceOpInstance_Half_GlobalPadding_M_S512<ADataType>;
    using DeviceOpStu_GlobalPadding_M_512_1024 = DeviceOpInstance_Half_GlobalPadding_M_512_1024<ADataType>;
    using DeviceOpStu_GlobalPadding_M_L1024    = DeviceOpInstance_Half_GlobalPadding_M_L1024<ADataType>;

    if(M <= 512) {
        launch_global_padding<DeviceOpStu_GlobalPadding_M_S512>(A,
                                                                C,
                                                                M,
                                                                K,
                                                                K_Padded, // K_Padded
                                                                StrideA,
                                                                StrideA_Padded,
                                                                padding_SplitK_Batch,
                                                                fill_zero_const, // fill_zero_const,
                                                                stream_id);
    } else if(M <= 1024) {
        launch_global_padding<DeviceOpStu_GlobalPadding_M_512_1024>(A,
                                                                    C,
                                                                    M,
                                                                    K,
                                                                    K_Padded, // K_Padded
                                                                    StrideA,
                                                                    StrideA_Padded,
                                                                    padding_SplitK_Batch,
                                                                    fill_zero_const, // fill_zero_const,
                                                                    stream_id);
    } else {
        launch_global_padding<DeviceOpStu_GlobalPadding_M_L1024>(A,
                                                                 C,
                                                                 M,
                                                                 K,
                                                                 K_Padded, // K_Padded
                                                                 StrideA,
                                                                 StrideA_Padded,
                                                                 padding_SplitK_Batch,
                                                                 fill_zero_const, // fill_zero_const,
                                                                 stream_id);
    }
}

} // namespace global_padding
