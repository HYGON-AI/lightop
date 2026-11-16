// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"

#include "gemm_utils.h"
#include "gemm_multiB_col_xdl_global_padding.h"
#include "instance_maps.h"
// 命名和profiler中gemm_weight_only_template_profiler.h写入文件名一致
#include "weightonly_best_all.h"
#include "weightonly_best_noAtomicAdd.h"
#include "weightonly_best_noSplitK.h"

namespace gemm_weight_only_all {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ALayout   = Row;
using BLayout   = Col;
using SpkLayout = Row;
using ELayout   = Row;

using AElementOp   = PassThrough;
using BElementOp   = weight_only_tools::Dequation;
using CDEElementOp = PassThrough;

using SpkDataType      = F16;
using EDataType        = F16;
using ADataType        = _Float16;
using BDataType0       = uint32_t;
using BDataType1       = uint32_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using COutDataType     = ck::half_t;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GemmWeightOnly {
    //////////////////////////////////////////////////// 计算workspace size ////////////////////////////////////////////////////
    __host__ static int get_splitK_workspace(int M, int N, int K, int SplitK_Batch) {
        const int splitK_element_cnt    = M * N * SplitK_Batch;
        const int splitK_workspace_size = splitK_element_cnt * sizeof(COutDataType);
        return splitK_workspace_size;
    }

    __host__ static int get_splitK_padA_workspace(int M,
                                                  int N,
                                                  int K,
                                                  int StrideA,
                                                  int StrideB,
                                                  int StrideB_padded, // 输入的权重矩阵添加pad后的K
                                                  int SplitK_Batch,
                                                  bool using_padding_A) {
        int splitK_workspace_size = get_splitK_workspace(M, N, K, SplitK_Batch);
        int padA_workspace_size   = 0;
        if(using_padding_A) {
            padA_workspace_size = global_padding::get_half_mat_global_padding_workspace_size(M, K, StrideB_padded);
        }

        return splitK_workspace_size + padA_workspace_size;
    }

    // 能够正常执行 SplitK 和 padding A 的显存空间
    __host__ static int get_weight_only_workspace_size(int N_max, // 模型权重的size N_max
                                                       int K_max, // 模型权重sizeK最大值，计算SplitK所需显存空间
                                                       int GroupSize) {
        const int M_max                 = 256;
        const int SplitK_Batch          = 8;
        const int splitK_element_cnt    = M_max * N_max * SplitK_Batch;
        const int splitK_workspace_size = splitK_element_cnt * sizeof(COutDataType);

        const int K_padded_max                    = K_max + GroupSize * 2; // 默认padding添加 2*GroupSize
        const int padA_workspace_size             = global_padding::get_half_mat_global_padding_workspace_size(M_max, K_max, K_padded_max);
        const int weight_only_gemm_workspace_size = padA_workspace_size + splitK_workspace_size;
        // printf("padA_workspace_size=%d, splitK_workspace_size=%d, weight_only_gemm_workspace_size=%d, \n",
        //     padA_workspace_size, splitK_workspace_size, weight_only_gemm_workspace_size);

        return weight_only_gemm_workspace_size;
    }

    //////////////////////////////////////////////////// 执行gemm计算 ////////////////////////////////////////////////////
    __host__ __forceinline__ static std::string get_batchsize_string(int M) {
        // 不同的M对应于模型输入的batchsize
        std::string str_m;
        if(M < 4) {
            str_m = "M1_4";
        } else if(M < 5) {
            str_m = "M4_5";
        } else if(M < 16) {
            str_m = "M5_16";
        } else if(M < 32) {
            str_m = "M16_32";
        } else if(M < 64) {
            str_m = "M32_64";
        } else if(M < 128) {
            str_m = "M64_128";
        } else if(M < 256) {
            str_m = "M128_256";
        } else {
            str_m = "M256Upper";
        }
        return str_m;
    }

    /* 根据值查找vector下标 */
    __host__ static int get_profiler_idx(int M, int N, int K, std::vector<std::string>& modelsize_vec, int batchsize_num) {
        int vec_idx = -1;

        // 不同的N、K对应特定的AI模型size
        std::string str_m         = get_batchsize_string(M);
        std::string modelsize_key = "_N" + std::to_string(N) + "_K" + std::to_string(K);

        int start_idx = 0;
        while(start_idx < modelsize_vec.size()) {
            auto tmp_name = modelsize_vec[start_idx];
            if(tmp_name.find(modelsize_key) == std::string::npos) {
                start_idx += batchsize_num; // 大小与profiler/profile_model_size.sh中batchsize可选个数有关
                continue;
            }

            if((str_m + modelsize_key) == tmp_name) {
                vec_idx = start_idx;
                break;
            } else {
                start_idx += 1;
            }
        }

        return vec_idx;
    }

    /*
        执行一次完整的反量化gemm计算
    */
    template <bool usingPaddingA, typename DeviceOpStu>
    __host__ void run_gemm_multiB(DeviceOpStu device_op,
                                  const void* A,
                                  const void* B0,
                                  const void* B1,
                                  void* C,
                                  int M,
                                  int N,
                                  int K,
                                  int StrideA,
                                  int StrideB,
                                  int StrideB_padded, // 输入的权重矩阵添加pad后的K
                                  int StrideC,
                                  int SplitK_Batch,
                                  int GroupSize,
                                  void* splitK_padA_workspace,
                                  int padA_workspace_size,
                                  hipStream_t stream_id) {
        if constexpr(usingPaddingA) { // 添加padA的情况
            // 设置添加padding的显存位置
            void* A_padded_workspace = splitK_padA_workspace;
            // 执行padding操作
            global_padding::launch_half_mat_global_padding(A,
                                                           A_padded_workspace,
                                                           M,
                                                           K,
                                                           StrideB_padded, // K_Padded
                                                           StrideA,
                                                           StrideB_padded,
                                                           /*fill_zero_const*/ false,
                                                           stream_id);
            auto splitK_workspace =
                reinterpret_cast<void*>(reinterpret_cast<COutDataType*>(splitK_padA_workspace) + padA_workspace_size / sizeof(COutDataType));
#if 0
            printf("usingPaddingA=%d, A_padded_workspace=%p, splitK_workspace=%p, padA_workspace_size=%d, \n",
                    usingPaddingA, A_padded_workspace, splitK_workspace, padA_workspace_size);
#endif
            /////////////////////////////////////////// 计算gemm ///////////////////////////////////////////
            device_op(A_padded_workspace, B0, B1, C, M, N, K, StrideB_padded, StrideB_padded, StrideC, SplitK_Batch, GroupSize, splitK_workspace, stream_id);
        } else {
            device_op(A, B0, B1, C, M, N, K, StrideA, StrideB_padded, StrideC, SplitK_Batch, GroupSize, splitK_padA_workspace, stream_id);
        }
    }

    /*
        根据不同的选项执行一次反量化计算
    */
    template <typename DeviceOpStu>
    __host__ void processing_once_gemm_multiB(DeviceOpStu device_op,
                                              const void* A,
                                              const void* B0,
                                              const void* B1,
                                              void* C,
                                              int M,
                                              int N,
                                              int K,
                                              int StrideA,
                                              int StrideB,
                                              int StrideB_padded, // 输入的权重矩阵添加pad后的K
                                              int StrideC,
                                              int SplitK_Batch,
                                              int GroupSize,
                                              void* splitK_padA_workspace,    // 用于SplitK和tensorA添加pad的显存空间
                                              int splitK_padA_workspace_size, // workspace有多少个B（字节）
                                              hipStream_t stream_id,
                                              bool padding_A_is_faster) {
        if(padding_A_is_faster) {
            // 给输入的特征矩阵A添加pad所需空间
            const int padA_workspace_size = global_padding::get_half_mat_global_padding_workspace_size(M, K, StrideB_padded);
            // 判断预留显存空间是否充足
            const bool padA_workspace_enough = padA_workspace_size <= splitK_padA_workspace_size;
            const bool paddingA              = padding_A_is_faster && (StrideB_padded != StrideB) && padA_workspace_enough;
            if(paddingA) {
                run_gemm_multiB<true>(device_op,
                                      A,
                                      B0,
                                      B1,
                                      C,
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideB_padded,
                                      StrideC,
                                      SplitK_Batch,
                                      GroupSize,
                                      splitK_padA_workspace,
                                      padA_workspace_size,
                                      stream_id);
            } else {
                run_gemm_multiB<false>(device_op,
                                       A,
                                       B0,
                                       B1,
                                       C,
                                       M,
                                       N,
                                       K,
                                       StrideA,
                                       StrideB,
                                       StrideB_padded,
                                       StrideC,
                                       SplitK_Batch,
                                       GroupSize,
                                       splitK_padA_workspace,
                                       0,//padA_workspace_size,
                                       stream_id);
            }

        } else {
            run_gemm_multiB<false>(device_op,
                                   A,
                                   B0,
                                   B1,
                                   C,
                                   M,
                                   N,
                                   K,
                                   StrideA,
                                   StrideB,
                                   StrideB_padded,
                                   StrideC,
                                   SplitK_Batch,
                                   GroupSize,
                                   splitK_padA_workspace,
                                   0,//padA_workspace_size,
                                   stream_id);
        }
    }

    __host__ void run_weight_only_gemm(const void* A,
                                       const void* B0,
                                       const void* B1,
                                       void* C,
                                       int M,
                                       int N,
                                       int K,
                                       int StrideA,
                                       int StrideB,
                                       int StrideB_padded, // 输入的权重矩阵添加pad后的K
                                       int StrideC,
                                       int GroupSize,
                                       void* splitK_padA_workspace,    // 用于SplitK和tensorA添加pad的显存空间
                                       int splitK_padA_workspace_size, // workspace有多少个字节
                                       hipStream_t stream_id,
                                       bool splitK_atomicAdd_available // 采用SplitK是否使用atomicAdd
    ) {
        std::string gemm_name    = "";
        bool padding_A_is_faster = false;
        int SplitK_Batch         = 1;
        int batchsize_num        = 8; // 大小与profiler/profile_model_size.sh中batchSizes可选个数有关

        //////////////////////////////////////// 1.查找最优策略 ////////////////////////////////////////
#if 1
        if(M < 4) {
            gemm_name = "GemmMultipleB_col_gemv";
        } else {
            // profiler生成的.h文件中，通过std::vector容器保存key值
            auto weightonly_best_all_modelsize_vec = weightonly_best_all::weightonly_best_all_modelsize_vec;
            // 找到最佳匹配的模板名称
            auto best_idx = get_profiler_idx(M, N, K, weightonly_best_all_modelsize_vec, batchsize_num);

            if(best_idx >= 0) {
                auto input_key = weightonly_best_all_modelsize_vec[best_idx];

                //// 首先判断是否可用atomicAdd
                if(splitK_atomicAdd_available) {
                    gemm_name           = weightonly_best_all::weightonly_best_all_instance_vec[best_idx];
                    padding_A_is_faster = weightonly_best_all::weightonly_best_all_padding_vec[best_idx];
                    SplitK_Batch        = weightonly_best_all::weightonly_best_all_kbatch_vec[best_idx];
                } else {
                    gemm_name           = weightonly_best_noAtomicAdd::weightonly_best_noAtomicAdd_instance_vec[best_idx];
                    padding_A_is_faster = weightonly_best_noAtomicAdd::weightonly_best_noAtomicAdd_padding_vec[best_idx];
                    SplitK_Batch        = weightonly_best_noAtomicAdd::weightonly_best_noAtomicAdd_kbatch_vec[best_idx];
                }

                /////////////////////////////// 如果选用的是 SplitK reduce，检查reduce显存空间是否足够 ///////////////////////////////
                if(input_key.find("SplitK_reduce") != std::string::npos) {
                    // 计算当前size需要reduce时的workspace_size
                    const int workspace_size_all = get_splitK_padA_workspace(M, N, K, StrideA, StrideB, StrideB_padded, SplitK_Batch, padding_A_is_faster);

                    // 判断padding+SplitK的size是否足够，如果不够则使用atomicAdd或者通用gemm
                    if(workspace_size_all > splitK_padA_workspace_size) {
                        gemm_name           = weightonly_best_noSplitK::weightonly_best_noSplitK_instance_vec[best_idx];
                        padding_A_is_faster = weightonly_best_noSplitK::weightonly_best_noSplitK_padding_vec[best_idx];
                    }
                }
            }
        }

        // std::cout << "gemm_name:" << gemm_name << ", padding_A_is_faster:" << padding_A_is_faster << ", SplitK_Batch:" << SplitK_Batch << std::endl;
#endif
        //////////////////////////////////////// 2.执行gemm ////////////////////////////////////////
#if 1
        if(gemm_name != "") {
            auto device_op = instance::int4Gemm_available_map.at(gemm_name);

            processing_once_gemm_multiB(device_op,
                                        A,
                                        B0,
                                        B1,
                                        C,
                                        M,
                                        N,
                                        K,
                                        StrideA,
                                        StrideB,
                                        StrideB_padded,
                                        StrideC,
                                        SplitK_Batch,
                                        GroupSize,
                                        splitK_padA_workspace,
                                        splitK_padA_workspace_size,
                                        stream_id,
                                        padding_A_is_faster);
        } else {
            //printf("Warning! The current size has not been selected by the profiler.\n");
            // 没有经过profiler筛选的size，使用通用gemm计算

            if(M <= 16) {
                auto device_op = instance::int4Gemm_available_map["GemmMultipleB_col_V3"];
                run_gemm_multiB<false>(
                    device_op, A, B0, B1, C, M, N, K, StrideA, StrideB, StrideB_padded, StrideC, SplitK_Batch, GroupSize, splitK_padA_workspace, 0, stream_id);
            } else if(M <= 64) {
                auto device_op = instance::int4Gemm_available_map["GemmMultipleB_col_V2"];
                run_gemm_multiB<false>(
                    device_op, A, B0, B1, C, M, N, K, StrideA, StrideB, StrideB_padded, StrideC, SplitK_Batch, GroupSize, splitK_padA_workspace, 0, stream_id);
            } else {
                auto device_op = instance::int4Gemm_available_map["GemmMultipleB_col_V1"];
                run_gemm_multiB<false>(
                    device_op, A, B0, B1, C, M, N, K, StrideA, StrideB, StrideB_padded, StrideC, SplitK_Batch, GroupSize, splitK_padA_workspace, 0, stream_id);
            }
        }
#endif
    }

}; // struct GemmWeightOnly

} // namespace gemm_weight_only_all
