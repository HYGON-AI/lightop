#include "miopen/miopen-wrapper.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <optional>
#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

static miopenTensorArgument_t setProblemTensorDescriptor(miopenProblem_t& problem,
                                                         miopenTensorArgumentId_t id,
                                                         TensorDescriptor& input,
                                                         const void* tensor_data_ptr)
{
    MIOPEN_CHECK(miopenSetProblemTensorDescriptor(problem, id, input.mut_desc()));
    return miopenTensorArgument_t{id, input.get_storage_addr(), const_cast<void*>(tensor_data_ptr)};
}

Tensor miopen_scaled_dot_product_attention(const Tensor& query,
                                           const Tensor& key,
                                           const Tensor& value,
                                           const std::optional<Tensor>& attn_mask_,
                                           const double droprate       = 0.0,
                                           const bool is_causal        = false,
                                           std::optional<double> scale = std::nullopt,
                                           const bool enable_gqa       = false)
{

    MhaDescriptor mha_descriptor;
    mha_descriptor.SetParams(1 / std::sqrt(query.sizes()[3]));
    miopenProblem_t problem = nullptr;
    MIOPEN_CHECK(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionForward));

    bool is_bshd       = false;
    const int n        = query.sizes()[0];
    const int head_q   = query.sizes()[1];
    const int seqlen_q = query.sizes()[2];
    const int seqlen_k = key.sizes()[2];
    const int dim_v    = value.sizes()[3];

    Tensor padding_mask = at::empty({n, 1, 1, 1}, query.options().dtype(at::kInt)).fill_(seqlen_k);
    Tensor bias         = at::empty({n, 1, seqlen_q, seqlen_k}, query.options().dtype(at::kFloat)).fill_(0);
    Tensor descaleQ     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat)).fill_(1);
    Tensor descaleK     = at::empty_like(descaleQ).fill_(1);
    Tensor descaleV     = at::empty_like(descaleQ).fill_(1);
    Tensor descaleS     = at::empty_like(descaleQ).fill_(1);
    Tensor scaleS       = at::empty_like(descaleQ).fill_(1);
    Tensor scaleO       = at::empty_like(descaleQ).fill_(1);
    Tensor dropProb     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat)).fill_(droprate);
    Tensor dropSeed     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kLong))
                          .fill_(c10::Scalar(static_cast<uint64_t>(0xAAFFFFFFFFull)));
    Tensor dropOffset = at::empty({1, 1, 1, 1}, query.options().dtype(at::kLong)).fill_(1);
    Tensor M          = at::empty({n, head_q, seqlen_q, 1}, query.options().dtype(at::kFloat));
    Tensor ZInv       = at::empty_like(M);
    Tensor LSE        = at::empty_like(M);
    Tensor AmaxO      = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat));
    Tensor AmaxS      = at::empty_like(AmaxO);
    Tensor O          = !is_bshd ? at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {head_q * seqlen_q * dim_v, seqlen_q * dim_v, dim_v, 1},
                                            query.options())
                                 : at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {head_q * seqlen_q * dim_v, dim_v, head_q * dim_v, 1},
                                            query.options());

    TensorDescriptor tquery(query), tkey(key), tvalue(value), tpadding_mask(padding_mask), tbias(bias),
        tdescaleQ(descaleQ), tdescaleK(descaleK), tdescaleV(descaleV), tdescaleS(descaleS), tscaleS(scaleS),
        tscaleO(scaleO), tdropProb(dropProb), tdropSeed(dropSeed), tdropOffset(dropOffset), tM(M), tZInv(ZInv),
        tLSE(LSE), tAmaxO(AmaxO), tAmaxS(AmaxS), tO(O);

    std::vector<miopenTensorArgument_t> args;

    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaQ, tquery, query.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaK, tkey, key.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaV, tvalue, value.data_ptr()));
    args.emplace_back(
        setProblemTensorDescriptor(problem, miopenTensorMhaPaddingMask, tpadding_mask, padding_mask.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaBias, tbias, bias.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaDescaleQ, tdescaleQ, descaleQ.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaDescaleK, tdescaleK, descaleK.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaDescaleV, tdescaleV, descaleV.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaDescaleS, tdescaleS, descaleS.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaScaleS, tscaleS, scaleS.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaScaleO, tscaleO, scaleO.data_ptr()));
    args.emplace_back(
        setProblemTensorDescriptor(problem, miopenTensorMhaDropoutProbability, tdropProb, dropProb.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaDropoutSeed, tdropSeed, dropSeed.data_ptr()));
    args.emplace_back(
        setProblemTensorDescriptor(problem, miopenTensorMhaDropoutOffset, tdropOffset, dropOffset.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaO, tO, O.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaM, tM, M.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaZInv, tZInv, ZInv.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaLSE, tLSE, LSE.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaAmaxO, tAmaxO, AmaxO.data_ptr()));
    args.emplace_back(setProblemTensorDescriptor(problem, miopenTensorMhaAmaxS, tAmaxS, AmaxS.data_ptr()));

    size_t found = 0;
    std::vector<miopenSolution_t> solutions(1);
    miopenHandle_t handle = getMiopenHandle_new();
    MIOPEN_CHECK(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()));
    if(found == 0)
        MIOPEN_THROW("no solution found");

    size_t workspace_size = 0;

    MIOPEN_CHECK(miopenGetSolutionWorkspaceSize(solutions[0], &workspace_size));
    Tensor workspace_tensor;

    if(workspace_size > 0)
    {
        workspace_tensor = at::empty(workspace_size, query.options());
    }

    MIOPEN_CHECK(miopenRunSolution(handle,
                                   solutions[0],
                                   args.size(),
                                   args.data(),
                                   workspace_tensor.defined() ? workspace_tensor.data_ptr() : nullptr,
                                   workspace_size));

    return O;
};

} // namespace native

} // namespace at
