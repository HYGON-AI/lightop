#include "miopen/miopen-wrapper.h"
#include "miopen/graph/tensor.hpp"
#include "miopen/Handle.h"
#include <miopen/Exceptions.h>
#include <vector>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

Tensor mha(const Tensor& query,
           const Tensor& key,
           const Tensor& value,
           //    const int n              = 1,
           //    const int head_q         = 4,
           //    const int head_k         = -1,
           //    const int seqlen_q       = 128,
           //    const int seqlen_k       = -1,
           //    const int dim_q          = 128,
           //    const int dim_v          = -1,
           const float droprate     = 0.5,
           const std::string layout = "BHSD")
{

    std::size_t found;
    MhaDescriptor mha_descriptor;
    std::vector<miopenSolution_t> solutions(1);

    InitTensor x;
    miopenProblem_t& problem = x.getProblem();
    bool is_bshd             = layout.compare("BSHD") == 0;
    const int n              = query.sizes()[0];
    const int head_q         = query.sizes()[1];
    const int head_k         = key.sizes()[1];
    const int seqlen_q       = query.sizes()[2];
    const int seqlen_k       = key.sizes()[2];
    const int dim_q          = query.sizes()[3];
    const int dim_v          = value.sizes()[3];

    mha_descriptor.SetParams(1);
    MIOPEN_CHECK(miopenCreateMhaProblem(&problem, &mha_descriptor, miopenProblemDirectionForward));

    Tensor padding_mask = at::empty({n, 1, 1, 1}, query.options().dtype(at::kInt)).fill_(seqlen_k);
    Tensor bias         = at::empty({n, 1, seqlen_q, seqlen_k}, query.options().dtype(at::kFloat)).fill_(0);
    Tensor descaleQ     = at::empty({1, 1, 1, 1}, query.options().dtype(at::kFloat)).fill_(1);
    Tensor descaleK     = at::empty_like(descaleQ);
    Tensor descaleV     = at::empty_like(descaleQ);
    Tensor descaleS     = at::empty_like(descaleQ);
    Tensor scaleS       = at::empty_like(descaleQ);
    Tensor scaleO       = at::empty_like(descaleQ);
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
                                            {n * head_q * seqlen_q, seqlen_q * dim_v, dim_v, 1},
                                            query.options())
                                 : at::empty_strided({n, head_q, seqlen_q, dim_v},
                                            {n * head_q * seqlen_q, dim_v, head_q * dim_v, 1},
                                            query.options());

    TensorDescriptor tquery(query);
    TensorDescriptor tkey(key);
    TensorDescriptor tvalue(value);
    TensorDescriptor tpadding_mask(padding_mask);
    TensorDescriptor tbias(bias);
    TensorDescriptor tdescaleQ(descaleQ);
    TensorDescriptor tdescaleK(descaleK);
    TensorDescriptor tdescaleV(descaleV);
    TensorDescriptor tdescaleS(descaleS);
    TensorDescriptor tscaleS(scaleS);
    TensorDescriptor tscaleO(scaleO);
    TensorDescriptor tdropProb(dropProb);
    TensorDescriptor tdropSeed(dropSeed);
    TensorDescriptor tdropOffset(dropOffset);
    TensorDescriptor tM(M);
    TensorDescriptor tZInv(ZInv);
    TensorDescriptor tLSE(LSE);
    TensorDescriptor tAmaxO(AmaxO);
    TensorDescriptor tAmaxS(AmaxS);
    TensorDescriptor tO(O);

    x.setTensor(miopenTensorMhaQ, tquery, query);
    x.setTensor(miopenTensorMhaK, tkey, key);
    x.setTensor(miopenTensorMhaV, tvalue, value);
    x.setTensor(miopenTensorMhaPaddingMask, tpadding_mask, padding_mask);
    x.setTensor(miopenTensorMhaBias, tbias, bias);
    x.setTensor(miopenTensorMhaDescaleQ, tdescaleQ, descaleQ);
    x.setTensor(miopenTensorMhaDescaleK, tdescaleK, descaleK);
    x.setTensor(miopenTensorMhaDescaleV, tdescaleV, descaleV);
    x.setTensor(miopenTensorMhaDescaleS, tdescaleS, descaleS);
    x.setTensor(miopenTensorMhaScaleS, tscaleS, scaleS);
    x.setTensor(miopenTensorMhaScaleO, tscaleO, scaleO);
    x.setTensor(miopenTensorMhaDropoutProbability, tdropProb, dropProb);
    x.setTensor(miopenTensorMhaDropoutSeed, tdropSeed, dropSeed);
    x.setTensor(miopenTensorMhaDropoutOffset, tdropOffset, dropOffset);
    x.setTensor(miopenTensorMhaO, tO, O);
    x.setTensor(miopenTensorMhaM, tM, M);
    x.setTensor(miopenTensorMhaZInv, tZInv, ZInv);
    x.setTensor(miopenTensorMhaLSE, tLSE, LSE);
    x.setTensor(miopenTensorMhaAmaxO, tAmaxO, AmaxO);
    x.setTensor(miopenTensorMhaAmaxS, tAmaxS, AmaxS);

    std::vector<miopenTensorArgument_t>& args        = x.getArgs();
    std::vector<miopenTensorDescriptor_t> descVector = x.getDescVector();
    for(size_t i = 0; i < descVector.size(); ++i)
    {
        args[i].descriptor = &descVector[i];
    }

    miopenHandle_t handle = getMiopenHandle_new();

    MIOPEN_CHECK(miopenFindSolutions(handle, problem, nullptr, solutions.data(), &found, solutions.size()));

    miopenSolution_t solution = solutions[0];

    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenGetSolutionWorkspaceSize(solution, &workspace_size));

    Tensor workspace_tensor;
    workspace_tensor = at::empty(workspace_size, query.options());

    MIOPEN_CHECK(
        miopenRunSolution(handle, solutions[0], args.size(), args.data(), workspace_tensor.data_ptr(), workspace_size));

    std::map<miopenTensorArgumentId_t, at::Tensor> tensors = x.getTensors();

    return tensors[miopenTensorMhaO];
};

} // namespace native

} // namespace at
