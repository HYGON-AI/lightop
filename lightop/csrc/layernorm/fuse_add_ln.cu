#include "layernorm.h"

namespace
{
constexpr int VEC = 4;
}

namespace at
{
namespace native
{

template <typename scalar_t, int VEC = 4, int blocksize = 512>
__global__ void add_layernorm_forward_kernel(scalar_t *input, scalar_t *residual, scalar_t *ret, scalar_t *ret_sum,
                                             float *mean, float *rstd, scalar_t *gamma, scalar_t *beta, int cols,
                                             float eps)
{
    int blockidx             = blockIdx.x;
    int threadidx            = threadIdx.x;
    int block_cols           = blockidx * cols;
    using T_ACC              = acc_type<scalar_t, true>;
    using WelfordType        = WelfordData<T_ACC, int, T_ACC>;
    using WelfordOp          = WelfordOps<T_ACC, T_ACC, int, T_ACC, thrust::pair<T_ACC, T_ACC>>;
    constexpr int share_size = blocksize / C10_WARP_SIZE;
    __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type val_shared[share_size];
    __shared__ T_ACC                                                                          s_mean;
    __shared__ T_ACC                                                                          s_rstd;
    WelfordType *val_shared_ptr = reinterpret_cast<WelfordType *>(val_shared);
    WelfordOp    welford_op;
    WelfordType  val;
    using VecTpye = memory::aligned_vector<scalar_t, VEC>;
    VecTpye *X    = (VecTpye *)(input + block_cols);
    VecTpye *R    = (VecTpye *)(residual + block_cols);
    VecTpye *Y    = (VecTpye *)(ret + block_cols);
    VecTpye *SUM  = (VecTpye *)(ret_sum + block_cols);
    scalar_t r_input[VEC];
    scalar_t r_residual[VEC];
    scalar_t r_output[VEC];
    int      tcol = cols / VEC;
    T_ACC    tmean;
    T_ACC    trstd;
    int      j = threadidx;
    if (j < tcol)
    {
        *(VecTpye *)r_input    = *(X + j);
        *(VecTpye *)r_residual = *(R + j);
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            r_input[i] = r_input[i] + r_residual[i];
            val        = welford_op.reduce(val, static_cast<T_ACC>(r_input[i]));
        }
        *(SUM + j) = *(VecTpye *)r_input;
    }

    val = BlockReduce(val, welford_op, val_shared_ptr);
    if (threadidx == 0)
    {
        // layernorm get mean and rstd
        thrust::tie(tmean, s_mean) = welford_op.project(val);
        s_rstd                     = c10::cuda::compat::rsqrt(tmean + eps);
    }
    __syncthreads();
    // s_mean,s_rstd为共享内存数据，块共享
    // 结果为块内所有数据的均值和标准差的倒数
    // 每个块为cols的数据，及特征维度的均值和标准差的倒数
    tmean = s_mean;
    trstd = s_rstd;

    if (j < tcol)
    {
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            int jj = j * VEC + i;
            r_output[i] =
                (static_cast<T_ACC>(r_input[i]) - tmean) * trstd * static_cast<T_ACC>(gamma[jj]) + static_cast<T_ACC>(beta[jj]);
        }
        *(Y + j) = *(VecTpye *)r_output;
    }
    if (threadIdx.x == 0 && rstd != NULL)
    {
        mean[blockidx] = tmean;
        rstd[blockidx] = trstd;
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> fused_add_ln_forward(const Tensor &self, const Tensor &other, const Tensor &weight,
                                                                const Tensor &bias, double ln_epsilon, bool training)
{
    AT_ASSERTM(self.sizes() == other.sizes(), "x must be the same size as y");
    AT_ASSERTM(self.is_cuda() && other.is_cuda(), "x and y must on HCU");
    // 特征维度
    const int cols = self.sizes().back();
    AT_ASSERT(cols >= 64, "cols must >= 64");
    AT_ASSERT(cols % 4 == 0, "cols%4 must be 0");

    const int nelem   = self.numel();
    const int rows    = nelem / cols;
    Tensor    ret     = at::empty_like(self);
    Tensor    ret_sum = at::empty_like(self);
    Tensor    mean;
    Tensor    rstd;
    if (training)
    {
        mean = at::empty({rows}, self.options().dtype(at::ScalarType::Float));
        rstd = at::empty({rows}, self.options().dtype(at::ScalarType::Float));
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "add_layernorm_forward", [&] {
            using T_ACC             = acc_type<scalar_t, true>;
            scalar_t *self_data     = self.data_ptr<scalar_t>();
            scalar_t *residual_data = other.data_ptr<scalar_t>();
            scalar_t *ret_data      = ret.data_ptr<scalar_t>();
            scalar_t *ret_sum_data  = ret_sum.data_ptr<scalar_t>();
            scalar_t *weight_data   = weight.data_ptr<scalar_t>();
            scalar_t *bias_data     = bias.data_ptr<scalar_t>();
            float  eps           = ln_epsilon;
            float *mean_data = NULL, *rstd_data = NULL;
            if (training)
            {
                mean_data = mean.data_ptr<float>();
                rstd_data = rstd.data_ptr<float>();
            }
            if (cols <= 1024)
            {
                add_layernorm_forward_kernel<scalar_t, 8, 128><<<rows, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
            }
            else if (cols <= 2048)
            {
                add_layernorm_forward_kernel<scalar_t, 8, 256><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
            }
            else if (cols <= 4096)
            {
                if (rows > 1200)
                {
                    add_layernorm_forward_kernel<scalar_t, 8, 512><<<rows, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                        self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols,
                        eps);
                }
                else
                {
                    add_layernorm_forward_kernel<scalar_t, 4, 1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                        self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols,
                        eps);
                }
            }
            else if (cols <= 8192)
            {
                add_layernorm_forward_kernel<scalar_t, 8, 1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
            }
            else
            {
                add_layernorm_forward_kernel<scalar_t, 16, 1024><<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_data, residual_data, ret_data, ret_sum_data, mean_data, rstd_data, weight_data, bias_data, cols, eps);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, ret_sum, mean, rstd);
}

template <typename scalar_t, int VEC = 4> __global__ void fused_add_ln_backward(int N, const scalar_t *dY, const scalar_t *X,
                                                                                const scalar_t *gamma, const float *mean,
                                                                                const float *rstd, scalar_t *dX)
{
    LAYERNORMBACKWARD_VEC_2048
    if (j < tcol)
    {
#pragma unroll
        for (int i = 0; i < VEC; i++)
        {
            pdx[i] = trstd * static_cast<T_ACC>(pdy[i]) * static_cast<T_ACC>(pgamma[i]) + b * static_cast<T_ACC>(px[i]) + c;
        }
        *(VecTpye *)(dX + index) = tdx;
    }
}

std::tuple<Tensor, Tensor, Tensor, Tensor> fused_add_ln_backward(const Tensor &grad, const Tensor &X, const Tensor &mean,
                                                                 const Tensor &rstd, const Tensor &gamma)
{
    int N = grad.sizes().back();
    AT_ASSERT(N >= 64, "N must >= 64");
    AT_ASSERT(N % 4 == 0, "N%4 must be 0");
    int    nelem       = grad.numel();
    int    M           = grad.numel() / N;
    auto   pgrad       = grad.expect_contiguous();
    Tensor dX          = at::empty_like(*pgrad);
    Tensor dgamma      = at::empty_like(gamma);
    Tensor dbeta       = at::empty_like(gamma);
    auto   cuda_stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, grad.scalar_type(), "layernorm_backward", [&] {
            scalar_t *dY_data     = pgrad->data_ptr<scalar_t>();
            scalar_t *X_data      = X.data_ptr<scalar_t>();
            float *mean_data   = mean.data_ptr<float>();
            float *rstd_data   = rstd.data_ptr<float>();
            scalar_t *gamma_data  = gamma.data_ptr<scalar_t>();
            scalar_t *dgamma_data = dgamma.data_ptr<scalar_t>();
            scalar_t *dbeta_data  = dbeta.data_ptr<scalar_t>();
            scalar_t *dX_data     = dX.data_ptr<scalar_t>();
            GammaBeta_Backward(nelem, N, dY_data, X_data, mean_data, rstd_data, dgamma_data, dbeta_data);
            if (N <= 1024)
            {
                fused_add_ln_backward<scalar_t, 8>
                    <<<M, 128, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
            }
            else if (N <= 2048)
            {
                fused_add_ln_backward<scalar_t, 8>
                    <<<M, 256, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
            }
            else if (N <= 4096)
            {
                if (M > 1200)
                {
                    fused_add_ln_backward<scalar_t, 8>
                        <<<M, 512, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
                }
                else
                {
                    fused_add_ln_backward<scalar_t, 4>
                        <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
                }
            }
            else if (N <= 8192)
            {
                fused_add_ln_backward<scalar_t, 8>
                    <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
            }
            else
            {
                fused_add_ln_backward<scalar_t, 16>
                    <<<M, 1024, 0, cuda_stream>>>(N, dY_data, X_data, gamma_data, mean_data, rstd_data, dX_data);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(dX, dX, dgamma, dbeta);
}

class fuseAddLnFunction : public torch::autograd::Function<fuseAddLnFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, const torch::autograd::Variable &x,
                                                  const torch::autograd::Variable &y, const torch::autograd::Variable &weight,
                                                  const torch::autograd::Variable &bias, double eps, bool training)
    {
        auto px      = x.expect_contiguous();
        auto py      = y.expect_contiguous();
        auto pweight = weight.expect_contiguous();
        auto pbias   = bias.expect_contiguous();
        auto ret     = fused_add_ln_forward(*px, *py, *pweight, *pbias, eps, training);
        ctx->save_for_backward({std::get<1>(ret), std::get<2>(ret), std::get<3>(ret), *pweight});
        return {std::get<0>(ret)};
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext     *ctx,
                                                   const torch::autograd::variable_list &gradOutput)
    {
        auto saved = ctx->get_saved_variables();
        auto X     = saved[0];
        auto mean  = saved[1];
        auto rstd  = saved[2];
        auto gamma = saved[3];
        auto ret   = fused_add_ln_backward(gradOutput[0], X, mean, rstd, gamma);
        return {std::get<0>(ret),
                std::get<1>(ret),
                std::get<2>(ret),
                std::get<3>(ret),
                torch::autograd::Variable(),
                torch::autograd::Variable(),
                torch::autograd::Variable()};
    }
};

Tensor fuse_add_ln_forward_autograd(const Tensor &x, const Tensor &y, const Tensor &weight, const Tensor &bias, double eps,
                                    bool training)
{
    auto result = fuseAddLnFunction::apply(x, y, weight, bias, eps, training);
    return result[0];
}

}  // namespace native
}  // namespace at