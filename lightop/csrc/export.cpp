#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <vector>

#include "paged_mqa_logits/paged_mqa_logits.h"
#include "sampling/sampling.h"

enum GRAD_TYPE {nograd, wgrad, xgrad};
namespace at {  namespace native {
Tensor rmsnorm_forward_autograd(const Tensor& x,const Tensor& weight,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_forward(const Tensor& self,const Tensor& weight,c10::optional<Tensor>& output_inplace,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma);
void fused_rms_norm_contiguous(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon);
void fuse_rmsnorm_rope_quant_qkv(torch::Tensor& positions, torch::Tensor& q_c, torch::Tensor& q_pe, torch::Tensor& q_quant, torch::Tensor& q_scale, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon);
void fuse_fill_rms_x2_concat(torch::Tensor& hidden_states_fuse, torch::Tensor& positions, torch::Tensor& inputs_embeds, torch::Tensor& previous_hidden_states, torch::Tensor& weight_inputs_embeds, torch::Tensor& weight_previous_hidden_states, float epsilon);
//void rms_rotary_embedding_fuse_with_kv_store(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, torch::Tensor& v, torch::Tensor& cos_sin_cache, int head_dim, int page_size, torch::Tensor& k_buffer, torch::Tensor& v_buffer, Tensor& kv_cache_loc, bool interleave, Tensor& weight_q, Tensor& weight_k, torch::Dtype output_dtype, std::optional<Tensor>& residual_q, std::optional<Tensor>& residual_k, std::optional<float>& k_scale, std::optional<float>& v_scale, std::optional<Tensor>& q_rope_out, std::optional<Tensor>& k_rope_out, std::optional<Tensor>& v_rope_out,  float epsilon);
//void rms_rotary_embedding_fuse_with_kv_store(torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key, torch::Tensor& v, int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& weight_q, torch::Tensor& weight_k, std::optional<Tensor>& residual_q, std::optional<Tensor>& residual_k, torch::Tensor& k_buffer, torch::Tensor& v_buffer, torch::Tensor& kv_cache_loc, torch::Tensor& q_rope_out, torch::Tensor& k_rope_out, torch::Tensor& v_rope_out, int page_size, double epsilon, bool use_fp8_e4m3);
void rms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions, 
    Tensor& query, 
    Tensor& key,
    //////////////
    Tensor& v,
    //////////////
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    Tensor& weight_q, 
    Tensor& weight_k, 
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    ///////////////////////////////////////////////////
    Tensor& k_buffer,
    Tensor& v_buffer,
    Tensor& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    /////////////////////////////////////////////////
    double epsilon,
    bool use_fp8_e4m3);
Tensor fuse_rn_dropout_forward_autograd(const Tensor& x,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor &bias, const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,bool training,bool inplace);
Tensor drop_path_forward_autograd(const Tensor& x,double p, bool training,bool scale_by_keep) ;
Tensor fuse_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,double rate,bool training);
Tensor fuse_layernorm_forward_autograd(const Tensor& x,const std::optional<Tensor>& weight,const std::optional<Tensor>& bias,double eps,bool training);
Tensor fuse_add_ln_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,const Tensor& bias,double eps,bool training);
Tensor fuse_ln_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,const Tensor& bias,double eps,double rate,bool training);
Tensor fuse_ln_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& attention_bias,const Tensor& weight,const Tensor& bias,double eps,double rate,bool training);
Tensor fuse_ln_dropout_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps,double rate,bool training);
Tensor fuse_layernorm_droppath_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,double eps, double keep_prob, bool scale_by_keep, bool training);
Tensor fused_linear_bias_forward_autograd(const Tensor& input, const Tensor& weight, const Tensor& bias);
std::tuple <Tensor, Tensor> fused_relu_dropout_forward(const Tensor &self, const double &p, const bool &use_relu, const bool &training);
Tensor fused_relu_dropout_backward(Tensor &grad, Tensor &mask);
Tensor fuse_softmax_forward_autograd(const Tensor &input, const int64_t dim, const bool half_to_float, const bool training);
Tensor fuse_logsoftmax_forward_autograd(const at::Tensor &input, const int64_t dim, const bool half_to_float, const bool training);
Tensor accelerate_rnnt_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp, bool compute_sum, bool compute_mean, bool check_lengths);
Tensor accelerate_ctc_loss(Tensor &logits, const Tensor &targets, const Tensor &logit_lengths, const Tensor &target_lengths, int blank, double clamp, bool logits_time_major, bool compute_sum, bool compute_mean, bool check_lengths);
Tensor fused_bias_gelu_forward(const Tensor &x, const Tensor &bias);
std::tuple <Tensor, Tensor> fused_bias_gelu_backward(const Tensor  &x, const Tensor &bias,const Tensor &grad);
Tensor fuse_scaled_upper_triang_masked_softmax_forward_autograd(torch::Tensor const& input, float scale_factor);
Tensor fuse_scaled_masked_softmax_forward_autograd(torch::Tensor const& input, torch::Tensor const& mask, float scale_factor);
int get_batch_per_block(int query_seq_len, int key_seq_len, int batches, int attn_heads);
Tensor fuse_scaled_upper_triang_masked_softmax_dropout_forward_autograd(torch::Tensor const& input, float scale_factor, float p);
Tensor fuse_scaled_masked_softmax_dropout_forward_autograd(torch::Tensor const& input, torch::Tensor const& mask, float scale_factor, double p);
Tensor RoPE_forward_gpu(int sq, int b, int np, int hn, Tensor tensor_q,Tensor tensor_cos, Tensor tensor_sin);
Tensor RoPE_backward_gpu(int sq, int b, int np, int hn,Tensor tensor_grad_out, Tensor tensor_cos, Tensor tensor_sin);
Tensor FusedBiasSwiGLU_forward_gpu(Tensor in, Tensor bias);
std::vector<Tensor> FusedBiasSwiGLU_backward_gpu(Tensor out_grad,Tensor in,Tensor bias);
std::tuple <Tensor, Tensor> fuse_bias_dropout_add_forward(const Tensor &x,const Tensor &bias,const Tensor &residual,double droprate,bool training);
std::tuple <Tensor, Tensor> fuse_bias_dropout_add_backward(const Tensor &grad,const Tensor &mask, double droprate);
Tensor fuse_groupnorm_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,bool training);
Tensor fuse_celu_gn_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,double alpha,bool training);
Tensor fuse_gn_silu_forward_autograd(const Tensor& x,const Tensor& weight,const Tensor& bias,int num_groups,int num_channels,double eps,bool training);
Tensor fused_bmm(torch::Tensor x, torch::Tensor w, torch::Tensor output, torch::Tensor x_scale, torch::Tensor w_scale, std::optional<torch::Tensor> bias, torch::Dtype dtype, bool transpose_bm, bool transpose_bm_in, int mode);
Tensor sum_opt(const Tensor& x,int dim);
Tensor mean_opt(const Tensor& x,int dim);
std::tuple<Tensor,Tensor> topk_opt(const Tensor& x,int k,int dim,bool sorted);
Tensor gemm_opt(const Tensor& x,const Tensor& y);
Tensor gemm_w4_a16(const Tensor& x,const Tensor& y,const Tensor& scale_m,const Tensor& zero_m);
void gemm_awq_w4a16(Tensor& ret,const Tensor& x,const Tensor& y,const Tensor& scales_zeros,int num_n_warps, int num_k_warps, int n_per_warp,int k_per_warp,int splitk,bool use_marlin);
void gemm_awq_w4a16_dequant(Tensor& ret,const Tensor& weight,const Tensor& scales_zeros,bool use_marlin);
Tensor w4_to_w16(const Tensor& x,const Tensor& scale_m,const Tensor& zero_m);
Tensor gemm_w8_a16_fp8_colmajor(const Tensor& x,const Tensor& y,const Tensor& scale,int warps,int n_per_block,int splitk);
void fp8to16(Tensor &ret, const Tensor &x,const Tensor& scale);
Tensor moe_gemm_w8a8(torch::Tensor input,Tensor b_qweight,Tensor output,Tensor a_scale,Tensor b_scale,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8(torch::Tensor input,Tensor b_qweight,Tensor output,Tensor a_scale,Tensor b_scale,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8_fp8(torch::Tensor input,Tensor b_qweight,Tensor output,Tensor a_scale,Tensor b_scale,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_marlin_w8a8_blockwise_fp8(torch::Tensor input,Tensor b_qweight,Tensor output,Tensor a_scale,Tensor b_scale,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_w4a8(torch::Tensor input,Tensor b_qweight,Tensor output,Tensor a_scale,Tensor b_scale,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor moe_gemm_w16a16(torch::Tensor input,Tensor b_qweight,Tensor output,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int64_t top_k, int mode, int delta);
Tensor gemm_w8_a8_smooth_colmajor(const Tensor& x,const Tensor& y,const Tensor& scale_a,const Tensor& scale_b,  const std::optional<Tensor> bias, at::ScalarType out_dtype,int warps,int n_per_block, int n_per_warp, int splitk, bool use_n_warps);

Tensor gemm_w8a8_smooth_asm(Tensor &A, Tensor &B, Tensor &As, Tensor &Bs, const std::optional<Tensor> bias, torch::Dtype output_dtype, int M, int N, int K, bool isTunning_slide_block, int tunning_slide_block);
Tensor gemm_w8a8_asm(Tensor &A, Tensor &B, Tensor &As, Tensor &Bs, c10::optional<Tensor> &C, c10::optional<Tensor> &D_, c10::optional<Tensor> bias, torch::Dtype output_dtype, int M, int N, int K, int gradType = 0);

std::vector<torch::Tensor> grougemm_w8a8_pertensor_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, std::vector<torch::Tensor>& As_list, std::vector<torch::Tensor>& Bs_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_nt_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_tn_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
std::vector<torch::Tensor> grougemm_bf16_fp16_nn_asm(std::vector<torch::Tensor>& A_list, std::vector<torch::Tensor>& B_list, std::vector<torch::Tensor>& D_list, 
                                                      std::vector<float>& alphas, c10::optional<std::vector<torch::Tensor>>& C_list, c10::optional<std::vector<float>>& betas, torch::Dtype output_dtype);
Tensor m_grouped_w8a8_gemm_nt_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, int mode);
Tensor m_grouped_w16a16_gemm_nt_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, int mode);
Tensor m_grouped_w8a8_gemm_nt_nopad_contig_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor m_indices, Tensor token_per_expert);

Tensor moe_w8a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_marlin_w8a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_marlin_w8a8_fp8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode, int delta);
Tensor moe_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode);
Tensor moe_w4a8_marlin_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, std::optional<Tensor> topk_weights, Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_pad, uint32_t top_k, int mode);
Tensor moe_groupgemm_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor problem_size, Tensor expert_offsets, int mode);
Tensor moe_groupgemm_marlin_w4a8_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor problem_size, Tensor expert_offsets, int mode);
Tensor m_grouped_marlin_w4a8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_marlin_w8a8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_marlin_fp8_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor a_scale, Tensor b_scale, Tensor masked_m, int expected_m_per_group, int mode);
Tensor m_grouped_bf16_gemm_nt_masked_asm(Tensor input, Tensor b_qweight, Tensor output, Tensor masked_m, int expected_m_per_group, int mode);
Tensor moe_w16a16_asm(Tensor input,Tensor b_qweight,Tensor output,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int top_k, int mode, int delta);
Tensor moe_marlin_w16a16_asm(Tensor input,Tensor b_qweight,Tensor output,std::optional<Tensor> topk_weights,Tensor sorted_token_ids,Tensor expert_ids,Tensor num_tokens_post_pad, int top_k, int mode, int delta);
Tensor moe_marlin_w4a16(torch::Tensor& a,
                        std::optional<torch::Tensor> const& c_or_none,
                        torch::Tensor& b_q_weight,
                        torch::Tensor& b_scale_zeros,
                        std::optional<torch::Tensor> const& g_idx_or_none,
                        std::optional<torch::Tensor> const& perm_or_none,
                        torch::Tensor& workspace,
                        torch::Tensor& sorted_token_ids,
                        torch::Tensor& expert_ids,
                        torch::Tensor& num_tokens_past_padded,
                        torch::Tensor& topk_weights,
                        int64_t moe_block_size,
                        int64_t top_k,
                        bool mul_topk_weights,
                        bool is_ep,
                        int64_t size_m,
                        int64_t size_n,
                        int64_t size_k,
                        bool is_k_full,
                        bool use_atomic_add,
                        bool use_fp32_reduce,
                        bool is_zp_float,
                        int64_t thread_n=-1, 
                        int64_t thread_k=-1,
                        int64_t blocks_per_cu=1
                        );
Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k, int64_t size_n, int64_t num_bits);
Tensor moe_marlin_w4a8(
                        torch::Tensor& a_q,
                        torch::Tensor& a_scales,
                        torch::Tensor& c,
                        torch::Tensor& b_q_weight,
                        torch::Tensor& b_scales,
                        torch::Tensor& workspace,
                        torch::Tensor& global_reduce_buffer,  // 用于全局规约的缓冲区
                        torch::Tensor& sorted_token_ids,
                        torch::Tensor& expert_ids,
                        torch::Tensor& num_tokens_past_padded,
                        std::optional<torch::Tensor> topk_weights,
                        int64_t moe_block_size,
                        int64_t top_k,
                        bool is_ep,
                        int64_t size_m,
                        int64_t size_n,
                        int64_t size_k,
                        int64_t thread_n,
                        int64_t thread_k,
                        int64_t blocks_per_cu,
                        bool is_reduce
                        );
Tensor moe_marlin_w4fp8(
                        torch::Tensor& a_q, 
                        torch::Tensor& a_scales, 
                        torch::Tensor& c,
                        torch::Tensor& b_q_weight, 
                        torch::Tensor& b_scales,
                        torch::Tensor& workspace, 
                        torch::Tensor& global_reduce_buffer,  // 用于全局规约的缓冲区
                        torch::Tensor& sorted_token_ids, 
                        torch::Tensor& expert_ids,
                        torch::Tensor& num_tokens_past_padded, 
                        std::optional<torch::Tensor> topk_weights,
                        int64_t moe_block_size, 
                        int64_t top_k, 
                        bool is_ep,
                        int64_t size_m, 
                        int64_t size_n,
                        int64_t size_k,
                        int64_t thread_n,
                        int64_t thread_k,
                        int64_t blocks_per_cu,
                        bool is_reduce
                        );
Tensor awq_marlin_repack_w4a8(torch::Tensor& b_q_weight, torch::Tensor& marlin_weight, int64_t size_batch, int64_t size_k, int64_t size_n);
Tensor ds_concat(torch::Tensor input_A, torch::Tensor input_B, torch::Tensor output, int32_t mode);
void blockwise_scaled_cast_transpose(const Tensor& input,std::optional<Tensor> output,std::optional<Tensor> scale_inv,
                                    std::optional<Tensor> output_t,std::optional<Tensor> scale_inv_t,std::optional<Tensor> noop,
                                    const int block_len,const int block_scaling_dim,const bool rowwise,const bool columnwise,
                                    const float epsilon,const bool pow2_scale,const bool all_gather_usage, at::ScalarType out_dtype);
#ifdef USE_MIOPEN
Tensor miopen_fused_conv_bias_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations);
Tensor miopen_fused_conv_bias_add_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const Tensor &add, const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations);
Tensor miopen_fused_conv_bias_add_relu_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const Tensor &add, const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations);
Tensor miopen_fused_conv_bias_relu_forward(const Tensor &x, const Tensor &weight, const Tensor &bias, const std::vector<int64_t> &pads, const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations);
std::vector<Tensor> miopen_add_rms_norm_dynamic_quant(const Tensor &x, const Tensor &x2, const Tensor &weight, const Tensor &smoothScale, const float epsilon);
std::vector<Tensor> miopen_rms_norm_dynamic_quant_without_residual(const Tensor &x, const Tensor &weight, const Tensor &smoothScale, const float epsilon);
Tensor miopen_head_rms_norm(const Tensor &x, const Tensor &weight, const float epsilon, const int norm_head_start, const int norm_head_end, const int norm_head_num);
std::vector<Tensor> miopen_groupnorm(const Tensor &x, const Tensor &weight, const Tensor &bias, const int num_groups, const float epsilon, const int mode);
// std::vector<Tensor> miopen_groupnorm_backward(const Tensor &x, const Tensor &dy, const Tensor &weight, const Tensor &mean, const Tensor &rstd, const int num_groups, const float epsilon, const int mode);
std::vector<Tensor> miopen_add_groupnorm(const Tensor &x1, const Tensor &x2, const Tensor &weight, const Tensor &bias, const int num_groups, const float epsilon, const int mode);
std::vector<Tensor> miopen_layernorm(const Tensor &x, const Tensor &weight, const Tensor &bias, const float epsilon, const int normalized_dim, const int mode);
std::vector<Tensor> miopen_add_layernorm(const Tensor &x1, const Tensor &x2, const Tensor &weight, const Tensor &bias, const float epsilon, const int normalized_dim, const int mode);
std::vector<Tensor> miopen_skip_layernorm(const Tensor &x1, const Tensor &x2, const Tensor &weight, const Tensor &bias, const float epsilon, const int normalized_dim, const int mode);
std::vector<Tensor> miopen_moe_scatter_dynamic_quant(const Tensor &hidden, const Tensor &selected_experts, const Tensor &moe_weights, const Tensor &smooth, Tensor& out_scatter_tokens, Tensor& out_scatter_per_token_scale, Tensor& out_scatter_tokens_offset, Tensor& out_experts_token_count, Tensor& out_experts_token_start, const int expert_start_idx, const int expert_end_idx, const int num_shared_experts, const int num_tokens_offset, const int num_tokens_per_rank, const int num_tokens);
std::vector<Tensor> miopen_moe_swiglu_dynamic_quant(const Tensor &scatter_tokens, const Tensor &smooth_scale, const Tensor &experts_tokens_count, const Tensor &exports_tokens_start, const float beta);
Tensor miopen_moe_gather(const Tensor &scatter_tokens, const Tensor &token_offset, const int real_scatter_tokens, const int num_tokens, const int hidden_size, const int topk, const int num_shared_experts);
Tensor miopen_scaled_dot_product_attention(const Tensor &query, const Tensor &key, const Tensor &value, const std::optional<Tensor> &attn_mask_, const double droprate, const bool is_causal, std::optional<double> scale, const bool enable_gqa, const bool is_training);
Tensor miopen_gemm_bias(const Tensor &input_a, const Tensor &input_b, const Tensor &bias);
Tensor miopen_transform_tensor(const Tensor &input, const int transpose_mode, const float alpha, const float beta);
void miopen_fused_adam(Tensor& params, const Tensor& grads, Tensor& exp_avgs, Tensor& exp_avg_sqs, c10::optional<Tensor>& max_exp_avg_sqs, Tensor& state_steps, const float lr, const float beta1, const float beta2, const float weight_decay, const float eps, const bool amsgrad, const bool maximize, const c10::optional<Tensor>& grad_scale, const c10::optional<Tensor>& found_inf);
#endif

void rms_norm_dynamic_per_token_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input);
void rms_norm_fp8_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input, int fp8type);
void create_gemmbias_handle();
void destroy_gemmbias_handle();
void create_gemmdbias_handle();
void destroy_gemmdbias_handle();
void store_kv_cache(torch::Tensor packed_qkv,torch::Tensor& K_cache,torch::Tensor& V_cache,torch::Tensor q_lens,torch::Tensor accum_q_lens,
                              torch::Tensor cache_lens,torch::Tensor cache_slot_ids,torch::Tensor k_scale,torch::Tensor v_scale,const std::string& kv_cache_dtype_str,int q_head_num,int kv_head_num);
void store_kv_cache_paged(const torch::Tensor& key, const torch::Tensor& value,torch::Tensor& K_cache, torch::Tensor& V_cache,const torch::Tensor& q_lens, const torch::Tensor& accum_q_lens,
                          const torch::Tensor& cache_lens, const torch::Tensor& block_table,const torch::Tensor& k_scale, const torch::Tensor& v_scale,const std::string& kv_cache_dtype_str);

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,torch::optional<torch::Tensor>& token_expert_indices,torch::Tensor& gating_output,  bool Is_renormalize = false);
void rotary_embedding_fuse(torch::Tensor& positions,  torch::Tensor& query, std::optional<torch::Tensor> key,int64_t head_size,torch::Tensor& cos_sin_cache, bool is_neox);    
void rotary_embedding_deepseek_fuse(torch::Tensor& positions,  torch::Tensor& query, std::optional<torch::Tensor> key,int64_t head_size,torch::Tensor& cos_sin_cache, bool is_neox, std::optional<torch::Tensor> offsets);
void moe_align_block_size(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor num_tokens_post_pad,
    const torch::optional<torch::Tensor> &expert_map =  torch::nullopt,
    const torch::optional<torch::Tensor> &expert_mask =  torch::nullopt, // bool
    const torch::optional<torch::Tensor> &num_local_tokens =  torch::nullopt,bool Is_EP = false,bool Is_fuse_fill = true); 

//------------------------------------------------- groupgemm ---------------------------------------//
Tensor groupgemm(Tensor &a, Tensor &b, Tensor &batch_sizes, bool trans_b);
// std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> groupgemm_permute(const Tensor& tokens, const Tensor& routing_map);
// Tensor groupgemm_unpermute(const Tensor& permuted_tokens, const Tensor& sorted_indices, const std::vector<int64_t>& restore_shape, std::optional<torch::Tensor> probs, std::optional<torch::Tensor> routing_map, const Tensor& expert_m_count);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> groupgemm_permute(const torch::Tensor& tokens, const torch::Tensor& routing_map);
torch::Tensor groupgemm_unpermute(const torch::Tensor& permuted_tokens, const torch::Tensor& sorted_indices, const std::vector<int64_t>& restore_shape, std::optional<torch::Tensor> probs, std::optional<torch::Tensor> routing_map, const torch::Tensor& expert_m_count, bool use_fp32);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fuse_groupgemm_permute_quant(const torch::Tensor& tokens, const torch::Tensor& routing_map, torch::Tensor& scales);
//------------------------------------------------- groupgemm ---------------------------------------//

std::vector<Tensor> moe_fused_gate(torch::Tensor& input, torch::Tensor& bias, int64_t num_expert_group, int64_t topk_group, int64_t topk, int64_t n_share_experts_fusion, double routed_scaling_factor);

void fuse_silu_mul_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids) ;
void fuse_silu_mul_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m);
void fuse_silu_mul_fp8_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids) ;
void fuse_silu_mul_fp8_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m);
void fuse_silu_and_mul(torch::Tensor& input, torch::Tensor& out);
void moe_sum(torch::Tensor &input,
             torch::Tensor &output, 
             const torch::optional<torch::Tensor> &bias = torch::nullopt ,
             const torch::optional<torch::Tensor> &expert_mask = torch::nullopt ,
             const torch::optional<torch::Tensor> &num_local_tokens = torch::nullopt ,
             const float factor = 1.f,const int expect_m = -1);    
Tensor ms_deform_attn_cuda_forward(const Tensor &value, const Tensor &spatial_shapes, const Tensor &level_start_index, const Tensor &sampling_loc, const Tensor &attn_weight, const int im2col_step);
void ms_deform_attn_cuda_backward(const Tensor &value, const Tensor &spatial_shapes, const Tensor &level_start_index, const Tensor &sampling_loc, const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value, Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);
void mha_norm_rope_qkv_inplace_cuda(at::Tensor qkv, c10::optional<at::Tensor> gamma_q, c10::optional<at::Tensor> beta_q, c10::optional<at::Tensor> gamma_k,c10::optional<at::Tensor> beta_k,at::Tensor cos, at::Tensor sin, double eps);
void cross_norm_rope_q_kv_inplace_cuda(at::Tensor q,at::Tensor kv,  c10::optional<at::Tensor> gamma_q, c10::optional<at::Tensor> beta_q,c10::optional<at::Tensor> gamma_k,c10::optional<at::Tensor> beta_k, at::Tensor cos,  at::Tensor sin,  double eps);
torch::Tensor box_cost(const torch::Tensor &pred, const torch::Tensor &target, const torch::Tensor &inst_weights, const torch::Tensor &dim_weights, const float cost_weight); 
torch::Tensor encode_reg_t(const torch::Tensor &input, const torch::Tensor &lengths); 
Tensor mqa_logits(torch::Tensor &Q, torch::Tensor &K, torch::Tensor &Weights, torch::Tensor &KS, torch::Tensor &KE, int seq_len, int kv_seq_len, int num_heads, int head_dim,bool clean_logits = true);
std::vector<torch::Tensor> batch_linear_assignment(torch::Tensor cost);
void fill_zeros(Tensor &out, int bs, int hidden_size);
void rms_rotary_embedding_fuse(Tensor &positions,Tensor &query,Tensor &key,int64_t head_size,Tensor &cos_sin_cache, bool is_neox,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);
void reshape_and_cache_cuda(torch::Tensor& key,torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping, const std::string& kv_cache_dtype, torch::Tensor& k_scale, torch::Tensor& v_scale);
void ep_fused_quant_scatter(torch::Tensor input,torch::Tensor topk_ids,torch::optional<torch::Tensor> expert_map_opt,torch::Tensor expert_num_tokens,torch::Tensor aq_out,torch::Tensor aq_scale_out,torch::Tensor m_indices,torch::Tensor inv_perm, int local_num_experts,int alignment);
void ep_scatter(torch::Tensor aq,torch::Tensor aq_scale,torch::Tensor topk_ids,torch::optional<torch::Tensor> expert_map_opt, torch::Tensor expert_num_tokens,torch::Tensor aq_out,torch::Tensor aq_scale_out,torch::Tensor m_indices,torch::Tensor inv_perm,int local_num_experts,int alignment);

void destindex_copy_quantize_int8_cuda(const torch::Tensor& input, torch::Tensor& scale_zero, torch::Tensor& output);
void destindex_dequantize_int8_cuda(const torch::Tensor& input, const torch::Tensor& scale_zero, torch::Tensor& output);
std::vector<at::Tensor> kimi_k2_moe_fused_gate(at::Tensor &input, at::Tensor &bias, int64_t topk,bool renormalize, double routed_scaling_factor,bool apply_routed_scaling_factor_on_output); 
void fuse_rms_mrope_cuda(Tensor &q, Tensor &k, const Tensor &cos, const Tensor &sin, const std::vector<int>& mrope_section, int head_size, int rotary_dim,bool is_interleaved,Tensor weight_q,Tensor weight_k,c10::optional<Tensor> residual_q,c10::optional<Tensor> residual_k,double epsilon);
void ep_gather(torch::Tensor& a,torch::Tensor& topk_ids,torch::Tensor& topk_weights,torch::Tensor& inv_perm,torch::optional<torch::Tensor> expert_map,torch::Tensor& output);
torch::Tensor backward_polyfill_cuda(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,torch::Tensor grad_output,torch::Tensor grad_vertices,int width,int height,float inv_smoothness,int mode);
std::vector<torch::Tensor> forward_polyfill_cuda(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,int width,int height,float inv_smoothness,int mode);
torch::Tensor backward_polyfill_lightop(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,torch::Tensor grad_output,torch::Tensor grad_vertices,int width,int height,float inv_smoothness,int mode);
std::vector<torch::Tensor> forward_polyfill_lightop(torch::Tensor vertices,torch::Tensor output,torch::Tensor nearest_edge_map,int width,int height,float inv_smoothness,int mode);
void ep_fused_smooth_quant_scatter(
    torch::Tensor input, 
    torch::Tensor topk_ids,
    torch::optional<torch::Tensor> expert_map_opt,
    torch::Tensor expert_offsets,
    torch::Tensor smooth_scale,
    torch::Tensor aq_out, 
    torch::Tensor aq_scale_out,
    torch::Tensor m_indices, 
    torch::Tensor inv_perm,
    int local_num_experts, int alignment); 
void per_token_dynamic_quant_int8(torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales, torch::Tensor const& smoothscale);
void rms_norm_opt(torch::Tensor& out,torch::Tensor& input,torch::Tensor& weight,double epsilon);
void fused_add_rms_norm_opt(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& weight,double epsilon); 
void silu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_tanh_and_mul_opt(torch::Tensor& out,torch::Tensor& input); 
}
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm_forward_autograd", &at::native::fuse_layernorm_forward_autograd, "layernorm forward");
  m.def("add_ln_forward_autograd", &at::native::fuse_add_ln_forward_autograd, "layernorm forward");
  m.def("ln_dropout_forward_autograd", &at::native::fuse_ln_dropout_forward_autograd, "fused_ln_dropout forward");
  m.def("ln_add_dropout_forward_autograd", &at::native::fuse_ln_add_dropout_forward_autograd, "fused_ln_add_dropout forward");
  m.def("ln_add_dropout_bias_forward_autograd", &at::native::fuse_ln_add_dropout_bias_forward_autograd, "fused_ln_add_dropout_bias forward");
  m.def("layernorm_droppath_forward_autograd", &at::native::fuse_layernorm_droppath_forward_autograd, "layernorm then droppath forward");
  m.def("rmsnorm_forward_autograd", &at::native::rmsnorm_forward_autograd, "rmsnorm forward autograd");
  m.def("rmsnorm_forward", &at::native::rmsnorm_forward, "rmsnorm forward");
  m.def("rmsnorm_backward", &at::native::rmsnorm_backward, "rmsnorm backward");
  m.def("fused_rms_norm_contiguous", &at::native::fused_rms_norm_contiguous, "fused_rms_norm_contiguous");
  m.def("fused_rms_norm_rope_contiguous", &at::native::fused_rms_norm_rope_contiguous, "fused_rms_norm_rope_contiguous");
  m.def("fuse_rmsnorm_rope_quant_qkv", &at::native::fuse_rmsnorm_rope_quant_qkv, "fuse_rmsnorm_rope_quant_qkv");
  m.def("rms_rotary_embedding_fuse_with_kv_store", &at::native::rms_rotary_embedding_fuse_with_kv_store, "rms_rotary_embedding_fuse_with_kv_store");
  m.def("fuse_fill_rms_x2_concat", &at::native::fuse_fill_rms_x2_concat, "fuse_fill_rms_x2_concat");
  m.def("rn_dropout_forward_autograd", &at::native::fuse_rn_dropout_forward_autograd, "fused_rn_dropout forward");
  m.def("rn_add_forward_autograd", &at::native::fuse_rn_add_forward_autograd, "fused_rn_add forward");
  m.def("rn_add_dropout_forward_autograd", &at::native::fuse_rn_add_dropout_forward_autograd, "fused_rn_add_dropout forward");
  m.def("rn_add_dropout_bias_forward_autograd", &at::native::fuse_rn_add_dropout_bias_forward_autograd, "fused_rn_add_dropout_bias forward");
  m.def("droppath_forward_autograd",&at::native::drop_path_forward_autograd,"droppath forward") ;
  m.def("add_dropout_forward_autograd", &at::native::fuse_add_dropout_forward_autograd, "fused_dropout forward");
  m.def("fused_linear_bias_forward_autograd", &at::native::fused_linear_bias_forward_autograd, "fused_linearbias forward autograd");
  m.def("relu_dropout_forward", &at::native::fused_relu_dropout_forward, "fused_relu_dropout forward");
  m.def("relu_dropout_backward", &at::native::fused_relu_dropout_backward, "fused_relu_dropout backward");
  m.def("softmax_forward_autograd", &at::native::fuse_softmax_forward_autograd, "softmax autograd of forward");
  m.def("logsoftmax_forward_autograd", &at::native::fuse_logsoftmax_forward_autograd, "logsoftmax autograd of forward");
  m.def("accelerate_rnnt_loss", &at::native::accelerate_rnnt_loss, "accelerated rnntloss");
  m.def("accelerate_ctc_loss", &at::native::accelerate_ctc_loss, "accelerated ctcloss");
  m.def("create_gemmbias_handle", &at::native::create_gemmbias_handle, "create_gemmbias_handle");
  m.def("destroy_gemmbias_handle", &at::native::destroy_gemmbias_handle, "destroy_gemmbias_handle");
  m.def("create_gemmdbias_handle", &at::native::create_gemmdbias_handle, "create_gemmdbias_handle");
  m.def("destroy_gemmdbias_handle", &at::native::destroy_gemmdbias_handle, "destroy_gemmdbias_handle");
  m.def("scaled_upper_triang_masked_softmax_forward_autograd", &at::native::fuse_scaled_upper_triang_masked_softmax_forward_autograd, "scaled_upper_masked_softmax forward");
  m.def("scaled_masked_softmax_forward_autograd", &at::native::fuse_scaled_masked_softmax_forward_autograd, "masked_softmax_forward  forward");
  m.def("scaled_upper_triang_masked_softmax_dropout_forward_autograd", &at::native::fuse_scaled_upper_triang_masked_softmax_dropout_forward_autograd, "scaled_upper_triang_masked_softmax_dropout forward");
  m.def("scaled_masked_softmax_dropout_forward_autograd", &at::native::fuse_scaled_masked_softmax_dropout_forward_autograd, "scaled_masked_softmax_dropout forward");
  m.def("get_batch_per_block",&at::native::get_batch_per_block, "get_batch_per_block");
  m.def("bias_gelu_forward", &at::native::fused_bias_gelu_forward, "fused_bias_gelu forward");
  m.def("bias_gelu_backward", &at::native::fused_bias_gelu_backward, "fused_bias_gelu backward");
  m.def("rotary_positional_embedding_forward", &at::native::RoPE_forward_gpu, "rotary positional embedding forward for gpu");
  m.def("rotary_positional_embedding_backward", &at::native::RoPE_backward_gpu, "rotary positional embedding backward for gpu");
  m.def("FusedBiasSwiGLU_forward", &at::native::FusedBiasSwiGLU_forward_gpu, "fused bias SwiGLU forward for gpu");
  m.def("FusedBiasSwiGLU_backward", &at::native::FusedBiasSwiGLU_backward_gpu, "fused bias SwiGLU backward for gpu");
  m.def("fused_bmm", &at::native::fused_bmm, "fused transepose bmm transpose");
  m.def("bias_dropout_add_forward", &at::native::fuse_bias_dropout_add_forward, "fuse_bias_dropout_add forward");
  m.def("bias_dropout_add_backward", &at::native::fuse_bias_dropout_add_backward, "fuse_bias_dropout_add backward");
  m.def("groupnorm_forward_autograd", &at::native::fuse_groupnorm_forward_autograd, "groupnorm forward");
  m.def("celu_gn_forward_autograd", &at::native::fuse_celu_gn_forward_autograd, "celu_gn forward");
  m.def("gn_silu_forward_autograd", &at::native::fuse_gn_silu_forward_autograd, "gn_silu forward");
  m.def("sum_opt", &at::native::sum_opt, "sum_opt");
  m.def("mean_opt", &at::native::mean_opt, "mean_opt");
  m.def("topk_opt", &at::native::topk_opt, "topk_opt");
  m.def("gemm_opt", &at::native::gemm_opt, "gemm_opt");
  m.def("gemm_w4_a16", &at::native::gemm_w4_a16, "gemm_w4_a16");
  m.def("gemm_awq_w4a16", &at::native::gemm_awq_w4a16, "gemm_awq_w4a16");
  m.def("gemm_awq_w4a16_dequant", &at::native::gemm_awq_w4a16_dequant, "gemm_awq_w4a16_dequant");
  m.def("w4_to_w16", &at::native::w4_to_w16, "w4_to_w16");
  m.def("gemm_w8_a16_fp8_colmajor", &at::native::gemm_w8_a16_fp8_colmajor, "gemm_w8_a16_fp8_colmajor");
  m.def("fp8to16", &at::native::fp8to16, "fp8to16");
  m.def("grougemm_w8a8_pertensor_asm", &at::native::grougemm_w8a8_pertensor_asm, "grougemm_w8a8_pertensor_asm");
  m.def("grougemm_bf16_fp16_nt_asm", &at::native::grougemm_bf16_fp16_nt_asm, "grougemm_bf16_fp16_nt_asm");
  m.def("grougemm_bf16_fp16_tn_asm", &at::native::grougemm_bf16_fp16_tn_asm, "grougemm_bf16_fp16_tn_asm");
  m.def("grougemm_bf16_fp16_nn_asm", &at::native::grougemm_bf16_fp16_nn_asm, "grougemm_bf16_fp16_nn_asm");
  m.def("m_grouped_w8a8_gemm_nt_contig_asm", &at::native::m_grouped_w8a8_gemm_nt_contig_asm, "m_grouped_w8a8_gemm_nt_contig_asm");
  m.def("m_grouped_w16a16_gemm_nt_contig_asm",&at::native::m_grouped_w16a16_gemm_nt_contig_asm,"m_grouped_w16a16_gemm_nt_contig_asm");
  m.def("m_grouped_w8a8_gemm_nt_nopad_contig_asm",&at::native::m_grouped_w8a8_gemm_nt_nopad_contig_asm,"m_grouped_w8a8_gemm_nt_nopad_contig_asm");
  m.def("moe_gemm_w8a8", &at::native::moe_gemm_w8a8, "moe_gemm_w8a8");
  m.def("moe_gemm_marlin_w8a8", &at::native::moe_gemm_marlin_w8a8, "moe_gemm_marlin_w8a8");
  m.def("moe_gemm_marlin_w8a8_fp8", &at::native::moe_gemm_marlin_w8a8_fp8, "moe_gemm_marlin_w8a8_fp8");
  m.def("moe_gemm_marlin_w8a8_blockwise_fp8", &at::native::moe_gemm_marlin_w8a8_blockwise_fp8, "moe_gemm_marlin_w8a8_blockwise_fp8");
  m.def("moe_gemm_w4a8", &at::native::moe_gemm_w4a8, "moe_gemm_w4a8");
  m.def("gemm_w8_a8_smooth_colmajor", &at::native::gemm_w8_a8_smooth_colmajor, "gemm_w8_a8_smooth_colmajor");
  m.def("gemm_w8a8_smooth_asm", &at::native::gemm_w8a8_smooth_asm, "gemm_w8a8_smooth_asm");
  m.def("moe_w16a16_asm", &at::native::moe_w16a16_asm, "moe_w16a16_asm");
  m.def("moe_marlin_w16a16_asm", &at::native::moe_marlin_w16a16_asm, "moe_marlin_w16a16_asm");
#ifdef USE_MIOPEN
  m.def("miopen_conv_bias", &at::native::miopen_fused_conv_bias_forward, "fused_conv_bias forward");
  m.def("miopen_conv_bias_add", &at::native::miopen_fused_conv_bias_add_forward, "fused_conv_bias_add forward");
  m.def("miopen_conv_bias_relu", &at::native::miopen_fused_conv_bias_relu_forward, "fused_conv_bias_relu forward");
  m.def("miopen_conv_bias_add_relu", &at::native::miopen_fused_conv_bias_add_relu_forward, "miopen_fused_conv_bias_add_relu_forward");
  m.def("miopen_add_rms_norm_dynamic_quant", &at::native::miopen_add_rms_norm_dynamic_quant, "fused add rms morm dynamic quant");
  m.def("miopen_rms_norm_dynamic_quant_without_residual", &at::native::miopen_rms_norm_dynamic_quant_without_residual, "rms_norm_dynamic_quant_without_residual");
  m.def("miopen_head_rms_norm", &at::native::miopen_head_rms_norm, "head_rms_norm");
  m.def("miopen_groupnorm", &at::native::miopen_groupnorm, "groupnorm");
  // m.def("miopen_groupnorm_backward", &at::native::miopen_groupnorm_backward, "groupnorm_backward");
  m.def("miopen_add_groupnorm", &at::native::miopen_add_groupnorm, "add_groupnorm");
  m.def("miopen_layernorm", &at::native::miopen_layernorm, "layernorm");
  m.def("miopen_add_layernorm", &at::native::miopen_add_layernorm, "add_layernorm");
  m.def("miopen_skip_layernorm", &at::native::miopen_skip_layernorm,"skip_layernorm");
  m.def("miopen_moe_scatter_dynamic_quant", &at::native::miopen_moe_scatter_dynamic_quant, "moe_scatter_dynamic_quant");
  m.def("miopen_moe_swiglu_dynamic_quant", &at::native::miopen_moe_swiglu_dynamic_quant, "moe_swiglu_dynamic_quant");
  m.def("miopen_moe_gather", &at::native::miopen_moe_gather, "moe_gather");
  m.def("miopen_scaled_dot_product_attention", &at::native::miopen_scaled_dot_product_attention, "miopen_scaled_dot_product_attention");
  m.def("miopen_gemm_bias", &at::native::miopen_gemm_bias, "miopen_gemm_bias");
  m.def("miopen_transform_tensor", &at::native::miopen_transform_tensor, "miopen_transform_tensor");
  m.def("miopen_fused_adam", &at::native::miopen_fused_adam, "miopen_fused_adam");
#endif
  m.def("store_kv_cache", &at::native::store_kv_cache, "store_kv_cache");
  m.def("store_kv_cache_paged", &at::native::store_kv_cache_paged, "store_kv_cache_paged");
  m.def("topk_softmax", &at::native::topk_softmax, "topk_softmax");
  m.def("gemm_w8a8_asm", &at::native::gemm_w8a8_asm, "gemm_w8a8_block_asm");
  m.def("moe_w8a8_asm", &at::native::moe_w8a8_asm, "moe_w8a8_asm");
  m.def("moe_marlin_w8a8_asm", &at::native::moe_marlin_w8a8_asm, "moe_marlin_w8a8_asm");
  m.def("moe_marlin_w8a8_fp8_asm", &at::native::moe_marlin_w8a8_fp8_asm, "moe_marlin_w8a8_fp8_asm");
  m.def("moe_w4a8_asm", &at::native::moe_w4a8_asm, "moe_w4a8_asm");
  m.def("moe_w4a8_marlin_asm", &at::native::moe_w4a8_marlin_asm, "moe_w4a8_marlin_asm");
  m.def("moe_gemm_w16a16", &at::native::moe_gemm_w16a16, "moe_gemm_w16a16");
  m.def("moe_groupgemm_w4a8_asm", &at::native::moe_groupgemm_w4a8_asm, "moe_groupgemm_w4a8_asm");
  m.def("moe_groupgemm_marlin_w4a8_asm", &at::native::moe_groupgemm_marlin_w4a8_asm, "moe_groupgemm_marlin_w4a8_asm");
  m.def("m_grouped_marlin_w4a8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_w4a8_gemm_nt_masked_asm, "m_grouped_marlin_w4a8_gemm_nt_masked_asm");
  m.def("m_grouped_marlin_w8a8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_w8a8_gemm_nt_masked_asm, "m_grouped_marlin_w8a8_gemm_nt_masked_asm");
  m.def("m_grouped_marlin_fp8_gemm_nt_masked_asm", &at::native::m_grouped_marlin_fp8_gemm_nt_masked_asm, "m_grouped_marlin_fp8_gemm_nt_masked_asm");
  m.def("m_grouped_bf16_gemm_nt_masked_asm", &at::native::m_grouped_bf16_gemm_nt_masked_asm, "m_grouped_bf16_gemm_nt_masked_asm");
  m.def("moe_marlin_w4a16", &at::native::moe_marlin_w4a16, "moe_marlin_w4a16");
  m.def("moe_marlin_w4a8", &at::native::moe_marlin_w4a8, "moe_marlin_w4a8");
  m.def("moe_marlin_w4fp8", &at::native::moe_marlin_w4fp8, "moe_marlin_w4fp8");
  m.def("awq_marlin_repack", &at::native::awq_marlin_repack, "awq_marlin_repack");
  m.def("awq_marlin_repack_w4a8", &at::native::awq_marlin_repack_w4a8, "awq_marlin_repack_w4a8");
  m.def("blockwise_scaled_cast_transpose", &at::native::blockwise_scaled_cast_transpose, "blockwise_scaled_cast_transpose");
  m.def("rms_norm_dynamic_per_token_quant",&at::native::rms_norm_dynamic_per_token_quant,"rms_norm_dynamic_per_token_quant");
  m.def("rms_norm_fp8_quant",&at::native::rms_norm_fp8_quant,"rms_norm_fp8_quant");
  m.def("rotary_embedding_fuse", &at::native::rotary_embedding_fuse, "rotary_embedding_fuse",py::arg("positions"),py::arg("query"),py::arg("key"),py::arg("head_size"),py::arg("cos_sin_cache"),py::arg("is_neox"));
  m.def("rotary_embedding_deepseek_fuse", &at::native::rotary_embedding_deepseek_fuse, "rotary_embedding_deepseek_fuse",py::arg("positions"),py::arg("query"),py::arg("key"),py::arg("head_size"),py::arg("cos_sin_cache"),py::arg("is_neox"),py::arg("offsets") = py::none());
  m.def("groupgemm",&at::native::groupgemm,"groupgemm");
  m.def("groupgemm_permute",&at::native::groupgemm_permute,"groupgemm_permute");
  m.def("groupgemm_unpermute",&at::native::groupgemm_unpermute,"groupgemm_unpermute");
  m.def("fuse_groupgemm_permute_quant",&at::native::fuse_groupgemm_permute_quant,"fuse_groupgemm_permute_quant");
  m.def("moe_fused_gate",&at::native::moe_fused_gate,"moe_fused_gate");
  m.def("fuse_silu_mul_quant",&at::native::fuse_silu_mul_quant,"fuse_silu_mul_quant");
  m.def("fuse_silu_mul_quant_ep",&at::native::fuse_silu_mul_quant_ep,"fuse_silu_mul_quant_ep");
  m.def("fuse_silu_mul_fp8_quant",&at::native::fuse_silu_mul_fp8_quant,"fuse_silu_mul_fp8_quant");
  m.def("fuse_silu_mul_fp8_quant_ep",&at::native::fuse_silu_mul_fp8_quant_ep,"fuse_silu_mul_fp8_quant_ep");
  m.def("fuse_silu_and_mul",&at::native::fuse_silu_and_mul,"fuse_silu_and_mul");
  m.def("ds_concat",&at::native::ds_concat,"ds_concat");
  m.def("ms_deform_attn_forward", &at::native::ms_deform_attn_cuda_forward,"ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &at::native::ms_deform_attn_cuda_backward,"ms_deform_attn_backward");
  m.def("moe_sum", &at::native::moe_sum, "moe_sum",
      py::arg("input"),
      py::arg("output"),
      py::arg("bias")  =   torch::nullopt ,
      py::arg("expert_mask") =   torch::nullopt,
      py::arg("num_local_tokens") =   torch::nullopt,
      py::arg("factor") = 1.0f,
      py::arg("expect_m") = -1
  );
  m.def("moe_align_block_size", &at::native::moe_align_block_size, "moe_align_block_size",
      py::arg("topk_ids"),
      py::arg("num_experts"),
      py::arg("block_size"),
      py::arg("sorted_token_ids"),
      py::arg("experts_ids"),
      py::arg("num_tokens_post_pad"),
      py::arg("expert_map") =   torch::nullopt,
      py::arg("expert_mask") =   torch::nullopt,
      py::arg("num_local_tokens") =   torch::nullopt,
      py::arg("Is_EP") = false,
      py::arg("Is_fuse_fill") = true
  );
  m.def("mha_norm_rope_qkv_inplace_cuda", &at::native::mha_norm_rope_qkv_inplace_cuda,"mha_norm_rope_qkv_inplace_cuda");
  m.def("cross_norm_rope_q_kv_inplace_cuda", &at::native::cross_norm_rope_q_kv_inplace_cuda,"cross_norm_rope_q_kv_inplace_cuda");
  m.def("box_cost", &at::native::box_cost, "box_cost");
  m.def("encode_reg_t", &at::native::encode_reg_t, "encode_reg_t");
  m.def("mqa_logits", &at::native::mqa_logits, "mqa_logits");
  m.def("get_paged_mqa_logits_metadata", &lightop::deep_gemm::get_paged_mqa_logits_metadata,"get_paged_mqa_logits_metadata");
  m.def("paged_mqa_logits", &lightop::deep_gemm::paged_mqa_logits,"paged_mqa_logits");
  m.def("batch_linear_assignment", &at::native::batch_linear_assignment, "batch_linear_assignment");
  m.def("fill_zeros", &at::native::fill_zeros,"fill_zeros");
  m.def("rms_rotary_embedding_fuse", &at::native::rms_rotary_embedding_fuse,"rms_rotary_embedding_fuse");
  m.def("reshape_and_cache_cuda", &at::native::reshape_and_cache_cuda,"reshape_and_cache_cuda");
  m.def("ep_fused_quant_scatter", &at::native::ep_fused_quant_scatter,"ep_fused_quant_scatter");
  m.def("ep_scatter", &at::native::ep_scatter,"ep_scatter");
  m.def("destindex_copy_quantize_int8_cuda", &at::native::destindex_copy_quantize_int8_cuda, "destindex_copy_quantize_int8_cuda");
  m.def("destindex_dequantize_int8_cuda", &at::native::destindex_dequantize_int8_cuda, "destindex_dequantize_int8_cuda");
  m.def("kimi_k2_moe_fused_gate", &at::native::kimi_k2_moe_fused_gate, "kimi_k2_moe_fused_gate");
  m.def("fuse_rms_mrope_cuda", &at::native::fuse_rms_mrope_cuda,"fuse_rms_mrope_cuda");
  m.def("ep_gather", &at::native::ep_gather, "ep_gather");
  m.def("backward_polyfill_cuda", &at::native::backward_polyfill_cuda, "backward_polyfill_cuda");
  m.def("forward_polyfill_cuda", &at::native::forward_polyfill_cuda, "forward_polyfill_cuda");
  m.def("forward_polyfill_lightop", &at::native::forward_polyfill_lightop, "forward_polyfill_lightop");
  m.def("backward_polyfill_lightop", &at::native::backward_polyfill_lightop, "backward_polyfill_lightop");
  m.def("ep_fused_smooth_quant_scatter", &at::native::ep_fused_smooth_quant_scatter, "ep_fused_smooth_quant_scatter");
  m.def("per_token_dynamic_quant_int8", &at::native::per_token_dynamic_quant_int8, "per_token_dynamic_quant_int8");
  m.def("top_k_sampling_from_probs", &lightop::sampling::top_k_sampling_from_probs, "top_k_sampling_from_probs");
  m.def("top_p_sampling_from_probs", &lightop::sampling::top_p_sampling_from_probs, "top_p_sampling_from_probs");
  m.def("top_k_top_p_sampling_from_probs", &lightop::sampling::top_k_top_p_sampling_from_probs, "top_k_top_p_sampling_from_probs");
  m.def("rms_norm_opt", &at::native::rms_norm_opt, "rms_norm_opt");
  m.def("fused_add_rms_norm_opt", &at::native::fused_add_rms_norm_opt, "fused_add_rms_norm_opt");
  m.def("silu_and_mul_opt", &at::native::silu_and_mul_opt, "silu_and_mul_opt");
  m.def("gelu_and_mul_opt", &at::native::gelu_and_mul_opt, "gelu_and_mul_opt");
  m.def("gelu_tanh_and_mul_opt", &at::native::gelu_tanh_and_mul_opt, "gelu_tanh_and_mul_opt");
}

