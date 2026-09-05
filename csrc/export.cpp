#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <vector>

#include "paged_mqa_logits/paged_mqa_logits.h"
#include "sparse_page_mqa/sparse_page_mqa.h"
#include "index_k_int8_dequant/index_k_int8_dequant.h"
#include "sampling/sampling.h"
enum GRAD_TYPE {nograd, wgrad, xgrad};
namespace at {  namespace native {
std::tuple<Tensor, Tensor> fused_deepep_prepare(
    const Tensor& hidden,
    const Tensor& smooth,
    const Tensor& ids,
    double min_limit,
    double max_limit);
void fused_metadata_kernel_general_launcher(
    Tensor seq_lens,
    Tensor req_to_token,
    Tensor req_pool_indices,
    Tensor cache_seqlens_int32,
    Tensor cu_seqlens_k,
    Tensor page_table,
    const c10::optional<Tensor>& swa_page_table,
    const c10::optional<Tensor>& full_to_swa_mapping,
    int64_t B,
    int64_t max_seq_pages,
    int64_t page_size,
    int64_t seq_len_delta,
    bool    use_swa);
Tensor rmsnorm_forward_autograd(const Tensor& x,const Tensor& weight,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_forward(const Tensor& self,const Tensor& weight,c10::optional<Tensor>& output_inplace,double eps,bool training);
std::tuple<Tensor,Tensor>rmsnorm_backward(const Tensor& grad, const Tensor& X,const Tensor& rstd,const Tensor& gamma);
void fused_rms_norm_contiguous(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
void fused_rms_norm_rope_contiguous(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);
void fused_rms_norm_rope_contiguous_q_cat(torch::Tensor& positions, torch::Tensor& q_pe, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed, torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, torch::Tensor& q_nope, torch::Tensor& q_cat, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit, bool flashmla_fp8_layout, float kv_quant_epsilon, const c10::optional<torch::Tensor>& kv_smooth_scale);
void fuse_rmsnorm_rope_quant_qkv(torch::Tensor& positions, torch::Tensor& q_c, torch::Tensor& q_pe, torch::Tensor& q_quant, torch::Tensor& q_scale, torch::Tensor& k_pe, torch::Tensor& kv_c, torch::Tensor& kv_c_normed,torch::Tensor& weight, torch::Tensor& cos_sin_cache, torch::Tensor& slot_mapping, torch::Tensor& kv_cache, std::string& kv_cache_dtype_str, float kv_cache_scale, bool is_neox_style, float epsilon, double limit);
Tensor grid_sample_forward_autograd(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
Tensor grid_sample_forward(const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
std::tuple<Tensor,Tensor> grid_sample_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, std::array<bool, 2> output_mask);
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
    std::optional<Tensor>& weight_q, 
    std::optional<Tensor>& weight_k, 
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
void gammarms_rotary_embedding_fuse_with_kv_store(
    Tensor& positions, 
    Tensor& query, 
    Tensor& key,
    //////////////
    Tensor& v,
    //////////////
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<Tensor>& weight_q, 
    std::optional<Tensor>& weight_k, 
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
    bool use_fp8_e4m3,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
    bool is_interleaved);
void split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant(
    Tensor& positions,
    Tensor& qkv,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    std::optional<Tensor>& weight_q,
    std::optional<Tensor>& weight_k,
    std::optional<Tensor>& residual_q,
    std::optional<Tensor>& residual_k,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_rope_out,
    Tensor& k_rope_out,
    Tensor& v_rope_out,
    int page_size,
    double epsilon,
    bool use_fp8_e4m3,
    bool use_fp8_cache);
Tensor fuse_rn_dropout_forward_autograd(const Tensor& x,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_dropout_bias_forward_autograd(const Tensor& x,const Tensor& y,const Tensor &bias, const Tensor& weight,double eps,double rate,bool training);
Tensor fuse_rn_add_forward_autograd(const Tensor& x,const Tensor& y,const Tensor& weight,double eps,bool training,bool inplace);
Tensor drop_path_forward_autograd(const Tensor& x,double p, bool training,bool scale_by_keep) ;
Tensor fuse_add_dropout_forward_autograd(const Tensor& x,const Tensor& y,double rate,bool training);
Tensor fuse_layernorm_forward_autograd(const Tensor& x,const std::optional<Tensor>& weight,const std::optional<Tensor>& bias,double eps,bool training);
Tensor fuse_layernorm_mixed_fp32_params_autograd(const Tensor& x,const std::optional<Tensor>& weight,const std::optional<Tensor>& bias,double eps,bool training);
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

void rms_norm_dynamic_per_token_quant(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit);
void rms_norm_dynamic_per_token_quant_sglang(Tensor &output,Tensor& input,const Tensor& weight,Tensor scales,double epsilon,std::optional<Tensor> residual,bool update_input,double limit,std::optional<Tensor> smooth_scale,std::optional<Tensor> norm_output,bool clamp_norm_output);
void mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang(Tensor& q_out,Tensor& packed_input,const Tensor& q_weight,const Tensor& kv_weight,
                                                       Tensor q_scales,double q_epsilon,double kv_epsilon);
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

std::vector<Tensor> moe_fused_gate(torch::Tensor& input, torch::Tensor& bias, int64_t num_expert_group, int64_t topk_group, int64_t topk, int64_t n_share_experts_fusion, double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
std::vector<Tensor> moe_fuse_gate_per_gourp_topk(torch::Tensor& input, torch::Tensor& bias, int64_t num_expert_group, int64_t topk_per_group, int64_t num_pad_experts, int64_t num_shared_experts, double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);

void fuse_silu_mul_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids) ;
void fuse_silu_mul_clamp_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, double limit, bool use_gelu, const c10::optional<torch::Tensor>& smooth_scale);
void fuse_silu_mul_clamp_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, double limit, const torch::Tensor& mask_m, int expect_m, bool use_gelu, const c10::optional<torch::Tensor>& smooth_scale);
void fuse_silu_mul_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m);
void fuse_silu_mul_fp8_quant(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &num_local_tokens_tensor, int topK, int expert_m, torch::optional<torch::Tensor> &expert_ids, float limit = -1.0f) ;
void fuse_silu_mul_fp8_quant_ep(torch::Tensor& input, torch::Tensor& output, torch::Tensor& scales, const int fp8type, torch::optional<torch::Tensor> &tokens_per_expert, torch::optional<torch::Tensor> &num_local_tokens_tensor,int topK,int expert_m, float limit);
void fuse_silu_and_mul(torch::Tensor& input, torch::Tensor& out);
void fuse_silu_and_mul_ep(torch::Tensor& input, torch::Tensor& out, const torch::Tensor& mask_m, int expect_m);
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
torch::Tensor mqa_logits(
    torch::Tensor& Q,
    torch::Tensor& K,
    torch::Tensor& Weights,
    torch::Tensor& KS,
    torch::Tensor& KE,
    int q_seq_len,
    int kv_seq_len,
    int num_heads,
    int head_dim,
    const torch::optional<torch::Tensor> &KV_scale =  torch::nullopt,
    bool clean_logits = true,
    const torch::optional<torch::Tensor> &D_out = torch::nullopt);
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
void per_token_dynamic_quant_int8(torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales, torch::Tensor const& smoothscale, bool round_after_smooth, double limit);
void rms_norm_opt(torch::Tensor& out,torch::Tensor& input,torch::Tensor& weight,double epsilon,double limit);
void rms_norm_no_weight(
    torch::optional<torch::Tensor> out,
    torch::Tensor& input,
    torch::optional<torch::Tensor> weight,
    double epsilon);
void fused_add_rms_norm_opt(torch::Tensor& input,torch::Tensor& residual, torch::Tensor& weight,double epsilon,double limit);
void silu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_and_mul_opt(torch::Tensor& out,torch::Tensor& input);
void gelu_tanh_and_mul_opt(torch::Tensor& out,torch::Tensor& input); 
void relu2_opt(torch::Tensor& out,torch::Tensor& input); 
torch::Tensor reshape_to_float(torch::Tensor input, int64_t vocab_size);
torch::Tensor add_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true);
torch::Tensor sub_int8(const torch::Tensor& a, const torch::Tensor& b, bool saturate=true);
std::tuple<torch::Tensor, torch::Tensor> pack_padded_sequence_custom(
    const torch::Tensor& input,
    const torch::Tensor& lengths,
    bool batch_first);
torch::Tensor fake_quantize_per_tensor_affine_custom(
    const torch::Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max);
torch::Tensor fake_quantize_per_channel_affine_custom(
    const torch::Tensor& input,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max);
torch::Tensor awq_w4a16_dequant_trans(
    torch::Tensor _weight, 
    torch::Tensor _scaling_factors, 
    torch::Tensor _zeros, 
    int IC, 
    int OC, 
    int group_size);
//------------------------------------------------- bilinear_upsample ---------------------------------------//
torch::Tensor bilinear_upsample(const torch::Tensor& x, const std::vector<int64_t>& output_size);
torch::Tensor bilinear_upsample_backward(const torch::Tensor& grad_output, const std::vector<int64_t>& input_size);
//------------------------------------------------- bilinear_upsample ---------------------------------------//
std::vector<torch::Tensor> ds_fused_qkv_cast_fp8(
  torch::Tensor q,
  torch::Tensor kv_nope, 
  torch::Tensor k_pe,   
  int64_t qk_nope_head_dim, 
  int64_t v_head_dim
);
void gather_convert_fp8_cache(
    torch::Tensor const& src_cache,    
    torch::Tensor const& dst,          
    torch::Tensor const& block_table,  
    torch::Tensor const& cu_seq_lens,  
    int64_t batch_size,
    double scale,                      
    const std::string& kv_cache_dtype, 
std::optional<torch::Tensor> seq_starts = std::nullopt);
void gather_and_maybe_dequant_cache(
    const torch::Tensor& src_cache,
    const torch::Tensor& dst,
    const torch::Tensor& block_table,
    const torch::Tensor& cu_seq_lens,
    const torch::Tensor& token_to_seq,
    int64_t num_tokens,
    const std::string& kv_cache_dtype,
    const torch::Tensor& scale,
    std::optional<torch::Tensor> seq_starts);
void fused_rms_norm_rope_int8quant_contiguous(
    torch::Tensor& positions, 
    torch::Tensor& q_pe, 
    torch::Tensor& k_pe, 
    torch::Tensor& kv_c, 
    torch::Tensor& kv_c_normed,      
    torch::Tensor& kv_c_normed_int8, 
    torch::Tensor& kv_c_normed_scale, 
    torch::Tensor& weight, 
    torch::Tensor& cos_sin_cache, 
    torch::Tensor& slot_mapping, 
    torch::Tensor& kv_cache, 
    std::string& kv_cache_dtype_str, 
    float kv_cache_scale, 
    bool is_neox_style, 
    float epsilon);
void per_token_quant_fp8(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales);
void rms_norm_smooth_per_token_dynamic_quant(
  Tensor &out,            
  Tensor& input,                
  std::optional<Tensor> weight,  
  const Tensor& smooth_scale,    
  Tensor scales,                
  double epsilon,
  std::optional<Tensor> residual,
  std::optional<Tensor> after_norm,    
  std::optional<Tensor> after_residual,
  bool update_input,               
  bool update_res                   
);
void moe_swiglu_dynamic_quant(torch::Tensor& scatter_tokens, torch::Tensor& smooth, torch::Tensor& experts_tokens_count, torch::Tensor& experts_tokens_start, torch::Tensor& output, torch::Tensor& scales, float beta);
void per_token_group_quant_fp8(
    torch::Tensor& out,        
    torch::Tensor  input, 
    torch::Tensor& scales,
    int group_size,            
    float eps = 1e-5f,         
    bool use_ue8m0 = false);    
void top_k_per_row_prefill(const torch::Tensor& logits,
        const torch::Tensor& rowStarts,
        const torch::Tensor& rowEnds, torch::Tensor& indices,
        int64_t numRows, int64_t stride0, int64_t stride1,
        int64_t topK);
void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
        const torch::Tensor& seqLens, torch::Tensor& indices,
        int64_t numRows, int64_t stride0, int64_t stride1,
        int64_t topK); 
torch::Tensor convert_req_index_to_global_index(
    torch::Tensor req_id,
    torch::Tensor block_table,
    torch::Tensor token_indices,
    int BLOCK_SIZE,
    int NUM_TOPK_TOKENS,
    bool HAS_PREFILL_WORKSPACE,
    std::optional<torch::Tensor> prefill_workspace_request_ids,
    std::optional<torch::Tensor> prefill_workspace_starts);
void fuse_qkv_head_rms_norm(
    torch::Tensor& token_data,   
    torch::Tensor& q_weight,     
    torch::Tensor& k_weight,    
    int q_head_num,
    int kv_head_num,
    int qk_head_dim,
    double epsilon); 
void gemma_fused_add_rmsnorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon);
void gemma_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);
void gemma_rms_norm_fp8_quant(
  Tensor &out, Tensor& input, const Tensor& weight, Tensor scales, 
  double epsilon, std::optional<Tensor> residual, bool update_input, int fp8type
);  
void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size); 
void prefill_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,
    torch::Tensor const& page_table_flat,
    torch::Tensor const& kv_dst);
void decode_gather_and_up_convert(
    torch::Tensor const& src_cache,
    torch::Tensor const& token_ids,
    torch::Tensor const& dst,
    torch::Tensor const& valid_lengths);
void decode_gather_and_up_convert_with_indices(
    torch::Tensor const& src_cache,
    torch::Tensor const& token_ids,
    torch::Tensor const& dst,
    torch::Tensor const& valid_lengths,
    torch::Tensor const& compact_indices);
void dsv4_dequantize_k_cache_paged_out(
    torch::Tensor const& cache_u8,
    torch::Tensor const& page_table,
    torch::Tensor const& out,
    int64_t page_size);
void dsv4_gather_upconvert_k_cache_paged(
    torch::Tensor const& cache,
    torch::Tensor const& indices,
    torch::Tensor const& lengths,
    torch::Tensor const& dst,
    torch::Tensor const& compact,
    int64_t page_size);
void dsv4_gather_upconvert_dual_k_cache_paged(
    torch::Tensor const& swa_cache,
    torch::Tensor const& swa_indices,
    torch::Tensor const& swa_lengths,
    torch::Tensor const& swa_dst,
    torch::Tensor const& swa_compact,
    int64_t swa_page_size,
    torch::Tensor const& extra_cache,
    torch::Tensor const& extra_indices,
    torch::Tensor const& extra_lengths,
    torch::Tensor const& extra_dst,
    torch::Tensor const& extra_compact,
    int64_t extra_page_size);
void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt);
void fast_topk_transform_sparse_mask_interface(
    const at::Tensor& score,
    const at::Tensor& nonzero_mask,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q);
void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt);
void fast_kpool_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    int64_t pool_size,
    const at::Tensor& seq_lens,
    std::optional<at::Tensor> page_table_opt,
    std::optional<at::Tensor> topk_indices_offset_opt,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> page_table_row_index_opt);
void fuse_act_quant_and_store_index_k_cache(
    torch::Tensor input,      
    torch::Tensor buf,     
    torch::Tensor loc,    
    int page_size,
    float eps = 1e-5f,         
    bool use_ue8m0 = false,
    bool is_e4m3 = true);   
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fuse_qk_quant_and_store_index_k_cache(
    torch::Tensor q_input,        // [num_tokens, q_hidden_size]
    torch::Tensor k_input,        // [num_tokens, k_hidden_size]
    torch::Tensor k_buf,          // The destination Paged Cache Buffer (uint8)
    torch::Tensor k_loc,          // [num_tokens] the cache block indices (int64)
    int page_size,
    c10::optional<torch::Tensor> weights_in_opt = c10::nullopt,
    float q_scale_factor = 1.0f,  
    float k_scale_factor = 1.0f,  
    float eps = 1e-5f,        
    bool use_ue8m0 = false,
    bool is_e4m3 = true);
void fuse_layernorm_rotary_embedding(
    at::Tensor& positions, 
    at::Tensor& query, 
    at::Tensor& key,
    int64_t head_size, 
    at::Tensor& cos_sin_cache, 
    bool is_neox, 
    std::optional<at::Tensor> weight_q, 
    std::optional<at::Tensor> bias_q,   
    std::optional<at::Tensor> weight_k, 
    std::optional<at::Tensor> bias_k,   
    std::optional<at::Tensor> residual_q,
    std::optional<at::Tensor> residual_k,
    double epsilon);   
std::vector<at::Tensor> fuse_norm_rope_quant_fp8(
    at::Tensor positions, 
    at::Tensor query, 
    at::Tensor key,
    int64_t head_size, 
    at::Tensor cos_sin_cache, 
    bool is_neox, 
    bool is_rmsnorm,
    std::optional<at::Tensor> weight_k, 
    std::optional<at::Tensor> bias_k,   
    double epsilon);      
void fused_quantize_and_store_mla_kv_cache(
    torch::Tensor k_nope,  
    torch::Tensor k_rope,    
    torch::Tensor kv_buffer, 
    torch::Tensor loc,        
    const std::string& fp8_dtype = "fp8_e4m3",
    float eps = 1e-5f
);
void fused_concat_and_store_mla_kv_cache(
    torch::Tensor k_nope,
    torch::Tensor k_rope,
    torch::Tensor kv_buffer,
    torch::Tensor loc,
    const std::string& fp8_dtype = "fp8_e4m3"
);
std::tuple<Tensor, Tensor, Tensor> qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, 
    Tensor& qkv, 
    int64_t q_size,
    int64_t kv_size,
    int64_t head_size,
    Tensor& cos_sin_cache, 
    bool is_neox, 
    Tensor weight_q, 
    Tensor weight_k, 
    std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k,
    double epsilon); 
std::tuple<Tensor, Tensor, Tensor> gemma_qkv_split_rms_rotary_embedding_fuse(
    Tensor& positions, Tensor& qkv, int64_t q_size, int64_t kv_size,
    int64_t head_size, Tensor& cos_sin_cache, bool is_neox, Tensor weight_q,
    Tensor weight_k, std::optional<Tensor> residual_q,
    std::optional<Tensor> residual_k, double epsilon); 
void layer_norm_fwd_1pass_opt(
    torch::Tensor& X,
    torch::Tensor& Y,
    torch::Tensor& W,
    std::optional<torch::Tensor> B,
    std::optional<torch::Tensor> Z,
    std::optional<torch::Tensor> Mean,
    torch::Tensor& Rstd,
    int64_t stride_x_row,
    int64_t stride_y_row,
    int64_t stride_z_row,
    int64_t M,
    int64_t N,
    double eps,
    int64_t BLOCK_N,
    int64_t ROWS_PER_BLOCK,
    bool HAS_BIAS,
    bool HAS_Z,
    bool NORM_BEFORE_GATE,
    bool IS_RMS_NORM,
    const std::string& ACTIVATION);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_nope_fp8_rope_bf16_pack(
    torch::Tensor k_bf16,
    float eps
); 
void quantize_nope_fp8_rope_bf16_pack_store(
    torch::Tensor k_bf16,
    torch::Tensor buf,
    torch::Tensor loc,
    int page_size,
    float eps
);   
bool supports_mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank);
torch::Tensor mhc_repeat_cp_sglang(
    const torch::Tensor& hidden_states, int64_t hc_mult,
    int64_t cp_size, int64_t cp_rank);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
    Tensor& q,
    const Tensor& kv,
    const Tensor& kv_norm_weight,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kv_rope(
    Tensor& q,
    Tensor& kv,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps);
void fused_deepseek_v4_qnorm_rope_kvnorm_rope(
    Tensor& q,
    Tensor& kv,
    const Tensor& kv_norm_weight,
    const Tensor& position_ids,
    const Tensor& cos_sin_cache,
    double eps);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& q_position_ids,
    const Tensor& kv_position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32(
    Tensor& q,
    const Tensor& kv,
    Tensor& k_cache,
    const Tensor& slot_mapping,
    const Tensor& q_position_ids,
    const Tensor& kv_position_ids,
    const Tensor& cos_sin_cache,
    double eps,
    int64_t cache_block_size);
void ep_build_m_indices(torch::Tensor topk_ids,torch::Tensor m_indices,int local_num_experts,int alignment);
void topk_ids_postprocess(torch::Tensor topk_ids, torch::optional<torch::Tensor> logical_to_physical_map, torch::optional<torch::Tensor> num_token_non_padded);
torch::Tensor dpskv4_hc_head_fused(torch::Tensor mixes, torch::Tensor x, torch::Tensor hc_scale, torch::Tensor hc_base, double hc_eps);
void ep_fused_fp8_quant_scatter(torch::Tensor input,torch::Tensor topk_ids,torch::optional<torch::Tensor> expert_map_opt,torch::Tensor expert_num_tokens,torch::Tensor aq_out,torch::Tensor aq_scale_out,torch::Tensor m_indices,torch::Tensor inv_perm, int local_num_experts,int alignment,int fp8type,bool fill_padded_m_indices);
std::vector<Tensor> moe_fused_gate_sqrtsoftplus(torch::Tensor& input, torch::Tensor& bias, int64_t topk, int64_t num_fused_shared_experts, bool renormalize, double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
void topk_transform_512(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt);
void topk_transform_1024(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    std::optional<at::Tensor> out_raw_indices_opt);
void lightop_hcu_alloc_extend_kernel(
    const Tensor pre_lens_ptr,
    const Tensor seq_lens_ptr,
    const Tensor last_loc_ptr,
    const Tensor free_page_ptr,
    Tensor out_indices,
    int64_t bs,
    int64_t page_size);
void mimo_v2_split_rope_vscale_kv_store(
    Tensor& positions,
    Tensor& qkv,
    int64_t q_size,
    int64_t k_size,
    int64_t v_size,
    int64_t head_size,
    int64_t v_head_size,
    Tensor& cos_sin_cache,
    bool is_neox,
    Tensor& k_buffer,
    Tensor& v_buffer,
    std::optional<Tensor>& kv_cache_loc,
    Tensor& q_out,
    Tensor& k_out,
    Tensor& v_out,
    double value_scale,
    std::optional<Tensor>& k_cache_scale,
    std::optional<Tensor>& v_cache_scale,
    bool use_fp8_e4m3,
    bool use_fp8_cache);
}
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm_forward_autograd", &at::native::fuse_layernorm_forward_autograd, "layernorm forward");
  m.def("layernorm_mixed_fp32_params_forward_autograd", &at::native::fuse_layernorm_mixed_fp32_params_autograd,
        "layernorm forward, activations half/bf16 and weight/bias float32");
  m.def("add_ln_forward_autograd", &at::native::fuse_add_ln_forward_autograd, "layernorm forward");
  m.def("ln_dropout_forward_autograd", &at::native::fuse_ln_dropout_forward_autograd, "fused_ln_dropout forward");
  m.def("ln_add_dropout_forward_autograd", &at::native::fuse_ln_add_dropout_forward_autograd, "fused_ln_add_dropout forward");
  m.def("ln_add_dropout_bias_forward_autograd", &at::native::fuse_ln_add_dropout_bias_forward_autograd, "fused_ln_add_dropout_bias forward");
  m.def("layernorm_droppath_forward_autograd", &at::native::fuse_layernorm_droppath_forward_autograd, "layernorm then droppath forward");
  m.def("rmsnorm_forward_autograd", &at::native::rmsnorm_forward_autograd, "rmsnorm forward autograd");
  m.def("rmsnorm_forward", &at::native::rmsnorm_forward, "rmsnorm forward");
  m.def("rmsnorm_backward", &at::native::rmsnorm_backward, "rmsnorm backward");
  m.def("fused_rms_norm_contiguous", &at::native::fused_rms_norm_contiguous, "fused_rms_norm_contiguous");
  m.def(
      "fused_rms_norm_rope_contiguous",
      &at::native::fused_rms_norm_rope_contiguous,
      "fused_rms_norm_rope_contiguous",
      py::arg("positions"),
      py::arg("q_pe"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0,
      py::arg("flashmla_fp8_layout") = false,
      py::arg("kv_quant_epsilon") = 1e-6f,
      py::arg("kv_smooth_scale") = py::none());
  m.def(
      "fused_rms_norm_rope_contiguous_q_cat",
      &at::native::fused_rms_norm_rope_contiguous_q_cat,
      "fused_rms_norm_rope_contiguous_q_cat",
      py::arg("positions"),
      py::arg("q_pe"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("q_nope"),
      py::arg("q_cat"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0,
      py::arg("flashmla_fp8_layout") = false,
      py::arg("kv_quant_epsilon") = 1e-6f,
      py::arg("kv_smooth_scale") = py::none());
  m.def(
      "fuse_rmsnorm_rope_quant_qkv",
      &at::native::fuse_rmsnorm_rope_quant_qkv,
      "fuse_rmsnorm_rope_quant_qkv",
      py::arg("positions"),
      py::arg("q_c"),
      py::arg("q_pe"),
      py::arg("q_quant"),
      py::arg("q_scale"),
      py::arg("k_pe"),
      py::arg("kv_c"),
      py::arg("kv_c_normed"),
      py::arg("weight"),
      py::arg("cos_sin_cache"),
      py::arg("slot_mapping"),
      py::arg("kv_cache"),
      py::arg("kv_cache_dtype_str"),
      py::arg("kv_cache_scale"),
      py::arg("is_neox_style"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0);
  m.def("grid_sample_forward_autograd", &at::native::grid_sample_forward_autograd, "grid_sample forward autograd");
  m.def("grid_sample_forward", &at::native::grid_sample_forward, "grid_sample forward");
  m.def("grid_sample_backward", &at::native::grid_sample_backward, "grid_sample backward");
  m.def("gammarms_rotary_embedding_fuse_with_kv_store", &at::native::gammarms_rotary_embedding_fuse_with_kv_store, "gammarms_rotary_embedding_fuse_with_kv_store");
  m.def("rms_rotary_embedding_fuse_with_kv_store", &at::native::rms_rotary_embedding_fuse_with_kv_store, "rms_rotary_embedding_fuse_with_kv_store");
  m.def("split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant", &at::native::split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant, "split_qkv_rms_rotary_embedding_fuse_with_kv_store_no_quant");
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
  m.def("bilinear_upsample", &at::native::bilinear_upsample, "bilinear_upsample");
  m.def("bilinear_upsample_backward", &at::native::bilinear_upsample_backward, "bilinear_upsample_backward");
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
  m.def(
      "rms_norm_dynamic_per_token_quant",
      &at::native::rms_norm_dynamic_per_token_quant,
      "rms_norm_dynamic_per_token_quant",
      py::arg("output"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("scales"),
      py::arg("epsilon"),
      py::arg("residual"),
      py::arg("update_input"),
      py::arg("limit") = -1.0);
  m.def(
      "rms_norm_dynamic_per_token_quant_sglang",
      &at::native::rms_norm_dynamic_per_token_quant_sglang,
      "rms_norm_dynamic_per_token_quant_sglang",
      py::arg("output"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("scales"),
      py::arg("epsilon"),
      py::arg("residual") = py::none(),
      py::arg("update_input") = true,
      py::arg("limit") = -1.0,
      py::arg("smooth_scale") = py::none(),
      py::arg("norm_output") = py::none(),
      py::arg("clamp_norm_output") = true);
  m.def("mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang",&at::native::mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang,
        "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang");
  m.def("rms_norm_fp8_quant",&at::native::rms_norm_fp8_quant,"rms_norm_fp8_quant");
  m.def("rotary_embedding_fuse", &at::native::rotary_embedding_fuse, "rotary_embedding_fuse",py::arg("positions"),py::arg("query"),py::arg("key"),py::arg("head_size"),py::arg("cos_sin_cache"),py::arg("is_neox"));
  m.def("rotary_embedding_deepseek_fuse", &at::native::rotary_embedding_deepseek_fuse, "rotary_embedding_deepseek_fuse",py::arg("positions"),py::arg("query"),py::arg("key"),py::arg("head_size"),py::arg("cos_sin_cache"),py::arg("is_neox"),py::arg("offsets") = py::none());
  m.def("groupgemm",&at::native::groupgemm,"groupgemm");
  m.def("groupgemm_permute",&at::native::groupgemm_permute,"groupgemm_permute");
  m.def("groupgemm_unpermute",&at::native::groupgemm_unpermute,"groupgemm_unpermute");
  m.def("fuse_groupgemm_permute_quant",&at::native::fuse_groupgemm_permute_quant,"fuse_groupgemm_permute_quant");
  m.def(
      "moe_fused_gate",
      &at::native::moe_fused_gate,
      "moe_fused_gate",
      py::arg("input"),
      py::arg("bias"),
      py::arg("num_expert_group"),
      py::arg("topk_group"),
      py::arg("topk"),
      py::arg("n_share_experts_fusion"),
      py::arg("routed_scaling_factor"),
      py::arg("apply_routed_scaling_factor_on_output") = true);
  m.def(
      "moe_fuse_gate_per_gourp_topk",
      &at::native::moe_fuse_gate_per_gourp_topk,
      "SparkV1 per-group top-k fused MoE gate",
      py::arg("input"),
      py::arg("bias"),
      py::arg("num_expert_group"),
      py::arg("topk_per_group"),
      py::arg("num_pad_experts"),
      py::arg("num_shared_experts"),
      py::arg("routed_scaling_factor"),
      py::arg("apply_routed_scaling_factor_on_output") = true);
  m.def("fuse_silu_mul_quant",&at::native::fuse_silu_mul_quant,"fuse_silu_mul_quant");
  m.def(
      "fuse_silu_mul_clamp_quant",
      &at::native::fuse_silu_mul_clamp_quant,
      "fuse_silu_mul_clamp_quant",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("limit"),
      py::arg("use_gelu") = false,
      py::arg("smooth_scale") = py::none());
  m.def(
      "fuse_silu_mul_clamp_quant_ep",
      &at::native::fuse_silu_mul_clamp_quant_ep,
      "fuse_silu_mul_clamp_quant_ep",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("limit"),
      py::arg("mask_m"),
      py::arg("expect_m") = -1,
      py::arg("use_gelu") = false,
      py::arg("smooth_scale") = py::none());
  m.def("fuse_silu_mul_quant_ep",&at::native::fuse_silu_mul_quant_ep,"fuse_silu_mul_quant_ep");
  m.def(
      "fuse_silu_mul_fp8_quant",
      &at::native::fuse_silu_mul_fp8_quant,
      "fuse_silu_mul_fp8_quant",
      py::arg("input"),
      py::arg("output"),
      py::arg("scales"),
      py::arg("fp8type"),
      py::arg("num_local_tokens_tensor"),
      py::arg("topK"),
      py::arg("expert_m"),
      py::arg("expert_ids"),
      py::arg("limit") = -1.0f);
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
  m.def("mqa_logits", &at::native::mqa_logits, "mqa_logits",
      py::arg("Q"),
      py::arg("K"),
      py::arg("Weights"),
      py::arg("KS"),
      py::arg("KE"),
      py::arg("seq_len"),
      py::arg("kv_seq_len"),
      py::arg("num_heads"),
      py::arg("head_dim"),
      py::arg("KV_scale") = torch::nullopt, 
      py::arg("clean_logits") = true,
      py::arg("D_out") = torch::nullopt);
  m.def("get_paged_mqa_logits_metadata", &lightop::deep_gemm::get_paged_mqa_logits_metadata,"get_paged_mqa_logits_metadata");
  m.def("paged_mqa_logits", &lightop::deep_gemm::paged_mqa_logits,"paged_mqa_logits");
  m.def(
      "dequantize_index_k_int8_paged",
      &lightop::deep_gemm::dequantize_index_k_int8_paged,
      "dequantize_index_k_int8_paged",
      py::arg("packed_cache"),
      py::arg("block_tables"),
      py::arg("context_lens"),
      py::arg("workspace"),
      py::arg("page_claims"),
      py::arg("waves_per_page") = 1);
  m.def(
      "page_mqa_logits_sparse_mask",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask,
      "page_mqa_logits_sparse_mask",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0);
  m.def(
      "page_mqa_logits_sparse_mask_grouped",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped,
      "page_mqa_logits_sparse_mask_grouped",
      py::arg("q"),
      py::arg("fused_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
  m.def(
      "page_mqa_logits_sparse_mask_grouped_int8",
      &lightop::deep_gemm::page_mqa_logits_sparse_mask_grouped_int8,
      "page_mqa_logits_sparse_mask_grouped_int8",
      py::arg("q"),
      py::arg("packed_kv_cache"),
      py::arg("weights"),
      py::arg("context_lens"),
      py::arg("block_table"),
      py::arg("schedule_meta"),
      py::arg("max_context_len"),
      py::arg("clean_logits") = true,
      py::arg("num_warps") = 4,
      py::arg("persistent_ctas") = 0,
      py::arg("group_size") = 5);
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
  m.def(
      "per_token_dynamic_quant_int8",
      &at::native::per_token_dynamic_quant_int8,
      "per_token_dynamic_quant_int8",
      py::arg("out"),
      py::arg("input"),
      py::arg("scales"),
      py::arg("smoothscale"),
      py::arg("round_after_smooth") = false,
      py::arg("limit") = -1.0);
  m.def("top_k_sampling_from_probs", &lightop::sampling::top_k_sampling_from_probs, "top_k_sampling_from_probs");
  m.def("top_p_sampling_from_probs", &lightop::sampling::top_p_sampling_from_probs, "top_p_sampling_from_probs");
  m.def("top_k_top_p_sampling_from_probs", &lightop::sampling::top_k_top_p_sampling_from_probs, "top_k_top_p_sampling_from_probs");
  m.def("top_k_first_top_p_sampling_from_probs", &lightop::sampling::top_k_first_top_p_sampling_from_probs, "top_k_first_top_p_sampling_from_probs");
  m.def(
      "rms_norm_opt",
      &at::native::rms_norm_opt,
      "rms_norm_opt",
      py::arg("out"),
      py::arg("input"),
      py::arg("weight"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0);
  m.def("rms_norm_no_weight", &at::native::rms_norm_no_weight, "rms_norm_no_weight");
  m.def(
      "fused_add_rms_norm_opt",
      &at::native::fused_add_rms_norm_opt,
      "fused_add_rms_norm_opt",
      py::arg("input"),
      py::arg("residual"),
      py::arg("weight"),
      py::arg("epsilon"),
      py::arg("limit") = -1.0);
  m.def("silu_and_mul_opt", &at::native::silu_and_mul_opt, "silu_and_mul_opt");
  m.def("gelu_and_mul_opt", &at::native::gelu_and_mul_opt, "gelu_and_mul_opt");
  m.def("gelu_tanh_and_mul_opt", &at::native::gelu_tanh_and_mul_opt, "gelu_tanh_and_mul_opt");
  m.def("relu2_opt", &at::native::relu2_opt, "relu2_opt");
  m.def("reshape_to_float", &at::native::reshape_to_float, "reshape_to_float");
  m.def("add_int8", &at::native::add_int8, "add_int8", py::arg("a"), py::arg("b"), py::arg("saturate") = true);
  m.def("sub_int8", &at::native::sub_int8, "sub_int8", py::arg("a"), py::arg("b"), py::arg("saturate") = true);
  m.def("pack_padded_sequence_custom", &at::native::pack_padded_sequence_custom, "pack_padded_sequence_custom");
  m.def("fake_quantize_per_tensor_affine_custom", &at::native::fake_quantize_per_tensor_affine_custom, "fake_quantize_per_tensor_affine_custom");
  m.def("fake_quantize_per_channel_affine_custom", &at::native::fake_quantize_per_channel_affine_custom, "fake_quantize_per_channel_affine_custom");
  m.def("awq_w4a16_dequant_trans",
          &at::native::awq_w4a16_dequant_trans, 
          py::arg("_weight"), 
          py::arg("_scaling_factors"), 
          py::arg("_zeros"), 
          py::arg("IC"), 
          py::arg("OC"), 
          py::arg("group_size")); 
 m.def("ds_fused_qkv_cast_fp8", &at::native::ds_fused_qkv_cast_fp8, "ds_fused_qkv_cast_fp8");    
 m.def("gather_convert_fp8_cache", &at::native::gather_convert_fp8_cache, "gather_convert_fp8_cache");    
 m.def(
     "gather_and_maybe_dequant_cache",
     &at::native::gather_and_maybe_dequant_cache,
     "Gather paged KV cache. Supported kv_cache_dtype values are auto, "
     "fp8/fp8_e4m3, fp8_ds_mla, and fp8_e5m2. FP8 inputs are "
     "dequantized as decode(cache) * scale to the FP32, FP16, or BF16 "
     "dtype selected by dst.",
     py::arg("src_cache"),
     py::arg("dst"),
     py::arg("block_table"),
     py::arg("cu_seq_lens"),
     py::arg("token_to_seq"),
     py::arg("num_tokens"),
     py::arg("kv_cache_dtype"),
     py::arg("scale"),
     py::arg("seq_starts") = py::none());
 m.def("fused_rms_norm_rope_int8quant_contiguous", &at::native::fused_rms_norm_rope_int8quant_contiguous, "fused_rms_norm_rope_int8quant_contiguous");    
 m.def("per_token_quant_fp8", &at::native::per_token_quant_fp8, "per_token_quant_fp8");    
 m.def("rms_norm_smooth_per_token_dynamic_quant", &at::native::rms_norm_smooth_per_token_dynamic_quant, "rms_norm_smooth_per_token_dynamic_quant");   
 m.def("moe_swiglu_dynamic_quant", &at::native::moe_swiglu_dynamic_quant, "moe_swiglu_dynamic_quant");  
 m.def("per_token_group_quant_fp8", &at::native::per_token_group_quant_fp8, "per_token_group_quant_fp8");  
 m.def("top_k_per_row_prefill", &at::native::top_k_per_row_prefill, "top_k_per_row_prefill");  
 m.def("top_k_per_row_decode", &at::native::top_k_per_row_decode, "top_k_per_row_decode");  
 m.def("convert_req_index_to_global_index", &at::native::convert_req_index_to_global_index, "convert_req_index_to_global_index");
 m.def("fuse_qkv_head_rms_norm", &at::native::fuse_qkv_head_rms_norm, "fuse_qkv_head_rms_norm");
 m.def("gemma_fused_add_rmsnorm", &at::native::gemma_fused_add_rmsnorm, "gemma_fused_add_rmsnorm");
 m.def("gemma_rmsnorm", &at::native::gemma_rmsnorm, "gemma_rmsnorm");
 m.def("gemma_rms_norm_fp8_quant", &at::native::gemma_rms_norm_fp8_quant, "gemma_rms_norm_fp8_quant");
 m.def("fast_topk_transform_interface", &at::native::fast_topk_transform_interface, "fast_topk_transform_interface");
 m.def(
     "fast_topk_transform_sparse_mask_interface",
     &at::native::fast_topk_transform_sparse_mask_interface,
     "fast_topk_transform_sparse_mask_interface");
 m.def("fast_topk_transform_ragged_interface", &at::native::fast_topk_transform_ragged_interface, "fast_topk_transform_ragged_interface");   
 m.def(
     "fast_kpool_topk_transform_interface",
     &at::native::fast_kpool_topk_transform_interface,
     "Fused kpool topk, token expansion, index transform, and tail append",
     py::arg("score"),
     py::arg("lengths"),
     py::arg("output"),
     py::arg("pool_size"),
     py::arg("seq_lens"),
     py::arg("page_table") = py::none(),
     py::arg("topk_indices_offset") = py::none(),
     py::arg("row_starts") = py::none(),
     py::arg("page_table_row_index") = py::none());
 m.def("cp_gather_and_upconvert_fp8_kv_cache", &at::native::cp_gather_and_upconvert_fp8_kv_cache, "cp_gather_and_upconvert_fp8_kv_cache");   
 m.def("prefill_gather_and_upconvert_fp8_kv_cache", &at::native::prefill_gather_and_upconvert_fp8_kv_cache, "prefill_gather_and_upconvert_fp8_kv_cache");
 m.def("decode_gather_and_up_convert", &at::native::decode_gather_and_up_convert, "decode_gather_and_up_convert");
 m.def("decode_gather_and_up_convert_with_indices", &at::native::decode_gather_and_up_convert_with_indices, "decode_gather_and_up_convert_with_indices");
 m.def(
     "dsv4_dequantize_k_cache_paged_out",
     &at::native::dsv4_dequantize_k_cache_paged_out,
     "DSV4 flat E4M3FN/UE8M0 paged dequantization into preallocated BF16 output",
     py::arg("cache_u8"), py::arg("page_table"), py::arg("out"), py::arg("page_size"));
 m.def(
     "dsv4_gather_upconvert_k_cache_paged",
     &at::native::dsv4_gather_upconvert_k_cache_paged,
     "DPSKV4 single-cache paged FP8/UE8M0 gather, BF16 upconvert, and compact indices");
 m.def(
     "dsv4_gather_upconvert_dual_k_cache_paged",
     &at::native::dsv4_gather_upconvert_dual_k_cache_paged,
     "DPSKV4 fused dual-cache paged gather, BF16 upconvert, and compact indices");
 m.def("fuse_act_quant_and_store_index_k_cache", &at::native::fuse_act_quant_and_store_index_k_cache, "fuse_act_quant_and_store_index_k_cache");
 m.def("fuse_qk_quant_and_store_index_k_cache", &at::native::fuse_qk_quant_and_store_index_k_cache, "fuse_qk_quant_and_store_index_k_cache");
 m.def("fuse_layernorm_rotary_embedding", &at::native::fuse_layernorm_rotary_embedding, "fuse_layernorm_rotary_embedding");
 m.def("fuse_norm_rope_quant_fp8", &at::native::fuse_norm_rope_quant_fp8, "fuse_norm_rope_quant_fp8");
 m.def("fused_quantize_and_store_mla_kv_cache", &at::native::fused_quantize_and_store_mla_kv_cache, "fused_quantize_and_store_mla_kv_cache");
 m.def("fused_concat_and_store_mla_kv_cache", &at::native::fused_concat_and_store_mla_kv_cache, "fused_concat_and_store_mla_kv_cache");
 m.def("qkv_split_rms_rotary_embedding_fuse", &at::native::qkv_split_rms_rotary_embedding_fuse, "qkv_split_rms_rotary_embedding_fuse");
 m.def("gemma_qkv_split_rms_rotary_embedding_fuse", &at::native::gemma_qkv_split_rms_rotary_embedding_fuse, "gemma_qkv_split_rms_rotary_embedding_fuse");
 m.def("layer_norm_fwd_1pass_opt", &at::native::layer_norm_fwd_1pass_opt,
       "layer_norm_fwd_1pass_opt",
       py::arg("X"),
       py::arg("Y"),
       py::arg("W"),
       py::arg("B"),
       py::arg("Z"),
       py::arg("Mean"),
       py::arg("Rstd"),
       py::arg("stride_x_row"),
       py::arg("stride_y_row"),
       py::arg("stride_z_row"),
       py::arg("M"),
       py::arg("N"),
       py::arg("eps"),
       py::arg("BLOCK_N"),
       py::arg("ROWS_PER_BLOCK"),
       py::arg("HAS_BIAS"),
       py::arg("HAS_Z"),
       py::arg("NORM_BEFORE_GATE"),
       py::arg("IS_RMS_NORM"),
       py::arg("ACTIVATION"));
 m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert");
 m.def("quantize_nope_fp8_rope_bf16_pack", &at::native::quantize_nope_fp8_rope_bf16_pack, "quantize_nope_fp8_rope_bf16_pack");
 m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_int32");
 m.def("fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32", &at::native::fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32, "fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32");
 m.def("fused_deepseek_v4_qnorm_rope_kv_rope", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope, "fused_deepseek_v4_qnorm_rope_kv_rope");
 m.def("fused_deepseek_v4_qnorm_rope_kvnorm_rope", &at::native::fused_deepseek_v4_qnorm_rope_kvnorm_rope, "fused_deepseek_v4_qnorm_rope_kvnorm_rope");
 m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp");
 m.def("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32", &at::native::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp_int32");
 m.def("ep_build_m_indices", &at::native::ep_build_m_indices,"ep_build_m_indices");
 m.def("ep_fused_fp8_quant_scatter", &at::native::ep_fused_fp8_quant_scatter,"ep_fused_fp8_quant_scatter");
 m.def("quantize_nope_fp8_rope_bf16_pack_store", &at::native::quantize_nope_fp8_rope_bf16_pack_store, "quantize_nope_fp8_rope_bf16_pack_store");
 m.def("supports_mhc_repeat_cp_sglang", &at::native::supports_mhc_repeat_cp_sglang,
       "Metadata-only SGLang mHC repeat+CP preflight",
       py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"), py::arg("cp_rank"));
 m.def("mhc_repeat_cp_sglang", &at::native::mhc_repeat_cp_sglang,
       "SGLang mHC CP-selected row broadcast",
       py::arg("hidden_states"), py::arg("hc_mult"), py::arg("cp_size"), py::arg("cp_rank"));
#if defined(USE_ROCM) && USE_ROCM
 m.attr("dsv4_mhc_repeat_cp_sglang") = py::bool_(true);
#else
 m.attr("dsv4_mhc_repeat_cp_sglang") = py::bool_(false);
#endif
 m.def("topk_ids_postprocess",&at::native::topk_ids_postprocess,"topk_ids_postprocess");
 m.def("dpskv4_hc_head_fused", &at::native::dpskv4_hc_head_fused, "dpskv4_hc_head_fused");
 m.def("fused_metadata_kernel_general", &at::native::fused_metadata_kernel_general_launcher,
       "fused_metadata_kernel_general",
       py::arg("seq_lens"),
       py::arg("req_to_token"),
       py::arg("req_pool_indices"),
       py::arg("cache_seqlens_int32"),
       py::arg("cu_seqlens_k"),
       py::arg("page_table"),
       py::arg("swa_page_table") = torch::nullopt,
       py::arg("full_to_swa_mapping") = torch::nullopt,
       py::arg("B"),
       py::arg("max_seq_pages"),
       py::arg("page_size"),
       py::arg("seq_len_delta"),
       py::arg("use_swa"));
m.def("moe_fused_gate_sqrtsoftplus",&at::native::moe_fused_gate_sqrtsoftplus,"moe_fused_gate_sqrtsoftplus");       
m.def("fused_deepep_prepare", &at::native::fused_deepep_prepare,
      "fused_deepep_prepare");
m.def("fuse_silu_and_mul_ep",&at::native::fuse_silu_and_mul_ep,"fuse_silu_and_mul_ep");
m.def(
     "topk_transform_512",
     &at::native::topk_transform_512,
     "DeepSeek V4 fused topk-512 and page-table transform",
     py::arg("scores"),
     py::arg("seq_lens"),
     py::arg("page_tables"),
     py::arg("out_page_indices"),
     py::arg("page_size"),
     py::arg("out_raw_indices") = py::none());       
m.def(
     "topk_transform_1024",
     &at::native::topk_transform_1024,
     "DeepSeek V4 fused topk-1024 and page-table transform",
     py::arg("scores"),
     py::arg("seq_lens"),
     py::arg("page_tables"),
     py::arg("out_page_indices"),
     py::arg("page_size"),
     py::arg("out_raw_indices") = py::none());
m.def("hcu_alloc_extend_kernel", &at::native::lightop_hcu_alloc_extend_kernel,
       "Parallel paged KV alloc-extend kernel",
       py::arg("pre_lens_ptr"),
       py::arg("seq_lens_ptr"),
       py::arg("last_loc_ptr"),
       py::arg("free_page_ptr"),
       py::arg("out_indices"),
       py::arg("bs"),
       py::arg("page_size"));
m.def("mimo_v2_split_rope_vscale_kv_store", &at::native::mimo_v2_split_rope_vscale_kv_store, "mimo_v2_split_rope_vscale_kv_store");            
}
