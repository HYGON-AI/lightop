/*
 * GDN recurrent HIP kernel: exported C entry points and their ABI.
 *
 * gdn_recurrent.py feeds this file to ffi.cdef() verbatim, so it stays a plain
 * declaration list: no preprocessor lines, no include guard, no extern "C".
 * Signatures are copied from the .hip definitions; keep them in sync by hand.
 * Notation: tokens = N*T; HV = H*hratio; strides are in elements; NULL allowed
 * where noted; softplus(x) = log1p(exp(beta*x + threshold)).
 */

void gdn_recurrent_update(
    const void *q, const void *k, const void *v, const void *a,
    const void *b, const void *A_log, const void *dt_bias, const void *cu_seqlens,
    void *h0_source, const void *h0_indices, void *o, void *snap_buf,
    const void *snap_idx, int cache_steps, int N, int T,
    int HV, int K, int V, int hratio,
    int stride_q, int stride_k, int stride_v, int stride_a,
    int stride_b, float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void *stream);

void gdn_recurrent_update_bplus(
    const void *q, const void *k, const void *v, const void *a,
    const void *b, const void *A_log, const void *dt_bias, const void *cu_seqlens,
    void *h0_source, const void *h0_indices, void *o, void *snap_buf,
    const void *snap_idx, int cache_steps, const void *store_step, const void *store_row,
    int N, int T, int HV, int K,
    int V, int hratio, int stride_q, int stride_k,
    int stride_v, int stride_a, int stride_b, float scale,
    float softplus_beta, float softplus_threshold, int use_initial_state, int varlen,
    int disable_state_update, void *stream);

void gdn_recurrent_update_bplus_state(
    const void *q, const void *k, const void *v, const void *a,
    const void *b, const void *A_log, const void *dt_bias, const void *cu_seqlens,
    void *h0_source, const void *h0_indices, void *o, void *snap_buf,
    const void *snap_idx, int cache_steps, const void *store_step, const void *store_row,
    int N, int T, int HV, int K,
    int V, int hratio, int stride_q, int stride_k,
    int stride_v, int stride_a, int stride_b, float scale,
    float softplus_beta, float softplus_threshold, int use_initial_state, int varlen,
    int disable_state_update, void *stream);

void gdn_recurrent_update_nofma(
    const void *q, const void *k, const void *v, const void *a,
    const void *b, const void *A_log, const void *dt_bias, const void *cu_seqlens,
    void *h0_source, const void *h0_indices, void *o, void *snap_buf,
    const void *snap_idx, int cache_steps, int N, int T,
    int HV, int K, int V, int hratio,
    int stride_q, int stride_k, int stride_v, int stride_a,
    int stride_b, float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void *stream);

void gdn_recurrent_update_bplus_state_multi(
    const void *layer_ptrs, const void *h0_indices, const void *snap_idx, int cache_steps,
    const void *store_step, int L, int N, int T,
    int HV, int K, int V, int hratio,
    float scale, float softplus_beta, float softplus_threshold, int use_initial_state,
    int varlen, int disable_state_update, void *stream);

void gdn_verify_norm_pass(
    const void *q, const void *k, const void *a, const void *b,
    const void *A_log, const void *dt_bias, void *qn, void *kn,
    void *eg_out, void *beta_out, int n_units, int H,
    int HV, int K, int hratio, int stride_q,
    int stride_k, int stride_a, int stride_b, float softplus_beta,
    float softplus_threshold, float scale, void *stream);

void gdn_recurrent_update_bplus_prenorm(
    const void *q, const void *k, const void *v, const void *a,
    const void *b, const void *A_log, const void *dt_bias, const void *cu_seqlens,
    void *h0_source, const void *h0_indices, void *o, void *snap_buf,
    const void *snap_idx, int cache_steps, const void *store_step, const void *store_row,
    int N, int T, int HV, int K,
    int V, int hratio, int stride_q, int stride_k,
    int stride_v, int stride_a, int stride_b, float scale,
    float softplus_beta, float softplus_threshold, int use_initial_state, int varlen,
    int disable_state_update, const void *qn, const void *kn, const void *eg_buf,
    const void *beta_buf, int stride_qn, int stride_ab, int gate,
    void *stream);

void gdn_causal_conv1d_update(
    const void *x, const void *w, const void *bias, void *conv_state,
    const void *conv_idx, void *window, const void *inter_idx, const void *nxt,
    const void *sib, void *par_out, void *o, int batch,
    int dim, int seqlen, int state_len, int num_cache_lines,
    int s_x_seq, int s_x_dim, int s_x_tok, int s_w_dim,
    int s_w_width, int s_cs_seq, int s_cs_dim, int s_cs_tok,
    int s_ci, int s_wseq, int s_wstep, int s_wdim,
    int s_win, int s_ii, int s_nseq, int s_ntok,
    int s_sseq, int s_stok, int s_pseq, int s_ptok,
    int s_o_seq, int s_o_dim, int s_o_tok, int pad_slot_id,
    int has_bias, int silu, int cont, int save_inter,
    int tree, int use_pad, int idx64, void *stream);
