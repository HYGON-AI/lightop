import subprocess
import sys


def test_category_exports_preserve_aliases():
    import lightop
    from lightop import attention, kvcache, norm
    from lightop.attention.mqa import mqa_logits_topk_sampled_int8
    from lightop.fused_metadata import dsa_cp_prefill_metadata_plan

    assert attention.mqa_logits_topk_sampled_int8 is mqa_logits_topk_sampled_int8
    assert kvcache.dsa_cp_prefill_metadata_plan is dsa_cp_prefill_metadata_plan
    for name in ("rms_norm_dynamic_per_token_quant_sglang",
                 "mla_qkv_a_rms_norm_dynamic_per_token_quant_sglang"):
        assert getattr(norm, name) is getattr(lightop, name)
        assert name in norm.__all__
    for category, names in (
        (attention, ("spark_mla_rope_cat_store", "indexer_q_rope_hadamard_quant",
                     "mla_q_absorb_rope_concat")),
        (kvcache, ("indexer_k_norm_rope_hadamard_store", "fill_swa_ring_mapping")),
    ):
        for name in names:
            assert getattr(category, name) is getattr(lightop.op, name)
            assert name in category.__all__
    for name in ("topk_transform_ragged_v2", "topk_transform_1024",
                 "fast_kpool_topk_transform_fused"):
        assert getattr(attention, name) is getattr(lightop, name)


def test_older_native_optional_symbols_do_not_break_imports():
    script = '''
import lightop
names = ("spark_mla_rope_cat_store", "indexer_q_rope_hadamard_quant",
         "mla_q_absorb_rope_concat", "indexer_k_norm_rope_hadamard_store",
         "fill_swa_ring_mapping")
for name in names:
    if hasattr(lightop.op, name):
        delattr(lightop.op, name)
from lightop import attention, kvcache, norm, quant
assert callable(attention.mqa_logits)
assert callable(norm.rms_norm)
assert callable(quant.per_token_dynamic_quant_int8)
assert not hasattr(kvcache, "fill_swa_ring_mapping")
'''
    subprocess.run([sys.executable, "-c", script], check=True)
