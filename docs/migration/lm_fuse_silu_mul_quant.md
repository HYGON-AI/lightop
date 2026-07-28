# lm_fuse_silu_mul_quant 迁移记录

- 分类：`lightop.activation`
- 来源：LMSlim `lmslim/quantize/quant_ops.py`
- 实现：从 `_lmslim_native.quantize.quant_ops` 按需直接导出，不复制或重写函数体。
- 覆盖：公开对象/签名；FP16、BF16；三组 token/hidden shape；输出 dtype/shape；per-token scale 与 INT8 数值。
- 参考：PyTorch FP32 `silu(gate) * value` + dynamic per-token INT8 quant，量化误差允许 1 LSB。
