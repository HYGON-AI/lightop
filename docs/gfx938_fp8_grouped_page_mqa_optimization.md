# gfx938 原生 FP8 Grouped Page MQA 优化设计

## 0. 实现与验证状态（2026-08-17）

本文方案已经完成首版实现：gfx938/64CU 的 FP8 Q 与 packed FP8 KV 直接进入
原生 FP8 MMAC 内核；MTP3 的 `group_size=4` 路径使用 256-thread CTA、双缓冲
KV 预取、row-pair KV 复用和交错双 Q MMAC。Python 路由不再将 FP8 Q 转为
BF16，输出继续与 mask-aware TopK 成对使用。

126 节点以 BS8、MTP 3/1/4、45K--46K context 验证：

| Context | grouped persistent FP8 | 旧 persistent FP8 |
|---|---:|---:|
| 45K | 145.6 us | 180.7 us |
| 45.5K | 150.7 us | 185.7 us |
| 46K | 151.2 us | 187.1 us |

正确性用例共 7 项，覆盖 `group_size=1/3/4/5`、跨 256-token 边界、强制单
CTA persistent 循环、NaN scale 和负零 scale，均已通过。端到端精度与
350 tok/s 目标仍需在完整双节点服务上验收。

## 1. 结论

本方案用于改造 LightOp 的 Sparse/Grouped Page MQA，使 gfx938 的 FP8
输入直接进入原生 FP8 特化内核，同时保留当前 MTP grouped 路径对同一请求
Indexer KV 的复用。

核心结论如下：

- gfx936 继续使用已验证的 BF16 Q + BF16 KV 路径。
- gfx938 使用 FP8 Q + packed FP8 KV 的原生 FP8 MMAC 路径。
- Sparse/Grouped Page MQA 内部不再执行 `q.to(torch.bfloat16)`。
- gfx938 内核不再把 FP8 KV tile 展开为 BF16 后执行 BF16 MMAC。
- MTP2/MTP3/MTP4 的 `group_size=3/4/5` 契约保持不变。
- Sparse logits 与 `nonzero_mask` 的生产者—消费者契约保持不变。
- 第一阶段只对 gfx938/64CU、E4M3FN、H32、page size 64 封板；其他组合
  继续走已有正确性回退路径，不因模板已经能实例化就直接宣称生产支持。

这不是恢复旧的单行 FP8 persistent MQA，而是将旧 FP8 内核中的原生
load/MMAC/scale 语义，合并到当前能够真正共享 KV tile 的 grouped MTP
结构中。

## 2. 改造前基线

### 2.1 Python 入口进行了 Q 反量化

文件：

```text
python/lightop/gemmopt.py
```

当前 `page_mqa_logits_sparse_mask_grouped` 对 packed FP8 KV + FP8 Q 执行：

```python
if fused_kv_cache.dtype == torch.uint8 and q.dtype in fp8_q_dtypes:
    q = q.to(torch.bfloat16)
```

该操作会在每次 Page MQA 调用前额外产生：

- 一个 FP8 -> BF16 cast kernel；
- 一份 BF16 Q 临时存储；
- graph capture/replay 中额外的 kernel 和内存生命周期；
- 与原生 FP8 MMAC 不同的计算路径。

`paged_mqa_logits` 中还存在 gfx936 的相似隐式转换。该兼容逻辑是否删除，
应与 sparse/grouped 快路径分开处理；本方案首先保证 gfx938 grouped 路径
不会命中任何 Python cast。

### 2.2 Native grouped 内核仍然以 BF16 Q 为唯一输入

文件：

```text
csrc/paged_mqa_logits/paged_mqa_logits_grouped_sparse.cu
```

当前内核签名固定为：

```cpp
const at::BFloat16* q
```

host 检查也要求：

```cpp
q.scalar_type() == torch::kBFloat16
```

模板参数 `kFp8Kv` 只表示 KV 的存储格式，不表示计算类型是 FP8。

### 2.3 FP8 KV 在内核中转换为 BF16

当前 `kFp8Kv=true` 路径执行：

```cpp
convert_fp8x8_e4m3fn_to_bf16_flush_subnormal(raw)
```

转换后的 KV 写入 BF16 LDS，随后调用：

```cpp
builtin_b16_mmac<false>(...)
```

因此当前 grouped FP8 路径实际是：

```text
FP8 Q --Python cast--> BF16 Q
FP8 KV --device convert--> BF16 LDS
BF16 Q x BF16 KV --BF16 MMAC--> FP32 accumulator
```

它减少了 KV 的全局存储带宽，但没有利用 gfx938 的原生 FP8 MMAC。

### 2.4 已有可复用的原生 FP8 代码

当前普通 dense Page MQA 已有：

```text
paged_mqa_logits_fp8
paged_mqa_logits_fp8_small_block
```

其关键实现位于：

```text
csrc/paged_mqa_logits/paged_mqa_logits_impl.hpp
csrc/paged_mqa_logits/paged_mqa_logits.cu
```

该路径已经具备：

- `f8x16_t` Q/KV fragment；
- `K_TILE=64`；
- `builtin_fp8_mmac<true>`；
- packed KV 中 FP8 数据与逐 token FP32 scale 的解析；
- FP32 accumulator、ReLU、head weight 和 KV scale 的既有数学顺序。

在 commit `6df3110` 之前，仓库还存在
`paged_mqa_logits_fp8_length_masked_persistent`。它可作为 gfx938 persistent
调度和原生 FP8 指令的第二份参考，但它仍是每行独立计算，不能替代当前
grouped MTP 的共享 KV 结构。

## 3. 目标与非目标

### 3.1 目标

- gfx938 Page MQA 热路径中没有 FP8 Q -> BF16 Q 的独立 cast kernel。
- gfx938 grouped 内核直接从 packed KV 读取 FP8 operand，不生成 BF16 KV
  中间副本。
- 使用 gfx938 原生 FP8 MMAC，FP32 累加。
- TARGET_VERIFY 中同一请求的多行 Q 只加载一次对应 KV tile。
- 保持 `group_size=1/3/4/5` 的接口和布局契约。
- 保持 Sparse MQA + mask-aware TopK 的一体化接口。
- 支持 eager 和 CUDA/HIP Graph capture/replay。
- 最终端到端性能不低于 350 tok/s 目标，并通过相同服务配置 A/B 验证。

### 3.2 非目标

- 第一阶段不重写 Mask TopK。
- 第一阶段不改变 packed Indexer KV 的 `[blocks,64,1,132]` 布局。
- 第一阶段不改变 Q scale 已经折叠到 head weights 的上游契约。
- 不在 gfx936 上模拟原生 FP8 MMAC。
- 不把 BF16 Q + FP8 KV mixed 路径作为新的生产快路径。
- 不在同一 Patch 中同时处理 TARGET_VERIFY metadata 空泡。

## 4. 最终路由矩阵

| 平台 | Q | KV | Heads | 路径 | 生产状态 |
|---|---|---|---:|---|---|
| gfx936/80CU | BF16 | BF16 `[B,64,1,128]` | 64 | 现有 BF16 sparse/grouped | 保持 |
| gfx938/64CU | FP8 E4M3FN | packed FP8 `[B,64,1,132]` | 32 | 新原生 FP8 sparse/grouped | 本次封板 |
| gfx938/64CU | BF16 | BF16 | 8/16/32/64 | 现有通用 grouped BF16 | 正确性回退，不作为本次性能目标 |
| gfx936 | FP8 | packed FP8 | 任意 | 不进入原生 FP8 sparse 快路径 | 回退或由调用方提供 BF16 输入 |
| 任意 | BF16 | packed FP8 | 任意 | mixed/dequant | 不进入新生产快路径 |

Python 与 C++ 必须使用同一张支持矩阵。禁止 Python 放行后在 native 层
隐式转换，也禁止 native 支持了一组模板后 Python 无条件放行所有 heads。

## 5. 必须保持的数据与数学契约

### 5.1 Q 与 Q scale

gfx938 输入 Q 为：

```text
dtype: torch.float8_e4m3fn
shape: [rows, 1, 32, 128]
layout: contiguous
```

上游 `fuse_qk_quant_and_store_index_k_cache` 会生成 Q FP8 和 Q scale；生产
调用中 Q scale 已经乘入 head weights。Page MQA 不应再次乘 Q scale，也不应
通过反量化 Q 改变这一契约。

实现前必须增加一项集成断言或测试，确认实际传入的 `weights` 已包含对应
Q scale。否则原生 FP8 与当前 BF16-cast 路径都可能得到数值一致但整体缩放
错误的结果。

### 5.2 Packed KV

KV 保持现有布局：

```text
dtype: uint8
shape: [num_blocks, 64, 1, 132]
stride: [8448, 132, 132, 1]

每个 page：
  [64 * 128 bytes FP8 K]
  [64 * sizeof(float) bytes per-token scale]
```

新内核直接把前 `64 * 128` 字节作为 E4M3FN operand，将尾部 scale 作为
FP32 读取。

### 5.3 数学顺序

每个 token 的语义保持为：

```text
dot[h]   = fp8_dot(q[h], k_fp8[token])
value[h] = max(dot[h], 0.0f) * weight[h]
logit    = sum_h(value[h]) * kv_scale[token]
```

要求：

- FP8 MMAC 使用 FP32 accumulator；
- KV scale 在 head reduction 后应用，与现有 dense FP8 内核一致；
- 不把 FP32 scale 预乘回 FP8 operand；
- reduction 顺序尽量与已有 gfx938 dense FP8 kernel一致；
- 允许 FP8 原生路径与 BF16-cast 路径存在量化容差内差异，不要求 bitwise
  相同。

### 5.4 Sparse mask

输出契约保持：

```text
logits: FP32 [rows, max_context_len]
nonzero_mask: int16 [rows, ceil(max_context_len / 16)]
```

只有 FP32 原始 bit 为全零的 `+0.0f` 可以不写 logits：

```text
raw_bits(logit) == 0 -> mask=0，logits 对应存储禁止读取
raw_bits(logit) != 0 -> mask=1，必须写 logits
```

`-0.0`、NaN 和 Inf 必须写入，不能被零 tile 快速路径误删。该 logits 只能
交给 `fast_topk_transform_sparse_mask_fused`，不能交给 dense TopK。

### 5.5 MTP grouped 契约

```text
MTP2 -> group_size=3
MTP3 -> group_size=4
MTP4 -> group_size=5
```

组内各行：

- Q、weights、context length 独立；
- 属于同一请求；
- block table 内容相同；
- 连续存放。

LightOp 不在热路径读取 GPU tensor 回 host 验证 block table 是否相等。
SGLang 调度层必须证明该契约；LightOp 的单元测试需要覆盖契约成立时的所有
group size，并加入错误构造用例防止框架仅按总 batch 猜测 grouped。

## 6. 新 gfx938 FP8 内核设计

### 6.1 保留现有 grouped 并行结构

继续使用当前 grouped kernel 的核心结构：

```text
一个 request group
  -> 一个 1024-thread CTA
  -> 最多四个 primary row slot 并行
  -> group_size=5 时 row slot 0 再计算 secondary row
  -> row slot 0 负责加载共享 KV tile
  -> 组内所有 Q 行复用该 KV tile
```

不要退回旧 FP8 length-masked 内核的“一行一个 CTA、仅依赖 L2 邻接复用”
模式。后者可以作为性能基线，但不满足本次 MTP KV 真共享目标。

### 6.2 使用独立 FP8 特化，避免过度模板化

建议新增：

```text
csrc/paged_mqa_logits/paged_mqa_logits_grouped_sparse_fp8_impl.hpp
```

建议内核名：

```cpp
template <int kGroupedRows, int kHeads>
__global__ void paged_mqa_logits_grouped_sparse_fp8_kernel(...);
```

BF16 与 FP8 的 `K_TILE`、operand 类型、LDS 字节数、finite 判定和 MMAC
指令不同。第一版应使用两个清晰的特化内核，复用 host dispatch 和输出
分配，不应把所有差异塞进多层 `if constexpr` 后难以检查生成代码。

### 6.3 Q load

FP8 路径使用：

```text
Q_TILE = 16
K_TILE = 64
Q_ITER = ceil(kHeads / 16)
K_BLOCKS = 128 / 64 = 2
operand = f8x16_t
```

Q 以 16-byte vector 从 global memory 写入 LDS，再为每个 primary row 缓存
到 register。`group_size=5` 的 secondary Q 继续保留在 LDS，由 row slot 0
分段读取，避免复制一整套 secondary Q VGPR。

### 6.4 KV load

每次 row slot 0 根据共享 block table 定位 physical page，将
`16 tokens * 128 FP8 bytes` 直接加载进 FP8 LDS buffer：

```text
global packed FP8 KV -> FP8 LDS -> f8x16_t register
```

禁止出现：

```text
FP8 KV -> convert_fp8x8_e4m3fn_to_bf16 -> BF16 LDS
```

可以复用或抽取现有的 `buffer_load_lds_x4` 和 direct-to-LDS 描述符逻辑。
地址计算必须使用 `kv_cache_stride_bytes`，scale 区不能被当作下一 token 的
K 数据。

### 6.5 FP8 MMAC

每个 K block 读取一个 `f8x16_t` Q fragment 和一个 `f8x16_t` KV fragment，
分别对 `front/rear` 调用：

```cpp
builtin_fp8_mmac<true>(kv.front, q.front, acc);
builtin_fp8_mmac<true>(kv.rear, q.rear, acc);
```

该 builtin 仅在 `__gfx938__` 下生成。host dispatch 也必须限定
gfx938/64CU，避免 gfx936 binary 意外进入没有实现的 FP8 builtin。

### 6.6 KV scale

row slot 0 在加载 KV tile 时，同时把每个 token 的 FP32 scale 写入共享
scale buffer。组内所有 row 读取同一个 scale：

```text
sum = reduce_heads(relu(fp8_dot) * weights)
sum *= kv_scale[token]
```

scale 只加载一次并被 grouped rows 复用。

### 6.7 零 tile 快速路径

现有 grouped kernel 会在输入有限时跳过全零 KV tile。FP8 特化应直接检查
FP8 原始字节，不先转换为 BF16。

允许跳过的充分条件：

```text
KV tile 的数值为零
AND 当前 Q 全部有限
AND 当前 weights 全部有限
AND 当前 token scale 全部有限
```

注意：

- E4M3FN 的 NaN 编码必须被识别；
- signed zero 可以保守地按非零 raw bit 进入完整计算；
- scale 为 NaN/Inf 时不能跳过，因为 `0 * scale` 可能产生 NaN；
- 任何不确定情况都执行完整 MMAC，正确性优先。

建议新增 `fp8x16_all_finite_e4m3fn`，不要复用 BF16 bit mask。

### 6.8 Mask 写回

每个 16-token tile 仍由一个 wave ballot 生成 `int16` mask。有效区以外的
mask 必须保持为 0。第一版继续使用 `torch::zeros` 创建 mask，避免未覆盖
尾部被消费者读取；后续如要消除 memset，应先证明每个 capture bucket 的
全部 mask tile 都由内核覆盖。

## 7. Host 与 Python 修改

### 7.1 `python/lightop/gemmopt.py`

修改 `page_mqa_logits_sparse_mask_grouped`：

- 删除 FP8 Q 的 `q.to(torch.bfloat16)`；
- gfx938 原生路径要求 `q.dtype == torch.float8_e4m3fn`；
- gfx938 原生路径要求 packed `uint8` KV 和 H32；
- gfx936 BF16 路径继续要求 BF16 Q/BF16 KV；
- mixed BF16-Q/FP8-KV 不进入新 sparse/grouped 快路径；
- 错误信息打印实际 `(arch, CU, q dtype, kv dtype, heads, group_size)`。

修改 `_use_page_mqa_logits_sparse_mask`，使其和 native 支持矩阵完全一致。

`page_mqa_logits_sparse_mask` 在 gfx938 FP8 时继续转发到 grouped
`group_size=1`，不另建一套单行 Python API。

### 7.2 `paged_mqa_logits_grouped_sparse.cu`

host 侧按计算路径分派：

```cpp
if (is_gfx936_bf16) {
    dispatch_grouped_bf16(...);
} else if (is_gfx938_fp8) {
    dispatch_grouped_fp8(...);
} else {
    TORCH_CHECK(false, ...);
}
```

不要再使用只表达 `kFp8Kv` 的布尔模板来决定完整计算路径。至少应区分：

```text
BF16 compute + BF16 storage
FP8 compute + FP8 storage
```

首批 FP8 实例化：

```text
kHeads=32
kGroupedRows=1/3/4/5
```

### 7.3 Export 与公开 API

如果保持现有函数签名，则以下文件不需要新增 symbol：

```text
csrc/paged_mqa_logits/paged_mqa_logits.h
csrc/export.cpp
python/lightop/attention/mqa.py
```

wheel 必须整体安装，不能只替换 `op.so`，因为 Python dtype gate 与 native
dispatch 必须处于同一版本。

## 8. 分阶段实施

### Patch 1：去掉隐式反量化并收紧契约

- 删除 grouped sparse wrapper 中的 FP8 Q -> BF16 转换。
- Python/native 同时收紧为 gfx936 BF16/BF16 与 gfx938 FP8/FP8 两条生产
  路径。
- 加入 dtype/shape/arch 单元测试。
- 此 Patch 可以先让 gfx938 sparse 路径明确报 unsupported，不能在 FP8
  内核尚未接入时静默走错计算路径。

### Patch 2：实现 gfx938 grouped FP8 kernel

- 从当前 grouped BF16 kernel 复制并保留 MTP KV 共享结构。
- 从 dense/旧 persistent FP8 kernel 复用 FP8 operand、load、MMAC 和 scale
  语义。
- 首先封板 H32、page64、E4M3FN、group size 1/3/4/5。
- 保持 sparse mask 输出。

### Patch 3：接通 dispatch 与框架联动

- gfx938 FP8 输入路由到新 kernel。
- `group_size=1` 用于普通 decode/draft。
- `group_size=3/4/5` 用于 TARGET_VERIFY。
- 与 Mask TopK 的路由条件前置绑定；任何条件不满足时，在执行 Sparse MQA
  前整体回退 dense MQA + dense TopK。

### Patch 4：gfx938 参数调优

独立调优：

```text
persistent_ctas per request
KV double-buffer stage 数
group_size=4/5 的 primary/secondary row 安排
Q/weights 放 LDS 或 VGPR 的边界
zero-tile classify 的粒度
```

不得直接复制 gfx936 的 `persistent_ctas=20/32` 作为 gfx938 最终值。gfx938
只有 64 CU，FP8 kernel 的 VGPR/LDS/occupancy 也不同。

## 9. 正确性验证

### 9.1 独立 FP8 reference

reference 应显式执行：

```text
Q_fp8 -> FP32 数值解释
K_fp8 -> FP32 数值解释
dot -> ReLU -> weight -> head sum -> KV scale
```

不要只拿当前 BF16-cast grouped kernel 当唯一 oracle。需要同时比较：

- FP32 数学 reference；
- 已有 dense `paged_mqa_logits_fp8`；
- 新 grouped FP8 kernel。

### 9.2 Case 矩阵

| 维度 | 覆盖值 |
|---|---|
| Platform | gfx938/64CU 必测；gfx936 BF16 回归 |
| Heads | H32 必测；其他 heads 验证正确回退 |
| Group size | 1、3、4、5 |
| Request batch | 1、8，以及生产 local BS |
| Context length | 0、1、15/16/17、63/64/65、长序列、容量边界 |
| KV page | 连续 page、随机 physical page、重复 page |
| Values | random、全零 tile、部分零、signed zero、NaN、Inf scale |
| Execution | eager、graph capture/replay |
| Consumer | sparse mask contract、mask TopK、paged transform |

grouped 测试必须构造同组共享 page table，但不同组使用不同随机 physical
page，防止内核错误地跨 request 复用 KV。

### 9.3 Mask 验证

逐位置验证：

```text
mask=1 -> logits 已初始化且 raw bits 与 reference 一致或在容差内
mask=0 -> reference 必须是 +0.0f
```

另外验证：

- 有效 token 范围外 mask 为 0；
- NaN/Inf/-0 对应 mask 为 1；
- sparse TopK 不读取 mask=0 的 logits；
- 大量零 tie 下满足 exact ordered-key TopK 合同。

### 9.4 API 回归

建议新增：

```text
test/test_paged_mqa_logits_sparse_fp8.py
```

现有 `test/test_paged_mqa_logits.py` 继续覆盖 dense BF16/FP8/mixed/INT8，
新测试只负责 sparse/grouped 原生 FP8 和 mask 合同，避免一个脚本同时承担
过多路由语义。

## 10. 性能验证

### 10.1 三组算子 A/B

必须在相同 Q/KV、context lengths、block table 和 stream 下比较：

```text
A. dense gfx938 native FP8 Page MQA
B. 当前 grouped：FP8 Q cast BF16 + FP8 KV convert BF16 + BF16 MMAC
C. 新 grouped gfx938 native FP8 MMAC
```

分别测量：

- Q cast kernel duration；
- Page MQA kernel duration；
- MQA 完整调用墙钟时间；
- DRAM read bytes 与 L2 hit/reuse；
- VGPR、LDS、waves/CU、occupancy；
- group size 1 与 TARGET_VERIFY group size 的差异。

### 10.2 Trace 命中证据

新路径的 profile 必须满足：

- Page MQA 前没有 `aten::to`、`_to_copy` 或 FP8->BF16 cast kernel；
- kernel 名明确包含 grouped/sparse/fp8 特化；
- ISA/code object 中命中 gfx938 FP8 MMAC，而不是 BF16 MMAC；
- TARGET_VERIFY 的 grouped kernel 只加载一份组共享 KV tile；
- 后续命中 sparse-mask TopK，而不是 dense TopK。

### 10.3 端到端验收

在相同服务配置下记录：

```text
Page MQA 平均/P50/P99 duration
Mask TopK duration
TARGET_VERIFY cycle duration
GPU idle
TPOT
throughput tok/s
acceptance length
```

验收原则：

- 新路径必须同时快于当前 BF16-dequant grouped 路径和不复用 KV 的 dense
  FP8 基线，至少不能以增加其他阶段耗时换取单 kernel 变快；
- 端到端目标不低于 350 tok/s；
- acceptance length 和精度评测不得出现统计显著回退；
- 性能结论以完整 MQA + TopK cycle 和端到端 A/B 为准，不只看一个 kernel。

## 11. 回退与上线

- LightOp 新内核复用现有 public API，不改变默认开关。
- SGLang 继续通过联动开关控制 Sparse MQA + Mask TopK。
- gfx938 原生 FP8 的所有支持条件必须在调用 Sparse MQA 前一次性判断。
- 条件不满足时整体回退到 dense native FP8 MQA + dense TopK。
- 禁止 Sparse MQA 已经产生未完整初始化 logits 后再回退 dense TopK。
- 出现 mismatch、graph capture 失败、hang 或性能回退时，关闭联动开关即可
  回到原路径，不需要把 FP8 输入临时反量化来维持 fast path。

## 12. 建议实施顺序

```text
1. 建立 gfx938 FP8 dense/current-grouped 基线
2. Patch 1：删除 grouped Python cast，收紧 dtype/arch gate
3. Patch 2：实现 H32 grouped native FP8 kernel
4. 单算子验证 group_size=1/3/4/5 与 mask 合同
5. Patch 3：接通 Sparse MQA + Mask TopK 框架路由
6. eager/graph、服务精度和 trace 验证
7. Patch 4：按 gfx938/64CU 调 persistent CTA 与流水
8. 端到端 350 tok/s A/B 验收
9. 验证完成后更新 PAGE_MQA_MASK_TOPK_INTEGRATION_GUIDE.md 的支持矩阵
```

每个 Patch 独立提交和 profile。不要把 dtype 路由、kernel 改造、TopK 改造
和 metadata 空泡优化合成一个无法归因的提交。
