# gemm_ops 原生接口按需加载记录

既有 W8A8 Grouped GEMM 回归会先导入系统 LMSlim。若 `lightop.gemm_ops` 在模块初始化时立即加载迁入的 LMSlim 私有实现，PyTorch 会检测到 `lmslim` namespace 下的 custom op 重复注册。

本次将 6 个 LMSlim 新接口和 `BlockSize` 改为模块属性按需加载：

- 只导入既有 LightOp GEMM/Grouped GEMM 接口时，不触发 LMSlim custom op 注册。
- 访问新迁入接口时，仍通过相同的 `lightop.gemm_ops.<name>` 或 `from lightop.gemm_ops import <name>` 方式加载。
- 新增隔离子进程测试，覆盖“先导入系统 LMSlim，再导入既有分类接口”的兼容场景。
