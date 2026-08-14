# LightOp GitHub Actions CI

本仓库的 CI 直接使用工作流中配置的当前构建环境，不使用版本锁、候选清单或自动更新版本的工作流，也不需要额外的更新 Token。

CI 在向 `master` 或 `lightop-dev` 提交或更新 Pull Request，以及手动触发时运行。出于内部 Runner 安全考虑，来自外部 fork 的 Pull Request 只执行公共校验任务。

## CI 流程

1. 公共 Runner 检查 Python 语法和公开命名规则。
2. 内部自托管 Runner 校验基础设施配置；外部 fork 不会直接在内部 Runner 上执行代码。
3. Checkout 关闭凭据持久化；Docker 构建上下文明确排除 `.git`、构建产物和测试产物。
4. 基础镜像使用显式 tag，并在每次运行时解析本次实际拉取的 digest。
5. 工具包使用工作流中配置的文件名下载；编译器安装包从共享目录动态选择最新候选。
6. 两个安装包在当次运行中计算 SHA-256 并写入构建环境记录，便于审计。
7. 编译原始 wheel，再通过 `auditwheel repair` 生成最终 wheel。
8. 从干净的编译镜像启动测试容器，只安装修复后的最终 wheel，并从 `/tmp` 执行导入检查和 OpTest。
9. OpTest 动态安装索引中的当前版本；支持 LightOp 时由 OpTest 执行，不支持时运行仓库自带的兜底测试集。兜底测试必须实际发现并执行用例，任一用例失败都会使任务失败，不允许静默跳过。
10. Docker 构建、编译和测试输出经过公开命名规则脱敏，同时保留原始命令退出码。
11. 上传前扫描日志、测试输出、wheel 成员路径和文本元数据；策略失败时不上传产物。

## 当前构建配置

构建服务地址、基础镜像 tag、工具包文件名、Python、Torch、NumPy 和设备列表直接维护在 `.github/workflows/ci.yml` 中，不需要配置 Repository Variables。Torch 固定为 `2.10.0`。OpTest 安装到独立临时目录，不改变 LightOp 测试环境；兜底测试使用仓库内不依赖外部二进制基准扩展的快速用例集。

当前下载地址使用 HTTP。Python 和 OpTest 共用同一个索引，并仅为该索引配置 `trusted-host`。

取消版本锁后，CI 会使用运行时可获得的基础镜像 digest、最新编译器安装包和当前 OpTest 版本。每次运行会记录实际选择和计算得到的摘要，但这些摘要不再与仓库中预先评审的固定值比较。

## 最终 wheel 验证边界

构建容器安装原始 wheel 仅用于帮助 `auditwheel` 定位包内动态库，不作为最终测试环境。测试容器重新从编译镜像启动，只安装 `/ci-dist` 中唯一的修复后 wheel，因此上传文件与 OpTest 实际验证的是同一个文件。

## 产物

```text
dist/*.whl
.ci-artifacts/<run-id>-<attempt>/
```

产物默认保留 14 天，只作为 GitHub Actions Artifact 保存，不发布到 Python 包索引。
