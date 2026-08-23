# LightOp CI 说明

本文档记录 `.github/workflows/ci.yml` 的策略取舍，方便下一个接手 CI 的人不用重新推理。

## 触发矩阵

| 事件 | validate | test | publish |
|---|:---:|:---:|:---:|
| PR → `master` | ✓ | ✓ | ✗ |
| PR → `master`（来自 fork） | ✓ | ✗（不在内网 runner 上执行外部代码） | ✗ |
| `push` 到 `master`（PR merge 之后） | ✓ | ✓ | ✓ |
| Actions 面板 `workflow_dispatch`，勾选 `publish_packages` | ✓ | ✓ | ✓ |
| Actions 面板 `workflow_dispatch`，不勾 `publish_packages` | ✓ | ✓ | ✗ |

## 关键策略：**测例失败不阻塞 merge / publish**

`Run functional tests` 步骤递归执行 `test/` 下全部 `test_*.py` 算子测试文件，以及分布式 all-reduce 测试 `benchmark_allreduce.py`；不按文件名跳过任何测例，并按其汇总退出码分类处理：

| 汇总 exit | 含义 | CI 行为 |
|---|---|---|
| `0` | 所有测例通过 | 正常绿 |
| `1` | 有测例失败 | **依然绿**，只在日志和 Step Summary 打 `::warning::` |
| 其他 | 测试容器、wheel 安装或运行器本身失败 | **红**，job fail |

**为什么这么设计**：算子库在版本迭代过程中经常有已知的、允许暂时挂的测例（性能回退、数值容差、上游 torch 变化等）。硬门槛会把这些噪音变成阻塞。管理员应通过 **JUnit 报告**判断这些失败是否可接受，再决定是否 merge。CI 只做"跑完并如实报告"，不做"通过/不通过"的判断。

如果未来需要**强制某些测试必须通过**，两种做法：

1. 另加一个 critical test step，不做上面的容错，任一脚本退出非 0 就 fail 整个 job；
2. GitHub 分支保护规则里勾 "Require status checks"，但因为 `test` job 对功能测例失败保持绿，只能挡基础设施失败或编译失败。

## 管理员如何决定是否 merge

1. 打开 PR，等 CI 跑完（validate + test × 每个 torch 版本）
2. 在 GitHub Actions run 页顶部看 **Step Summary**：一个 `Total/Passed/Failed/Errors/Skipped` 表格 + 折叠的失败测例列表（每条第一行错误消息）
3. 如果失败细节不够，在节点目录 `${LIGHTOP_CI_OUTPUT_ROOT}/lightop/<run-id>/<attempt>/torch-<ver>/lightop_test.xml` 查看完整 JUnit 报告
4. wheel 保存在同一节点目录，仅在 test job 成功（含"仅测例失败"）时写入
5. 确认可接受 → merge → `push` 事件触发 publish，两个 torch lane 各上传一个 `*manylinux*.whl` 到 devpi

## 应急：不合并直接发包

Actions → LightOp CI → **Run workflow** → 选目标分支 → 勾 `publish_packages: true` → Run。

跳过 PR 流程，仍然要求 validate + test job 拉起来（test 恒绿，除非测试基础设施失败）。用于修复紧急问题而不想走 PR review 的场景。

## GPU allocation

The CI uses fixed devices and does not install or invoke CIsmi. The Torch 2.10
lane uses device `0`; the Torch 2.11 lane uses device `1`. Each lane requires
one device for compilation, so the two matrix lanes require two devices when
they run in parallel. The selected device is passed to the compile container
through `ROCR_VISIBLE_DEVICES`.

The test container does not inherit that single-device filter. It follows the
`optest` allocation model: `hy-smi` is sampled twice and normal operator tests
run on one stable idle HCU through `HIP_VISIBLE_DEVICES`; scripts containing
`init_process_group` wait until every HCU is idle and run with `torchrun` across
all detected devices. A host lock serializes LightOp test suites on the runner,
so a multi-HCU test cannot overlap another LightOp test lane.

## Build and test isolation

Each Torch lane uses two fresh containers with separate responsibilities. The
build container contains the compiler, matched Torch/Triton, and wheel-building
tools. After the wheel is copied out, it is stopped. The test container is
pulled from the pre-built vLLM/HCU image matching its Torch lane: vLLM 0.21.0
for Torch 2.10 and vLLM 0.25.1 for Torch 2.11. CI uninstalls the LightOp package
already present in that test image, force-installs the wheel built by this run
without dependencies, imports the vLLM native extension, and then runs the CI
operator test scripts under `test/`. Thus vLLM, llguidance, and their native dependencies are never
resolved or built during the LightOp test stage.

## 可覆盖的仓库变量 / secrets

`vars.*`（Settings → Secrets and variables → Actions → Variables）：

- `LIGHTOP_TORCH_VERSIONS` — JSON 数组，默认 `["2.10.0", "2.11.0"]`
- `LIGHTOP_CI_BASE_IMAGE_REPOSITORY` — 编译 base image
- `LIGHTOP_DTK_PKG_PATH` — DTK 包下载地址
- `LIGHTOP_PYTHON_INDEX_URL` / `LIGHTOP_PYTHON_TRUSTED_HOST` — 内部 pip 索引
- `LIGHTOP_CI_OUTPUT_ROOT` — 首选节点输出目录，默认 `/public/home/guochy1/guochy1/framework/workspace/logs`；只读时回退到 runner 工作区旁的 `.lightop-ci-outputs`
- `LIGHTOP_DEVPI_URL` — devpi 发布地址

`secrets.*`：

- `LIGHTOP_DEVPI_USER` / `LIGHTOP_DEVPI_PASSWORD` — devpi 发布凭据（缺失时 publish 会 fail-fast 并 mask 打印）
