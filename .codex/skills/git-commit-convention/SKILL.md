---
name: git-commit-convention
description: Use when preparing or reviewing git commits for the lightop repository in the current workspace. Check staged files for generated artifacts, inspect changes for hard-coded absolute paths and common bug patterns, suggest splitting unrelated changes into separate commits, and recommend commit messages such as [feat]: description.
---

# lightop Git 提交规范

当用户要求为 `lightop` 仓库准备提交、检查提交内容、生成 commit 信息时，使用这个 skill。

## 使用方式

当前环境可能不会自动加载仓库内 skill。需要显式使用时，可以这样请求：

```text
使用 lightop/.codex/skills/git-commit-convention/SKILL.md 帮我检查 staged 修改并生成 commit 信息
```

## 仓库路径

从以下目录执行命令：

```bash
lightop/.
```

如果当前会话在仓库上级目录，需要进入 `lightop/` 后再执行 git 命令。

## 第一步：检查 staged 缓存区

先检查 staged 缓存区是否有文件。如果 staged 区为空，提示用户需要先手动执行 `git add <files>` 加入要提交的文件，然后停止本次 commit 信息生成流程。

```bash
git status --short
git diff --cached --name-status
```

如果 staged 区非空，再继续检查 staged 文件。目标是在生成 commit 信息前，发现是否误加入了编译产物、缓存文件或临时文件。

```bash
git diff --cached --stat
```

重点检查以下通常不应该提交的文件，除非用户明确要求：
- 生成的 HIP 文件，例如 `*.hip`
- 编译产物和库文件，例如 `*.o`、`*.so`、`*.a`、`*.dll`、`*.dylib`
- Python 缓存，例如 `__pycache__/`、`*.pyc`、`.pytest_cache/`
- 构建输出，例如 `build/`、`dist/`、`*.egg-info/`
- 临时 benchmark 日志、profiler trace、本地输出文件

如果 staged 区存在可疑文件，先报告这些文件，并建议从 staged 区移除。除非用户明确要求，不要主动 unstage 或删除文件。

## 第二步：基础语法和常见 bug 检查

检查 staged 修改中的基础语法和常见 bug 风险，其中包含硬编码绝对路径检查。文件已经 staged 时，优先只检查 staged diff：

```bash
git diff --cached
```

### 绝对路径检查

重点搜索以下路径形式：
- `/public/home/...`
- `/public4/home/...`
- `/home/<user>/...`
- `/data/...`
- `/tmp/...`

如果发现绝对路径，需要说明风险，并给出修改建议。优先建议改成仓库相对路径、环境变量、配置项，或运行时自动发现的路径。

示例：
- 将 `/path/to/workspace/lightop` 替换为基于仓库根目录推导出的路径
- 将硬编码构建输出目录改为 `os.environ`、命令行参数或项目已有配置
- 只有当路径确实应该相对源码文件定位时，才使用 `Path(__file__).resolve()`

不要静默改写路径。除非用户明确要求直接修改文件，否则只给出具体修改建议。

### 基础语法检查

根据 staged 文件类型选择轻量检查：
- Python 文件：优先建议运行 `python3 -m py_compile <changed .py files>`，至少人工检查缩进、括号、逗号、变量名拼写、未定义变量、错误的 import。
- C++/CUDA/HIP 文件：人工检查括号匹配、宏条件、模板参数、kernel launch 参数、stream 参数、dtype 分支和头文件 include。
- JSON 配置文件：检查是否为合法 JSON，字段名是否拼写一致，数值类型是否符合已有配置模式。

如果无法实际运行检查，需要在回复中说明“未运行，仅基于 diff 人工检查”。

### 常见 bug 风险检查

优先参考本仓库历史中的 `fix`、`bugfix`、`BUG FIX` 提交信息；可用以下命令从本地历史抽样：

```bash
git log --all --grep='fix\|bugfix\|bug fix' --regexp-ignore-case --format='%h %s' -n 80
```

根据历史 fix/bugfix 模式，检查 staged 修改是否可能引入以下问题：
- OOM、memory overflow、vmfault、core dump：关注 batch size、buffer size、LDS/shared memory、workspace、临时 tensor 分配。
- dtype 或精度错误：关注 fp8、bf16、fp32、int8、uint8 之间的转换链路，避免读错 dtype 或错误 cast。
- contiguous、stride、layout、alignment 问题：关注 NCHW/NCHWc、非 contiguous 输入、16-byte/4-byte 对齐、stride 64、unaligned batch size。
- stream/context 问题：关注是否使用正确 CUDA/HIP stream，是否缺少 invalid context length 或 batch size 边界检查。
- config 查找或命名错误：关注 config 文件名、shape key、GFX/CU 后缀、TP/EP 配置是否和代码查找逻辑一致。
- scale/quant 参数访问错误：关注 a_scale、b_scale、xgrad scale、p_size、kv_cache_loc 等索引、shape 和负值检查。
- kernel 名称、注册和 import 错误：关注 `__init__.py`、Python wrapper、kernel 名称、算子导出是否一致。
- 测试脚本参数错误：关注默认 batch size、TP/EP 参数、模型尺寸、precision 阈值、compare failed 场景。

如果发现风险，给出具体文件和修改位置，并说明建议检查或修复方向；不要把所有清单机械列出，只报告和 staged diff 相关的风险。

## 第三步：判断是否需要拆分提交

检查 staged 修改是否把多个相互独立的功能或主题放在同一个提交里。

建议拆分的常见情况：
- 功能代码、测试代码、文档修改彼此可以独立提交
- 多个算子、多个 kernel、多个配置文件变更没有直接依赖关系
- bug 修复和重构混在一起
- 构建/打包修改和业务逻辑修改混在一起
- 大量自动生成配置或 benchmark 结果混入功能修改

如果发现多个独立功能混在一起，先建议用户拆分提交，并按主题给出不同 commit 信息供用户选择。每个建议 commit 必须同时列出对应应包含的提交文件，方便用户按文件分组 staged。

示例：

```text
建议拆分为 2 个提交：

1. [feat]: add W8A8 smooth quantization support
   建议包含文件：
   - lightop/quant_xxx.py
   - csrc/xxx_kernel.cu
   - lightop/configs/xxx.json

2. [test]: add W8A8 smooth quantization coverage
   建议包含文件：
   - test/w8a8/test_xxx.py
   - test/python/smoothquant/w8a8_test.py
```

如果修改内容高度相关、无法自然拆分，则说明可以保持一个提交。

## 第四步：生成推荐 commit 信息

完成 staged 文件检查、基础语法/常见 bug 检查和拆分判断后，生成推荐 commit 信息。

推荐格式：

```text
[<type>]: <description>

<optional body>
```

常用类型：
- `[feat]`：新增功能、算子、API、配置或能力
- `[fix]`：bug 修复或正确性修复
- `[test]`：仅测试相关修改
- `[docs]`：仅文档相关修改
- `[build]`：构建、打包、依赖或 setup 相关修改
- `[perf]`：性能优化
- `[refactor]`：不改变行为的结构调整
- `[chore]`：维护类修改

commit 描述信息默认使用简洁中文短句。只有用户明确要求英文，或项目已有提交规范明确要求英文时，才使用英文描述。描述应具体，不要使用过于笼统的表达。

示例：

```text
[feat]: 添加 W8A8 smooth quantization 支持
```

```text
[fix]: 移除 lightop 源码硬编码路径
```

```text
[build]: 将生成的 HIP kernel 加入 setup 打包数据
```

## 最终回复

回复用户时包含：
- 是否存在可疑 staged 文件；如果存在，列出文件并说明建议
- 是否发现绝对路径；如果存在，列出位置并给出修改建议
- 是否发现基础语法或常见 bug 风险；如果存在，列出文件、风险和建议
- 是否建议拆分提交；如果建议拆分，给出多个 commit 信息供用户选择，并为每个 commit 列出对应提交文件
- 如果不需要拆分，给出一条推荐 commit 信息
- staged 修改是否看起来可以提交
- 提醒用户提交 merge 申请，并 `@wangaq` 或 `@luoyao` 进行 code review
