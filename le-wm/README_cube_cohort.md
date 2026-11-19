# Cube N 可解开发子集

- **官方参考评估**：[README_cube_reference.md](README_cube_reference.md)，使用 `cube_reference.py`。不是本子集的总体 benchmark。
- **N 可解子集的适应研究**：本页，使用 `research_cube.py screen` 和 `run --cohort`。成功定义不变：同一专家轨迹片段的当前状态到 25 步后目标，方块中心距目标 ≤4 cm；不代表完整抓取放置。
- **困难案例诊断**：[旧运行记录](cube_reference_prior_runs.md)。`e22d467bfaaa9357`、`e1db09ce9f216bf6` 保留为困难开发案例，不删、不重新归因于 MPS。

## 本地证据和清理

2026-11-19 从仓库源目录删除 78 个可再生缓存文件（Python 字节码、pytest 缓存、`.DS_Store`），共 1,206,637 字节逻辑大小，约 1.33 MiB 先前占用块；另移除空缓存目录和空的 `runs/cube/expert-replay`。没有跟随符号链接或清理 Python 环境。保留约 5.9 GiB 的既有 `runs/cube` 证据、所有失败/中断记录和快照，特别是 `cases.json`、`e0-dev/histories/`、`planning-diagnosis/report.json`、`reference-20261119-audit/`。`runs/cube/cache`（空的 checkpoints 子目录）、Matplotlib 缓存及约 2 GiB 的 `.cache/mineru-paper-reader` 模型/解析缓存未动；没有确认其可丢弃性和再下载成本。保留外置盘原始数据、权重和配置。Git 跟踪代码、通用配置、测试和说明，不跟踪本机配置、运行目录或大二进制产物。

新目录见 `runs/cube/cohort-20261119-batch0/` 和 `runs/cube/cohort-20261119-e0/`。不同运行必须使用新目录。不要从路径名称推断状态，读 `STATUS.json`；未完成批次不能供后续运行使用。

## 锁定规则和协议

配置 [cube-cohort.yaml](config/research/cube-cohort.yaml) 自动覆盖 `screen` 和 `run --cohort` 的规划设置。只对原始 manifest 的 `dev` 按顺序提取，再以 NumPy `default_rng(20261119)` 排列；`--batch 0` 是前 6 个候选，未来再次授权筛查可用 `--batch 1` 在新目录取接下来的 6 个，不改既有 cohort。本轮不启动下一批。筛选前写 `screen-plan.json`，包含配置、候选、源 manifest、权重/模型配置与关键实现哈希；筛选时核验源数据完整 SHA-256。后续同一文件只在设备/文件编号、字节数、修改及状态变更时间均相同时复用校验，身份变化或移动到另一机器时重算 SHA-256。

每个候选从确定性 `N` reset 实际执行 **50 步零命令**；`t=0` 已经成功或零命令后任何一步成功都排除。余下分别用种子 **42、43** 各运行一次 Frozen N；两次均须在 `t>0` 成功、最大方块位移至少 **1 cm**，且执行后有 pad-object 接触。只收前 3 个符合者，但记录全部 6 个；没有根据 I/D 或恢复器输出筛选。`screen-results.json` 保留全体结果，逐回合目录另含允许的历史、评价轨迹和账本；`cohort.json` 带封存哈希、全部筛选摘要及与源 dev 的映射。**2/2 是操作性门槛，不是长期稳定性证明。**

同一套 Frozen checkpoint、`FrozenModel` 评分桥和有界研究 `Planner` 用于筛选及所有比较：300 候选、10 轮、30 elites；每块 5 个独立的 5D 动作；时域、执行时域都是 5 块（25 步）；最多 50 步，首次成功即停。每个 5D 命令按数据统计标准化、按时间展平成 25D。候选在标准化空间采样，以反标准化并裁剪后的动作评分，使用 NumPy 总体标准差及 0.03 下限，最后执行有界分布均值，无 warm start。同一回合的 RNG 连续，跨条件/案例新建；E1 恢复在第 25 步使用同一前缀 RNG 状态和空计划缓存。E2 各恢复策略也从同一前缀 RNG 状态开始，不给任何方法旧动作缓存。每回合在 `data_collection` 模式用相同案例派生的 seed、数据 qpos/qvel 和目标初始化，使用实时首帧。**裁剪、RNG、std 和 reset 均不等同于官方参考入口。**

E0 只取封存的首例、独立评估种子 **142**，同一个案例、起点和目标跑 N/I/D，不接受 `--case-id` 重选。E0 匹配审计重放 N 的已发命令到 N/I/D-identity/D，并把这部分和快照重放作为离线步数单独记账；不向策略给真值接触、物体状态、扰动标签或专家命令。N 独立成功、至少一种干预通过实际核查且有 `N 成功/该条件失败` 的任务损失，并在第 25 步有未完成的适应决策，才允许 E1。D 需要 D-identity 与 N 方块轨迹在 `1e-7 m` 内相符、两边有带力且实际摩擦值匹配的接触、匹配命令下 D 对比 D-identity 方块响应大于 `1e-6 m`；I 需要实际 xyz 命令与大小匹配的增益，以及大于 `1e-6 m` 的末端响应。无接触、响应不足、N 失败或早成功都不能替换首例。

E1 的 Frozen/R-I/R-D/R-U 从相同历史、同一快照恢复校验后分支；每个分支新建回合内参数，记录调用、更新、参数量和计时。历史完成任务则标 `early_success`，无更新、无额外恢复动作，不计入恢复效果。没有观察到可信损失或有效修正时不扩大。E2 `collect` 仅接入 dev 接口验证；保留完整允许历史的 passive、同预算 continue、固定试探、从相同合法库随机选的试探。任何实际收集前都应先看到有效 E0/E1；不把选中的 dev 集用于训练后再在同一批报告泛化。

## 命令

以下在仓库根目录执行。`CONFIG` 提供显式本机数据与权重路径；迁移到 Linux 可改用自己的 YAML 或 CLI `--dataset` / `--weights`，不需先前 Mac 审计目录。Mac 可用虚拟环境的 `mjpython` 和 GLFW；此仓库的本机 Python 若找不到动态库，使用下面的 `DYLD_LIBRARY_PATH`。Linux 可用自己的 `python`、`--device cuda` 或 `--device cpu`，渲染 backend 由环境支持决定。CUDA 路径保留但本轮无真机验证。

```zsh
cd /Users/xuhaoquan/project/lewm-research
export DYLD_LIBRARY_PATH="$PWD/.python/cpython-3.11.15-macos-aarch64-none/lib"
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
CONFIG=le-wm/config/research/cube-local.yaml
CASES=runs/cube/cases.json
cube() { .venv-mac/bin/mjpython le-wm/research_cube.py --config "$CONFIG" --device mps --render-backend glfw "$@"; }

# 新输出示例。已跑的 batch0 和 E0 见下节；不要覆盖。
SCREEN=runs/cube/cohort-local-v1-screen
E0=runs/cube/cohort-local-v1-e0
cube screen --cases "$CASES" --batch 0 --out "$SCREEN"
cube run --stage e0 --cases "$CASES" --cohort "$SCREEN/cohort.json" --out "$E0"

# 仅在 E0 pair-audit.json 的 e1_allowed=true 时运行。
cube run --stage e1 --cases "$CASES" --cohort "$SCREEN/cohort.json" \
  --paired-e0 "$E0" --out runs/cube/cohort-local-v1-e1

# 仅在 E1 证据支持继续时。这个命令只收集 dev，不训练诊断器。
cube run --stage collect --limit 1 --cases "$CASES" --cohort "$SCREEN/cohort.json" \
  --out runs/cube/cohort-local-v1-e2-dev
```

## 本轮真实结果

实际用 MPS/GLFW 和现有数据、权重运行。零命令 6 回合、300 步；Frozen 筛选 6 回合、87 步。没有因为入选名额而丢弃失败或跳过记录。

| 固定顺序案例 | Episode，起止步 | `t=0` 距离 | Frozen N 42 / 43 | 结论 |
| --- | --- | ---: | --- | --- |
| `e6d14fe677e8a56a` | 2553，120–145 | 45.54 mm | 成功于 1 / 19 步 | 入选；边界例，种子 42 最大位移 10.07 mm |
| `f98411fee6571747` | 6665，135–160 | 289.69 mm | 成功于 17 / 16 步 | 入选 |
| `cba4e92adcb900f3` | 7444，51–76 | 253.67 mm | 成功于 16 / 18 步 | 入选 |
| `cc45e5cd46c1091e` | 4938，54–79 | 108.60 mm | 跳过 | 零命令下完成 |
| `616c47260a91e79a` | 9654，82–107 | 0 mm | 跳过 | 初始成功 |
| `d25bb86da1832b1e` | 9645，74–99 | 约 0 mm | 跳过 | 初始成功 |

**配对 E0**：种子 142、首例 `e6d14fe677e8a56a`，N/I/D 均第 1 步成功；最终距离分别为 **34.74 / 38.65 / 34.74 mm**。I 的实际控制分量最大变化 0.194，末端响应约 3.08 mm。D 与 D-identity 均有 16 个受力接触记录，求解器摩擦与各自配置相符；D-identity 与 N 一致，D 的方块位移差约 `1.26e-7 m`，小于预设阈值。这里只比较了同一个命令，不据此判定 D 普遍无效。E0 真实任务步数 3，另有 24 步快照复放和 4 步匹配干预复放。

**停止**：没有扰动造成的任务损失，三个条件都在第 25 步前成功，未运行 E1。E2 接入只做接口/合成测试，未批量采集、训练或运行正式 benchmark；也未换第二例追求正结果。没有理由用这一次 E0 启动 E2。后续若扩展，单独预先固定其余已入选案例的配对验证顺序，并保留本次首例结果；先确认是否存在可测扰动损失和未完成的适应决策点。

下一条本地命令可生成现有 E0 汇总，不再做环境交互：

```zsh
.venv-mac/bin/python le-wm/research_cube.py analyze \
  --runs runs/cube/cohort-20261119-e0 --out runs/cube/cohort-20261119-e0-summary
```
