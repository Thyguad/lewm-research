# 历史研究索引

旧研究完整保存在分支 `research/cube-e0-e2`，本轮整理前的固定提交为 [`6bc69c4`](https://github.com/Thyguad/lewm-research/tree/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b)。以下材料解释过去做过什么，不规定新任务必须继续做什么。

| 历史材料 | 固定版本链接 |
| --- | --- |
| PushT 冻结规划完整教程 | [原 README](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/README_frozen_diagnostics.md) |
| Mac 原安装与实验记录 | [原教程](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/README_macos.md) |
| 服务器完整执行教程 | [原教程](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/04-LeWM-server-experiments-guide.md) |
| Cube E0–E2 计划与命令 | [原 README](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/README_cube.md) |
| 正常可解子集、扰动网格与固定案例恢复 | [原 cohort 记录](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/README_cube_cohort.md) |
| Cube 参考接口核查与两个案例结果 | [原参考报告](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/README_cube_reference.md) |
| 更早 Cube 运行结果说明 | [prior runs](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/le-wm/cube_reference_prior_runs.md) |
| PushT 实验报告与产物 | [历史报告](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/experiments/2026-11-18-macos-pusht/README.md) |
| Cube 最初工程验证记录 | [历史报告](https://github.com/Thyguad/lewm-research/blob/6bc69c4dc21f97b42f4d5328a09b062ebda88b7b/experiments/2026-11-19-cube-engineering/README.md) |

本轮移出当前代码树的专用实现：`cube_research/perturbation_recovery.py`（固定单案例 E1）、`cube_research/reference_audit.py`（依赖旧运行目录的两案例对照）。它们及对应测试仍存在于上述提交；通用环境、模型、CEM、适应/诊断组件与历史结果数据保留。

需要执行旧命令时，使用旧提交的独立 checkout/worktree及其对应依赖和产物。不要把旧审计回执改字段以适配新源码，也不要把旧报告中的“下一步”当成当前要求。
