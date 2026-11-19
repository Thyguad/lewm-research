# LeWM planning research

当前工作区用于探索冻结 LeWM 上的规划方法。新的研究方向以用户当次任务和明确选定的方案为准；此前的 PushT 预算实验、Cube E0–E2 适应实验是历史工作，不是新任务的前置流程。

## 从这里开始

- AI 编码约定：[AGENTS.md](AGENTS.md)。
- 安装、设备和本地数据：[docs/setup.md](docs/setup.md)。
- Cube 现有能力与入口：[le-wm/README_cube.md](le-wm/README_cube.md)。
- Cube 参考评估入口：[le-wm/README_cube_reference.md](le-wm/README_cube_reference.md)。
- Cube 目标集合规划：[le-wm/README_goal_set.md](le-wm/README_goal_set.md)。
- PushT 可复用工具：[le-wm/README_frozen_diagnostics.md](le-wm/README_frozen_diagnostics.md)。
- 旧研究与文档查阅：[docs/history.md](docs/history.md)。

## 代码布局

| 路径 | 用途 |
| --- | --- |
| `le-wm/jepa.py`、`module.py`、`train.py`、`eval.py` | LeWM 模型及上游训练/评估代码 |
| `le-wm/cube_research/` | Cube 数据、环境、模型桥接、规划及保留的实验组件 |
| `le-wm/research_*.py` | 研究 CLI、设备和日志工具 |
| `stable-worldmodel/` | 随仓库保存的依赖源码；不默认跟随远端升级 |
| `experiments/` | 已完成工作的历史记录，不是当前任务清单 |

已有适应器、诊断器和筛选组件可以作为对照或复用材料。它们的存在不代表新方法必须采用 N/I/D 分类、先筛出 N 成功案例，或执行 E0→E1→E2。

新方法使用独立配置和输出目录，保留同条件基线。数据、权重、虚拟环境、缓存和新运行产物由 `.gitignore` 排除；已提交历史记录保留原样。当前分支不预填新方法效果，也不将开发案例结果当作论文复现成功率。
