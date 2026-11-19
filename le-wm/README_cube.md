# Cube 工具入口

本文描述当前代码，不规定新研究必须沿用旧 E0–E2 流程。旧协议与结果见 [历史索引](../docs/history.md)；安装与设备见 [环境说明](../docs/setup.md)。

## 可复用部分

| 文件 | 能力 |
| --- | --- |
| `cube_research/data.py` | 本地 HDF5、动作统计、按完整轨迹划分案例 |
| `cube_research/model.py` | 冻结权重桥接、图像预处理、CEM |
| `cube_research/simulator.py` | Cube 执行、评价通道、完整快照与回放 |
| `cube_research/protocol.py` | 动作转换、允许的观测、历史和预算账本 |
| `cube_research/reference.py` | 使用本地 stable-worldmodel 的参考评估组件 |
| `research_device.py`、`research_common.py` | 设备、身份记录与输出管理 |

Cube 是 5 维动作，一个模型 block 含 5 个不同底层动作。模型归一化与环境动作转换是不同环节，新增算法应沿用已核查的形状和顺序。

## 已有 CLI

```bash
python le-wm/research_cube.py --help
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml doctor
python le-wm/cube_reference.py --help
```

`research_cube.py` 保留原通用研究工具，包括案例生成、普通运行、适应/诊断及分析。`screen`、`perturb-e0` 和 `run --cohort` 是旧研究专用路径，不是新方法的默认入口。绑定单一历史案例的 `perturb-e1` 已移除；需要复现它时使用历史提交。

`cube_reference.py` 是独立的 N 条件参考入口，说明见 [README_cube_reference.md](README_cube_reference.md)。它不再要求先取得两个历史案例的审计目录。

## 使用边界

现有研究协议是同轨迹未来 25 步的物体位置目标、50 个控制步预算、约 40 mm 位置达标阈值，不检查朝向。它不能直接解释为完整抓取放置任务。研究 CEM 与参考 CEM 的采样、裁剪、随机流及执行频率并非完全相同。

新增方法应明确案例来源、目标、成功定义、执行频率和计算预算，并为全部对照使用相同设置；无需继承旧案例筛选规则。图像/目标/声明传感输入与评价真值保持分开，既有实验记录只用于其原有范围的结论。
