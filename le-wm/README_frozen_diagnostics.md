# PushT 冻结规划工具

这组代码可用于 PushT 基线与规划诊断。此前的规划预算研究已成为历史记录，不是新方法的默认研究路线。环境安装见 [setup](../docs/setup.md)，原教程与结果见 [历史索引](../docs/history.md)。

| 入口 | 用途 |
| --- | --- |
| `research_cem.py` | 案例生成、权重预检、规划采集 |
| `research_diag.py` | 诊断、候选重放与结果分析 |
| `research_runtime.py` | 权重、预处理、环境重建 |
| `research_device.py` | CPU/MPS/CUDA 选择与张量边界 |
| `research_common.py` | 案例、身份记录与状态文件 |

从仓库根目录查看当前参数：

```bash
python le-wm/research_cem.py --help
python le-wm/research_diag.py --help
python le-wm/tests/test_research_contracts.py -v
```

本地数据和权重路径按对应 CLI/配置填写。数据集标识、环境动作接口和 Cube 不同，不能仅替换路径后复用 Cube 的 5 维动作逻辑。

已有研究 CEM 支持分批评分、独立随机流、明确的执行缓冲和计时记录。新的方法比较应保持相同案例、模型、执行与计算预算；历史统计不能直接充当改变协议后的基线。没有在本轮文档清理中重新运行 PushT 任务实验。
