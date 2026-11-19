# Cube N 条件参考入口

`cube_reference.py` 复用本地 `stable-worldmodel` 的 World、policy 和 CEM，以及匹配的 LeWM 权重。这里描述可运行入口；历史数值和接口核查报告见 [历史索引](../docs/history.md)。这不是已经完成论文成功率复现的声明。

## 参数与配置

- 模式：`manifest` 或 `evaluate-50`。
- 规划档位：`paper-300x10-top30`、`public-300x30-top30`。
- 图像、动作、目标与执行配置来自 `config/eval/cube.yaml` 和对应 solver 配置。
- `--config` 提供本地数据/权重路径；`--device`、`--render-backend` 可覆盖设备与渲染配置。
- `--out` 必须是新目录。

参考流程保留原案例抽样和每次执行完整 25 步计划的语义。它与研究入口默认每次执行 5 步不同，不可将两者结果直接混为同一基线。

## 命令

先根据实际本机配置生成清单：

```bash
python le-wm/cube_reference.py manifest \
  --config le-wm/config/research/cube-local.yaml \
  --out runs/cube/reference-manifest
```

只检查清单、权重身份和依赖，不执行环境回合：

```bash
python le-wm/cube_reference.py evaluate-50 \
  --config le-wm/config/research/cube-local.yaml \
  --manifest runs/cube/reference-manifest/reference-50.json \
  --out runs/cube/reference-check --check-only
```

`--check-only` 仍会读取数据、拟合动作 scaler、加载权重并记录环境身份；不是零成本检查，也不验证闭环成功。

实际执行时去掉 `--check-only` 并使用新输出目录，将运行 50 个案例的 CEM、zero、random 三组评估。这是批量实验，仅在用户要求参考评估时启动。macOS/GLFW 使用 `mjpython` 替换 `python`。

## 本轮清理改变了什么

移除了依赖 `runs/cube/...` 历史产物、硬编码两个案例的 `audit-two` 和 `--verified-audit` 前置流程。当前清单仍须与实际数据的参考抽样结果完全一致；每次仍保存 scaler、权重、源码、设备和协议身份。底层抽样、求解器、模型预处理与成功汇总实现没有改动。

保留的接口与模型数值测试位于 `tests/test_cube_reference.py`；新的入口契约测试位于 `tests/test_cube_reference_cli.py`。单元检查不替代目标机器上的真实权重与环境验证。
