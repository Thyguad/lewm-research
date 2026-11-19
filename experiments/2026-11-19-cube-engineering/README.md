# Cube E0–E2 工程验证记录（2026-11-19）

基线 main：`a1158f1b9b8281c8afa0ddbec9d0a884648d58a2`。工作分支：`research/cube-e0-e2`。
本目录只记录工程验证，不是 Cube 基准实验结果。未启动正式批量实验、长期训练或权重/数据下载。

## 实际修改

新增 `le-wm/research_cube.py`、`le-wm/config/research/cube.yaml`、`le-wm/cube_research/`（协议、数据、仿真、冻结模型桥接/CEM、适应、诊断、实验、接触审计、统计），以及 `le-wm/README_cube.md`、4个测试文件和1个真实模拟器检查脚本。

已有源文件仅修改 Cube.reset 的两处 seed 传递；新增 Cube 安装 extra，更新根 README 和忽略规则。没有修改 PushT 的模型/规划/记录入口、配置或历史实验结果。

实现：E0 的 N/I/D；E1 的 Frozen/R-I/R-D/R-U 配对交叉；E2 的完整被动历史、继续任务、固定/随机试探，独立训练/校准与锁定测试；共享完整状态快照/检查、信息隔离、预算/模型调用/适应/时间记录、案例级配对统计。没有 E3–E5 或自适应主动试探实现。

## 实际验证

| 检查 | 执行结果 | 证据含义 |
| --- | --- | --- |
| Cube 协议、动作变换、隔离、完整历史、分组、适应数学/重置、统计 | 通过 | 使用合成数据，不证明 Cube 任务效果 |
| 新 runner 采集→适应→保存，以及训练/校准→锁定评估 | 通过 | 注入合成模型/环境的执行契约测试，不是官方权重前向 |
| 故意模拟器失败 | 通过 | 12个失败恢复分支仍保存部分历史、原因、已用预算，运行标为 incomplete |
| 以上4个新测试文件 | **14 passed** | CPU/Python 单元与契约测试 |
| `test_research_contracts.py` | **11 passed** | 旧 PushT 轻量契约仍通过 |
| `test_research_devices.py` | 因缺 torch 阻塞 | 不宣称旧 PushT 设备/端到端已在本环境重跑 |
| 真实 Cube/MuJoCo CPU+EGL 检查 | 通过 | 本仓库真实 Cube、控制器、MuJoCo 和渲染，采用 isolated-imports 绕过 torch-heavy 包初始化 |
| 静态检查 | ruff F/E9、compileall、CLI help、TOML解析、git diff --check 通过 | 仅实现检查 |
| doctor | blocked | 无本地权重/HDF5；缺 torch、torchvision、hydra、stable_pretraining |

真实模拟器检查命令：

```bash
python le-wm/tests/cube_simulator_check.py --isolated-imports --render-backend egl
```

在 MuJoCo 3.3.6、OGBench 1.2.0、dm-control 1.0.34、Gymnasium 1.4.0、Python 3.12.14 下：

- 使用 OGBench 真值专家生成70个控制命令，只作离线夹爪接触测试；分别在 N、显式接触 pair 倍率1、倍率0.3下重放。
- 倍率1和未改环境的物体轨迹最大坐标差：`0.0 m`。
- 倍率0.3相对倍率1的物体轨迹最大坐标差：`0.03118909463990377 m`，实际求解器接触摩擦与设定一致。
- 同一初态，xyz 接口倍率1与0.6的一步末端最大坐标差：`0.004201148801643684 m`。
- 接触状态保存/恢复后，2步控制序列重复执行的 pixels、声明传感、qpos、qvel、ctrl、contact friction/force、success 逐项一致；规划 RNG/缓冲被恢复；4步离线回放单列，未记为在线免费反馈。
- 完整快照 pickle 写入/读取检查通过。首轮检查暴露了 variation space 内 lambda 不可 pickle，已改为只保存/恢复 variation 数值与 RNG 状态，再通过检查。

上述位移差是干预生效的工程证据，不是恢复器性能、任务可完成性或试探价值证据。这个脚本用真实物理但没有官方权重、HDF5 或论文成功率评估。

## 尚未真实验证

1. 官方 Cube 权重加载、RGB预处理和前向的完整入口；匹配官方 HDF5 的实际内容/哈希。
2. 官方权重下的一次真实适应、CEM闭环和记录读取；本次只检查其契约与数学路径。
3. Mac/MPS 和 Linux/CUDA 的模型路径；CPU权重路径也未运行。
4. 开发案例下正常基线能力、扰动幅度的可恢复性、E1修正偏好、E2诊断/恢复收益。
5. 本轮没有任何正式 Cube 成功率；下表故意留空。

| 条件 | Frozen | R-I | R-D | R-U |
| --- | --- | --- | --- | --- |
| N | — | — | — | — |
| I | — | — | — | — |
| D | — | — | — | — |

## 补验入口和停止条件

按 [教程](../../le-wm/README_cube.md) 安装 Cube extra，将本地数据/权重绝对路径填入 cube-local.yaml。然后：

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml doctor
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml cases --out runs/cube/cases.json
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage e1 --split dev --limit 1 --smoke --cases runs/cube/cases.json --out runs/cube/smoke
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/smoke --out runs/cube/smoke-summary
```

只有真实权重 smoke 的 STATUS=complete、12分支保存完整、非Frozen确实有更新、回放通过后才进入 E0。若 smoke 案例在适应前已经成功，用 manifest 内另一个 dev case 的 `--case-id` 重跑到新目录。若 E0无接触响应、基线失败或任务不可恢复，停下检查，不启动正式 E1/E2。后续研究继续条件见教程。
