# Cube E0–E2：运行、审计与继续条件

**正常基线先行：**见 [Cube 参考复现核查](README_cube_reference.md)。独立入口 `cube_reference.py` 不使用下面的研究 planner；两例 N 回归已跑完，正式 50-case 和 E1/E2 尚未启动。

本入口建立实验基础，不宣称 I 与 D 已混淆、短试探有效或形成了新的主动辨识方法。
基于 `a1158f1b9b8281c8afa0ddbec9d0a884648d58a2`：核查时 main 无后续 Cube 代码/结果、无 AGENTS.md、无开放 PR。历史 PushT 结果及所有原运行入口保留。

## 1. 已实现的协议和边界

- `research_cube.py` 是独立入口；`research_cem.py/research_diag.py` 仍是 PushT 专用，保留二维动作断言和原行为。
- **只支持 Cube-single 专家轨迹片段目标**：从同一 episode 取起点和 25 步后图像/物体目标，最多 50 个环境控制步。成功为任一步物体中心距目标 ≤0.04 m，沿用环境规则，不检验朝向。不是完整抓取放置协议，也不是官方 OGBench 多任务成功率。初始即成功的案例保留并单独计数。
- 匹配本仓库 `config/train/data/ogb.yaml` 和 `config/eval/cube.yaml` 的 `ogbench/cube_single_expert`。官方 `quentinll/lewm-cube` 模型配置为 RGB224、192 维 latent、25 维 action input。运行时检查本地 config/数据形状并记录 SHA-256；本轮没有下载/验证本地官方权重或数据内容。
- 环境动作是 5 维 `[-1,1]`：xyz 位移、yaw、夹爪相对闭合量。环境转换范围是 xyz ±0.05 m、yaw ±0.3 rad、夹爪 ±1，另有 workspace 和夹爪位置限制。正夹爪命令趋向闭合。
- 每动作块包含 **5 个不同的底层命令**，不是单动作重复。默认规划 5 块、只执行 1 块即 5 步后重规划；与旧 Cube 配置执行 5 块的设置不同，作为明确的新研究协议。控制周期 0.05 s，物理步 0.002 s；实际值写入记录。
- 研究 CEM 在数据集标准化动作空间采样，再映射回有界环境命令；使用终点 latent 距离评分，执行最后一轮分布均值。预测保留最多 3 块上下文。此实现不是未经修改的官方 CEM 复现；所有分支共用它。

### 扰动和修正方向

```
规划变量 c → clip[-1,1] → 环境 xyz 真值缩放 → clip[-1,1]
             → 环境物理单位转换 → IK/PD/workspace 限制 → 物理执行
模型输入 c → clip[-1,1] → 估计 xyz 缩放 → clip[-1,1]
             → (effective action − 离线 action mean) / scale → 5步展平25维 → LeWM
```

规划端估计有效动作，只用于预测；实际发送的仍是 c，不再次逆映射。yaw 和夹爪不参与缩放。N 为恒等映射；I 只改变位移增益；D 只改变夹爪四个碰撞接触垫与 `object_0` 的滑动摩擦。变化从回合起点生效，回合内固定。

D 使用明确的 pad–object contact pairs。当前 OGBench pad priority=1、cube priority=0；从编译后的高优先级 pad 复制 condim、solref、solimp、扭转/滚动摩擦，margin/gap 按双方最大值处理，仅缩放两个切向摩擦分量。不会更改 cube–floor 摩擦。遇到不匹配 priority、未知 geom 或已有冲突 pair 会报错。`audit` 同时比较 N、I、D、D(scale=1)，验证求解器中实际 contact.friction 和轨迹响应；无接触时结论是 inconclusive。

### 恢复器

| 方法 | 更新对象 | 初始化与训练 | 默认更新量 |
| --- | --- | --- | --- |
| Frozen | 无 | 原始模型 | 0 |
| R-I | 1 个公共 xyz 增益 | 增益=1，log-gain 有界坐标搜索，范围 [0.25,4] | 20 轮，每轮至多2个新候选 |
| R-D | 小型线性残差输出矩阵 | 固定随机 latent 投影 rank=4，拼接块平均动作5维和常数；矩阵初始0；ridge 拟合 | 1 次线性求解，1920 个参数 |
| R-U | 增益＋同一残差矩阵 | 变量投影联合优化：每个增益候选重新求残差最优解，再比较联合目标 | 20 轮，1921 个参数 |

全部冻结 encoder、action encoder、predictor、projector、pred_proj，始终 eval/inference mode。
损失为所有可用 5 步转移的 latent 均方误差，加 `λ‖W‖² / latent_dim + λ(log gain)²`，λ=0.01。
拟合使用完整历史的每个可用滑动 5 步窗口（单块条件，非独立统计样本）；多步规划仍用最多3块自回归上下文。
每回合、每分支新建适应对象，没有保留的优化器、历史或持续学习。R-D 一次闭式求解与 R-U 多次求解的计算量不同，记录实际次数和耗时，不声称计算预算已匹配。没有额外预训练适应模块。

### E2 和信息边界

- 被动诊断使用从任务开始到决策点的**全部允许历史**；继续任务、固定试探、随机试探获得相同额外交互上限。已经成功或环境截断则提前终止，并记录实际步数。
- 动作库只有 x/y/抬升小幅往返；夹爪相对动作保持0（请求当前开度），不会发出松手命令。由最近已发夹爪命令标识“commanded_closing/unconfirmed”，**不声称等于真实接触阶段**；这个保守库在两种阶段都合法，不使用接触标签筛选。奇数预算最后一步为零位移，仍计步；回程、重新接近包含在预算中，不免费回位。
- `sensors: rgb` 或 `rgb_proprio`。后者公开关节位置/速度、末端位置/yaw、夹爪开度/速度；明确不公开 gripper_contact/力、物体真实状态、真实缩放/摩擦或变化标签。所有方法获得同一协议的输入。恢复器和规划器目前使用 RGB/动作；诊断器还使用全部声明的传感量。
- 诊断参考拼接完整、保序、补零而不截断的 latent、14×14 RGB空间均值、动作、传感和有效帧掩码，加目标 latent；不是只用末帧或平均历史。采用 dual ridge；它是一个可检查的基线，不是关于不可辨识性的证明。
- 同一诊断器训练/校准架构用于全部采集策略，训练数据包含各策略同样数量的案例和预算。监督标签仅来自训练分区的评价通道。独立 episode 内的规划种子和各策略始终同组；训练分区再按完整案例分 80% 拟合、20% 温度校准。开发集不用来拟合这个诊断器；锁定测试集不参与任何拟合。
- 除变化分类，还学习四种恢复器的**实际后续任务失败损失**，测试选择预测损失最低的恢复器。因此不把分类准确率直接当作任务收益。该方案是 E2 的基础诊断/恢复对照，尚无自适应主动选试探、触发或 E3–E5 系统；正式路径不含 active 占位函数。
- `collect` 在离线环境中分支运行四恢复器以形成监督结果，这些交互成本按分支记录。`evaluate` 每种采集策略只执行诊断器选出的一个真实恢复分支，不能查看其他未执行分支。

## 2. 安装和设备

从仓库根目录执行（Python 3.11/3.12；Apple Silicon 使用原生 arm64 Python）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './stable-worldmodel[cube-research]'
python -m pip check
python le-wm/research_cube.py --help
```

已有 PushT 环境可直接安装 Cube extra，不需要重建现有 PushT 文件。Linux CUDA 用户使用与驱动匹配的 PyTorch 构建；Cube extra 不固定 CUDA 版本。

`--device auto|cpu|mps|cuda[:index]` 使用原有设备解析器，浮点模型输入统一 float32；模拟器和 ridge 求解在 CPU。
`--render-backend auto|egl|osmesa|glfw` 在导入 MuJoCo 前设置；auto 不覆盖外部设置。Mac 通常先用 auto；Linux 无显示先试 egl，软件渲染可选 osmesa（需系统 OSMesa）。
本轮真实验证范围是 Linux MuJoCo CPU/EGL；完整 PyTorch CPU、MPS、CUDA 权重路径均待本机冒烟。不能把“实现了分支”理解为这些设备已经通过实际权重测试。

## 3. 权重、数据和配置

官方来源：[权重](https://huggingface.co/quentinll/lewm-cube)、[数据](https://huggingface.co/datasets/quentinll/lewm-cube)、[配置](https://huggingface.co/quentinll/lewm-cube/raw/main/config.json)。数据须是对应的 Cube-single expert HDF5，不是原始 OGBench 的不同存储格式、Scene 或 multi-view 数据。

以下是你主动准备输入时执行的下载命令，本次实现过程没有运行：

```bash
hf download quentinll/lewm-cube config.json weights.pt --local-dir checkpoints/lewm-cube
hf download quentinll/lewm-cube --repo-type dataset --local-dir datasets/lewm-cube
```

当前官方压缩文件为 `cube_single_expert.tar.zst`（约46.2 GB，解压另需空间），手动确认资源后再下载。解压命令：

```bash
tar --zstd -xf datasets/lewm-cube/cube_single_expert.tar.zst -C datasets/lewm-cube
```

Mac 可先安装系统 zstd。将解压目录内真实 `.h5` 路径写入配置；不要把压缩文件路径当 HDF5 路径。

```bash
cp le-wm/config/research/cube.yaml le-wm/config/research/cube-local.yaml
```

编辑副本的 `dataset` 为解压 HDF5 的绝对路径、`weights` 为 weights.pt 的绝对路径；config.json 必须相邻。按机器设置 device/render_backend。现有配置中的缩放0.6、摩擦0.3、lift_cycle 都是**待开发集验证的起步值**。

下面全部命令使用这个同一配置：

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml doctor
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml cases --out runs/cube/cases.json
```

`doctor` 只检查依赖/本地输入是否存在；不会加载模型或下载。`cases` 流式计算原始离线预训练数据的动作 StandardScaler 统计，按完整 episode 分 dev/train/test，一条轨迹只取一个起点目标，冻结 manifest 和数据哈希。这是对固定预训练离线统计的复用，不是用测试扰动轨迹拟合归一化器；不能据此宣称基础模型未见过源数据。

## 4. 测试与最小真实权重冒烟

```bash
python -m pytest le-wm/tests/test_cube_protocol.py le-wm/tests/test_cube_analysis.py le-wm/tests/test_cube_runner.py le-wm/tests/test_cube_data.py -q
python le-wm/tests/test_research_contracts.py
python le-wm/tests/cube_simulator_check.py --render-backend egl
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage e1 --split dev --limit 1 --smoke --cases runs/cube/cases.json --out runs/cube/smoke
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/smoke --out runs/cube/smoke-summary
```

Mac 将模拟器测试的 `egl` 换为 `auto`；脚本的 auto 不指定 MUJOCO_GL。若只安装物理依赖、没有 torch，可给模拟器检查加 `--isolated-imports`：只绕过 stable_worldmodel 包的重型初始化，仍加载本仓库真实 Cube/spaces/env utils 和 MuJoCo；这不是完整产品入口验证。

smoke 使用 1 个 dev 案例，N/I/D，各4分支，总7步、历史5步、CEM4候选/1轮、适应1轮。要求实际执行适应，初始/历史即成功而没有适应则报失败，需要用 `--case-id` 指定 manifest 中另一个 dev 案例（保持正式 manifest 不变）；不要把 smoke 成功率汇入正式表。检查 `STATUS.json=complete`、12个恢复结果、适应记录、真实权重 SHA、动作/图像可读、回放通过。缺任一项先停止。

## 5. E0 → E1 → E2

### E0：基线和干预

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage e0 --split dev --limit 3 --cases runs/cube/cases.json --out runs/cube/e0-dev
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml audit --cases runs/cube/cases.json --limit 3 --steps 20 --out runs/cube/contact-audit
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/e0-dev --out runs/cube/e0-summary
```

先核对可见目标、起点、动作/渲染和初始成功比例。audit 重放数据集真实专家命令，作为特权离线诊断；不提供给部署策略。若短段没有接触，audit 会 inconclusive，不能认定扰动有效；在开发集检查接触段/增加 audit 步数，保留失败记录。若正常模型没有可测能力，先查兼容/规划/目标；若 D 改了参数却没改运动则停止；任务是否仍可完成需在开发范围检查专家/真实动力学参考，不能用当前工程测试代替。

### E1：共享历史后的修正交叉

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage e1 --split dev --limit 3 --cases runs/cube/cases.json --out runs/cube/e1-dev
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/e1-dev --out runs/cube/e1-summary
```

同条件下四个分支读取完全相同的 `history_sha256`，从同一完整模拟器快照恢复、同一基础模型、适应初态和规划种子开始。默认前10步正常历史，余下40步恢复。所有分支一律在适应边界清空动作计划缓存。不能把分支真实交互加总后说每个策略获得了这些反馈。

若所有修正都无效，先解决恢复能力；若 R-U 同预算已足够好，路由动机减弱；只有错误修正有明显任务代价才继续研究选择问题。

### E2：采集、训练/校准、锁定评估

开发比较（不训练测试诊断器）：

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage collect --split dev --limit 3 --cases runs/cube/cases.json --out runs/cube/e2-dev
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/e2-dev --out runs/cube/e2-dev-summary
```

默认每案例/种子/条件比较 passive(0)、continue/fixed/random(5,10,20)，各运行4种恢复器。此命令工作量明显超过 smoke；先通过前序阶段再执行。以下规模只是命令示例，不是已运行结果：

```bash
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage collect --split train --limit 30 --cases runs/cube/cases.json --out runs/cube/e2-train
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml train --cases runs/cube/cases.json --runs runs/cube/e2-train --out runs/cube/diagnostic
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml lock --cases runs/cube/cases.json --diagnostic runs/cube/diagnostic --out runs/cube/lock.json
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml run --stage evaluate --split test --limit 10 --cases runs/cube/cases.json --lock runs/cube/lock.json --diagnostic runs/cube/diagnostic --out runs/cube/e2-test
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml analyze --runs runs/cube/e2-test --out runs/cube/e2-test-summary
```

至少5个独立训练案例，每个包含 N/I/D；不是5个规划种子。训练拒绝 dev/test/smoke、缺失恢复分支、重复记录和不同模型/协议。若训练过程资源过大先减开发规模测量，不要在测试标签上调配置。正式 E1 也可用 `--split test --lock ...` 单独运行；它是离线四分支比较。测试标签只进入评价文件。

先在开发集固定配置；训练后将诊断器文件哈希一并 lock，再开始测试。冻结项：数据/目标协议、完整 case manifest、开发确定的扰动幅度/候选动作库/固定试探、观测协议、总步数/历史/额外预算、CEM、动作块/执行缓冲、适应损失/正则/迭代、全部种子、模型和config哈希。设备/文件位置/评分分批可以更换但记录实际值；数值可比性仍需按设备补验。

如果完整普通历史已足够，报告边界；如果固定动作同样好，保留简单方法；只有诊断改善进一步减少任务失败、且扣除额外动作后仍获益，才考虑后续主动方法。本轮不实现“可靠估计试探减少错误适应决策损失”的完整算法，也不把主动试探、任务辨识或候选模型选择本身称为创新。

## 6. 输出、预算与状态恢复

每个运行目录拒绝覆盖。更换目录名重试，保留失败记录。

- `metadata.json/model.json/plan.json/STATUS.json`：代码/source hashes、配置、依赖、模型/数据、种子、计划与完成情况。
- `histories/<id>/allowed.npz`：全部允许图像、传感、已发命令、目标；`evaluation.json` 和 metadata 中 `truth` 为独立评价通道。
- `evaluator_snapshot.pkl`：仅可信本地离线快照，包含完整 MjData（warm start/ctrl/activation/mocap/外力等）、IK 自身 MjData/工作数组、控制器/任务字段、扰动、随机数及 variation 状态。前缀快照还保存 CEM RNG/动作缓冲。读取此文件不属于部署算法 API。
- `recoveries/<id>/result.json/allowed.npz/evaluation.json/adaptation.json`：状态、原因、成功/距离/首达时刻、各阶段步数、预测/编码/规划调用及 batch item 数、更新次数/适应/规划/恢复/端到端耗时、适应参数和全部后续真实历史。
- `branch-replay.json`：每个新增历史后的状态恢复检查；前缀检查在 metadata。实际恢复会重新执行短序列，比较 pixels/sensors/qpos/qvel/ctrl/contact/成功，不只检查 qpos/qvel。离线回放步数和次数在 ledger 中单列，不能视为策略免费反馈。在线策略账本是 normal_history+continue/probe+recovery，总和不超 total_steps。
- 时间是仿真等待计算的墙钟时间，不是实时机器人控制指标。单分支计入共享前缀/采集成本，运行总时间含所有离线分支、快照检查、I/O；STATUS 中的 evaluator_actual_environment_steps 是实际执行总数（含检查），offline_replay_steps 单列，不能把拷贝给多个分支的共享账本直接求和当实际总成本。GPU/MPS 在模型输出回 CPU 和求解结束时同步。
- 统计使用完整独立案例：先聚合同案例规划种子，再做配对 case bootstrap。`summary.json` 包含 cross cells、恢复器配对、采集策略配对、budget/收益、N误适应代价、分类 confusion/balanced accuracy/NLL/Brier。失败/缺失显式列出；报告 observed estimate 和 missing-as-failure 分开，不能静默删除。

## 7. 本轮实际验证状态

详见 [工程验证记录](../experiments/2026-11-19-cube-engineering/README.md)。没有正式 Cube 成功率，交叉表无实际运行数据时保持 `—`。后续先跑 doctor、生成 cases、真实权重 smoke，再审查 E0；不要直接启动批量 E2。
