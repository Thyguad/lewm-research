# Cube 执行监督的动作块价值

本入口实现 `cube-prefix-value-v1`：冻结 LeWM，训练一个小网络评价“执行前 5 步动作，再由固定 L2+CEM 续接，在剩余预算内得到什么结果”。当前是已实现、待本机真实运行的方法，不预填成功率。旧目标集合入口和 PushT 入口保留。

## 方法与对照

网络输入只有当前 RGB 特征、目标 RGB 特征、实际裁剪后的 5×5 命令、冻结 LeWM 的一步动作块预测和剩余步数。两层 256 隐藏单元 MLP 输出 [0,1] 价值。`value-no-future` 去掉预测特征；两个头从头训练，使用相同采集数据和按来源轨迹采样的 minibatch。训练和小头评分使用 CPU float32；LeWM 支持配置选择 CPU/MPS/CUDA。没有新增框架、自动下载或在线测试更新。

每次使用原始 CEM 300 候选、10 轮、30 精英，在归一化动作空间搜索 25 步。通过只读观察接口保存每轮较好的候选，按前缀质量与多样性构造至多 16 个候选，保留最终均值。另含有限 xyz 编辑与夹爪命令延迟一控制步的编辑；不读真实接触阶段。所有动作先逆标准化再裁剪，实际执行的命令与价值网络输入一致。

四种方法都使用相同候选构造与额外前向预算：

| 方法 | 最后执行哪个前 5 步 |
| --- | --- |
| `cem-mean` | 原 CEM 最终均值；额外构造候选池但不改变原动作，是匹配计算开销的基线 |
| `pool-l2` | 同候选池中预测 25 步终点 L2 最低的候选 |
| `value-no-future` | 去掉预测特征的价值头所选候选 |
| `value` | 完整价值头所选候选 |

每次只执行选中候选的前 5 步，随后重新观察。不会再次平均不同候选。价值网络不读取未执行的后 20 步；同一状态、目标、前缀和剩余预算的价值与计划尾部无关。CEM 均值/局部编辑的额外终点预测、池中前缀预测和评分次数全部计账；相同状态和种子下四方法的候选池相同。闭环分岔后各自重新规划，不能要求后续候选始终相同。

任务仍为同轨迹未来 25 步的位置目标、总预算 50、首次中心距离不超过 40 mm；不检查朝向，不是完整抓取放置或官方 benchmark 的复现。

## 数据与标签

`prepare` 要求已有 idea1 `prepare` 产出的完整 cache。它原样保留其中 12 个开发案例与测试案例，从其训练轨迹中划出 16 条校准轨迹，剩余轨迹产生 128 个训练起点，校准产生 32 个起点。先分整条轨迹，再取起点；同轨迹可有多个训练状态，但不能跨集合。实际源缓存不足时明确报错，不补选测试轨迹。

状态生成可执行 0/5/10/15/20 步固定 L2 warmup；补采轮改由第一轮 `value` 策略生成状态。提前成功/终止的状态记录为跳过，不偷偷替换。每个合法状态取 6 个候选，完整恢复同一模拟器/控制器/RNG 状态，逐个执行前 5 步，再由固定原始 L2+CEM 续接。每个分支使用相同的续接 RNG 种子、空动作缓冲；后续动作来自分支自己的画面。每个分支最多使用该状态剩余的预算，不是额外获得 50 步。

标签中的成功、距离来自独立评价通道。收益规则固定为：

- 成功：`1 - 0.1 * first_hit_step / remaining`，包括前缀执行时间。
- 失败：`0.2 * clip((initial_distance - min_distance) / max(initial_distance, 0.04), 0, 1)`。

它估计的是**固定 L2 续接策略下的价值**，不是最优成功概率。所有真实分支只供离线训练/诊断，在线策略没有模拟器分支或真值访问权限。

训练使用 Huber 回归与同起点排序损失，收益差 ≤0.02 的配对不做排序惩罚。默认 4,000 次更新，每 100 次记录训练/校准曲线，按来源轨迹平均的校准选择后悔值保存最佳权重，平局使用 MSE。同时报告候选池最优收益、所选收益、有非平局配对的状态比例和成功分支比例，避免把“全部候选同样失败”的零后悔值当成学会选择。允许最佳权重来自第 0 步，报告中如实显示。仅更改训练超参数可以复用采集缓存，但评估与所加载头必须使用相同训练配置。

补采只允许一轮、只允许训练集合：选择新头高分、与 L2 排名分歧大的候选，并保留随机覆盖；执行后才生成标签。续接器始终不变。重新训练时合并第一轮和补采数据；校准集保持原样。

## 本机环境与命令准备

沿用 [安装说明](../docs/setup.md) 和已有数据/权重。以下命令在仓库根目录执行，所有输出必须是新目录。先把 `PV_GOAL_CACHE` 改为你实际完成的 idea1 cache 目录，里面应有 `manifest.json` 和 `STATUS.json`。

```bash
# Apple Silicon + GLFW
PV_PYTHON=mjpython
PV_DEVICE=mps
PV_RENDER=glfw

# Linux/CUDA 无显示环境改用：
# PV_PYTHON=python
# PV_DEVICE=cuda
# PV_RENDER=egl
# CPU 可用 PV_DEVICE=cpu，渲染后端仍按本机环境选择。

PV_INPUTS=le-wm/config/research/cube-local.yaml
PV_GOAL_CACHE=runs/goal-set/cache

pv() {
  "$PV_PYTHON" le-wm/research_prefix_value.py \
    --inputs "$PV_INPUTS" --device "$PV_DEVICE" --render-backend "$PV_RENDER" "$@"
}

pv --help
```

`--inputs` 只读取本机数据、权重、设备、渲染四个字段；不会继承旧 E0/E1 的协议。数值配置默认来自 `config/research/ideas/prefix-value.yaml`。路径可跨机器迁移；权重/config 校验和、数据大小及每个使用案例的初态/目标图内容会核对。动作统计复用源缓存；不对约 95GB 数据每次做全文件哈希，复制时仍需确保使用同一数据文件。

## 首先运行：独立的真实冒烟链路

冒烟使用 CEM 12/2/3、总预算 10、每状态 3 个分支、每头 20 次更新。产物带 smoke 标识，不能混进正式采集或测试。真实数据上此链路覆盖权重前向、MuJoCo 分支恢复、一次以上参数更新、四方法输出；测试代码里的合成环境不能代替它。

```bash
pv --smoke prepare --goal-cache "$PV_GOAL_CACHE" \
  --out runs/prefix-value/smoke/manifest

pv --smoke collect --manifest runs/prefix-value/smoke/manifest --split train \
  --out runs/prefix-value/smoke/train-data
pv --smoke collect --manifest runs/prefix-value/smoke/manifest --split calibration \
  --out runs/prefix-value/smoke/calibration-data

pv --smoke train --manifest runs/prefix-value/smoke/manifest \
  --train-data runs/prefix-value/smoke/train-data \
  --calibration-data runs/prefix-value/smoke/calibration-data \
  --out runs/prefix-value/smoke/heads

pv --smoke evaluate --manifest runs/prefix-value/smoke/manifest \
  --trained runs/prefix-value/smoke/heads --out runs/prefix-value/smoke/dev
```

检查各阶段 `STATUS.json` 为 complete，已采集状态有通过的 `replay.json`，`state.json` 中 `budget.accounted=true`，训练曲线有限且至少更新过，评估四方法记录完整且没有缺失。冒烟是否成功完成任务不是继续条件。所有起点都在状态生成时已完成、渲染失败或预算不一致时，保留原因并处理该问题；不要把这种情况写成方法成功或失败。

## 正式开发：采集、训练、补采、评估

以下命令会执行有实际成本的采集，交由本地 agent 启动，不由代码安装或导入自动触发。

```bash
pv prepare --goal-cache "$PV_GOAL_CACHE" --out runs/prefix-value/manifest

pv collect --manifest runs/prefix-value/manifest --split train \
  --out runs/prefix-value/train-r0
pv collect --manifest runs/prefix-value/manifest --split calibration \
  --out runs/prefix-value/calibration

pv train --manifest runs/prefix-value/manifest \
  --train-data runs/prefix-value/train-r0 \
  --calibration-data runs/prefix-value/calibration \
  --out runs/prefix-value/heads-r0

# 新评分器生成状态并挑选高分/分歧候选；仍由固定 L2 续接。
pv collect --manifest runs/prefix-value/manifest --split train --round 1 \
  --trained runs/prefix-value/heads-r0 --limit 32 \
  --out runs/prefix-value/train-r1

pv train --manifest runs/prefix-value/manifest \
  --train-data runs/prefix-value/train-r0 runs/prefix-value/train-r1 \
  --calibration-data runs/prefix-value/calibration \
  --out runs/prefix-value/heads-r1

pv evaluate --manifest runs/prefix-value/manifest \
  --trained runs/prefix-value/heads-r1 --out runs/prefix-value/dev-r1
```

可以用同一个 `evaluate` 入口评估 `heads-r0`，比较补采前后。两者训练数据量不同，这本身不能证明定向补采优于同量随机补采；目前未实现后者的独立实验。

首轮训练/校准分支最多 `160×6×50=48,000` 步；上述 32 状态补采最多 9,600 步。状态生成和每个有效状态额外 10 步的恢复校验另计；提前成功会缩短执行。主要计算成本来自分支续接的 CEM，不能只按物理步数估计耗时。采集完成后，重复训练小头不再执行环境。

长采集建议预先分块，例如 train 的 `--offset 0 --limit 16`、`--offset 16 --limit 16` 等，每块使用新输出目录，`train --train-data` 可以接收多个目录。失败时保留目录，按 `states.jsonl` 定位原因后，以原 offset/limit 重跑整个小块到新目录；不要跳过失败块里先前完成的状态，因为该失败目录整体不会进入训练。训练只接受完整块，拒绝重复 state ID 和篡改后的标签。当前不支持原目录原地续跑。丢弃的失败运行成本保留在各自日志中，需与训练报告列出的完整源运行成本一起统计。

测试集在方法确定后才运行：

```bash
pv evaluate --manifest runs/prefix-value/manifest --trained runs/prefix-value/heads-r1 \
  --split test --dev-run runs/prefix-value/dev-r1 --out runs/prefix-value/test-r1
```

测试要求完全相同的权重、manifest 和协议已有完整开发运行；冒烟/截断开发记录不能冒充。这个入口不会自动根据开发结果改参数或挑测试案例。

## 输出、分析与交接

- `manifest/manifest.json`：源 idea1 manifest、训练/校准轨迹、原开发/测试案例、权重与输入标识。
- 采集目录 `states.jsonl`：全部请求状态的完成/跳过/失败记录、预算与耗时。
- `states/<id>/inputs.npz`：部署允许的特征、候选前缀、剩余预算。`labels.json` 独立保存真实收益；分支 `*-evaluation.json` 保存评价真值，`*.npz` 保存图像。
- `heads*/curves.jsonl`、`training.json`、`heads.pt`：学习曲线、按验证选择的更新步、参数量、更新数、耗时、额外数据成本和头权重。
- `dev*/episodes.jsonl`：真实动作、距离、成功、终止、计算账本。每回合有单独的观测、评价与决策记录；`restore-checks.jsonl` 单列离线校验成本。
- `summary.json`、`comparison.md`：完整性、全部/起点未成功子集、案例内先平均规划种子的配对差值与区间。失败或缺失不静默丢弃；不完整方法不报告一个伪完整成功率，同时给出成功率上下界和配对排除清单。

独立分析不需要数据、权重或 GPU，可分析失败/中断评估：

```bash
python le-wm/research_prefix_value.py analyze --run runs/prefix-value/dev-r1 \
  --out runs/prefix-value/dev-r1-analysis
```

正式前固定：manifest、动作/成功协议、CEM/候选池、收益标签、训练配置、补采状态范围、规划种子。开发案例已被查看，不作为最终测试证据。继续投入看闭环成功、失败惩罚步数和距离；价值网络若只赢原均值但不赢 `pool-l2`，收益可能来自最后选择方式；不优于 `value-no-future` 时，不声称世界模型预测带来增益。没有候选能产生有效动作时，评分头本身无法创造它。

## 本次工程验证边界

新增测试覆盖 CEM 原均值保持、实际前缀/尾部隔离、动作编辑范围、价值损失与真实 PyTorch 更新、分组数据隔离、完整状态恢复的调用路径、预算、权重选择/加载、一次补采、失败与缺失汇总。集成测试注入合成数据/物理环境，**不是 MuJoCo 或官方权重验证**。本次没有启动正式采集、LeWM 训练或 CUDA/MPS 实验。

```bash
python -m pytest -q le-wm/tests/test_prefix_value.py le-wm/tests/test_goal_set_planning.py
python le-wm/tests/test_research_contracts.py -v
```

实际物理/权重/设备是否可用，以本机上面的真实冒烟结果为准。
