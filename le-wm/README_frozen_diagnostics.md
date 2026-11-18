# LeWM 冻结 PushT 规划诊断

这组脚本冻结世界模型，只研究规划与诊断，不训练、不修改模型权重。支持服务器 CUDA、Apple Silicon MPS 和 CPU。Mac 用户先阅读 [Mac 安装与实验教程](README_macos.md)，使用 `config/research/pusht_mac.yaml`；本文保留通用协议和服务器示例。CPU 回归与小流程集成已经验证，真实预训练权重实验和 MPS 实机验收仍需在目标机器执行。

## 修复范围与协议

- 显式 `world.set_policy(policy)`；修复 `configure` 的 NumPy 引用和 `torch.device` 同步。
- warm start 保留剩余 `H-K` 个模型动作块，尾部补零至 H；`K=H` 的空余量也能开始下一次规划。
- 配置优先级：代码默认值 < YAML < 显式 CLI；未知字段、非法参数、重复案例会报错。
- 正式计时默认关闭诊断。`solver_seconds` 在正式模式覆盖整个求解过程，包括采样、模型评分、top-k、均值/标准差更新和结果转 CPU；不包含策略预处理、环境执行和后续日志写盘。`episode_seconds` 是整个上游评估调用的墙钟时间，不能当作求解延迟。首轮冷启动包含在计时内，没有暗中预热或扣除。
- 诊断模式额外评分每轮**更新后的均值**，最终均值才是实际返回计划。候选评分、额外均值评分、快照写盘和离线真实回放分别计数/计时。诊断运行不用于正式耗时图。
- RNG：所有设备统一使用独立的 CPU `torch.Generator` 采样，诊断候选采样使用另一个 NumPy RNG。它是明确的研究基线，**不是上游 CUDA CEM 的逐位复现声明**。默认固定 `topk=10`，仅改变 N 和 I；H、K、动作块和回合预算均固定。
- `case_id` 由 episode/start/offset 决定；候选主键为 `(run_id, case_id, plan_id, iteration, candidate_id)`。每例/每次规划独立快照目录，禁止覆盖旧结果。
- 结果目录有 `STATUS.json`：running / failed / complete。审计与绘图拒绝未完成、旧版 v1 或不兼容结果。旧结果缺少必要上下文，必须重新采集，不能靠改字段升级。

调用链：`research_cem.py → World.evaluate_from_dataset → WorldModelPolicy → LoggingCEMSolver → 实际加载模型.get_cost`。当前实现针对仓库内的 `stable_worldmodel.wm.lewm.lewm.LeWM`。每次记录模型类型、权重/config SHA256、安装源码 SHA256、研究脚本 SHA256、依赖版本、设备与运算配置、数据集 SHA256 和实际使用的案例；不能仅根据顶层 `jepa.py` 推断运行代码。

## 文件

| 文件 | 用途 |
| --- | --- |
| `research_common.py` | 清单、身份校验、状态文件和日志 |
| `research_cem.py` | 固定案例、模型预检、正式收集与诊断收集 |
| `research_runtime.py` | 权重/预处理、PushT 重建、状态校验和任务指标 |
| `research_device.py` | 设备选择、模型 float32 输入、无窗口模式与设备元数据 |
| `research_diag.py` | 候选回放审计、固定状态重规划、配对检查和绘图 |
| `tests/test_research_contracts.py` | 不加载模型/仿真器的配置与数据契约测试 |
| `tests/test_research_server.py` | 跨平台 CPU 张量回归测试，无需权重/数据 |
| `tests/test_research_replay_server.py` | 跨平台真实 PushT 回放测试，无需权重/数据 |
| `tests/test_research_devices.py` / `test_research_lewm_devices.py` | 设备边界、分批评分、CPU 与条件 MPS 测试 |
| `tests/test_research_pipeline.py` | 本地小模型/小数据的完整 CPU 流程测试 |

## 服务器部署

仓库本身包含 `le-wm/` 和 `stable-worldmodel/` 两个子目录，不需要另找同级 Git 仓库。下面从本仓库根目录开始，在服务器上执行。使用 Python 3.11；PyTorch/CUDA 构建需与服务器驱动相容。

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision
uv pip install -e './stable-worldmodel[train]'
uv pip install scikit-learn scipy matplotlib pyyaml pygame 'pymunk>=7,<8' shapely opencv-python pillow
uv pip check
export STABLEWM_HOME=/data/lewm/stablewm  # 换成服务器可写目录
export SDL_VIDEODRIVER=dummy
cd le-wm
python -c "import stable_worldmodel as s; print(s.__file__)"
python tests/test_research_contracts.py -v
python tests/test_research_server.py -v
python tests/test_research_replay_server.py -v
uv pip freeze > server-environment.txt
```

安装包版本不是已经验证的锁定环境：先通过以上服务器测试，再固定该环境。若导入/仿真 API 不兼容，应处理版本兼容问题，不能跳过回放校验。

随后在服务器准备权重及数据：

- 权重目录：`$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht/`，其中有 `weights.pt` 与 `config.json`。也可用 `--policy /绝对路径/weights.pt` 指向同一对文件。研究加载器只查找已有文件，不自动下载模型。
- 数据：`$STABLEWM_HOME/datasets/pusht_expert_train.h5`。`--dataset` 是相对 datasets 目录的名称，**不带 `.h5`**，使用本仓库 HDF5Dataset 的接口。
- 权重、预处理和数据来源必须符合预期；每次运行的完整数据 SHA256 扫描与 scaler 拟合在计时区间之外，可能耗时。scaler 沿用原评估方案拟合整份给定数据，不是另行训练世界模型。

```bash
python research_cem.py preflight --policy quentinll/lewm-pusht --device cuda
python research_cem.py generate-cases --dataset pusht_expert_train \
  --num-cases 20 --goal-offset 25 --case-seed 123 --output runs/cases.json
```

每个案例来自不同 episode；使用的是给定数据集中的开发案例。默认 `pusht_expert_train` 不能用于宣称未见任务泛化。案例种子与规划种子分开；环境优先使用数据集 seed，缺失时才使用配置中的 environment_seed 派生值，实际 reset seed 会记录。

## 先做真实权重的配对冒烟

同一预算、案例、种子，分别关闭/开启诊断。两次输出目录都必须尚不存在。

```bash
python research_cem.py run --config config/research/pusht_budget.yaml \
  --cases runs/cases.json --limit-cases 2 --num-samples 30 --iterations 2 \
  --planning-seed 42 --output runs/smoke_formal
python research_cem.py run --config config/research/pusht_budget.yaml \
  --cases runs/cases.json --limit-cases 2 --num-samples 30 --iterations 2 \
  --planning-seed 42 --save-snapshots --max-snapshot-plans 2 --output runs/smoke_diag
python research_diag.py check-pair --formal runs/smoke_formal --diagnostic runs/smoke_diag
python research_diag.py audit --run runs/smoke_diag --output runs/smoke_audit
python research_diag.py plot-diagnostics --run runs/smoke_diag \
  --audit runs/smoke_audit --output runs/smoke_figures
```

`check-pair` 比较环境初始化和**整段实际执行动作的逐位一致性**。`audit` 对所有保存候选各回放两次，检查重建起点、整段物理轨迹、终点像素与成功标记，并重算 J_pred 对照收集值。默认保存每次迭代 2 个最好候选、2 个非最好随机候选和 1 个更新均值；可通过 `--keep-best/--keep-random` 控制数量。默认只保存每例第一个规划点，示例用两个规划点覆盖非空动作前缀。

目标机器验收至少需要：依赖路径正确、参数冻结、张量测试通过、真实 PushT 测试通过、真实权重的 check-pair 通过、audit complete。开发环境的小模型测试不代替目标机器和真实权重验收。

## 正式预算实验

保持所有配置相同，只有 N/I 和各条件共享的一组规划种子变化；`topk` 固定为 10。不要给正式运行加 `--diagnostics` 或 `--save-snapshots`。

```bash
for seed in 42 43 44; do
  for budget in '30 2' '100 10' '300 30'; do
    read -r n iterations <<< "$budget"
    python research_cem.py run --config config/research/pusht_budget.yaml \
      --cases runs/cases.json --num-samples "$n" --iterations "$iterations" \
      --planning-seed "$seed" --output "runs/formal_N${n}_I${iterations}_S${seed}"
  done
done
python research_diag.py plot --run runs/formal_N*_I*_S* --output runs/budget_figures
```

绘图会核对案例/种子配对、权重、数据、预处理、代码、设备、依赖和固定配置；重复行、缺失案例、不同 topk、诊断运行混入均报错。输出 `budget_summary.json` 和 `budget_success_time.png`。

独立单位是案例（每个来自不同 episode），不把多个规划种子视作额外案例。一个种子时使用 Wilson 95% 区间；多个种子先求每例平均成功率，再使用独立有界案例均值的 Hoeffding 95% 区间，较保守，但全成功/全失败时不会错误地产生零宽区间。图是描述性比较，不是跨预算差异的显著性检验。

## 审计定义与限制

每个快照保存模型实际接收的归一化输入、完整候选动作块、返回均值、规划 RNG 状态、warm start、reset seed/options、初始/目标 state、已执行原始动作前缀、物理指纹、环境渲染和在线观测图像。

PushT 的七维 `_set_state` 不是完整模拟器快照。这里重现相同 reset、上游的初始 state/goal 设置顺序和已执行动作前缀，再校验 agent/block 的位置、线速度、角度、角速度、力、力矩、质心和渲染。审计中每个候选都独立重建并回放两次。此检查能发现可观测重建误差，不构成对模拟器全部隐藏接触缓存的数学证明；任何检查失败均终止审计，不能通过 `--allow-inexact-state` 绕过。

首次规划时，上游可能把数据集图像直接作为观测，它与重建环境的渲染不完全一致。`observation_render_pixel_max_abs` 专门暴露这个问题，不会悄悄替换输入。若不为零，必须单独分析该图像域偏差，不能把所有预测/真实差异都解释为动力学误差。重建环境与收集时环境渲染之间仍必须通过严格一致性检查。

| 指标 | 定义 |
| --- | --- |
| J_pred | 冻结模型预测的 H 步终点表征到同一目标表征的 SSE |
| J_enc | 真实执行 H × action_block 个连续动作后，终点图像编码到同一目标表征的 SSE |
| J_task | `max(位置误差/20, 周期角度误差/(π/9))`，明确的研究任务代理代价 |
| official_state_distance | 上游 `eval_state` 的原始七维距离，另行记录，不混称 J_task |
| endpoint_success / ever_success | 终点官方成功 / 此候选动作执行期间任一步成功 |
| selection_loss_on_audited_set | 最终返回均值的 J_task 减去已审计候选中最小 J_task |

动作块展开为连续的二维动作序列，使用在线相同 StandardScaler 反标准化；不会把一个动作重复 action_block 次。审计执行全 H 个块，不重新规划。在线只执行前 K 个块，因此全 H 候选审计与在线闭环成功率回答不同问题。

候选主键包括 iteration，实际返回均值独立保存并标记 selected；同一轮精英代价均值不等于该动作均值的代价。Spearman 相关按每个案例/规划点分别计算；常数序列或样本太少返回 null。候选来自有选择的子集，选择损失不是全局 regret，不能把各候选当独立任务计算置信区间。`plot-diagnostics` 也逐个案例/规划点绘图，不串接无关曲线。

默认重建容差是物理绝对误差 1e-7、像素差 0；J_pred 重评分 atol/rtol 为 1e-4。容差可显式修改并记录，但调大容差必须说明原因，不能据此宣称精确恢复。若迁移机器，可通过 audit 的 `--policy/--cache-dir/--device` 指定路径/设备；权重、源码和相关依赖版本仍须匹配，且重评分与重复回放仍须通过。

离线报告包含真实动作步数（含重复前缀回放）、环境重建次数和总耗时；reset/内部初始化物理子步不计入动作步数。所有离线成本独立于在线规划成本。

## 固定同一个起点比较预算

闭环预算改变后，后续访问状态也会改变。要隔离搜索预算的影响，可选择诊断运行中的一个 `snapshot.json`，冻结其模型输入、真实起点和 warm start：

```bash
# 将下一行替换为上面运行实际产生的快照路径。
SNAPSHOT=runs/smoke_diag/snapshots/实际case_id/plan_000000/snapshot.json
python research_diag.py replan --run runs/smoke_diag --snapshot "$SNAPSHOT" \
  --num-samples 100 --iterations 10 --topk 10 --output runs/replan_N100_I10
python research_diag.py audit --run runs/replan_N100_I10 --output runs/replan_audit
```

默认从保存的规划 RNG 状态开始；显式 `--planning-seed` 可研究新的随机搜索。不同 N 的采样张量形状不同，不保证所有候选逐一嵌套。`replan` 结果标为 counterfactual，只比较固定状态上的候选质量，不产生闭环成功率，也不能混入正式预算图。先使用与源快照相同的 N/I/topk 重规划，检查返回动作与 J_pred 一致，再改变预算。

## 本次校验边界

- 已执行：语法与 CLI 检查、11 项轻量契约、7 项设备/分批 CPU 回归、5 项原求解器回归、1 项实际 LeWM 小模型 rollout、1 项真实 PushT 回放、1 项完整小流程集成，共 26 项通过。3 项 MPS 专项测试在 Linux 跳过。
- 未执行：Mac 安装与 MPS 实机验收、官方预训练权重前向、真实数据实验、训练、GPU 耗时和成功率测量。小流程使用临时生成的测试权重/数据，不是论文复现结果。
- 目标机器验收通过后再启动大预算；保留失败目录及 STATUS 错误信息，用新的输出目录重跑。详细版本与 Mac 验收步骤见 [Mac 教程](README_macos.md)。
