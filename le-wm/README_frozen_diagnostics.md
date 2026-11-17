# LeWM 冻结 PushT 研究工具

这组脚本只做冻结模型的规划、记录和离线诊断；不训练、不修改权重，也不改变官方 `config/eval/pusht.yaml` 的默认行为。当前代码只在源码层面完成，下面的服务器命令是待执行步骤，不能把静态检查当作实验结果。

## 本地核查结果与调用链

本仓库当前分支为 `research/frozen-diagnostics`，HEAD 为 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，同级依赖仓库为本地 `../stable-worldmodel`，实际 HEAD 为 `21446f1ede6d5284e981bd7b47f432b994e6d812`，且该依赖工作树已有修改。它不是此前参考的 `464f97f51a287f4d16e0caad02e8a9fe87c3847c`；本阶段没有强制切换、覆盖或清理这些修改，服务器部署时必须把实际依赖提交和差异记录下来。没有发现项目内更深层的 `AGENTS.md`。依赖版本没有被脚本强制切换。

在线路径是：`research_cem.py` → `stable_worldmodel.data.HDF5Dataset` → `World.evaluate_from_dataset` → `WorldModelPolicy.get_action` → `LoggingCEMSolver.solve` → 加载对象中的 `get_cost`。`load_pretrained`/`AutoCostModel` 可能返回 `stable_worldmodel.wm.lewm.lewm.LeWM`；脚本会打印实际模块类型和源码路径，不能只根据本仓库的 `jepa.py` 推断修改已生效。

关键形状如下：PushT 图像经过 `HWC uint8 → CHW float32 → ImageNet Normalize → 224×224`；策略输入通常是 `(B, 1, C, H, W)`，CEM 候选是 `(B, N, H, action_dim*action_block)`，模型代价是 `(B, N)`，返回动作是 CEM 更新后的均值 `(B, H, action_dim*action_block)`。`WorldModelPolicy` 再把前 `receding_horizon` 个模型步展开成连续环境动作并反标准化。`CEMSolver` 的 `var` 更新使用 `std`；`costs` 是精英平均代价，不是返回均值的代价。研究求解器会额外评分返回均值并单独写入日志。

## 文件

* `research_common.py`：案例清单、运行元数据、冻结模型和设备同步辅助函数。
* `research_cem.py`：`generate-cases` 固定案例；`run` 执行一个预算条件，记录逐回合结果、逐轮 CEM 日志和可选快照。
* `research_diag.py`：读取快照并生成审计表与图。完整 `J_enc/J_task` 需要服务器上的真实 PushT 重放表；脚本不会把缺失值填成零。
* `config/research/pusht_budget.yaml`：研究默认值；N 和 I 是唯一计划改变的预算字段，`topk` 需明确保持规则。

## 服务器安装与目录

在两个仓库的父目录执行，使用与当前源码兼容的 Python 3.10/3.11 环境。不要在本地运行以下命令，也不要在服务器之外下载权重或数据。

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision
uv pip install -e '../stable-worldmodel[train,env]'
uv pip install stable-pretraining hydra-core scikit-learn pandas matplotlib pyyaml
export PYTHONPATH="$PWD:$PYTHONPATH"  # le-wm 当前没有独立 pyproject
uv pip check
export STABLEWM_HOME=/data/lewm/stablewm
export SDL_VIDEODRIVER=dummy
```

权重、数据和输出均通过参数或 `STABLEWM_HOME` 指定；没有写死本地绝对路径。服务器应先下载公开 PushT 权重与数据，再记录下载 URL、文件校验值和目录清单。

## 推荐执行顺序

1. **预检（待执行）**：确认两个仓库的 `git rev-parse HEAD`、`python research_cem.py --help`、PyTorch 设备和 `stable_worldmodel.__file__`。模型预检命令会打印实际类型、源码路径、参数量并断言所有参数冻结：

   ```bash
   python research_cem.py preflight --policy pusht/lewm --cache-dir "$STABLEWM_HOME" --device cuda
   ```
2. **案例清单**：

   ```bash
   python research_cem.py generate-cases --dataset pusht_expert_train --cache-dir "$STABLEWM_HOME" --num-cases 20 --goal-offset 25 --case-seed 123 --output runs/cases.json
   ```

   清单记录 `case_id, episode, start_step, goal_offset`。所有预算条件必须复用这一个文件；案例种子、环境种子和规划种子分开记录。

3. **低预算冒烟**（先用 1–2 个案例）：

   ```bash
   python research_cem.py run --cases runs/cases.json --policy pusht/lewm --dataset pusht_expert_train --cache-dir "$STABLEWM_HOME" --device cuda --num-samples 30 --iterations 2 --topk 3 --output runs/smoke --save-snapshots
   ```

   过关条件是加载类/路径正确、权重严格加载、设备可用、动作形状和反标准化无异常、同种子重复动作一致。

4. **预算条件**：只改变 `--num-samples N` 与 `--iterations I`，固定 H、K、`action_block`、`eval_budget`、warm-start、权重、预处理和案例清单。推荐少数组合，例如 `(30,2),(100,10),(300,30)`；每个条件使用不同的规划种子重复，但分析时先按案例汇总，不能把重复种子当成独立案例。

   ```bash
   python research_cem.py run ... --num-samples 100 --iterations 10 --topk 10 --output runs/N100_I10 --save-snapshots
   ```

5. **绘图**：

   ```bash
   python research_diag.py plot --run runs/N100_I10
   ```

   `budget_success_time.png` 报告成功率、回合时间和近似候选计算量（成功率误差条为 95% 正态区间）；`cem_cost_by_iteration.png` 区分每轮最小代价、精英均值和返回均值；`audit_cost_scatter.png`（有审计数据时）比较代价。多条件可一次传入多个 `--run`，单次运行的图只作检查，不能报告跨条件结论。

## CEM 日志和快照

`cem_iterations.jsonl` 每行包含 `plan_id、iteration、candidate_count、horizon、topk、iteration_min_cost、elite_mean_cost、distribution_std_mean`。`return_mean` 行给出更新后均值的独立预测代价。候选快照保存动作序列、每个候选的 `J_pred`、精英索引、更新后的均值和 `std`，并尽量保存在线状态/目标图像；诊断采样 RNG 是独立的 NumPy 生成器，不会改动规划器的 torch RNG。正式计时应关闭 `--save-snapshots`，并单独报告诊断模式开销。

## 候选审计和状态恢复限制

PushT 的 `_set_state` 只接收七维观测，内部还会推进一个物理步，不能声称精确恢复完整物理快照。服务器必须先做“固定 reset seed + 重放已执行动作前缀”或完整模拟器快照的重复恢复检查，比较位置、线速度、角速度和渲染图像，并核对首次规划数据集图像与恢复后的图像；任何不一致都应停止对应审计。只有通过后，才把真实重放表（每个候选包含 `plan_id,candidate_id,J_enc,J_task,endpoint_success,ever_success`）交给：

```bash
python research_diag.py audit --run runs/N100_I10 --replay server_replay.jsonl --keep 10
```

审计必须从同一物理起点执行完整 `H × action_block` 个连续动作，不重新规划、不重复同一 block 动作，使用在线相同的反标准化、渲染和图像预处理。所有代价“小为好”：`J_pred` 是模型终点表征与目标表征的 SSE，`J_enc` 是真实终点图像经同一编码器后的 SSE，`J_task` 是明确的 PushT 任务误差；另外记录官方成功和途中成功。被选候选相对已审计候选集中实际最好者只能称为“已审计候选集上的选择损失”。

## 输出与验收

每个运行目录应保存 `run_metadata.json`（代码版本、配置、案例文件、模型身份、冻结断言）、`episodes.jsonl`（成功、环境步数、规划次数、求解/回合时间和候选预测近似计算量）、`cem_iterations.jsonl` 及可选 `snapshots/`。运行未完成时保留目录并标注未完成，不静默覆盖已有结果。

尚未执行的运行时验证包括：模型实际类与权重兼容性、CUDA/MPS 前向、日志开关是否保持同种子动作、返回均值评分、动作时间对齐、PushT 状态恢复和候选重放一致性、不同预算案例一致性，以及正式计时边界。当前已完成的检查仅包括源码阅读、配置/调用链核对、Python 语法编译和 Git 差异审查；没有成功率、耗时或研究结论可报告。
