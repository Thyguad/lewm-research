# LeWM Cube 参考复现核查

2026-11-19，`research/cube-e0-e2` / `f28fa97`。未改 E0–E2、PushT、原始 HDF5 或 checkpoint，未训练、未下载数据或权重。本轮只执行两个既有 N 回归例。旧运行证据见 [prior runs](cube_reference_prior_runs.md)。

## 协议与来源

- [论文 v3](https://arxiv.org/pdf/2603.19312v3) 附录 D：Cube 为 300 条候选、10 轮、top-30，5 个动作块，每块 5 步，一次执行整段 25 步；附录 F.1：同一条离线轨迹取当前状态和 25 步后的目标，给 50 个环境步。图中的 74% 不是固定桌面初态的完整抓取放置成功率。
- 核对公开 LeWM `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` 的 [eval.py](https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/eval.py)、[Cube 配置](https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/cube.yaml) 和 [CEM 配置](https://raw.githubusercontent.com/lucas-maes/le-wm/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml)：与本地参考入口/配置一致，CEM 默认是 **30 轮**。本轮主协议 `paper-300x10-top30`；独立可选 `public-300x30-top30`，后者未实跑。既有研究高预算 `300/30/5` 不属于任何一个参考协议。
- 使用本地 vendored `stable_worldmodel` 0.0.6，未更新到远端 main。`runtime.json` 记录 Python、`HDF5Dataset`、`World`、`WorldModelPolicy`、`CEMSolver`、`LeWM` 的实际路径/源码哈希、依赖和设备。这次是 Apple Silicon 上 MPS/GLFW：Torch 2.15.0、torchvision 0.29.2、stable-pretraining 0.1.8、OGBench 1.2.0、MuJoCo 3.3.6、Gymnasium 1.4.0、NumPy 2.4.6、scikit-learn 1.9.1。
- 本地 `quentinll/lewm-cube` 权重 SHA-256：`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`；模型配置 SHA-256：`4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999`。数据为本地 `cube_single_expert.h5`，101,942,558,720 字节；相邻 `source-revisions.json` 记录模型 `b0747c5002e86d2ce8f3cd8178004b97524c587d`、数据 `02a19a67a0dc8c9d6215f89c19e0a597691e152a`。本轮未重新全量哈希 HDF5，也未独立确认这就是论文出图时的权重和数据版本。

## 兼容修复

| 核查 | 发现、修复和验证 |
| --- | --- |
| 数据读取及 scaler | `load_chunk` 在 `frameskip=5` 时把 25 步窗口重排为 `(5,25)`；实际评估的 26 帧窗口甚至会变成 `(5,26)`。随后 `WorldModelPolicy._prepare_info` 将 25/26 维数据交给 5 维 scaler；强行拟合 25 维又使 `inverse_transform` 不接受策略缓存拆出的 5 维。用评估默认 `frameskip=1`，窗口含 26 个 RGB/5D 动作条目，目标在 `t+25`。只用去 NaN 后的 2,000,000 个 5D 数据动作拟合一次。固定候选先按每步 5D 标准化，再按时间拼成 `(5,25)`；参考 CEM 原生在这个空间优化，不对候选二次标准化，也不重复命令。执行由原策略拆成 25 个 5D 向量，逐行用同一个 scaler 反变换。回归测试走实际 HDF5/策略接口。 |
| MPS dtype 与信息边界 | 旧调用栈 `CEMSolver.solve → LeWM.get_cost → info_dict[k].to(device)` 将 `qpos` 等 float64 张量搬到 MPS 后报错。新入口仅给模型 RGB、目标 RGB，以及 `get_cost` 要求存在的 5D action 占位键；模型会在目标编码时删除该键，在初态 rollout 时以当前候选覆盖它。占位不包含过去专家动作。模型浮点张量在设备搬运前使用 float32；`qpos/qvel` 保持原 float64，真值、接触力与条件只用于仿真初始化或评价。见 `reference-20261119-probes/interfaces.log` 的调用栈、CPU/MPS 测试和固定候选对照。 |
| World 与平台 | 公共 `eval.py` 调用的 `World.evaluate(..., goal_offset=..., video=...)` 与本地库的 dataset API 不匹配；用实际 `World.evaluate_from_dataset`，复用原 Cube callables、`terminate_at_goal=True`、100 步环境上限、50 步评估、关闭视频。保留 task reset → `set_state` → `set_target_pos` → 数据集首帧/目标帧覆盖 → 后续实时 RGB；不把旧研究的 `data_collection` reset、`pre_step/post_step` 或派生环境 seed 混进来。HDF5 无 `seed`/variation 列，参照入口让环境用默认 reset，日志保留实际 reset seed、t=0 qpos/qvel/ctrl。macOS 用 `mjpython` + GLFW，动态库路径见下。两例实跑检查了 t=0、每一步和重规划时刻。 |

### CEM 剩余差异

新入口直接调用本地参考 `CEMSolver.solve`，只旁路记录逐轮最优、最佳样本和最终均值的重新评分：

| 项目 | 参考入口 | 原研究入口 |
| --- | --- | --- |
| 评分前裁剪 | 不裁剪；直接评分标准化 CEM 候选。反标准化后也不额外裁到 `[-1,1]`；当前 OGBench 的 `unnormalize_action` 线性映射，随后仍有 workspace/夹爪控制限制 | 反标准化后裁剪再重标准化评分 |
| Elite / std | 主协议 30；Torch `std(dim=1)`（correction=1），无下限 | 默认 5；NumPy `std`（ddof=0），下限 `.03` |
| 均值 | 每轮第一个样本是当前均值；最终执行更新后的均值。原 `outputs['costs']` 是最后一轮 elite 平均，不是均值计划的 cost | 当前代码也保留均值样本并执行最终均值，但采样/裁剪/更新不同 |
| 缓存与 warm start | 25 个环境步缓存，实际在步 0、25 重规划；`warm_start=True` 但尾部是 `(N,0,25)`，第二次零均值起步，无前次动作保留 | 默认 5 步缓存、重规划重置分布 |
| RNG | 一次运行一个 device-local Torch generator、seed 42，`batch_size=1` 按向量 slot 顺序为每例采样，跨案例/重规划不重置。2 例与 50 例的流不同，不保证跨 CPU/MPS/CUDA 同序列 | 独立研究 Planner 用 NumPy `default_rng`，每 Planner 初始化；同 seed 不是同候选 |

实跑 t=0、25 的命令均逐步等于重评分均值反变换后的 5D 命令。首个计划中 e22d、e1db 各有 6、5 个分量超出 `[-1,1]`，第二个计划各有 4、6 个；未偷偷裁剪。没有线上专家注入、接触奖励或参数更新。

## 固定候选对照与两个真实 N 结果

证据在 `runs/cube/reference-20261119-audit`。对每个案例固定同一个数据集首帧、目标和 25 步候选数组（专家、零命令、固定随机、重构旧 CEM）。旧 CEM 在旧 RGB 上重构后首块与旧在线日志完全相同，旧代价也重现到 `6.2e-5` 内；不能把这个开环序列当成旧闭环的 25 步。在线之后另外将首计划的最佳样本和最终 mean 加入离线对照，没有额外仿真。

- 同设备：图像预处理逐元素相同，标准化块最大差 `4.77e-7`，初始/目标 embedding 最大差 `3.6e-7`，5 个预测块最大差 `1.91e-6`，成本最大差 `6.11e-5`，排序一致。重新拟合的 scaler 和旧 manifest 均值/尺度最大差 `1.12e-16 / 6.43e-13`。
- CPU/MPS 使用相同的四个固定候选：参考 cost 最大差 `1.53e-4`，rollout 最大绝对差 `7.63e-6`，两例排序不变。不能推出 CEM 在不同设备下运行会做同样选择。

| N 回归例 | t=0 距离 | 50 步结果 | 首计划：专家 / 最佳样本 / 执行均值的 cost | 第二次均值 cost |
| --- | ---: | --- | ---: | ---: |
| `e22d467bfaaa9357` | 0.2450 m | 失败，最终 0.2450 m；无 pad–cube 接触，方块未移动 | 17.82 / 11.08 / 27.41 | 179.34 |
| `e1db09ce9f216bf6` | 0.2761 m | 失败，最小 0.1605 m，最终 0.1648 m；17 步有 pad–cube 接触，方块移动过 | 28.45 / 244.19 / 245.27 | 186.03 |

两例均非 t=0 成功，均确实执行 50 步。e1db 的搜索在首个预算内错过了**同一模型**评分明显更低的专家序列；e22d 不能用“CEM 找不到低分”解释：它找到了比专家更低的样本，执行均值也得到较低预测 cost，但没有发生接触；该最佳样本只评分、没有执行，不知它能否成功。该结果是**公开 checkpoint + 参考搜索与执行组合在这两个案例的已观测限制**，不是模型本身的能力上限；预测/目标表示、均值选择、越界命令和 reset 未逐项因果分离。之前专家可达性的 5.42 / 2.78 mm 结果来自研究 reset。旧 N/I/D 的 1/3 来自同一个 t=0 例，不能归因为扰动。

## 下一步：固定 50-case

`reference-50.json` 已完整生成。按公开入口计算每个 episode 的 `max(step_idx)+1`，有效起点满足 `step_idx <= length-25-1`；在 1,760,000 个有效 HDF5 全局行中，保留公开实现 `choice(len(valid)-1, ...)` 对最后一个有效行 `2009974` 的排除。用 NumPy `default_rng(42)` 按行无放回抽 50，再按 HDF5 全局行号排序。没有改成按 episode 抽样、不按结果筛选、不排除 t=0 成功。先固定全表，再离线读取第 1/2 行 `128267` / `136513` 验证 26 帧、5D 单步和对齐。两个在线回归例不在表中，不计入正式结果。

现在可以启动**本地参考评估**。入口校验同一清单、设备/依赖/源码、权重和配置哈希、scaler、固定候选和首计划评分对照。本轮以 `--check-only` 跑过下面命令的前置校验，独立输出 `runs/cube/reference-20261119-preflight/STATUS.json` 为 `ready-no-rollouts`，未创建环境。50 个 slot 一次构建，不把 2 例结果外推；50 个渲染环境的资源占用尚未实跑验证。命令也会用同一清单分别运行零命令和固定 seed 的均匀随机 `[-1,1]`，各自保留参考 reset（没有数据种子，因此不是隐藏控制器状态的逐位配对）；随机流是本地定义的对照，不冒称跟论文随机命令相同。

测试：新增 6 项通过，覆盖实际 HDF5 读法、时间拼块/反变换、CPU/MPS 参考策略和求解器的缓存、指标与放行条件。按 `--ignore=le-wm/tests/test_research_contracts.py` 和单独运行 `le-wm/tests/test_research_contracts.py` 分组，共 49 项通过；合并收集有 1 个既有的进程全局导入测试失败，不收集任何新增测试也能复现该失败。

```zsh
cd /Users/xuhaoquan/project/lewm-research
DYLD_LIBRARY_PATH="$PWD/.python/cpython-3.11.15-macos-aarch64-none/lib" \
OMP_NUM_THREADS=4 MUJOCO_GL=glfw .venv-mac/bin/mjpython le-wm/cube_reference.py evaluate-50 \
  --device mps --protocol paper-300x10-top30 \
  --manifest runs/cube/reference-20261119-audit/reference-50.json \
  --verified-audit runs/cube/reference-20261119-audit \
  --out runs/cube/reference-20261119-paper50
```

每种策略保存 `steps.jsonl`、`reference-metrics.json` 和 `summary.json`。初始成功仅由 t=0 几何判据单独统计；公开代码口径为 50 步内环境 `terminated` 的并集，不把 t=0 手动加为成功，不删掉初始成功例；另统计初始未成功子集。逐例保存失败、首次成功步、接触和完成状态。异常结束保留 `STATUS.json` 和已记录的步；未开始的策略对照 `planned-evaluation.json`，未完成不折算成失败。真实 50-case 总体结果、30 轮公开配置结果、论文出图依赖/随机流和精确的 reset 上下文仍未核实；不能宣称严格复现论文 74%，也不能据此解除 E1/E2 的归因暂停。
