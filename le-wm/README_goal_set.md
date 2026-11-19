# 冻结 LeWM 的 Cube 目标集合规划

本入口比较 `l2`、`scalar`、`position`、`global-set` 和 `goal-set`。目标集合以 RGB 目标潜变量为中心，允许沿低秩正交子空间移动，但位移不能超过训练集标定的半径。只有终点评分改变，LeWM 编码器、动作编码器、预测器和 CEM 行为保持冻结或一致。全局集合与条件集合共享秩、半径、潜空间尺度和配对；位置回归按米制欧氏距离评分。

## 运行

从仓库根目录执行。`--inputs` 仅从本机 YAML 读取数据、权重、设备和渲染后端，不继承旧 Cube 实验协议。也可以直接设置 `--dataset`、`--weights`、`--device` 和 `--render-backend`。每个 `--out` 必须是新目录。

```bash
python le-wm/research_planning_ideas.py --help

python le-wm/research_planning_ideas.py --inputs le-wm/config/research/cube-local.yaml \
  prepare --out runs/goal-set/cache

python le-wm/research_planning_ideas.py --inputs le-wm/config/research/cube-local.yaml \
  train --cache runs/goal-set/cache --out runs/goal-set/heads

mjpython le-wm/research_planning_ideas.py --inputs le-wm/config/research/cube-local.yaml \
  evaluate --cache runs/goal-set/cache --trained runs/goal-set/heads \
  --out runs/goal-set/dev
```

macOS/GLFW 用环境中的 `mjpython`；其他平台按渲染后端用 `python`。如果本地独立 Python 安装在启动 `mjpython` 时找不到 `libpython*.dylib`，可通过同一虚拟环境的 `python` 查询库目录：

```bash
export DYLD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

`evaluate --smoke` 仅用开发清单首个起点未达标的案例、一个种子和降预算 CEM 验证链路，成功率不可与正式协议比较。`--dev-prefix N` 用开发清单前 N 例、全部开发种子和正式 CEM。`--mechanism` 添加替代目标图，也可配合 `--smoke` 核对管线。`diagnose --cache ... --trained ... --out ...` 在开发案例上对同批随机候选比较五个头的预测评分与物理终点距离，不反馈给策略。待方法固定后，`evaluate --split test --dev-run runs/goal-set/dev ...` 使用同一数据清单和权重运行保留集；测试入口拒绝以冒烟或前缀开发记录作为依据。

## 协议和监督

- `cube-planner-study-v1`：同轨迹随机起点、未来 25 步的图像目标、最多 50 个环境步。CEM 每次预测 5 个 block、每 block 5 步，执行第一个 block 后重规划；每次 300 候选、10 轮、30 精英，执行最终均值。首次方块中心距目标不超过 40 mm 成功。无朝向要求；不是完整抓取放置或 OGBench 排行榜复现。
- 先按整条轨迹分组，再取帧。每个配对束包含相同目标的一正一负，均为真实专家图像。离线用 `privileged_block_0_pos`、`proprio_joint_pos` 和 `proprio_gripper_opening` 构建配对。机械臂相似度是关节角 RMS 和夹爪开度组合的代理量，不等同于像素相似度；抽查 `*-examples.png`。位置头只用同一批被配对帧的坐标标签。新增头的测试轨迹是留出的，不代表 LeWM 预训练没见过这些轨迹。
- 真实位置只用于离线标注、环境初始化和评价。策略仅有当前与目标 RGB 编码、已生成的命令前缀及冻结模型，不接收 simulator info。开发清单不按基线成败筛选；正式汇总同时报告全部案例和排除第 0 步成功的子集。
- 机制案例用原图、两张独立轨迹的等价目标图和一张错误物体目标图。等价目标图离原目标不超过 5 mm；错误目标距原目标超过 80 mm，避免两个 40 mm 成功球重叠。对于错误图，分别记录是否接近新和原物体目标。替代图不足的案例明确记录，不合成、不放宽阈值，也不按结果补挑。

## 输出和边界

`prepare` 写出 `manifest.json`、带轨迹和原始行号的 `train.npz` / `dev.npz` 及配对图片；`train` 写出 `heads.pt` 和开发配对排名 `training.json`；`evaluate` 保存逐回合动作、位置距离和计算账本 `episodes.jsonl`、配对统计 `summary.json`，以及可选的替代目标图与 `mechanism.json`。统计先在案例内平均规划种子，再按案例 bootstrap；更换目标图片不会增加独立样本数。CUDA 的峰值是 PyTorch allocator 峰值，MPS 只记录预测调用返回时的采样峰值，RSS 是进程高水位，三者不能直接对比。

此入口仅实现 Cube。评分头本身不依赖动作维度，但 PushT 的图像标签、位姿等价和闭环评估需要独立适配。请勿从真实帧排名或小规模闭环冒烟推断任务成功率。
