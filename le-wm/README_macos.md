# 在 Apple Silicon Mac 上运行 LeWM 冻结规划实验

适用：M 系列 Mac，尤其是本项目使用者的 M4 Pro / 24 GB 统一内存。研究目标仍然是冻结 LeWM、比较规划预算与候选质量，不训练模型。以下命令先在仓库根目录执行，到明确的 `cd le-wm` 后再运行实验命令。

本次已经在 Linux CPU 上完成回归和小规模流程测试，**没有在真实 Mac/MPS 或官方预训练权重上完成验收**。下面的 MPS 测试和两案例冒烟是你开始正式实验前的验收步骤。不要把安装成功或 `doctor` 通过理解为研究实验已经复现。

## 1. 获取这次适配代码

已有仓库时，先确认 `git status`，保存自己的未提交工作，再切到适配分支：

```bash
git fetch origin
git switch fix/macos-mps-support
git pull --ff-only
```

尚未下载仓库时：

```bash
git clone --branch fix/macos-mps-support https://github.com/Thyguad/lewm-research.git
cd lewm-research
```

等这次 PR 合并后，也可以使用更新后的 `main`。之前服务器教程中的固定旧提交不包含这次适配，不要再切回旧提交。

## 2. 安装原生 arm64 环境

需要原生 arm64 Python 3.11 或 3.12。已有合适的 Python 就直接使用；如果你已经使用 Homebrew，可以执行 `brew install python@3.11`。不要在 Rosetta 的 x86_64 Python 环境安装。

```bash
python3.11 -c 'import platform, sys; print(platform.machine(), sys.version)'
bash scripts/setup_macos.sh
source .venv-mac/bin/activate
```

第一条应显示 `arm64`。如果你的 Python 是 3.12，安装时改用：

```bash
LEWM_PYTHON=python3.12 bash scripts/setup_macos.sh
source .venv-mac/bin/activate
```

脚本创建 `.venv-mac`，安装本仓库 `stable-worldmodel[pusht-research]`，执行依赖检查和设备探测。这个 extra 包含 PushT、权重实例化和诊断所需依赖，不需要安装完整的 `[env]` 环境合集。脚本不下载权重/数据，不启动实验。依赖版本尚不是经过 Mac 实机验证的锁文件；成功验收后用 `pip freeze` 保存实际环境。

```bash
export STABLEWM_HOME="$HOME/lewm-data"
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht"
cd le-wm
mkdir -p runs/mac_environment
git rev-parse HEAD > runs/mac_environment/git-commit.txt
python -m pip freeze > runs/mac_environment/pip-freeze.txt
python -c 'import stable_worldmodel as s; print(s.__file__)'
python research_cem.py doctor --device mps
```

导入路径必须指向你这个仓库的 `stable-worldmodel/stable_worldmodel`。以后新开终端，需要重新激活虚拟环境、设置 `STABLEWM_HOME`，并进入 `le-wm` 目录。

`doctor` 只验证分配、基础矩阵运算、归约、索引和同步；它不加载权重。默认 `--device auto` 依次选 CUDA、MPS、CPU，并记录实际设备；Mac 配置明确指定 `mps`，不可用就报错，避免悄悄跑到 CPU。

## 3. 先运行不需要下载数据的测试

```bash
python tests/test_research_contracts.py -v
python tests/test_research_server.py -v
python tests/test_research_devices.py --require-mps -v
python tests/test_research_lewm_devices.py --require-mps -v
python tests/test_research_replay_server.py -v
python tests/test_research_pipeline.py -v
```

`server` 是历史文件名，这两项现在也适用于 Mac。`--require-mps` 会在 MPS 不可用时直接失败，不能通过跳过测试得到虚假的 MPS 验收。最后一项用本地生成的小权重、小数据和真实 PushT，在 CPU 上检查收集→配对→审计→重规划→绘图；它不下载数据、不训练模型，也不能用于报告成功率。

如果决定先用 CPU，把两个 `--require-mps` 去掉，并将后续 `preflight`、两次冒烟和所有正式运行统一加上 `--device cpu`。这时 MPS 测试会被跳过，意味着 MPS 仍未验证。

## 4. 准备官方权重与数据

如已有文件，可直接放入下列位置，无需重复下载：

| 内容 | 路径 |
| --- | --- |
| 权重 | `$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht/weights.pt` |
| 模型配置 | 同目录的 `config.json` |
| 解压后的数据 | `$STABLEWM_HOME/datasets/pusht_expert_train.h5` |

官方来源：[模型文件](https://huggingface.co/quentinll/lewm-pusht/tree/main)、[数据文件](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main)。数据压缩文件约 13.1 GB，解压后还会占用额外磁盘空间；下载前检查磁盘容量。不要将整个压缩包读进内存，也不要把 `.h5.zst` 当作 `.tar.zst` 解包。

下载命令会使用网络，并占用上述磁盘空间：

```bash
hf download quentinll/lewm-pusht weights.pt config.json \
  --local-dir "$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht"
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
  --repo-type dataset --local-dir "$STABLEWM_HOME/datasets"
```

用 Python 流式解压，写完后再改成最终文件名，防止把中断的文件误当成完整数据：

```bash
python - <<'PY'
import os
from pathlib import Path
import h5py
import zstandard as zstd
root = Path(os.environ['STABLEWM_HOME']) / 'datasets'
target = root / 'pusht_expert_train.h5'
partial = target.with_suffix('.h5.partial')
if target.exists():
    raise SystemExit(f'Already exists; inspect before replacing: {target}')
with (root / 'pusht_expert_train.h5.zst').open('rb') as src, partial.open('wb') as dst:
    zstd.ZstdDecompressor().copy_stream(src, dst)
with h5py.File(partial, 'r') as f:
    required = {'pixels', 'action', 'state', 'proprio', 'ep_len', 'ep_offset'}
    if not required.issubset(f.keys()):
        raise RuntimeError(f'Missing datasets: {required - set(f.keys())}')
    print({k: f[k].shape for k in sorted(required)})
partial.replace(target)
print(target)
PY
```

`--dataset` 始终填 `pusht_expert_train`，不带 `.h5`。本仓库研究入口从 `STABLEWM_HOME/datasets` 读取，不要使用旧上游说明里省略 `datasets/` 的路径。数据通常按需读像素，但索引、元数据和 scaler 拟合仍需内存/时间；`--limit-cases 2` 不会把整份数据的准备开销也降为两条样本。

## 5. 实际权重的 MPS 前向预检

```bash
python research_cem.py preflight --policy quentinll/lewm-pusht --device mps
```

应看到 `device: mps`、`frozen: true`、`forward_check: passed ...`。这一步会用真实权重和合成输入计算两个候选的五步预测代价，覆盖真实编码器/预测器的设备兼容性，但不代表 PushT 成功率。`--no-forward-check` 仅加载权重，不作为完整前向验收。

## 6. 两案例配对冒烟和回放审计

```bash
python research_cem.py generate-cases --dataset pusht_expert_train \
  --num-cases 20 --goal-offset 25 --case-seed 123 --output runs/mac_cases.json

python research_cem.py run --config config/research/pusht_mac.yaml \
  --cases runs/mac_cases.json --limit-cases 2 --planning-seed 42 \
  --output runs/mac_smoke_formal

python research_cem.py run --config config/research/pusht_mac.yaml \
  --cases runs/mac_cases.json --limit-cases 2 --planning-seed 42 \
  --save-snapshots --max-snapshot-plans 2 --output runs/mac_smoke_diag

python research_diag.py check-pair --formal runs/mac_smoke_formal \
  --diagnostic runs/mac_smoke_diag
python research_diag.py audit --run runs/mac_smoke_diag --output runs/mac_smoke_audit
python research_diag.py plot-diagnostics --run runs/mac_smoke_diag \
  --audit runs/mac_smoke_audit --output runs/mac_smoke_figures
```

验收条件：两次收集的 `STATUS.json` 都是 `complete`；`check-pair` 通过；audit 为 `complete`，且生成诊断图。`check-pair` 要求实际执行动作逐位一致；不通过就保留目录定位原因，不应为了通过测试而放宽标准。审计会重复执行候选真实动作，比单次规划慢是正常的。

每次输出目录必须是新的。失败目录留作排查，修复后用新名字重跑。即使刚刚的两案例冒烟成功，也不代表已经做完 20 案例的预算实验。

## 7. 24 GB Mac 的起步预算

`config/research/pusht_mac.yaml` 是起步配置，不是已实测的峰值内存保证：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| device | mps | 使用 Apple GPU |
| candidate_batch_size | 16 | 每次模型前向最多处理 16 个候选 |
| num_samples / iterations | 30 / 2 | 每轮 30 个候选、更新 2 轮 |
| topk | 10 | 在完整 N 个候选中选 10 个精英 |
| horizon / receding_horizon | 5 / 5 | 预测 5 块、执行 5 块后再规划 |
| action_block | 5 | 每块展开成 5 个连续动作 |
| eval_budget / goal_offset | 50 / 25 | 保持原实验任务设置 |

分批只限制单次模型前向的候选数；仍然评分全部 N 个候选，最后统一做全局 top-k 和 CEM 更新。总候选数、规划深度和任务定义不因此减少。所有候选动作/代价仍需保存，因此不能将它理解为内存完全与 N 无关。

先比较 N/I = 30/2、60/5、100/10。用显式 Bash 执行循环，避免 macOS 默认 zsh 与 Bash 的语法差异：

```bash
bash <<'BASH'
set -euo pipefail
for seed in 42 43 44; do
  for budget in '30 2' '60 5' '100 10'; do
    read -r n iterations <<< "$budget"
    python research_cem.py run --config config/research/pusht_mac.yaml \
      --cases runs/mac_cases.json --num-samples "$n" --iterations "$iterations" \
      --planning-seed "$seed" --output "runs/mac_formal_N${n}_I${iterations}_S${seed}"
  done
done
BASH
python research_diag.py plot --run runs/mac_formal_N*_I*_S* --output runs/mac_budget_figures
```

这组总计 3 预算 × 3 规划种子 × 20 案例 = 180 回合。先从单预算/单种子确认耗时能接受，再跑整组。可用 macOS `caffeinate -i` 前缀避免长任务期间因空闲睡眠；耗时受电源模式、温度和其他应用负载影响，没有预先承诺完成时间。

正式实验不加 `--save-snapshots` / `--diagnostics`。诊断单独做，不能混入正式时间图。`solver_seconds` 包含求解期间的采样、设备传输、全部分批评分、精英更新和结果转 CPU；数据准备、策略预处理、环境执行等不都在其中。详细指标、固定状态重规划命令和统计解释见 [冻结规划诊断协议](README_frozen_diagnostics.md)。

## 8. Mac 上的比较规则与排错

- **MPS 不可用**：检查 `platform.machine()` 是否 `arm64`，确认 PyTorch 的 `torch.backends.mps.is_built()` / `is_available()`。macOS 最低版本以实际安装的 PyTorch 版本为准。可统一改 `--device cpu` 先验证流程。
- **float64 报错**：本次只将模型输入/参数转换为 float32，且在传到 MPS 前转换；物理状态仍保留 CPU float64。不要为了消除错误而把整个模拟器状态降精度。若报错来自新增代码，检查那条数据到设备的路径。
- **MPS 内存不足或系统明显换页**：退出占用大量内存的其他程序；把配置里的 `candidate_batch_size` 从 16 降为 8，重新做配对冒烟。整组对比统一用 8 并使用新输出目录。不要设置 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` 来取消内存保护。
- **某个 MPS 算子尚不支持**：先保存完整错误。最明确的替代是整组使用 CPU。如果只为排查而尝试 `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`，必须在 Python 启动前设置；它只处理部分不支持的算子，不能解决 float64 或 OOM，也不代表纯 MPS 性能。重新做配对验收，整组保持同样设置。
- **不要混合设备/分批大小**：CPU、MPS、CUDA 的耗时分开分析；同一预算曲线保持分批大小、软件版本、环境变量一致。分批和不分批可能有浮点舍入差异，不能保证真实模型的精英选择逐位相同；代码会将分批大小写入运行元数据，并拒绝混合协议绘图。`candidate_batch_size: 0` 表示整批评分，仅用于内存足够时。
- **审计优先留在收集机器**：跨 OS/仿真依赖可能产生不同渲染/物理误差。不要用放宽回放容差的方法把不同环境冒充成精确重建。
- **旧快照不能直接升级**：本次修改了模型设备输入路径和代码身份记录。旧代码采集的结果无法通过当前源码校验，应在这次版本重新采集，不能修改旧元数据冒充新结果。

实验上首先看“增加搜索预算是否提高闭环成功率、是否降低真实 J_task”，再结合 J_pred/J_enc 排名及固定状态重规划分析原因。这里使用 `pusht_expert_train` 的开发案例，不能据此宣称未见任务泛化；跨设备速度也不能直接解释为规划算法改进。

官方设备说明：[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)、[MPS 环境变量](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)。

## 9. 此次代码校验记录

在 Linux CPU / Python 3.12 / PyTorch 2.15.0+cpu 环境：11 项配置契约、7 项设备/分批 CPU 回归、5 项原求解器回归、1 项实际 LeWM rollout 小模型测试、1 项真实 PushT 回放测试、1 项完整小流程集成测试通过，共 26 项。3 项 MPS 专项测试因没有 Apple GPU 跳过。另检查了 Python 语法、CLI 帮助、安装脚本 Shell 语法与 Git diff。

没有执行 macOS 安装器、MPS 实机测试、官方权重前向或真实数据实验；Mac 依赖解析和峰值内存仍需按本教程实机验收。验证中的小模型复用了真实 LeWM/Predictor，但编码器是本地测试替身，不等同于完整官方模型。
