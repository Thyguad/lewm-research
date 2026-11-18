# LeWM 冻结世界模型：服务器部署、实验与诊断教程

更新日期：2026-11-17。

适用仓库：[Thyguad/lewm-research](https://github.com/Thyguad/lewm-research)。本教程依据已经合并的 [PR #1](https://github.com/Thyguad/lewm-research/pull/1)，对应合并提交 `dda52d87f2dd6fe357a95df7920703374b5a4f53`，研究脚本采用 v2 结果格式。

**目标：冻结 LeWM，不训练权重，研究“规划器搜得更多，是否真的控制得更好；如果没有，问题出在哪里”。**

所有安装、下载、模型前向、仿真和实验命令均在服务器执行。电脑只用于 SSH、查看代码和下载图表。本教程是待执行流程，不代表服务器实验已经通过：此前完成的是语法、CLI 入口和 9 项轻量契约测试。

---

## 0. 先知道自己要做什么

建议按下表顺序推进，不要一上来跑最大的预算。

| 阶段 | 要解决的问题 | 主要产物 | 继续条件 |
| --- | --- | --- | --- |
| 部署 | 导入的是这份仓库吗，GPU/依赖能用吗？ | 环境信息、依赖清单 | 路径和 CUDA 检查通过 |
| 无权重测试 | 求解器和环境重建的基础逻辑正确吗？ | 测试日志 | 三个测试脚本都通过 |
| 配对冒烟 | 真实模型能运行吗，开启诊断会改变动作吗？ | 两例正式/诊断运行 | check-pair 通过 |
| 回放验收 | 保存的候选能从同一个真实起点执行吗？ | 审计表、诊断图 | audit complete |
| 实验 A | 增加搜索预算，成功率和开销如何变化？ | 预算比较图、汇总表 | 所有条件用同一案例/种子 |
| 实验 B | 在固定起点，更多搜索改善的是哪种代价？ | replan 与 audit 结果 | 先通过同预算重规划核对 |
| 研究分析 | 动力学误差、表征代价、搜索选择，哪一环值得改？ | 学习记录、下一步假设 | 结论与数据支持范围一致 |

**几个术语：**

- **世界模型**：预测做某组动作之后会发生什么。这里预测的是图像的潜在表征，不生成完整未来图像。
- **冻结**：模型处于 eval 状态，不计算用于训练的梯度，不更新参数。
- **CEM**：反复采样动作序列，选出预测代价较小的一部分，再更新采样分布。
- **闭环**：执行一段动作，重新观察，再次规划。
- **冒烟测试**：只用极少案例检查整条链路是否可用，不用来得出研究结论。
- **候选审计**：把规划器考虑过的动作放进真实仿真环境，看看预测和实际表现是否一致。

## 1. 服务器准备与路径约定

以下按 **Linux + Bash + NVIDIA GPU** 写。只使用一张 GPU，不用 torchrun 或多卡训练。先从学校平台进入已分配的计算实例；如果使用调度系统，应在获批的计算作业中运行，具体队列/申请参数按学校说明设置。

```bash
nvidia-smi
python3.11 --version
git --version
```

`nvidia-smi` 应能看到分配给你的 GPU。如果看不到，先处理资源分配/驱动问题。本文不承诺某个显存容量必然能跑所有预算；先通过小规模运行测量峰值显存和耗时。

选一个**空间足够、重启后不会被清空**的位置。下面默认使用 home 下的目录；学校提供数据盘时，修改第一行后再执行。

```bash
export LEWM_WORK="$HOME/lewm-study"
export REPO_ROOT="$LEWM_WORK/lewm-research"
export STABLEWM_HOME="$LEWM_WORK/cache/stablewm"
export RUN_ROOT="$LEWM_WORK/results/exp01"
export SDL_VIDEODRIVER=dummy
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p "$LEWM_WORK" "$STABLEWM_HOME" "$RUN_ROOT/logs"
df -h "$LEWM_WORK"
```

路径用途：

| 变量/目录 | 用途 |
| --- | --- |
| `$REPO_ROOT` | Git 仓库源码 |
| `$REPO_ROOT/.venv` | 独立 Python 环境 |
| `$STABLEWM_HOME/checkpoints/` | 模型权重与模型配置 |
| `$STABLEWM_HOME/datasets/` | 解压后的 HDF5 数据 |
| `$STABLEWM_HOME/downloads/` | 下载的压缩数据与来源记录 |
| `$RUN_ROOT` | 本批实验结果；每次新批次用 exp02、exp03 等新目录 |

已核对官方数据仓库当前提供 `pusht_expert_train.h5.zst`，页面标示压缩文件约 13.1 GB。还要为解压后的 HDF5、Python 依赖、缓存和结果留空间，不能只按 13.1 GB 准备磁盘。[来源：官方数据文件列表](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main)。

## 2. 获取修复后的代码

第一次部署：

```bash
cd "$LEWM_WORK"
git clone https://github.com/Thyguad/lewm-research.git "$REPO_ROOT"
cd "$REPO_ROOT"
git switch --detach dda52d87f2dd6fe357a95df7920703374b5a4f53
git rev-parse HEAD
```

这里固定到本教程对应的合并提交，避免之后 main 更新改变实验代码。显示 detached HEAD 是正常的，表示固定版本，不影响运行。

如果已经有这个仓库，不要重复 clone，也不要直接覆盖本地修改。先运行 `git status --short` 检查，再 fetch 并选择实验提交。

仓库内已经同时包含 `le-wm/` 和 `stable-worldmodel/`。检查：

```bash
ls "$REPO_ROOT/le-wm/research_cem.py"
ls "$REPO_ROOT/le-wm/research_diag.py"
ls "$REPO_ROOT/le-wm/research_runtime.py"
ls "$REPO_ROOT/stable-worldmodel/pyproject.toml"
```

后面的研究命令在 `$REPO_ROOT/le-wm` 执行。**本流程使用 `research_cem.py` 和 `research_diag.py`，不需要把权重转换成旧 `eval.py` 使用的 `_object.ckpt`。**

## 3. 创建 Python 环境并安装依赖

以下从仓库根目录执行。服务器没有 Python 3.11 或 venv 时，先通过学校提供的环境/模块准备好；不要把所有依赖安装到系统 Python。

```bash
cd "$REPO_ROOT"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先安装与服务器驱动兼容的 PyTorch 和 torchvision。普通环境可从以下命令开始；学校有指定镜像/安装命令时使用学校提供的方案。需要指定 CUDA wheel 时，到 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择适合驱动的构建，不要把示例中的任意 CUDA 版本硬套到服务器上。

```bash
python -m pip install torch torchvision
python -m pip install -e './stable-worldmodel[train]'
python -m pip install scikit-learn scipy matplotlib pyyaml pygame pymunk shapely opencv-python pillow huggingface_hub zstandard
python -m pip check
```

这不是已完成服务器验收的版本锁文件。首次成功后要保存实际依赖版本，后续对比实验不要中途升级包。

检查 GPU、安装路径和模型类：

```bash
python - <<'PY'
import sys
from pathlib import Path
import os
import torch
import stable_worldmodel as swm
from stable_worldmodel.wm.lewm import LeWM

print('Python:', sys.executable)
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
print('stable_worldmodel:', swm.__file__)
expected = Path(os.environ['REPO_ROOT']) / 'stable-worldmodel' / 'stable_worldmodel'
assert Path(swm.__file__).resolve().parent == expected.resolve(), '导入了另一份 stable-worldmodel'
assert torch.cuda.is_available(), 'CUDA 不可用，请先修复 GPU 环境'
print('GPU:', torch.cuda.get_device_name(0))
print('LeWM class:', LeWM.__module__, LeWM.__qualname__)
PY
```

**通过标准：**导入路径位于本仓库的 `stable-worldmodel/stable_worldmodel/`，CUDA 检查通过。`pip check` 成功只证明依赖声明没有明显冲突，不能替代后面的运行测试。

## 4. 先跑无需权重和数据的测试

```bash
cd "$REPO_ROOT/le-wm"
python research_cem.py --help
python research_diag.py --help
python tests/test_research_contracts.py -v
python tests/test_research_server.py -v
python tests/test_research_replay_server.py -v
```

| 测试 | 检查什么 | 需要什么 |
| --- | --- | --- |
| contracts | 配置覆盖、参数拒绝、案例/结果完整性、统计分组 | 轻量 Python 环境和 YAML |
| server | warm start、诊断开关、候选身份、动作反标准化 | PyTorch/NumPy 等，无真实权重 |
| replay_server | 重建环境、执行动作前缀、继续执行的状态与图像一致性 | 真实 PushT 仿真，无数据集/权重 |

三个测试都应以 `OK` 结束。某项失败就停在这里，保存完整 traceback；不要绕过失败继续跑正式预算。

通过后记录环境：

```bash
python -m pip freeze > "$RUN_ROOT/environment.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/code_commit.txt"
git -C "$REPO_ROOT" diff > "$RUN_ROOT/code_changes.patch"
nvidia-smi > "$RUN_ROOT/gpu_info.txt"
```

## 5. 下载权重与数据

### 5.1 下载并记录 Hugging Face 版本

官方模型与数据仓库都叫 `quentinll/lewm-pusht`，但一个是 model，一个是 dataset，不要混淆。下载前固定各自当时的完整 revision，并写入来源文件；中断后重跑此块会复用已记录的版本。

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

root = Path(os.environ['STABLEWM_HOME'])
downloads = root / 'downloads'
downloads.mkdir(parents=True, exist_ok=True)
manifest = downloads / 'source_revisions.json'
repo = 'quentinll/lewm-pusht'
if manifest.exists():
    versions = json.loads(manifest.read_text())
else:
    api = HfApi()
    versions = {'repo': repo, 'model_revision': api.model_info(repo).sha,
                'dataset_revision': api.dataset_info(repo).sha}
    with manifest.open('x') as handle:
        json.dump(versions, handle, indent=2)
assert versions['repo'] == repo
for name in ('weights.pt', 'config.json'):
    hf_hub_download(repo_id=repo, filename=name, revision=versions['model_revision'],
                    local_dir=root / 'checkpoints' / 'quentinll' / 'lewm-pusht')
hf_hub_download(repo_id=repo, repo_type='dataset', filename='pusht_expert_train.h5.zst',
                revision=versions['dataset_revision'], local_dir=downloads)
print(json.dumps(versions, indent=2))
PY
```

如果服务器无法访问 Hugging Face，可在能访问的机器按同一 revision 下载这三个文件，再传到上面对应目录，同时保存 `source_revisions.json`。不要用随机找到的另一个权重代替后仍称为同一实验。

下载接口和 revision/local_dir 参数依据 [Hugging Face 官方下载说明](https://huggingface.co/docs/huggingface_hub/guides/download)。

### 5.2 解压：当前文件不是 tar 包

`pusht_expert_train.h5.zst` 是单个 HDF5 的 zstd 压缩文件，不使用 `tar --zstd -xvf`。下面流式解压，不把整个数据集读入内存；先写 `.partial`，验证可打开后才改成最终文件名。

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
import h5py
import hdf5plugin
import zstandard as zstd

root = Path(os.environ['STABLEWM_HOME'])
src = root / 'downloads' / 'pusht_expert_train.h5.zst'
dst = root / 'datasets' / 'pusht_expert_train.h5'
dst.parent.mkdir(parents=True, exist_ok=True)
if not dst.exists():
    partial = dst.with_suffix('.h5.partial')
    with src.open('rb') as compressed, partial.open('wb') as output:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            shutil.copyfileobj(reader, output, length=8 * 1024 * 1024)
    with h5py.File(partial, 'r') as handle:
        assert {'ep_len', 'ep_offset', 'pixels', 'state', 'action'} <= set(handle.keys())
    partial.replace(dst)
with h5py.File(dst, 'r') as handle:
    print('File:', dst)
    print('Episodes:', len(handle['ep_len']))
    for key in ('pixels', 'state', 'action'):
        print(key, handle[key].shape, handle[key].dtype)
    print('Episodes long enough for offset=25:', int((handle['ep_len'][:] > 25).sum()))
PY
```

**最终路径必须是：**

- `$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht/weights.pt`
- `$STABLEWM_HOME/checkpoints/quentinll/lewm-pusht/config.json`
- `$STABLEWM_HOME/datasets/pusht_expert_train.h5`

研究脚本使用 `--dataset pusht_expert_train`，不带 `.h5`。原仓库某些旧说明把数据写在 cache 根目录；本教程按当前 HDF5Dataset 的真实实现使用 `datasets/` 子目录。

### 5.3 模型预检

```bash
python research_cem.py preflight \
  --policy quentinll/lewm-pusht --cache-dir "$STABLEWM_HOME" --device cuda
```

**通过标准：**实际类型为 `stable_worldmodel.wm.lewm.lewm.LeWM`，打印 `frozen: true` 和权重/config 哈希，没有缺失权重或形状不匹配错误。预检主要检查模型加载与冻结，真实前向要靠下一步冒烟覆盖。

## 6. 固定案例与实验参数

### 6.1 只生成一次案例清单

```bash
python research_cem.py generate-cases \
  --dataset pusht_expert_train --cache-dir "$STABLEWM_HOME" \
  --num-cases 20 --goal-offset 25 --case-seed 123 \
  --output "$RUN_ROOT/cases.json"
```

一个案例是“某条数据轨迹中的一个起点，及其后 25 步的目标”。每例来自不同 episode（数据轨迹），清单固定 episode/start/offset 和 case_id。所有预算条件复用这一份文件，不要分别重新抽案例。

当前默认数据为 expert_train，且 scaler 按给定整份数据拟合。它适合第一轮开发诊断，**不能据此声称未见任务泛化或独立测试集成绩**。后续做严格测试集研究时，还要明确拆分、训练集预处理统计和新代码协议，不能仅换一个数据文件名就认为无泄漏。

### 6.2 参数含义

| 参数 | 本教程默认 | 含义 |
| --- | --- | --- |
| N / num_samples | 30、100、300 | 每轮采样多少条候选动作序列 |
| I / iterations | 2、10、30 | CEM 更新多少轮 |
| topk | 10 | 每轮取最好的 10 条更新分布 |
| H / horizon | 5 | 模型向前规划多少个动作块 |
| K / receding_horizon | 5 | 每次规划后实际执行多少个块 |
| action_block | 5 | 一个模型动作块包含 5 个连续二维环境动作 |
| eval_budget | 50 | 每例闭环执行的环境动作总步数 |
| goal_offset | 25 | 数据集目标与起点之间的步数 |
| warm_start | true | 下次规划保留未执行的剩余块，并将尾部补齐 |
| planning_seed | 42、43、44 | 控制 CEM 随机采样 |

本设置 `H × action_block = 25`；每次执行 `K × action_block = 25` 步，50 步一例通常会规划两次。K=H 时剩余块为空，因此默认设置虽启用 warm start，却没有可保留的尾部。要研究 warm start 本身，需要另立实验令 K<H，并固定其他条件，不能混进本轮 N/I 比较。

固定 topk=10 会使不同 N 的精英比例不同；本轮研究的是固定 topk 规则下的预算变化，不是固定精英比例的结论。配置优先级为：代码默认 < YAML < 显式命令行。

## 7. 两个案例的配对冒烟：先确认链路能跑

以下两个运行使用同一清单的前两例、同一个种子、N=30、I=2、topk=10。输出子目录不能预先创建。

### 7.1 诊断关闭

```bash
python research_cem.py run \
  --config config/research/pusht_budget.yaml \
  --cases "$RUN_ROOT/cases.json" --limit-cases 2 \
  --device cuda --num-samples 30 --iterations 2 --topk 10 \
  --planning-seed 42 --no-diagnostics --no-save-snapshots \
  --output "$RUN_ROOT/smoke_formal"
```

### 7.2 同配置开启诊断与快照

```bash
python research_cem.py run \
  --config config/research/pusht_budget.yaml \
  --cases "$RUN_ROOT/cases.json" --limit-cases 2 \
  --device cuda --num-samples 30 --iterations 2 --topk 10 \
  --planning-seed 42 --save-snapshots --max-snapshot-plans 2 \
  --output "$RUN_ROOT/smoke_diag"
```

`--save-snapshots` 会自动启用 diagnostics。这里每例保存两个规划点，既检查第一次规划，也覆盖执行动作前缀后的第二次规划。

每次运行都会在计时开始前扫描数据 SHA256 并拟合预处理统计，首次启动可能等待较久；不能把这段时间当作单次规划延迟。

### 7.3 检查运行完整性和动作一致性

```bash
cat "$RUN_ROOT/smoke_formal/STATUS.json"
cat "$RUN_ROOT/smoke_diag/STATUS.json"
python research_diag.py check-pair \
  --formal "$RUN_ROOT/smoke_formal" --diagnostic "$RUN_ROOT/smoke_diag"
```

**通过标准：**两个 STATUS 都是 complete，check-pair 输出 PASS。它核对配置、环境初始化和整段执行动作逐位一致，回答“记录诊断是否改变了策略行为”。冒烟成功率低本身不等于程序有 bug；但动作配对检查失败必须先查清。

运行中断/失败会留下目录；旧目录不能直接作为输出重用。保留日志，改用新的输出名，并让后续命令引用新目录。当前没有自动断点续跑功能，不能把 STATUS 手工改成 complete。

## 8. 验收真实候选回放，再看第一批诊断图

```bash
python research_diag.py audit \
  --run "$RUN_ROOT/smoke_diag" --output "$RUN_ROOT/smoke_audit"
cat "$RUN_ROOT/smoke_audit/STATUS.json"
python research_diag.py plot-diagnostics \
  --run "$RUN_ROOT/smoke_diag" --audit "$RUN_ROOT/smoke_audit" \
  --output "$RUN_ROOT/smoke_figures"
```

`audit` 会重建相同 reset、初始化状态/目标和已执行动作前缀，检查收集时的物理指纹与渲染；每个候选独立执行两次，检查整段物理轨迹与终点图像一致性。随后用冻结编码器计算真实终点代价。

默认每轮保存 2 条最好候选、2 条非最好随机候选和更新后的均值。N=30、I=2、两例、每例两个规划点时，通常是 40 条审计候选，每条回放两次，因此即使是冒烟也会有一定离线成本。

**通过标准：**audit 的 STATUS 为 complete，无重建/重复执行/J_pred 重评分不一致。默认物理绝对容差 1e-7、像素容差 0；不要为了“跑完”随意放宽阈值。

### 8.1 三种核心代价

| 指标 | 怎么得到 | 用来判断什么 |
| --- | --- | --- |
| J_pred | 模型预测终点表征与目标表征的平方误差和（SSE） | 规划器认为这条动作有多好 |
| J_enc | 真实执行后的终点图像，经同一编码器得到的表征与目标的 SSE | 模型代价对真实未来的描述是否靠谱 |
| J_task | `max(位置误差/20, 周期角度误差/(π/9))` | 真实任务状态离目标有多远 |
| endpoint_success | 终点的官方成功判断 | 最后是否成功 |
| ever_success | 候选执行途中是否有任一步成功 | 是否曾达到成功状态 |

J_task 是本实验明确选择的代理代价；不要称为 LeWM 原论文的官方损失。`official_state_distance` 另行记录上游原始七维距离，不与 J_task 混用。

所有候选审计执行完整 H 个动作块、不重规划；闭环运行则执行 K 个块后重新观察。未来改为 K<H 后，这两者回答的问题更加不同。

### 8.2 先看哪些文件

| 位置 | 内容 |
| --- | --- |
| `smoke_diag/cem_iterations.jsonl` | 每轮最小候选代价、精英代价均值、更新后的动作均值代价 |
| `smoke_diag/plans.jsonl` | 每次规划的时间、计算量、最终返回均值代价 |
| `smoke_diag/snapshots/<case_id>/plan_000000/` | 该案例该规划点的 JSON 元数据和 NPZ 数组 |
| `smoke_audit/audit_records.jsonl` | 每条候选的 J_pred/J_enc/J_task、成功、回放一致性 |
| `smoke_audit/audit_summary.json` | 每个规划点的相关性、选择损失和最终计划代价 |
| `smoke_figures/cem_*.png` | 各案例/规划点的 CEM 曲线 |
| `smoke_figures/audit_*.png` | 三组代价散点图，星号是最终返回计划 |

**返回的是更新后的动作均值。**“精英候选代价的平均值”不等于“精英动作平均后再评分的代价”；曲线必须区分这两者。

另一个重要字段是 `observation_render_pixel_max_abs`。首次规划时，上游可能将数据集图像直接当作观测，它与重建环境渲染不完全相同。此字段非零时，应标记为输入图像域偏差，不能把差异全部归咎于动力学模型。它与“回放起点未恢复成功”是两个不同检查。

## 9. 实验 A：搜索预算与闭环成功率

### 9.1 第一轮做三个预算条件

| 条件 | N | I | 用途 |
| --- | --- | --- | --- |
| 小预算 | 30 | 2 | 对照起点 |
| 中预算 | 100 | 10 | 检查收益 |
| 大预算 | 300 | 30 | 资源允许且中预算完成后再运行 |

三个条件同时改变了 N 和 I，所以只能首先回答“这几档预算的整体效果如何”。若要分辨增加 N 与增加 I 各自的作用，下一轮固定 I 扫 N、固定 N 扫 I，不能仅凭上表归因。

正式运行必须关闭 diagnostics/save_snapshots；所有条件复用同一 20 例清单和种子 42、43、44。不要在同一张卡上并发这些计时实验，也不要边跑边升级包、修改代码或更换设备。

```bash
(
  set -euo pipefail
  for seed in 42 43 44; do
    for budget in '30 2' '100 10' '300 30'; do
      read -r n iterations <<< "$budget"
      name="formal_N${n}_I${iterations}_S${seed}"
      python research_cem.py run \
        --config config/research/pusht_budget.yaml \
        --cases "$RUN_ROOT/cases.json" --device cuda \
        --num-samples "$n" --iterations "$iterations" --topk 10 \
        --planning-seed "$seed" --no-diagnostics --no-save-snapshots \
        --output "$RUN_ROOT/$name" \
        2>&1 | tee "$RUN_ROOT/logs/$name.log"
    done
  done
)
```

脚本在命令失败时停止，不会把失败的 run 当作成功继续汇总。20 例 × 3 个预算 × 3 个种子，共 180 个回合；默认每回合 50 步，共 9000 个闭环环境动作步。这只是计划数量，不是运行时间估计，不含离线审计和模型内部预测。

显存/时间不够时，可先删去大预算这一档，只保留两档做完整配对。不要在某一档单独缩 H、K、图像尺寸或删掉失败案例后继续称为同一对比实验。

### 9.2 汇总与画图

```bash
python research_diag.py plot \
  --run "$RUN_ROOT"/formal_N*_I*_S* \
  --output "$RUN_ROOT/budget_figures"
cat "$RUN_ROOT/budget_figures/budget_summary.json"
```

如果重跑目录也以 formal 开头，通配符可能同时匹配旧失败目录或重复条件。此时改成显式列出本次要汇总的完整成功目录，不要删除数据校验来绕过报错。

输出图 `budget_success_time.png` 的三部分分别是：

1. 各预算成功率及案例级区间。
2. 每回合累计 solver 秒数。
3. 近似候选预测步数与成功率的关系。

`solver_seconds` 在正式模式覆盖采样、评分、top-k、分布更新和结果转 CPU；不包含环境执行、策略预处理和后续写日志。首轮冷启动包含在时间里。`episode_seconds` 是整个评估调用的时间，不能当作单次模型前向或单次规划延迟。

`candidate_predictions` 按 N × I × H × 规划次数计数，只是近似预测步数，不是 FLOPs。诊断额外评分另有计数，审计真实环境开销另行报告。

### 9.3 结果怎么写才严谨

独立单位是 20 个案例，而不是 60 条“案例 × 种子”记录。当前代码在每例只有一个种子时使用 Wilson 95% 区间；多种子先对每例求均值，再使用较保守的 Hoeffding 95% 区间。20 例的区间可能很宽，这是样本量的信息，不是绘图出错。

区间重叠与否不等于正式的配对显著性检验。先报告成功率、案例数、种子数和成本；要做可靠研究结论，再预先扩大独立案例数量并设计配对统计。不要在看完结果后只保留“有趋势”的种子。

## 10. 实验 B：固定同一个物理起点，诊断预算增加的作用

实验 A 中，不同预算执行的动作不同，第二次规划时看到的状态也可能不同。直接对比不同闭环轨迹的后续 J_pred，不能把差异都解释为搜索本身。

`replan` 把某个源快照的模型输入、物理起点、动作前缀和 warm start 固定下来，只更改搜索预算。它不产生新的闭环成功率。

### 10.1 自动选一个实际存在的源快照

```bash
export SOURCE_RUN="$RUN_ROOT/smoke_diag"
export SNAPSHOT="$(python - <<'PY'
import os
from pathlib import Path
paths = sorted((Path(os.environ['SOURCE_RUN']) / 'snapshots').glob('*/plan_000000/snapshot.json'))
assert paths, '没有找到源快照'
print(paths[0])
PY
)"
printf '%s\n' "$SNAPSHOT"
```

第一次先用 plan_000000 完成操作；之后扩展到多个预先确定的案例，并纳入 plan_000001，覆盖非空动作前缀。一个快照上的结果仅用于诊断，不是总体性能证据。

### 10.2 先用同预算重规划验证动作一致

源运行是 N=30、I=2、topk=10；不指定新 planning seed，默认恢复保存的规划 RNG 状态。

```bash
python research_diag.py replan \
  --run "$SOURCE_RUN" --snapshot "$SNAPSHOT" \
  --num-samples 30 --iterations 2 --topk 10 \
  --output "$RUN_ROOT/replan_same"
python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
source = Path(os.environ['SNAPSHOT'])
meta = json.loads(source.read_text())
target = (Path(os.environ['RUN_ROOT']) / 'replan_same' / 'snapshots' /
          meta['case_id'] / f"plan_{meta['plan_id']:06d}" / 'arrays.npz')
with np.load(source.parent / 'arrays.npz', allow_pickle=False) as a, np.load(target, allow_pickle=False) as b:
    np.testing.assert_array_equal(a['returned_actions'], b['returned_actions'])
    target_meta = json.loads(target.with_name('snapshot.json').read_text())
    source_selected = next(i for i, r in enumerate(meta['candidates']) if r['selected'])
    target_selected = next(i for i, r in enumerate(target_meta['candidates']) if r['selected'])
    np.testing.assert_array_equal(a['J_pred'][source_selected], b['J_pred'][target_selected])
print('PASS: 同预算返回动作与最终计划代价一致')
PY
```

这里核对的是实际返回动作及其预测代价，不要求诊断用随机抽样候选完全相同。不同硬件/数值环境可能无法逐位一致；失败时先定位差异与运行环境，不能直接跳过这一步。

### 10.3 改预算，分别审计

```bash
python research_diag.py audit \
  --run "$RUN_ROOT/replan_same" --output "$RUN_ROOT/replan_same_audit"
python research_diag.py replan \
  --run "$SOURCE_RUN" --snapshot "$SNAPSHOT" \
  --num-samples 100 --iterations 10 --topk 10 \
  --output "$RUN_ROOT/replan_N100_I10"
python research_diag.py audit \
  --run "$RUN_ROOT/replan_N100_I10" --output "$RUN_ROOT/replan_N100_I10_audit"
python research_diag.py plot-diagnostics \
  --run "$RUN_ROOT/replan_N100_I10" --audit "$RUN_ROOT/replan_N100_I10_audit" \
  --output "$RUN_ROOT/replan_N100_I10_figures"
```

读取两个审计汇总进行初步对照：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['RUN_ROOT'])
fields = ['case_id', 'plan_id', 'selected_J_pred', 'selected_J_enc', 'selected_J_task',
          'selection_loss_on_audited_set', 'rho_pred_enc', 'rho_enc_task', 'rho_pred_task',
          'observation_render_pixel_max_abs']
for name in ('replan_same_audit', 'replan_N100_I10_audit'):
    print('\n', name)
    for row in json.loads((root / name / 'audit_summary.json').read_text()):
        print(json.dumps({key: row[key] for key in fields}, ensure_ascii=False, indent=2))
PY
```

默认每轮保存的候选数量相近，I 越大审计集合越大。因此跨预算的“已审计集合上选择损失”也会受到集合大小和组成影响；不能只看这个数变大就断言规划器变差。优先比较同一起点的最终返回计划 J_pred/J_enc/J_task 与成功，再结合各自集合的选择损失分析。

如需更强的受控比较，可另加固定大小、相同动作集合的审计方案；当前仓库不会自动生成这种共同候选集合。不同 N 下使用相同 RNG 起点，也不保证候选严格嵌套。

### 10.4 从观察提出下一步研究假设

| 观察（需多案例重复） | 可以提出的假设 | 需要排除/补查 |
| --- | --- | --- |
| J_pred 下降，J_enc 不降或变差 | 搜索可能利用了模型预测误差 | 初始图像域偏差、数值重评分、状态重建 |
| J_enc 下降，J_task 不降 | 潜在表征距离可能与任务进展不充分一致 | 代理代价定义、成功指标、目标图像细节 |
| 集合中存在 J_task 更好的候选，但最终计划更差 | 模型代价/均值选择值得进一步研究 | 候选集合偏差、均值动作的非线性效果 |
| 同起点质量改善，但闭环成功不升 | 后续状态、反馈时机或任务难度可能限制收益 | 闭环轨迹、K、是否已达性能饱和 |
| 三种代价和成功均随预算改善 | 当前条件中搜索预算可能是有效杠杆 | 多种子、多案例及成本收益是否稳定 |

这些是待验证解释，不是单张散点图能证明的因果结论。Spearman 相关衡量排序是否一致；样本太少或代价为常数时输出 null 是合理行为。每条候选不是独立研究案例，不能拿所有候选行数当统计样本量。

## 11. 正式实验后的诊断扩展

先完成上述小流程，再扩大诊断范围。可以固定预先选定的少量案例和预算，另开一个诊断运行：

```bash
python research_cem.py run \
  --config config/research/pusht_budget.yaml \
  --cases "$RUN_ROOT/cases.json" --limit-cases 5 --device cuda \
  --num-samples 100 --iterations 10 --topk 10 --planning-seed 42 \
  --save-snapshots --max-snapshot-plans 2 \
  --output "$RUN_ROOT/diag_N100_I10_first5"
python research_diag.py audit \
  --run "$RUN_ROOT/diag_N100_I10_first5" --output "$RUN_ROOT/audit_N100_I10_first5"
```

这部分只用于诊断，不加入实验 A 的正式耗时图。按默认保存规则，上述配置约有 5 × 2 × 10 × 5 = 500 条候选，各回放两次，成本显著高于单次闭环评估。先估算小审计耗时，再决定是否扩大案例数/迭代数；可在采集前降低 keep_best/keep_random，但要记录并固定抽样规则。

## 12. 常见报错与处理

| 现象 | 优先检查 |
| --- | --- |
| 找不到 research_cem.py | 当前目录应是 `$REPO_ROOT/le-wm` |
| 导入了错误的 stable_worldmodel | 重新在仓库根目录 editable 安装，并检查 `swm.__file__` |
| CUDA 不可用 | 是否在获配 GPU 的计算节点；驱动、PyTorch 构建是否相容 |
| 缺 libGL/图形库 | 检查 OpenCV/系统图形依赖；请管理员处理对应运行库，勿用跳过仿真替代 |
| SDL/display 错误 | 同一运行环境设置 `SDL_VIDEODRIVER=dummy`，再跑回放测试 |
| 找不到 HDF5 | 应在 `$STABLEWM_HOME/datasets/`，参数名称不带 `.h5` |
| HDF5 signature 错误 | 是否误把 `.zst` 当 `.h5`，下载/解压是否完整 |
| 模型类型或 state_dict 错误 | 权重和 config 是否成对，是否是官方 PushT checkpoint，依赖版本是否兼容 |
| 输出目录已存在 | 换新输出名；不要先手工创建那个子目录，也不要覆盖已有实验 |
| 回放 mismatch | 保存 traceback、运行元数据与 snapshot；检查源码/依赖、初始化、动作前缀和渲染 |
| J_pred 重评分不一致 | 模型/config/设备数值设置是否变化；先复核，不直接调大容差 |
| Cannot pool / duplicate | 是否混入诊断结果、旧失败目录、不同 topk/案例/种子/环境或重复运行 |
| 显存不足 | 先停大预算并量测小预算，必要时重新定义较小的完整对比矩阵；不偷偷改 H/图像尺寸 |
| 进程启动很久没到第一例 | 查看 CPU/I/O；数据 SHA256 与 scaler 拟合在正式计时之前 |

重建检查不能证明全部隐藏接触缓存在数学意义上等价，但会检查物理指纹、渲染和重复轨迹。发生差异应调查，不应使用“允许不精确状态”的方式生成看起来完整的审计结果。

## 13. 断开 SSH、恢复会话与保存结果

学校有作业调度器时按学校规范提交任务；直接使用长期实例时可在平台提供的持久终端或已安装的 tmux 会话中执行。普通 SSH 断开可能终止进程，不要假定后台任务一定继续。

重新连接后，恢复第 1 节的变量和同一个 exp01 路径，然后：

```bash
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT/le-wm"
```

本地要下载结果时，可使用学校平台文件下载功能或 SFTP，把以下内容一起保留：

- `cases.json`、代码 commit、环境依赖清单、GPU 信息。
- 每个正式 run 的 `run_metadata.json`、`processors.json`、`STATUS.json`、`episodes.jsonl`、`plans.jsonl`。
- 诊断 run 的逐轮日志、cases 中的动作/初始化记录和完整 snapshots。
- audit 目录、budget/diagnostic 图表和下载来源的 `source_revisions.json`。

结果目录放在 Git 仓库外，避免误提交权重/数据。只留 PNG 会丢失重新分析所需的数据。

## 14. 每轮实验记录模板

```markdown
### 本轮问题
- 想检验的具体假设：
- 实验 A（闭环）还是实验 B（固定起点）：

### 配置与样本
- Git commit / 环境清单 / GPU：
- 权重和数据来源 revision：
- cases 文件 / 独立案例数 / 规划种子：
- N、I、topk、H、K、action_block、eval_budget：
- 是否诊断运行 / 保存候选规则：

### 验收
- 三个测试脚本：
- check-pair：
- audit 状态与容差：
- 是否出现 observation_render_pixel_max_abs 非零：

### 结果
- 成功率、区间、每回合 solver 时间：
- 最终计划 J_pred/J_enc/J_task：
- 已审计集合上的选择损失与集合大小：
- 失败案例与异常情况：

### 解释和下一步
- 数据直接支持什么：
- 目前只是推测什么：
- 下一轮只改变哪一个因素：
```

## 15. 最终验收清单

- [ ] 使用本文对应的修复提交，记录实际版本。
- [ ] GPU 可用，安装路径指向本仓库。
- [ ] 三个测试脚本通过。
- [ ] 权重/数据完整，版本与目录正确。
- [ ] 模型预检冻结断言通过。
- [ ] 两例 formal/diag 都 complete，check-pair 通过。
- [ ] 候选 audit complete，并理解初始图像偏差字段。
- [ ] 同预算 replan 返回动作/代价核对通过。
- [ ] 正式实验配置和案例/种子配对，诊断开销不混入计时。
- [ ] 原始结果、环境和来源记录保留完整。

第一轮做到这里，产物应是：一张预算—成功率—成本图、一组按案例/规划点分开的候选诊断图，以及能解释这些图的原始日志。它们可以帮助你选择下一步规划研究问题；是否值得写成论文，需要更多独立案例、受控对比及稳定复现来支撑。

---

## 参考与版本依据

- [本仓库 PR #1（已合并）](https://github.com/Thyguad/lewm-research/pull/1)
- [本文对应提交的研究 README](https://github.com/Thyguad/lewm-research/blob/dda52d87f2dd6fe357a95df7920703374b5a4f53/le-wm/README_frozen_diagnostics.md)
- [研究收集脚本](https://github.com/Thyguad/lewm-research/blob/dda52d87f2dd6fe357a95df7920703374b5a4f53/le-wm/research_cem.py)
- [审计与重规划脚本](https://github.com/Thyguad/lewm-research/blob/dda52d87f2dd6fe357a95df7920703374b5a4f53/le-wm/research_diag.py)
- [官方模型文件](https://huggingface.co/quentinll/lewm-pusht/tree/main)
- [官方数据文件](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main)
- [Hugging Face 下载接口](https://huggingface.co/docs/huggingface_hub/guides/download)
- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)

外部模型/数据文件名与下载接口于 2026-11-17 查阅核对；教程没有预设服务器实验已经成功，也没有改写任何实测成功率或耗时。
