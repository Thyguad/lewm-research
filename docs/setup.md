# 安装与设备

从仓库根目录执行。使用已有兼容环境即可，不必因文档整理重新安装或下载数据。

## 环境

Python 3.11/3.12 的通用安装示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e './stable-worldmodel[cube-research,pusht-research]'
python -m pip check
```

只研究一个环境时，可仅安装 `cube-research` 或 `pusht-research` extra。Linux/CUDA 的 PyTorch 构建按实际驱动选择；仓库不指定服务器地址或 GPU 编号。

Apple Silicon 使用原生 arm64 Python。现有 `scripts/setup_macos.sh` 创建 `.venv-mac` 并安装 PushT extra；需要 Cube 时补装：

```bash
bash scripts/setup_macos.sh
source .venv-mac/bin/activate
python -m pip install -e './stable-worldmodel[cube-research]'
```

Cube/GLFW 在 macOS 上通常使用该环境的 `mjpython` 启动。Linux 无显示环境可配置 EGL；OSMesa 需要系统库。设备用 `auto`、`cpu`、`mps` 或 `cuda[:index]` 配置；不要将某机器的验证结果视为所有设备均已通过。

## 本地输入

复用已有权重与数据，不运行自动下载：

- Cube：`cube_single_expert.h5`、匹配的 `weights.pt` 与相邻 `config.json`。
- PushT：`pusht_expert_train.h5` 与该环境对应的权重。
- 官方资源：https://huggingface.co/collections/quentinll/lewm 。

Cube 配置副本：

```bash
cp le-wm/config/research/cube.yaml le-wm/config/research/cube-local.yaml
```

编辑副本中的 `dataset`、`weights`、`device`、`render_backend`。已有副本时直接编辑，不覆盖本机路径。`cube.yaml` 保留旧实验的完整配置字段，新方法应使用自己明确声明的配置，不能把这些旧超参数当成已选择的新方案。

## 轻量入口检查

```bash
python le-wm/research_cube.py --help
python le-wm/cube_reference.py --help
python le-wm/research_cem.py --help
python le-wm/research_cube.py --config le-wm/config/research/cube-local.yaml doctor
```

`doctor` 检查依赖和本地文件是否存在，不证明模型前向、渲染或任务成功。具体实验与验证按当前任务选择，安装本身不自动启动批量实验。
