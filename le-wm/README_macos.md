# Apple Silicon 环境入口

Mac 安装、设备与 Cube 渲染说明已合并到 [docs/setup.md](../docs/setup.md)。此文件仅保留稳定链接，供现有安装脚本提示使用。

- 使用原生 arm64 Python 3.11/3.12。
- `scripts/setup_macos.sh` 创建或复用 `.venv-mac`，安装 PushT 依赖。
- Cube 额外安装 `cube-research`；GLFW 通常使用同一环境的 `mjpython`。
- 新研究按当前任务选择配置，不必切换到旧适配分支或重新执行旧实验清单。
