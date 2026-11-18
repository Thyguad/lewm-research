#!/usr/bin/env bash
# Invoke with: bash scripts/setup_macos.sh
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Darwin ]]; then
  echo 'This installer is for macOS. Use the repository installation instructions on Linux.' >&2
  exit 1
fi
python_bin="${LEWM_PYTHON:-python3.11}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo 'Install native Python 3.11/3.12 first, then set LEWM_PYTHON to its executable.' >&2
  exit 1
fi
"$python_bin" - <<'PY'
import platform, sys
if platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 Python on Apple Silicon; exit Rosetta/x86_64 first.')
if sys.version_info[:2] not in ((3, 11), (3, 12)):
    raise SystemExit('This setup script targets Python 3.11 or 3.12.')
PY
venv_dir="${LEWM_MAC_VENV:-$repo_root/.venv-mac}"
if [[ ! -d "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" - <<'PY'
import platform, sys
if platform.machine() != 'arm64' or sys.version_info[:2] not in ((3, 11), (3, 12)):
    raise SystemExit('Existing virtual environment is incompatible; choose a new LEWM_MAC_VENV.')
PY
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -e "$repo_root/stable-worldmodel[pusht-research]"
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" "$repo_root/le-wm/research_cem.py" doctor --device auto
printf '\nActivate with: source "%s/bin/activate"\n' "$venv_dir"
printf 'Then follow: %s/le-wm/README_macos.md\n' "$repo_root"
