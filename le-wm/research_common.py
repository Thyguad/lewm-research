"""Shared, import-light helpers for frozen LeWM experiments.

This module deliberately does not import torch, gymnasium, or stable-worldmodel
at module import time.  It is therefore safe to use for manifest/config checks
on a machine that only has the source tree installed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import importlib.metadata
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Case:
    case_id: str
    episode: int
    start_step: int
    goal_offset: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n"
    if overwrite:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(text)
        os.replace(temp, path)
    else:
        with path.open('x') as handle:
            handle.write(text)


def json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def append_jsonl(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, default=json_default, allow_nan=False) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(seed, case_id):
    return (int(seed) + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)) % (2**31)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def new_run(path):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'STATUS.json', {'state': 'running', 'started_at': utc_now()})
    return root


def finish_run(root, state, **extra):
    write_json(Path(root) / 'STATUS.json', {'state': state, 'updated_at': utc_now(), **extra}, overwrite=True)


def require_complete(root):
    status = read_json(Path(root) / 'STATUS.json')
    if status.get('state') != 'complete':
        raise ValueError(f'Incomplete run: {root}')


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_cases(path: str | Path, cases: Iterable[Case], *, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "schema": "lewm-frozen-cases-v1",
        "created_at": utc_now(),
        "metadata": metadata or {},
        "cases": [asdict(case) for case in cases],
    }
    write_json(path, payload)


def read_cases(path: str | Path) -> tuple[list[Case], dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema") != "lewm-frozen-cases-v1":
        raise ValueError(f"Unsupported case manifest schema in {path}")
    cases = [Case(**row) for row in payload.get("cases", [])]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Case manifest contains duplicate case_id values")
    for case in cases:
        if any(type(value) is not int for value in (case.episode, case.start_step, case.goal_offset)):
            raise ValueError('episode/start_step/goal_offset must be integers')
        if case.case_id != stable_case_id(case.episode, case.start_step, case.goal_offset):
            raise ValueError('case_id must match episode/start_step/goal_offset')
        if min(case.episode, case.start_step) < 0 or case.goal_offset <= 0:
            raise ValueError(f'Invalid case: {case}')
    return cases, payload.get("metadata", {})


def stable_case_id(episode: int, start_step: int, goal_offset: int) -> str:
    raw = f"{episode}:{start_step}:{goal_offset}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def runtime_metadata(root: str | Path = ".") -> dict[str, Any]:
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'gymnasium', 'pymunk', 'pygame',
                 'transformers', 'stable-pretraining', 'stable-worldmodel', 'h5py',
                 'hdf5plugin', 'scikit-learn', 'scipy', 'pandas', 'matplotlib', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": utc_now(),
        "git_revision": git_revision(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        'packages': packages,
    }


def sync_device(device) -> None:
    """Accept strings and torch.device; CPU needs no synchronization."""
    kind = str(device).split(':')[0]
    if kind == 'cuda':
        import torch
        torch.cuda.synchronize(device)
    elif kind == 'mps':
        import torch
        torch.mps.synchronize()


def freeze_eval(model: Any) -> Any:
    model.eval()
    model.requires_grad_(False)
    return model


def source_identity(obj: Any) -> dict[str, str]:
    import inspect

    cls = type(obj)
    try:
        source = inspect.getsourcefile(cls) or "unknown"
    except (OSError, TypeError):
        source = "unknown"
    return {"type": f"{cls.__module__}.{cls.__qualname__}", "source": source}
