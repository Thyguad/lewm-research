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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


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
    return cases, payload.get("metadata", {})


def stable_case_id(episode: int, start_step: int, goal_offset: int) -> str:
    raw = f"{episode}:{start_step}:{goal_offset}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def runtime_metadata(root: str | Path = ".") -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "git_revision": git_revision(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def sync_device(device: str) -> None:
    """Synchronize CUDA before/after formal timing; no-op on other devices."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


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
