"""Device selection and float32 model boundary; simulator state stays on CPU."""
from __future__ import annotations

import os
import platform
import re


def validate_device_name(requested):
    value = str(requested)
    if not re.fullmatch(r'auto|cpu|mps(?::0)?|cuda(?::[0-9]+)?', value):
        raise ValueError('device must be auto, cpu, mps, or cuda[:index]')
    return value


def resolve_device(requested='auto'):
    import torch

    value = validate_device_name(requested)
    mps = getattr(torch.backends, 'mps', None)
    if value == 'auto':
        if torch.cuda.is_available():
            value = 'cuda'
        elif mps is not None and mps.is_available():
            value = 'mps'
        else:
            value = 'cpu'
    if value.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. On Apple Silicon use --device mps or --device cpu.')
        index = int(value.split(':')[1]) if ':' in value else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA index {index} is outside the visible device range')
        return f'cuda:{index}'
    if value.startswith('mps'):
        if mps is None or not mps.is_built():
            raise RuntimeError('This PyTorch build has no MPS support. Install native macOS PyTorch or use --device cpu.')
        if not mps.is_available():
            raise RuntimeError('MPS unavailable. Check macOS/PyTorch compatibility and native arm64 Python, or use --device cpu.')
        return 'mps'
    return 'cpu'


def prepare_headless():
    # Respect an explicit user choice. Must happen before importing pygame.
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('MPLBACKEND', 'Agg')


def model_tensor(value, device):
    """Cast before transfer: .to('mps').float() fails on CPU float64 input."""
    import torch

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(device=device)


def prepare_model_info(info, device):
    # Only these inputs enter the supported LeWM. In particular, float64 physics
    # state/goal_state must not be copied to MPS or rounded in replay recipes.
    required = ('pixels', 'goal', 'action')
    missing = set(required) - set(info)
    if missing:
        raise ValueError(f'Missing model inputs: {sorted(missing)}')
    return {name: model_tensor(info[name], device) for name in required}


def backend_metadata(device):
    import torch

    mps = getattr(torch.backends, 'mps', None)
    info = {'machine': platform.machine(), 'macos': platform.mac_ver()[0],
            'model_dtype': 'float32', 'sdl_video_driver': os.getenv('SDL_VIDEODRIVER')}
    if str(device).startswith('mps'):
        info.update(mps_built=mps.is_built(), mps_available=mps.is_available(),
                    mps_environment={name: os.getenv(name) for name in (
                        'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH',
                        'PYTORCH_MPS_PREFER_METAL', 'PYTORCH_MPS_METAL_VERSION',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO')})
        if hasattr(torch.mps, 'recommended_max_memory'):
            info['mps_recommended_max_memory'] = torch.mps.recommended_max_memory()
    return info


def probe_device(device):
    """Small allocation/reduction check, not a checkpoint compatibility test."""
    import torch
    from research_common import sync_device

    with torch.inference_mode():
        x = model_tensor(torch.arange(24, dtype=torch.float64).reshape(4, 6), device)
        y = (x @ x.T).std(dim=1)
        values, indices = y.topk(2, largest=False)
        selected = x[indices]
        sync_device(device)
        if not torch.isfinite(values).all() or selected.shape != (2, 6):
            raise RuntimeError('Device probe failed')
    return {'device': str(device), 'probe': 'passed', **backend_metadata(device)}
