"""Server-only model, preprocessing and deterministic PushT replay helpers.

Imports stay inside functions so static inspection and CLI help need no ML stack.
"""
from __future__ import annotations

from pathlib import Path
import copy

from research_common import file_hash, freeze_eval, source_identity


def resolve_checkpoint(policy, cache_dir=None):
    import stable_worldmodel as swm

    root = swm.data.utils.get_cache_dir(cache_dir, sub_folder='checkpoints')
    path = Path(root) / policy
    if not path.exists():
        path = Path(root) / ('models--' + policy.replace('/', '--'))
    if path.is_dir():
        files = list(path.glob('*.pt'))
        if len(files) != 1:
            raise ValueError(f'Expected exactly one .pt in {path}; use an explicit weights path')
        path = files[0]
    if not path.is_file() or not (path.parent / 'config.json').is_file():
        raise FileNotFoundError('Download weights.pt and config.json on the server first: ' + str(path))
    return path.resolve()


def load_model(policy, cache_dir, device):
    import stable_worldmodel as swm

    path = resolve_checkpoint(policy, cache_dir)
    model = freeze_eval(swm.wm.utils.load_pretrained(str(path), cache_dir=cache_dir)).to(device)
    if source_identity(model)['type'] != 'stable_worldmodel.wm.lewm.lewm.LeWM':
        raise ValueError('This audit protocol requires stable_worldmodel.wm.lewm.lewm.LeWM')
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('World model is not frozen')
    identity = {
        **source_identity(model), 'weights_sha256': file_hash(path),
        'config_sha256': file_hash(path.parent / 'config.json'),
        'checkpoint_path': str(path),
    }
    return model, identity


def source_fingerprints():
    import stable_worldmodel as swm

    root = Path(swm.__file__).parent
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*.py'))}


def execution_identity(device):
    import platform
    import torch

    info = {'device': str(device), 'cpu': platform.processor(),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'cuda_runtime': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision()}
    if str(device).startswith('cuda'):
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, gpu_memory=props.total_memory,
                    capability=[props.major, props.minor], cudnn=torch.backends.cudnn.version(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32)
    return info


def image_transform(size=224):
    import torch
    from torchvision.transforms import v2 as T

    return T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize(size=size),
    ])


def encode_frame(model, frame, device, size=224):
    import numpy as np
    import torch
    from torchvision import tv_tensors

    x = torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1)
    x = image_transform(size)(tv_tensors.Image(x))[None, None].to(device)
    with torch.inference_mode():
        z = model.encode({'pixels': x})['emb'][0, -1]
    return z.detach().cpu().numpy()


def fit_processors(dataset):
    """Fit the reference scalers without caching pixels or all numeric rows."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    processors, state = {}, {}
    with h5py.File(dataset.h5_path, 'r') as handle:
        for name in ('action', 'proprio', 'state'):
            if name not in handle:
                continue
            scaler = StandardScaler()
            for start in range(0, len(handle[name]), 100000):
                x = handle[name][start:start + 100000]
                if x.ndim != 2:
                    raise ValueError(f'{name} must be a matrix')
                x = x[np.isfinite(x).all(axis=1)]
                if len(x):
                    scaler.partial_fit(x)
            if not hasattr(scaler, 'mean_'):
                raise ValueError(f'No finite rows for {name}')
            processors[name] = scaler
            state[name] = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
            if name != 'action':
                processors['goal_' + name] = scaler
    if 'action' not in state or len(state['action']['mean']) != 2:
        raise ValueError('This experiment requires two-dimensional PushT actions')
    return processors, state


def inverse_actions(normalized, action_stats):
    """Match StandardScaler.inverse_transform, including in-place dtype rules."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(action_stats['mean'], dtype=np.float64)
    scaler.scale_ = np.asarray(action_stats['scale'], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    x = np.asarray(normalized).reshape(-1, scaler.n_features_in_)
    return scaler.inverse_transform(x)


def make_world(cfg):
    import stable_worldmodel as swm

    return swm.World(
        env_name=cfg['env_name'], num_envs=1,
        image_shape=(cfg['img_size'], cfg['img_size']), history_size=1,
        max_episode_steps=2 * cfg['eval_budget'], verbose=0,
    )


def base_env(world):
    return world.envs.unwrapped.envs[0].unwrapped


def physics_vector(world):
    """Diagnostics only: never used as model/planner input."""
    import numpy as np

    values = []
    for body in (base_env(world).agent, base_env(world).block):
        values.extend([*body.position, *body.velocity, body.angle,
                       body.angular_velocity, *body.force, body.torque,
                       *body.center_of_gravity])
    return np.asarray(values, dtype=np.float64)


def render_frame(world, size):
    import numpy as np
    from PIL import Image

    # Identical resize to AddPixelsWrapper; no extra normalization here.
    return np.array(Image.fromarray(base_env(world).render()).resize(
        (size, size), Image.Resampling.BILINEAR))


def install_reset_recorder(world, fallback_seed):
    """Capture the exact reset used by evaluate_from_dataset (one env only)."""
    import numpy as np

    original = world.reset
    recipe = {}

    def reset(seed=None, options=None):
        value = None if seed is None else np.asarray(seed).reshape(-1)[0]
        effective = int(fallback_seed if value is None else value)
        single_options = (options[0] if isinstance(options, list) else options) or {}
        recipe.update(reset_seed=effective, reset_options=copy.deepcopy(single_options))
        return original(seed=[effective], options=[copy.deepcopy(single_options)])

    world.reset = reset
    return recipe


def advance_world(world, action):
    import numpy as np

    # Official dataset evaluation also continues after success without autoreset.
    world.envs.unwrapped._autoreset_envs = np.zeros(1, dtype=bool)
    obs, reward, terminated, truncated, info = world.envs.step(np.asarray(action)[None])
    if bool(truncated[0]):
        raise RuntimeError('Replay exceeded the environment time limit')
    return obs, reward, terminated, info


def restore_world(cfg, recipe, prefix_actions):
    """Reconstruct from reset + official initialization + real action prefix.

    No claim that the seven-dimensional state is itself a complete snapshot.
    Replaying the same initialization preserves the initial simulator history.
    """
    world = make_world(cfg)
    try:
        world.reset(seed=[int(recipe['reset_seed'])],
                    options=[copy.deepcopy(recipe['reset_options'])])
        env = base_env(world)
        import numpy as np
        env._set_state(np.asarray(recipe['initial_state'], dtype=np.float64))
        env._set_goal_state(np.asarray(recipe['goal_state'], dtype=np.float64))
        for action in prefix_actions:
            advance_world(world, action)
        return world
    except BaseException:
        world.close()
        raise


def assert_restored(world, snapshot, size, state_atol=1e-7, pixel_atol=0):
    import numpy as np

    actual = physics_vector(world)
    expected = snapshot['physics']
    state_error = float(np.max(np.abs(actual - expected)))
    frame = render_frame(world, size)
    pixel_error = int(np.max(np.abs(frame.astype(int) - snapshot['rendered_pixels'].astype(int))))
    if not np.isfinite(state_error) or state_error > state_atol or pixel_error > pixel_atol:
        raise RuntimeError(f'Replay mismatch: state={state_error}, pixels={pixel_error}')
    return {'state_max_abs': state_error, 'pixel_max_abs': pixel_error}


def task_measurements(world):
    import numpy as np

    env = base_env(world)
    state, goal = env._get_obs(), np.asarray(env.goal_state)
    position = float(np.linalg.norm(goal[:4] - state[:4]))
    angle = float(abs((goal[4] - state[4] + np.pi) % (2 * np.pi) - np.pi))
    success, official_distance = env.eval_state(goal, state)
    # Explicit research cost: worst normalized error for the two success thresholds.
    return {
        'J_task': max(position / 20.0, angle / (np.pi / 9)),
        'position_error': position, 'angle_error': angle,
        'official_state_distance': float(official_distance),
        'endpoint_success': bool(success),
    }
