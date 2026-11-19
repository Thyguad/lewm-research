import sys
from pathlib import Path
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research.data import Dataset, make_manifest
from cube_research.protocol import read_config, validate_manifest
from research_common import read_json


def test_hdf5_alignment_population_scaler_and_grouped_cases(tmp_path):
    path = tmp_path / "cube.h5"
    n = 30
    episodes = 10
    actions = np.linspace(-1, 1, episodes * n * 5).reshape(episodes * n, 5)
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", shape=(episodes * n, 224, 224, 3), dtype="u1", fillvalue=0)
        for k, v in dict(
            action=actions,
            qpos=np.arange(episodes * n)[:, None],
            qvel=np.zeros((episodes * n, 1)),
            ep_len=[n] * episodes,
            ep_offset=np.arange(episodes) * n,
            privileged_block_0_pos=np.repeat(np.arange(episodes * n)[:, None], 3, axis=1),
            privileged_block_0_quat=np.tile([1, 0, 0, 0], (episodes * n, 1)),
        ).items():
            f[k] = v
    d = Dataset(path)
    try:
        stats = d.stats()
        assert np.allclose(stats["mean"], actions.mean(0))
        assert np.allclose(stats["scale"], actions.std(0))
        initial, target, goal = d.case(dict(episode=2, start_step=1, goal_offset=25))
        assert initial["qpos"][0] == 61 and target["pos"][0] == 86
        assert goal.shape == (224, 224, 3)
    finally:
        d.close()
    cfg = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")
    cfg["dataset"] = str(path)
    make_manifest(cfg, tmp_path / "cases.json")
    m = read_json(tmp_path / "cases.json")
    validate_manifest(m)
    assert len(m["cases"]) == episodes
    with pytest.raises(FileExistsError):
        make_manifest(cfg, tmp_path / "cases.json")
