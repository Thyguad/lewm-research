"""Evaluator-only Cube harness. Policies receive Observation, never this object."""

from __future__ import annotations
import copy
import os
import random
import numpy as np
from .protocol import observation, transform_command

PAD_NAMES = tuple(
    "ur5e/robotiq/" + side + "_pad" + str(i) for side in ("left", "right") for i in (1, 2)
)


def space_state(space):
    state = {}
    if hasattr(space, "_value"):
        state["value"] = copy.deepcopy(space._value)
    if getattr(space, "_np_random", None) is not None:
        state["rng"] = copy.deepcopy(space._np_random.bit_generator.state)
    if hasattr(space, "spaces"):
        state["children"] = {k: space_state(v) for k, v in space.spaces.items()}
    return state


def restore_space(space, state):
    if "value" in state:
        space._value = copy.deepcopy(state["value"])
    if "rng" in state:
        space.np_random.bit_generator.state = copy.deepcopy(state["rng"])
    for k, v in state.get("children", {}).items():
        restore_space(space.spaces[k], v)


def research_env_class(base):
    class ResearchCube(base):
        def __init__(self, *args, contact_scale=None, **kwargs):
            self.contact_scale = contact_scale
            super().__init__(*args, **kwargs)

        def build_mjcf_model(self):
            root = super().build_mjcf_model()
            if self.contact_scale is not None:
                # Named collision pads only. No changes to cube-floor friction.
                for i, name in enumerate(PAD_NAMES):
                    pad = root.find("geom", name)
                    obj = root.find("geom", "object_0")
                    if pad is None or obj is None:
                        raise ValueError("OGBench contact geometry changed; inspect before running")
                    root.contact.add("pair", name=f"research_pad_{i}", geom1=pad, geom2=obj)
            return root

        def post_compilation(self):
            super().post_compilation()
            m = self._model
            self.research_pairs = []
            for i, name in enumerate(PAD_NAMES):
                g, o = m.geom(name).id, m.geom("object_0").id
                # Current OGBench pads have priority 1, cube 0. Refuse a changed
                # model instead of silently implementing the wrong mixing rule.
                if not m.geom_priority[g] > m.geom_priority[o]:
                    raise ValueError("Expected pad priority > object priority")
                matching = [
                    p
                    for p in range(m.npair)
                    if {int(m.pair_geom1[p]), int(m.pair_geom2[p])} == {g, o}
                ]
                if self.contact_scale is None:
                    if matching:
                        raise ValueError(
                            "Unexpected existing pad-object pair; inspect upstream model"
                        )
                    continue
                p = m.pair(f"research_pad_{i}").id
                if matching != [p]:
                    raise ValueError("Overlapping explicit contact pair")
                # High-priority pad determines condim, friction, solref/solimp;
                # margins and gaps use max independently of geom priority.
                friction = m.geom_friction[g][[0, 0, 1, 2, 2]].copy()
                friction[:2] *= self.contact_scale
                m.pair_friction[p] = friction
                m.pair_dim[p] = m.geom_condim[g]
                m.pair_solref[p] = m.geom_solref[g]
                m.pair_solreffriction[p] = 0
                m.pair_solimp[p] = m.geom_solimp[g]
                m.pair_margin[p] = max(m.geom_margin[g], m.geom_margin[o])
                m.pair_gap[p] = max(m.geom_gap[g], m.geom_gap[o])
                self.research_pairs.append(p)

    return ResearchCube


def make_env(cfg, condition, seed, initial, target):
    backend = cfg["render_backend"]
    if backend != "auto":
        os.environ["MUJOCO_GL"] = backend
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    env = research_env_class(CubeEnv)(
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
        contact_scale=cfg["friction_scale"] if condition == "D" else None,
    )
    # data_collection avoids hidden initialization actions in OGBench task reset.
    env.reset(seed=seed, options={"variation": []})
    env.set_state(np.asarray(initial["qpos"]), np.asarray(initial["qvel"]))
    env.set_target_pos(0, target["pos"], target["quat"])
    env.pre_step()
    env.post_step()
    if (
        env.action_space.shape != (5,)
        or not np.all(env.action_space.low == -1)
        or not np.all(env.action_space.high == 1)
    ):
        raise ValueError("Cube action semantics changed")
    return Harness(env, cfg, condition)


class Harness:
    def __init__(self, env, cfg, condition):
        self.env, self.cfg, self.condition = env, cfg, condition
        self.scale = cfg["interface_scale"] if condition == "I" else 1.0
        self.steps = 0
        self.actual_steps = 0
        self.actual_offline_replay_steps = 0

    def observe(self):
        return observation(self.env.render(), self.env.compute_ob_info(), self.cfg["sensors"])

    def step(self, command, ledger, phase):
        self.last_attempted_command = np.asarray(command).tolist()
        ledger.step(phase)  # count every attempted actual environment action
        applied = transform_command(command, self.scale)
        self.actual_steps += 1
        if phase == "offline_check":
            self.actual_offline_replay_steps += 1
        _, _, terminated, truncated, _ = self.env.step(applied)
        self.steps += 1
        return self.observe(), self.evaluate(applied, terminated, truncated)

    def evaluate(self, applied=None, terminated=False, truncated=False):
        e, m, d = self.env, self.env._model, self.env._data
        import mujoco

        contacts = []
        for k in range(d.ncon):
            c = d.contact[k]
            names = (m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name)
            if "object_0" in names and any(x in PAD_NAMES for x in names):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, k, force)
                contacts.append(
                    dict(geoms=names, friction=c.friction.copy().tolist(), force=force.tolist())
                )
        pos = d.joint("object_joint_0").qpos[:3].copy()
        target = d.mocap_pos[e._cube_target_mocap_ids[0]]
        return dict(
            success=bool(e._compute_successes()[0]),
            distance=float(np.linalg.norm(pos - target)),
            object_pos=pos.tolist(),
            effector_pos=d.site_xpos[e._pinch_site_id].tolist(),
            applied_action=None if applied is None else applied.tolist(),
            contacts=contacts,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def audit_model(self):
        m = self.env._model
        return dict(
            condition=self.condition,
            interface_scale=self.scale,
            friction_scale=self.cfg["friction_scale"] if self.condition == "D" else 1.0,
            control_timestep=self.env.control_timestep,
            physics_timestep=self.env.physics_timestep,
            geoms=[
                dict(
                    name=n,
                    priority=int(m.geom_priority[m.geom(n).id]),
                    friction=m.geom_friction[m.geom(n).id].tolist(),
                )
                for n in (*PAD_NAMES, "object_0")
            ],
            pairs=[
                dict(
                    geom1=m.geom(int(m.pair_geom1[i])).name,
                    geom2=m.geom(int(m.pair_geom2[i])).name,
                    friction=m.pair_friction[i].tolist(),
                    dim=int(m.pair_dim[i]),
                )
                for i in range(m.npair)
            ],
        )

    def snapshot(self, planner=None):
        e = self.env
        # MjData.__copy__ copies ALL simulator fields, including warm start,
        # controls, activation, mocap, applied forces, plugin/history buffers.
        # IK has its own MjData and mutable work arrays, also copied.
        fields = (
            "_prev_qpos",
            "_prev_qvel",
            "_prev_ob_info",
            "_success",
            "_target_effector_pose",
            "_target_task",
            "_target_block",
            "_cur_goal_ob",
            "_reset_next_step",
        )
        return dict(
            data=copy.copy(e._data),
            ik={
                k: copy.copy(v) if k == "_data" else copy.deepcopy(v)
                for k, v in vars(e._ik).items()
                if k != "_model"
            },
            fields={k: copy.deepcopy(getattr(e, k)) for k in fields if hasattr(e, k)},
            rng=copy.deepcopy(e.np_random.bit_generator.state),
            variation=space_state(e.variation_space),
            py_rng=random.getstate(),
            np_rng=np.random.get_state(),
            model_arrays={
                k: getattr(e._model, k).copy()
                for k in ("geom_friction", "geom_priority", "pair_friction", "geom_rgba")
            },
            steps=self.steps,
            scale=self.scale,
            condition=self.condition,
            planner=None
            if planner is None
            else dict(
                buffer=copy.deepcopy(planner.buffer),
                rng=copy.deepcopy(planner.rng.bit_generator.state),
            ),
        )

    def restore(self, snap, planner=None):
        e = self.env
        e._data = copy.copy(snap["data"])
        for k, v in snap["ik"].items():
            setattr(e._ik, k, copy.copy(v) if k == "_data" else copy.deepcopy(v))
        for k, v in snap["fields"].items():
            setattr(e, k, copy.deepcopy(v))
        for k, v in snap["model_arrays"].items():
            getattr(e._model, k)[:] = v
        e.np_random.bit_generator.state = copy.deepcopy(snap["rng"])
        restore_space(e.variation_space, snap["variation"])
        random.setstate(snap["py_rng"])
        np.random.set_state(snap["np_rng"])
        self.steps, self.scale, self.condition = snap["steps"], snap["scale"], snap["condition"]
        if snap["planner"] is not None:
            if planner is None:
                raise ValueError("Snapshot includes a planner; restore it explicitly")
            planner.buffer = copy.deepcopy(snap["planner"]["buffer"])
            planner.rng.bit_generator.state = copy.deepcopy(snap["planner"]["rng"])
        # No mj_forward: would overwrite solver warm starts from the snapshot.

    def replay_check(self, commands, ledger, planner=None):
        """Offline diagnostic only. All simulated steps charged separately."""
        from .protocol import Ledger

        snap = self.snapshot(planner)
        trajectories = []
        try:
            for _ in range(2):
                self.restore(snap, planner)
                rows = []
                for a in commands:
                    obs, truth = self.step(a, Ledger(len(commands)), "offline_check")
                    ledger.offline_replay_steps += 1
                    rows.append(
                        (
                            obs.pixels,
                            obs.sensors,
                            self.env._data.qpos.copy(),
                            self.env._data.qvel.copy(),
                            self.env._data.ctrl.copy(),
                            truth,
                        )
                    )
                trajectories.append(rows)
            for x, y in zip(*trajectories):
                for a, b in zip(x[:5], y[:5]):
                    if not np.array_equal(a, b):
                        raise RuntimeError(
                            "Full-state replay mismatch (pixels/sensors/physics/controller)"
                        )
                if x[5] != y[5]:
                    raise RuntimeError("Replay mismatch in contacts or evaluation")
            ledger.restore_checks += 1
            return dict(
                status="passed",
                steps=len(commands),
                comparisons="pixels,sensors,qpos,qvel,ctrl,contacts,success",
            )
        finally:
            self.restore(snap, planner)

    def close(self):
        if self.env._renderer is not None:
            self.env._renderer.close()
        self.env.close()
