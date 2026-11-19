"""Offline paired intervention audit using recorded expert actions, never policy feedback."""

from dataclasses import asdict
import numpy as np
from research_common import new_run, finish_run, read_json, write_json, file_hash
from .protocol import Ledger, validate_manifest
from .data import Dataset
from .simulator import make_env


def audit(cfg, manifest, out, limit=3, steps=20):
    root = new_run(out)
    results = []
    data = None
    try:
        if limit <= 0 or steps <= 0:
            raise ValueError("Positive audit limit/steps required")
        m = read_json(manifest)
        validate_manifest(m)
        if file_hash(cfg["dataset"]) != m["dataset_sha256"]:
            raise ValueError("Dataset changed")
        cases = [c for c in m["cases"] if c["split"] == "dev"][:limit]
        if not cases:
            raise ValueError("No development cases")
        data = Dataset(cfg["dataset"])
        for case in cases:
            initial, target, _ = data.case(case)
            start = int(data.offsets[case["episode"]] + case["start_step"])
            count = min(steps, int(data.lengths[case["episode"]] - case["start_step"] - 1))
            actions = data.f["action"][start : start + count]
            if not np.isfinite(actions).all():
                raise ValueError("Non-finite expert action")
            trajectories = {}
            models = {}
            for name, condition, friction in [
                ("N", "N", 1.0),
                ("I", "I", 1.0),
                ("D_identity", "D", 1.0),
                ("D", "D", cfg["friction_scale"]),
            ]:
                harness = make_env(
                    {**cfg, "friction_scale": friction}, condition, 42, initial, target
                )
                ledger = Ledger(count)
                try:
                    models[name] = harness.audit_model()
                    replay = harness.replay_check(actions[: min(3, count)], ledger)
                    traj = [harness.evaluate()]
                    for a in actions:
                        _, truth = harness.step(a, ledger, "offline_expert_replay")
                        traj.append(truth)
                    trajectories[name] = traj
                    write_json(
                        root / f"{case['case_id']}-{name}.json",
                        dict(
                            trajectory=traj,
                            model=models[name],
                            ledger=asdict(ledger),
                            replay=replay,
                            role="privileged offline diagnostic; not deployment interaction",
                        ),
                    )
                finally:
                    harness.close()

            def pos(name, key):
                return np.array([x[key] for x in trajectories[name]])

            identity = float(
                np.max(np.abs(pos("N", "object_pos") - pos("D_identity", "object_pos")))
            )
            dynamic = float(
                np.max(np.abs(pos("D", "object_pos") - pos("D_identity", "object_pos")))
            )
            interface = float(np.max(np.abs(pos("I", "effector_pos") - pos("N", "effector_pos"))))
            observed = all(any(t["contacts"] for t in trajectories[n]) for n in ("D", "D_identity"))
            # Check actual solver contact friction, not just model-array assignments.
            effective = True
            for name in ("D", "D_identity"):
                expected = {
                    tuple(sorted((p["geom1"], p["geom2"]))): p["friction"]
                    for p in models[name]["pairs"]
                }
                for t in trajectories[name]:
                    for contact in t["contacts"]:
                        effective &= bool(
                            np.allclose(
                                contact["friction"], expected[tuple(sorted(contact["geoms"]))]
                            )
                        )
            ok = identity < 1e-7 and observed and effective and dynamic > 1e-6 and interface > 1e-6
            results.append(
                dict(
                    case_id=case["case_id"],
                    status="passed" if ok else "inconclusive",
                    identity_object_max_abs=identity,
                    interface_effector_max_abs=interface,
                    contact_response_object_max_abs=dynamic,
                    contacts_observed=observed,
                    effective_friction_verified=effective,
                    note="A no-contact/no-response result is not proof of an effective perturbation",
                )
            )
        write_json(
            root / "audit.json",
            dict(cases=results, config=cfg, manifest_sha256=file_hash(manifest)),
        )
        ok = bool(results) and all(x["status"] == "passed" for x in results)
        finish_run(root, "complete" if ok else "inconclusive", diagnostic_only=True)
        return ok
    except BaseException as exc:
        finish_run(root, "failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if data:
            data.close()
