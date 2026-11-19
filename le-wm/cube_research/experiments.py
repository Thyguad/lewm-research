"""E0/E1/E2 orchestration; evaluator owns simulator and labels."""

from __future__ import annotations
import copy
import importlib.metadata
import pickle
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from research_common import (
    write_json,
    read_json,
    file_hash,
    runtime_metadata,
    new_run,
    finish_run,
    derived_seed,
)
from .protocol import (
    CONDITIONS,
    METHODS,
    History,
    Ledger,
    digest,
    validate_manifest,
    probe_actions,
    policy_config,
)
from .data import Dataset, local_inputs
from .adaptation import Adapter
from .diagnosis import Diagnostic, features


def protocol_identity(cfg):
    return digest(
        {
            k: v
            for k, v in cfg.items()
            if k
            not in (
                "dataset",
                "weights",
                "output_root",
                "device",
                "render_backend",
                "score_batch_size",
            )
        }
    )


def lock(cfg, manifest, out, diagnostic_path=None):
    local_inputs(cfg)
    m = read_json(manifest)
    validate_manifest(m)
    if m["dataset_sha256"] != file_hash(cfg["dataset"]):
        raise ValueError("Dataset changed")
    diagnostic_hash = None
    if diagnostic_path:
        if read_json(Path(diagnostic_path) / "STATUS.json")["state"] != "complete":
            raise ValueError("Diagnostic training did not complete")
        diagnostic_hash = file_hash(Path(diagnostic_path) / "diagnostic.npz")
    write_json(
        out,
        dict(
            diagnostic_sha256=diagnostic_hash,
            schema="cube-lock-v1",
            protocol_sha256=protocol_identity(cfg),
            manifest_sha256=file_hash(manifest),
            weights_sha256=file_hash(cfg["weights"]),
            model_config_sha256=file_hash(Path(cfg["weights"]).parent / "config.json"),
            config=cfg,
            created_from="Development choices must be fixed before held-out evaluation",
        ),
    )


def check_lock(cfg, manifest, lock_path):
    if not lock_path:
        raise ValueError("Locked test requires --lock created before testing")
    record = read_json(lock_path)
    expected = dict(
        protocol_sha256=protocol_identity(cfg),
        manifest_sha256=file_hash(manifest),
        weights_sha256=file_hash(cfg["weights"]),
        model_config_sha256=file_hash(Path(cfg["weights"]).parent / "config.json"),
    )
    if any(record[k] != v for k, v in expected.items()):
        raise ValueError("Frozen protocol/model/manifest changed; do not tune on test results")
    return record


def provenance(cfg, manifest):
    meta = runtime_metadata(Path(__file__).resolve().parents[2])
    for pkg in ("mujoco", "ogbench", "dm-control"):
        try:
            meta["packages"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            meta["packages"][pkg] = None
    meta.update(
        config=cfg,
        protocol_sha256=protocol_identity(cfg),
        manifest_sha256=file_hash(manifest),
        weights_sha256=file_hash(cfg["weights"]),
        model_config_sha256=file_hash(Path(cfg["weights"]).parent / "config.json"),
        input_boundary="RGB + goal + issued commands; optional listed proprio, never force/state/labels",
        success_protocol="first hit cube centre <=0.04m to expert-segment target; orientation not tested",
        adaptation_protocol="per episode reset; no backbone update; no cross-episode memory",
    )
    # Hash current Python sources even in a dirty working tree.
    root = Path(__file__).resolve().parents[2]
    meta["source_sha256"] = {
        str(p.relative_to(root)): file_hash(p)
        for folder in (root / "le-wm", root / "stable-worldmodel/stable_worldmodel")
        for p in sorted(folder.rglob("*.py"))
    }
    return meta


def execute(harness, history, truth, planner, model, ledger, count, phase, goal_z, scripted=None):
    for i in range(count):
        if truth[-1]["success"] or truth[-1]["truncated"] or ledger.used >= ledger.limit:
            break
        if scripted is None:
            z = model.encode([history.observations[-1].pixels], ledger)[0]
            command = planner.action(z, goal_z)
        else:
            command = scripted[i]
        obs, evaluation = harness.step(command, ledger, phase)
        history.append(command, obs)
        truth.append(evaluation)


def start_frozen(cfg, model, seed, ledger, harness, goal):
    from .model import Planner

    goal_z = model.encode([goal], ledger)[0]
    history, truth = History([harness.observe()]), [harness.evaluate()]
    adapter = Adapter(model, "Frozen", policy_config(cfg), ledger)
    return history, truth, goal_z, adapter, Planner(adapter, policy_config(cfg), seed)


def save_history(path, history, truth, goal):
    np.savez_compressed(path / "allowed.npz", **history.arrays(), goal=goal)
    write_json(path / "evaluation.json", truth)


def recover(harness, history, truth, model, goal_z, method, cfg, seed, ledger, rng_state=None):
    from .model import Planner

    adapter = Adapter(model, method, policy_config(cfg), ledger)
    if method != "Frozen" and not (truth[-1]["success"] or truth[-1]["truncated"]):
        embeddings = model.encode([o.pixels for o in history.observations], ledger)
        adapter.fit(embeddings, np.asarray(history.actions))
    planner = Planner(
        adapter, policy_config(cfg), seed
    )  # always fresh optimizer, RNG and empty action buffer
    if rng_state is not None:
        planner.rng.bit_generator.state = copy.deepcopy(rng_state)
    execute(
        harness,
        history,
        truth,
        planner,
        model,
        ledger,
        ledger.limit - ledger.used,
        "recovery",
        goal_z,
    )
    return adapter


def run(
    cfg,
    manifest,
    out,
    stage,
    split="dev",
    limit=3,
    lock_path=None,
    diagnostic_path=None,
    smoke=False,
    case_id=None,
    cohort=None,
    paired_e0=None,
    *,
    cohort_position=None,
    perturbations=None,
    extension_seal=None,
):
    from .model import FrozenModel, Planner
    from .simulator import make_env

    root = new_run(out)
    started = time.perf_counter()
    completed = []
    failures = []
    actual_steps = 0
    actual_offline_steps = 0
    write_json(
        root / "request.json",
        dict(
            config=cfg,
            stage=stage,
            split=split,
            limit=limit,
            case_id=case_id,
            smoke=smoke,
            manifest=str(manifest),
            cohort=str(cohort) if cohort else None,
            cohort_position=cohort_position,
            paired_e0=str(paired_e0) if paired_e0 else None,
            runtime=runtime_metadata(),
        ),
    )
    try:
        if perturbations and not (cohort and stage == "e0" and extension_seal):
            raise ValueError("Perturbation grid requires sealed cohort E0")
        base_cfg = dict(cfg)
        variants = [(p["condition"], p) for p in perturbations] if perturbations else [
            (c, {}) for c in CONDITIONS
        ]
        if cohort_position is not None and not (cohort and stage == "e0"):
            raise ValueError("Cohort position is only for cohort E0")
        local_inputs(cfg)
        m = read_json(manifest)
        validate_manifest(m)
        if any(c["goal_offset"] != cfg["goal_offset"] for c in m["cases"]):
            raise ValueError("Goal offset differs from the case manifest")
        if m["protocol"] != cfg["protocol"] or m["dataset_source"] != cfg["dataset_source"]:
            raise ValueError("Manifest and config protocol/data identity mismatch")
        if cohort:
            from .cohort import load_cohort

            source_cfg = ({k: v for k, v in cfg.items()
                           if k not in ('interface_parameterization', 'experiment_protocol')}
                          if perturbations else cfg)
            selected_cases = load_cohort(source_cfg, manifest, m, cohort, stage, split, case_id, smoke,
                                         paired_e0, position=cohort_position)
        elif m["dataset_sha256"] != file_hash(cfg["dataset"]):
            raise ValueError("Dataset content changed after split/stats freezing")
        if split == "test":
            locked = check_lock(cfg, manifest, lock_path)
        if stage == "collect" and split == "test":
            raise ValueError(
                "Collect/train only on dev/train; evaluate locked test with --stage evaluate"
            )
        diagnostic = None
        if stage == "evaluate":
            if split != "test" or not diagnostic_path:
                raise ValueError("evaluate requires locked test and --diagnostic")
            if read_json(Path(diagnostic_path) / "STATUS.json")["state"] != "complete":
                raise ValueError("Diagnostic training did not complete")
            if locked.get("diagnostic_sha256") != file_hash(
                Path(diagnostic_path) / "diagnostic.npz"
            ):
                raise ValueError("Lock must include this frozen diagnostic via lock --diagnostic")
            metadata = read_json(Path(diagnostic_path) / "metadata.json")
            if metadata["manifest_sha256"] != file_hash(manifest) or metadata[
                "protocol_sha256"
            ] != protocol_identity(cfg):
                raise ValueError("Diagnostic training protocol mismatch")
            if metadata["weights_sha256"] != file_hash(cfg["weights"]):
                raise ValueError("Diagnostic used a different world model")
            diagnostic = Diagnostic.load(Path(diagnostic_path) / "diagnostic.npz")
        cases = selected_cases[:limit] if cohort else [
            x
            for x in m["cases"]
            if x["split"] == split and (case_id is None or x["case_id"] == case_id)
        ][:limit]
        if not cases:
            raise ValueError("No cases in requested split")
        strategies = (
            [("passive", 0)]
            if stage in ("e0", "e1")
            else [("passive", 0)]
            + [
                (s, b)
                for s in ("continue", "fixed", "random")
                for b in cfg["extra_budgets"]
                if b > 0
            ]
        )
        methods = (
            ["Frozen"] if stage == "e0" else list(METHODS) if stage != "evaluate" else ["Selected"]
        )
        plan = [
            dict(
                case_id=c["case_id"],
                condition=cond,
                planning_seed=seed,
                strategy=s,
                extra_budget=b,
                method=method,
                **{k: v for k, v in variant.items() if k != "condition"},
            )
            for c in cases
            for seed in cfg["planning_seeds"]
            for cond, variant in variants
            for s, b in strategies
            for method in methods
        ]
        write_json(root / "plan.json", plan)
        meta = provenance(cfg, manifest)
        meta.update(stage=stage, split=split, smoke=smoke, case_count=len(cases))
        if perturbations:
            meta.update(extension_seal=extension_seal, experiment_protocol=cfg["experiment_protocol"])
        if cohort:
            meta.update(cohort_sha256=file_hash(cohort), cohort_scope="selected dev only")
            if stage == "e0":
                meta["cohort_position"] = 1 if cohort_position is None else cohort_position
        if diagnostic_path:
            meta["diagnostic_sha256"] = file_hash(Path(diagnostic_path) / "diagnostic.npz")
            meta["diagnostic_metadata"] = read_json(Path(diagnostic_path) / "metadata.json")
        write_json(root / "metadata.json", meta)
        data = Dataset(cfg["dataset"])
        model = FrozenModel(cfg, m["stats"])
        write_json(root / "model.json", model.identity)
        try:
            for case in cases:
                initial, target, goal = data.case(case)
                for seed in cfg["planning_seeds"]:
                    for condition, variant in variants:
                        variant_fields = {k: v for k, v in variant.items() if k != "condition"}
                        key = variant.get("perturbation_id", condition)
                        harness = None
                        history = truth = None
                        prefix_ledger = None
                        try:
                            env_cfg = dict(base_cfg, interface_scale=variant["xyz_gain"],
                                           friction_scale=variant["pad_friction_scale"]) if variant else cfg
                            harness = make_env(
                                env_cfg,
                                condition,
                                derived_seed(cfg["environment_seed"], case["case_id"]),
                                initial,
                                target,
                            )
                            prefix_started = time.perf_counter()
                            prefix_ledger = Ledger(cfg["total_steps"])
                            history, truth, goal_z, frozen, prefix_planner = start_frozen(
                                cfg, model, seed, prefix_ledger, harness, goal
                            )
                            execute(
                                harness,
                                history,
                                truth,
                                prefix_planner,
                                model,
                                prefix_ledger,
                                cfg["total_steps"] if stage == "e0" else cfg["history_steps"],
                                "normal_history",
                                goal_z,
                            )
                            if variant:
                                from .cohort_pair import assert_commands

                                assert_commands(history.actions, truth, variant["xyz_gain"])
                            # Snapshot contains controller, physics, random streams and
                            # prefix action buffer; restore it before EVERY strategy.
                            prefix_seconds = time.perf_counter() - prefix_started
                            snap = harness.snapshot(prefix_planner)
                            rng_state = copy.deepcopy(prefix_planner.rng.bit_generator.state) if cohort else None
                            check_actions = np.array([[0.02, 0, 0, 0, 0], [0, 0.02, 0, 0, 0]])
                            check_started = time.perf_counter()
                            check = harness.replay_check(
                                check_actions, prefix_ledger, prefix_planner
                            )
                            prefix_check_seconds = time.perf_counter() - check_started
                            for strategy, budget in strategies:
                                episode_started = time.perf_counter()
                                harness.restore(snap, prefix_planner)
                                h = copy.deepcopy(history)
                                t = copy.deepcopy(truth)
                                ledger = copy.deepcopy(prefix_ledger)
                                # Uniform decision boundary flush for every strategy.
                                planner = Planner(
                                    Adapter(model, "Frozen", policy_config(cfg), ledger),
                                    policy_config(cfg),
                                    seed,
                                )
                                if rng_state is not None:
                                    planner.rng.bit_generator.state = copy.deepcopy(rng_state)
                                probe_name = None
                                phase = None
                                if strategy in ("fixed", "random"):
                                    probe_name = (
                                        cfg["fixed_probe"]
                                        if strategy == "fixed"
                                        else np.random.default_rng(
                                            derived_seed(cfg["probe_seed"] + seed, case["case_id"])
                                        )
                                        .choice(["x_cycle", "y_cycle", "lift_cycle"])
                                        .item()
                                    )
                                    actions, phase = probe_actions(
                                        probe_name, budget, cfg["probe_amplitude"], h
                                    )
                                    execute(
                                        harness,
                                        h,
                                        t,
                                        planner,
                                        model,
                                        ledger,
                                        budget,
                                        "probe_including_return",
                                        goal_z,
                                        actions,
                                    )
                                elif strategy == "continue":
                                    execute(
                                        harness,
                                        h,
                                        t,
                                        planner,
                                        model,
                                        ledger,
                                        budget,
                                        "continue_task",
                                        goal_z,
                                    )
                                hid = digest([case["case_id"], key, seed, strategy, budget])[
                                    :20
                                ]
                                hp = root / "histories" / hid
                                hp.mkdir(parents=True)
                                save_history(hp, h, t, goal)
                                write_json(
                                    hp / "metadata.json",
                                    dict(
                                        case=case,
                                        condition=condition,
                                        planning_seed=seed,
                                        strategy=strategy,
                                        extra_budget=budget,
                                        actual_extra_steps=ledger.used - prefix_ledger.used,
                                        prefix_success=bool(truth[-1]["success"]),
                                        prefix_steps=prefix_ledger.used,
                                        probe=probe_name,
                                        observable_phase=phase,
                                        history_sha256=h.fingerprint(),
                                        ledger=asdict(ledger),
                                        restore_check=check,
                                        truth=harness.audit_model(),
                                        **variant_fields,
                                    ),
                                )
                                check_started = time.perf_counter()
                                branch_check = harness.replay_check(check_actions, ledger)
                                snapshot_check_seconds = prefix_check_seconds + time.perf_counter() - check_started
                                write_json(hp / "branch-replay.json", branch_check)
                                recovery_snap = harness.snapshot()
                                # Trusted local offline checkpoint, never passed to policy/diagnostic.
                                with (hp / "evaluator_snapshot.pkl").open("wb") as f:
                                    pickle.dump(recovery_snap, f)
                                selected = None
                                if diagnostic:
                                    emb = model.encode([o.pixels for o in h.observations], ledger)
                                    selected = diagnostic.predict(
                                        features(emb, h, goal_z, cfg["total_steps"])
                                    )
                                acquisition_seconds = time.perf_counter() - episode_started
                                for requested in methods:
                                    recovery_started = time.perf_counter()
                                    method = (
                                        selected["method"] if requested == "Selected" else requested
                                    )
                                    row = dict(
                                        case_id=case["case_id"],
                                        episode=case["episode"],
                                        condition=condition,
                                        planning_seed=seed,
                                        strategy=strategy,
                                        extra_budget=budget,
                                        method=requested,
                                        selected_method=method,
                                        history_id=hid,
                                        split=split,
                                        history_sha256=h.fingerprint(),
                                        status="running",
                                        **variant_fields,
                                    )
                                    rid = digest(row)[:20]
                                    if variant:
                                        row['snapshot_replay_seconds'] = snapshot_check_seconds
                                    rp = root / "recoveries" / rid
                                    rp.mkdir(parents=True)
                                    write_json(rp / "result.json", row)
                                    harness.restore(recovery_snap)
                                    hh, tt, ll = (
                                        copy.deepcopy(h),
                                        copy.deepcopy(t),
                                        copy.deepcopy(ledger),
                                    )
                                    adapter = None
                                    try:
                                        adapter = (
                                            recover(
                                                harness,
                                                hh,
                                                tt,
                                                model,
                                                goal_z,
                                                method,
                                                cfg,
                                                seed,
                                                ll,
                                                rng_state,
                                            )
                                            if stage != "e0"
                                            else frozen
                                        )
                                        save_history(rp, hh, tt, goal)
                                        write_json(rp / "adaptation.json", adapter.record())
                                        success = any(x["success"] for x in tt)
                                        row.update(
                                            status="complete",
                                            success=success,
                                            distance=tt[-1]["distance"],
                                            task_loss=float(not success),
                                            initial_success=bool(tt[0]["success"]),
                                            prefix_success=bool(truth[-1]["success"]) if stage != "e0" else None,
                                            recovery_state=("not_applicable" if stage == "e0" else
                                                            "early_success" if t[-1]["success"] else
                                                            "truncated" if t[-1]["truncated"] else "decision"),
                                            termination="success"
                                            if success
                                            else "environment_truncated"
                                            if tt[-1]["truncated"]
                                            else "budget",
                                            ledger=asdict(ll),
                                            environment_steps=ll.used,
                                            first_success_step=next(
                                                (i for i, x in enumerate(tt) if x["success"]), None
                                            ),
                                            diagnostic=selected,
                                            wall_seconds=prefix_seconds
                                            + acquisition_seconds
                                            + time.perf_counter()
                                            - recovery_started,
                                            recovery_seconds=time.perf_counter() - recovery_started,
                                            cumulative_distance=float(
                                                sum(x["distance"] for x in tt[1:])
                                            ),
                                            requested_extra_steps=budget,
                                            actual_extra_steps=ledger.used - prefix_ledger.used,
                                        )
                                        if smoke and method != "Frozen" and not adapter.trace:
                                            raise RuntimeError(
                                                "Smoke did not exercise adaptation; choose a nontrivial development case"
                                            )
                                        completed.append(row)
                                    except Exception as exc:
                                        row.update(
                                            status="failed",
                                            reason=f"{type(exc).__name__}: {exc}",
                                            ledger=asdict(ll),
                                            environment_steps=ll.used,
                                            last_attempted_command=getattr(
                                                harness, "last_attempted_command", None
                                            ),
                                        )
                                        if not (rp / "allowed.npz").exists():
                                            np.savez_compressed(
                                                rp / "allowed.npz", **hh.arrays(), goal=goal
                                            )
                                        if not (rp / "evaluation.json").exists():
                                            write_json(rp / "evaluation.json", tt)
                                        failures.append(row)
                                        if variant:
                                            write_json(rp / "result.json", row, overwrite=True)
                                            raise
                                    write_json(rp / "result.json", row, overwrite=True)
                        except Exception as exc:
                            failure = dict(
                                case_id=case["case_id"],
                                condition=condition,
                                planning_seed=seed,
                                status="failed",
                                reason=f"{type(exc).__name__}: {exc}",
                                **variant_fields,
                            )
                            if history is not None and truth is not None:
                                partial = (
                                    root / "failed-prefix" / f"{case['case_id']}-{seed}-{key}"
                                )
                                partial.mkdir(parents=True, exist_ok=True)
                                save_history(partial, history, truth, goal)
                                failure["ledger"] = asdict(prefix_ledger)
                            failures.append(failure)
                            write_json(
                                root / f"failure-{case['case_id']}-{seed}-{key}.json", failure
                            )
                            if variant:
                                raise
                        finally:
                            if harness:
                                actual_steps += getattr(harness, "actual_steps", 0)
                                actual_offline_steps += getattr(
                                    harness, "actual_offline_replay_steps", 0
                                )
                                harness.close()
        finally:
            data.close()
        if cohort and stage == "e0" and not perturbations and not failures and len(completed) == len(plan):
            from .cohort_pair import verify_pair

            offline = verify_pair(cfg, m, cases[0], root)
            actual_steps += offline
            actual_offline_steps += offline
        finish_run(
            root,
            "complete" if not failures and len(completed) == len(plan) else "incomplete",
            expected=len(plan),
            completed=len(completed),
            failures=failures,
            wall_seconds=time.perf_counter() - started,
            evaluator_actual_environment_steps=actual_steps,
            offline_replay_steps=actual_offline_steps,
        )
        return not failures and len(completed) == len(plan)
    except BaseException as exc:
        finish_run(
            root,
            "failed",
            reason=f"{type(exc).__name__}: {exc}",
            evaluator_actual_environment_steps=actual_steps,
            offline_replay_steps=actual_offline_steps,
            wall_seconds=time.perf_counter() - started,
        )
        raise


def train(cfg, manifest, roots, out):
    root = new_run(out)
    try:
        m = read_json(manifest)
        validate_manifest(m)
        rows = []
        x = []
        groups = []
        labels = []
        losses = []
        seen = set()
        for source in roots:
            source = Path(source)
            meta = read_json(source / "metadata.json")
            if meta["split"] != "train" or meta["stage"] != "collect" or meta["smoke"]:
                raise ValueError("Training accepts only real training-split collect runs")
            if meta["weights_sha256"] != file_hash(cfg["weights"]) or meta[
                "model_config_sha256"
            ] != file_hash(Path(cfg["weights"]).parent / "config.json"):
                raise ValueError("Training outcomes used a different checkpoint")
            if meta["protocol_sha256"] != protocol_identity(cfg) or meta[
                "manifest_sha256"
            ] != file_hash(manifest):
                raise ValueError("Training data protocol/manifest mismatch")
            if read_json(source / "STATUS.json")["state"] != "complete":
                raise ValueError(
                    "Incomplete collection: preserve and resolve failed/missing rows before training"
                )
            results = [read_json(p) for p in sorted((source / "recoveries").glob("*/result.json"))]
            for hp in sorted((source / "histories").iterdir()):
                hm = read_json(hp / "metadata.json")
                cid = hm["case"]["case_id"]
                if not any(c["case_id"] == cid and c["split"] == "train" for c in m["cases"]):
                    raise ValueError("Training case outside manifest training partition")
                key = (
                    cid,
                    hm["condition"],
                    hm["planning_seed"],
                    hm["strategy"],
                    hm["extra_budget"],
                )
                if key in seen:
                    raise ValueError("Duplicate training history")
                seen.add(key)
                outcomes = {r["method"]: r for r in results if r["history_id"] == hp.name}
                if set(outcomes) != set(METHODS) or any(
                    r["status"] != "complete" for r in outcomes.values()
                ):
                    raise ValueError("Every training history needs four completed paired outcomes")
                rows.append((hp, hm))
                groups.append(cid)
                labels.append(CONDITIONS.index(hm["condition"]))
                losses.append([outcomes[a]["task_loss"] for a in METHODS])
        from .model import FrozenModel

        model = FrozenModel(cfg, m["stats"])
        ledger = Ledger(0)
        from .protocol import Observation

        for hp, hm in rows:
            with np.load(hp / "allowed.npz", allow_pickle=False) as a:
                h = History(
                    [Observation(p, s) for p, s in zip(a["pixels"], a["sensors"])],
                    list(a["commands"]),
                )
                emb = model.encode(a["pixels"], ledger)
                gz = model.encode([a["goal"]], ledger)[0]
                x.append(features(emb, h, gz, cfg["total_steps"]))
        diagnostic = Diagnostic()
        fit = diagnostic.fit(
            np.asarray(x), np.asarray(labels), np.asarray(losses), groups, cfg["split_seed"]
        )
        diagnostic.save(root / "diagnostic.npz")
        write_json(
            root / "metadata.json",
            dict(
                **fit,
                protocol_sha256=protocol_identity(cfg),
                manifest_sha256=file_hash(manifest),
                weights_sha256=file_hash(cfg["weights"]),
                sources=[
                    dict(
                        path=str(p),
                        metadata_sha256=file_hash(Path(p) / "metadata.json"),
                        outcomes_sha256={
                            str(q.relative_to(p)): file_hash(q)
                            for q in sorted((Path(p) / "recoveries").glob("*/result.json"))
                        },
                    )
                    for p in roots
                ],
                total_histories=len(rows),
                ledger=asdict(ledger),
                architecture="full padded RGB pools + latent + commands + declared proprio; dual ridge; temperature calibration",
            ),
        )
        finish_run(root, "complete")
    except BaseException as exc:
        finish_run(root, "failed", reason=f"{type(exc).__name__}: {exc}")
        raise
