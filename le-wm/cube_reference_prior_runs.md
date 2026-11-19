# Cube reference audit: prior runs

Read-only audit of saved runs at `research/cube-e0-e2` / `f28fa97`. No new rollouts. Run paths below are relative to `runs/cube/`.

## Cases and executed outcomes

`cases.json` (file SHA-256 `178a47350ae8c9001d319394b9ab75558b5a02c8d132b9c9dfeb24ec3845be86`, matching the E0 `metadata.json` files) and `e0-dev/plan.json` identify these three **dev** cases, in manifest order. These are zero-based within-episode steps, not global HDF5 rows or a reference 50-case sample.

| Case ID | Episode | Start -> goal | `e0-dev` N/I/D |
| --- | ---: | ---: | --- |
| `e70853564fc1d758` | 9539 | 72 -> 97 | Success at t=0 in all three; zero actions |
| `e1db09ce9f216bf6` | 9879 | 45 -> 70 | Budget failure at 50 steps in all three |
| `e22d467bfaaa9357` | 9989 | 17 -> 42 | Budget failure at 50 steps in all three |

Sources: `e0-dev/{STATUS,plan}.json`, `e0-dev/recoveries/*/result.json`, `e0-summary-2/summary.json`. Nine runs completed, but only three independent cases: 1/3 per condition, all from the same initial success (distance ~`1.58e-8 m`). There were no post-action successes in this run and no evidence that I/D caused these failures.

| Later E0 run(s) | `e1db09ce9f216bf6`, N | `e22d467bfaaa9357`, N |
| --- | --- | --- |
| `e0-normalized-cem` | Fail, 0.1630 m | Fail, 0.2450 m |
| `e0-dataset-initial` / `e0-dataset-initial-e22d` | Fail, 0.2163 m | Fail, 0.2450 m |
| `e0-high-cem` / `e0-high-cem-e22d` | Success at step 43, 0.0375 m | Fail, 0.2301 m |
| `e0-h1-e22d` / `e0-h1-e22d-2` | Not run | Fail, 0.2450 m in both |
| `e0-fullplan-e22d` | Not run | Fail, 0.2465 m |
| `e0-mean-cem-e22d` / `e0-cpu-e22d-auto` | Not run | Fail, 0.2450 m in both |

Distances are final; each listed failure spent 50 steps. The high-budget e1db also succeeded in D at step 45, but not I. Both high-budget requests differ from `e0-normalized-cem/request.json` only in samples and iterations within `config`: **300/30/5**, still 5 steps per replan, not either top-30 reference setting. Their planner source hashes match the normalized run. `e0-fullplan-e22d` used **30/2/5** with 25 steps per replan. Both dataset-initial runs logged a first RGB exactly equal to the HDF5 row; the old `e0-dev` and normalized runs did not. The dataset-initial planner/source hashes also changed, so this is not a clean image-only comparison against the normalized run. Sources: each run's `request.json`, `metadata.json`, `histories/*/allowed.npz`, `recoveries/*/result.json`; read-only HDF5 rows `1985724` and `2007806`. `e0-cpu-e22d` (no `-auto`) was **incomplete, 0/3**, with `MUJOCO_GL=egl` rejected, not a policy failure (`STATUS.json`).

## Expert replay and score comparability

- `expert-replay-2/<case_id>.json`: both expert replays used 25 actions and their trajectories exactly match the corresponding `planning-diagnosis/<case_id>-expert.json`. e22d first succeeded at step 24, ended at `0.005421 m`, and had `5.42 mm` maximum position error vs the dataset. e1db first succeeded at step 18, ended at `0.002782 m`, and had `3.09 mm` maximum error. This establishes reachability under the replay setup, not official reset/controller or policy-input parity.
- **Comparable within the old research scorer.** `planning-diagnosis/diagnose.py:11-32` uses the same logged `e0-dev` N first RGB and HDF5 goal RGB for all candidates within each case. The expert and sampled CEM actions are both `5 x 5 x 5` (25 steps, 5 controls each) and are re-scored together via `FrozenModel.predict` and the same final squared latent L2. The bridge standardizes each 5D action and flattens each block to 25D ([model.py](cube_research/model.py#L48)). `report.json` records MPS and scaler max differences of `0` (mean) and `2.68e-14` (scale) against a `StandardScaler` fitted on all recorded actions. e22d expert/CEM/zero: **18.59 / 281.31 / 367.36**; e1db: **29.00 / 305.72 / 342.36**. e22d's diagnostic CEM replay had no logged contacts and no meaningful object motion (max coordinate change `<1e-16 m`).
- **Not a score for the closed-loop trajectory or the reference solver.** The diagnostic reconstructed the old raw-command, clipped `30/2/5` search and chose the last iteration's best sample, not the final mean or the best-ever sample (e1db best-ever was `304.73`). Against `e0-dev/histories/*/allowed.npz`, the replayed first block exactly matches both online command logs, but the second block differs by up to `1.58` (e22d) and `1.97` (e1db) in command units: the diagnostic executes all 25 actions open-loop. The expert's lower cost establishes that this old initial search missed a sequence the same research scorer rated much better. These records do not test official `LeWM.rollout/get_cost` on identical candidates, official CEM, or CPU/MPS score parity, and cannot attribute N failures to a checkpoint or official-planner limit.
