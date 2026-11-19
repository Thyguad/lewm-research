# Repository guidance

## Current scope

This workspace supports new planning-method research with frozen LeWM. Follow the user's current task and its selected design. No new method is selected by default. The earlier Cube adaptation campaign is historical context, not the mandatory research agenda.

Read the root README first. Read the relevant current entry-point guide and implementation only as needed. `docs/history.md` and `experiments/` contain historical evidence; consult them when reproducing or interpreting that work, not as default implementation instructions. Upstream READMEs describe their own projects, not this workspace's current research plan.

## Reuse without inheriting an old experiment

- Reuse model loading, preprocessing, device support, environment execution, action conversion, snapshots and logging when appropriate.
- Do not automatically invoke cohort screening, fixed-case perturbation grids, recovery gates, or E0→E1→E2 because an old module or result mentions them.
- Use a new entry point or explicit method configuration for a new experiment. State any protocol change, including action execution length and replanning frequency.
- Select evaluation cases by the declared protocol, not by whether the baseline succeeds. Previously inspected cases are development evidence.
- Keep task success, model prediction quality and implementation checks distinct. A historical 3-case result is not a benchmark success rate.

## Engineering

Preserve compatible PushT, CPU, MPS and CUDA paths. Resolve paths and devices from configuration. Keep weights, datasets and generated caches out of commits. Do not delete results or untracked user files as a side effect of code cleanup.

Run checks relevant to changed behavior. Report missing real-data/device validation accurately. Existing generic adaptation or diagnostic components may remain for comparison; their presence is not authorization to launch training or batch experiments.

When a document stops describing the current entry point, replace it with a short current guide and link its exact historical commit from `docs/history.md`; do not leave contradictory task instructions in active READMEs.
