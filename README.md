# LeWM frozen diagnostics workspace

This monorepo keeps the LeWM research checkout and its stable-worldmodel dependency together:

- `le-wm/`: the personal LeWM fork, branch `research/frozen-diagnostics`.
- `stable-worldmodel/`: the dependency source used by the research scripts.

The research scripts and Chinese execution guide are under `le-wm/`. Runtime data, checkpoints, virtual environments, and experiment outputs are ignored by the top-level Git configuration.

The original child repository metadata and the pre-existing stable-worldmodel working-tree state were backed up outside this workspace in `../lewm-research-git-backup-20261117/` before migration. The dependency had local changes at migration time; inspect and explicitly decide whether to preserve those changes before server deployment.

This repository is a local workspace repository. The former `le-wm` GitHub fork remains available at https://github.com/Thyguad/le-wm; create a separate remote for this monorepo only if and when a GitHub monorepo is desired.

## Cube E0–E2 research

The independent Cube implementation, exact commands, protocol boundaries and
validation limits are in [le-wm/README_cube.md](le-wm/README_cube.md).
The existing PushT experiments and commands are preserved.
