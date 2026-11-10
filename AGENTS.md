# Agent guide

This repository is a RotorPy plant **adapter** around the pinned
`motion_planning_MPC` git submodule. The upstream algorithms are the subject of
study; the adapter (`src/uav3d/`) only connects them to a 6-DoF quadrotor plant,
a scheduler, logging, metrics, and replay. See `README.md` for setup, CLI usage,
and timing semantics, and `VALIDATION.md` for what has been validated.

## Rule 1: never modify the upstream submodule

- Do not edit, add, delete, or reformat any file under `motion_planning_MPC/`.
- Do not commit a new submodule pointer, check out another upstream revision,
  or rewrite upstream YAML/scenario files. Scenario changes are overlays built
  in the adapter (as the nine-drone overlay already is).
- Do not monkey-patch upstream functions to change algorithm behavior.
- Before finishing, confirm the submodule is clean and still pinned:

  ```bash
  git -C motion_planning_MPC status --porcelain   # must print nothing
  git submodule status                            # must show 188edd7… with no +/- prefix
  ```

If a task seems to require changing upstream, stop and explain why to the user
instead of editing it.

## Rule 2: the adapter preserves upstream, then adds safety and compatibility

Every edit in `src/uav3d/` (especially `upstream.py`, `scheduler.py`,
`physics.py`, `experiment.py`) must keep as much of the core algorithm inside
upstream as possible:

- **Call upstream, don't reimplement it.** Sensing, estimation, formation,
  planning, optimization, and fallback logic stay in upstream code paths. Do not
  copy upstream logic into the adapter and alter it.
- **Wrap at the boundary.** Adapter code may translate inputs and outputs
  (state conversion, units, planar ↔ 3D, timing, command delivery) and must
  leave the decision logic in between untouched.
- **Safety layers are allowed and expected** when they sit outside the
  algorithm: command saturation, braking once a plan sequence runs out,
  altitude/attitude control, collision detection, timeouts, and explicit
  failure reporting. Keep each one small and visible, and record it in the run
  artifacts (for example `config.json`) so results stay interpretable.
- **Compatibility layers are allowed** to bridge the plant, scheduler, or new
  scenarios (e.g. more drones) to upstream interfaces, provided upstream
  defaults (`scen1`, its parameters and nominal period) stay reproducible
  unchanged.
- Prefer the least invasive option. If a safety or compatibility layer changes
  what an algorithm effectively does, say so in the README and in your summary.
- Keep timing honest: physics, logging, metrics, and rendering wall time must
  never be counted as planner delay.

## Rule 3: validate every edit with a short mission

After **every** code change, before reporting done:

1. Run the test suite:

   ```bash
   uv run pytest -q
   ```

2. Run a short mission into a new output directory (directories must not exist
   yet), then recompute its metrics:

   ```bash
   uv run uav3d run --duration 2 --out runs/check-<short-name>
   uv run uav3d metrics runs/check-<short-name>
   ```

   If the change touches a specific controller, scheduling mode, drone count, or
   the benchmark path, validate that path too, e.g.
   `--controller apf`, `--scheduling sequential`, `--drones 9`, or
   `uav3d benchmark --duration 2 --out runs/check-<short-name>-bench`.

3. Report the outcome faithfully: tests passed/failed, whether the mission
   completed or recorded a collision, and any command-delivery gaps. Do not
   claim success for a step that was skipped.

`runs/` and `exports/` are git-ignored; do not commit their contents.

## Rule 4: generate a video when the user asks for one

If the user's prompt asks for a video (or animation / MP4 / replay clip), do
produce one, rendering from a recorded run rather than describing how to:

```bash
uv run uav3d run --duration 10 --out runs/<name>
uv run uav3d replay runs/<name> --out exports/<name>.mp4 --speed 5 --fps 24
```

Use `--zoom N` to follow the target or `--camera overview` for the full map
when requested. Re-use an existing run directory if it already matches the
request. Confirm the MP4 file exists and is non-empty, then give the user its
path (and send the file when the environment allows it). If no video was
requested, a PNG replay is optional, not required.

## Workflow notes

- Python 3.12, dependencies via `uv sync --locked`; do not loosen pins in
  `pyproject.toml` / `uv.lock` without a reason.
- Add or update tests in `tests/` for adapter behavior you change.
- Update `README.md` when CLI flags, timing semantics, or safety/compatibility
  layers change.
