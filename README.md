# UAV simulation in 3D

A standalone RotorPy plant adapter for the pinned `motion_planning_MPC`
submodule. **The default is upstream `scen1`, unchanged:** five starts, target
trajectory, obstacle map, controller parameters, and nominal planning period.
The aircraft are physical 6-DoF quadrotors flying at controlled altitude; the
upstream algorithms remain planar.

## Setup

Python 3.12, Git, a C compiler, Python development headers, and GMP development
headers are required. On Debian/Ubuntu the native prerequisites are
`build-essential python3-dev libgmp-dev`. Docker and NVIDIA hardware are not
required.

```bash
git submodule update --init --recursive
# Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked
uv run pytest -q
```

`uv` creates `.venv` automatically. Dependencies are locked, including RotorPy
at `40b09697a6a1596d404a67f459941d46a39eedd3`, `pycddlib<3`, and `pyny3d==0.1.1`.
The latter satisfies upstream while avoiding a malformed newer wheel.
Upstream is pinned to `188edd780e40e771cba37ecd5c6a50ca37f2f89f`.
Use an editable project checkout; `UAV3D_UPSTREAM` can override the submodule
location if the adapter is installed elsewhere.

## Running

```bash
# Full scen1 mission, ours, five drones (may take substantial wall time).
uv run uav3d run

# Short run; duration is simulated mission time, not wall time.
uv run uav3d run --duration 10 --out runs/ours-demo

# Sequential, isolated runs of all five algorithms, using the same preset.
uv run uav3d benchmark --duration 10 --out runs/comparison

# Explicit nine-drone scenario overlay; upstream YAML is never rewritten.
uv run uav3d benchmark --drones 9 --duration 10 --out runs/comparison-nine

uv run uav3d metrics runs/ours-demo
uv run uav3d replay runs/ours-demo --out exports/ours.png
uv run uav3d replay runs/ours-demo --out exports/ours.mp4 --speed 5 --fps 24
# Follow the target at 4× magnification, or restore the full-map view.
uv run uav3d replay runs/ours-demo --out exports/closeup.mp4 --zoom 4
uv run uav3d replay runs/ours-demo --out exports/overview.mp4 --camera overview

# Reuse a run's measured command delays; keep all experiment options identical.
uv run uav3d run --duration 10 --timings-from runs/ours-demo --out runs/ours-repeated
```

Select `--controller ours|apf|pure_mpc|cbf_qp_lag|impc_ob_rec`, `--scenario scen1`,
`--seed 0`, or `--physics-dt 0.0025` explicitly as needed. Benchmark supports a
`--controllers` subset. Output directories must be new, preventing accidental
overwrites. A completed run or a detected collision exits successfully; the
result artifact distinguishes them. A benchmark continues after a failed child
process and returns nonzero if any worker failed.

The nine-drone overlay preserves the original five starts and the already
resolved map. Four new starts are chosen from a deterministic grid, ordered by
distance from the original team centroid and then coordinates, and checked for
obstacle and pairwise clearance. Failure to find valid starts is explicit.

## Timing semantics

The experiment represents **one shared ground-computer worker** for every
algorithm, with UAVs processed in ascending ID. At each admitted nominal
planning release, every planner observes the same team/target snapshot. If its
planning-call duration is `C_i`, command `i` arrives at:

```text
release_time + C_0 + C_1 + ... + C_i
```

During these delays RotorPy and the target advance in simulation time. The
previous horizontal command remains active; attitude and altitude control keep
running. Results and predictions are published only on completion. Nominal
releases while the worker is busy are dropped, without creating a request
backlog. Upstream shared leader/slot bookkeeping and sequential access to
already-completed plans are retained.

Host execution can be sequential and slower than real time. **Physics,
logging, metrics, and rendering wall time are not injected as planner delay.**
Planning measurements cover the upstream sensing, estimation, formation,
planning, optimization, and fallback call. Queue wait and total observation age
are reported separately. Each run uses a fresh subprocess and single-thread
numerical-library limits; benchmark runs execute sequentially. For publication
timings, avoid unrelated concurrent CPU workloads.

Numba and solver compilation are warmed on disposable robots. Live robots have
fresh maps, estimators, controller state, seeds, and empty histories; warmed
CasADi solver workspaces are reused with fresh numerical initial guesses to
avoid counting lazy solver compilation. Later problem-size growth is real work
and remains timed. Initialization wall time is recorded separately.

Prediction buffers and fallback cursors age by elapsed nominal intervals.
The target filter's transition/noise matrices use actual observation spacing.
Upstream cycle-count hysteresis is retained, so missed releases slow those
decisions. `cbf_qp_lag` retains its fixed upstream lookahead; it is not retuned
automatically for the measured computation delays. Delayed flight and model
mismatch can invalidate original safety guarantees; failures are recorded.

Identical seeds do not reproduce OS-dependent solve durations. `--timings-from`
reuses recorded delays while recomputing commands in closed loop. It requires
the same controller, scenario, seed, starts, duration, and upstream revision. The
physics step may differ for a controlled numerical-convergence comparison. It
cannot extend a recording past its captured calls. This is distinct from
`replay`, which only renders recorded aircraft poses.

## Physical model and collision definition

All methods command **world-frame horizontal acceleration in m/s²**, including
APF (which converts an internal velocity reference into acceleration). The
adapter sends RotorPy `cmd_acc = [ux, uy, g + 4(h-z) - 4*vz]`. RotorPy interprets
this as mass-normalized thrust, then simulates attitude control, motor response,
thrust limits, and Newton–Euler rigid-body dynamics. Each drone holds its
initial altitude, rather than having `z` clamped. Acceleration feedback is
measured from the plant and mapped to the upstream `a = dv/dt + D*v` state.
Upstream's artificial plant lag is not applied a second time.

The default Hummingbird has no motor noise. Aerodynamics and wind are disabled;
the physical response generally differs from upstream's predictive first-order
lag. A scenario with nonzero predictive `D_FRAC` still uses this drag-free plant;
its configured model mismatch is recorded, not silently compensated. The
native acceleration interface fixes a heading reference; yaw is not a planning
input. Ideal stabilized camera footprints and planar LiDAR remain upstream
assumptions; there is no rendered perception or camera occlusion model.

Physics/control run at 200 Hz by default, with exact event-boundary step splits
and RK45 tolerances `rtol=1e-6`, `atol=1e-9`. The backend interface supports a
future implementation; CUDA/batched physics is not implemented in v1.

Obstacle footprints are vertical columns through the flight region. Collision
checks sweep the scenario's safety-radius sphere against other drones, and its
horizontal disk against circles and rounded expanded rectangles between
physics samples. The earliest intersection terminates the entire team. Ground
contact, safety-envelope exit from horizontal world bounds, and nonfinite
states also fail the run. This is a conservative safety-envelope test, not
mesh-level impact dynamics. Sweeps approximate the curved motion within each
5 ms physics interval by a line. No bouncing/contact response is simulated.

## Artifacts and metrics

Each run saves:

- `config.json`: resolved scenario/starts, all parameters, commits, dependency
  versions, model settings, initialization cost, and numerical thread limits.
- `trajectory.npz`: mission `time`; `aircraft` shaped `(samples, drones, 17)`
  with `[x,y,z,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,rotor_speed_1..4]`;
  `target`, applied `command`, and published `status` (`TRACK`, leader, fail streak).
- `events.jsonl`: each measured planning call, observation/start/completion
  times, requested acceleration, queue wait, latency, and whether it arrived
  before termination. An undelivered last result can lie beyond the horizon.
- `plans.npz`: published predictions, their observation/publication epochs,
  and drone IDs. Prediction layout follows upstream `[x,y,z,vx,vy,vz]`.
- `result.json`, `planning.log`, and `metrics.json`/`.csv`/`.md`.

Mission metrics integrate sampled values with actual timestamp intervals.
Coverage uses upstream's exact footprint-union calculation and its metric
leader (nearest drone to target), which is distinct from the controller's
sticky leader. Gathering/re-formation use published controller modes. The
mean gathering convention follows upstream's positive join times; it excludes
zero-time joins, not a separately inferred leader identity.

Visibility and coverage divide by the requested mission horizon, contributing
zero after first collision. `--duration` explicitly defines a shorter test
horizon; full missions use the upstream target-sample horizon. Runs that never
gather report unavailable gathering times. Re-formation SEARCH percentage is
unavailable when its observation window is shorter than upstream's 20% rule.
Execution, queue wait, and latency are in seconds; visibility and completion
are percentages; coverage efficiency is in `[0,1]`. Metrics count executed
planning calls, including a pending call that completes after termination.
Errors and interrupted runs are labeled, not presented as successful missions.

Replay reads these artifacts without importing planners. Arm/attitude glyphs
are enlarged for field-scale visibility; the vertical display aspect is also
exaggerated when needed, with its scale factor labeled. Aircraft
positions, target, footprints, and obstacles use their recorded coordinates.
The default camera follows the recorded target at 3× magnification with a
55-degree downward view and fixed heading. The target stays centered even near
map edges; distant drones may leave the frame. Tracking pans the camera projection;
world axis limits, ground paving, and obstacle coordinates stay fixed. `--zoom` adjusts magnification,
and `--camera overview` restores the full-map view. Camera tracking does not
change the trajectory or simulation.
MP4 export uses an environment-managed FFmpeg binary. Replay uses dark gray
running-bond brick paving, windowed buildings on rectangular obstacle footprints,
and low-poly trees within circular obstacle footprints. Pavers are enlarged to
remain visible at map scale. This decoration only affects replay: collision and
LiDAR still use the original vertical obstacle columns, including the space
around tree trunks and above the decorative roofs/crowns. High-quality artistic
rendering and a live viewer are outside v1. Aircraft and trajectory overlays
draw above the scenery to remain readable; they are not depth-occluded by buildings.

## Validation

`uv run pytest -q` checks continuous collision detection, irregular timestamp
metrics, cumulative delays, pending-command motion, failure during a pending
solve, exact preset loading, deterministic nine-drone placement, estimator and
prediction aging, hover, acceleration/braking, and 5 ms/2.5 ms convergence.

Integration validation runs all five methods with both five and nine drones.
These are short smoke experiments, not evidence of full-mission safety or
paper-quality performance. The scalar CPU RotorPy implementation on this
machine is slower than real time in these checks; simulator throughput is
reported independently of algorithm execution time. Use recorded timing replay
when comparing physics resolutions so OS timing variation is not mistaken for
integration error.
