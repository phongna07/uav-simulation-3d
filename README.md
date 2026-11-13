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
# Full scen1 mission, ours (may take substantial wall time). The number of
# drones is the number of `starts` in the scenario YAML.
uv run uav3d run

# Short run; duration is simulated mission time, not wall time.
uv run uav3d run --duration 10 --out runs/ours-demo

# Sequential, isolated runs of all five algorithms, using the same preset.
uv run uav3d benchmark --duration 10 --out runs/comparison

uv run uav3d metrics runs/ours-demo
uv run uav3d replay runs/ours-demo --out exports/ours.png
uv run uav3d replay runs/ours-demo --out exports/ours.mp4 --speed 5 --fps 24
# Follow the target at 4× magnification, or restore the full-map view.
uv run uav3d replay runs/ours-demo --out exports/closeup.mp4 --zoom 4
uv run uav3d replay runs/ours-demo --out exports/overview.mp4 --camera overview

# Reuse a run's measured command delays; keep all experiment options identical.
uv run uav3d run --duration 10 --timings-from runs/ours-demo --out runs/ours-repeated

# Add recorded runs to the per-preset comparison tables in metrics/ (no simulation).
uv run uav3d collect ours-demo            # a run under runs/ (or a path)
uv run uav3d collect comparison           # every method of a benchmark directory
uv run uav3d collect --latest             # the most recently finished run
uv run uav3d collect                      # pick from a numbered list of runs
```

Select `--controller ours|apf|pure_mpc|cbf_qp_lag|impc_ob_rec`, `--scenario scen1`,
`--seed 0`, `--physics-dt 0.0025`, or `--physics-workers 1` explicitly as needed. Benchmark supports a
`--controllers` subset. Output directories must be new, preventing accidental
overwrites. A completed run or a detected collision exits successfully; the
result artifact distinguishes them. A benchmark continues after a failed child
process and returns nonzero if any worker failed.

The fleet size is not a CLI option: it is inferred from the scenario's
`starts` list (upstream sets `NUM_ROBOT = len(starts)`), so a different fleet
needs a scenario with a different `starts` list.

### Console progress

`run`, `benchmark`, `metrics` and `replay` write timestamped progress lines to
stderr (`HH:MM:SS [run ours] ...`); results stay on stdout (the final run line,
the metrics JSON, the replay path), so `uav3d metrics DIR > m.json` still
captures clean JSON. Lines never use carriage returns, so a redirected log file
(`... > /tmp/name.log 2>&1`) reads the same as the terminal.

- **run**: setup stages (scenario load, fleet size, horizon, physics workers,
  planner warm-up time), each UAV's first switch to target tracking, then a
  periodic status line: mission time and percentage, planning round, wall time,
  mission-to-wall speed and ETA, round solve time (mean/max of the slowest UAV
  per round since the last line), UAVs tracking, UAVs in watchdog braking,
  rejected stale plans, upstream stop fallbacks and skipped releases. A
  collision, interruption or error is reported as it ends the mission, followed
  by artifact writing and a one-line metrics digest.
- **benchmark**: `[k/N] controller` before and after each child run, with its
  termination, wall time and remaining-time estimate, and a final summary.
- **metrics**: sample and planner-call counts, progress through the
  visibility/coverage pass (about 10 s for a full mission), and a digest.
- **replay**: scene building and drawing stages for PNG; frame count, worker
  plan, time to first frame, and frame/rate/ETA lines for serial and parallel
  MP4 export.

`--progress-interval SECONDS` (env `UAV3D_PROGRESS_INTERVAL`, default 10) sets
the wall-time spacing of the periodic lines; `0` keeps only stage and event
lines. Status lines are produced between planning rounds, outside every timed
planner call, so they never count as planning delay and do not change any
trajectory (only `wall_time_s`/`throughput` include their negligible cost).

## Timing semantics

Scheduling is always **parallel**: one planning worker per UAV (there is no
`--scheduling` option; `config.json` records `scheduling: parallel`).
At each admitted nominal planning release, every planner observes the same
team/target snapshot, including the other UAVs' predictions from before the
round (never a same-round plan). With planning-call durations `C_i`, all
commands arrive together at:

```text
release_time + d,   d = max(C_0, ..., C_N-1)
```

Host solves still run one after another, each measured single-threaded, so
`d` models dedicated hardware rather than host parallelism. Events record each
drone's own `C_i` as `execution_s`/`injected_delay_s`; `latency_s` is `d`.

`--compute-ratio r` (default `0`) models faster (or slower) planning hardware,
e.g. a powerful onboard computer: every measured duration is multiplied by `r`
before it is injected as mission-time delay, so the
commands arrive at `release_time + r*max(C_i)`. `r=0.5` halves the delay;
`r=0` delivers every command at its release (instantaneous planning: physics is
frozen while the host solves, so no planning period is ever skipped); `r=1`
injects the measured host duration unchanged. It is
a compatibility layer on timing only; the planner calls are unchanged. Events
keep the raw host duration in `execution_s` and the scaled delay in
`injected_delay_s`; `config.json` records `compute_ratio`. `--timings-from`
divides the source run's delays by its recorded ratio and applies the new one,
so the same measured timings can be replayed at different ratios; a ratio-0
source injected no delay, so its measured `execution_s` is replayed instead.
Events also record `used_backup` (upstream's contingency branch issued the
command) and `solver_failures`, upstream's own cumulative failed/infeasible
solve counter for that UAV (read only; absent for APF).

Safety-layer activity is logged for `ours`: `trajectory.npz` has a per-sample,
per-drone `watchdog_braking` flag (the drone is past its plan's stop point and
braking to rest), events record `upstream_stop` (upstream fell back to its
backup, so the aircraft stops along its current plan) and `accepted: false`
(stale plan rejected). `metrics.json` summarizes these as
`watchdog_braking_pct_*`, `watchdog_braking_episodes`, `upstream_stop_fallbacks`
and `rejected_stale_plans`.

During planning delays RotorPy and the target advance in simulation time;
attitude and altitude control keep running. Other methods hold their latest
horizontal command. For `ours` (`command_execution: tracked_plan_with_watchdog`),
each delivered result is the whole plan, followed by mission time at the
physics rate:

- **Time-indexed plan.** `U[k]` is the feedforward on
  `[release + k*dt, release + (k+1)*dt)`; a late plan starts at its current
  step, not `U[0]`. The previous plan keeps playing until a new one is accepted.
- **Tracking.** `u = U[k] + 9*(p_plan - p) + 6*(v_plan - v)` (3 rad/s,
  critically damped) removes the drift between upstream's lag model and
  RotorPy. Only the correction may raise the command above `UMAX`, up to
  `1.5*UMAX`; the Hummingbird's thrust/weight is about 10, so `UMAX` limits the
  planner, not the aircraft.
- **Watchdog.** Each plan stops being followed as-is at the last planned step
  from which its remaining path suffices to stop at `0.8*UMAX`. The reference
  then slows to rest along that remaining, corridor-verified path and holds its
  end point. When upstream falls back to its CONTINGENCY backup (failed solve),
  the aircraft has been flying the plan, not the backup branch, so it starts this
  stop immediately and publishes it as its prediction.
- **Stale-plan rejection.** A plan solved for an earlier state is accepted only
  if, at delivery, the aircraft is within `0.5*ROBOT_RADIUS` and 2 m/s of the
  plan's state for that time. Otherwise the current schedule continues, its
  trajectory is published, and the next release replans from the real state.
  Events record `accepted: false`.
- **Speed guard.** Within 0.5 m/s of `VMAX` the speed-increasing part of the
  command is removed; above it, it brakes by `6*(VMAX - speed)`. Holding one
  command for several periods had driven aircraft far past `VMAX`, after which
  the MPC's hard speed bound is infeasible.

These gains and ratios are adapter constants in `upstream.py`, recorded in
`config.json` under `plan_tracking`. Results and predictions are published only on completion. Nominal
releases while the worker is busy are dropped, without creating a request
backlog. Upstream shared leader/slot bookkeeping is retained.

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

`--physics-workers auto|N` (default `auto`: one process per drone, capped by
available CPUs; env `UAV3D_PHYSICS_WORKERS`) steps the aircraft in parallel
processes, one physics step at a time. It is a host-speed option only and is
**bit-identical** to serial stepping (`--physics-workers 1`): each RotorPy
step depends only on that vehicle's state and command, the main process keeps
every state (so snapshot/restore and collision step splits are unchanged),
and the main process consumes RotorPy's zero-scale motor-noise draws from the
global NumPy RNG exactly as serial stepping does, so upstream's random stream
is unaffected. It applies to every controller. Workers block idle while
planners run, so planner timings are not affected. `config.json` records
`physics_workers`. Planner solves stay sequential: upstream robots share
per-round leader/slot state and (for IMPC) each other's solver objects, so
they cannot be split across processes without changing upstream.

Obstacle footprints are vertical columns through the flight region. Collision
checks sweep the scenario's safety-radius sphere against other drones, and its
horizontal disk against circles and rounded expanded rectangles between
physics samples. The earliest intersection terminates the entire team. Ground
contact, safety-envelope exit from horizontal world bounds, and nonfinite
states also fail the run. This is a conservative safety-envelope test, not
mesh-level impact dynamics. Sweeps approximate the curved motion within each
5 ms physics interval by a line. No bouncing/contact response is simulated.
A conservative bounding-box broad phase skips only obstacles, boundaries and
drone pairs that cannot be reached within a step; the remaining candidates go
through the same exact tests in the same order, so the detected hit (time,
reason, drones, and ties) is identical to testing everything.

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
mean gathering time is over the followers: upstream's leader is in TRACK from
its election at t = 0, while here the first status arrives with the first
delivered command, so a UAV whose first TRACK sample carries the leader role
(or t = 0) is excluded.

Visibility and coverage divide by the requested mission horizon, contributing
zero after first collision. `--duration` explicitly defines a shorter test
horizon; full missions use the upstream target-sample horizon. Runs that never
gather report unavailable gathering times. Re-formation SEARCH percentage is
unavailable when its observation window is shorter than upstream's 20% rule.
Execution, queue wait, and latency are in seconds; visibility and completion
are percentages; coverage efficiency is in `[0,1]`. Metrics count executed
planning calls, including a pending call that completes after termination.
Errors and interrupted runs are labeled, not presented as successful missions.

Replay reads these artifacts without importing planners. Quadcopter glyphs with
colored bodies, four rotor rings, and white nose stripes follow recorded attitude
and are enlarged for field-scale visibility; the vertical display aspect is also
exaggerated when needed. Aircraft
positions, target, footprints, and obstacles use their recorded coordinates.
The default camera follows the recorded target at 6× magnification with a
55-degree downward view and fixed heading. The camera aims at the point above
the target at the recorded flying altitude (median aircraft altitude), so the
swarm hovering over the target sits at the frame center and the target appears
below it by the projected altitude; this holds even near map edges, and distant
drones may leave the frame. Tracking pans the camera projection;
world axis limits, ground paving, and obstacle coordinates stay fixed. `--zoom` adjusts magnification,
and `--camera overview` restores the full-map view. Camera tracking does not
change the trajectory or simulation.
The scenery viewport fills the entire output frame without stretching the map;
the title sits on an opaque white overlay, sized to give the two lines equal
top and bottom padding, and the UAV legend (UAVs numbered from 1) sits below it.
The first line reads `Scenario: … · Map size: WxH m · Algorithm: … · N UAVs · Video speed: Sx ·
t = … s`, where the video speed is `--speed` (mission seconds per video second).
The second line shows the max solve time
`d = max_i C_i` of the latest planning round released at or before the frame
time, read from `events.jsonl`. `C_i` is the raw host duration (the injected
delay divided by `--compute-ratio`, or the measured duration when the ratio is
0), so `d` does not depend on the ratio. It continues with the minimum UAV
separation distance (smallest 3D centre-to-centre distance between any two
UAVs in that frame, as in the `min_inter_uav_m` metric) and the minimum UAV -
obstacles distance (smallest horizontal distance from any UAV centre to the
nearest rectangle or circle obstacle footprint, matching the collision model's
vertical obstacle columns; 0 inside one). Values are
bold. Mission time, `d` and both distances use fixed-width,
digit-wide padding so the centered title does not shift between frames. World axes and their labels are hidden for a clean video frame.
MP4 export uses an environment-managed FFmpeg binary. Replay uses dark gray
running-bond brick paving and buildings on rectangular obstacle footprints
with framed glass, entrances, parapets, and rooftop solar panels,
plus branched trees with irregular, directionally shaded leaf clusters within
circular obstacle footprints. MP4s render at three times the output resolution
then downsample to 1920 × 1280, smoothing fine window edges during camera pans.
The paving extends one world span beyond every map edge, including negative X and Y, to keep ground
visible around the origin and during camera tracking. Pavers are enlarged to
remain visible at map scale. This decoration only affects replay: collision and
LiDAR still use the original vertical obstacle columns, including the space
around tree trunks and above the decorative roofs/crowns. High-quality artistic
rendering and a live viewer are outside v1. Aircraft glyphs, trails, FOV
squares and the target path are depth-occluded by buildings and trees: each
frame, any part whose view ray toward the camera crosses an obstacle's bounding
volume is drawn beneath exactly those occluding obstacles (and any obstacle in
front of them), which are lifted out of the scenery layer and ordered far to
near. Solid facades and crowns therefore cover it and gaps between leaves still
show it, while scenery behind it (e.g. a building behind a UAV hidden by a
tree) never paints over it. Unblocked parts draw above all scenery. Aircraft
glyphs are split piece by piece: hull faces into small triangles (back faces
culled), and arms, rotor rings, hull outlines and the nose stripe into short
segments, so a building edge sweeps across the body and rotors gradually
instead of whole parts popping in or out. Hidden parts of different overlays
that share a layer are drawn in overlay construction order (a 1e-10 zorder
tie-break), so each frame depends only on its mission time, not on which
frames were drawn before it.

### Parallel MP4 export

MP4 frames are drawn by several worker processes by default. Each worker builds
its own scene, draws an interleaved share of the frames through matplotlib's
normal animation path, and runs a private ffmpeg that applies the same Lanczos
downscale and yuv420p conversion as the serial export. The main process puts
the frames back in order through small in-memory queues and streams them into
one x264 encoder (CRF 18, same settings as before). There are no intermediate
frame or segment files, so disk use is just the final MP4. It is written as a
hidden `.<name>.partial.mp4` beside the output and renamed when complete, so a
failed or interrupted export leaves no truncated video. Frames reach the encoder
exactly as a serial export produces them, so for the same settings the MP4 is
byte-identical to `--jobs 1` (checked in `tests/test_replay_video.py`).

| Flag | Environment default | Meaning |
|---|---|---|
| `--jobs auto\|N` | `UAV3D_REPLAY_JOBS` | worker processes; `auto` takes the smallest of the CPU limit `CPUs − 4`, the memory budget divided by the per-worker estimate, and one worker per 8 frames; `1` runs the original serial export |
| `--max-memory auto\|GiB` | `UAV3D_REPLAY_MAX_MEMORY_GB` | memory budget for `--jobs auto`; `auto` = 80% of available memory (cgroup limits respected) |
| `--queue-frames N` | `UAV3D_REPLAY_QUEUE_FRAMES` | delivery-size frames (3.7 MB each) a worker may buffer ahead of the encoder (default 4) |
| `--encoder-threads auto\|N` | `UAV3D_REPLAY_ENCODER_THREADS` | x264 threads (default: ffmpeg's choice); a different count may change the bitstream slightly, not the quality target |

CPU count follows the process affinity mask and any cgroup CPU quota. Each
worker needs about 2.1 GiB at the 5760 × 3840 render size: ~0.9 GiB for the
drawing process and ~1.2 GiB for its private scaling ffmpeg, which queues
several raw frames between its threads. The encoder needs about 3.5 GiB. On a
48-CPU, 47 GiB host this means `auto` picks ~14 workers rather than 44. An
explicit `--jobs` above the CPU count or the memory budget is honoured with a
warning. If the encoder dies mid-export (for example, killed by the kernel when
memory runs out), the export stops at once with an error, stops its workers,
and leaves no partial MP4 behind. Before rendering, replay warns if the output
filesystem has less than twice the estimated video size free (~40 kB per frame).
Progress (frame, frames/s, remaining time) is printed to stderr every
`--progress-interval` seconds (default 10; see [Console progress](#console-progress)).

```bash
uv run uav3d replay runs/ours-demo --out exports/ours.mp4 --speed 5 --fps 24            # auto
uv run uav3d replay runs/ours-demo --out exports/ours.mp4 --jobs 8 --max-memory 16      # shared host
UAV3D_REPLAY_JOBS=1 uv run uav3d replay runs/ours-demo --out exports/ours.mp4           # serial
```

## Comparison tables

`uav3d collect` reads finished runs (`config.json`, `result.json`,
`events.jsonl`, `trajectory.npz`) and never runs a planner. It writes
`metrics/<preset>.xlsx`, `.csv` and `.md` (git-ignored), where `<preset>` is
the scenario and `_<T>s` for a
`--duration` shorter than the route or `_full_mission` otherwise (e.g.
`scen1_150s`). Each file has one entry per algorithm: collecting a run
replaces only its algorithm's row, so each file holds the latest run of every
algorithm. A run that finished earlier than the stored one is skipped unless
`--force` is given. The CSV is the store read back on the next collection.

The paper metrics come first: the Markdown has a separate *Paper metrics*
table, the XLSX a *Paper metrics* sheet (then *All metrics* and
*Definitions*), and the CSV puts those columns first. They follow Section IV-C
of the paper and upstream `thesis_metrics.py`: gathering time (mean
follower `t_i^g`), re-formation time `rho_ref` (Eq. 33), mission visibility
and target loss `T_loss` (Eq. 34), mission coverage `C` (Eq. 35) and the
execution time per UAV and step (mean, p95). All are cut at the first
collision `t_c`. A metric the run cannot define (e.g. a short mission in which
not every UAV joins the formation, or a re-formation window shorter than 20 %
of the mission) is left empty (`–` in Markdown), and the *Paper metric notes*
entry says why. The other tables add run settings, safety (clearances,
collision), formation, tracking distances, computation (fleet round time,
latency, late delivery) and fallback/safety-layer counts; the Markdown and the
XLSX *Definitions* sheet define every entry. Compare rows only when their
compute ratio and scheduling match (runs recorded before sequential scheduling
was removed may differ); `collect` warns when a file mixes them.

## Validation

`uv run pytest -q` checks continuous collision detection, irregular timestamp
metrics, parallel delivery delays, pending-command motion, failure during a pending
solve, exact preset loading, estimator and
prediction aging, hover, acceleration/braking, and 5 ms/2.5 ms convergence.

Integration validation runs all five methods on the scenario's own fleet.
These are short smoke experiments, not evidence of full-mission safety or
paper-quality performance. The scalar CPU RotorPy implementation on this
machine is slower than real time in these checks; simulator throughput is
reported independently of algorithm execution time. Use recorded timing replay
when comparing physics resolutions so OS timing variation is not mistaken for
integration error.
