# Implementation validation

Validated locally with Python 3.12.3, CPU-only RotorPy, and the versions in
`uv.lock`. The upstream submodule stayed at
`188edd780e40e771cba37ecd5c6a50ca37f2f89f` without tracked changes.

- A second, empty environment successfully installed with `uv sync --locked`.
  All five algorithm modules and the RotorPy plant imported successfully there.
- The automated suite passed 18 tests, covering preset defaults, deterministic
  starts, collision sweeps, timing, estimator/prediction aging, metrics, and
  physical response/convergence.
- All five algorithms completed two mission seconds of upstream `scen1` with
  five drones and nine drones. Each delivered a command to all five aircraft.
  An additional four-second `impc_ob_rec` run verified command delivery to all
  nine aircraft; its initial two-second run ended while the ninth result was
  pending. The other nine-drone runs delivered to all nine within two seconds.
- Recomputed commands using the five-drone `ours` run's recorded delays
  reproduced `time`, `aircraft`, `target`, `command`, and `status` arrays exactly.
- A controlled `ours` repeat using the same delays and a 2.5 ms physics step
  differed from the 5 ms trajectory by at most 0.008854 m in position and
  0.007691 m/s in velocity over the two-second check. This includes the
  closed-loop change in commands caused by the slightly different plant state.
- PNG and H.264 MP4 exports succeeded from recorded artifacts, without rerunning
  algorithms. `metrics` also successfully recomputed the saved report.

Local artifacts (ignored by Git):

```text
runs/validation-five/
runs/validation-nine/
runs/validation-nine-impc-extended/
runs/timing-replay-ours/
runs/convergence-ours-fine/
exports/scen1-ours.png
exports/scen1-apf.mp4
```

These are integration smoke runs, not full missions or publishable comparative
benchmarks. The scalar CPU backend ran slower than real time. Physics and
rendering costs are excluded from injected planner delay; real planner costs
and shared-worker queue waits remain included. No full-mission safety or
GPU-speed claim is made.
