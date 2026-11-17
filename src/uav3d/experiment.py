"""Single-run worker: upstream planners on RotorPy aircraft, one planning round per control period.

Commands are delivered at their planning release (no computation delay): the
plant is paused while the planners run, and host solve times are only measured.
"""
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .physics import RotorPyBackend, resolve_physics_workers
from .scheduler import SPEED_GUARD_KD, SpeedGuard, plan_against_snapshot
from . import progress, upstream


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return str(value)


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data),indent=2,allow_nan=False)+"\n")


class Simulation:
    def __init__(self, backend, world, target_samples, target_dt, horizon, sample_dt):
        self.backend, self.world = backend, world
        self.target_samples = target_samples
        self.target_dt = target_dt
        self.horizon = horizon
        self.sample_dt = sample_dt
        self.t = 0.
        self.hit = None
        n = len(backend.commands)
        # Every drone holds its delivered command (self.held) until the next
        # one, through self.guard (a SpeedGuard) when one is set.
        self.held = backend.commands.copy()
        self.guard = None
        self.status = np.zeros((n,3),int)
        self.times, self.aircraft, self.targets, self.commands, self.statuses = [],[],[],[],[]
        self.sample()

    def target(self, t):
        tick = max(0.,t/self.target_dt)
        k = min(int(tick),len(self.target_samples)-1)
        j = min(k+1,len(self.target_samples)-1)
        f = min(1.,tick-k)
        return self.target_samples[k]*(1-f)+self.target_samples[j]*f

    def sample(self):
        values = (self.t,self.backend.packed().copy(),self.target(self.t),
                  self.backend.commands.copy(),self.status.copy())
        seqs = (self.times,self.aircraft,self.targets,self.commands,self.statuses)
        if self.times and abs(self.times[-1]-self.t) < 1e-12:
            for seq,v in zip(seqs,values):
                seq[-1] = v
        else:
            for seq,v in zip(seqs,values):
                seq.append(v)

    def hold(self, i, command):
        """Deliver a command that is held until the next one."""
        self.held[i] = command
        self.backend.commands[i] = command
        if self.guard is not None:
            self.apply_guard()          # the command in effect at this instant

    def apply_guard(self):
        """Pass every held command through the speed guard at the current velocity."""
        if self.guard is None:
            return
        velocities = self.backend.velocities()
        for i in range(len(self.held)):
            self.backend.commands[i,:2] = self.guard.command(self.held[i,:2],velocities[i])

    def advance(self, until):
        until = min(until,self.horizon)
        while self.t < until-1e-12 and self.hit is None:
            # Retain the global 200-Hz grid even when a command splits a step.
            grid = (math.floor((self.t+1e-10)/self.sample_dt)+1)*self.sample_dt
            end = min(grid,until)
            dt = end-self.t
            # The speed guard acts at the physics rate on the current velocity.
            self.apply_guard()
            old = self.backend.snapshot()
            p = self.backend.positions()
            self.backend.step(dt)
            hit = self.world.first_hit(p,self.backend.positions())
            if not np.isfinite(self.backend.packed()).all():
                from .collision import Hit
                hit = Hit(0.,"non_finite_state",())
            if hit is not None:
                self.backend.restore(old)
                if hit.fraction > 1e-12:
                    self.backend.step(dt*hit.fraction)
                self.t += dt*hit.fraction
                self.hit = dict(time=self.t,reason=hit.reason,drones=list(hit.drones))
            else:
                self.t = end
            self.sample()


class MissionProgress:
    """Console status for a running mission. It reads recorded state only and
    runs between planning rounds, outside every timed planner call."""

    def __init__(self, tag, horizon, interval, drones):
        self.tag, self.horizon, self.drones = tag, horizon, drones
        self.ticker = progress.Ticker(interval)
        self.tracking = np.zeros(drones,bool)
        self.solves = []

    def log(self, message):
        progress.log(self.tag,message)

    def round(self, sim, round_index, round_records, events):
        if round_records:
            self.solves.append(max(r["execution_s"] for r in round_records))
        tracking = sim.status[:,0].astype(bool)
        for i in np.flatnonzero(tracking & ~self.tracking):
            self.log(f"UAV {i+1} tracking the target at t={sim.t:.2f}s "
                     f"({int(tracking.sum())}/{self.drones} tracking)")
        self.tracking |= tracking
        if self.ticker.due():
            self.report(sim,round_index,events)

    def report(self, sim, round_index, events):
        wall = self.ticker.elapsed()
        pct = 100*sim.t/self.horizon if self.horizon else 100.
        parts = [f"t {sim.t:.2f}/{self.horizon:.2f}s ({pct:.1f}%)",f"round {round_index+1}",
                 f"wall {progress.duration(wall)}"]
        if wall > 0:
            parts.append(f"{sim.t/wall:.2f}x real time, "
                         f"ETA {progress.duration(progress.eta(sim.t,self.horizon,wall))}")
        if self.solves:
            parts.append(f"round solve mean {np.mean(self.solves):.3f}s max {max(self.solves):.3f}s")
        self.solves = []
        parts += [f"tracking {int(sim.status[:,0].astype(bool).sum())}/{self.drones}",
                  f"backup commands {sum(bool(e.get('used_backup')) for e in events)}"]
        self.log(" | ".join(parts))


def _git_sha(path):
    return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()


def adapter_provenance():
    head = subprocess.run(["git","-C",str(upstream.ROOT),"rev-parse","HEAD"],
                          capture_output=True,text=True)
    status = subprocess.run(["git","-C",str(upstream.ROOT),"status","--porcelain"],
                            capture_output=True,text=True)
    return dict(adapter_sha=head.stdout.strip() if head.returncode==0 else None,
                adapter_dirty=bool(status.stdout.strip()),python=platform.python_version(),
                platform=platform.platform(),cpu_affinity_count=len(os.sched_getaffinity(0)))


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(filename=out/"planning.log",level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    np.random.seed(args.seed)
    tag = f"run {args.controller}"
    interval = progress.parse_interval(getattr(args,"progress_interval",progress.DEFAULT_INTERVAL_S))
    progress.log(tag,f"scenario {args.scenario}, seed {args.seed}, output {out}")
    init_start = time.perf_counter()
    progress.log(tag,"loading upstream scenario and controller ...")
    cfg,world,starts = upstream.load(args.scenario,args.controller)
    issues = cfg.validate_config()
    for level,message in issues:
        if level != "INFO":
            logging.warning("upstream configuration %s: %s",level,message)
    warnings = sum(level != "INFO" for level,_ in issues)
    if warnings:
        progress.log(tag,f"upstream configuration: {warnings} warning(s), see {out/'planning.log'}")
    errors = [message for level,message in issues if level == "ERROR"]
    if errors:
        raise ValueError("Upstream configuration errors: "+"; ".join(errors))
    from target_manual import create_target
    target = create_target(verbose=False)
    target_samples = np.array(target.trajectory)
    full_duration = (len(target_samples)-1)*cfg.TIMESTEP
    horizon = min(full_duration,args.duration) if args.duration is not None else full_duration
    if horizon <= 0:
        raise ValueError("Mission duration must be positive")
    # Parallel stepping is bit-identical to serial stepping; it only uses more cores.
    physics_workers = resolve_physics_workers(getattr(args,"physics_workers","auto"),len(starts))
    progress.log(tag,f"{len(starts)} UAVs, mission horizon {horizon:.2f}s of {full_duration:.2f}s, "
                     f"planning period {cfg.TIMESTEP:g}s, physics dt {args.physics_dt:g}s "
                     f"on {physics_workers} worker(s); commands delivered at release")
    backend = RotorPyBackend(starts,args.physics_dt,physics_workers)
    # Every controller holds its latest command for the period, through the
    # speed guard (same for all controllers; --speed-guard-band off disables it).
    band = getattr(args,"speed_guard_band",None)
    guard = None if band is None else SpeedGuard(cfg.VMAX,band,SPEED_GUARD_KD)
    if guard is not None:
        progress.log(tag,f"speed guard: no speed-up above {cfg.VMAX-band:g} m/s, "
                         f"brake {guard.kd:g}*(VMAX-speed) above VMAX={cfg.VMAX:g} m/s "
                         "(the planner keeps VMAX)")
    else:
        progress.log(tag,"speed guard off")
    with progress.stage(tag,"initializing upstream planners (warm-up)"):
        robots = upstream.initialize(cfg,starts,target_samples[0])
    # Disposable warmup consumes RNG samples; measured experiments get a fresh stream.
    np.random.seed(args.seed)
    initialization = time.perf_counter()-init_start
    configuration = dict(schema_version=1,scenario=cfg.SCENARIO_NAME,controller=cfg.CONTROLLER,
                         seed=args.seed,starts=starts,scenario_definition=cfg.SCENARIO_DEF,
                         params={k:v for k,v in vars(cfg).items() if k.isupper()},
                         horizon_s=horizon,full_target_duration_s=full_duration,
                         physics_dt=args.physics_dt,physics_workers=physics_workers,
                         planning_period=cfg.TIMESTEP,
                         scheduling="parallel",planning_delay="none",command_execution="hold_latest",
                         speed_guard=None if guard is None else
                         dict(vmax=guard.vmax,band=guard.band,kd=guard.kd,
                              speedup_cut_above=guard.vmax-guard.band,applies_to="all_controllers"),
                         deployment="one_worker_per_uav",order="ascending_drone_id",
                         physics="rotorpy_hummingbird_cmd_acc",aerodynamics=False,
                         upstream_sha=_git_sha(upstream.UPSTREAM),
                         rotorpy_sha="40b09697a6a1596d404a67f459941d46a39eedd3",
                         package_versions={p:importlib.metadata.version(p)
                                           for p in ("rotorpy","numpy","scipy","casadi","pydecomp")},
                         initialization_s=initialization,
                         provenance=adapter_provenance(),
                         thread_limits={k:os.environ.get(k) for k in
                                        ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    write_json(out/"config.json",configuration)
    sim = Simulation(backend,world,target_samples,cfg.TIMESTEP,horizon,args.physics_dt)
    sim.guard = guard
    initial = world.first_hit(starts,starts)
    if initial:
        sim.hit = dict(time=0.,reason=initial.reason,drones=list(initial.drones))
    monitor = MissionProgress(tag,horizon,interval,len(starts))
    progress.log(tag,f"mission started; status every {interval:g}s of wall time"
                 if interval > 0 else "mission started")
    round_index = 0
    events,plans,plan_epochs,plan_drones = [],[],[],[]
    wall_start = time.perf_counter()
    interrupted = False
    failure = None
    try:
        while (release := round_index*cfg.TIMESTEP) < horizon-1e-12:
            sim.advance(release)
            if sim.hit:
                break
            pv,net_acc = backend.observations()
            for i,robot in enumerate(robots):
                upstream.age_robot(robot)
                robot.state = pv[i].copy()
                robot.accel = net_acc[i]+cfg.D_FRAC*pv[i,3:6]
                robot.control = backend.commands[i].copy()
                robot.time_stamp = release
                robot.goal = sim.target(release).copy()
            # The plant is paused while the planners run: every command is
            # delivered at this release, and the solve time is only measured.
            def plan(i, robot):
                robot.requested_command = None
                robot.used_backup = False
                before = time.perf_counter()
                robot.computeControlSignal(robots)
                measured = time.perf_counter()-before
                if robot.requested_command is None:
                    raise RuntimeError("Upstream failed to issue an acceleration command")
                robot.planned = True
                return measured

            def deliver(i, robot, measured):
                record = dict(round=round_index,drone=i,observed_at=release,completed_at=release,
                              execution_s=measured,requested_command=robot.requested_command.tolist(),
                              used_backup=bool(getattr(robot,"used_backup",False)),
                              solver_failures=upstream.failure_count(robot,cfg.CONTROLLER))
                sim.hold(i,robot.requested_command)
                sim.status[i] = (robot.mode == cfg.MODE_TRACK,robot.is_leader_role,
                                 getattr(robot,"mpc_fail_streak",0))
                plans.append(robot.states_prediction.copy())
                plan_epochs.append([release,release])
                plan_drones.append(i)
                events.append(record)
                # Persist events as they complete, so partial runs are inspectable.
                with (out/"events.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")

            # One planner per UAV: every planner sees the pre-round team predictions.
            measured = plan_against_snapshot(robots,plan)
            for i,(robot,duration) in enumerate(zip(robots,measured)):
                deliver(i,robot,duration)
            sim.sample()
            monitor.round(sim,round_index,events[-len(robots):],events)
            round_index += 1
        if sim.hit is None:
            sim.advance(horizon)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        logging.exception("Experiment failed")
        raise
    finally:
        wall = time.perf_counter()-wall_start
        if interrupted:
            progress.log(tag,f"interrupted at t={sim.t:.2f}s; saving the partial run")
        elif failure:
            progress.log(tag,f"failed at t={sim.t:.2f}s: {failure}")
        if sim.hit:
            drones = ", ".join(f"UAV {d+1}" for d in sim.hit["drones"]) or "no UAV"
            progress.log(tag,f"collision at t={sim.hit['time']:.3f}s: {sim.hit['reason']} ({drones})")
        progress.log(tag,f"mission loop ended at t={sim.t:.2f}s after {progress.duration(wall)}; "
                         f"writing trajectory ({len(sim.times)} samples) and plans ...")
        backend.close()
        np.savez_compressed(out/"trajectory.npz",time=np.array(sim.times),
                            aircraft=np.array(sim.aircraft),target=np.array(sim.targets),
                            command=np.array(sim.commands),status=np.array(sim.statuses))
        np.savez_compressed(out/"plans.npz",prediction=np.array(plans),
                            epochs=np.array(plan_epochs),drone=np.array(plan_drones))
        (out/"events.jsonl").touch(exist_ok=True)
        result = dict(status="error" if failure else "interrupted" if interrupted else "collision" if sim.hit else "completed",
                      collision=sim.hit,mission_time_s=sim.t,wall_time_s=wall,
                      initialization_s=initialization,
                      planning_rounds=len({event["round"] for event in events}),error=failure,
                      throughput=sim.t/wall if wall else 0.)
        write_json(out/"result.json",result)
        from .metrics import report, summary
        metrics = report(out,tag=tag,interval=interval)
    progress.log(tag,summary(metrics))
    print(f"{cfg.CONTROLLER}: {result['status']}, {sim.t:.3f}s mission, "
          f"{wall:.2f}s wall; {out}",flush=True)
    return 130 if interrupted else 0
