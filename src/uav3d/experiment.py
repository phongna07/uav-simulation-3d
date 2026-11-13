"""Single-run worker: host computation durations become mission-time events."""
from dataclasses import asdict, replace
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
from .scheduler import next_release, parallel_events, plan_against_snapshot, scaled_delay
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
        # Per-drone PlanSchedule; None holds backend.commands as delivered.
        self.schedules = [None]*n
        self.status = np.zeros((n,3),int)
        self.times, self.aircraft, self.targets, self.commands, self.statuses = [],[],[],[],[]
        self.braking = []
        self.sample()

    def target(self, t):
        tick = max(0.,t/self.target_dt)
        k = min(int(tick),len(self.target_samples)-1)
        j = min(k+1,len(self.target_samples)-1)
        f = min(1.,tick-k)
        return self.target_samples[k]*(1-f)+self.target_samples[j]*f

    def sample(self):
        # Watchdog flag: the drone is past its plan's stop point, i.e. braking to rest.
        braking = np.array([s is not None and self.t > s.stop_at+1e-9 for s in self.schedules])
        values = (self.t,self.backend.packed().copy(),self.target(self.t),
                  self.backend.commands.copy(),self.status.copy(),braking)
        seqs = (self.times,self.aircraft,self.targets,self.commands,self.statuses,self.braking)
        if self.times and abs(self.times[-1]-self.t) < 1e-12:
            for seq,v in zip(seqs,values):
                seq[-1] = v
        else:
            for seq,v in zip(seqs,values):
                seq.append(v)

    def apply_schedules(self, t):
        if not any(s is not None for s in self.schedules):
            return
        positions,velocities = self.backend.positions(),self.backend.velocities()
        for i,schedule in enumerate(self.schedules):
            if schedule is not None:
                self.backend.commands[i,:2] = schedule.command(t,positions[i],velocities[i])

    def advance(self, until):
        until = min(until,self.horizon)
        while self.t < until-1e-12 and self.hit is None:
            # Retain the global 200-Hz grid even when a command splits a step.
            grid = (math.floor((self.t+1e-10)/self.sample_dt)+1)*self.sample_dt
            end = min(grid,until)
            dt = end-self.t
            # The command in effect for this step: evaluated at its midpoint, so
            # plan step boundaries on the physics grid are not subject to rounding.
            self.apply_schedules(self.t+dt/2)
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

    def round(self, sim, round_index, round_records, events, skipped):
        if round_records:
            self.solves.append(max(r["execution_s"] for r in round_records))
        tracking = sim.status[:,0].astype(bool)
        for i in np.flatnonzero(tracking & ~self.tracking):
            self.log(f"UAV {i+1} tracking the target at t={sim.t:.2f}s "
                     f"({int(tracking.sum())}/{self.drones} tracking)")
        self.tracking |= tracking
        if self.ticker.due():
            self.report(sim,round_index,events,skipped)

    def report(self, sim, round_index, events, skipped):
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
        braking = int(sim.braking[-1].sum()) if sim.braking else 0
        parts += [f"tracking {int(sim.status[:,0].astype(bool).sum())}/{self.drones}",
                  f"braking {braking}",
                  f"stale plans {sum(e.get('accepted') is False for e in events)}",
                  f"stop fallbacks {sum(bool(e.get('upstream_stop')) and e.get('delivered',True) for e in events)}",
                  f"skipped releases {skipped}"]
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
                     f"on {physics_workers} worker(s), compute ratio {args.compute_ratio:g}")
    backend = RotorPyBackend(starts,args.physics_dt,physics_workers)
    # "ours" executes its whole plan by mission time; other methods hold the latest command.
    follow_plan = cfg.CONTROLLER == "ours"
    execution = "tracked_plan_with_watchdog" if follow_plan else "hold_latest"
    replay_config = None
    timings = None
    if args.timings_from:
        source = Path(args.timings_from)
        replay_config = json.loads((source/"config.json").read_text())
        expected = (cfg.SCENARIO_NAME,cfg.CONTROLLER,args.seed,starts.tolist(),horizon)
        previous = (replay_config["scenario"],replay_config["controller"],replay_config["seed"],
                    replay_config["starts"],replay_config["horizon_s"])
        if expected != previous:
            raise ValueError("Timing replay requires matching scenario, controller, seed, starts and horizon")
        if replay_config.get("scheduling","sequential") != "parallel":
            raise ValueError("Timing replay requires a source run with parallel scheduling")
        if replay_config.get("command_execution","hold_latest") != execution:
            raise ValueError("Timing replay requires matching command execution")
        if replay_config["upstream_sha"] != _git_sha(upstream.UPSTREAM):
            raise ValueError("Timing replay requires matching upstream revision")
        for key in ("rects","circles","waypoints","speeds","params"):
            if jsonable(cfg.SCENARIO_DEF[key]) != replay_config["scenario_definition"][key]:
                raise ValueError(f"Timing replay scenario differs in {key}")
        timings = [json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
        # Recorded delays were scaled by the source run's ratio; recover the raw duration.
        # A ratio-0 source injected no delay, so its measured solve time is used instead.
        source_ratio = float(replay_config.get("compute_ratio",1.))
    ratio = args.compute_ratio
    if timings is not None:
        progress.log(tag,f"replaying {len(timings)} recorded planning delays from {source}")
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
                         scheduling="parallel",command_execution=execution,
                         compute_ratio=ratio,
                         plan_tracking=dict(kp=upstream.TRACK_KP,kd=upstream.TRACK_KD,
                                            stop_accel_ratio=upstream.STOP_ACCEL_RATIO,
                                            track_accel_ratio=upstream.TRACK_ACCEL_RATIO,
                                            vmax_guard_band=upstream.VMAX_GUARD_BAND,
                                            accept_position_radius_ratio=upstream.ACCEPT_POSITION_RADIUS_RATIO,
                                            accept_speed_error=upstream.ACCEPT_SPEED_ERROR)
                         if follow_plan else None,
                         deployment="one_worker_per_uav",order="ascending_drone_id",
                         physics="rotorpy_hummingbird_cmd_acc",aerodynamics=False,
                         upstream_sha=_git_sha(upstream.UPSTREAM),
                         rotorpy_sha="40b09697a6a1596d404a67f459941d46a39eedd3",
                         package_versions={p:importlib.metadata.version(p)
                                           for p in ("rotorpy","numpy","scipy","casadi","pydecomp")},
                         initialization_s=initialization,timings_from=args.timings_from,
                         provenance=adapter_provenance(),
                         thread_limits={k:os.environ.get(k) for k in
                                        ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    write_json(out/"config.json",configuration)
    sim = Simulation(backend,world,target_samples,cfg.TIMESTEP,horizon,args.physics_dt)
    initial = world.first_hit(starts,starts)
    if initial:
        sim.hit = dict(time=0.,reason=initial.reason,drones=list(initial.drones))
    monitor = MissionProgress(tag,horizon,interval,len(starts))
    progress.log(tag,f"mission started; status every {interval:g}s of wall time"
                 if interval > 0 else "mission started")
    release,round_index,skipped = 0.,0,0
    events,plans,plan_epochs,plan_drones = [],[],[],[]
    wall_start = time.perf_counter()
    interrupted = False
    failure = None
    try:
        while release < horizon-1e-12 and sim.hit is None:
            sim.advance(release)
            if sim.hit:
                break
            pv,net_acc = backend.observations()
            for i,robot in enumerate(robots):
                upstream.age_robot(robot,release,cfg.TIMESTEP)
                robot.state = pv[i].copy()
                robot.accel = net_acc[i]+cfg.D_FRAC*pv[i,3:6]
                robot.control = backend.commands[i].copy()
                robot.time_stamp = release
                robot.goal = sim.target(release).copy()
                if hasattr(robot,"last_observation"):
                    upstream.set_tracker_interval(robot.target_tracker,release-robot.last_observation)
                robot.last_observation = release
            # Aircraft observations stay frozen for this round. The physical plant
            # advances separately; no later UAV accidentally sees a future pose.
            def plan(i, robot):
                robot.requested_command = None
                robot.used_backup = False
                backup = getattr(robot,"_backup",None)
                before = time.perf_counter()
                robot.computeControlSignal(robots)
                measured = time.perf_counter()-before
                if robot.requested_command is None:
                    raise RuntimeError("Upstream failed to issue an acceleration command")
                robot.prediction_epoch = release
                # Built outside the timed call; the shifted buffers change next round.
                robot.schedule,robot.stop_current = (upstream.plan_schedule(robot,cfg,release,backup)
                                                     if follow_plan else (None,False))
                if timings is not None:
                    index = len(events)+i
                    if index >= len(timings):
                        raise ValueError("Timing replay exhausted before mission completion")
                    recorded = timings[index]
                    if recorded["round"] != round_index or recorded["drone"] != i:
                        raise ValueError("Timing replay planning sequence does not match")
                    raw = (float(recorded["injected_delay_s"])/source_ratio if source_ratio > 0
                           else float(recorded["execution_s"]))
                    delay = scaled_delay(raw,ratio)
                else:
                    delay = scaled_delay(measured,ratio)
                if not np.isfinite(delay) or delay < 0:
                    raise ValueError("Invalid recorded computation duration")
                return measured,delay

            def deliver(event, robot):
                record = asdict(event)|dict(queue_wait_s=event.queue_wait_s,latency_s=event.latency_s,
                                           requested_command=robot.requested_command.tolist(),delivered=False,
                                           upstream_stop=bool(robot.stop_current),
                                           used_backup=bool(getattr(robot,"used_backup",False)),
                                           solver_failures=upstream.failure_count(robot,cfg.CONTROLLER))
                if sim.hit is None and event.completed_at <= horizon+1e-12:
                    backend.commands[event.drone] = robot.requested_command
                    current = sim.schedules[event.drone]
                    if robot.stop_current and current is not None:
                        # Upstream fell back to stopping. The aircraft flew the current
                        # plan, not the backup branch, so it stops along that plan now,
                        # and publishes that stop as its prediction.
                        sim.schedules[event.drone] = current.stopped(sim.t)
                        upstream.publish_prediction(robot,sim.schedules[event.drone],release)
                    elif robot.schedule is not None and current is not None and not upstream.plan_valid(
                            robot.schedule,sim.t,backend.positions()[event.drone],
                            backend.velocities()[event.drone],cfg):
                        # Stale: solved for a state the aircraft has left. Keep flying
                        # (or stopping along) the current plan and publish that instead.
                        record["accepted"] = False
                        upstream.publish_prediction(robot,current,release)
                    else:
                        sim.schedules[event.drone] = robot.schedule
                    if sim.schedules[event.drone] is not None:
                        # Delivered late: continue at the plan step for now, not U[0].
                        sim.apply_schedules(sim.t)
                    sim.status[event.drone] = (robot.mode == cfg.MODE_TRACK,robot.is_leader_role,
                                               getattr(robot,"mpc_fail_streak",0))
                    record["delivered"] = True
                    plans.append(robot.states_prediction.copy())
                    plan_epochs.append([release,event.completed_at])
                    plan_drones.append(event.drone)
                events.append(record)
                # Persist events as they complete, so partial runs are inspectable.
                with (out/"events.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")

            # One worker per UAV: every planner sees the pre-round team predictions,
            # and all commands arrive together after the slowest solve, d = max(C_i).
            timed = plan_against_snapshot(robots,lambda i,robot: plan(i,robot))
            round_events = list(parallel_events([d for _,d in timed],release,round_index))
            sim.advance(round_events[0].completed_at)
            for event,(measured,_),robot in zip(round_events,timed,robots):
                deliver(replace(event,execution_s=measured),robot)
            sim.sample()
            monitor.round(sim,round_index,events[-len(round_events):],events,skipped)
            if sim.hit is not None or sim.t >= horizon-1e-12:
                break
            release,missed = next_release(sim.t,release,cfg.TIMESTEP)
            skipped += missed
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
                            command=np.array(sim.commands),status=np.array(sim.statuses),
                            watchdog_braking=np.array(sim.braking))
        np.savez_compressed(out/"plans.npz",prediction=np.array(plans),
                            epochs=np.array(plan_epochs),drone=np.array(plan_drones))
        (out/"events.jsonl").touch(exist_ok=True)
        actual_rounds = len({event["round"] for event in events})
        nominal_releases = max(0,math.ceil(sim.t/cfg.TIMESTEP-1e-10))
        result = dict(status="error" if failure else "interrupted" if interrupted else "collision" if sim.hit else "completed",
                      collision=sim.hit,mission_time_s=sim.t,wall_time_s=wall,
                      initialization_s=initialization,skipped_releases=skipped,
                      planning_rounds=actual_rounds,error=failure,
                      throughput=sim.t/wall if wall else 0.)
        result["skipped_releases"] = max(0,nominal_releases-actual_rounds)
        write_json(out/"result.json",result)
        from .metrics import report, summary
        metrics = report(out,tag=tag,interval=interval)
    progress.log(tag,summary(metrics))
    print(f"{cfg.CONTROLLER}: {result['status']}, {sim.t:.3f}s mission, "
          f"{wall:.2f}s wall; {out}",flush=True)
    return 130 if interrupted else 0
