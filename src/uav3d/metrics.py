"""Upstream metric definitions integrated over explicit mission timestamps."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import progress
from .upstream import UPSTREAM


def braking_summary(time, braking):
    """Watchdog braking: per-drone time share and episode count (adapter safety layer)."""
    time,braking = np.asarray(time,float),np.asarray(braking,bool)
    weights = np.diff(time)
    span = time[-1]-time[0]
    shares = [float(weights@braking[:-1,i])/span*100 if span > 0 else 0. for i in range(braking.shape[1])]
    episodes = [int(np.sum(braking[1:,i] & ~braking[:-1,i])+braking[0,i]) for i in range(braking.shape[1])]
    return dict(watchdog_braking_pct_max=max(shares,default=0.),
                watchdog_braking_pct_mean=float(np.mean(shares)) if shares else 0.,
                watchdog_braking_episodes=int(sum(episodes)))


def target_views(aircraft, target, fov_half, on_progress=None):
    """Per-interval target visibility and coverage efficiency (upstream definitions).

    Entry k describes the interval [time[k], time[k+1]); the last sample has none.
    on_progress(done, total), if given, is called after each interval."""
    sys.path.insert(0,str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    from metrics import fov_coverage
    pos = np.asarray(aircraft)[:-1,:,:2]
    target = np.asarray(target)[:-1]
    n = pos.shape[1]
    visible = np.any(np.max(np.abs(pos-target[:,None,:2]),axis=2) <= fov_half,axis=1)
    coverage = []
    for k,(p,t) in enumerate(zip(pos,target)):
        if on_progress is not None:
            on_progress(k,len(pos))
        # Upstream run_metrics defines the metric leader as the nearest UAV,
        # independent of the sticky controller leader. Preserve that definition.
        leader = int(np.argmin(np.linalg.norm(p-t[:2],axis=1)))
        area,_ = fov_coverage(p,2*fov_half,leader_idx=leader)
        coverage.append(area/(n*(2*fov_half)**2))
    return visible,np.array(coverage,float)


def calculate(time, aircraft, target, status, horizon, fov_half, events=(), result=None, views=None):
    time = np.asarray(time,float)
    status = np.asarray(status)
    n = np.asarray(aircraft).shape[1]
    weights = np.diff(time)
    visible,coverage = views if views is not None else target_views(aircraft,target,fov_half)
    # Terminal contact sample does not count as joining the formation.
    alive = len(time)-1 if result and result.get("collision") else len(time)
    joins,search_time,window,episodes = [],0.,0.,[]
    follower_times = []
    for i in range(n):
        track = status[:,i,0].astype(bool)
        found = np.flatnonzero(track[:alive])
        if not len(found):
            continue
        k = int(found[0])
        joins.append(float(time[k]))
        # Upstream's leader is in TRACK from its election (t = 0 there). Here the
        # first status arrives with the first delivered command, so the leader is
        # recognised by its recorded role instead of by t = 0.
        if not (status[k,i,1] or time[k] <= 0):
            follower_times.append(float(time[k]))
        window += time[-1]-time[k]
        episode = None
        for j in range(k,len(time)-1):
            if not track[j]:
                search_time += weights[j]
                if episode is None:
                    episode = time[j]
            elif episode is not None:
                episodes.append(float(time[j]-episode))
                episode = None
        if episode is not None:
            episodes.append(float(time[-1]-episode))
    def stats(key, scale=1.):
        vals = [e[key]*scale for e in events]
        return {f"{key}_{name}":float(fn(vals)) if vals else None
                for name,fn in (("mean",np.mean),("p95",lambda x:np.percentile(x,95)),("max",np.max))}
    all_joined = len(joins)==n
    m = dict(gathered_uavs=len(joins),drone_count=n,
             gather_time_team_s=max(joins) if all_joined else None,
             gather_time_mean_s=float(np.mean(follower_times)) if all_joined and follower_times else 0. if all_joined else None,
             reform_search_pct=100*search_time/window if window and len(joins) and window/len(joins)>=.2*horizon else None,
             reform_episode_s=float(np.mean(episodes)) if episodes else None,
             reform_episodes_per_joined_uav=len(episodes)/len(joins) if joins else None,
             reform_window_mean_s=window/len(joins) if joins else None,
             mission_visibility=100*float(weights@visible)/horizon,
             mission_cov_eff=float(weights@np.array(coverage))/horizon,
             target_loss_pct=100-100*float(weights@visible)/horizon,
             completion_pct=100*time[-1]/horizon,
             success=bool(result and result.get("status")=="completed"),
             mission_time_s=float(time[-1]),horizon_s=horizon,
             computation_calls=len(events),
             rejected_stale_plans=sum(e.get("accepted") is False for e in events),
             upstream_stop_fallbacks=sum(bool(e.get("upstream_stop")) and e.get("delivered",True) for e in events))
    for key in ("execution_s","queue_wait_s","latency_s"):
        m.update(stats(key))
    if result:
        m.update(wall_time_s=result.get("wall_time_s"),throughput=result.get("throughput"),
                 skipped_releases=result.get("skipped_releases"),
                 termination=result.get("status"),collision=result.get("collision"))
    return m


def _value(value, fmt):
    return "n/a" if value is None else format(value,fmt)


def summary(m):
    """One-line human-readable digest of a metrics dict."""
    gathered = f"{m['gathered_uavs']}/{m['drone_count']} UAVs gathered"
    if m.get("gather_time_team_s") is not None:
        gathered += f" by t={m['gather_time_team_s']:.2f}s"
    parts = [f"{m.get('termination') or 'unknown'}",
             f"{m['completion_pct']:.1f}% of {m['horizon_s']:.2f}s horizon",
             gathered,
             f"visibility {m['mission_visibility']:.1f}%",
             f"coverage eff {m['mission_cov_eff']:.3f}",
             f"latency p95 {_value(m.get('latency_s_p95'),'.3f')}s",
             f"{m['computation_calls']} planner calls"]
    if m.get("collision"):
        parts.insert(1,f"collision at t={m['collision']['time']:.3f}s ({m['collision']['reason']})")
    return "metrics: "+" | ".join(parts)


def report(directory, tag="metrics", interval=None):
    """Recompute and write metrics.{json,csv,md}; log stages under tag on stderr."""
    directory = Path(directory)
    cfg = json.loads((directory/"config.json").read_text())
    result = json.loads((directory/"result.json").read_text())
    ticker = progress.Ticker(progress.parse_interval(progress.default_interval())
                             if interval is None else interval)
    def on_progress(done, total):
        if ticker.due():
            progress.log(tag,f"target visibility/coverage {done}/{total} samples "
                             f"({100*done/total:.0f}%), ETA "
                             f"{progress.duration(progress.eta(done,total,ticker.elapsed()))}")
    with np.load(directory/"trajectory.npz") as data:
        events = [json.loads(l) for l in (directory/"events.jsonl").read_text().splitlines()]
        progress.log(tag,f"computing metrics for {directory}: {len(data['time'])} samples, "
                         f"{len(events)} planner calls")
        start = time.perf_counter()
        views = target_views(data["aircraft"],data["target"],cfg["params"]["VIEWING_RADIUS"],
                             on_progress)
        m = calculate(data["time"],data["aircraft"],data["target"],data["status"],
                      cfg["horizon_s"],cfg["params"]["VIEWING_RADIUS"],events,result,views)
        if "watchdog_braking" in data:
            m.update(braking_summary(data["time"],data["watchdog_braking"]))
    m.update(compute_ratio=cfg.get("compute_ratio",1.))
    m.update(controller=cfg["controller"],scenario=cfg["scenario"],seed=cfg["seed"])
    from .experiment import write_json
    write_json(directory/"metrics.json",m)
    with (directory/"metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(m))
        writer.writeheader()
        writer.writerow(m)
    lines = ["| Metric | Value |","|---|---|"]
    lines += [f"| {key} | {value if value is not None else 'unavailable'} |" for key,value in m.items()]
    (directory/"metrics.md").write_text("\n".join(lines)+"\n")
    progress.log(tag,f"wrote metrics.json, metrics.csv, metrics.md in "
                     f"{progress.duration(time.perf_counter()-start)}")
    return m
