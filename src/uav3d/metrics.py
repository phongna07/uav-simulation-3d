"""Upstream metric definitions integrated over explicit mission timestamps."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
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


def calculate(time, aircraft, target, status, horizon, fov_half, events=(), result=None):
    sys.path.insert(0,str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    from metrics import fov_coverage
    time = np.asarray(time,float)
    pos = np.asarray(aircraft)[:,:,:2]
    status = np.asarray(status)
    n = pos.shape[1]
    weights = np.diff(time)
    visible = np.any(np.max(np.abs(pos[:-1]-target[:-1,None,:2]),axis=2) <= fov_half,axis=1)
    coverage = []
    for p,t in zip(pos[:-1],target[:-1]):
        # Upstream run_metrics defines the metric leader as the nearest UAV,
        # independent of the sticky controller leader. Preserve that definition.
        leader = int(np.argmin(np.linalg.norm(p-t[:2],axis=1)))
        area,_ = fov_coverage(p,2*fov_half,leader_idx=leader)
        coverage.append(area/(n*(2*fov_half)**2))
    # Terminal contact sample does not count as joining the formation.
    alive = len(time)-1 if result and result.get("collision") else len(time)
    joins,search_time,window,episodes = [],0.,0.,[]
    for i in range(n):
        track = status[:,i,0].astype(bool)
        found = np.flatnonzero(track[:alive])
        if not len(found):
            continue
        k = int(found[0])
        joins.append(float(time[k]))
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
    follower_times = [t for t in joins if t>0]
    m = dict(gathered_uavs=len(joins),drone_count=n,
             gather_time_team_s=max(joins) if all_joined else None,
             gather_time_mean_s=float(np.mean(follower_times)) if all_joined and follower_times else 0. if all_joined else None,
             reform_search_pct=100*search_time/window if window and len(joins) and window/len(joins)>=.2*horizon else None,
             reform_episode_s=float(np.mean(episodes)) if episodes else None,
             reform_episodes_per_joined_uav=len(episodes)/len(joins) if joins else None,
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


def report(directory):
    directory = Path(directory)
    cfg = json.loads((directory/"config.json").read_text())
    result = json.loads((directory/"result.json").read_text())
    with np.load(directory/"trajectory.npz") as data:
        events = [json.loads(l) for l in (directory/"events.jsonl").read_text().splitlines()]
        m = calculate(data["time"],data["aircraft"],data["target"],data["status"],
                      cfg["horizon_s"],cfg["params"]["VIEWING_RADIUS"],events,result)
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
    return m
