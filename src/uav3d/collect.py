"""Per-preset comparison tables from recorded runs; never runs a planner.

``uav3d collect`` reads a run directory under runs/ (config.json, result.json,
events.jsonl, trajectory.npz) and writes metrics/<preset>.{xlsx,csv,md}, where
<preset> is the scenario plus the mission length (``scen1_150s`` or
``scen1_full_mission``). Each table has one entry per algorithm: collecting a
run replaces only that algorithm's row, so every file holds the latest run of
each algorithm. The CSV is the store read back on the next collection.
"""
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import sys
import numpy as np
from . import CONTROLLERS
from .metrics import braking_summary, calculate, target_views
from .upstream import UPSTREAM

PAPER, RUN, SAFETY, FORMATION, TRACKING, COMPUTATION, FALLBACKS = (
    "Paper metrics", "Run", "Safety and completion", "Formation", "Tracking",
    "Computation", "Fallbacks and safety layers")
SECTIONS = (PAPER, RUN, SAFETY, FORMATION, TRACKING, COMPUTATION, FALLBACKS)
REFORM_MIN_WINDOW = .2      # upstream thesis_metrics: mean window >= 20 % of the mission
WAYPOINT_TOLERANCE_M = 1.


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    section: str
    fmt: str | None          # None: text
    definition: str


M = Metric
METRICS = [
    M("gather_time_s","Gathering time [s]",PAPER,"{:.1f}",
      "Mean over the followers of t_i^g, the first time UAV i is in TRACK (the leader joins at "
      "its election); blank unless every UAV joined before t_c"),
    M("reform_pct","Re-formation time rho_ref [%]",PAPER,"{:.1f}",
      "Eq. (33): UAV time in SEARCH within [t_i^g, t_c) over sum_i (t_c - t_i^g); blank if the "
      "mean window is < 20 % of the mission"),
    M("mission_visibility_pct","Mission visibility [%]",PAPER,"{:.2f}",
      "100 - T_loss: share of the mission [0, T) with the target inside >= 1 square FOV"),
    M("target_loss_pct","Target loss T_loss [%]",PAPER,"{:.2f}",
      "Eq. (34): share of [0, T) with no UAV within L of the target in the infinity norm; "
      "time from t_c on counts as lost"),
    M("coverage","Mission coverage C",PAPER,"{:.3f}",
      "Eq. (35): FOV union area connected to the leader's (nearest UAV's) FOV / (N (2L)^2), "
      "averaged over [0, T) with A_k = 0 from t_c on"),
    M("exec_mean_ms","Execution time per UAV and step, mean [ms]",PAPER,"{:.1f}",
      "Host wall time of one upstream computeControlSignal call (planning + optimisation) "
      "per UAV and control step before t_c"),
    M("exec_p95_ms","Execution time per UAV and step, p95 [ms]",PAPER,"{:.1f}",
      "As above, 95th percentile"),
    M("paper_notes","Paper metric notes",PAPER,None,
      "Why a paper metric is unavailable, and the collision cut if one applies"),

    M("run_id","Run ID",RUN,None,"Run directory"),
    M("finished_utc","Run finished (UTC)",RUN,None,"Modification time of result.json"),
    M("uavs","UAVs",RUN,"{:d}","Number of UAVs"),
    M("seed","Seed",RUN,"{:d}","run --seed"),
    M("scheduling","Scheduling",RUN,None,"Recorded scheduling (parallel; older runs may be sequential)"),
    M("compute_ratio","Compute ratio r",RUN,"{:g}",
      "run --compute-ratio: mission-time delay = r x measured solve time (0: instantaneous)"),
    M("command_execution","Command execution",RUN,None,
      "tracked_plan_with_watchdog (ours) or hold_latest"),
    M("horizon_s","Mission length T [s]",RUN,"{:.1f}","Evaluated mission length"),
    M("mission_time_s","Flown duration [s]",RUN,"{:.1f}","Simulation time reached (t_c on a collision)"),
    M("termination","Termination",RUN,None,"completed, collision, interrupted or error"),
    M("waypoints_reached","Target waypoints reached",RUN,None,
      f"Route waypoints the target came within {WAYPOINT_TOLERANCE_M:g} m of, in order, counting its start"),
    M("wall_time_s","Wall time [s]",RUN,"{:.1f}","Host wall time of the mission loop"),
    M("real_time_factor","Real-time factor",RUN,"{:.2f}","Mission seconds per wall second"),

    M("collision_free","Collision-free",SAFETY,None,
      "No obstacle, world boundary, ground or UAV-UAV contact (continuous test)"),
    M("completion_pct","Completion [%]",SAFETY,"{:.1f}","t_c / T (100 without a collision)"),
    M("first_collision_s","First collision t_c [s]",SAFETY,"{:.2f}","Mission time of the first contact"),
    M("collision_reason","Collision kind",SAFETY,None,"Contact kind and UAVs involved"),
    M("min_obstacle_clearance_m","Min obstacle clearance beyond UAV radius [m]",SAFETY,"{:.2f}",
      "Smallest planar obstacle distance minus ROBOT_RADIUS (negative: contact)"),
    M("min_inter_uav_m","Min inter-UAV distance [m]",SAFETY,"{:.2f}","Smallest 3D distance between two UAVs"),
    M("min_altitude_m","Min altitude [m]",SAFETY,"{:.2f}","Lowest UAV altitude"),

    M("track_pct","Time in TRACK mode [%]",FORMATION,"{:.1f}","UAV time in TRACK over [0, t_c)"),
    M("gathered","UAVs that joined the formation",FORMATION,None,"UAVs in TRACK at least once before t_c"),
    M("gather_team_s","Gathering time, whole team [s]",FORMATION,"{:.1f}",
      "Time until the last UAV first enters TRACK; blank if one never joins"),
    M("reform_episodes","Re-formation episodes per UAV",FORMATION,"{:.1f}",
      "SEARCH episodes after joining, per joined UAV"),
    M("reform_episode_s","Re-formation episode mean length [s]",FORMATION,"{:.1f}",
      "Mean length of those SEARCH episodes"),
    M("leader_present_pct","Leader present after formation [%]",FORMATION,"{:.1f}",
      "Time with an elected leader after the whole team first was in TRACK"),
    M("leader_losses","Leader losses after formation",FORMATION,"{:d}",
      "Times the team lost its leader after the whole team first was in TRACK"),

    M("visibility_after_gather_pct","Mission visibility after gathering [%]",TRACKING,"{:.2f}",
      "Target visibility over [t_team, T), time after t_c counted as lost"),
    M("coverage_after_gather","Mission coverage after gathering",TRACKING,"{:.3f}",
      "Coverage C over [t_team, T), zero after t_c"),
    M("visibility_flown_pct","Visibility before t_c [%]",TRACKING,"{:.2f}",
      "Target visibility over the flown interval [0, t_c) only"),
    M("coverage_flown","Coverage before t_c",TRACKING,"{:.3f}","Coverage C over [0, t_c) only"),
    M("nearest_median_m","Nearest UAV to target, median [m]",TRACKING,"{:.2f}",
      "Planar distance of the nearest UAV to the target after gathering"),
    M("nearest_p95_m","Nearest UAV to target, p95 [m]",TRACKING,"{:.2f}","As above, 95th percentile"),
    M("nearest_max_m","Nearest UAV to target, max [m]",TRACKING,"{:.2f}","As above, maximum"),
    M("fleet_dist_median_m","Mean fleet distance to target, median [m]",TRACKING,"{:.2f}",
      "Mean planar UAV-to-target distance after gathering"),
    M("fleet_dist_p95_m","Mean fleet distance to target, p95 [m]",TRACKING,"{:.2f}","As above, 95th percentile"),

    M("solve_mean_ms","Solve time per UAV, mean [ms]",COMPUTATION,"{:.1f}",
      "Host wall time per UAV and step, all recorded solves"),
    M("solve_p95_ms","Solve time per UAV, p95 [ms]",COMPUTATION,"{:.1f}","As above, 95th percentile"),
    M("solve_max_ms","Solve time per UAV, max [ms]",COMPUTATION,"{:.1f}","As above, maximum"),
    M("round_median_ms","Fleet planning round, median [ms]",COMPUTATION,"{:.1f}",
      "Host compute per control step for the fleet: slowest UAV (parallel) or sum (sequential)"),
    M("round_p95_ms","Fleet planning round, p95 [ms]",COMPUTATION,"{:.1f}","As above, 95th percentile"),
    M("round_max_ms","Fleet planning round, max [ms]",COMPUTATION,"{:.1f}","As above, maximum"),
    M("slow_rounds_pct","Rounds slower than the control period [%]",COMPUTATION,"{:.1f}",
      "Fleet planning rounds above TIMESTEP (host compute)"),
    M("latency_mean_ms","Command latency, mean [ms]",COMPUTATION,"{:.1f}",
      "Mission time from observation to command delivery (r x solve time, plus queueing)"),
    M("latency_p95_ms","Command latency, p95 [ms]",COMPUTATION,"{:.1f}","As above, 95th percentile"),
    M("late_rounds_pct","Late command delivery [%]",COMPUTATION,"{:.2f}",
      "Rounds whose last command arrived more than one control period after release"),
    M("skipped_releases","Skipped planning releases",COMPUTATION,"{:d}",
      "Control periods without a planning round because the previous one was still running"),

    M("solver_failures","Solver failures",FALLBACKS,"{:d}",
      "Failed or infeasible solves counted by upstream (mpc / pure MPC / IMPC n_fail, "
      "CBF-QP n_infeasible); blank for APF or runs recorded without the counter"),
    M("backup_used","Contingency backup commands",FALLBACKS,"{:d}",
      "Commands taken from upstream's backup branch (ours)"),
    M("upstream_stops","Upstream stop fallbacks",FALLBACKS,"{:d}",
      "Delivered results in which upstream fell back to stopping along the current plan"),
    M("stale_plans","Stale plans rejected",FALLBACKS,"{:d}",
      "Adapter: delivered plans not accepted because the aircraft had left their start state"),
    M("undelivered","Results not delivered",FALLBACKS,"{:d}",
      "Solves finished after the first collision or the mission end"),
    M("watchdog_pct_max","Watchdog braking, max UAV [%]",FALLBACKS,"{:.1f}",
      "Adapter: time a UAV braked past the end of its plan (ours), worst UAV"),
    M("watchdog_episodes","Watchdog braking episodes",FALLBACKS,"{:d}","Adapter: braking episodes, all UAVs"),
]
del M
BY_KEY = {m.key:m for m in METRICS}
BY_LABEL = {m.label:m for m in METRICS}
ALGORITHM = "Algorithm"


def preset_name(cfg):
    """scenario + _<T>s, or _full_mission for the whole route."""
    name = str(cfg["scenario"])
    horizon,full = float(cfg["horizon_s"]),float(cfg.get("full_target_duration_s",cfg["horizon_s"]))
    return name+("_full_mission" if horizon >= full-1e-9 else f"_{horizon:g}s")


def _stats(values, scale=1.):
    values = np.asarray(values,float)*scale
    if not values.size:
        return None,None,None
    return float(values.mean()),float(np.percentile(values,95)),float(values.max())


def _waypoints_reached(waypoints, target):
    reached = 0
    waypoints = np.asarray(waypoints,float)
    for w in waypoints.reshape(len(waypoints),-1)[:,:2]:
        if not len(target) or np.min(np.linalg.norm(target[:,:2]-w,axis=1)) > WAYPOINT_TOLERANCE_M:
            break
        reached += 1
    return f"{reached}/{len(waypoints)}"


def _obstacle_clearance(points, scenario):
    sys.path.insert(0,str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    from evaluate import obstacle_clearance
    rects,circles = scenario.get("rects") or [],scenario.get("circles") or []
    if not rects and not len(circles):
        return None
    return float(np.min(obstacle_clearance(points,rects,circles)))


def _min_pair_distance(aircraft):
    n = aircraft.shape[1]
    if n < 2:
        return None
    return float(min(np.min(np.linalg.norm(aircraft[:,i]-aircraft[:,j],axis=1))
                     for i in range(n) for j in range(i)))


def _rounds(events, scheduling):
    rounds = {}
    for e in events:
        rounds.setdefault(e["round"],[]).append(e)
    host = [(max if scheduling == "parallel" else sum)(e["execution_s"] for e in r) for r in rounds.values()]
    late = [max(e["completed_at"] for e in r)-r[0]["observed_at"] for r in rounds.values()]
    return np.array(host,float),np.array(late,float)


def evaluate(directory):
    """All table metrics of one run directory; None marks a metric the run cannot define."""
    directory = Path(directory)
    cfg = json.loads((directory/"config.json").read_text())
    result = json.loads((directory/"result.json").read_text())
    events = [json.loads(l) for l in (directory/"events.jsonl").read_text().splitlines() if l.strip()]
    with np.load(directory/"trajectory.npz") as data:
        time,aircraft,target,status = (np.asarray(data[k]) for k in ("time","aircraft","target","status"))
        braking = np.asarray(data["watchdog_braking"]) if "watchdog_braking" in data else None
    prm = cfg["params"]
    horizon,fov_half,period = float(cfg["horizon_s"]),float(prm["VIEWING_RADIUS"]),float(cfg["planning_period"])
    n = aircraft.shape[1]
    views = target_views(aircraft,target,fov_half)
    base = calculate(time,aircraft,target,status,horizon,fov_half,events,result,views=views)
    visible,coverage = views
    weights = np.diff(time)
    t_end = float(time[-1])
    collision = result.get("collision")
    m = dict(controller=cfg["controller"],scenario=cfg["scenario"],preset=preset_name(cfg))

    # Paper metrics (Section IV-C): cut at t_c, the rest of [0, T) counts as a failure.
    m["gather_time_s"] = base["gather_time_mean_s"]
    m["reform_pct"] = base["reform_search_pct"]
    m["mission_visibility_pct"] = base["mission_visibility"]
    m["target_loss_pct"] = base["target_loss_pct"]
    m["coverage"] = base["mission_cov_eff"]
    execution = [e["execution_s"] for e in events]
    mean,p95,peak = _stats(execution,1e3)
    m["exec_mean_ms"],m["exec_p95_ms"] = mean,p95
    m["solve_mean_ms"],m["solve_p95_ms"],m["solve_max_ms"] = mean,p95,peak
    cut = "t_c" if collision else "the mission end"
    notes = []
    if collision:
        notes.append(f"collision at t_c = {collision['time']:.2f} s; metrics cut there")
    if m["gather_time_s"] is None:
        notes.append(f"gathering: only {base['gathered_uavs']}/{n} UAVs reached TRACK before {cut} "
                     f"({t_end:.1f} s)")
    if m["reform_pct"] is None:
        if not base["gathered_uavs"]:
            notes.append("re-formation: no UAV joined the formation")
        else:
            notes.append(f"re-formation: mean window {base['reform_window_mean_s']:.1f} s after joining "
                         f"< {100*REFORM_MIN_WINDOW:.0f} % of the {horizon:.1f} s mission")
    if not execution:
        notes.append("execution time: no planning step recorded")
    m["paper_notes"] = "; ".join(notes) or "none"

    finished = datetime.fromtimestamp((directory/"result.json").stat().st_mtime,timezone.utc)
    try:
        run_id = directory.resolve().relative_to(Path("runs").resolve()).as_posix()
    except ValueError:
        run_id = directory.name
    m.update(run_id=run_id,finished_utc=finished.strftime("%Y-%m-%d %H:%M:%S"),uavs=n,
             seed=cfg.get("seed"),scheduling=cfg.get("scheduling","sequential"),
             compute_ratio=cfg.get("compute_ratio",1.),
             command_execution=cfg.get("command_execution","hold_latest"),horizon_s=horizon,
             mission_time_s=t_end,termination=result.get("status"),
             waypoints_reached=_waypoints_reached(cfg["scenario_definition"].get("waypoints",[]),target),
             wall_time_s=result.get("wall_time_s"),real_time_factor=result.get("throughput"))

    m["collision_free"] = "no" if collision else "yes"
    m["completion_pct"] = base["completion_pct"]
    m["first_collision_s"] = collision["time"] if collision else None
    if collision:
        drones = ", ".join(str(d) for d in collision.get("drones",[]))
        m["collision_reason"] = collision["reason"]+(f" (UAV {drones})" if drones else "")
    else:
        m["collision_reason"] = "none"
    clearance = _obstacle_clearance(aircraft[:,:,:2].reshape(-1,2),cfg["scenario_definition"])
    m["min_obstacle_clearance_m"] = None if clearance is None else clearance-float(prm["ROBOT_RADIUS"])
    m["min_inter_uav_m"] = _min_pair_distance(aircraft[:,:,:3])
    m["min_altitude_m"] = float(aircraft[:,:,2].min())

    track = status[:,:,0].astype(bool)
    leader = status[:,:,1].astype(bool).any(axis=1)
    m["track_pct"] = 100*float(weights@track[:-1].mean(axis=1))/t_end if t_end > 0 else None
    m["gathered"] = f"{base['gathered_uavs']}/{n}"
    m["gather_team_s"] = base["gather_time_team_s"]
    m["reform_episodes"] = base["reform_episodes_per_joined_uav"]
    m["reform_episode_s"] = (base["reform_episode_s"] if base["reform_episode_s"] is not None
                             else 0. if base["gathered_uavs"] else None)
    # After the first instant at which the whole team is in TRACK.
    together = np.flatnonzero(track.all(axis=1))
    formed = int(together[0]) if len(together) else None
    if formed is not None and formed < len(time)-1:
        after = slice(formed,len(time)-1)
        m["leader_present_pct"] = 100*float(weights[after]@leader[after])/float(t_end-time[formed])
        m["leader_losses"] = int(np.sum(leader[formed:-1] & ~leader[formed+1:]))
    else:
        m["leader_present_pct"] = m["leader_losses"] = None

    t_team = base["gather_time_team_s"]
    if t_team is not None and horizon-t_team > 0 and len(time) > 1:
        mask = time[:-1] >= t_team
        m["visibility_after_gather_pct"] = 100*float(weights[mask]@visible[mask])/(horizon-t_team)
        m["coverage_after_gather"] = float(weights[mask]@coverage[mask])/(horizon-t_team)
        planar = np.linalg.norm(aircraft[:,:,:2]-target[:,None,:2],axis=2)[time >= t_team]
        nearest,fleet = planar.min(axis=1),planar.mean(axis=1)
        m["nearest_median_m"],m["nearest_p95_m"],m["nearest_max_m"] = (
            float(np.median(nearest)),float(np.percentile(nearest,95)),float(nearest.max()))
        m["fleet_dist_median_m"],m["fleet_dist_p95_m"] = float(np.median(fleet)),float(np.percentile(fleet,95))
    m["visibility_flown_pct"] = 100*float(weights@visible)/t_end if t_end > 0 else None
    m["coverage_flown"] = float(weights@coverage)/t_end if t_end > 0 else None

    host,late = _rounds(events,m["scheduling"])
    if host.size:
        m["round_median_ms"] = float(np.median(host))*1e3
        m["round_p95_ms"],m["round_max_ms"] = float(np.percentile(host,95))*1e3,float(host.max())*1e3
        m["slow_rounds_pct"] = 100*float(np.mean(host > period+1e-12))
        m["late_rounds_pct"] = 100*float(np.mean(late > period+1e-9))
    lat_mean,lat_p95,_ = _stats([e["latency_s"] for e in events],1e3)
    m["latency_mean_ms"],m["latency_p95_ms"] = lat_mean,lat_p95
    m["skipped_releases"] = result.get("skipped_releases")

    counters = {}
    for e in events:
        if e.get("solver_failures") is not None:
            counters[e["drone"]] = max(counters.get(e["drone"],0),int(e["solver_failures"]))
    m["solver_failures"] = sum(counters.values()) if counters else None
    m["backup_used"] = sum(bool(e.get("used_backup")) for e in events) if any(
        "used_backup" in e for e in events) else None
    m["upstream_stops"] = base["upstream_stop_fallbacks"]
    m["stale_plans"] = base["rejected_stale_plans"]
    m["undelivered"] = sum(e.get("delivered") is False for e in events)
    if braking is not None and len(time) > 1 and cfg.get("command_execution") == "tracked_plan_with_watchdog":
        b = braking_summary(time,braking)
        m["watchdog_pct_max"],m["watchdog_episodes"] = b["watchdog_braking_pct_max"],b["watchdog_braking_episodes"]
    return {k:_clean(v) for k,v in m.items()}


def _clean(value):
    if isinstance(value,(np.generic,)):
        value = value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------- tables

def _display(metric, value):
    if value is None or value == "":
        return "–"
    if metric.fmt is None:
        return str(value)
    if metric.fmt == "{:d}":
        return str(int(round(float(value))))
    return metric.fmt.format(float(value))


def _stored(metric, value):
    """CSV cell: full-precision number or text; empty when unavailable."""
    if value is None:
        return ""
    if metric.fmt is None:
        return str(value)
    if metric.fmt == "{:d}":
        return str(int(round(float(value))))
    return f"{float(value):.6g}"


def _parse(metric, text):
    if text is None or text == "":
        return None
    if metric.fmt is None:
        return text
    try:
        return float(text)
    except ValueError:
        return None


def _order(algorithms):
    known = [c for c in CONTROLLERS if c in algorithms]
    return known+sorted(a for a in algorithms if a not in CONTROLLERS)


def read_rows(path):
    """{algorithm: {key: value}} from a table CSV written by write_tables."""
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    with path.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get(ALGORITHM)
            if name:
                rows[name] = {BY_LABEL[k].key:_parse(BY_LABEL[k],v) for k,v in row.items() if k in BY_LABEL}
    return rows


def _md_section(section, rows, algorithms):
    lines = [f"## {section}","","| Metric | "+" | ".join(algorithms)+" |",
             "|---|"+"---:|"*len(algorithms)]
    for metric in (m for m in METRICS if m.section == section):
        if section != PAPER and all(rows[a].get(metric.key) is None for a in algorithms):
            continue
        lines.append(f"| {metric.label} | "+" | ".join(_display(metric,rows[a].get(metric.key))
                                                    for a in algorithms)+" |")
    return lines


def markdown(preset, rows, updated):
    algorithms = _order(rows)
    lines = [f"# Metrics: {preset}","",
             f"Updated {updated}. One column per algorithm (its latest collected run); the CSV and "
             "XLSX hold the same values with one row per algorithm.",""]
    lines += _md_section(PAPER,rows,algorithms)
    lines += ["","Evaluated on [0, T) with the first collision t_c as the cut (Section IV-C); \"–\" marks "
              "a metric the run cannot define, with the reason in the notes row (e.g. a mission too "
              "short for the team to gather or for a re-formation window).",""]
    for section in SECTIONS[1:]:
        lines += _md_section(section,rows,algorithms)+[""]
    lines += ["## Definitions","","| Metric | Definition |","|---|---|",f"| {ALGORITHM} | run --controller |"]
    lines += [f"| {m.label} | {m.definition} |" for m in METRICS]
    lines += ["","Gathering, re-formation, mission coverage and target loss follow upstream "
              "`thesis_metrics.py`; FOVs are squares of side 2 x VIEWING_RADIUS = 2L. Samples are "
              "time-weighted on the recorded physics grid. Compare rows only when their compute "
              "ratio and scheduling match.",""]
    return "\n".join(lines)


def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([ALGORITHM]+[m.label for m in METRICS])
        for a in _order(rows):
            writer.writerow([a]+[_stored(m,rows[a].get(m.key)) for m in METRICS])


def _excel_format(fmt):
    if fmt == "{:d}":
        return "0"
    match = re.fullmatch(r"\{:\.(\d+)f\}",fmt or "")
    if not match:
        return None
    digits = int(match.group(1))
    return "0."+"0"*digits if digits else "0"


def write_xlsx(path, rows, preset, updated):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    book = Workbook()
    head_font,head_fill = Font(bold=True),PatternFill("solid",fgColor="DDE5F0")
    algorithms = _order(rows)

    def sheet(ws, metrics):
        ws.append([ALGORITHM]+[m.label for m in metrics])
        for a in algorithms:
            cells = [a]
            for m in metrics:
                v = rows[a].get(m.key)
                cells.append(None if v is None else v if m.fmt is None
                             else int(round(float(v))) if m.fmt == "{:d}" else float(v))
            ws.append(cells)
        for cell in ws[1]:
            cell.font,cell.fill = head_font,head_fill
            cell.alignment = Alignment(wrap_text=True,vertical="top")
        for j,m in enumerate(metrics,start=2):
            number = _excel_format(m.fmt)
            for row in ws.iter_rows(min_row=2,min_col=j,max_col=j):
                if number:
                    row[0].number_format = number
            ws.column_dimensions[get_column_letter(j)].width = 40 if m.fmt is None else 16
        ws.column_dimensions["A"].width = 14
        ws.row_dimensions[1].height = 48
        ws.freeze_panes = "B2"

    paper = book.active
    paper.title = PAPER
    sheet(paper,[m for m in METRICS if m.section == PAPER])
    paper.append([])
    paper.append([f"{preset}; updated {updated}. Empty cell: unavailable, see the notes column."])
    sheet(book.create_sheet("All metrics"),METRICS)
    defs = book.create_sheet("Definitions")
    defs.append(["Metric","Section","Definition"])
    for m in METRICS:
        defs.append([m.label,m.section,m.definition])
    for cell in defs[1]:
        cell.font,cell.fill = head_font,head_fill
    defs.column_dimensions["A"].width,defs.column_dimensions["B"].width = 44,24
    defs.column_dimensions["C"].width = 110
    book.save(path)


def write_tables(out_dir, preset, rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    base = out_dir/preset
    write_csv(base.with_suffix(".csv"),rows)
    write_xlsx(base.with_suffix(".xlsx"),rows,preset,updated)
    base.with_suffix(".md").write_text(markdown(preset,rows,updated),encoding="utf-8")
    return [base.with_suffix(s) for s in (".xlsx",".csv",".md")]


def add_run(directory, out_dir="metrics", force=False):
    """Collect one run; returns (preset, algorithm, replaced row or None, skipped reason or None)."""
    m = evaluate(directory)
    preset,algorithm = m["preset"],m["controller"]
    path = Path(out_dir)/f"{preset}.csv"
    rows = read_rows(path)
    old = rows.get(algorithm)
    if old and not force and (old.get("finished_utc") or "") > m["finished_utc"]:
        return preset,algorithm,old,(f"stored run {old.get('run_id')} finished later "
                                     f"({old['finished_utc']} UTC); use --force to replace it")
    rows[algorithm] = {k:m.get(k) for k in BY_KEY}
    write_tables(out_dir,preset,rows)
    return preset,algorithm,old,None


# ---------------------------------------------------------------- selection

def _is_run(path):
    return all((path/name).exists() for name in ("config.json","result.json","trajectory.npz"))


def expand(path):
    """A run directory, or every run in a benchmark directory."""
    path = Path(path)
    if _is_run(path):
        return [path]
    children = sorted((p for p in path.iterdir() if p.is_dir() and _is_run(p)),key=lambda p:p.name) \
        if path.is_dir() else []
    if not children:
        raise SystemExit(f"collect: {path} is not a finished run (config.json, result.json, "
                         "trajectory.npz) or a benchmark directory of runs")
    return children


def resolve(name, root=Path("runs")):
    for candidate in (Path(name),root/name):
        if candidate.exists():
            return expand(candidate)
    raise SystemExit(f"collect: no run named {name} (looked in . and {root}/)")


def available(root=Path("runs")):
    """Finished runs under root (benchmark children included), newest first."""
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and _is_run(p)]
    runs += [c for p in root.iterdir() if p.is_dir() and not _is_run(p)
             for c in p.iterdir() if c.is_dir() and _is_run(c)]
    return sorted(runs,key=lambda p:(p/"result.json").stat().st_mtime,reverse=True)


def describe(path):
    try:
        cfg = json.loads((path/"config.json").read_text())
        result = json.loads((path/"result.json").read_text())
        return (f"{cfg['controller']:<12} {preset_name(cfg):<24} r={cfg.get('compute_ratio',1.):g} "
                f"{result.get('status')}")
    except (OSError,KeyError,ValueError):
        return "unreadable"


def choose(root=Path("runs")):
    runs = available(root)
    if not runs:
        raise SystemExit(f"collect: no finished runs under {root}/")
    if not sys.stdin.isatty():
        listing = "\n".join(f"  {p.relative_to(root)}" for p in runs[:20])
        raise SystemExit("collect: name the runs to collect, or pass --latest. Newest runs:\n"+listing)
    for k,p in enumerate(runs,start=1):
        print(f"{k:>3}  {str(p.relative_to(root)):<44} {describe(p)}")
    answer = input("Runs to collect (numbers separated by spaces or commas, Enter = 1): ").strip() or "1"
    picked = []
    for token in answer.replace(","," ").split():
        if not token.isdigit() or not 1 <= int(token) <= len(runs):
            raise SystemExit(f"collect: invalid choice {token!r}")
        picked.append(runs[int(token)-1])
    return picked


def main(args):
    if args.latest:
        runs = available()[:1]
        if not runs:
            raise SystemExit("collect: no finished runs under runs/")
    elif args.runs:
        runs = [r for name in args.runs for r in resolve(name)]
    else:
        runs = choose()
    written = set()
    for run in runs:
        print(f"collect: {run} ...",flush=True)
        preset,algorithm,old,skipped = add_run(run,args.out_dir,args.force)
        if skipped:
            print(f"  skipped {algorithm} in {preset}: {skipped}")
            continue
        replaced = f" (replaced {old.get('run_id')})" if old else ""
        print(f"  {algorithm} -> {Path(args.out_dir)/preset}.{{xlsx,csv,md}}{replaced}")
        written.add(preset)
        rows = read_rows(Path(args.out_dir)/f"{preset}.csv")
        mixed = {(r.get("compute_ratio"),r.get("scheduling")) for r in rows.values()}
        if len(mixed) > 1:
            print(f"  warning: {preset} mixes compute ratio / scheduling across algorithms: "
                  + ", ".join(f"r={r:g} {s}" for r,s in sorted(mixed,key=str) if r is not None))
    return 0
