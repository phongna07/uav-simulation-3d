"""Commands always start experiments in fresh, single-thread-limited processes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from . import CONTROLLERS


def parser():
    p = argparse.ArgumentParser(description="RotorPy adapter; default preset: upstream scen1")
    sub = p.add_subparsers(dest="command",required=True)
    for name in ("run","benchmark","_worker"):
        q = sub.add_parser(name,help="isolated experiment" if name=="run" else "all five methods" if name=="benchmark" else argparse.SUPPRESS)
        q.add_argument("--scenario",default="scen1")
        q.add_argument("--controller",choices=CONTROLLERS,default="ours")
        q.add_argument("--seed",type=int,default=0)
        q.add_argument("--duration",type=float,help="mission seconds; default: full target trajectory")
        q.add_argument("--physics-dt",type=float,default=.005)
        q.add_argument("--physics-workers",default=os.environ.get("UAV3D_PHYSICS_WORKERS","auto"),
                       help="processes that step the aircraft each physics step: 'auto' "
                            "(default; one per drone, capped by available CPUs) or N; 1 = serial "
                            "in-process stepping. Results are bit-identical for every value "
                            "[env UAV3D_PHYSICS_WORKERS]")
        q.add_argument("--out",help="new output directory (must not already exist)")
        q.add_argument("--timings-from",help="run directory containing previously measured delays")
        q.add_argument("--compute-ratio",type=float,default=0.,
                       help="scale every measured planning duration by r before injecting it as "
                            "mission-time delay (e.g. 0.5: onboard computer twice as fast, 1: host "
                            "speed); default 0: instantaneous planning, physics paused while solving")
        if name=="benchmark":
            q.add_argument("--controllers",nargs="+",choices=CONTROLLERS,default=list(CONTROLLERS))
    q = sub.add_parser("metrics",help="recompute timestamp-based metrics from recorded data")
    q.add_argument("directory")
    q = sub.add_parser("collect",help="add recorded runs to the per-preset tables in metrics/")
    q.add_argument("runs",nargs="*",help="run directories, or names under runs/ (a benchmark "
                   "directory adds each method); default: choose interactively")
    q.add_argument("--latest",action="store_true",help="collect the most recently finished run")
    q.add_argument("--out-dir",default="metrics",help="table directory (default: metrics)")
    q.add_argument("--force",action="store_true",
                   help="replace an algorithm's row even if the stored run finished later")
    q = sub.add_parser("replay",help="offline 3D PNG/MP4 rendering; never runs planners")
    q.add_argument("directory")
    q.add_argument("--out",help=".png or .mp4 output path; default: run/replay.png")
    q.add_argument("--time",type=float,default=None,help="mission time for PNG; default: final recorded time")
    q.add_argument("--fps",type=int,default=24)
    q.add_argument("--speed",type=float,default=1.,help="mission seconds per video second")
    q.add_argument("--camera",choices=("follow","overview"),default="follow",
                   help="follow the target (default) or show the full map")
    q.add_argument("--zoom",type=float,default=6.,
                   help="follow camera magnification relative to the full map (default: 6)")
    env = os.environ.get
    q.add_argument("--jobs",default=env("UAV3D_REPLAY_JOBS","auto"),
                   help="MP4 render worker processes: 'auto' (default; sized from available "
                        "CPUs, memory and frame count) or N; 1 = serial export "
                        "[env UAV3D_REPLAY_JOBS]")
    q.add_argument("--max-memory",default=env("UAV3D_REPLAY_MAX_MEMORY_GB","auto"),
                   help="memory budget in GiB used to cap '--jobs auto' (default: 80%% of "
                        "available memory) [env UAV3D_REPLAY_MAX_MEMORY_GB]")
    q.add_argument("--queue-frames",type=int,default=int(env("UAV3D_REPLAY_QUEUE_FRAMES","4")),
                   help="encoded-size frames each worker may buffer in memory ahead of the "
                        "encoder (default: 4) [env UAV3D_REPLAY_QUEUE_FRAMES]")
    q.add_argument("--encoder-threads",default=env("UAV3D_REPLAY_ENCODER_THREADS","auto"),
                   help="x264 threads: 'auto' (default, ffmpeg's choice) or N "
                        "[env UAV3D_REPLAY_ENCODER_THREADS]")
    return p


def worker_command(args, controller, output):
    command = [sys.executable,"-m","uav3d.cli","_worker","--scenario",args.scenario,
               "--controller",controller,"--seed",str(args.seed),"--physics-dt",str(args.physics_dt),
               "--compute-ratio",repr(args.compute_ratio),
               "--physics-workers",str(args.physics_workers),"--out",str(output)]
    for name in ("duration","timings_from"):
        value = getattr(args,name)
        if value is not None:
            command += ["--"+name.replace("_","-"),str(value)]
    return command


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command in ("run","benchmark","_worker"):
        import math
        if not math.isfinite(args.physics_dt) or args.physics_dt<=0:
            p.error("--physics-dt must be finite and positive")
        if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0):
            p.error("--duration must be finite and positive")
        if not math.isfinite(args.compute_ratio) or args.compute_ratio<0:
            p.error("--compute-ratio must be finite and nonnegative")
        if args.physics_workers!="auto" and not (args.physics_workers.isdigit() and int(args.physics_workers)>=1):
            p.error("--physics-workers must be 'auto' or a positive integer")
    if args.command=="_worker":
        from threadpoolctl import threadpool_limits
        from .experiment import run
        with threadpool_limits(limits=1):
            return run(args)
    if args.command in ("run","benchmark"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out = Path(args.out or f"runs/{args.scenario}_{stamp}")
        env = os.environ.copy()
        env.update({k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")})
        env.update(PYTHONHASHSEED=str(args.seed),MPLBACKEND="Agg",PYTHONUNBUFFERED="1")
        if args.command=="run":
            return subprocess.call(worker_command(args,args.controller,out),env=env)
        if args.timings_from:
            p.error("--timings-from is supported by run, not a multi-method benchmark")
        out.mkdir(parents=True,exist_ok=False)
        summary = []
        for controller in args.controllers:
            child = out/controller
            code = subprocess.call(worker_command(args,controller,child),env=env)
            record = dict(controller=controller,exit_code=code,directory=str(child))
            if (child/"metrics.json").exists():
                record.update(json.loads((child/"metrics.json").read_text()))
            summary.append(record)
            (out/"benchmark.json").write_text(json.dumps(summary,indent=2)+"\n")
            if code==130:
                return code
        import csv
        keys = sorted({key for row in summary for key in row})
        with (out/"benchmark.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)
        return int(any(row["exit_code"] for row in summary))
    if args.command=="metrics":
        from .metrics import report
        print(json.dumps(report(args.directory),indent=2))
        return 0
    if args.command=="collect":
        from .collect import main as collect
        return collect(args)
    if args.command=="replay":
        import math
        if args.fps<=0 or args.speed<=0:
            p.error("--fps and --speed must be positive")
        if not math.isfinite(args.zoom) or args.zoom<=0:
            p.error("--zoom must be finite and positive")
        from .video import plan_export
        try:
            plan_export(1, args.jobs, args.max_memory, args.queue_frames, args.encoder_threads,
                        cpus=1, memory=None)
        except ValueError as error:
            p.error(str(error))
        from .replay import render
        render(args)
        return 0


if __name__=="__main__":
    raise SystemExit(main())
