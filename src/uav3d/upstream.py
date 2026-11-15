"""Read-only integration with the pinned upstream's import-time configuration."""
import os
import sys
import copy
from pathlib import Path
import numpy as np
from .collision import CollisionWorld

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("UAV3D_UPSTREAM", ROOT / "motion_planning_MPC"))


def load(scenario="scen1", controller="ours"):
    """Upstream config for a scenario; the drone count is len(cfg.STARTS)."""
    if "config" in sys.modules:
        raise RuntimeError("Use a fresh process for each scenario/controller experiment")
    os.environ["SCENARIO"] = scenario
    os.environ["CONTROLLER"] = controller
    os.environ["TARGET_MODE"] = "manual"
    sys.path.insert(0, str(UPSTREAM))
    import config as cfg
    cfg.CONTROLLER = controller
    if cfg.MPC_BACKEND != "parametric":
        raise ValueError("3D adapter requires the upstream parametric MPC backend")
    world = CollisionWorld(cfg.RECTANGLE_OBSTACLES, cfg.OBSTACLES,
                           cfg.WORLD_BOUNDS, cfg.ROBOT_RADIUS)
    starts = cfg.STARTS.copy()
    return cfg, world, starts


def make_robots(cfg, starts):
    from robot_jps import Robot
    class AdapterRobot(Robot):
        def updateState(self, control, dt):
            if abs(dt-cfg.TIMESTEP) > 1e-12:
                raise ValueError("Unexpected planner period")
            u = np.asarray(control,float)
            if u.shape != (3,) or not np.isfinite(u).all() or abs(u[2]) > 1e-9:
                raise ValueError(f"Invalid horizontal acceleration command: {u}")
            self.requested_command = u.copy()

        def _backup_control(self):
            # Flag the CONTINGENCY fallback: its command comes from Ub, not controls_prediction.
            self.used_backup = True
            return super()._backup_control()

    Robot._team_assign = {"step":None,"key":None,"assign":None}
    Robot._team_leader = {"step":None,"leader":None}
    return [AdapterRobot(i,np.r_[p,np.zeros(3)],p.copy()) for i,p in enumerate(starts)]


# Plan tracking for "ours" (adapter-side; upstream is unchanged). The PD loop
# has natural frequency 3 rad/s and damping 1; with upstream's ACCEL_TAU = 0.2 s
# response lag it stays stable (tau*kp < kd). A watchdog stop may use up to
# 80% of UMAX, leaving the rest for the tracking correction.
TRACK_KP, TRACK_KD, STOP_ACCEL_RATIO = 9.0, 6.0, 0.8
# UMAX limits the planner; the Hummingbird (thrust/weight ~10) can do more. The
# tracking correction may raise the horizontal command to 1.5*UMAX (~37 deg tilt
# at UMAX = 5), so a plan that brakes at UMAX can still be followed.
TRACK_ACCEL_RATIO = 1.5
VMAX_GUARD_BAND = 0.5          # m/s below VMAX at which speed-up is cut


def plan_schedule(robot, cfg, release, previous_backup):
    """The plan behind the command "ours" just issued, and whether it is a stop.

    Returns (schedule, stop_current). A fresh plan, braking prediction, or
    shifted (age_robot) previous plan starts at release, row 0. When upstream
    used its CONTINGENCY backup (a failed solve), the aircraft has been flying
    the current plan rather than the backup branch, so stop_current asks to
    stop along the current plan; the backup branch is only a fallback when no
    plan was ever delivered.
    """
    from .scheduler import PlanSchedule
    backup = getattr(robot,"_backup",None)
    if backup is not None and backup is not previous_backup:
        robot.backup_epoch = release      # a successful solve made a fresh backup branch
    stop = getattr(robot,"used_backup",False)
    if stop:
        start,states,controls = robot.backup_epoch,backup[0][:,:4],backup[1]
    else:
        start = release
        states = robot.states_prediction[:,[0,1,3,4]]
        controls = robot.controls_prediction[:,:2]
    schedule = PlanSchedule(start,cfg.TIMESTEP,np.array(states,float,copy=True),
                            np.array(controls,float,copy=True),cfg.UMAX,
                            TRACK_KP,TRACK_KD,STOP_ACCEL_RATIO*cfg.UMAX,cfg.VMAX,
                            TRACK_ACCEL_RATIO*cfg.UMAX,VMAX_GUARD_BAND)
    k = schedule.index(release+cfg.TIMESTEP/2)
    if k < len(controls) and not np.allclose(controls[k],robot.requested_command[:2],atol=1e-9):
        raise RuntimeError("Plan schedule does not reproduce the issued command")
    return schedule,stop


# Upstream's own cumulative solver-failure counter per controller (read only).
FAILURE_COUNTERS = {"ours":("mpc","n_fail"),"pure_mpc":("simple","n_fail"),
                    "cbf_qp_lag":("cbfqp","n_infeasible"),"impc_ob_rec":("impc.prob","n_fail")}


def failure_count(robot, controller):
    """Upstream's cumulative failed/infeasible solves of this robot, or None (e.g. APF)."""
    if controller not in FAILURE_COUNTERS:
        return None
    path,name = FAILURE_COUNTERS[controller]
    owner = robot
    for part in path.split("."):
        owner = getattr(owner,part,None)
    value = getattr(owner,name,None)
    return None if value is None else int(value)


def publish_prediction(robot, schedule, release):
    """Replace upstream's prediction by the trajectory this aircraft really follows."""
    rows = schedule.prediction(release,len(robot.states_prediction))
    z = robot.state[2]
    robot.states_prediction = np.column_stack([rows[:,:2],np.full(len(rows),z),
                                               rows[:,2:],np.zeros(len(rows))])


def age_robot(robot):
    """Shift the horizon buffers by one control step, as upstream's updateState does.

    The adapter replaces updateState (the plant is RotorPy), so this shift runs
    here, before every planning round except the first."""
    if not getattr(robot,"planned",False):
        return
    for name in ("states_prediction","controls_prediction","_mpc_X","_mpc_U"):
        array = getattr(robot,name,None)
        if array is not None:
            array[:-1] = array[1:]


def initialize(cfg, starts, target):
    """Warm compilation/solves on disposable state, build fresh live solver objects."""
    from robot_jps import _MPC_CFG
    if cfg.PLANNER == "local":
        from planner_grid import warmup
        warmup()
    disposable = make_robots(cfg,starts)
    for tick in range(2):
        for robot in disposable:
            robot.time_stamp = tick*cfg.TIMESTEP
            robot.goal = target.copy()
            robot.computeControlSignal(disposable)
    live = make_robots(cfg,starts)
    conf = _MPC_CFG()
    for robot,warm in zip(live,disposable):
        if cfg.CONTROLLER == "ours" and warm.mpc is not None:
            # CasADi compiles each solver instance lazily on its first solve.
            # Keep the warmed solver workspace, but discard all experiment state.
            robot.mpc = warm.mpc
            robot.mpc.last_backup = None
            robot.mpc._bk_init = None
            robot.mpc.solve_times = []
            robot.mpc.n_fail = robot.mpc.n_fallback = 0
        elif cfg.CONTROLLER == "apf":
            from apf import APF
            robot.simple = APF(robot,conf)
        elif cfg.CONTROLLER == "pure_mpc":
            robot.simple = copy.copy(warm.simple)
            robot.simple.robot = robot
            robot.simple.Xp = robot.simple.Up = None
            robot.simple.left = robot.simple.n_fail = 0
            robot.simple.solve_times = []
        elif cfg.CONTROLLER == "cbf_qp_lag":
            from cbfqp import CBFQP
            robot.cbfqp = CBFQP(robot,conf,lag_aware=True)
        elif cfg.CONTROLLER == "impc_ob_rec":
            from impc_ob import IMPCOB
            robot.impc = IMPCOB(robot,len(live)-1,conf,recover=True)
            warm_impc = getattr(warm,"impc",None)
            if warm_impc is not None:
                robot.impc.n_faces = warm_impc.n_faces
                robot.impc.prob = warm_impc.prob
                robot.impc.prob.n_fail = 0
                robot.impc.prob.solve_times = []
    return live
