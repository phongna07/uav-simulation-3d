"""Read-only integration with the pinned upstream's import-time configuration."""
import os
import sys
import copy
from pathlib import Path
import numpy as np
from .collision import CollisionWorld

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("UAV3D_UPSTREAM", ROOT / "motion_planning_MPC"))


def load(scenario="scen1", controller="ours", drones=None):
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
    if drones is not None and drones != len(starts):
        if drones != 9 or len(starts) != 5:
            raise ValueError("Drone count must match the scenario, or expand five starts to nine")
        xmin,ymin,xmax,ymax = cfg.WORLD_BOUNDS
        spacing = max(cfg.GRID_RESOLUTION, 2*cfg.ROBOT_RADIUS+0.01)
        xs,ys = np.meshgrid(np.arange(xmin+spacing,xmax,spacing),
                            np.arange(ymin+spacing,ymax,spacing))
        candidates = np.column_stack([xs.ravel(),ys.ravel()])
        center = starts[:,:2].mean(axis=0)
        order = np.lexsort((candidates[:,1],candidates[:,0],np.sum((candidates-center)**2,axis=1)))
        points = list(starts)
        for k in order:
            p = np.r_[candidates[k],starts[0,2]]
            if world.valid_start(p, points):
                points.append(p)
                if len(points) == drones:
                    break
        if len(points) != drones:
            raise ValueError("Cannot place four additional drones safely on the resolved map")
        starts = np.array(points)
        cfg.STARTS, cfg.GOALS, cfg.NUM_ROBOT = starts, starts.copy(), drones
        cfg.SCENARIO_DEF["starts"] = starts.tolist()
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


def plan_schedule(robot, cfg, release, previous_backup):
    """The time-indexed control sequence behind the command "ours" just issued.

    Upstream applies controls_prediction[0] for one period and the rest of the
    sequence on later periods; the CONTINGENCY fallback applies Ub, aligned with
    the solve that produced it. Executing the whole sequence keeps the aircraft
    on the trajectory the MPC verified, however late the next plan arrives.
    """
    from .scheduler import PlanSchedule
    backup = getattr(robot,"_backup",None)
    if backup is not None and backup is not previous_backup:
        robot.backup_epoch = release      # a successful solve made a fresh backup branch
    if getattr(robot,"used_backup",False):
        start,controls = robot.backup_epoch,backup[1]
    else:
        # Fresh plan, braking prediction, or shifted (age_robot) previous plan: row 0 is now.
        start,controls = release,robot.controls_prediction[:,:2]
    tau = getattr(cfg,"ACCEL_TAU",0.)
    schedule = PlanSchedule(start,cfg.TIMESTEP,np.array(controls,float,copy=True),
                            1/(2*tau) if tau > 0 else 1/cfg.TIMESTEP,cfg.UMAX)
    k = schedule.index(release+cfg.TIMESTEP/2)
    if k < len(controls) and not np.allclose(controls[k],robot.requested_command[:2],atol=1e-9):
        raise RuntimeError("Plan schedule does not reproduce the issued command")
    return schedule


def set_tracker_interval(tracker, dt):
    """Keep the upstream CV filter state, updating its F/Q for elapsed mission time."""
    q2 = float(tracker.Q[2,2]/tracker.dt**2)
    tracker.dt = dt
    tracker.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
    tracker.Q = q2*np.array([[dt**4/4,0,dt**3/2,0],
                            [0,dt**4/4,0,dt**3/2],
                            [dt**3/2,0,dt**2,0],
                            [0,dt**3/2,0,dt**2]])


def shift_rows(array, steps):
    if array is None or steps <= 0:
        return
    s = min(steps,len(array)-1)
    array[:] = np.concatenate([array[s:],np.repeat(array[-1:],s,axis=0)])


def age_robot(robot, released, period):
    """Age horizon buffers once per elapsed planning interval, never per physics step."""
    previous = getattr(robot,"prediction_epoch",released)
    steps = max(0,int(np.floor((released-previous)/period+1e-9)))
    for name in ("states_prediction","controls_prediction","_mpc_X","_mpc_U"):
        shift_rows(getattr(robot,name,None),steps)
    # Upstream fallback methods advance one entry themselves when invoked.
    # Prepare their cursors at elapsed_steps-1 so that they select time-correct entries.
    if hasattr(robot,"_backup") and robot._backup is not None:
        robot._bk_i = min(len(robot._backup[1]),getattr(robot,"_bk_i",0)+max(0,steps-1))
    if hasattr(robot,"_plan_left"):
        robot._plan_left = max(0,robot._plan_left-max(0,steps-1))
    for planner in (getattr(robot,"simple",None),getattr(robot,"impc",None)):
        if planner is None:
            continue
        if hasattr(planner,"left"):
            planner.left = max(0,planner.left-max(0,steps-1))
        # PureMPC shifts its warm-start once internally. IMPC stores an already-shifted warm start.
        for name in ("Xp","Up","_X","_Uw"):
            shift_rows(getattr(planner,name,None),max(0,steps-1))
    robot.prediction_epoch = previous+steps*period


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
