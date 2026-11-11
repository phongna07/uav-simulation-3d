import numpy as np
import pytest
from uav3d.scheduler import sequential_events,next_release
from uav3d.physics import RotorPyBackend
from uav3d.collision import CollisionWorld
from uav3d.experiment import Simulation


def test_queue_and_grid():
    events=list(sequential_events([.04,.08,.15]))
    assert [e.completed_at for e in events]==pytest.approx([.04,.12,.27])
    assert events[2].queue_wait_s==pytest.approx(.12)
    assert next_release(.27,0,.1)==pytest.approx((.3,2))
    assert next_release(.3,0,.1)==pytest.approx((.3,2))
    with pytest.raises(ValueError):
        list(sequential_events([-1]))


def test_motion_continues_during_pending_solve():
    physics=RotorPyBackend([[0,0,3]])
    physics.states[0]["v"][0]=1.
    collision=CollisionWorld([],[],[-10,-10,10,10],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,1.,.005)
    sim.advance(.07)
    assert sim.t==pytest.approx(.07)
    assert physics.positions()[0,0]==pytest.approx(.07,abs=1e-6)
    assert sim.target(.07)[0]==pytest.approx(.07)
    assert not physics.commands.any()
    assert not sim.status.any()  # No output published before command delivery.


def test_collision_during_pending_solve_stops_before_delivery():
    physics=RotorPyBackend([[0,0,3]])
    physics.states[0]["v"][0]=10.
    collision=CollisionWorld([[1,-1,1,2]],[],[-10,-10,10,10],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,1.,.005)
    sim.advance(.2)
    assert sim.hit["reason"]=="obstacle_rectangle"
    assert sim.t==pytest.approx(.08,abs=1e-6)
    assert not physics.commands.any()


def test_parallel_round_delivers_together_after_slowest():
    from uav3d.scheduler import parallel_events
    events=list(parallel_events([.04,.15,.08],release=.3,round_index=2))
    assert [e.started_at for e in events]==pytest.approx([.3,.3,.3])
    assert [e.completed_at for e in events]==pytest.approx([.45,.45,.45])
    assert [e.queue_wait_s for e in events]==pytest.approx([0,0,0])
    assert [e.injected_delay_s for e in events]==pytest.approx([.04,.15,.08])
    with pytest.raises(ValueError):
        list(parallel_events([.1,float("nan")]))


def test_parallel_planners_see_only_pre_round_predictions():
    from types import SimpleNamespace
    from uav3d.scheduler import plan_against_snapshot
    robots=[SimpleNamespace(states_prediction=np.full((3,2),float(i))) for i in range(3)]
    seen=[]
    def plan(i,robot):
        seen.append([float(r.states_prediction[0,0]) for r in robots])
        robot.states_prediction=np.full((3,2),10.+i)  # fresh plan
        return i
    assert plan_against_snapshot(robots,plan)==[0,1,2]
    others=[[v for j,v in enumerate(row) if j!=i] for i,row in enumerate(seen)]
    assert others==[[1,2],[0,2],[0,1]]  # never a same-round plan
    assert [r.states_prediction[0,0] for r in robots]==[10,11,12]



def line_plan(start, speed, steps, period=.1):
    """A constant-velocity plan along +x from the origin."""
    t=np.arange(steps+1)*period
    states=np.column_stack([speed*t,0*t,np.full_like(t,speed),0*t])
    return states,np.zeros((steps,2))


def test_plan_schedule_feedforward_and_tracking():
    from uav3d.scheduler import PlanSchedule
    states,controls=line_plan(0,1.,20)
    controls[3]=[2.,0]
    s=PlanSchedule(.3,.1,states,controls,5.,9.,6.,4.)
    p,v,u=s.reference(.65)                                  # step 3, halfway
    assert p==pytest.approx([.35,0]) and v==pytest.approx([1,0]) and u==pytest.approx([2,0])
    assert s.reference(.4+1e-12)[2]==pytest.approx([0,0])   # boundary rounding
    assert s.command(.65,[.35,.1],[1,0])==pytest.approx([2,-.9])  # pulls back onto the path
    assert np.linalg.norm(s.command(.65,[0,-5],[0,0]))==pytest.approx(5)  # limited to umax
    wide=PlanSchedule(.3,.1,states,controls,5.,9.,6.,4.,track_umax=7.5)
    u=wide.command(.65,[0,-5],[0,0])
    assert np.linalg.norm(u)==pytest.approx(7.5)                # correction may exceed umax
    braking=PlanSchedule(0.,.1,states,np.tile([-5.,0],(20,1)),5.,9.,6.,4.,track_umax=7.5)
    assert braking.command(.05,[.05,3],[1,0])[0]==pytest.approx(-5,abs=1e-9)  # planned braking kept
    with pytest.raises(ValueError):
        s.command(.2,[0,0],[0,0])


def test_watchdog_stops_along_remaining_plan_path():
    from uav3d.scheduler import PlanSchedule
    states,controls=line_plan(0,8.,20)                       # 2 s plan, 16 m
    s=PlanSchedule(0.,.1,states,controls,5.,9.,6.,4.)
    # Latest step from which 8 m/s stops within the remaining plan at <= 4 m/s^2.
    assert s.stop_at==pytest.approx(1.)
    brake=2*(s.end-s.stop_at)
    p,v,_=s.reference(s.stop_at+brake/2)
    assert 0<v[0]<8 and p[0]<16
    p,v,_=s.reference(s.stop_at+brake+1)
    assert p==pytest.approx([16,0]) and v==pytest.approx([0,0])  # holds the plan's end
    early=s.stopped(.25)
    assert early.stop_at==pytest.approx(.25)
    assert s.stopped(5.).stop_at==pytest.approx(1.)          # never later than its own watchdog
    path=early.prediction(0.,40)
    assert np.all(np.diff(path[:,0])>=-1e-9) and path[-1,0]==pytest.approx(16)
    assert path[-1,2]==pytest.approx(0)


def test_tracked_plan_resists_drift_and_stops():
    from uav3d.scheduler import PlanSchedule
    physics=RotorPyBackend([[0,0,3]])
    collision=CollisionWorld([],[],[-50,-50,50,50],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,6.,.005)
    states,controls=line_plan(0,0.,20)
    sim.schedules[0]=PlanSchedule(0.,.1,states,controls,5.,9.,6.,4.)
    physics.states[0]["v"][:2]=[1.,1.]                      # disturbance off the plan
    sim.advance(3.)
    assert np.linalg.norm(physics.positions()[0,:2])<.05     # pulled back to the held point
    assert np.linalg.norm(physics.velocities()[0,:2])<.05


def test_speed_guard_and_stale_plan_rejection():
    from types import SimpleNamespace
    from uav3d.scheduler import PlanSchedule
    from uav3d.upstream import plan_valid
    states,controls=line_plan(0,2.,20)
    controls[:]=[3.,1.]
    s=PlanSchedule(0.,.1,states,controls,5.,9.,6.,4.,vmax=2.)
    assert s.command(.05,[.1,0],[2,0])==pytest.approx([0,1])   # no further speed-up at vmax
    banded=PlanSchedule(0.,.1,states,controls,5.,9.,6.,4.,vmax=2.,vmax_band=.5)
    assert banded.command(.05,[.1,0],[1.6,0])[0]==pytest.approx(0)  # inside the band
    assert banded.command(.05,[.1,0],[2.5,0])[0]==pytest.approx(-3)  # above: kd*(vmax-speed)
    assert s.command(.05,[.1,0],[1,0])[0]>0
    cfg=SimpleNamespace(ROBOT_RADIUS=.5)
    assert plan_valid(s,.5,[1.1,.1],[2,0],cfg)
    assert not plan_valid(s,.5,[1.5,0],[2,0],cfg)               # 0.5 m behind the plan
    assert not plan_valid(s,.5,[1,0],[-.5,0],cfg)               # stopped while it assumed 2 m/s
