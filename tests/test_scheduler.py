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


def test_plan_schedule_indexes_by_mission_time_then_brakes():
    from uav3d.scheduler import PlanSchedule
    s=PlanSchedule(.3,.1,np.array([[1.,0],[2,0],[3,0]]),2.5,5.)
    assert s.command(.3,[0,0])==pytest.approx([1,0])
    assert s.command(.4+1e-12,[0,0])==pytest.approx([2,0])  # boundary rounding
    assert s.command(.55,[0,0])==pytest.approx([3,0])
    assert s.command(.65,[1,0,0])==pytest.approx([-2.5,0])  # brake tail, u=-gain*v
    assert s.command(.65,[10,0,0])==pytest.approx([-5,0])   # limited to umax
    with pytest.raises(ValueError):
        s.command(.2,[0,0])


def test_late_plan_is_executed_from_its_current_step():
    from uav3d.scheduler import PlanSchedule
    physics=RotorPyBackend([[0,0,3]])
    collision=CollisionWorld([],[],[-50,-50,50,50],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,2.,.005)
    sim.advance(.25)                    # plan released at 0, delivered 0.25 s late
    sim.schedules[0]=PlanSchedule(0.,.1,np.array([[5.,0],[5,0],[0,0],[-5,0]]),2.5,5.)
    sim.apply_schedules(sim.t)
    assert physics.commands[0,:2]==pytest.approx([0,0])  # step 2, not U[0]
    sim.advance(.35)
    assert physics.commands[0,:2]==pytest.approx([-5,0])
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,4.,.005)
    sim.t=.35
    sim.schedules[0]=PlanSchedule(0.,.1,np.zeros((4,2)),2.5,5.)
    sim.advance(4.)                     # plan exhausted: brake tail settles to rest
    assert np.linalg.norm(physics.velocities()[0,:2])<.05
