import numpy as np
import pytest
from uav3d.physics import RotorPyBackend
from uav3d.collision import CollisionWorld
from uav3d.experiment import Simulation


def test_motion_between_releases_without_commands():
    physics=RotorPyBackend([[0,0,3]])
    physics.states[0]["v"][0]=1.
    collision=CollisionWorld([],[],[-10,-10,10,10],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,1.,.005)
    sim.advance(.07)
    assert sim.t==pytest.approx(.07)
    assert physics.positions()[0,0]==pytest.approx(.07,abs=1e-6)
    assert sim.target(.07)[0]==pytest.approx(.07)
    assert not physics.commands.any()
    assert not sim.status.any()  # No output published before the first command.


def test_collision_between_releases_stops_the_mission():
    physics=RotorPyBackend([[0,0,3]])
    physics.states[0]["v"][0]=10.
    collision=CollisionWorld([[1,-1,1,2]],[],[-10,-10,10,10],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,1.,.005)
    sim.advance(.2)
    assert sim.hit["reason"]=="obstacle_rectangle"
    assert sim.t==pytest.approx(.08,abs=1e-6)
    assert not physics.commands.any()


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



def test_cli_has_no_delay_options():
    from uav3d.cli import parser, worker_command
    args = parser().parse_args(["run","--duration","2"])
    assert not hasattr(args,"compute_ratio") and not hasattr(args,"timings_from")
    command = worker_command(args,"ours","runs/x")
    assert command[command.index("--duration")+1] == "2.0"
    assert "--compute-ratio" not in command
    with pytest.raises(SystemExit):
        parser().parse_args(["run","--compute-ratio","0.5"])


def test_speed_guard_cuts_speed_up_near_vmax_and_brakes_above():
    from uav3d.scheduler import SpeedGuard
    guard=SpeedGuard(vmax=10.,band=.5,kd=6.)
    u=np.array([3.,2.])
    assert np.allclose(guard.command(u,[9.,0.]),u)            # below VMAX-band: untouched
    assert np.allclose(guard.command(u,[9.6,0.]),[0.,2.])     # speed-up removed, lateral kept
    assert np.allclose(guard.command([-1.,0.],[9.8,0.]),[-1.,0.])  # braking is kept
    assert np.allclose(guard.command(u,[10.5,0.]),[-3.,2.])   # above VMAX: brake 6*(10-10.5)


def test_held_command_passes_through_the_speed_guard_at_physics_rate():
    from uav3d.scheduler import SpeedGuard
    physics=RotorPyBackend([[0,0,3]])
    physics.states[0]["v"][0]=9.8
    collision=CollisionWorld([],[],[-100,-100,100,100],.2)
    sim=Simulation(physics,collision,np.array([[0,0,0],[1,0,0]]),1.,1.,.005)
    sim.guard=SpeedGuard(10.,.5,6.)
    sim.hold(0,np.array([5.,1.,0.]))
    assert np.allclose(physics.commands[0,:2],[0.,1.])
    assert np.allclose(sim.held[0],[5.,1.,0.])                 # the delivered command is kept
    sim.advance(.5)
    assert np.linalg.norm(physics.velocities()[0,:2])<10.


def test_speed_guard_is_on_by_default_and_can_be_disabled():
    from uav3d.cli import parser, worker_command
    args=parser().parse_args(["run","--duration","2"])
    assert args.speed_guard_band==.5
    assert not hasattr(args,"command_execution")
    command=worker_command(args,"apf","runs/x")
    assert command[command.index("--speed-guard-band")+1]=="0.5"
    off=parser().parse_args(["run","--speed-guard-band","off"])
    assert off.speed_guard_band is None
    command=worker_command(off,"ours","runs/x")
    assert command[command.index("--speed-guard-band")+1]=="off"
    assert parser().parse_args(["_worker","--speed-guard-band","off"]).speed_guard_band is None
    for bad in ("-1","nan","fast"):
        with pytest.raises(SystemExit):
            parser().parse_args(["run","--speed-guard-band",bad])
    with pytest.raises(SystemExit):
        parser().parse_args(["run","--command-execution","hold_latest"])
