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
