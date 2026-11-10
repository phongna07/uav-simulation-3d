import numpy as np
from uav3d.physics import RotorPyBackend


def flight(dt):
    p=RotorPyBackend([[0,0,3]],dt)
    for _ in range(round(10/dt)):
        p.step(dt)
    assert abs(p.positions()[0,2]-3)<.05
    initial_quat=p.states[0]["q"].copy()
    p.commands[0,0]=5
    for _ in range(round(2/dt)):
        p.step(dt)
    assert p.states[0]["v"][0]>5
    assert np.linalg.norm(p.states[0]["q"]-initial_quat)>.05
    p.commands[0,0]=-5
    for _ in range(round(2/dt)):
        p.step(dt)
    p.commands[0,0]=0
    for _ in range(round(4/dt)):
        p.step(dt)
    assert abs(p.positions()[0,2]-3)<.1
    return p.packed()


def test_hover_acceleration_braking_and_step_convergence():
    coarse=flight(.005)
    fine=flight(.0025)
    assert np.max(np.abs(coarse[:,:3]-fine[:,:3]))<.1
    assert np.max(np.abs(coarse[:,3:6]-fine[:,3:6]))<.1
