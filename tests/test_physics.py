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


def test_parallel_stepping_is_bit_identical_to_serial():
    starts=[[0,0,3],[2,1,3.5],[-1,4,2.5],[3,-2,3],[5,5,3]]
    serial=RotorPyBackend(starts)
    parallel=RotorPyBackend(starts,workers=3)
    try:
        assert parallel.workers==3
        np.random.seed(11)
        rng_serial=[]
        for p in (serial,parallel):
            np.random.seed(11)
            for k in range(60):
                p.commands[:,:2]=np.column_stack([np.sin(k/7+np.arange(5)),np.cos(k/5-np.arange(5))])*3
                p.step(.005 if k%9 else .0013)       # include a split (partial) step
            rng_serial.append(np.random.random_sample(4))
        assert np.array_equal(serial.packed(),parallel.packed())
        for a,b in zip(serial.states,parallel.states):
            assert all(np.array_equal(a[key],b[key]) for key in a)
        # The global RNG stream (shared with upstream) is consumed identically.
        assert np.array_equal(*rng_serial)
        snap=parallel.snapshot()
        parallel.step(.005)
        parallel.restore(snap)
        assert np.array_equal(serial.packed(),parallel.packed())
    finally:
        parallel.close()


def test_physics_worker_count_resolution():
    from uav3d.physics import resolve_physics_workers
    assert resolve_physics_workers("auto",5,cpus=48)==5
    assert resolve_physics_workers("auto",5,cpus=2)==2
    assert resolve_physics_workers("8",5,cpus=48)==5
    assert resolve_physics_workers("1",5,cpus=48)==1
