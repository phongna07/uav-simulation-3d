"""RotorPy implementation of the replaceable physics boundary."""
import multiprocessing
import os
from typing import Protocol
import numpy as np


class PhysicsBackend(Protocol):
    commands: np.ndarray
    def positions(self) -> np.ndarray: ...
    def velocities(self) -> np.ndarray: ...
    def observations(self) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, duration: float) -> None: ...
    def snapshot(self) -> list[dict]: ...
    def restore(self, snapshot: list[dict]) -> None: ...
    def packed(self) -> np.ndarray: ...


def _make_vehicle(params, state):
    from rotorpy.vehicles.multirotor import Multirotor
    return Multirotor(params, initial_state=state, control_abstraction="cmd_acc",
                      aero=False, enable_ground=False,
                      integrator_kwargs={"rtol":1e-6, "atol":1e-9})


def _step_worker(connection, params, state):
    """Step vehicles on request. Multirotor.step is a pure function of
    (state, control, duration), so one vehicle object serves every aircraft."""
    vehicle = _make_vehicle(params, state)
    connection.send("ready")
    while True:
        request = connection.recv()
        if request is None:
            return
        jobs, duration = request
        connection.send([vehicle.step(s, c, duration) for s, c in jobs])


def resolve_physics_workers(value, drones, cpus=None):
    """Worker processes for `value` ("auto" or N >= 1), never more than one per drone."""
    if cpus is None:
        cpus = len(os.sched_getaffinity(0))
    if value == "auto":
        count = cpus
    else:
        count = int(value)
        if count < 1:
            raise ValueError("--physics-workers must be 'auto' or a positive integer")
    return max(1, min(count, drones))


class _StepPool:
    """Vehicle steps split across processes; chunk 0 is stepped in the caller.

    Workers are stateless (the caller owns every state and sends it each step),
    so snapshot/restore and collision splitting behave as in serial stepping.
    """

    def __init__(self, params, states, chunks):
        self.chunks = chunks
        context = multiprocessing.get_context("spawn")
        self.connections, self.processes = [], []
        for chunk in chunks[1:]:
            parent, child = context.Pipe()
            process = context.Process(target=_step_worker, daemon=True,
                                      args=(child, params, states[chunk[0]]))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        for connection in self.connections:
            if connection.recv() != "ready":
                raise RuntimeError("Physics worker failed to start")

    def step(self, vehicle, states, controls, duration):
        for connection, chunk in zip(self.connections, self.chunks[1:]):
            connection.send(([(states[i], controls[i]) for i in chunk], duration))
        new = list(states)
        for i in self.chunks[0]:
            new[i] = vehicle.step(states[i], controls[i], duration)
        for connection, chunk in zip(self.connections, self.chunks[1:]):
            for i, state in zip(chunk, connection.recv()):
                new[i] = state
        return new

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
                connection.close()
            except OSError:
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.connections, self.processes = [], []


class RotorPyBackend:
    def __init__(self, starts, physics_dt=0.005, workers=1):
        from rotorpy.vehicles.hummingbird_params import quad_params
        self.params = dict(quad_params)
        self.params["motor_noise_std"] = 0.0
        self.dt = physics_dt
        self.altitudes = np.asarray(starts, float)[:,2].copy()
        self.commands = np.zeros((len(starts), 3))
        hover = np.sqrt(self.params["mass"]*9.81/(4*self.params["k_eta"]))
        self.states = [dict(x=np.array(p, float), v=np.zeros(3), q=np.array([0.,0.,0.,1.]),
                            w=np.zeros(3), wind=np.zeros(3), rotor_speeds=np.full(4, hover))
                       for p in starts]
        self.vehicles = [_make_vehicle(self.params, s) for s in self.states]
        self.workers = max(1, min(int(workers), len(self.states)))
        self.pool = None
        if self.workers > 1:
            # Zero motor noise makes every vehicle's step deterministic, so a worker's
            # own RNG cannot change the result (see step() for the global RNG stream).
            if any(v.motor_noise != 0 for v in self.vehicles):
                raise ValueError("Parallel physics requires zero motor noise")
            chunks = [list(c) for c in np.array_split(np.arange(len(self.states)), self.workers)]
            self.pool = _StepPool(self.params, self.states, chunks)

    def control(self, i):
        s = self.states[i]
        return {"cmd_acc": np.array([*self.commands[i,:2],
                                    9.81+4*(self.altitudes[i]-s["x"][2])-4*s["v"][2]])}

    def positions(self):
        return np.array([s["x"] for s in self.states])

    def velocities(self):
        return np.array([s["v"] for s in self.states])

    def observations(self):
        pv = np.array([np.r_[s["x"],s["v"]] for s in self.states])
        acc = np.array([v.statedot(s, self.control(i), self.dt)["vdot"]
                        for i,(s,v) in enumerate(zip(self.states,self.vehicles))])
        return pv, acc

    def step(self, duration):
        if self.pool is None:
            self.states = [v.step(s, self.control(i), duration)
                           for i,(v,s) in enumerate(zip(self.vehicles,self.states))]
            return
        controls = [self.control(i) for i in range(len(self.states))]
        self.states = self.pool.step(self.vehicles[0], self.states, controls, duration)
        # Multirotor.step draws its (zero-scale) motor noise from the global NumPy
        # RNG, which upstream also uses. Workers draw from their own stream, so the
        # same draws are consumed here, in vehicle order, to keep the main-process
        # RNG state identical to serial stepping.
        for v in self.vehicles[self.pool.chunks[0][-1]+1:]:
            np.random.normal(scale=np.abs(v.motor_noise), size=(v.num_rotors,))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def snapshot(self):
        return [{k:np.array(v,copy=True) for k,v in s.items()} for s in self.states]

    def restore(self, snapshot):
        self.states = snapshot

    def packed(self):
        return np.array([np.r_[s["x"],s["v"],s["q"],s["w"],s["rotor_speeds"]]
                         for s in self.states])
