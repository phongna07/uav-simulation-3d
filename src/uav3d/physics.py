"""RotorPy implementation of the replaceable physics boundary."""
from typing import Protocol
import numpy as np


class PhysicsBackend(Protocol):
    commands: np.ndarray
    def positions(self) -> np.ndarray: ...
    def observations(self) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, duration: float) -> None: ...
    def snapshot(self) -> list[dict]: ...
    def restore(self, snapshot: list[dict]) -> None: ...
    def packed(self) -> np.ndarray: ...


class RotorPyBackend:
    def __init__(self, starts, physics_dt=0.005):
        from rotorpy.vehicles.multirotor import Multirotor
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
        self.vehicles = [Multirotor(self.params, initial_state=s, control_abstraction="cmd_acc",
                                   aero=False, enable_ground=False,
                                   integrator_kwargs={"rtol":1e-6, "atol":1e-9})
                         for s in self.states]

    def control(self, i):
        s = self.states[i]
        return {"cmd_acc": np.array([*self.commands[i,:2],
                                    9.81+4*(self.altitudes[i]-s["x"][2])-4*s["v"][2]])}

    def positions(self):
        return np.array([s["x"] for s in self.states])

    def observations(self):
        pv = np.array([np.r_[s["x"],s["v"]] for s in self.states])
        acc = np.array([v.statedot(s, self.control(i), self.dt)["vdot"]
                        for i,(s,v) in enumerate(zip(self.states,self.vehicles))])
        return pv, acc

    def step(self, duration):
        self.states = [v.step(s, self.control(i), duration)
                       for i,(v,s) in enumerate(zip(self.vehicles,self.states))]

    def snapshot(self):
        return [{k:np.array(v,copy=True) for k,v in s.items()} for s in self.states]

    def restore(self, snapshot):
        self.states = snapshot

    def packed(self):
        return np.array([np.r_[s["x"],s["v"],s["q"],s["w"],s["rotor_speeds"]]
                         for s in self.states])
