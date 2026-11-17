"""Planning-round rules and the speed guard, independent of host execution speed and physics."""
from dataclasses import dataclass
import numpy as np


def plan_against_snapshot(robots, plan):
    """Call plan(i, robot) for each robot while every other robot exposes the
    states_prediction it had before the round, as truly concurrent planners
    would see. Each robot's fresh prediction is published after all return."""
    before = [None if r.states_prediction is None else r.states_prediction.copy() for r in robots]
    fresh = [r.states_prediction for r in robots]
    results = []
    try:
        for i, robot in enumerate(robots):
            results.append(plan(i, robot))
            fresh[i] = robot.states_prediction
            robot.states_prediction = before[i]
    finally:
        for robot, prediction in zip(robots, fresh):
            robot.states_prediction = prediction
    return results


# Speed guard (adapter safety layer, on by default for every controller). The
# planner keeps its own VMAX; the aircraft stops speeding up SPEED_GUARD_BAND
# below it, and above VMAX brakes by SPEED_GUARD_KD*(VMAX-speed), so the next
# solve starts inside its hard speed bound.
SPEED_GUARD_BAND = 0.5   # m/s
SPEED_GUARD_KD = 6.0     # 1/s


def speed_guard(u, velocity, vmax, band, kd):
    """Horizontal command u with its speed-increasing part removed once the
    horizontal speed is within band of vmax; above vmax that part becomes
    kd*(vmax-speed), a brake. Components across the velocity are kept."""
    u = np.asarray(u, float)
    velocity = np.asarray(velocity, float)[:2]
    speed = np.linalg.norm(velocity)
    if speed >= vmax-band and speed > 0:
        along = u@velocity/speed
        limit_along = kd*(vmax-speed) if speed > vmax else 0.
        if along > limit_along:
            u = u+(limit_along-along)*velocity/speed
    return u


@dataclass(frozen=True)
class SpeedGuard:
    """Speed guard for the held command: the planner keeps its own vmax, the
    aircraft stops speeding up at vmax-band (see speed_guard)."""
    vmax: float
    band: float
    kd: float

    def command(self, u, velocity):
        return speed_guard(u, velocity, self.vmax, self.band, self.kd)

