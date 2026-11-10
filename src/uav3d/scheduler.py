"""Pure scheduling rules, independent of host execution speed and physics."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningEvent:
    round: int
    drone: int
    observed_at: float
    started_at: float
    completed_at: float
    execution_s: float
    injected_delay_s: float

    @property
    def queue_wait_s(self):
        return self.started_at-self.observed_at

    @property
    def latency_s(self):
        return self.completed_at-self.observed_at


def next_release(completed, released, period):
    """Next grid boundary; an exact completion boundary is available."""
    tick = max(round(released/period)+1, math.ceil(completed/period-1e-10))
    return tick*period, tick-round(released/period)-1


SCHEDULES = ("parallel", "sequential")


def parallel_events(durations, release=0., round_index=0):
    """One worker per UAV: all start at release, all deliver at release+max(C_i)."""
    durations = list(durations)
    for duration in durations:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Planning duration must be finite and nonnegative")
    end = release+max(durations, default=0.)
    for i, duration in enumerate(durations):
        yield PlanningEvent(round_index,i,release,release,end,duration,duration)


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


def sequential_events(durations, release=0., round_index=0):
    t = release
    for i, duration in enumerate(durations):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Planning duration must be finite and nonnegative")
        end = t+duration
        yield PlanningEvent(round_index,i,release,t,end,duration,duration)
        t = end
