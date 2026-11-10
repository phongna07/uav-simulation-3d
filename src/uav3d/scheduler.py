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


def sequential_events(durations, release=0., round_index=0):
    t = release
    for i, duration in enumerate(durations):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Planning duration must be finite and nonnegative")
        end = t+duration
        yield PlanningEvent(round_index,i,release,t,end,duration,duration)
        t = end
