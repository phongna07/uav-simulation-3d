"""Pure scheduling rules, independent of host execution speed and physics."""
import math
from dataclasses import dataclass
import numpy as np


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



@dataclass(frozen=True)
class PlanSchedule:
    """A delivered plan, tracked by mission time.

    states[k] = [px, py, vx, vy] is planned at start+k*period and controls[k]
    is applied on [start+k*period, start+(k+1)*period). The aircraft tracks the
    plan, u = u_ff + kp*(p_ref-p) + kd*(v_ref-v), so model mismatch does not
    accumulate however long the next plan takes. The planned u_ff already lies
    within the planner's limit; the correction may add up to track_umax in
    total, so a plan at its limit can still be tracked. Within vmax_band of vmax
    the component of u that would speed the aircraft up is removed (the attitude
    response lags), and above vmax it is replaced by kd*(vmax-speed), so the next
    MPC solve starts inside its hard speed bound.

    Watchdog: from stop_at the reference slows to rest along the plan's own
    (corridor-verified) remaining path, reaching its last point exactly as the
    progress rate reaches zero, then holds that point. By default stop_at is the
    last planned step from which that stop needs at most about a_stop.
    """
    start: float
    period: float
    states: np.ndarray
    controls: np.ndarray
    umax: float
    kp: float
    kd: float
    a_stop: float
    vmax: float = math.inf
    track_umax: float | None = None
    vmax_band: float = 0.
    stop_at: float | None = None

    def __post_init__(self):
        states = np.asarray(self.states, float)
        controls = np.asarray(self.controls, float)
        if states.shape != (len(controls)+1, 4) or controls.shape[1:] != (2,) or not len(controls):
            raise ValueError("Plan schedule needs K controls and K+1 [px,py,vx,vy] states")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)
        if self.stop_at is None:
            # Stopping linearly in progress rate over the remaining plan time R
            # decelerates by ~|v|/(2R); keep that within a_stop.
            stop = self.start
            for j, state in enumerate(states):
                remaining = (len(controls)-j)*self.period
                if np.linalg.norm(state[2:]) > 2*self.a_stop*remaining:
                    break
                stop = self.start+j*self.period
            object.__setattr__(self, "stop_at", stop)

    @property
    def end(self):
        return self.start+len(self.controls)*self.period

    def index(self, t):
        return math.floor((t-self.start)/self.period+1e-9)

    def stopped(self, t):
        """The same plan with its watchdog stop starting no later than t."""
        return PlanSchedule(self.start, self.period, self.states, self.controls, self.umax,
                            self.kp, self.kd, self.a_stop, self.vmax, self.track_umax,
                            self.vmax_band, min(self.stop_at, max(t, self.start)))

    def _planned(self, tau):
        """Planned position/velocity (cubic Hermite through the plan states) and control."""
        k = min(max(self.index(tau), 0), len(self.controls)-1)
        h = self.period
        s = min(max((tau-self.start-k*h)/h, 0.), 1.)
        p0, v0 = self.states[k, :2], self.states[k, 2:]
        p1, v1 = self.states[k+1, :2], self.states[k+1, 2:]
        p = ((2*s**3-3*s**2+1)*p0+(s**3-2*s**2+s)*h*v0
             +(-2*s**3+3*s**2)*p1+(s**3-s**2)*h*v1)
        v = ((6*s**2-6*s)*p0/h+(3*s**2-4*s+1)*v0
             +(-6*s**2+6*s)*p1/h+(3*s**2-2*s)*v1)
        return p, v, self.controls[k]

    def reference(self, t):
        """(position, velocity, feedforward acceleration) the aircraft should have at t."""
        if t < self.start-1e-9:
            raise ValueError("Plan schedule queried before its start")
        hold = self.states[-1, :2], np.zeros(2), np.zeros(2)
        if t <= self.stop_at:
            if t >= self.end:
                return hold
            return self._planned(t)
        brake = 2*(self.end-self.stop_at)
        elapsed = t-self.stop_at
        if brake <= 0 or elapsed >= brake:
            return hold
        rate = 1-elapsed/brake                     # d(tau)/dt, reaches 0 at the plan end
        p, v, u = self._planned(self.stop_at+elapsed-elapsed**2/(2*brake))
        return p, v*rate, u*rate**2-v/brake

    def command(self, t, position, velocity):
        p, v, u = self.reference(t)
        velocity = np.asarray(velocity, float)[:2]
        n = np.linalg.norm(u)
        if n > self.umax:
            u = u*(self.umax/n)
        limit = self.umax if self.track_umax is None else max(self.track_umax, self.umax)
        # Clip the correction alone: it never scales down the planned braking.
        feedback = self.kp*(p-np.asarray(position, float)[:2])+self.kd*(v-velocity)
        room = _room(u, feedback, limit)
        u = u+room*feedback
        speed = np.linalg.norm(velocity)
        if speed >= self.vmax-self.vmax_band and speed > 0:
            along = u@velocity/speed
            limit_along = self.kd*(self.vmax-speed) if speed > self.vmax else 0.
            if along > limit_along:
                u = u+(limit_along-along)*velocity/speed
        return u

    def prediction(self, t0, steps):
        """Reference [px, py, vx, vy] at t0, t0+period, ..., as published to other UAVs."""
        rows = [self.reference(max(t0+k*self.period, self.start))[:2] for k in range(steps)]
        return np.array([np.r_[p, v] for p, v in rows])


def _room(base, step, limit):
    """Largest s in [0, 1] with |base + s*step| <= limit, given |base| <= limit."""
    a, b, c = step@step, 2*base@step, base@base-limit**2
    if a == 0 or a+b+c <= 0:
        return 1.
    return max(0., (-b+math.sqrt(max(b*b-4*a*c, 0.)))/(2*a))
