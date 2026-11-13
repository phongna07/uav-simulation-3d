"""Console progress for long CLI commands: timestamped plain lines on stderr.

Lines never use carriage returns, so the output reads the same in a terminal
and in a redirected log file (``... > run.log 2>&1``). Results stay on stdout.
Progress is only ever produced outside timed planner calls, so it is never
counted as planning delay.
"""
from contextlib import contextmanager
import math
import os
import sys
import time

DEFAULT_INTERVAL_S = 10.
ENV_INTERVAL = "UAV3D_PROGRESS_INTERVAL"


def default_interval():
    """Seconds between periodic progress lines, from the environment (0 = off)."""
    return os.environ.get(ENV_INTERVAL, str(DEFAULT_INTERVAL_S))


def parse_interval(value):
    """Validated progress interval in seconds; 0 turns periodic lines off."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise ValueError("--progress-interval must be a number of seconds") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("--progress-interval must be finite and nonnegative")
    return seconds


def log(tag, message):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {message}", file=sys.stderr, flush=True)


def duration(seconds):
    """Compact wall-clock duration: 4.2s, 3m05s, 1h02m."""
    if seconds is None or not math.isfinite(seconds):
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class Ticker:
    """Rate limiter for periodic progress lines on the wall clock."""

    def __init__(self, interval, clock=time.monotonic):
        self.interval, self.clock = float(interval), clock
        self.start = self.last = clock()

    def due(self):
        if self.interval <= 0:
            return False
        now = self.clock()
        if now-self.last < self.interval:
            return False
        self.last = now
        return True

    def elapsed(self):
        return self.clock()-self.start


def eta(done, total, elapsed):
    """Remaining wall time, extrapolated from the rate so far (None if unknown)."""
    if done <= 0 or elapsed <= 0 or total < done:
        return None
    return (total-done)*elapsed/done


@contextmanager
def stage(tag, message):
    """Log a stage when it starts and how long it took when it ends."""
    log(tag, f"{message} ...")
    start = time.perf_counter()
    yield
    log(tag, f"{message}: done in {duration(time.perf_counter()-start)}")
