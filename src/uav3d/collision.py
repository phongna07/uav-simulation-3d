"""Continuous tests on the line segments between physics samples."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Hit:
    fraction: float
    reason: str
    drones: tuple[int, ...]


def sphere_entry(p, d, radius):
    """First intersection of p + t*d with a closed sphere, t in [0,1]."""
    c = float(p @ p - radius * radius)
    if c <= 1e-12:
        return 0.0
    a, b = float(d @ d), float(2 * p @ d)
    disc = b * b - 4 * a * c
    if a < 1e-24 or disc < 0:
        return None
    t = (-b - np.sqrt(disc)) / (2 * a)
    return float(np.clip(t, 0, 1)) if -1e-12 <= t <= 1 + 1e-12 else None


def box_entry(p, d, lo, hi):
    enter, leave = 0.0, 1.0
    for k in range(len(p)):
        if abs(d[k]) < 1e-15:
            if p[k] < lo[k] or p[k] > hi[k]:
                return None
        else:
            a, b = sorted(((lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]))
            enter, leave = max(enter, a), min(leave, b)
            if enter > leave:
                return None
    return float(enter)


def rectangle_entry(p, d, rect, radius):
    x, y, w, h = rect
    # Rounded Minkowski expansion: two strips plus four corner disks.
    candidates = [box_entry(p, d, np.array([x-radius, y]), np.array([x+w+radius, y+h])),
                  box_entry(p, d, np.array([x, y-radius]), np.array([x+w, y+h+radius]))]
    candidates += [sphere_entry(p-np.array(c), d, radius)
                   for c in ((x,y), (x+w,y), (x,y+h), (x+w,y+h))]
    return min((t for t in candidates if t is not None), default=None)


class CollisionWorld:
    def __init__(self, rects, circles, bounds, radius):
        self.rects = np.asarray(rects, float).reshape(-1, 4)
        self.circles = np.asarray(circles, float).reshape(-1, 3)
        self.bounds = np.asarray(bounds, float)
        self.radius = float(radius)

    def first_hit(self, old, new):
        old, new = np.asarray(old), np.asarray(new)
        if not np.isfinite(new).all():
            return Hit(0.0, "non_finite_state", ())
        hits = []
        R = self.radius
        for i, (p, q) in enumerate(zip(old, new)):
            d = q-p
            for rect in self.rects:
                t = rectangle_entry(p[:2], d[:2], rect, R)
                if t is not None:
                    hits.append(Hit(t, "obstacle_rectangle", (i,)))
            for x, y, r in self.circles:
                t = sphere_entry(p[:2]-[x,y], d[:2], r+R)
                if t is not None:
                    hits.append(Hit(t, "obstacle_circle", (i,)))
            xmin, ymin, xmax, ymax = self.bounds
            for axis, bound, sign, reason in ((0,xmin+R,1,"world_boundary"),
                                              (0,xmax-R,-1,"world_boundary"),
                                              (1,ymin+R,1,"world_boundary"),
                                              (1,ymax-R,-1,"world_boundary"),
                                              (2,R,1,"ground_contact")):
                a, b = sign*(p[axis]-bound), sign*(q[axis]-bound)
                if a <= 0:
                    hits.append(Hit(0.0, reason, (i,)))
                elif b <= 0:
                    hits.append(Hit(float(a/(a-b)), reason, (i,)))
        for i in range(len(old)):
            for j in range(i):
                t = sphere_entry(old[i]-old[j], (new[i]-old[i])-(new[j]-old[j]), 2*R)
                if t is not None:
                    hits.append(Hit(t, "drone_collision", (j,i)))
        return min(hits, key=lambda hit: hit.fraction, default=None)

    def valid_start(self, p, others):
        points = np.array([*others, p], float).reshape(-1, 3)
        return self.first_hit(points, points) is None
