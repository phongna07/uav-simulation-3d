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


# Broad-phase slack in metres. The exact tests accept contacts within ~1e-12
# (squared) of the boundary; this margin is many orders of magnitude larger.
BROAD_PHASE_MARGIN = 1e-6


class CollisionWorld:
    def __init__(self, rects, circles, bounds, radius):
        self.rects = np.asarray(rects, float).reshape(-1, 4)
        self.circles = np.asarray(circles, float).reshape(-1, 3)
        self.bounds = np.asarray(bounds, float)
        self.radius = float(radius)

    def first_hit(self, old, new):
        """Earliest hit on the segments old -> new, identical to first_hit_exhaustive.

        A conservative broad phase (axis-aligned bounds with a margin far above
        the exact tests' tolerances) only discards obstacles, boundaries and drone
        pairs that cannot be hit. The rest go through the same exact tests in the
        same order, so the hit list, its ties and the result are unchanged.
        """
        old, new = np.asarray(old), np.asarray(new)
        if not np.isfinite(new).all():
            return Hit(0.0, "non_finite_state", ())
        n = len(old)
        if n == 0:
            return None
        R, m = self.radius, BROAD_PHASE_MARGIN
        # The filter uses float copies; the exact tests below see the inputs as given.
        old_f, new_f = old.astype(float), new.astype(float)
        lo = np.minimum(old_f[:, :2], new_f[:, :2]) - R - m
        hi = np.maximum(old_f[:, :2], new_f[:, :2]) + R + m
        x, y, w, h = self.rects.T
        rect_near = ((lo[:, :1] <= (x+w)[None]) & (hi[:, :1] >= x[None]) &
                     (lo[:, 1:2] <= (y+h)[None]) & (hi[:, 1:2] >= y[None]))
        cx, cy, cr = self.circles.T
        circle_near = ((lo[:, :1] <= (cx+cr)[None]) & (hi[:, :1] >= (cx-cr)[None]) &
                       (lo[:, 1:2] <= (cy+cr)[None]) & (hi[:, 1:2] >= (cy-cr)[None]))
        xmin, ymin, xmax, ymax = self.bounds
        inner = np.array([xmin+R, ymin+R]) + m
        outer = np.array([xmax-R, ymax-R]) - m
        both = np.stack([old_f, new_f])
        bound_near = ((both[..., :2] <= inner) | (both[..., :2] >= outer)).any(axis=(0, 2))
        bound_near |= (both[..., 2] <= R+m).any(axis=0)
        hits = []
        for i in np.flatnonzero(rect_near.any(axis=1) | circle_near.any(axis=1) | bound_near):
            p, q = old[i], new[i]
            d = q-p
            for rect in self.rects[rect_near[i]]:
                t = rectangle_entry(p[:2], d[:2], rect, R)
                if t is not None:
                    hits.append(Hit(t, "obstacle_rectangle", (int(i),)))
            for cx_, cy_, r in self.circles[circle_near[i]]:
                t = sphere_entry(p[:2]-[cx_,cy_], d[:2], r+R)
                if t is not None:
                    hits.append(Hit(t, "obstacle_circle", (int(i),)))
            if bound_near[i]:
                for axis, bound, sign, reason in ((0,xmin+R,1,"world_boundary"),
                                                  (0,xmax-R,-1,"world_boundary"),
                                                  (1,ymin+R,1,"world_boundary"),
                                                  (1,ymax-R,-1,"world_boundary"),
                                                  (2,R,1,"ground_contact")):
                    a, b = sign*(p[axis]-bound), sign*(q[axis]-bound)
                    if a <= 0:
                        hits.append(Hit(0.0, reason, (int(i),)))
                    elif b <= 0:
                        hits.append(Hit(float(a/(a-b)), reason, (int(i),)))
        if n > 1:
            # |rel(t)| >= |rel(0)| - t*|drel|: pairs farther apart than that can never meet.
            rel = old_f[:, None]-old_f[None]
            drel = (new_f-old_f)[:, None]-(new_f-old_f)[None]
            gap = np.linalg.norm(rel, axis=2)-np.linalg.norm(drel, axis=2)
            near = np.tril(gap <= 2*R+m, k=-1)
            for i, j in zip(*np.nonzero(near)):
                t = sphere_entry(old[i]-old[j], (new[i]-old[i])-(new[j]-old[j]), 2*R)
                if t is not None:
                    hits.append(Hit(t, "drone_collision", (int(j),int(i))))
        return min(hits, key=lambda hit: hit.fraction, default=None)

    def first_hit_exhaustive(self, old, new):
        """Reference: every exact test on every obstacle, boundary and drone pair."""
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
