"""Offline pose rendering using only experiment artifacts."""
import json
from pathlib import Path
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from . import progress


class TargetFollowAxes(Axes3D):
    """Fill a rectangular viewport and pan without changing world geometry."""

    camera_target = None
    camera_zoom = 1.

    def apply_aspect(self, position=None):
        # Axes3D normally shrinks the viewport to a square, leaving side bars.
        # Use the entire allocated rectangle; get_proj compensates its aspect.
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, which="active")

    def get_proj(self):
        projection = super().get_proj()
        if self.camera_target is not None:
            target = projection @ np.r_[self.camera_target, 1.]
            # Translate the camera in its screen plane, preserving heading.
            # Project the fixed world axes together with the scenery.
            for axis in (0, 1):
                projection[axis] = self.camera_zoom * (
                    projection[axis] - target[axis]/target[3]*projection[3])
        # Keep aircraft, trees and paving proportional in a wide viewport.
        projection[0] *= self.bbox.height / self.bbox.width
        return projection


def follow_camera(ax, target, bounds, max_z, zoom, lift=0.):
    """Move the camera to the target, keeping world bounds and heading fixed.

    ``lift`` aims the camera at the point ``lift`` metres above the target
    instead of the target itself, so aircraft flying at that altitude over the
    target sit at the frame center and the target appears below it. Zoom and
    heading are unchanged; only the screen-plane pan moves.

    Returns the vertical display scale and the nominal horizontal view window.
    Scenery stays in world coordinates and must not be rebuilt from this window.
    """
    xmin, ymin, xmax, ymax = bounds
    full_span = max(xmax-xmin, ymax-ymin)
    span = full_span/zoom
    display_z = max(max_z, .11*full_span)
    scale = display_z/max_z
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(0, max_z))
    ax.set_box_aspect((xmax-xmin, ymax-ymin, display_z))
    ax.set_zticks(np.linspace(0, max_z, 3))
    ax.camera_target = np.asarray(target, dtype=float).copy()
    ax.camera_target[2] += lift
    ax.camera_zoom = zoom
    x, y = target[:2]
    return scale, (x-span/2, y-span/2, x+span/2, y+span/2)


# Layering with ``computed_zorder=False``: paving (0) < depth-sorted scenery
# (1) < raised obstacle groups and the hidden overlays between them (OCCLUSION
# band) < visible overlays (>= 2). An overlay part whose view ray is blocked by
# obstacle volumes is drawn just beneath the farthest of those obstacles: they
# (and any obstacle in front of them) cover it where their mesh is solid, it
# still shows through gaps (e.g. between tree crown lobes), and scenery farther
# from the camera than its occluders never paints over it.
OCCLUSION = (1.2, 1.9)
HIDDEN_LAYER = 1.1  # hidden overlays before a LayerPlan places them
# Hidden parts of different overlays can share a layer zorder; matplotlib then
# draws them in artist creation order, and hidden-part artists are created on
# demand, so that order would depend on which frames were drawn before. A tiny
# per-overlay offset (far below the smallest gap between distinct layer zorders)
# draws such ties in overlay construction order instead, so every frame is a
# function of its mission time alone, as parallel MP4 export requires.
TIE_BREAK = 1e-10
# Replay trail fading: opacity time constant (mission s) and lower bound.
TRAIL_FADE_S = 60.
TRAIL_MIN_ALPHA = .2


def overlay_rank(ax):
    """Construction index of the next depth-split overlay on ``ax``."""
    rank = getattr(ax, "_uav3d_overlay_count", 0)
    ax._uav3d_overlay_count = rank + 1
    return rank


def view_directions(projection, points):
    """Unit world-space directions from each point toward the camera."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points))]
    screen = homogeneous @ projection.T
    screen = screen/screen[:, 3:]
    # Smaller projected depth is nearer the camera (ortho and perspective).
    screen[:, 2] -= 1e-3
    nearer = screen @ np.linalg.inv(projection).T
    direction = nearer[:, :3]/nearer[:, 3:] - points
    return direction/np.linalg.norm(direction, axis=1, keepdims=True)


def occluders(projection, points, boxes):
    """(points × boxes) mask: the ray from a point to the camera crosses a box."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0 or len(boxes) == 0:
        return np.zeros((len(points), len(boxes)), dtype=bool)
    direction = view_directions(projection, points)
    direction[np.abs(direction) < 1e-12] = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        lower = (boxes[None, :, 0] - points[:, None])/direction[:, None]
        upper = (boxes[None, :, 1] - points[:, None])/direction[:, None]
    near = np.minimum(lower, upper).max(axis=2)
    far = np.maximum(lower, upper).min(axis=2)
    return far >= np.maximum(near, 0.)


def occluded(projection, points, boxes):
    """True where the ray from a point to the camera crosses an obstacle box."""
    return occluders(projection, points, boxes).any(axis=1)


def box_depths(projection, boxes):
    """Projected box corners: screen bounds (low, high) and mean depth per box."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
    corners = np.array([[[box[i][0], box[j][1], box[k][2]]
                         for i in (0, 1) for j in (0, 1) for k in (0, 1)]
                        for box in boxes]).reshape(-1, 3)
    screen = np.c_[corners, np.ones(len(corners))] @ projection.T
    screen = (screen[:, :3]/screen[:, 3:]).reshape(len(boxes), 8, 3)
    # Smaller depth is nearer the camera.
    return screen[:, :, :2].min(axis=1), screen[:, :, :2].max(axis=1), screen[:, :, 2].mean(axis=1)


def front_obstacles(projection, boxes, seeds):
    """Occluding boxes plus every box drawn over them (nearer and overlapping).

    Raising only ``seeds`` above the scenery would let an occluder paint over a
    nearer obstacle in front of it, so those are raised with it.
    """
    front = set(int(i) for i in seeds)
    if not front:
        return front
    low, high, depth = box_depths(projection, boxes)
    pending = list(front)
    while pending:
        i = pending.pop()
        overlap = (low <= high[i]).all(axis=1) & (high >= low[i]).all(axis=1)
        for j in np.flatnonzero(overlap & (depth < depth[i])):
            if j not in front:
                front.add(int(j))
                pending.append(int(j))
    return front


class LayerPlan:
    """Interleave raised obstacles and hidden overlays for one frame.

    ``keys`` are the occluder index sets of every hidden overlay part. Raised
    obstacles are ranked far to near; each hidden part sits just beneath its
    farthest occluder, so nearer raised obstacles cover it and farther ones
    (raised for some other part) stay beneath it.
    """

    def __init__(self, projection, boxes, keys):
        keys = {frozenset(int(j) for j in key) for key in keys if key}
        front = front_obstacles(projection, boxes, set().union(*keys) if keys else ())
        depth = box_depths(projection, boxes)[2] if front else None
        order = sorted(front, key=lambda j: -depth[j])
        self.rank = {j: r for r, j in enumerate(order)}
        cuts = sorted({min(self.rank[j] for j in key) for key in keys})
        low, high = OCCLUSION
        self.step = (high-low)/(len(cuts)+1)
        self.base = low
        self.group_of_cut = {cut: g+1 for g, cut in enumerate(cuts)}
        groups = {}
        for j, r in self.rank.items():
            # Group 0 (farther than every hidden part) stays in the scenery.
            groups.setdefault(int(np.searchsorted(cuts, r, side="right")), []).append(j)
        self.obstacle_layers = [(low + g*self.step, sorted(indices))
                                for g, indices in sorted(groups.items()) if g > 0]

    def layer(self, key):
        """(offset, scale) placing an overlay hidden by ``key`` boxes."""
        g = self.group_of_cut[min(self.rank[int(j)] for j in key)]
        # Overlay-internal zorders (< 10) stay inside the gap below group g.
        return self.base + (g-1)*self.step + self.step/4, self.step/40


class DepthSplitLine:
    """A 3D polyline drawn above the scenery only where nothing occludes it."""

    def __init__(self, ax, zorder=2., segment=None, **style):
        self.segment = segment  # optional maximum segment length for testing
        self.ax, self.zorder = ax, zorder
        self.rank = overlay_rank(ax)
        label = style.pop("label", None)
        self.style = style
        self.front = self._new_artist(zorder)
        if label is not None:
            self.front.set_label(label)
        # One artist per layer used by hidden parts; more are added on demand.
        self.backs = [self._new_artist(HIDDEN_LAYER+zorder/100)]
        self.pieces = []  # (occluder key, points) of hidden parts, in order

    def _new_artist(self, zorder):
        return self.ax.plot([], [], [], zorder=zorder, **self.style)[0]

    def _show(self, artist, chunks):
        """Draw ``chunks`` (point arrays) as separate paths of ``artist``."""
        parts = []
        for chunk in chunks:
            # A NaN row breaks the path between chunks.
            parts += [chunk, np.full((1, chunk.shape[1]), np.nan)]
        p = np.concatenate(parts) if parts else np.empty((0, 3))
        artist.set_data_3d(p[:, 0], p[:, 1], p[:, 2])

    @property
    def back(self):
        """The first hidden-part artist (all hidden parts before ``place``)."""
        return self.backs[0]

    def _subdivide(self, points):
        if self.segment is None or len(points) < 2:
            return points
        pieces = [points[:1]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm((end-start)[:3])/self.segment)))
            pieces.append(start + np.outer(np.arange(1, count+1)/count, end-start))
        return np.concatenate(pieces)

    def update(self, points, projection, boxes):
        """Split the line by occlusion; return the occluder sets of hidden parts.

        Hidden parts are drawn at HIDDEN_LAYER until ``place`` assigns layers.
        Columns after the first three (x, y, z) are carried along with each
        point (e.g. a fade weight) and ignored for occlusion.
        """
        points = np.asarray(points, dtype=float)
        points = self._subdivide(points.reshape(-1, points.shape[-1] if points.ndim > 1 else 3))
        single = len(points) < 2
        xyz = points[:, :3]
        blocked = occluders(projection, xyz if single else (xyz[:-1]+xyz[1:])/2, boxes)
        keys = [frozenset(np.flatnonzero(row).tolist()) for row in blocked]
        # Consecutive segments with one occluder set stay one path (so dash
        # patterns continue); the path breaks where it changes.
        visible, self.pieces = [], []
        start = 0
        for stop in range(1, len(keys)+1):
            if stop < len(keys) and keys[stop] == keys[start]:
                continue
            chunk = points[start:stop] if single else points[start:stop+1]
            if keys[start]:
                self.pieces.append((keys[start], chunk))
            else:
                visible.append(chunk)
            start = stop
        self._show(self.front, visible)
        self.place(None)
        return {key for key, _ in self.pieces}

    def place(self, plan):
        """Draw hidden parts at the layers ``plan`` (a LayerPlan) assigns."""
        layers = {}
        for key, chunk in self.pieces:
            offset, scale = plan.layer(key) if plan is not None else (HIDDEN_LAYER, .01)
            layers.setdefault(offset + scale*self.zorder, []).append(chunk)
        while len(self.backs) < len(layers):
            self.backs.append(self._new_artist(HIDDEN_LAYER))
        for artist, (zorder, chunks) in zip(self.backs, sorted(layers.items())):
            artist.set_zorder(zorder + self.rank*TIE_BREAK)
            self._show(artist, chunks)
        for artist in self.backs[len(layers):]:
            self._show(artist, [])


def trail_fade(ages, window, floor):
    """Trail opacity factor per point: 1 at the aircraft, easing to ``floor``.

    ``ages`` are mission seconds since each point was recorded. The factor
    decays exponentially with time constant ``window`` and never drops below
    ``floor``, so old trail stays faintly visible. ``window <= 0`` disables it.
    """
    ages = np.asarray(ages, dtype=float)
    if window <= 0:
        return np.ones_like(ages)
    return floor + (1-floor)*np.exp(-np.maximum(ages, 0.)/window)


class FadingTrail(DepthSplitLine):
    """A depth-split polyline whose opacity varies along it.

    ``update`` takes (x, y, z, weight) rows; each segment is drawn with the
    base colour's alpha scaled by the mean weight of its two end points.
    """

    def __init__(self, ax, color, zorder=2., **style):
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
        self.rgba = np.array(to_rgba(color))
        label = style.pop("label", None)
        super().__init__(ax, zorder=zorder, **style)
        # Collections draw per-segment colours; the legend uses a plain line.
        self.handle = Line2D([], [], color=color, label=label,
                             lw=style.get("lw", style.get("linewidth")))

    def _new_artist(self, zorder):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        artist = Line3DCollection([], zorder=zorder, capstyle="butt", **self.style)
        self.ax.add_collection3d(artist, autolim=False)  # world limits stay fixed
        return artist

    def _show(self, artist, chunks):
        chunks = [c for c in chunks if len(c) > 1]
        if not chunks:
            artist.set_segments([])
            return
        segments = np.concatenate([np.stack([c[:-1, :3], c[1:, :3]], axis=1) for c in chunks])
        weights = np.concatenate([(c[:-1, 3] + c[1:, 3])/2 for c in chunks])
        colors = np.tile(self.rgba, (len(segments), 1))
        colors[:, 3] *= weights
        artist.set_segments(segments)
        artist.set_color(colors)


class DepthSplitPolys:
    """3D polygons drawn above the scenery only where nothing occludes them.

    Each polygon is tested at its centroid, so callers pass small pieces (e.g.
    subdivided faces) to let an obstacle edge sweep across a surface gradually.
    """

    def __init__(self, ax, zorder=2., **style):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.ax, self.zorder, self.style = ax, zorder, style
        self.rank = overlay_rank(ax)
        self._make = lambda z: ax.add_collection3d(Poly3DCollection([], zorder=z, **style))
        self.front = self._make(zorder)
        self.backs = []
        self.pieces = []  # (occluder key, polygon indices) of hidden groups
        self.polygons, self.colors = np.empty((0, 3, 3)), np.empty((0, 4))

    @staticmethod
    def _fill(artist, polygons, colors):
        artist.set_verts(list(polygons))
        artist.set_facecolor(colors)
        artist.set_edgecolor(colors)  # same-colour strokes seal piece seams

    def update(self, polygons, colors, projection, boxes):
        """Split polygons by occlusion; return the occluder sets of hidden ones."""
        from matplotlib.colors import to_rgba_array
        self.polygons = np.asarray(polygons, dtype=float).reshape(-1, *np.shape(polygons)[-2:])
        self.colors = to_rgba_array(colors)
        blocked = occluders(projection, self.polygons.mean(axis=1), boxes)
        groups = {}
        for i, row in enumerate(blocked):
            groups.setdefault(frozenset(np.flatnonzero(row).tolist()), []).append(i)
        visible = groups.pop(frozenset(), [])
        self._fill(self.front, self.polygons[visible], self.colors[visible])
        self.pieces = list(groups.items())
        self.place(None)
        return set(groups)

    def place(self, plan):
        """Draw hidden polygons at the layers ``plan`` (a LayerPlan) assigns."""
        layers = {}
        for key, indices in self.pieces:
            offset, scale = plan.layer(key) if plan is not None else (HIDDEN_LAYER, .01)
            layers.setdefault(offset + scale*self.zorder, []).extend(indices)
        while len(self.backs) < len(layers):
            self.backs.append(self._make(HIDDEN_LAYER))
        for artist, (zorder, indices) in zip(self.backs, sorted(layers.items())):
            artist.set_zorder(zorder + self.rank*TIE_BREAK)
            self._fill(artist, self.polygons[indices], self.colors[indices])
        for artist in self.backs[len(layers):]:
            self._fill(artist, np.empty((0, 3, 3)), np.empty((0, 4)))


FIGURE_SPACE = "\u2007"  # as wide as a digit in proportional fonts such as DejaVu Sans


def fixed_width(value, width, decimals):
    """Right-align a number with digit-wide padding so centered text keeps its width."""
    return f"{value:{width}.{decimals}f}".replace(" ", FIGURE_SPACE)


def round_solve_times(events, ratio):
    """(release times, d) per planning round, with d = max_i C_i in host seconds.

    The raw host duration is recovered from the injected delay (delay = C*r), so
    d does not depend on --compute-ratio; with r = 0 the measured duration is used."""
    rounds = {}
    for event in events:
        raw = event["injected_delay_s"]/ratio if ratio > 0 else event["execution_s"]
        release, d = rounds.get(event["round"], (event["observed_at"], 0.))
        rounds[event["round"]] = (min(release, event["observed_at"]), max(d, raw))
    ordered = sorted(rounds.values())
    return np.array([r for r, _ in ordered]), np.array([d for _, d in ordered])


def bold(text):
    """Bold a value inside an otherwise regular title line (mathtext, DejaVu Sans Bold)."""
    return "$\\mathbf{" + str(text).replace("_", "\\_").replace(" ", "\\ ") + "}$"


def solve_time_label(releases, solves, t):
    """Max solve time of the latest round released at or before mission time t."""
    index = np.searchsorted(releases, t+1e-12, side="right")-1
    value = fixed_width(solves[index]*1e3, 6, 1) if index >= 0 else FIGURE_SPACE*3+"n/a"
    return f"Max solve time d = {bold(value)} ms"


def min_separations(states):
    """Smallest 3D centre distance between any two UAVs, per recorded sample."""
    positions = np.asarray(states, dtype=float)[..., :3]
    if positions.shape[1] < 2:
        return np.full(len(positions), np.nan)
    gaps = positions[:, :, None] - positions[:, None, :]
    distance = np.linalg.norm(gaps, axis=-1)
    distance[:, np.arange(positions.shape[1]), np.arange(positions.shape[1])] = np.inf
    return distance.min(axis=(1, 2))


def min_obstacle_distances(states, rects, circles):
    """Smallest UAV-obstacle distance per recorded sample.

    Matches the collision model: obstacles are vertical columns, so this is the
    horizontal distance from each UAV centre to the nearest rectangle or circle
    footprint edge (0 inside one), minimised over UAVs. NaN without obstacles.
    """
    xy = np.asarray(states, dtype=float)[..., :2]
    distances = []
    for x, y, w, h in np.asarray(rects, dtype=float).reshape(-1, 4):
        dx = np.maximum(np.maximum(x-xy[..., 0], xy[..., 0]-(x+w)), 0.)
        dy = np.maximum(np.maximum(y-xy[..., 1], xy[..., 1]-(y+h)), 0.)
        distances.append(np.hypot(dx, dy))
    for x, y, r in np.asarray(circles, dtype=float).reshape(-1, 3):
        distances.append(np.maximum(np.hypot(xy[..., 0]-x, xy[..., 1]-y)-r, 0.))
    if not distances:
        return np.full(len(xy), np.nan)
    return np.min(distances, axis=0).min(axis=1)


def map_size_label(bounds):
    """World extent as 'width x height m', e.g. 1000x1000 m."""
    xmin, ymin, xmax, ymax = bounds
    return f"{xmax-xmin:g}x{ymax-ymin:g} m"


def distance_label(value):
    """Fixed-width distance, or n/a when undefined."""
    return FIGURE_SPACE*3+"n/a" if not np.isfinite(value) else fixed_width(value, 6, 2)


def speed_label(speed):
    """Video speed as mission seconds per video second, e.g. 5x or 0.5x."""
    return f"{speed:g}x"


def title_text(scenario, map_size, controller, n, speed, t, solve_label, separation,
               clearance):
    """Two-line replay header; values are bold and numbers keep a fixed width."""
    return (f"Scenario: {bold(scenario)} · Map size: {bold(map_size)} · "
            f"Algorithm: {bold(controller)} · "
            f"{bold(n)} UAVs · Video speed: {bold(speed_label(speed))} · "
            f"t = {bold(fixed_width(t, 7, 3))} s\n"
            f"{solve_label} · Minimum UAV Separation Distance: {bold(distance_label(separation))} m · "
            f"Minimum UAV - Obstacles Distance: {bold(distance_label(clearance))} m")


def video_frame_times(times, speed, fps):
    """Mission times of the MP4 frames; shared by serial and parallel export."""
    return np.r_[np.arange(0, times[-1], speed/fps), times[-1]]


def build_scene(args):
    """Build the replay figure; return (figure, update(t), recorded times).

    Each frame is a function of its mission time only, so worker processes can
    each build their own scene and draw any subset of frames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .scene import draw_city, occluder_bounds
    from .aircraft import QuadcopterGlyph
    directory = Path(args.directory)
    cfg = json.loads((directory/"config.json").read_text())
    with np.load(directory/"trajectory.npz") as archive:
        times,states,targets = (archive[k].copy() for k in ("time","aircraft","target"))
    events_path = directory/"events.jsonl"
    events = ([json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
              if events_path.exists() else [])
    releases, solves = round_solve_times(events, float(cfg.get("compute_ratio", 1.)))
    separations = min_separations(states)
    clearances = min_obstacle_distances(states, cfg["scenario_definition"]["rects"],
                                        cfg["scenario_definition"]["circles"])
    map_size = map_size_label(cfg["params"]["WORLD_BOUNDS"])
    speed = float(getattr(args, "speed", 1.))
    fig = plt.figure(figsize=(12,8),facecolor="#f3f5f9")
    # Explicit layers: ground, depth-sorted obstacles, then aircraft overlays.
    # Automatic collection ordering can put the ground above whole buildings.
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes,
                      facecolor="#f3f5f9", computed_zorder=False)
    xmin,ymin,xmax,ymax = cfg["params"]["WORLD_BOUNDS"]
    max_z = max(10.,float(states[:,:,2].max())*1.7)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(0,max_z),
           xlabel="East [m]",ylabel="North [m]",zlabel="Altitude [m]")
    display_z = max(max_z,.22*max(xmax-xmin,ymax-ymin))
    ax.set_box_aspect((xmax-xmin,ymax-ymin,display_z))
    ax.set_zticks(np.linspace(0,max_z,3))
    ax.view_init(elev=35,azim=-65)
    following = getattr(args, "camera", "follow") == "follow"
    zoom = getattr(args, "zoom", 6.)
    # Trails fade with age (mission seconds) but never vanish.
    fade_window = float(getattr(args, "trail_fade", TRAIL_FADE_S))
    fade_floor = float(getattr(args, "trail_min_alpha", TRAIL_MIN_ALPHA))
    # Center the swarm rather than the ground target: aim at the target lifted
    # to the recorded flying altitude. The aircraft hold altitude, so a single
    # robust value keeps the pan smooth (no per-frame vertical jitter).
    flight_altitude = float(np.clip(np.median(states[:,:,2]), 0., max_z))
    if following:
        ax.view_init(elev=55, azim=-65)
        ax.set_proj_type("ortho")
    ax.set_axis_off()
    # Figure overlays do not reserve space or shrink the scenery viewport.
    title_top = .988
    title = fig.text(.5, title_top, title_text(cfg["scenario"], map_size, cfg["controller"],
                                               states.shape[1], speed, 0., solve_time_label(
                                                   releases, solves, 0.), 0., 0.),
                     ha="center", va="top", fontsize=11, color="#111111", zorder=11)
    # Size the white band to the two lines: equal padding above and below.
    extent = title.get_window_extent(fig.canvas.get_renderer())
    band = 1 - (extent.y0/fig.bbox.height - (1-title_top))
    fig.add_artist(Rectangle((0, 1-band), 1, band, transform=fig.transFigure,
                             facecolor="white", edgecolor="none", zorder=10))
    height = max(5.,float(states[:,:,2].max())*1.3)
    city = draw_city(ax, (xmin,ymin,xmax,ymax), cfg["scenario_definition"]["rects"],
              cfg["scenario_definition"]["circles"], height)
    boxes = occluder_bounds(cfg["scenario_definition"]["rects"],
                            cfg["scenario_definition"]["circles"], height)
    n = states.shape[1]
    colors = plt.cm.Set1(np.arange(n)%9)
    # Enlarge only the aircraft illustration for field-scale readability.
    glyph = max(1.,(xmax-xmin)*.0025)
    half = cfg["params"]["VIEWING_RADIUS"]
    trails,aircraft,footprints = [],[],[]
    for i in range(n):
        trails.append(FadingTrail(ax,colors[i],lw=1.5,label=f"UAV {i+1}"))
        aircraft.append(QuadcopterGlyph(ax, colors[i], glyph))
        # Long square edges are split so an obstacle can hide part of one.
        footprints.append(DepthSplitLine(ax,segment=half/8,color=colors[i],lw=.8,alpha=.6))
    target_line = DepthSplitLine(ax,color="#f1f5f9",ls="--",lw=1)
    target_marker = DepthSplitLine(ax,marker="*",color="#dc2626",ms=12,ls="none")
    ax.legend(handles=[trail.handle for trail in trails],
              loc="upper left", bbox_to_anchor=(.008, 1-band-.01), fontsize=8)
    square = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0],[-half,-half,0]])
    def update(t):
        index = max(0,min(len(times)-1,np.searchsorted(times,t,side="right")-1))
        s = states[index]
        # Set the camera first: occlusion is tested against this frame's view.
        if following:
            follow_camera(ax, targets[index], (xmin,ymin,xmax,ymax), max_z, zoom,
                          flight_altitude)
        projection = ax.get_proj()
        stride = max(1,index//1000)
        trail_times = np.r_[times[:index:stride], times[index]]
        keys = set()
        for i in range(n):
            # Keep the newest sample so the trail reaches the aircraft.
            fade = trail_fade(times[index] - trail_times, fade_window, fade_floor)
            keys |= trails[i].update(np.c_[np.r_[states[:index:stride,i,:3], s[None,i,:3]],
                                           fade], projection, boxes)
            keys |= aircraft[i].update(s[i,:3], s[i,6:10], projection, boxes)
            keys |= footprints[i].update(square+np.r_[s[i,:2],.05], projection, boxes)
        keys |= target_line.update(np.r_[targets[:index:stride], targets[None,index]],
                                   projection, boxes)
        keys |= target_marker.update(targets[index,None,:], projection, boxes)
        # Raise only obstacles that hide an overlay, interleaved by depth.
        plan = LayerPlan(projection, boxes, keys)
        city.set_layers(plan.obstacle_layers)
        for part in (*trails, *footprints, target_line, target_marker, *aircraft):
            part.place(plan)
        title.set_text(title_text(cfg["scenario"], map_size, cfg["controller"], n, speed,
                                  times[index], solve_time_label(releases, solves, times[index]),
                                  separations[index], clearances[index]))
        return []
    return fig, update, times


def render(args):
    import matplotlib.pyplot as plt
    from . import video
    directory = Path(args.directory)
    output = Path(args.out or directory/"replay.png")
    if output.suffix.lower() not in (".png",".mp4"):
        raise ValueError("Replay output must end in .png or .mp4")
    output.parent.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    try:
        cfg = json.loads((directory/"config.json").read_text())
        about = f" ({cfg.get('scenario','?')}, {cfg.get('controller','?')})"
    except (OSError,ValueError):
        about = ""  # reported by the scene builder, which needs the config
    progress.log("replay",f"{directory}{about} -> {output}")
    if output.suffix.lower()==".png":
        with progress.stage("replay","building scene"):
            fig, update, times = build_scene(args)
        t = times[-1] if args.time is None else args.time
        with progress.stage("replay",f"drawing PNG at t={t:.2f}s"):
            update(t)
            fig.savefig(output,dpi=180)
        plt.close(fig)
    else:
        video.export(args, output)
    size = output.stat().st_size
    progress.log("replay",f"wrote {output} ({size/2**20:.1f} MiB) in "
                          f"{progress.duration(time.perf_counter()-started)}")
    print(output)
