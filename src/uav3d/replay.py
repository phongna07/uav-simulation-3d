"""Offline pose rendering using only experiment artifacts."""
import json
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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


def follow_camera(ax, target, bounds, max_z, zoom):
    """Move the camera to the target, keeping world bounds and heading fixed.

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
    ax.camera_zoom = zoom
    x, y = target[:2]
    return scale, (x-span/2, y-span/2, x+span/2, y+span/2)


# Layering with ``computed_zorder=False``: paving (0) < hidden overlays
# (HIDDEN_LAYER) < depth-sorted scenery (1) < visible overlays (>= 2).
# Overlay parts whose view ray is blocked by an obstacle volume are drawn
# beneath the scenery, so the obstacle covers them where its mesh is solid
# and they still show through gaps (e.g. between tree crown lobes).
HIDDEN_LAYER = .5


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


def occluded(projection, points, boxes):
    """True where the ray from a point to the camera crosses an obstacle box."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0 or len(boxes) == 0:
        return np.zeros(len(points), dtype=bool)
    direction = view_directions(projection, points)
    direction[np.abs(direction) < 1e-12] = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        lower = (boxes[None, :, 0] - points[:, None])/direction[:, None]
        upper = (boxes[None, :, 1] - points[:, None])/direction[:, None]
    near = np.minimum(lower, upper).max(axis=2)
    far = np.maximum(lower, upper).min(axis=2)
    return (far >= np.maximum(near, 0.)).any(axis=1)


class DepthSplitLine:
    """A 3D polyline drawn above the scenery only where nothing occludes it."""

    def __init__(self, ax, zorder=2., segment=None, **style):
        self.segment = segment  # optional maximum segment length for testing
        self.front = ax.plot([], [], [], zorder=zorder, **style)[0]
        style.pop("label", None)
        self.back = ax.plot([], [], [], zorder=HIDDEN_LAYER+zorder/100, **style)[0]

    def _subdivide(self, points):
        if self.segment is None or len(points) < 2:
            return points
        pieces = [points[:1]]
        for start, end in zip(points[:-1], points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(end-start)/self.segment)))
            pieces.append(start + np.outer(np.arange(1, count+1)/count, end-start))
        return np.concatenate(pieces)

    def update(self, points, projection, boxes):
        points = self._subdivide(np.asarray(points, dtype=float).reshape(-1, 3))
        if len(points) < 2:
            hidden = occluded(projection, points, boxes)
            for artist, mask in ((self.front, ~hidden), (self.back, hidden)):
                p = points[mask]
                artist.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
            return
        hidden = occluded(projection, (points[:-1]+points[1:])/2, boxes)
        # Consecutive segments of one layer stay one path (so dash patterns
        # continue); a NaN row breaks the path where the layer changes.
        breaks = np.flatnonzero(np.diff(hidden)) + 1
        runs = zip(np.r_[0, breaks], np.r_[breaks, len(hidden)])
        paths = {False: [], True: []}
        gap = np.full((1, 3), np.nan)
        for start, stop in runs:
            paths[bool(hidden[start])] += [points[start:stop+1], gap]
        for artist, key in ((self.front, False), (self.back, True)):
            p = np.concatenate(paths[key]) if paths[key] else np.empty((0, 3))
            artist.set_data_3d(p[:, 0], p[:, 1], p[:, 2])


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.patches import Rectangle
    from .scene import draw_city, occluder_bounds
    from .aircraft import QuadcopterGlyph
    import imageio_ffmpeg
    directory = Path(args.directory)
    cfg = json.loads((directory/"config.json").read_text())
    with np.load(directory/"trajectory.npz") as archive:
        times,states,targets = (archive[k].copy() for k in ("time","aircraft","target"))
    output = Path(args.out or directory/"replay.png")
    if output.suffix.lower() not in (".png",".mp4"):
        raise ValueError("Replay output must end in .png or .mp4")
    output.parent.mkdir(parents=True,exist_ok=True)
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
    z_scale = display_z/max_z
    ax.set_box_aspect((xmax-xmin,ymax-ymin,display_z))
    ax.set_zticks(np.linspace(0,max_z,3))
    ax.view_init(elev=35,azim=-65)
    following = getattr(args, "camera", "follow") == "follow"
    zoom = getattr(args, "zoom", 3.)
    if following:
        ax.view_init(elev=55, azim=-65)
        ax.set_proj_type("ortho")
    ax.set_axis_off()
    # Figure overlays do not reserve space or shrink the scenery viewport.
    fig.add_artist(Rectangle((0, .925), 1, .075, transform=fig.transFigure,
                             facecolor="white", edgecolor="none", zorder=10))
    title = fig.text(.5, .988, "", ha="center", va="top", fontsize=11,
                     color="#111111", zorder=11)
    height = max(5.,float(states[:,:,2].max())*1.3)
    draw_city(ax, (xmin,ymin,xmax,ymax), cfg["scenario_definition"]["rects"],
              cfg["scenario_definition"]["circles"], height)
    boxes = occluder_bounds(cfg["scenario_definition"]["rects"],
                            cfg["scenario_definition"]["circles"], height)
    n = states.shape[1]
    colors = plt.cm.Set1(np.arange(n)%9)
    # Enlarge only the aircraft illustration for field-scale readability.
    glyph = max(1.,(xmax-xmin)*.008)
    half = cfg["params"]["VIEWING_RADIUS"]
    trails,aircraft,footprints = [],[],[]
    for i in range(n):
        trails.append(DepthSplitLine(ax,color=colors[i],lw=1.5,label=f"UAV {i}"))
        aircraft.append(QuadcopterGlyph(ax, colors[i], glyph))
        # Long square edges are split so an obstacle can hide part of one.
        footprints.append(DepthSplitLine(ax,segment=half/8,color=colors[i],lw=.8,alpha=.6))
    target_line = DepthSplitLine(ax,color="#f1f5f9",ls="--",lw=1)
    target_marker = DepthSplitLine(ax,marker="*",color="#dc2626",ms=12,ls="none")
    ax.legend(handles=[trail.front for trail in trails],
              loc="upper left", bbox_to_anchor=(.008, .915), fontsize=8)
    square = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0],[-half,-half,0]])
    def update(t):
        index = max(0,min(len(times)-1,np.searchsorted(times,t,side="right")-1))
        s = states[index]
        # Set the camera first: occlusion is tested against this frame's view.
        if following:
            scale, _ = follow_camera(ax, targets[index], (xmin,ymin,xmax,ymax), max_z, zoom)
        else:
            scale = z_scale
        projection = ax.get_proj()
        stride = max(1,index//1000)
        hidden = occluded(projection, s[:,:3], boxes)
        for i in range(n):
            # Keep the newest sample so the trail reaches the aircraft.
            trails[i].update(np.r_[states[:index:stride,i,:3], s[None,i,:3]],
                             projection, boxes)
            aircraft[i].update(s[i,:3], s[i,6:10])
            aircraft[i].set_layer(*((HIDDEN_LAYER, .01) if hidden[i] else (0., 1.)))
            footprints[i].update(square+np.r_[s[i,:2],.05], projection, boxes)
        target_line.update(np.r_[targets[:index:stride], targets[None,index]],
                           projection, boxes)
        target_marker.update(targets[index,None,:], projection, boxes)
        camera_label = f"Target-follow camera · zoom ×{zoom:g}" if following else "Full-map camera"
        title.set_text(f"{cfg['scenario']} · {cfg['controller']} · {n} UAVs · t = {times[index]:.3f} s\n"
                       f"{camera_label} · enlarged aircraft glyphs · vertical display scale ×{scale:.1f}")
        return []
    if output.suffix.lower()==".png":
        update(times[-1] if args.time is None else args.time)
        fig.savefig(output,dpi=180)
    else:
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        frames = np.r_[np.arange(0,times[-1],args.speed/args.fps),times[-1]]
        animation = FuncAnimation(fig,update,frames=frames,interval=1000/args.fps,blit=False)
        # Supersample smoothed mesh faces, then filter to the delivery resolution.
        # Direct rasterization makes subpixel window mullions jump between pixel
        # columns during camera pans. Matching edge strokes seal facade tiles;
        # downsampling the completed frame also stabilizes subpixel details.
        animation.save(output, dpi=480,
                       writer=FFMpegWriter(
                           fps=args.fps, codec="libx264",
                           extra_args=["-vf", "scale=iw/3:ih/3:flags=lanczos",
                                       "-crf", "18", "-pix_fmt", "yuv420p",
                                       "-movflags", "+faststart"]))
    plt.close(fig)
    print(output)
