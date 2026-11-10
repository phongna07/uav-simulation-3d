"""Offline pose rendering using only experiment artifacts."""
import json
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class TargetFollowAxes(Axes3D):
    """Pan an orthographic camera without changing world axes or geometry."""

    camera_target = None
    camera_zoom = 1.

    def get_proj(self):
        projection = super().get_proj()
        if self.camera_target is not None:
            target = projection @ np.r_[self.camera_target, 1.]
            # Translate the camera in its screen plane, preserving heading.
            # Project the fixed world axes together with the scenery.
            for axis in (0, 1):
                projection[axis] = self.camera_zoom * (
                    projection[axis] - target[axis]/target[3]*projection[3])
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



def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from .scene import draw_city
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
    ax = fig.add_subplot(111, axes_class=TargetFollowAxes,
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
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d6dce3")
    height = max(5.,float(states[:,:,2].max())*1.3)
    draw_city(ax, (xmin,ymin,xmax,ymax), cfg["scenario_definition"]["rects"],
              cfg["scenario_definition"]["circles"], height)
    n = states.shape[1]
    colors = plt.cm.Set1(np.arange(n)%9)
    # Enlarge only the aircraft illustration for field-scale readability.
    glyph = max(1.,(xmax-xmin)*.008)
    trails,aircraft,footprints = [],[],[]
    for i in range(n):
        trails.append(ax.plot([],[],[],color=colors[i],lw=1.5,label=f"UAV {i}")[0])
        aircraft.append(QuadcopterGlyph(ax, colors[i], glyph))
        footprints.append(ax.plot([],[],[],color=colors[i],lw=.8,alpha=.6)[0])
    target_line = ax.plot([],[],[],color="#f1f5f9",ls="--",lw=1)[0]
    target_marker = ax.plot([],[],[],marker="*",color="#dc2626",ms=12)[0]
    ax.legend(loc="upper left",fontsize=8)
    half = cfg["params"]["VIEWING_RADIUS"]
    square = np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0],[-half,-half,0]])
    def line(artist,p):
        artist.set_data_3d(p[:,0],p[:,1],p[:,2])
    def update(t):
        index = max(0,min(len(times)-1,np.searchsorted(times,t,side="right")-1))
        s = states[index]
        for i in range(n):
            stride = max(1,index//1000)
            line(trails[i],states[:index+1:stride,i,:3])
            aircraft[i].update(s[i,:3], s[i,6:10])
            line(footprints[i],square+np.r_[s[i,:2],.05])
        line(target_line,targets[:index+1:max(1,index//1000)])
        line(target_marker,targets[index,None,:])
        if following:
            scale, _ = follow_camera(ax, targets[index], (xmin,ymin,xmax,ymax), max_z, zoom)
        else:
            scale = z_scale
        camera_label = f"Target-follow camera · zoom ×{zoom:g}" if following else "Full-map camera"
        ax.set_title(f"{cfg['scenario']} · {cfg['controller']} · {n} UAVs · t = {times[index]:.3f} s\n"
                     f"{camera_label} · enlarged aircraft glyphs · vertical display scale ×{scale:.1f}",fontsize=11)
        return []
    if output.suffix.lower()==".png":
        update(times[-1] if args.time is None else args.time)
        fig.savefig(output,dpi=180)
    else:
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        frames = np.r_[np.arange(0,times[-1],args.speed/args.fps),times[-1]]
        animation = FuncAnimation(fig,update,frames=frames,interval=1000/args.fps,blit=False)
        animation.save(output, dpi=160,
                       writer=FFMpegWriter(fps=args.fps, codec="libx264", bitrate=6000,
                                           extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print(output)
