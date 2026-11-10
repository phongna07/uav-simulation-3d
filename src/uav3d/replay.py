"""Offline pose rendering using only experiment artifacts."""
import json
from pathlib import Path
import numpy as np


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial.transform import Rotation
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
    ax = fig.add_subplot(111,projection="3d",facecolor="#f3f5f9")
    xmin,ymin,xmax,ymax = cfg["params"]["WORLD_BOUNDS"]
    max_z = max(10.,float(states[:,:,2].max())*1.7)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(0,max_z),
           xlabel="East [m]",ylabel="North [m]",zlabel="Altitude [m]")
    display_z = max(max_z,.22*max(xmax-xmin,ymax-ymin))
    z_scale = display_z/max_z
    ax.set_box_aspect((xmax-xmin,ymax-ymin,display_z))
    ax.set_zticks(np.linspace(0,max_z,3))
    ax.view_init(elev=35,azim=-65)
    height = max(5.,float(states[:,:,2].max())*1.3)
    def box(x,y,w,h):
        v = np.array([[x,y,0],[x+w,y,0],[x+w,y+h,0],[x,y+h,0],
                      [x,y,height],[x+w,y,height],[x+w,y+h,height],[x,y+h,height]])
        faces = [[v[j] for j in face] for face in
                 ((0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]
        ax.add_collection3d(Poly3DCollection(faces,facecolor="#8c9aaa",alpha=.35,
                                            edgecolor="#607080",linewidth=.3))
    for rect in cfg["scenario_definition"]["rects"]:
        box(*rect)
    for x,y,r in cfg["scenario_definition"]["circles"]:
        a = np.linspace(0,2*np.pi,24)
        z = np.array([0,height])
        ax.plot_surface(x+r*np.cos(a)[None,:]*np.ones((2,1)),
                        y+r*np.sin(a)[None,:]*np.ones((2,1)),z[:,None]*np.ones((1,len(a))),
                        color="#8c9aaa",alpha=.35)
    n = states.shape[1]
    colors = plt.cm.tab10(np.arange(n)%10)
    trails,arms,footprints,markers = [],[],[],[]
    for i in range(n):
        trails.append(ax.plot([],[],[],color=colors[i],lw=1.5,label=f"UAV {i}")[0])
        arms.append(ax.plot([],[],[],color=colors[i],lw=2)[0])
        footprints.append(ax.plot([],[],[],color=colors[i],lw=.8,alpha=.6)[0])
        markers.append(ax.plot([],[],[],marker="^",color=colors[i],ms=7,linestyle="")[0])
    target_line = ax.plot([],[],[],color="#111827",ls="--",lw=1)[0]
    target_marker = ax.plot([],[],[],marker="*",color="#dc2626",ms=12)[0]
    ax.legend(loc="upper left",fontsize=8)
    # Enlarge physical arms for readability on field-scale maps, while rotating
    # the glyph by the recorded physical attitude. Trajectories retain true scale.
    glyph = max(1.,(xmax-xmin)*.008)
    body = glyph*np.array([[-1,0,0],[1,0,0],[0,0,0],[0,-1,0],[0,1,0],[0,0,0],[0,0,.5]])
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
            rotated = Rotation.from_quat(s[i,6:10]).apply(body)+s[i,:3]
            line(arms[i],rotated)
            line(markers[i],s[i,None,:3])
            line(footprints[i],square+np.r_[s[i,:2],.05])
        line(target_line,targets[:index+1:max(1,index//1000)])
        line(target_marker,targets[index,None,:])
        ax.set_title(f"{cfg['scenario']} · {cfg['controller']} · {n} UAVs · t = {times[index]:.3f} s\n"
                     f"Recorded motion · enlarged aircraft glyphs · vertical display scale ×{z_scale:.1f}",fontsize=11)
        return []
    if output.suffix.lower()==".png":
        update(times[-1] if args.time is None else args.time)
        fig.savefig(output,dpi=180,bbox_inches="tight")
    else:
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        frames = np.r_[np.arange(0,times[-1],args.speed/args.fps),times[-1]]
        animation = FuncAnimation(fig,update,frames=frames,interval=1000/args.fps,blit=False)
        animation.save(output,writer=FFMpegWriter(fps=args.fps,codec="libx264",bitrate=1800))
    plt.close(fig)
    print(output)
