"""Decorative replay geometry. Never used by sensing or collision detection."""
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _mesh(meshes, faces, colors, **kwargs):
    # Batch obstacle faces so mplot3d sorts individual faces rather than whole
    # trees/buildings. Ground is kept in its own background layer below them.
    for index, face in enumerate(faces):
        color = colors if isinstance(colors, str) else colors[index % len(colors)]
        meshes.append((face, to_rgba(color),
                       to_rgba(kwargs.get("edgecolors", color)),
                       kwargs.get("linewidths", 0)))


def _box(meshes, x, y, width, depth, bottom, top, colors):
    vertices = np.array([
        [x, y, bottom], [x+width, y, bottom],
        [x+width, y+depth, bottom], [x, y+depth, bottom],
        [x, y, top], [x+width, y, top],
        [x+width, y+depth, top], [x, y+depth, top],
    ])
    faces = vertices[np.array([(0, 1, 5, 4), (1, 2, 6, 5),
                               (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)])]
    _mesh(meshes, faces, colors, edgecolors="#343a40", linewidths=.25)


def _building(meshes, x, y, width, depth, height, index):
    palettes = [
        ["#aa9586", "#84776e", "#918175", "#b6a293", "#d0c7bd"],
        ["#9eabb3", "#74838e", "#84939d", "#b0bbc2", "#d2d9dc"],
        ["#b29e8b", "#8c7968", "#9b8878", "#c1ad98", "#ded2c4"],
    ]
    _box(meshes, x, y, width, depth, 0, height, palettes[index % len(palettes)])
    # Windows sit just outside each facade; the building's footprint is exact.
    windows = []
    rows = 4
    eps = min(width, depth)*.001
    for axis, span in ((0, width), (1, depth)):
        columns = max(2, min(7, int(span / 8)))
        for column in range(columns):
            lo = span*(column+.24)/columns
            hi = span*(column+.76)/columns
            for row in range(rows):
                z0, z1 = height*(.12+row*.20), height*(.24+row*.20)
                for far in (False, True):
                    if axis == 0:
                        wall = y+depth+eps if far else y-eps
                        face = [(x+lo, wall, z0), (x+hi, wall, z0),
                                (x+hi, wall, z1), (x+lo, wall, z1)]
                    else:
                        wall = x+width+eps if far else x-eps
                        face = [(wall, y+lo, z0), (wall, y+hi, z0),
                                (wall, y+hi, z1), (wall, y+lo, z1)]
                    windows.append(face)
    _mesh(meshes, windows, ["#b9d5df" if i % 7 else "#f2d6a0"
                        for i in range(len(windows))],
          edgecolors="#53636d", linewidths=.2)
    # Recessed rooftop equipment leaves the original obstacle outline intact.
    _box(meshes, x+.32*width, y+.32*depth, .36*width, .36*depth,
         height, height*1.04, ["#68737c"]*4+["#9aa4ac"])


def _tree(meshes, x, y, radius, height, index):
    angles = np.linspace(0, 2*np.pi, 17)
    # Low-poly trunk and broad crown; every ring fits inside the recorded disk.
    def rings(profile, palette):
        faces, colors = [], []
        for band, ((z0, r0), (z1, r1)) in enumerate(zip(profile, profile[1:])):
            for sector in range(len(angles)-1):
                a, b = angles[sector:sector+2]
                faces.append([(x+r0*np.cos(a), y+r0*np.sin(a), z0),
                              (x+r0*np.cos(b), y+r0*np.sin(b), z0),
                              (x+r1*np.cos(b), y+r1*np.sin(b), z1),
                              (x+r1*np.cos(a), y+r1*np.sin(a), z1)])
                colors.append(palette[(sector+band+index) % len(palette)])
        _mesh(meshes, faces, colors, linewidths=0)
    rings([(0, radius*.15), (height*.48, radius*.12)],
          ["#70503a", "#826044", "#5d4331"])
    rings([(height*.25, 0), (height*.38, radius*.78),
           (height*.58, radius), (height*.80, radius*.80),
           (height*.96, radius*.40), (height, 0)],
          ["#39734d", "#448558", "#508d5b", "#326846", "#619664"])


def draw_city(ax, bounds, rects, circles, height):
    """Draw display-scale brick paving, buildings, and trees from saved footprints."""
    xmin, ymin, xmax, ymax = bounds
    # Coarse pavers keep the bond visible on kilometre-scale overview maps and
    # bound mesh complexity. Geometry and colors are deterministic across frames.
    brick = max(xmax-xmin, ymax-ymin)/48
    faces, colors = [], []
    palette = ["#414448", "#484b4f", "#3c3f43", "#4d5054", "#45484c"]
    for row, y in enumerate(np.arange(ymin, ymax, brick)):
        offset = brick if row % 2 else 0
        for column, x in enumerate(np.arange(xmin-offset, xmax, 2*brick)):
            left, right = max(xmin, x), min(xmax, x+2*brick)
            top = min(ymax, y+brick)
            faces.append([(left, y, 0), (right, y, 0),
                          (right, top, 0), (left, top, 0)])
            colors.append(palette[(row*7+column*3) % len(palette)])
    # The camera always looks down on z=0. Draw paving first: averaging depths
    # of wide brick faces with tall obstacle faces can otherwise paint bricks
    # over trunks and lower walls, especially in the zoomed follow view.
    ground = Poly3DCollection([], edgecolors="#292c30", linewidths=.45, zorder=0)
    ax.add_collection3d(ground, autolim=False)
    obstacles = []
    for index, (x, y, width, depth) in enumerate(rects):
        meshes = []
        _building(meshes, x, y, width, depth, height, index)
        obstacles.append(((x, y, x+width, y+depth), meshes))
    for index, (x, y, radius) in enumerate(circles):
        meshes = []
        _tree(meshes, x, y, radius, height, index)
        obstacles.append(((x-radius, y-radius, x+radius, y+radius), meshes))
    scenery = Poly3DCollection([], zorder=1)
    ax.add_collection3d(scenery, autolim=False)

    def clip_view(window):
        """Show only scenery inside window; mplot3d does not clip polygons.

        Pavers are cut exactly at the window edge. Obstacles are shown whole
        or not at all, so none overhangs the axes box. Returns the paving.
        """
        x0, y0, x1, y1 = window
        kept, fills = [], []
        for face, color in zip(faces, colors):
            (left, bottom, _), _, (right, top, _), _ = face
            left, right = max(left, x0), min(right, x1)
            bottom, top = max(bottom, y0), min(top, y1)
            if left < right and bottom < top:
                kept.append([(left, bottom, 0), (right, bottom, 0),
                             (right, top, 0), (left, top, 0)])
                fills.append(color)
        ground.set_verts(kept)
        ground.set_facecolor(fills)
        visible = [mesh for (left, bottom, right, top), meshes in obstacles
                   if x0 <= left and right <= x1 and y0 <= bottom and top <= y1
                   for mesh in meshes]
        # Batch faces so mplot3d sorts individual faces, not whole objects.
        polys, face_colors, edges, widths = zip(*visible) if visible else ([],)*4
        scenery.set_verts(list(polys))
        scenery.set_facecolor(list(face_colors))
        scenery.set_edgecolor(list(edges))
        scenery.set_linewidth(list(widths))
        return kept

    clip_view(bounds)
    return clip_view
