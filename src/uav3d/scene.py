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
    palette = palettes[index % len(palettes)]
    # Tile each facade around its windows instead of laying glass over a huge
    # wall polygon. This avoids mplot3d hiding lower windows during face sorting.
    rows = 4
    for side, span in enumerate((width, depth, width, depth)):
        columns = max(2, min(7, int(span / 9)))

        def panel(u0, u1, z0, z1, color):
            if side in (0, 2):
                wall = y if side == 0 else y+depth
                face = [(x+u0, wall, z0), (x+u1, wall, z0),
                        (x+u1, wall, z1), (x+u0, wall, z1)]
            else:
                wall = x+width if side == 1 else x
                face = [(wall, y+u0, z0), (wall, y+u1, z0),
                        (wall, y+u1, z1), (wall, y+u0, z1)]
            _mesh(meshes, [face], color)

        for row in range(rows):
            bottom, top = height*row/rows, height*(row+1)/rows
            for col in range(columns):
                left, right = span*col/columns, span*(col+1)/columns
                lo, hi = left+(right-left)*.20, right-(right-left)*.20
                z0, z1 = bottom+height*.055, top-height*.045
                door = row == 0 and col == columns//2 and side in (0, 3)
                if door:
                    z0 = 0
                panel(left, lo, bottom, top, palette[side])
                panel(hi, right, bottom, top, palette[side])
                panel(lo, hi, bottom, z0, palette[side])
                panel(lo, hi, z1, top, palette[side])
                # A dark frame, narrow mullion and two blue glass panes.
                frame = min(span/columns*.045, height*.012)
                panel(lo, hi, z0, z0+frame, "#46545c")
                panel(lo, hi, z1-frame, z1, "#46545c")
                panel(lo, lo+frame, z0+frame, z1-frame, "#46545c")
                panel(hi-frame, hi, z0+frame, z1-frame, "#46545c")
                mid = (lo+hi)/2
                panel(mid-frame/2, mid+frame/2, z0+frame, z1-frame, "#46545c")
                lit = (index*3+row*7+col+side) % 13 == 0
                panel(lo+frame, mid-frame/2, z0+frame, z1-frame,
                      "#ebd6a5" if lit else "#6894a6")
                panel(mid+frame/2, hi-frame, z0+frame, z1-frame,
                      "#f5e5bf" if lit else "#9fc1cd")
            # Pale stone courses and darker foundations emphasize each floor.
            band = height*(.018 if row else .035)
            panel(0, span, bottom, bottom+band, "#6a7072" if row == 0 else palette[4])

    # Inset roof, solid coping and four parapets, all inside the footprint.
    rim = min(width, depth)*.035
    _box(meshes, x+rim, y+rim, width-2*rim, depth-2*rim,
         height*.995, height, ["#667078"]*5)
    for bx, by, bw, bd in ((x, y, width, rim), (x, y+depth-rim, width, rim),
                            (x, y+rim, rim, depth-2*rim),
                            (x+width-rim, y+rim, rim, depth-2*rim)):
        _box(meshes, bx, by, bw, bd, height*.98, height*1.055,
             [palette[0], palette[1], palette[2], palette[3], "#e0ddd4"])
    _box(meshes, x+.60*width, y+.57*depth, .23*width, .25*depth,
         height, height*1.09, ["#899397"]*4+["#c5cccd"])
    # Small sloping solar panels create recognizable rooftop silhouettes.
    for row in range(2):
        for col in range(3):
            px, py = x+width*(.14+col*.13), y+depth*(.23+row*.22)
            face = [(px, py, height*1.015), (px+width*.11, py, height*1.015),
                    (px+width*.11, py+depth*.16, height*1.065),
                    (px, py+depth*.16, height*1.065)]
            _mesh(meshes, [face], "#24485e", edgecolors="#9db2ba", linewidths=.35)


def _branch(meshes, start, end, radius):
    start, end = np.asarray(start), np.asarray(end)
    axis = end-start
    axis = axis/np.linalg.norm(axis)
    across = np.cross(axis, [0., 1., 0.])
    across /= np.linalg.norm(across)
    other = np.cross(axis, across)
    angles = np.arange(9)*2*np.pi/8
    ring = np.array([np.cos(a)*across+np.sin(a)*other for a in angles])
    faces = [[start+radius*ring[i], start+radius*ring[i+1],
              end+radius*.55*ring[i+1], end+radius*.55*ring[i]] for i in range(8)]
    _mesh(meshes, faces, ["#70503a", "#826044", "#5d4331", "#987454"])


def _crown(meshes, center, radii, rng, base):
    # Irregular triangulated ellipsoids with a consistent sunlight direction.
    # Adjacent triangles share vertices: no cracks or regular striped bands.
    sectors, bands = 10, 6
    vertices = []
    for band in range(bands+1):
        latitude = np.pi*band/bands
        phase = (band % 2)*np.pi/sectors
        ring = []
        for sector in range(sectors):
            angle = sector*2*np.pi/sectors+phase
            rough = rng.uniform(.90, 1.08) if band not in (0, bands) else 1.
            ring.append(np.asarray(center)+np.asarray(radii)*rough*np.array([
                np.sin(latitude)*np.cos(angle), np.sin(latitude)*np.sin(angle),
                np.cos(latitude)]))
        vertices.append(ring)
    sun = np.array([-.45, -.6, .8])
    sun /= np.linalg.norm(sun)
    faces, colors = [], []
    for band in range(bands):
        for sector in range(sectors):
            nxt = (sector+1) % sectors
            quad = [vertices[band][sector], vertices[band+1][sector],
                    vertices[band+1][nxt], vertices[band][nxt]]
            for triangle in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                length = np.linalg.norm(normal)
                if length < 1e-10:
                    continue
                normal /= length
                if np.dot(normal, np.mean(triangle, axis=0)-center) < 0:
                    normal = -normal
                light = .72+.35*max(0, np.dot(normal, sun))+rng.uniform(-.045, .045)
                faces.append(triangle)
                colors.append(tuple(np.clip(np.asarray(base)*light, 0, 1)))
    _mesh(meshes, faces, colors)


def _tree(meshes, x, y, radius, height, index):
    rng = np.random.default_rng(4100+index)
    # Each tree has a branched trunk and an asymmetric, clustered leaf crown.
    # Offsets plus lobe radii stay strictly inside the original obstacle disk.
    top = height*(.92+.07*rng.random())
    _branch(meshes, (x, y, 0), (x, y, top*.66), radius*.10)
    base = [( .24, .47, .27), (.29, .51, .29), (.34, .53, .27)][index % 3]
    for lobe in range(3):
        angle = index*.73+lobe*2*np.pi/3
        cx, cy = x+radius*.34*np.cos(angle), y+radius*.34*np.sin(angle)
        cz = top*(.57+.035*(lobe % 2))
        _branch(meshes, (x, y, top*.30), (cx, cy, cz), radius*.045)
        _crown(meshes, (cx, cy, cz), (radius*.57, radius*.57, top*.30), rng, base)
    _crown(meshes, (x-radius*.08, y+radius*.04, top*.76),
           (radius*.62, radius*.62, top*.24), rng, base)


def draw_city(ax, bounds, rects, circles, height):
    """Draw display-scale brick paving, buildings, and trees from saved footprints."""
    xmin, ymin, xmax, ymax = bounds
    # Coarse pavers keep the bond visible on kilometre-scale overview maps and
    # bound mesh complexity. Geometry and colors are deterministic across frames.
    brick = max(xmax-xmin, ymax-ymin)/48
    # Extend the decorative floor on every side of the simulation domain.
    # A fixed world-span apron keeps paving under the panning camera near map
    # edges, including negative X/Y, without moving the world axes or obstacles.
    apron = 48*brick
    ground_bounds = (xmin-apron, ymin-apron, xmax+apron, ymax+apron)
    gxmin, gymin, gxmax, gymax = ground_bounds
    faces, colors = [], []
    palette = ["#414448", "#484b4f", "#3c3f43", "#4d5054", "#45484c"]
    for row in range(-48, int(np.ceil((ymax-ymin)/brick))+48):
        y = ymin+row*brick
        offset = brick if row % 2 else 0
        for column in range(-24, int(np.ceil((xmax-xmin+offset)/(2*brick)))+24):
            x = xmin-offset+column*2*brick
            left, right = max(gxmin, x), min(gxmax, x+2*brick)
            top = min(gymax, y+brick)
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
    # Smooth moving window edges; matching edge strokes seal adjacent tiles.
    scenery = Poly3DCollection([], zorder=1, antialiaseds=True, snap=False)
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
        scenery.set_linewidth([max(width, .2) for width in widths])
        return kept

    clip_view(ground_bounds)
    return clip_view
