"""Reusable quadcopter glyph, expressed in aircraft-local coordinates."""
import numpy as np
from scipy.spatial.transform import Rotation


def subdivide_triangle(a, b, c, n):
    """Split a triangle into n² congruent triangles (barycentric grid)."""
    def point(i, j):
        return a + (b-a)*i/n + (c-a)*j/n
    triangles = []
    for i in range(n):
        for j in range(n-i):
            triangles.append([point(i, j), point(i+1, j), point(i, j+1)])
            if i+j < n-1:
                triangles.append([point(i+1, j), point(i+1, j+1), point(i, j+1)])
    return triangles


class QuadcopterGlyph:
    """Update a field-scale aircraft illustration from a recorded rigid pose.

    Every part is depth-split against the obstacles independently: hull faces
    are cut into small triangles, and arms, rotor rings, hull outlines and the
    nose stripe into short segments. An obstacle edge therefore sweeps across
    the aircraft gradually instead of whole parts popping in or out.
    """

    def __init__(self, ax, color, scale, ring_points=49, face_pieces=4):
        from .replay import DepthSplitLine, DepthSplitPolys
        # The nose points along body +X, matching the simulation's attitude.
        lower = np.array([[-.38, -.24, -.10], [.30, -.24, -.10],
                          [.48, 0, -.10], [.30, .24, -.10], [-.38, .24, -.10]])
        upper = lower.copy()
        upper[:, 2] = .18
        # Convex prism: bottom, top, then side j between corners j and j+1.
        faces = [lower[::-1], upper, *[np.array([lower[j], lower[(j+1) % 5],
                                                 upper[(j+1) % 5], upper[j]])
                                       for j in range(5)]]
        face_colors = ["#334155", color, *[color]*5]
        centre = (lower.mean(axis=0) + upper.mean(axis=0))/2
        normals = []
        for face in faces:
            normal = np.cross(face[1]-face[0], face[2]-face[0])
            normals.append(normal*np.sign(normal @ (face.mean(axis=0)-centre)))
        self.normals = np.array(normals)/np.linalg.norm(normals, axis=1, keepdims=True)
        pieces, self.piece_face, colors = [], [], []
        for f, face in enumerate(faces):
            middle = face.mean(axis=0)
            for k in range(len(face)):
                for tri in subdivide_triangle(middle, face[k], face[(k+1) % len(face)],
                                              face_pieces):
                    pieces.append(tri)
                    self.piece_face.append(f)
                    colors.append(face_colors[f])
        self.pieces = scale*np.array(pieces)
        self.piece_face = np.array(self.piece_face)
        self.piece_colors = colors
        self.shell = DepthSplitPolys(ax, zorder=6, linewidths=.3)
        # Each outline edge with the two faces that meet there.
        self.edges = ([(scale*lower[j], scale*lower[(j+1) % 5], 0, 2+j) for j in range(5)] +
                      [(scale*upper[j], scale*upper[(j+1) % 5], 1, 2+j) for j in range(5)] +
                      [(scale*lower[j], scale*upper[j], 2+(j-1) % 5, 2+j) for j in range(5)])
        edge_length = scale*.18
        self.outlines = [DepthSplitLine(ax, zorder=6.5, segment=edge_length, color="#1e293b",
                                        lw=.5, solid_capstyle="round")
                         for _ in self.edges]

        motors = np.array([[x, y, 0.] for x, y in
                           ((.72, .72), (.72, -.72), (-.72, -.72), (-.72, .72))])
        self.struts = scale * np.array([[np.zeros(3), motor] for motor in motors])
        arm_length = np.linalg.norm(self.struts[0, 1])
        self.arms = [DepthSplitLine(ax, zorder=5, segment=arm_length/8, color="#334155",
                                    lw=2.2, solid_capstyle="butt")
                     for _ in self.struts]
        theta = np.linspace(0, 2*np.pi, ring_points)
        circle = np.column_stack((.32*np.cos(theta), .32*np.sin(theta),
                                  np.full_like(theta, .10)))
        self.rings = scale * np.array([circle + motor for motor in motors])
        self.rotors = [DepthSplitLine(ax, zorder=7, color=color, lw=1.3,
                                      solid_capstyle="butt") for _ in self.rings]
        # A white canopy/nose stripe gives an attitude cue without an arrow.
        self.nose = scale * np.array([[.02, 0, .20], [.40, 0, .20]])
        self.indicator = DepthSplitLine(ax, zorder=8, segment=scale*.05, color="#f8fafc",
                                        lw=1.6, solid_capstyle="butt")

    @property
    def parts(self):
        """Every depth-split part of the glyph."""
        return (self.shell, *self.outlines, *self.arms, *self.rotors, self.indicator)

    def place(self, plan=None):
        """Draw hidden pieces at the layers ``plan`` (a LayerPlan) assigns.

        Unblocked pieces always draw above the scenery, at their own zorder.
        """
        for part in self.parts:
            part.place(plan)

    def update(self, position, quaternion, projection=None, boxes=()):
        """Pose the glyph; return the occluder sets of its hidden pieces."""
        from .replay import view_directions
        rotation = Rotation.from_quat(quaternion)

        def world(points):
            return rotation.apply(points.reshape(-1, 3)).reshape(points.shape) + position

        boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
        # Back faces of the convex hull are culled, so the remaining (front)
        # faces never overlap on screen and need no depth sort across layers.
        if projection is None:
            facing = np.ones(len(self.normals), dtype=bool)
        else:
            toward = view_directions(projection, [position])[0]
            facing = rotation.apply(self.normals) @ toward > 0
        shown = facing[self.piece_face]
        keys = self.shell.update(world(self.pieces[shown]),
                                 [c for c, s in zip(self.piece_colors, shown) if s],
                                 projection, boxes)
        empty = np.empty((0, 3))
        for line, (a, b, f, g) in zip(self.outlines, self.edges):
            # A convex hull's edge is visible when either adjacent face is.
            points = world(np.array([a, b])) if facing[f] or facing[g] else empty
            keys |= line.update(points, projection, boxes)
        for part, points in zip((*self.arms, *self.rotors),
                                (*world(self.struts), *world(self.rings))):
            keys |= part.update(points, projection, boxes)
        keys |= self.indicator.update(world(self.nose), projection, boxes)
        return keys
