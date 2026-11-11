"""Reusable quadcopter glyph, expressed in aircraft-local coordinates."""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation


class QuadcopterGlyph:
    """Update a field-scale aircraft illustration from a recorded rigid pose."""

    def __init__(self, ax, color, scale):
        # The nose points along body +X, matching the simulation's attitude.
        lower = np.array([[-.38, -.24, -.10], [.30, -.24, -.10],
                          [.48, 0, -.10], [.30, .24, -.10], [-.38, .24, -.10]])
        upper = lower.copy()
        upper[:, 2] = .18
        self.hull = scale * np.array([
            lower[::-1], upper,
            # Side faces are padded to five vertices for a uniform array.
            *[np.array([lower[j], lower[(j+1) % 5], upper[(j+1) % 5],
                        upper[j], lower[j]]) for j in range(5)],
        ])
        self.shell = Poly3DCollection(
            self.hull, facecolors=["#334155", color, *[color]*5],
            edgecolors="#1e293b", linewidths=.5, zorder=6)
        ax.add_collection3d(self.shell)

        motors = np.array([[x, y, 0.] for x, y in
                           ((.72, .72), (.72, -.72), (-.72, -.72), (-.72, .72))])
        self.struts = scale * np.array([[np.zeros(3), motor] for motor in motors])
        self.arms = Line3DCollection(self.struts, colors="#334155", linewidths=2.2,
                                     zorder=5)
        ax.add_collection3d(self.arms)
        theta = np.linspace(0, 2*np.pi, 25)
        circle = np.column_stack((.32*np.cos(theta), .32*np.sin(theta),
                                  np.full_like(theta, .10)))
        self.rings = scale * np.array([circle + motor for motor in motors])
        self.rotors = Line3DCollection(self.rings, colors=[color], linewidths=1.3,
                                       zorder=7)
        ax.add_collection3d(self.rotors)
        # A white canopy/nose stripe gives an attitude cue without an arrow.
        self.nose = scale * np.array([[.02, 0, .20], [.40, 0, .20]])
        self.indicator, = ax.plot([], [], [], color="#f8fafc", lw=1.6, zorder=8)
        self._layers = [(artist, artist.get_zorder()) for artist in
                        (self.shell, self.arms, self.rotors, self.indicator)]

    def set_layer(self, offset=0., scale=1.):
        """Move the whole glyph to another layer, keeping its internal order."""
        for artist, zorder in self._layers:
            artist.set_zorder(offset+scale*zorder)

    def update(self, position, quaternion):
        rotation = Rotation.from_quat(quaternion)

        def world(points):
            return rotation.apply(points.reshape(-1, 3)).reshape(points.shape) + position

        self.shell.set_verts(world(self.hull))
        self.arms.set_segments(world(self.struts))
        self.rotors.set_segments(world(self.rings))
        nose = world(self.nose)
        self.indicator.set_data_3d(nose[:, 0], nose[:, 1], nose[:, 2])
