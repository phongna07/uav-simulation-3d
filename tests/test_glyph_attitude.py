"""The aircraft glyph must show its true shape and attitude at any altitude.

The replay axes stretch altitude (by up to 11x for low flight on a wide map),
so the glyph compresses its body-relative heights to cancel that stretch.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from uav3d.aircraft import QuadcopterGlyph, vertical_stretch
from uav3d.replay import TargetFollowAxes, follow_camera, view_directions


def stretched_axes(max_z):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    follow_camera(ax, (500, 500, 0), (0, 0, 1000, 1000), max_z, 6)
    return fig, ax


def arm_tips(glyph):
    """World endpoints of the four arms (visible, no occluders)."""
    return np.array([np.asarray(arm.front.get_data_3d()).T[-2] for arm in glyph.arms])


@pytest.mark.parametrize("max_z, expected", [(10, 11.), (51, 110/51), (200, 1.)])
def test_vertical_stretch_matches_follow_camera(max_z, expected):
    fig, ax = stretched_axes(max_z)
    assert vertical_stretch(ax) == pytest.approx(expected)
    plt.close(fig)


@pytest.mark.parametrize("max_z", [10, 51, 200])  # 3 m, 30 m and high flight
def test_displayed_tilt_equals_true_tilt(max_z):
    fig, ax = stretched_axes(max_z)
    stretch = vertical_stretch(ax)
    glyph = QuadcopterGlyph(ax, "red", 2.5)
    position = np.array([500., 500., 3.])
    for roll_deg in (5., 20.):
        quat = Rotation.from_euler("x", roll_deg, degrees=True).as_quat()
        glyph.update(position, quat)
        # Undo the axes' stretch: what the viewer sees, in metres.
        seen = (arm_tips(glyph) - position)*[1, 1, stretch]
        expected = Rotation.from_quat(quat).apply(glyph.struts[:, 1])
        np.testing.assert_allclose(seen, expected, atol=1e-9)
        # The plane through the arm tips is tilted by exactly the true roll.
        normal = np.cross(seen[0]-seen[2], seen[1]-seen[3])
        tilt = np.degrees(np.arccos(abs(normal[2])/np.linalg.norm(normal)))
        assert tilt == pytest.approx(roll_deg)
    plt.close(fig)


def test_tilted_glyph_culls_faces_in_display_space():
    """A strongly stretched axes must not flip which hull faces are visible."""
    fig, ax = stretched_axes(10)
    glyph = QuadcopterGlyph(ax, "red", 2.5)
    projection = ax.get_proj()
    position = np.array([500., 500., 3.])
    # Tilt the top face 30 deg toward the camera: top stays visible, belly hidden.
    toward = view_directions(projection, [position])[0]
    axis = np.cross([0, 0, 1], toward)
    quat = Rotation.from_rotvec(np.radians(30)*axis/np.linalg.norm(axis)).as_quat()
    glyph.update(position, quat, projection, ())
    colors = glyph.shell.front.get_facecolor()
    belly = matplotlib.colors.to_rgba("#334155")
    assert len(colors) and not np.isclose(colors, belly).all(axis=1).any()
    plt.close(fig)
