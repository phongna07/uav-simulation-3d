import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from uav3d import scene
from uav3d.aircraft import QuadcopterGlyph
from uav3d.replay import (HIDDEN_LAYER, DepthSplitLine, TargetFollowAxes,
                          follow_camera, occluded, view_directions)

BUILDING = (480, 480, 40, 40)


def axes(proj_type):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
    ax.set_axis_off()
    ax.view_init(elev=35, azim=-65)
    ax.set_proj_type(proj_type)
    follow_camera(ax, (500, 500, 0), (0, 0, 1000, 1000), 50, 3)
    return fig, ax


def camera_side(ax):
    """A horizontal unit vector pointing from the scene toward the camera."""
    direction = view_directions(ax.get_proj(), [[500, 500, 0]])[0]
    direction[2] = 0
    return direction/np.linalg.norm(direction)


@pytest.mark.parametrize("proj_type", ["ortho", "persp"])
def test_points_behind_obstacles_are_occluded(proj_type):
    fig, ax = axes(proj_type)
    boxes = scene.occluder_bounds([BUILDING], [], 40)
    toward = camera_side(ax)
    centre = np.array([500, 500, 5.])
    points = [centre - 60*toward,          # behind the building, low
              centre + 60*toward,          # in front of it
              centre - 60*toward + [0, 0, 200],  # far above, seen over it
              (500, 500, .05)]             # ground under the building
    np.testing.assert_array_equal(occluded(ax.get_proj(), points, boxes),
                                  [True, False, False, True])
    plt.close(fig)


def test_split_line_and_glyph_change_layers():
    fig, ax = axes("ortho")
    boxes = scene.occluder_bounds([BUILDING], [], 40)
    toward = camera_side(ax)
    across = np.cross(toward, [0, 0, 1])
    centre = np.array([500, 500, 5.])
    behind = centre - 60*toward
    line = DepthSplitLine(ax, segment=5, color="red")
    # A line passing behind the building, extending well to each side.
    line.update([behind - 150*across, behind + 150*across], ax.get_proj(), boxes)
    front, back = (np.asarray(a.get_data_3d()[0]) for a in (line.front, line.back))
    assert np.isfinite(front).sum() > 0 and np.isfinite(back).sum() > 0
    assert line.back.get_zorder() < 1 < line.front.get_zorder()

    glyph = QuadcopterGlyph(ax, "red", 2)
    glyph.set_layer(HIDDEN_LAYER, .01)
    assert all(artist.get_zorder() < 1 for artist, _ in glyph._layers)
    glyph.set_layer()
    assert [a.get_zorder() for a, _ in glyph._layers] == [6, 5, 7, 8]
    plt.close(fig)


@pytest.mark.parametrize("proj_type", ["ortho", "persp"])
def test_hidden_trail_is_covered_by_building_pixels(proj_type):
    """A trail behind a building must not change the building's pixels."""
    def render(building=True, line=True):
        fig, ax = axes(proj_type)
        clip = scene.draw_city(ax, (0, 0, 1000, 1000),
                               [BUILDING] if building else [], [], 40)
        clip((380, 380, 620, 620))
        ax.collections[0].set_verts([])  # drop paving; keep a white backdrop
        if line:
            boxes = scene.occluder_bounds([BUILDING], [], 40)
            toward = camera_side(ax)
            across = np.cross(toward, [0, 0, 1])
            behind = np.array([500, 500, 10.]) - 45*toward
            trail = DepthSplitLine(ax, segment=2, color="red", lw=4)
            trail.update([behind - 80*across, behind + 80*across],
                         ax.get_proj(), boxes)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        plt.close(fig)
        return pixels

    from scipy.ndimage import binary_erosion
    blank = render(building=False, line=False)
    reference = render(line=False)
    building = binary_erosion(np.abs(reference - blank).max(axis=2) > 0, iterations=4)
    actual = render()
    line_only = np.abs(render(building=False) - blank).max(axis=2) > 50
    overlap = building & line_only
    assert overlap.sum() > 50  # the trail crosses the building on screen
    assert (line_only & ~building).sum() > 50  # and is visible beside it
    # Antialiased silhouette and corner seams may blend with whatever lies
    # beneath; a trail painted on top would change the whole overlap band.
    changed = np.abs(actual - reference)[overlap].max(axis=1) > 3
    assert changed.mean() < .1
