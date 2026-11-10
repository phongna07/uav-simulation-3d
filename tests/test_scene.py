import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from uav3d import scene
from uav3d.replay import TargetFollowAxes, follow_camera


@pytest.mark.parametrize("elevation", [35, 55])
def test_paving_does_not_paint_over_obstacles(elevation):
    """Adding the ground must preserve solid pixels of above-ground objects."""
    def render(ground=True):
        fig = plt.figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.view_init(elev=elevation, azim=-65)
        ax.set_proj_type("ortho")
        _, window = follow_camera(ax, (100, 100, 0), (0, 0, 1000, 1000), 50, 3)
        clip = scene.draw_city(ax, (0, 0, 1000, 1000),
                               [(80, 80, 40, 35)], [(160, 100, 22)], 40)
        clip(window)
        if not ground:
            ax.collections[0].set_verts([])  # paving is the first layer
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        return pixels

    reference = render(ground=False)
    actual = render()
    # Inspect fully opaque obstacle interiors. Antialiased edges (including
    # internal tile seams) intentionally blend with their backdrop.
    mask = reference[:, :, 3] == 255
    # Erode the mask to remove boundary pixels whose antialiasing blends with
    # the background even when the obstacle is correctly drawn on top.
    mask = binary_erosion(mask)
    assert mask.sum() > 100
    # Seams between adjacent antialiased faces let the backdrop tint a pixel
    # by a level or two; a paver painted on top would differ by far more.
    np.testing.assert_allclose(actual[mask].astype(int), reference[mask].astype(int), atol=3)


@pytest.mark.parametrize("target", [(0, 0, 0), (900, 950, 0), (1000, 1000, 0)])
def test_follow_ground_sits_on_box_floor_and_is_clipped_to_view(target):
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    try:
        _, window = follow_camera(ax, target, (0, 0, 1000, 1000), 50, 3)
        clip = scene.draw_city(ax, (0, 0, 1000, 1000), [], [], 40)
        vertices = np.concatenate(clip(window))
        assert ax.get_zlim()[0] == 0
        np.testing.assert_allclose(vertices[:, :2].min(axis=0), window[:2])
        np.testing.assert_allclose(vertices[:, :2].max(axis=0), window[2:])
        np.testing.assert_array_equal(vertices[:, 2], 0)
    finally:
        plt.close(fig)


def test_ground_extends_beyond_all_world_edges_without_changing_axes():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes)
    try:
        follow_camera(ax, (0, 0, 0), (0, 0, 1000, 1000), 50, 3)
        clip = scene.draw_city(ax, (0, 0, 1000, 1000), [], [], 40)
        vertices = np.concatenate(clip((-1000, -1000, 2000, 2000)))
        np.testing.assert_allclose(vertices[:, :2].min(axis=0), (-1000, -1000))
        np.testing.assert_allclose(vertices[:, :2].max(axis=0), (2000, 2000))
        assert ax.get_xlim() == (0, 1000)
        assert ax.get_ylim() == (0, 1000)
    finally:
        plt.close(fig)


def test_window_edges_remain_stable_during_subpixel_camera_pan():
    """Fine facade details should fade smoothly between pixels, not jump."""
    def flicker(smooth):
        fig = plt.figure(figsize=(4, 3), dpi=120)
        ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
        ax.set_axis_off()
        ax.view_init(elev=55, azim=-65)
        ax.set_proj_type("ortho")
        follow_camera(ax, (100, 100, 0), (0, 0, 1000, 1000), 50, 9)
        scene.draw_city(ax, (0, 0, 1000, 1000), [(80, 80, 40, 35)], [], 40)
        ax.collections[0].set_verts([])
        if not smooth:
            # Reproduce the previous opaque, unsmoothed window rendering.
            ax.collections[1].set_antialiased(False)
            ax.collections[1].set_linewidth(0)
        frames = []
        try:
            for offset in np.arange(9)*.04:
                follow_camera(ax, (100+offset, 100, 0), (0, 0, 1000, 1000), 50, 9)
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(float))
        finally:
            plt.close(fig)
        # Constant-speed panning should not produce abrupt intensity changes.
        return np.abs(np.diff(frames, n=2, axis=0)).sum()

    assert flicker(smooth=True) < .65*flicker(smooth=False)
