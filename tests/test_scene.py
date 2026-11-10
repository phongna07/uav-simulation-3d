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
    # Exact facade, roof, trunk and foliage colors exclude antialiased edges.
    colors = ["#aa9586", "#84776e", "#918175", "#b6a293", "#d0c7bd",
              "#70503a", "#826044", "#5d4331", "#39734d", "#448558",
              "#508d5b", "#326846", "#619664"]
    mask = np.zeros(reference.shape[:2], dtype=bool)
    for color in colors:
        rgb = np.round(np.array(matplotlib.colors.to_rgb(color))*255).astype(np.uint8)
        mask |= np.all(reference[:, :, :3] == rgb, axis=2)
    # Erode the mask to remove boundary pixels whose antialiasing blends with
    # the background even when the obstacle is correctly drawn on top.
    mask = binary_erosion(mask)
    assert mask.sum() > 100
    # Seams between adjacent antialiased faces let the backdrop tint a pixel
    # by a level or two; a paver painted on top would differ by far more.
    np.testing.assert_allclose(actual[mask].astype(int), reference[mask].astype(int), atol=3)


def test_follow_ground_sits_on_box_floor_and_is_clipped_to_view():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    try:
        _, window = follow_camera(ax, (900, 950, 0), (0, 0, 1000, 1000), 50, 3)
        clip = scene.draw_city(ax, (0, 0, 1000, 1000), [], [], 40)
        vertices = np.concatenate(clip(window))
        assert ax.get_zlim()[0] == 0
        assert vertices[:, 0].min() >= ax.get_xlim()[0] - 1e-9
        assert vertices[:, 0].max() <= min(1000, ax.get_xlim()[1]) + 1e-9
        assert vertices[:, 1].min() >= ax.get_ylim()[0] - 1e-9
        assert vertices[:, 1].max() <= min(1000, ax.get_ylim()[1]) + 1e-9
        np.testing.assert_array_equal(vertices[:, 2], 0)
    finally:
        plt.close(fig)
