import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pytest

from uav3d.cli import main, parser
from uav3d.replay import TargetFollowAxes, follow_camera


@pytest.mark.parametrize("figsize", [(12, 8), (8, 12)])
def test_camera_fills_frame_without_stretching_geometry(figsize):
    fig = plt.figure(figsize=figsize, dpi=60)
    ax = fig.add_axes((0, 0, 1, 1), axes_class=TargetFollowAxes)
    ax.set_axis_off()
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    follow_camera(ax, [500, 500, 0], (0, 0, 1000, 1000), 50, 3)
    try:
        fig.canvas.draw()
        np.testing.assert_allclose(ax.bbox.bounds, fig.bbox.bounds)
        # Equal display-space steps along screen right/up must have equal pixel
        # lengths (after accounting for intentional altitude exaggeration).
        center = np.array([500., 500., 0.])
        world_scale = ax._box_aspect / [1000, 1000, 50]
        points = [center, center+.01*ax._view_u/world_scale,
                  center+.01*ax._view_v/world_scale]
        pixels = np.array([ax.transData.transform(
            proj3d.proj_transform(*p, ax.get_proj())[:2]) for p in points])
        assert np.linalg.norm(pixels[1]-pixels[0]) == pytest.approx(
            np.linalg.norm(pixels[2]-pixels[0]))
    finally:
        plt.close(fig)


def test_follow_keeps_target_projected_at_center_without_zoom_pumping():
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    previous_offset = None
    try:
        for target in ([100, 100, 0], [0, 0, 0], [1000, 1000, 20]):
            _, window = follow_camera(ax, target, (0, 0, 1000, 1000), 50, 3)
            projected = proj3d.proj_transform(*target, ax.get_proj())
            np.testing.assert_allclose(projected[:2], (0, 0), atol=1e-12)
            assert ax.get_xlim() == (0, 1000)
            assert ax.get_ylim() == (0, 1000)
            assert window[2]-window[0] == pytest.approx(1000/3)
            assert ax.get_zlim() == (0, 50)
            nearby = np.asarray(target) + (10, 20, 5)
            offset = proj3d.proj_transform(*nearby, ax.get_proj())
            if previous_offset is not None:
                np.testing.assert_allclose(offset[:2], previous_offset, atol=1e-12)
            previous_offset = offset[:2]
            assert (ax.elev, ax.azim) == (55, -65)
    finally:
        plt.close(fig)


def test_replay_defaults_to_target_follow():
    args = parser().parse_args(["replay", "recording"])
    assert args.camera == "follow"
    assert args.zoom == 6


@pytest.mark.parametrize("zoom", ["0", "-1", "nan", "inf"])
def test_replay_rejects_invalid_zoom(zoom):
    with pytest.raises(SystemExit) as error:
        main(["replay", "recording", "--zoom", zoom])
    assert error.value.code == 2


def test_follow_preserves_scenery_geometry_across_frames():
    from uav3d.scene import draw_city

    fig = plt.figure()
    ax = fig.add_subplot(axes_class=TargetFollowAxes, computed_zorder=False)
    ax.view_init(elev=55, azim=-65)
    ax.set_proj_type("ortho")
    draw_city(ax, (0, 0, 1000, 1000), [(80, 80, 40, 35)],
              [(160, 100, 22)], 40)
    from unittest.mock import Mock

    setters = []
    for collection in ax.collections:
        collection.set_verts = Mock(wraps=collection.set_verts)
        setters.append(collection.set_verts)
    try:
        for target in ([100, 100, 0], [900, 950, 20]):
            follow_camera(ax, target, (0, 0, 1000, 1000), 50, 3)
            fig.canvas.draw()
            for setter in setters:
                setter.assert_not_called()
            projected = proj3d.proj_transform(*target, ax.get_proj())
            np.testing.assert_allclose(projected[:2], (0, 0), atol=1e-12)
    finally:
        plt.close(fig)
