import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from uav3d import scene
from uav3d.aircraft import QuadcopterGlyph
from uav3d.replay import (DepthSplitLine, TargetFollowAxes,
                          LayerPlan, follow_camera, front_obstacles, occluded, occluders,
                          view_directions)

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
    assert 1 < line.back.get_zorder() < 2 <= line.front.get_zorder()

    glyph = QuadcopterGlyph(ax, "red", 2)
    glyph.update([200, 200, 5], [0, 0, 0, 1], ax.get_proj(), boxes)
    glyph.place()
    assert [p.front.get_zorder() for p in glyph.parts] == [6] + [6.5]*15 + [5]*4 + [7]*4 + [8]
    # A level aircraft seen from above shows its top and some sides, not its belly.
    colors = glyph.shell.front.get_facecolor()
    assert len(colors) and not np.isclose(colors, plt.matplotlib.colors.to_rgba("#334155")).all(axis=1).any()
    plt.close(fig)


@pytest.mark.parametrize("proj_type", ["ortho", "persp"])
def test_glyph_parts_are_partly_hidden_at_a_building_edge(proj_type):
    """A ring straddling a building silhouette is split, not hidden whole."""
    fig, ax = axes(proj_type)
    boxes = scene.occluder_bounds([BUILDING], [], 40)
    toward = camera_side(ax)
    across = np.cross(toward, [0, 0, 1])
    projection = ax.get_proj()
    glyph = QuadcopterGlyph(ax, "red", 20)

    def split_counts(part):
        if hasattr(part, "polygons"):
            front = len(part.front.get_paths())
            back = sum(len(b.get_paths()) for b in part.backs)
        else:
            front = np.isfinite(part.front.get_data_3d()[0]).sum()
            back = sum(np.isfinite(b.get_data_3d()[0]).sum() for b in part.backs)
        return front > 0 and back > 0

    # Slide the aircraft sideways out from behind the building: each rotor
    # ring and the hull must pass through a partly hidden state at the edge.
    ever_split = np.zeros(6, dtype=bool)
    for offset in np.linspace(0, 80, 321):
        centre = np.array([500, 500, 10.]) - 45*toward + offset*across
        keys = glyph.update(centre, [0, 0, 0, 1], projection, boxes)
        plan = LayerPlan(projection, boxes, keys)
        glyph.place(plan)
        fig.canvas.draw()  # Poly3DCollection paths are projected on draw
        ever_split |= [split_counts(p) for p in (*glyph.rotors, glyph.shell, glyph.indicator)]
    assert ever_split[:5].all()
    for part in glyph.parts:
        assert all(1 < b.get_zorder() < 2 for b in part.backs)
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
            keys = trail.update([behind - 80*across, behind + 80*across],
                                ax.get_proj(), boxes)
            plan = LayerPlan(ax.get_proj(), boxes, keys)
            clip.set_layers(plan.obstacle_layers)
            trail.place(plan)
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


@pytest.mark.parametrize("proj_type", ["ortho", "persp"])
def test_hidden_overlay_is_not_covered_by_farther_scenery(proj_type):
    """Overlays sit beneath their own occluders only, not farther obstacles.

    A trail hidden by a far building raises that building; an aircraft hidden
    only by a tree in front of it must still be drawn over that building.
    """
    fig, ax = axes(proj_type)
    toward = camera_side(ax)
    across = np.cross(toward, [0, 0, 1])
    centre = np.array([500, 500, 0.])
    far_xy, near_xy = centre - 60*toward, centre + 60*toward
    rects = [(far_xy[0]-20, far_xy[1]-20, 40, 40)]
    circles = [(near_xy[0], near_xy[1], 6)]
    boxes = scene.occluder_bounds(rects, circles, 40)
    clip = scene.draw_city(ax, (0, 0, 1000, 1000), rects, circles, 40)
    clip((300, 300, 700, 700))
    projection = ax.get_proj()
    aircraft = DepthSplitLine(ax, segment=2, color="red")
    trail = DepthSplitLine(ax, segment=2, color="blue")
    point = centre + [0, 0, 5]
    behind = far_xy.copy() - 40*toward + [0, 0, 5]
    keys = aircraft.update([point - 30*across, point + 30*across], projection, boxes)
    assert keys == {frozenset({1})}  # only the tree in front hides it
    keys |= trail.update([behind - 60*across, behind + 60*across], projection, boxes)
    assert frozenset({0}) in keys
    plan = LayerPlan(projection, boxes, keys)
    clip.set_layers(plan.obstacle_layers)
    aircraft.place(plan)
    trail.place(plan)
    layer = {j: z for z, indices in plan.obstacle_layers for j in indices}
    hidden_aircraft = max(a.get_zorder() for a in aircraft.backs
                          if len(a.get_data_3d()[0]))
    hidden_trail = max(a.get_zorder() for a in trail.backs if len(a.get_data_3d()[0]))
    assert 1 < hidden_trail < layer[0] < hidden_aircraft < layer[1] < 2
    fig.canvas.draw()
    raised = [c for c in ax.collections[2:] if isinstance(c, type(ax.collections[1]))]
    assert sum(len(c.get_paths()) > 0 for c in raised) == 2
    plt.close(fig)


def test_front_obstacles_include_nearer_overlapping_boxes():
    fig, ax = axes("ortho")
    toward = camera_side(ax)
    centre = np.array([500, 500, 0.])
    a, b = centre, centre + 40*toward  # b stands in front of a on screen
    far = centre + [400, -400, 0]      # elsewhere on screen
    rects = [(x-10, y-10, 20, 20) for x, y, _ in (a, b, far)]
    boxes = scene.occluder_bounds(rects, [], 40)
    assert front_obstacles(ax.get_proj(), boxes, {0}) == {0, 1}
    assert front_obstacles(ax.get_proj(), boxes, {1}) == {1}
    assert front_obstacles(ax.get_proj(), boxes, set()) == set()
    # The nearer box b is layered above a, and a part hidden by a sits below both.
    plan = LayerPlan(ax.get_proj(), boxes, [frozenset({0})])
    assert [indices for _, indices in plan.obstacle_layers] == [[0, 1]]
    assert plan.layer({0})[0] < plan.obstacle_layers[0][0]
    plt.close(fig)
