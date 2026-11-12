import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav3d.replay import (FIGURE_SPACE, bold, fixed_width, min_separations, round_solve_times,
                          solve_time_label, speed_label, title_text)


def event(round, drone, observed, execution, delay):
    return dict(round=round, drone=drone, observed_at=observed,
                execution_s=execution, injected_delay_s=delay)


def test_solve_time_is_round_max_independent_of_ratio():
    raw = [[.02, .05, .03], [.04, .01, .02]]
    for ratio in (.5, 1., 2.):
        events = [event(r, i, .1*r, 9., c*ratio) for r, row in enumerate(raw) for i, c in enumerate(row)]
        releases, solves = round_solve_times(events, ratio)
        assert np.allclose(releases, [0., .1])
        assert np.allclose(solves, [.05, .04])
    # Ratio 0 injects no delay; the measured duration is the solve time.
    events = [event(0, i, 0., c, 0.) for i, c in enumerate(raw[0])]
    assert np.allclose(round_solve_times(events, 0.)[1], [.05])


def test_label_tracks_latest_released_round():
    releases, solves = np.array([0., .1]), np.array([.05, .004])
    assert "50.0}$ ms" in solve_time_label(releases, solves, .09)
    assert solve_time_label(releases, solves, .1).endswith(bold(f"{FIGURE_SPACE*3}4.0")+" ms")
    assert "n/a" in solve_time_label(np.array([]), np.array([]), 0.)


def test_fixed_width_text_keeps_rendered_width():
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()
    times = {fig.text(.5, .5, f"t = {fixed_width(v, 7, 3)} s").get_window_extent(renderer).width
              for v in (0., 1.111, 9.999, 10., 123.456, 749.)}
    solves = {fig.text(.5, .5, solve_time_label(np.array([0.]), np.array([v]), 0.)).get_window_extent(renderer).width
               for v in (.0011, .0518, .9999, 2.0359)}
    gaps = {fig.text(.5, .5, title_text("s", "c", 2, 5., 1., "", v, False)).get_window_extent(renderer).width
            for v in (.5, 9.99, 43.07, 120.)}
    plt.close(fig)
    for widths in (times, solves, gaps):
        assert max(widths)-min(widths) < .5


def test_title_follows_template_with_bold_values():
    text = title_text("scen1", "pure_mpc", 5, 5., 194.58, "Max solve time d = x ms", 43.07, False)
    first, second = text.split("\n")
    assert first.startswith("Scenario: $\\mathbf{scen1}$ · Algorithm: $\\mathbf{pure\\_mpc}$ · ")
    assert "$\\mathbf{5}$ UAVs · Video speed: $\\mathbf{5x}$ · t = $\\mathbf{" in first
    assert second.startswith("Max solve time d = x ms · Minimum UAV Separation Distance: ")
    assert "43.07}$ m · Is UAV colliding? $\\mathbf{False}$" in second
    assert "n/a" in title_text("s", "c", 1, 1., 0., "", np.nan, True)
    assert speed_label(.5) == "0.5x" and speed_label(5.) == "5x"


def test_min_separation_is_closest_pair_per_sample():
    states = np.zeros((2, 3, 13))
    states[:, 1, 0] = [3., 1.]
    states[:, 2, :3] = [[0., 4., 0.], [0., 0., 2.]]
    assert np.allclose(min_separations(states), [3., 1.])
    assert np.isnan(min_separations(states[:, :1])).all()
