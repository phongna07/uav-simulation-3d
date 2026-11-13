import numpy as np
import pytest
from uav3d import progress
from uav3d.cli import main, parser, worker_command
from uav3d.metrics import calculate, summary, target_views


def test_duration_formatting():
    assert progress.duration(4.24) == "4.2s"
    assert progress.duration(185) == "3m05s"
    assert progress.duration(3725) == "1h02m"
    assert progress.duration(None) == "?"


def test_ticker_is_rate_limited_and_can_be_disabled():
    now = [0.]
    ticker = progress.Ticker(10, clock=lambda: now[0])
    now[0] = 9.9
    assert not ticker.due()
    now[0] = 10.
    assert ticker.due()
    assert not ticker.due()
    assert ticker.elapsed() == 10.
    off = progress.Ticker(0, clock=lambda: now[0])
    now[0] = 1e6
    assert not off.due()


def test_eta_extrapolates_rate():
    assert progress.eta(25, 100, 10.) == pytest.approx(30.)
    assert progress.eta(0, 100, 10.) is None


def test_log_goes_to_stderr_not_stdout(capsys):
    progress.log("run ours", "hello")
    out, err = capsys.readouterr()
    assert out == ""
    assert err.rstrip().endswith("[run ours] hello")


def test_progress_interval_flag_is_validated_and_forwarded(monkeypatch):
    for bad in ("-1", "nan", "soon"):
        with pytest.raises(SystemExit):
            main(["metrics", "runs/missing", "--progress-interval", bad])
    args = parser().parse_args(["run", "--progress-interval", "2.5"])
    args.progress_interval = progress.parse_interval(args.progress_interval)
    command = worker_command(args, "ours", "runs/x")
    assert float(command[command.index("--progress-interval")+1]) == 2.5
    monkeypatch.setenv(progress.ENV_INTERVAL, "0")
    assert progress.parse_interval(parser().parse_args(["replay", "x"]).progress_interval) == 0


def test_target_views_reports_progress_without_changing_results():
    aircraft = np.zeros((4, 2, 17))
    target = np.zeros((4, 3))
    calls = []
    with_progress = target_views(aircraft, target, 1., lambda done, total: calls.append((done, total)))
    plain = target_views(aircraft, target, 1.)
    assert calls == [(0, 3), (1, 3), (2, 3)]
    np.testing.assert_array_equal(with_progress[0], plain[0])
    np.testing.assert_array_equal(with_progress[1], plain[1])


def test_metrics_summary_line():
    time = np.array([0, .5, 1.])
    status = np.zeros((3, 1, 3))
    status[:, 0, 0] = 1
    m = calculate(time, np.zeros((3, 1, 17)), np.zeros((3, 3)), status, 2, 1,
                  result={"status": "collision", "collision": {"time": 1., "reason": "obstacle"}})
    line = summary(m)
    assert "collision at t=1.000s (obstacle)" in line
    assert "50.0% of 2.00s horizon" in line
    assert "1/1 UAVs gathered" in line
