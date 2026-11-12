import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from uav3d import video
from uav3d.cli import main, parser
from uav3d.replay import DepthSplitLine, TIE_BREAK, build_scene, video_frame_times

GiB = 2**30


def make_run(directory, samples=40, duration=2.):
    """A tiny two-UAV run whose trails pass behind a building and a tree."""
    directory.mkdir()
    t = np.linspace(0, duration, samples)
    aircraft = np.zeros((samples, 2, 13))
    for i, offset in enumerate((-4., 4.)):
        aircraft[:, i, 0] = 10 + 30*t/duration
        aircraft[:, i, 1] = 25 + offset
        aircraft[:, i, 2] = 3.
        aircraft[:, i, 9] = 1.  # identity quaternion (scalar last)
    target = np.c_[10 + 30*t/duration, np.full(samples, 25.), np.full(samples, 3.)]
    np.savez(directory/"trajectory.npz", time=t, aircraft=aircraft, target=target)
    (directory/"config.json").write_text(json.dumps({
        "scenario": "test", "controller": "ours", "compute_ratio": 0.,
        "params": {"WORLD_BOUNDS": [0, 0, 50, 50], "VIEWING_RADIUS": 3.},
        "scenario_definition": {"rects": [[22, 26, 6, 6]], "circles": [[30, 18, 3]]}}))
    return directory


def replay_args(directory, out):
    return SimpleNamespace(directory=str(directory), out=str(out), time=None, fps=4,
                           speed=1., camera="follow", zoom=2.)


def test_auto_jobs_respect_cpus_memory_and_frames():
    plenty = 512*GiB
    by_cpu = video.plan_export(10_000, cpus=48, memory=plenty)
    assert by_cpu.jobs == int((48 - video.ENCODER_CORES)/video.WORKER_CORES)
    assert video.plan_export(10_000, cpus=2, memory=plenty).jobs == 1
    assert video.plan_export(80, cpus=48, memory=plenty).jobs == 80//video.MIN_FRAMES_PER_WORKER
    assert video.plan_export(5, cpus=48, memory=plenty).jobs == 1
    per_worker = video.worker_memory(4)
    budget = video.ENCODER_BYTES + 3.5*per_worker
    by_memory = video.plan_export(10_000, cpus=48, memory=budget/video.MEMORY_FRACTION)
    assert by_memory.jobs == 3 and "memory" in by_memory.notes[0]
    explicit = video.plan_export(10_000, max_memory_gb=budget/GiB, cpus=48, memory=1)
    assert explicit.jobs == 3 and explicit.memory_budget == int(budget)


def test_explicit_jobs_are_honoured_with_warnings():
    plan = video.plan_export(10_000, jobs="64", cpus=8, memory=4*GiB)
    assert plan.jobs == 64
    assert any("memory budget" in note for note in plan.notes)
    assert any("8 available CPUs" in note for note in plan.notes)
    assert video.plan_export(3, jobs=8, cpus=8, memory=None).jobs == 3
    assert video.plan_export(100, jobs=1, encoder_threads="6").encoder_threads == 6


@pytest.mark.parametrize("option, value", [("--jobs", "0"), ("--jobs", "many"),
                                           ("--max-memory", "-1"), ("--queue-frames", "0"),
                                           ("--encoder-threads", "0")])
def test_replay_rejects_invalid_export_options(option, value):
    with pytest.raises(SystemExit):
        main(["replay", "runs/missing", "--out", "x.mp4", option, value])


def test_export_options_default_from_environment(monkeypatch):
    assert parser().parse_args(["replay", "r"]).jobs == "auto"
    monkeypatch.setenv("UAV3D_REPLAY_JOBS", "6")
    monkeypatch.setenv("UAV3D_REPLAY_MAX_MEMORY_GB", "12")
    monkeypatch.setenv("UAV3D_REPLAY_QUEUE_FRAMES", "2")
    monkeypatch.setenv("UAV3D_REPLAY_ENCODER_THREADS", "8")
    args = parser().parse_args(["replay", "r"])
    assert (args.jobs, args.max_memory, args.queue_frames, args.encoder_threads) == \
        ("6", "12", 2, "8")
    assert parser().parse_args(["replay", "r", "--jobs", "2"]).jobs == "2"


def test_disk_warning_only_when_space_is_short(tmp_path, monkeypatch):
    assert video.check_disk(tmp_path, 10) is None
    monkeypatch.setattr(video.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=video.BYTES_PER_FRAME_ESTIMATE))
    assert "free" in video.check_disk(tmp_path, 10)


def test_equal_layer_ties_draw_in_construction_order():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class Plan:
        def layer(self, key):
            return 1.5 + .1*min(key), .01

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    first, second = DepthSplitLine(ax), DepthSplitLine(ax)
    # The later overlay grows a new hidden-part artist first (created earlier).
    for line, keys in ((second, (1, 2)), (first, (1, 2))):
        line.pieces = [(frozenset({k}), np.zeros((2, 3))) for k in keys]
        line.place(Plan())
    zorders = [first.backs[1].get_zorder(), second.backs[1].get_zorder()]
    assert zorders[0] < zorders[1] and zorders[1] - zorders[0] == pytest.approx(TIE_BREAK)
    plt.close(fig)


def test_frame_does_not_depend_on_previously_drawn_frames(tmp_path):
    args = replay_args(make_run(tmp_path/"run"), tmp_path/"x.mp4")

    def frame(history):
        fig, update, times = build_scene(args)
        frames = video_frame_times(times, args.speed, args.fps)
        for i in history:
            update(frames[i])
            fig.canvas.draw()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="rgba", dpi=60)
        return buffer.getvalue()

    assert frame([6]) == frame([0, 2, 4, 6]) == frame([5, 1, 6])


def test_parallel_export_matches_serial_byte_for_byte(tmp_path):
    run = make_run(tmp_path/"run")
    serial, parallel = tmp_path/"serial.mp4", tmp_path/"parallel.mp4"
    main(["replay", str(run), "--out", str(serial), "--fps", "4", "--zoom", "2", "--jobs", "1"])
    main(["replay", str(run), "--out", str(parallel), "--fps", "4", "--zoom", "2",
          "--jobs", "3", "--queue-frames", "1"])
    assert serial.stat().st_size > 0
    assert serial.read_bytes() == parallel.read_bytes()
    assert not list(tmp_path.glob(".*partial*"))


def test_failed_worker_leaves_no_output(tmp_path):
    run = make_run(tmp_path/"run")
    (run/"config.json").unlink()  # the parent only reads times; workers need config
    out = tmp_path/"broken.mp4"
    with pytest.raises(RuntimeError, match="worker"):
        main(["replay", str(run), "--out", str(out), "--fps", "4", "--jobs", "2"])
    assert not out.exists() and not list(tmp_path.glob(".*partial*"))
