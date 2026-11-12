"""MP4 export: serial, or frames drawn by worker processes into one continuous encode.

Parallel export keeps the serial output format. Each worker draws an interleaved
subset of frames with matplotlib's own ``FuncAnimation.save`` path. It pipes them
into a private ffmpeg that applies the same supersampling filter (Lanczos
downscale and yuv420p conversion) and returns delivery-size raw frames. The
parent reorders frames through small bounded in-memory queues and streams them
into a single x264 encoder with the serial CRF settings. There are no segment
files and no concatenation: disk use is the final MP4 alone (written as a
partial file, renamed on success), and the keyframe structure and rate control
match a serial encode.
"""
from dataclasses import dataclass
import math
import multiprocessing
import os
from pathlib import Path
import queue as queue_module
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace

SUPERSAMPLE = 3
RENDER_DPI = 480
SCALE_FILTER = f"scale=iw/{SUPERSAMPLE}:ih/{SUPERSAMPLE}:flags=lanczos"
ENCODE_ARGS = ["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
FIGSIZE = (12, 8)  # must match replay.build_scene

# Calibration for automatic sizing (measured on a 48-core x86-64 host, scen1).
# Each worker keeps about one core busy (matplotlib plus its scaling ffmpeg); the
# shared encoder needs a few cores. Memory scales with the render buffer.
WORKER_CORES = 1.0  # throughput peaked at CPUs - 4 workers (48 cores: 44)
ENCODER_CORES = 4
WORKER_BASE_BYTES = 450 * 2**20
WORKER_RENDER_COPIES = 3          # Agg buffer + savefig stream + scaler input
ENCODER_BYTES = 3584 * 2**20  # x264 lookahead/threads plus parent buffers (measured)
MEMORY_FRACTION = .8              # share of available memory used by "auto"
BYTES_PER_FRAME_ESTIMATE = 40_000  # ~31 kB/frame observed for scen1 at CRF 18
PROGRESS_INTERVAL_S = 15.
MIN_FRAMES_PER_WORKER = 8  # "auto" avoids spawning workers that would barely draw


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def render_size():
    return tuple(int(round(v*RENDER_DPI)) for v in FIGSIZE)


def output_size():
    width, height = render_size()
    if width % SUPERSAMPLE or height % SUPERSAMPLE:
        raise ValueError("render size must be divisible by the supersampling factor")
    size = (width//SUPERSAMPLE, height//SUPERSAMPLE)
    if size[0] % 2 or size[1] % 2:
        raise ValueError("yuv420p output needs an even frame size")
    return size


# --- machine resources ----------------------------------------------------

def available_cpus():
    """CPUs this process may use: affinity mask, capped by a cgroup quota."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = None
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text[0] != "max":
            quota = int(text[0])/int(text[1])
    except (OSError, ValueError, IndexError):
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if q > 0:
                quota = q/period
        except (OSError, ValueError):
            pass
    if quota is not None:
        count = min(count, max(1, math.floor(quota)))
    return max(1, count)


def available_memory():
    """Bytes available now: MemAvailable, capped by cgroup headroom."""
    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1])*1024)
    except (OSError, ValueError):
        pass
    for limit, usage in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            text = Path(limit).read_text().strip()
            if text != "max" and int(text) < 2**60:
                candidates.append(int(text) - int(Path(usage).read_text()))
            break
        except (OSError, ValueError):
            continue
    if not candidates:
        try:
            candidates.append(os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            return None
    return max(0, min(candidates))


def worker_memory(queue_frames):
    width, height = render_size()
    out_w, out_h = output_size()
    return (WORKER_BASE_BYTES + WORKER_RENDER_COPIES*width*height*4
            + (queue_frames+2)*out_w*out_h*3//2)


# --- planning -------------------------------------------------------------

@dataclass
class ExportPlan:
    jobs: int
    queue_frames: int
    encoder_threads: int | None   # None: ffmpeg default (automatic)
    memory_budget: int | None
    notes: list


def parse_auto(value, kind, name, minimum):
    """'auto'/None -> None, otherwise a number of ``kind`` >= minimum."""
    if value is None or str(value).strip().lower() == "auto":
        return None
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be 'auto' or a number, got {value!r}") from None
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return number


def plan_export(frames, jobs="auto", max_memory_gb="auto", queue_frames=4,
                encoder_threads="auto", cpus=None, memory=None):
    """Choose the worker count from CPUs, a memory budget and the frame count."""
    requested = parse_auto(jobs, int, "--jobs", 1)
    budget_gb = parse_auto(max_memory_gb, float, "--max-memory", 0.)
    queue_frames = int(queue_frames)
    if queue_frames < 1:
        raise ValueError("--queue-frames must be at least 1")
    threads = parse_auto(encoder_threads, int, "--encoder-threads", 1)
    cpus = available_cpus() if cpus is None else cpus
    memory = available_memory() if memory is None else memory
    if budget_gb is not None:
        budget = int(budget_gb*2**30)
    elif memory is not None:
        budget = int(memory*MEMORY_FRACTION)
    else:
        budget = None
    per_worker = worker_memory(queue_frames)
    memory_jobs = (None if budget is None else
                   max(1, int((budget - ENCODER_BYTES)//per_worker)))
    notes = []
    if requested is None:
        cpu_jobs = max(1, int((cpus - ENCODER_CORES)/WORKER_CORES))
        frame_jobs = max(1, frames//MIN_FRAMES_PER_WORKER)
        chosen = min(cpu_jobs, memory_jobs or cpu_jobs, frame_jobs)
        limit = ("frame count" if chosen == frame_jobs else
                 "memory budget" if memory_jobs is not None and chosen == memory_jobs
                 else "CPU count")
        notes.append(f"auto jobs={chosen} (limited by {limit}; {cpus} CPUs, "
                     f"~{per_worker/2**30:.2f} GiB per worker"
                     + (f", budget {budget/2**30:.1f} GiB)" if budget is not None else ")"))
    else:
        chosen = min(requested, max(1, frames))
        if memory_jobs is not None and requested > memory_jobs:
            notes.append(f"warning: --jobs {requested} may need ~"
                         f"{(requested*per_worker+ENCODER_BYTES)/2**30:.1f} GiB, above the "
                         f"{budget/2**30:.1f} GiB memory budget")
        if requested > cpus:
            notes.append(f"warning: --jobs {requested} exceeds the {cpus} available CPUs")
    return ExportPlan(chosen, queue_frames, threads, budget, notes)


def check_disk(path, frames):
    """Warn early when the output filesystem is unlikely to hold the video."""
    try:
        free = shutil.disk_usage(path).free
    except OSError:
        return None
    estimate = frames*BYTES_PER_FRAME_ESTIMATE
    if free < 2*estimate:
        return (f"warning: only {free/2**20:.0f} MiB free at {path}; this video is "
                f"estimated at ~{estimate/2**20:.0f} MiB")
    return None


# --- export ---------------------------------------------------------------

def _log(message):
    print(f"[replay] {message}", file=sys.stderr, flush=True)


def _partial_path(output):
    return output.with_name(f".{output.stem}.partial{output.suffix}")


def export(args, output):
    import numpy as np
    from .replay import video_frame_times
    with np.load(Path(args.directory)/"trajectory.npz") as archive:
        times = archive["time"].copy()
    frames = video_frame_times(times, args.speed, args.fps)
    plan = plan_export(len(frames), getattr(args, "jobs", "auto"),
                       getattr(args, "max_memory", "auto"),
                       getattr(args, "queue_frames", 4),
                       getattr(args, "encoder_threads", "auto"))
    for note in plan.notes:
        _log(note)
    disk = check_disk(output.parent, len(frames))
    if disk:
        _log(disk)
    partial = _partial_path(output)
    partial.unlink(missing_ok=True)
    try:
        if plan.jobs == 1:
            _log(f"rendering {len(frames)} frames serially")
            export_serial(args, partial, frames, plan.encoder_threads)
        else:
            _log(f"rendering {len(frames)} frames with {plan.jobs} workers")
            export_parallel(args, partial, frames, plan)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _encoder_threads_args(threads):
    return [] if threads is None else ["-threads", str(threads)]


def export_serial(args, output, frames, encoder_threads=None):
    """The original single-process path: matplotlib drives one ffmpeg."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from .replay import build_scene
    fig, update, _ = build_scene(args)
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
    animation = FuncAnimation(fig, update, frames=frames, interval=1000/args.fps, blit=False)
    # Supersample smoothed mesh faces, then filter to the delivery resolution.
    # Direct rasterization makes subpixel window mullions jump between pixel
    # columns during camera pans. Matching edge strokes seal facade tiles;
    # downsampling the completed frame also stabilizes subpixel details.
    animation.save(output, dpi=RENDER_DPI,
                   writer=FFMpegWriter(fps=args.fps, codec="libx264",
                                       extra_args=["-vf", SCALE_FILTER, *ENCODE_ARGS,
                                                   *_encoder_threads_args(encoder_threads)]))
    plt.close(fig)


def _scene_spec(args):
    return {k: getattr(args, k, None) for k in ("directory", "speed", "fps", "camera", "zoom")}


def _scaler_writer_class():
    from matplotlib.animation import FFMpegWriter

    class ScaledFrameWriter(FFMpegWriter):
        """Matplotlib's ffmpeg writer, but scaled raw frames come back on stdout."""

        def __init__(self, emit, **kwargs):
            super().__init__(**kwargs)
            self._emit, self._failure = emit, None
            out_w, out_h = output_size()
            self._frame_bytes = out_w*out_h*3//2

        def _args(self):
            args = super()._args()
            head = args[:args.index("-i")+2]  # unchanged raw RGBA input options
            # One thread each: parallelism comes from the workers themselves.
            return [head[0], "-threads", "1", "-filter_threads", "1", *head[1:],
                    "-vf", SCALE_FILTER, "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"]

        def _run(self):
            super()._run()
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()

        def _read(self):
            try:
                while data := self._proc.stdout.read(self._frame_bytes):
                    if len(data) != self._frame_bytes:
                        raise RuntimeError(f"scaler returned a short frame ({len(data)} bytes)")
                    self._emit(data)
            except BaseException as error:  # reported by finish()
                self._failure = error

        def finish(self):
            self._proc.stdin.close()
            self._reader.join()
            err = self._proc.stderr.read().decode(errors="replace")
            self._proc.wait()
            if self._proc.returncode:
                raise subprocess.CalledProcessError(self._proc.returncode, self._proc.args,
                                                    None, err)
            if self._failure is not None:
                raise self._failure

    return ScaledFrameWriter


def _worker(spec, index, jobs, frames, frames_queue):
    """Draw frames[index::jobs] and queue their scaled yuv420p bytes in order."""
    try:
        from threadpoolctl import threadpool_limits
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from .replay import build_scene
        threadpool_limits(limits=1)
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe()
        args = SimpleNamespace(**spec)
        fig, update, _ = build_scene(args)
        mine = frames[index::jobs]
        animation = FuncAnimation(fig, update, frames=mine, interval=1000/args.fps, blit=False)
        writer = _scaler_writer_class()(lambda data: frames_queue.put(("frame", data)),
                                        fps=args.fps, codec="rawvideo")
        animation.save(os.devnull, dpi=RENDER_DPI, writer=writer)
        plt.close(fig)
        frames_queue.put(("done", None))
    except BaseException:
        frames_queue.put(("error", traceback.format_exc()))
        raise SystemExit(1)


def _next(frames_queue, process, index):
    while True:
        try:
            return frames_queue.get(timeout=1.)
        except queue_module.Empty:
            if not process.is_alive():
                try:
                    return frames_queue.get(timeout=1.)
                except queue_module.Empty:
                    raise RuntimeError(f"replay worker {index} exited (code "
                                       f"{process.exitcode}) before sending all frames") from None


def encoder_command(output, fps, threads=None):
    width, height = output_size()
    return [ffmpeg_exe(), "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}",
            "-pix_fmt", "yuv420p", "-framerate", str(fps), "-loglevel", "error",
            "-i", "pipe:", "-vcodec", "libx264", *ENCODE_ARGS, *_encoder_threads_args(threads),
            "-y", str(output)]


def export_parallel(args, output, frames, plan):
    context = multiprocessing.get_context("spawn")
    jobs = plan.jobs
    queues = [context.Queue(maxsize=plan.queue_frames) for _ in range(jobs)]
    spec = _scene_spec(args)
    workers = [context.Process(target=_worker, args=(spec, k, jobs, frames, queues[k]),
                               daemon=True, name=f"replay-worker-{k}") for k in range(jobs)]
    with tempfile.TemporaryFile() as errors:
        encoder = subprocess.Popen(encoder_command(output, args.fps, plan.encoder_threads),
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=errors)
        try:
            for worker in workers:
                worker.start()
            start = last = time.monotonic()
            for i in range(len(frames)):
                k = i % jobs
                kind, data = _next(queues[k], workers[k], k)
                if kind != "frame":
                    raise RuntimeError(f"replay worker {k} failed:\n{data}" if kind == "error"
                                       else f"replay worker {k} ended early")
                try:
                    encoder.stdin.write(data)
                except BrokenPipeError:
                    break  # reported below from the encoder's exit status
                now = time.monotonic()
                if now - last >= PROGRESS_INTERVAL_S:
                    last = now
                    rate = (i+1)/(now-start)
                    _log(f"frame {i+1}/{len(frames)} ({100*(i+1)/len(frames):.0f}%), "
                         f"{rate:.1f} frames/s, ~{(len(frames)-i-1)/rate:.0f} s left")
            for k, worker in enumerate(workers):
                kind, data = _next(queues[k], worker, k)
                if kind == "error":
                    raise RuntimeError(f"replay worker {k} failed:\n{data}")
                worker.join()
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
            encoder.wait()
            if encoder.returncode:
                errors.seek(0)
                raise RuntimeError("ffmpeg encoder failed:\n"
                                   + errors.read().decode(errors="replace"))
            _log(f"encoded {len(frames)} frames in {time.monotonic()-start:.0f} s")
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
            for q in queues:
                q.close()
                q.cancel_join_thread()
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
