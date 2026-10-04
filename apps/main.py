"""
MediaPipe real-time demo (camera or video) with:
  - PoseLandmarker (pose skeleton)
  - HandLandmarker (hand skeleton) + lightweight V-sign rule

Run (camera):
  .venv\\Scripts\\python.exe main.py

Run (video file):
  .venv\\Scripts\\python.exe main.py --source path\\to\\demo.mp4 --out out.mp4

Quit:
  press "q" in the preview window
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import cv2

from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.paths import models_dir
from apps.camera_enum import open_camera
from core.parallel_pose_engine import ParallelPoseEngine, default_pipeline_factory


@dataclass(frozen=True)
class LatestFrameItem:
    index: int
    frame: object
    captured_at: float
    timestamp_ms: int


@dataclass(frozen=True)
class RealtimeLatestFrameMetrics:
    captured_frames: int
    processed_frames: int
    rendered_frames: int
    dropped_frames: int
    capture_fps: float | None
    infer_fps: float | None
    render_fps: float | None
    latency_p90_ms: float | None


class LatestFrameQueue:
    """Single-slot queue that keeps only the newest frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: LatestFrameItem | None = None
        self._dropped = 0
        self._closed = False

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def put(self, item: LatestFrameItem) -> None:
        with self._lock:
            if self._item is not None:
                self._dropped += 1
            self._item = item

    def get_latest(self) -> LatestFrameItem | None:
        with self._lock:
            item = self._item
            self._item = None
            return item

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


def _open_capture(source: str) -> cv2.VideoCapture:
    # If source is a digit, treat as camera index.
    if source.isdigit():
        # open_camera negotiates a fast capture format (MSMF / MJPG) and sets
        # the requested resolution; default CAP_DSHOW+YUY2 caps many webcams at ~10fps.
        return open_camera(int(source))
    return cv2.VideoCapture(source)


def _fmt_seconds(s: float) -> str:
    s = max(0.0, float(s))
    m, sec = divmod(int(s + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def _print_progress(done: int, total: int, t0: float) -> None:
    if total <= 0:
        elapsed = time.monotonic() - t0
        rate = done / max(1e-9, elapsed)
        msg = f"processed={done}  {rate:5.1f} fps  elapsed {_fmt_seconds(elapsed)}"
        print("\r" + msg, end="", flush=True)
        return

    frac = min(1.0, max(0.0, done / total))
    width = 28
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - t0
    rate = done / max(1e-9, elapsed)
    eta = (total - done) / max(1e-9, rate)
    msg = f"[{bar}] {done}/{total} ({frac*100:5.1f}%)  {rate:5.1f} fps  ETA {_fmt_seconds(eta)}"
    print("\r" + msg, end="", flush=True)
    if done >= total:
        print()


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _rate(count: int, seconds: float) -> float | None:
    if count <= 0 or seconds <= 1e-9:
        return None
    return round(count / seconds, 3)


def _round_ms(value: float | None) -> float | None:
    return None if value is None else round(value * 1000.0, 3)


def _metrics_to_dict(metrics: RealtimeLatestFrameMetrics) -> dict[str, Any]:
    return asdict(metrics)


def _process_video_multithread(
    *,
    cap: cv2.VideoCapture,
    out_path: str | None,
    show: bool,
    pose_variant: str,
    enable_hands: bool,
    workers: int,
    delegate: str = "cpu",
) -> None:
    # This mode parallelizes per-frame processing using IMAGE mode pipelines.
    # Trade-off: no temporal tracking/smoothing, but better CPU utilization and throughput.
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_for_ts = src_fps if src_fps > 1e-3 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer: cv2.VideoWriter | None = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_for_ts, (w, h))

    stop_evt = threading.Event()
    frame_q: "Queue[tuple[int, object] | None]" = Queue(maxsize=workers * 2)
    result_q: "Queue[tuple[int, object, list[str]]]" = Queue(maxsize=workers * 2)

    models_dir_path = models_dir()

    def reader() -> None:
        idx = 0
        while not stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_q.put((idx, frame))
            idx += 1
        for _ in range(workers):
            frame_q.put(None)

    def worker(worker_id: int) -> None:
        pipe = MediaPipePipeline(
            models_dir=models_dir_path,
            cfg=PipelineConfig(
                pose_variant=pose_variant,
                running_mode="image",
                enable_hands=enable_hands,
                delegate=delegate,
            ),
        )
        while True:
            item = frame_q.get()
            if item is None:
                break
            if stop_evt.is_set():
                continue
            idx, frame = item
            annotated, actions = pipe.annotate(frame, timestamp_ms=None)
            result_q.put((idx, annotated, actions))

    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()
    workers_ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(workers)]
    for t in workers_ts:
        t.start()

    next_idx = 0
    pending: dict[int, tuple[object, list[str]]] = {}

    t0 = time.monotonic()
    written = 0
    printed_done = 0

    while True:
        if next_idx in pending:
            annotated, actions = pending.pop(next_idx)
            # Overlay labels + fps
            y = 30
            for a in actions[:5]:
                cv2.putText(annotated, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
                y += 30
            fps = written / max(1e-6, (time.monotonic() - t0))
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(
                annotated,
                "Press 'q' to quit",
                (10, annotated.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            if writer is not None:
                writer.write(annotated)
                written += 1
                if written != printed_done:
                    _print_progress(written, total, t0)
                    printed_done = written

            if show:
                cv2.imshow("MediaPipe (Video/Camera)", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop_evt.set()
                    break

            next_idx += 1
            continue

        # When exporting, wait for results; for preview-only, still need ordering.
        try:
            idx, annotated, actions = result_q.get(timeout=0.2)
        except Exception:
            # All workers finished and buffer drained.
            alive = any(t.is_alive() for t in workers_ts)
            if (not alive) and (not pending):
                break
            continue

        pending[int(idx)] = (annotated, actions)

        # Exit when we've written all frames (if frame count is known).
        if total > 0 and next_idx >= total and not pending:
            break

    stop_evt.set()
    t_reader.join(timeout=2.0)
    for t in workers_ts:
        t.join(timeout=2.0)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_realtime_latest_frame_smoke(
    source: str,
    *,
    show: bool = True,
    pose_variant: str = "full",
    enable_hands: bool = True,
    delegate: str = "cpu",
    limit_frames: int | None = None,
) -> RealtimeLatestFrameMetrics:
    """Opt-in realtime smoke path: capture and inference share latest-frame semantics."""
    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    is_file = not source.isdigit()
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_for_ts = src_fps if (is_file and src_fps > 1e-3) else 30.0

    frame_q = LatestFrameQueue()
    result_q: "Queue[tuple[int, object, list[str], float, float]]" = Queue(maxsize=1)
    error_q: "Queue[BaseException]" = Queue()
    stop_evt = threading.Event()
    limit = None if limit_frames is None else max(0, int(limit_frames))

    counts = {"captured": 0, "processed": 0, "rendered": 0}
    latency_sec: list[float] = []
    capture_started = time.monotonic()
    capture_ended = capture_started
    infer_started: float | None = None
    infer_ended: float | None = None
    render_started: float | None = None
    render_ended: float | None = None

    def capture_loop() -> None:
        nonlocal capture_ended
        idx = 0
        try:
            while not stop_evt.is_set():
                if limit is not None and idx >= limit:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                captured_at = time.monotonic()
                ts = int(idx * 1000.0 / max(1e-6, fps_for_ts)) if is_file else int((captured_at - capture_started) * 1000)
                frame_q.put(LatestFrameItem(index=idx, frame=frame, captured_at=captured_at, timestamp_ms=ts))
                counts["captured"] += 1
                idx += 1
        except BaseException as exc:  # noqa: BLE001
            error_q.put(exc)
            stop_evt.set()
        finally:
            capture_ended = time.monotonic()
            frame_q.close()

    def infer_loop() -> None:
        nonlocal infer_started, infer_ended
        try:
            pipe = MediaPipePipeline(
                models_dir=models_dir(),
                cfg=PipelineConfig(
                    pose_variant=pose_variant,
                    running_mode="video",
                    enable_hands=enable_hands,
                    delegate=delegate,
                ),
            )
            infer_started = time.monotonic()
            while not stop_evt.is_set():
                item = frame_q.get_latest()
                if item is None:
                    if frame_q.is_closed():
                        break
                    time.sleep(0.001)
                    continue
                annotated, actions = pipe.annotate(item.frame, timestamp_ms=item.timestamp_ms)
                inferred_at = time.monotonic()
                counts["processed"] += 1
                try:
                    result_q.put((item.index, annotated, actions, item.captured_at, inferred_at), timeout=0.1)
                except Exception:
                    pass
        except BaseException as exc:  # noqa: BLE001
            error_q.put(exc)
            stop_evt.set()
        finally:
            infer_ended = time.monotonic()

    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_infer = threading.Thread(target=infer_loop, daemon=True)
    t_capture.start()
    t_infer.start()

    try:
        while True:
            if t_capture.is_alive() or t_infer.is_alive() or not result_q.empty():
                try:
                    _idx, annotated, actions, captured_at, _inferred_at = result_q.get(timeout=0.05)
                except Empty:
                    continue
                now = time.monotonic()
                if render_started is None:
                    render_started = now
                latency_sec.append(now - captured_at)
                counts["rendered"] += 1

                y = 30
                for action in actions[:5]:
                    cv2.putText(annotated, action, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
                    y += 30
                cv2.putText(
                    annotated,
                    f"Latest-frame smoke FPS: {counts['rendered'] / max(1e-6, now - (render_started or now)):.1f}",
                    (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                )
                if show:
                    cv2.imshow("MediaPipe (Latest Frame Smoke)", annotated)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        stop_evt.set()
                        break
                render_ended = time.monotonic()
                continue
            break
    finally:
        stop_evt.set()
        t_capture.join(timeout=2.0)
        t_infer.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()

    if not error_q.empty():
        raise RuntimeError("Latest-frame smoke worker failed") from error_q.get()

    infer_duration = 0.0 if infer_started is None or infer_ended is None else infer_ended - infer_started
    render_duration = 0.0 if render_started is None or render_ended is None else render_ended - render_started
    return RealtimeLatestFrameMetrics(
        captured_frames=counts["captured"],
        processed_frames=counts["processed"],
        rendered_frames=counts["rendered"],
        dropped_frames=frame_q.dropped_frames,
        capture_fps=_rate(counts["captured"], capture_ended - capture_started),
        infer_fps=_rate(counts["processed"], infer_duration),
        render_fps=_rate(counts["rendered"], render_duration),
        latency_p90_ms=_round_ms(_percentile(latency_sec, 90)),
    )


def run(
    source: str,
    *,
    show: bool = True,
    out_path: str | None = None,
    pose_variant: str = "full",
    enable_hands: bool = True,
    delegate: str = "cpu",
) -> None:
    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Try to request a reasonable camera resolution; for files it is ignored.
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    is_file = not source.isdigit()
    fps_for_ts = src_fps if (is_file and src_fps > 1e-3) else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(
            pose_variant=pose_variant,
            running_mode="video",
            enable_hands=enable_hands,
            delegate=delegate,
        ),
    )

    writer: cv2.VideoWriter | None = None
    if out_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_for_ts, (w, h))

    t0 = time.monotonic()
    frame_count = 0
    printed_done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = pipe.next_timestamp_ms(is_file=is_file, fps_for_ts=fps_for_ts)
        annotated, actions = pipe.annotate(frame, timestamp_ms=timestamp_ms)

        # Overlay labels
        h, w = annotated.shape[:2]
        y = 30
        for a in actions[:5]:
            cv2.putText(annotated, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
            y += 30

        frame_count += 1
        fps = frame_count / max(1e-6, (time.monotonic() - t0))
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(annotated, "Press 'q' to quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if writer is not None:
            writer.write(annotated)
            if is_file and out_path:
                if frame_count != printed_done and (frame_count % 5 == 0 or (total > 0 and frame_count == total)):
                    _print_progress(frame_count, total, t0)
                    printed_done = frame_count

        if show:
            cv2.imshow("MediaPipe (Video/Camera)", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def run_realtime_parallel(
    source: str,
    *,
    show: bool = True,
    out_path: str | None = None,
    pose_variant: str = "full",
    enable_hands: bool = True,
    workers: int = 4,
    delegate: str = "cpu",
) -> None:
    """Realtime camera path that fans inference out across multiple IMAGE-mode pipelines.

    Trade-off vs single-threaded VIDEO mode: much higher throughput on many-core CPUs
    (a single VIDEO pipeline leaves most cores idle), at the cost of temporal
    tracking/smoothing (skeleton is jITTERier). A bounded input queue drops the newest
    frame when workers can't keep up, keeping end-to-end latency in check for live preview.
    """
    cap = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    writer: cv2.VideoWriter | None = None
    if out_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0, (w, h))

    engine = ParallelPoseEngine(
        pipeline_factory=default_pipeline_factory(
            models_dir=models_dir(),
            pose_variant=pose_variant,
            enable_hands=enable_hands,
            delegate=delegate,
        ),
        workers=max(1, int(workers)),
        drop_when_full=True,
    )
    engine.start()

    stop_evt = threading.Event()

    def reader() -> None:
        try:
            while not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                engine.submit(frame)
        finally:
            engine.signal_input_done()

    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()

    t0 = time.monotonic()
    rendered = 0
    try:
        while True:
            res = engine.get(timeout=0.2)
            if res is None:
                if engine.is_drained():
                    break
                continue
            annotated = res.annotated
            h, w = annotated.shape[:2]
            y = 30
            for a in res.actions[:5]:
                cv2.putText(annotated, a, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 240), 2, cv2.LINE_AA)
                y += 30
            rendered += 1
            fps = rendered / max(1e-6, (time.monotonic() - t0))
            cv2.putText(annotated, f"FPS: {fps:.1f} (x{workers})", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(annotated, "Press 'q' to quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("MediaPipe (Realtime Parallel)", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        stop_evt.set()
        engine.close()
        t_reader.join(timeout=2.0)
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    err = engine.take_error()
    if err is not None:
        raise RuntimeError("Realtime parallel inference worker failed") from err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="Camera index like 0/1, or a video file path like demo.mp4")
    p.add_argument("--pose", default="full", choices=["lite", "full", "heavy"], help="Pose model variant")
    p.add_argument("--no-hands", action="store_true", help="Disable hand landmarker for pose-only preview/export")
    p.add_argument("--no-show", action="store_true", help="Disable preview window (useful for batch)")
    p.add_argument("--out", default=None, help="Optional output video path, e.g. out.mp4")
    p.add_argument(
        "--realtime-latest-frame",
        action="store_true",
        help="Opt-in smoke mode: capture/infer/render use latest-frame queues and print latency metrics.",
    )
    p.add_argument("--limit-frames", type=int, default=None, help="Optional frame cap for smoke modes")
    p.add_argument(
        "--delegate",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Explicit MediaPipe delegate opt-in. Defaults to cpu; gpu falls back to cpu if unavailable.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads (>=1). For >1, uses per-frame IMAGE mode (better throughput on "
        "many-core CPUs, less temporal smoothing). Works for both offline videos and live cameras; "
        "the camera path drops frames under load to keep latency bounded.",
    )
    args = p.parse_args()

    if args.realtime_latest_frame:
        metrics = run_realtime_latest_frame_smoke(
            args.source,
            show=not args.no_show,
            pose_variant=args.pose,
            enable_hands=not args.no_hands,
            delegate=args.delegate,
            limit_frames=args.limit_frames,
        )
        print(_metrics_to_dict(metrics))
        return

    # Route to multithreaded path only for offline videos.
    if (not args.source.isdigit()) and args.workers and args.workers > 1:
        cap = _open_capture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        _process_video_multithread(
            cap=cap,
            out_path=args.out,
            show=not args.no_show,
            pose_variant=args.pose,
            enable_hands=not args.no_hands,
            workers=max(1, int(args.workers)),
            delegate=args.delegate,
        )
    elif args.source.isdigit() and args.workers and args.workers > 1:
        # Realtime camera with multi-core parallel inference (IMAGE mode, drops frames
        # under load to keep latency bounded). Trades temporal smoothing for throughput.
        run_realtime_parallel(
            args.source,
            show=not args.no_show,
            out_path=args.out,
            pose_variant=args.pose,
            enable_hands=not args.no_hands,
            workers=max(1, int(args.workers)),
            delegate=args.delegate,
        )
    else:
        run(
            args.source,
            show=not args.no_show,
            out_path=args.out,
            pose_variant=args.pose,
            enable_hands=not args.no_hands,
            delegate=args.delegate,
        )


if __name__ == "__main__":
    main()
