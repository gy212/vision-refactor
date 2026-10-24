"""``main`` CLI：摄像头或视频文件的实时骨架预览与导出，逐帧处理都经 ``core.live.SessionRunner``。

- 默认：顺序 VIDEO 推理；``--workers`` > 1 时并行推理（摄像头满则丢帧，视频保序）。
- ``--realtime-latest-frame``：采集与推理分线程、推理只取最新帧，结束时打印延迟与吞吐指标（冒烟用）。
- 导出用 mp4v 编码写到 ``--out``：视频文件逐帧写叠加帧；摄像头先按真实时间录原始画面，会话结束后
  再画骨架写成 ``--out``（D-26）。按 ``q`` 退出预览窗口。

Run (camera):  .venv\\Scripts\\python.exe apps/main.py
Run (video):   .venv\\Scripts\\python.exe apps/main.py --source demo.mp4 --out out.mp4
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from core.foundation import CancelToken
from core.live import (
    CameraInput,
    FrameInput,
    FramePacket,
    SessionEvent,
    SessionOutcome,
    SessionRunner,
    VideoFileInput,
    build_stages,
    camera_session,
    video_session,
)
from core.media import PRIMARY, CameraService, CaptureSpec, capture_clock_ns, first_frame_ok, open_recording_writer
from core.pose import EngineSpec, annotate_video
from core.recording import SegmentMeta, SegmentRecorder, skeleton_engine

WINDOW_TITLE = "MediaPipe (Video/Camera)"
PARALLEL_TITLE = "MediaPipe (Realtime Parallel)"
SMOKE_TITLE = "MediaPipe (Latest Frame Smoke)"


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


class _ServiceCameraInput(CameraInput):
    """CLI 自带一个 ``CameraService``：会话结束时连同服务一起关闭。"""

    def __init__(self, service: CameraService, spec: CaptureSpec) -> None:
        super().__init__(service.open_stream({PRIMARY: spec}))
        self._service = service

    def close(self, *, keep_warm: bool) -> None:
        try:
            super().close(keep_warm=False)
        finally:
            self._service.close()


def open_input(source: str) -> FrameInput:
    """数字是摄像头编号（只经 ``CameraService``），否则是视频文件路径。"""
    if source.isdigit():
        return _ServiceCameraInput(CameraService(), CaptureSpec(int(source)))
    return VideoFileInput(source)


def _open_mp4v(path: Path, *, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, str]:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), tuple(size))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法写出视频：{path}")
    return writer, Path(path), "mp4v"


class CliExport:
    """CLI 的 ``--out`` 导出；``path`` 为空时什么都不做。

    - 视频文件：会话线程逐帧写主路叠加帧（mp4v）。
    - 摄像头：录制层的录像线程按真实时间把原始画面录到 ``--out`` 旁的临时文件；会话结束后 ``finish``
      再画骨架写成 ``--out``，成片校验可读后才删除临时文件，失败时保留它（D-26）。
    """

    def __init__(self, path: str | None, engine: EngineSpec) -> None:
        self._path = Path(path) if path else None
        self._engine = engine
        self._writer: cv2.VideoWriter | None = None
        self._recorder: SegmentRecorder | None = None
        self._raw: Path | None = None
        self.error: str | None = None

    def begin_session(self, sizes: Mapping[str, tuple[int, int]], fps: float, *, feed=None,
                      rotation_of: Callable[[str], int] | None = None) -> None:
        if self._path is None:
            return
        if feed is None:
            self._writer, _path, _codec = _open_mp4v(self._path, fps=fps, size=tuple(sizes["front"]))
            return
        temporary = self._path.with_name(f".{self._path.stem}.raw.avi")
        self._recorder = SegmentRecorder(lambda _target, rate, size, rotation: open_recording_writer(
            temporary, fps=rate, size=size, rotation=rotation))
        self._recorder.begin_session(sizes, fps, feed=feed, rotation_of=rotation_of)
        if not self._recorder.start_segment(SegmentMeta(self._path.parent), self._confirmed):
            raise RuntimeError("无法开始录制摄像头画面")

    def _confirmed(self, _segment, error: str | None) -> None:
        """开录确认（录制的任务线程）：没能写出首帧时录制端已结束这一段，记下原因。"""
        if error is not None:
            self.error = f"摄像头录像没有开始：{error}"

    def write(self, frames: Mapping[str, np.ndarray]) -> None:
        if self._writer is not None:
            self._writer.write(frames["front"])

    def close_session(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._recorder is not None:
            finished = self._recorder.close_session()
            self._recorder = None
            view = None if finished is None else finished.views.get("front")
            if self.error is not None:
                return
            if view is None or view.path is None or view.frames <= 0:
                self.error = "摄像头录像没有写出画面"
            elif view.error:
                self.error = f"摄像头录像失败：{view.error}（原始录像保留在 {view.path}）"
            else:
                self._raw = view.path

    def finish(self, progress: bool = True) -> None:
        """摄像头导出：会话结束后把原始录像画上骨架写成 ``--out``。"""
        raw, self._raw = self._raw, None
        if raw is None or self._path is None:
            return

        def report(done: int, total: int) -> None:
            if progress:
                print(f"\r骨架成片 {done}/{total}", end="", flush=True)

        try:
            path, _codec, frames = annotate_video(raw, self._path, self._engine, open_writer=_open_mp4v,
                                                  progress=report)
        except Exception as exc:
            self.error = f"骨架成片失败（原始录像保留在 {raw}）：{exc}"
            return
        finally:
            if progress:
                print()
        if frames <= 0 or not first_frame_ok(path):
            self.error = f"骨架成片不可读：{path}（原始录像保留在 {raw}）"
            return
        raw.unlink(missing_ok=True)


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


def _window(title: str, show: bool, on_frame=None):
    """预览窗口回调：显示主路叠加帧，按 q 返回 False 结束会话。"""

    def display(packet: FramePacket) -> bool:
        if on_frame is not None:
            on_frame(packet)
        if not show:
            return True
        cv2.imshow(title, packet.out[PRIMARY])
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    return display


def _run_session(config, source: str, *, out_path: str | None, display, progress: bool) -> tuple[SessionOutcome, SessionRunner]:
    """在当前线程跑完一个会话；窗口在结束时关闭。"""
    t0 = time.monotonic()

    def on_event(event: SessionEvent) -> None:
        if progress and event.kind == "progress":
            done, total = event.payload
            if total > 0 and done > 0:
                _print_progress(done, total, t0)

    pose, stages = build_stages(config)
    export = CliExport(out_path, skeleton_engine(config.pose_variant, config.enable_hands))
    runner = SessionRunner(
        config, open_input(source), pose, stages, export, None, on_event, 1, CancelToken(),
        display=display,
    )
    try:
        outcome = runner.run()
    finally:
        cv2.destroyAllWindows()
    if not outcome.failed:
        export.finish()
    if export.error is not None and not outcome.failed:
        outcome = replace(outcome, outcome="failed", status=export.error, error=RuntimeError(export.error))
    return outcome, runner


def _raise_on_failure(outcome: SessionOutcome, message: str | None = None) -> None:
    if outcome.failed:
        raise RuntimeError(message or outcome.status or "session failed") from outcome.error


def run(
    source: str,
    *,
    show: bool = True,
    out_path: str | None = None,
    pose_variant: str = "full",
    enable_hands: bool = True,
    delegate: str = "cpu",
) -> None:
    """顺序 VIDEO 推理的预览/导出。"""
    if source.isdigit():
        config = camera_session(
            {PRIMARY: CaptureSpec(int(source))}, pose_variant=pose_variant, enable_hands=enable_hands,
            skeleton=True, online_match=False, smoothing=False, loading_raw_preview=False,
            hud="cli", delegate=delegate,
        )
    else:
        config = video_session(source, pose_variant=pose_variant, enable_hands=enable_hands, hud="cli",
                               delegate=delegate)
    outcome, _ = _run_session(
        config, source, out_path=out_path, display=_window(WINDOW_TITLE, show),
        progress=bool(out_path) and not source.isdigit(),
    )
    _raise_on_failure(outcome)


def run_parallel(
    source: str,
    *,
    show: bool = True,
    out_path: str | None = None,
    pose_variant: str = "full",
    enable_hands: bool = True,
    workers: int = 4,
    delegate: str = "cpu",
) -> None:
    """多 worker 并行推理（IMAGE 模式）：摄像头满则丢帧控制延迟，视频按帧序导出。"""
    workers = max(1, int(workers))
    if source.isdigit():
        config = camera_session(
            {PRIMARY: CaptureSpec(int(source))}, pose_variant=pose_variant, enable_hands=enable_hands,
            workers=max(2, workers), skeleton=True, online_match=False, smoothing=False, hud="cli",
            delegate=delegate,
        )
        title = PARALLEL_TITLE
    else:
        config = video_session(source, pose_variant=pose_variant, enable_hands=enable_hands,
                               workers=max(2, workers), hud="cli", delegate=delegate)
        title = WINDOW_TITLE
    outcome, _ = _run_session(
        config, source, out_path=out_path, display=_window(title, show),
        progress=bool(out_path) and not source.isdigit(),
    )
    _raise_on_failure(outcome, "Realtime parallel inference worker failed" if source.isdigit() else None)


def run_realtime_latest_frame_smoke(
    source: str,
    *,
    show: bool = True,
    pose_variant: str = "full",
    enable_hands: bool = True,
    delegate: str = "cpu",
    limit_frames: int | None = None,
) -> RealtimeLatestFrameMetrics:
    """latest-frame 冒烟：采集与推理分线程，推理只取最新帧；返回吞吐与延迟指标。"""
    if source.isdigit():
        config = camera_session(
            {PRIMARY: CaptureSpec(int(source))}, pose_variant=pose_variant, enable_hands=enable_hands,
            skeleton=True, online_match=False, smoothing=False, loading_raw_preview=False, hud="cli",
            delegate=delegate,
        )
    else:
        config = video_session(source, pose_variant=pose_variant, enable_hands=enable_hands, hud="cli",
                               delegate=delegate)
    config = replace(config, latest_frame=True, hud_hint=False, hud_title="Latest-frame smoke FPS",
                   limit_frames=None if limit_frames is None else max(0, int(limit_frames)))
    latency: list[float] = []
    rendered: list[float] = []

    def on_frame(packet: FramePacket) -> None:
        rendered.append(time.monotonic())
        latency.append((capture_clock_ns() - packet.captured_at_ns) / 1e9)

    outcome, runner = _run_session(
        config, source, out_path=None, display=_window(SMOKE_TITLE, show, on_frame), progress=False,
    )
    _raise_on_failure(outcome, "Latest-frame smoke worker failed")
    stats = runner.stats
    capture_s = stats["capture_ended"] - stats["capture_started"]
    render_s = rendered[-1] - rendered[0] if rendered else 0.0
    return RealtimeLatestFrameMetrics(
        captured_frames=int(stats["captured"]),
        processed_frames=outcome.frames,
        rendered_frames=len(rendered),
        dropped_frames=int(stats["dropped"]),
        capture_fps=_rate(int(stats["captured"]), capture_s),
        infer_fps=_rate(outcome.frames, render_s),
        render_fps=_rate(len(rendered), render_s),
        latency_p90_ms=_round_ms(_percentile(latency, 90)),
    )


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

    if args.workers and args.workers > 1:
        run_parallel(
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
