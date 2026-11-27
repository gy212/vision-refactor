"""视频写出：open_video_writer 编码回退（离线导出与视频文件会话用；摄像头录像用 ``mjpeg_writer``）、MJPG AVI 包级无损裁剪与 ffmpeg H.264 转码。"""

from __future__ import annotations

import math
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event

import cv2

from core.foundation.atomic_io import replace_file
from core.media.ffmpeg import NO_WINDOW, ffmpeg_executable
from core.media.video_source import first_frame_ok, probe_video

FFMPEG_TRANSCODE_TIMEOUT_S = 30 * 60
_FFMPEG_POLL_INTERVAL_S = 0.05
_FFMPEG_TERMINATE_TIMEOUT_S = 1.0


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def open_video_writer(
    preferred_path: Path, *, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, Path, str]:
    """
    Open a VideoWriter with best-effort codec/container compatibility.

    Returns: (writer, actual_path, codec_tag)
    """
    w, h = int(size[0]), int(size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid frame size: {size}")

    fps = float(fps) if fps and fps > 1e-6 else 30.0
    p = Path(preferred_path)
    suf = p.suffix.lower()

    candidates: list[tuple[Path, str]] = []
    if suf == ".mp4":
        # H.264 is widely supported, but may be unavailable in some OpenCV builds.
        # Keep every non-H.264 fallback in AVI so it can be transcoded reliably.
        avi_path = p.with_suffix(".avi")
        candidates = [
            (p, "avc1"),
            (p, "H264"),
            (avi_path, "MJPG"),
            (avi_path, "XVID"),
        ]
    elif suf == ".avi":
        candidates = [(p, "MJPG"), (p, "XVID")]
    else:
        # Unknown containers are not safe codec signals; normalize fallbacks to AVI.
        avi_path = p.with_suffix(".avi")
        candidates = [(avi_path, "MJPG"), (avi_path, "XVID")]

    last_err: str | None = None
    for out_path, codec in candidates:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(out_path), _fourcc(codec), fps, (w, h))
        if vw.isOpened():
            return vw, out_path, codec
        vw.release()
        last_err = f"VideoWriter open failed: path={out_path}, codec={codec}"

    raise RuntimeError(last_err or "VideoWriter open failed")


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=_FFMPEG_TERMINATE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=_FFMPEG_TERMINATE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _is_cancelled(
    stop_evt: Event | None,
    cancel_check: Callable[[], bool] | None,
) -> bool:
    if stop_evt is not None and stop_evt.is_set():
        return True
    return cancel_check is not None and bool(cancel_check())


def _discard_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _run_ffmpeg_cancellable(
    command: list[str],
    stop_evt: Event | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
    timeout_s: float = FFMPEG_TRANSCODE_TIMEOUT_S,
    cancel_message: str = "视频转码已取消",
) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            if _is_cancelled(stop_evt, cancel_check):
                _terminate_process(process)
                raise InterruptedError(cancel_message)

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                return

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process(process)
                raise subprocess.TimeoutExpired(command, timeout_s)
            wait_s = min(_FFMPEG_POLL_INTERVAL_S, remaining)
            if stop_evt is not None:
                stop_evt.wait(wait_s)
            else:
                try:
                    process.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    pass
    except BaseException:
        _terminate_process(process)
        raise


def _fourcc_code(value: float | int) -> str:
    fourcc_int = int(value) if value and math.isfinite(float(value)) else 0
    return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))


def _probe_mjpeg_avi(path: Path) -> tuple[float, int] | None:
    """探测是否为带有效帧率的全关键帧 MJPG AVI；返回 ``(fps, frame_count)``，否则返回 None。"""
    if not path.is_file() or path.suffix.lower() != ".avi":
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        if _fourcc_code(cap.get(cv2.CAP_PROP_FOURCC)).upper() != "MJPG":
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0.0:
            return None
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        frame_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else 0
        return fps, frame_count
    finally:
        cap.release()


def is_mjpeg_avi(path: Path | str) -> bool:
    """判断路径是否为带有效帧率的全关键帧 MJPG AVI。"""
    return _probe_mjpeg_avi(Path(path)) is not None


def _validate_frame_range(start_frame: int, end_frame: int) -> tuple[int, int]:
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise TypeError(f"start_frame 必须为整数：{start_frame!r}")
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise TypeError(f"end_frame 必须为整数：{end_frame!r}")
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"裁剪帧区间无效：[{start_frame}, {end_frame})")
    return start_frame, end_frame


def _verify_trimmed_avi(path: Path) -> int:
    """校验无损裁剪输出的 AVI 非空、首帧可读且含有效帧数，返回实际帧数。"""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg 裁剪未产出有效文件：{path}")
    if not first_frame_ok(path):
        raise RuntimeError(f"裁剪输出首帧不可解码：{path}")
    written = probe_video(path).frame_count
    if written <= 0:
        raise RuntimeError(f"裁剪输出帧数无效：{path}")
    return written


def trim_mjpeg_avi(
    src: Path | str,
    dst: Path | str,
    *,
    start_frame: int,
    end_frame: int,
    stop_evt: Event | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """对全关键帧 MJPG AVI 按精确帧区间 ``[start_frame, end_frame)`` 执行包级无损流拷贝裁剪。

    复用随包 ffmpeg（``-c:v copy -an``）先写入同目录临时 AVI，校验非空且首帧可解码后原子替换 ``dst``。
    返回实际写出帧数；响应 ``stop_evt`` / ``cancel_check`` 取消（抛 ``InterruptedError`` 并清理临时文件）。
    """
    start_frame, end_frame = _validate_frame_range(start_frame, end_frame)
    src_path = Path(src)
    dst_path = Path(dst)
    if dst_path.suffix.lower() != ".avi":
        raise ValueError(f"MJPG 无损裁剪输出路径必须为 .avi：{dst_path}")
    if dst_path.is_dir():
        raise IsADirectoryError(f"裁剪输出路径是目录：{dst_path}")

    probed = _probe_mjpeg_avi(src_path)
    if probed is None:
        raise ValueError(f"源视频不是有效的 MJPG AVI：{src_path}")
    fps, src_frames = probed
    if src_frames > 0 and start_frame >= src_frames:
        raise ValueError(f"裁剪起始帧越界：start={start_frame}, total={src_frames}")
    effective_end = min(end_frame, src_frames) if src_frames > 0 else end_frame
    target_frames = effective_end - start_frame
    if target_frames <= 0:
        raise ValueError(f"裁剪帧区间为空：[{start_frame}, {effective_end})")

    if _is_cancelled(stop_evt, cancel_check):
        raise InterruptedError("视频裁剪已取消")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{dst_path.stem}.",
        suffix=".tmp.avi",
        dir=dst_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_dst = Path(temporary_file.name)

    command = [ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error"]
    if start_frame > 0:
        command.extend(["-ss", f"{start_frame / fps:.9f}"])
    command.extend(
        [
            "-i",
            str(src_path),
            "-frames:v",
            str(target_frames),
            "-c:v",
            "copy",
            "-an",
            "-f",
            "avi",
            str(temporary_dst),
        ]
    )

    try:
        if stop_evt is None and cancel_check is None:
            subprocess.run(
                command,
                check=True,
                timeout=FFMPEG_TRANSCODE_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            )
        else:
            _run_ffmpeg_cancellable(
                command,
                stop_evt,
                cancel_check=cancel_check,
                cancel_message="视频裁剪已取消",
            )
            if _is_cancelled(stop_evt, cancel_check):
                raise InterruptedError("视频裁剪已取消")
        written = _verify_trimmed_avi(temporary_dst)
        if _is_cancelled(stop_evt, cancel_check):
            raise InterruptedError("视频裁剪已取消")
        replace_file(temporary_dst, dst_path)
    except BaseException:
        _discard_temp(temporary_dst)
        raise
    return written


def transcode_to_h264(
    src: Path | None,
    *,
    stop_evt: Event | None = None,
) -> Path | None:
    """Transcode an AVI recording to H.264 MP4 without risking the source."""
    if src is None:
        return None

    src = Path(src)
    if not src.exists() or src.suffix.lower() != ".avi":
        return src
    if stop_evt is not None and stop_evt.is_set():
        raise InterruptedError("视频转码已取消")

    dst = src.with_suffix(".mp4")
    if dst == src:
        return src

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{dst.stem}.",
            suffix=".tmp.mp4",
            dir=dst.parent,
            delete=False,
        ) as temporary_file:
            temporary_dst = Path(temporary_file.name)
    except OSError:
        return src

    command = [
        ffmpeg_executable(),
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(temporary_dst),
    ]
    try:
        if stop_evt is None:
            subprocess.run(
                command,
                check=True,
                timeout=FFMPEG_TRANSCODE_TIMEOUT_S,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW,
            )
        else:
            _run_ffmpeg_cancellable(command, stop_evt)
            if stop_evt.is_set():
                raise InterruptedError("视频转码已取消")
        if not temporary_dst.is_file() or temporary_dst.stat().st_size <= 0:
            raise OSError("ffmpeg produced no output")
        replace_file(temporary_dst, dst)
    except InterruptedError:
        try:
            temporary_dst.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        try:
            temporary_dst.unlink(missing_ok=True)
        except OSError:
            pass
        return src

    try:
        src.unlink()
    except OSError:
        pass
    return dst
