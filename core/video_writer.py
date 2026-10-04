from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Event

import cv2

FFMPEG_TRANSCODE_TIMEOUT_S = 30 * 60
_FFMPEG_POLL_INTERVAL_S = 0.05
_FFMPEG_TERMINATE_TIMEOUT_S = 1.0


def _fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def open_video_writer(preferred_path: Path, *, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path, str]:
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


def _run_ffmpeg_cancellable(command: list[str], stop_evt: Event) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    deadline = time.monotonic() + FFMPEG_TRANSCODE_TIMEOUT_S
    while True:
        if stop_evt.is_set():
            _terminate_process(process)
            raise InterruptedError("视频转码已取消")

        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            return

        if time.monotonic() >= deadline:
            _terminate_process(process)
            raise subprocess.TimeoutExpired(command, FFMPEG_TRANSCODE_TIMEOUT_S)
        stop_evt.wait(_FFMPEG_POLL_INTERVAL_S)


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

    bundled_ffmpeg = list((Path(getattr(sys, "_MEIPASS", "")) / "ffmpeg").glob("*.exe")) if getattr(sys, "frozen", False) else []
    command = [
        str(bundled_ffmpeg[0]) if bundled_ffmpeg else "ffmpeg",
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
            subprocess.run(command, check=True, timeout=FFMPEG_TRANSCODE_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        else:
            _run_ffmpeg_cancellable(command, stop_evt)
            if stop_evt.is_set():
                raise InterruptedError("视频转码已取消")
        if not temporary_dst.is_file() or temporary_dst.stat().st_size <= 0:
            raise OSError("ffmpeg produced no output")
        temporary_dst.replace(dst)
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
