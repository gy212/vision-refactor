# -*- coding: utf-8 -*-
"""摄像头录像专用的写出 ``MjpegPipeWriter``：每帧在调用线程按本段方向转正、用 libjpeg-turbo（``cv2.imencode``）编成 JPEG，
交给 ffmpeg 子进程原样封装（``-c:v copy``）成 MJPG AVI；录完照常转成 H.264 MP4（D-31，设计文档 §8.8）。

- 接口与 ``open_video_writer`` 的 writer 相同（``write``/``release``），``open_recording_writer`` 返回
  ``(writer, 实际路径, "MJPG")``；``write`` 收未转正的源帧，按打开时给的旋转转正（D-29：一段之内方向不变）。
- 连续传入同一个数组对象（时间轴补帧）时复用上一帧的 JPEG，不再转正、编码。
- JPEG 由一个任务线程经有界队列（``FEED_QUEUE_FRAMES`` 帧）送进 ffmpeg 的 stdin，录像线程不会卡在管道上；
  队列满时 ``write`` 最多等 ``FEED_STALL_S``。ffmpeg 退出、管道断开、超过这段时间仍不接收或画面尺寸与本段不符时，
  ``write`` 抛出写明原因的 ``RecordingWriterError``，之后的 ``write`` 都抛同一个原因。
- ``release`` 最多用 ``RELEASE_TIMEOUT_S``：送完队列、关闭 stdin、等 ffmpeg 写完 AVI 索引；超时就结束进程并报错，
  ffmpeg 退出码非 0 也报错（附 ffmpeg 的错误输出）。一帧都没写时删除输出文件。重复调用不再做任何事。
- ffmpeg 子进程经注入的 ``spawn`` 启动（默认 ``subprocess.Popen``），测试换成假进程。
"""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Protocol

import cv2
import numpy as np

from core.foundation.tasks import CancelToken, TaskGroup
from core.media.ffmpeg import NO_WINDOW, ffmpeg_executable
from core.media.transforms import apply_rotation

CODEC = "MJPG"
JPEG_QUALITY = 85
FEED_QUEUE_FRAMES = 30       # 约 1 秒的 JPEG（1080p 约 0.1～0.3 MB/帧），内存有界
FEED_STALL_S = 1.0           # 队列满时 ``write`` 最多等这么久（开录积压一次写出、ffmpeg 刚启动时会短暂排满）
RELEASE_TIMEOUT_S = 2.0      # 小于录像线程结束一段的等待上限（``STOP_WAIT_S``）
_STDERR_TAIL_CHARS = 400


class RecordingWriterError(RuntimeError):
    """录像写出失败；消息直接给用户看。"""


class PipeProcess(Protocol):
    """``subprocess.Popen`` 中本模块用到的部分。"""

    stdin: Any

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


Spawn = Callable[[list[str], IO[bytes]], PipeProcess]   # (命令, 错误输出文件) → 进程


def _popen(command: list[str], stderr: IO[bytes]) -> PipeProcess:
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr,
                            creationflags=NO_WINDOW)


def mux_command(executable: str, fps: float, out_path: Path) -> list[str]:
    """把 stdin 上的 JPEG 流按 ``fps`` 原样封装成 AVI 的 ffmpeg 命令。输入用 ``image2pipe``：
    裸 ``mjpeg`` 输入格式不认 ``-framerate``，AVI 头会写成 600 fps。"""
    return [executable, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:.6g}",
            "-c:v", "mjpeg", "-i", "pipe:0", "-c:v", "copy", "-y", str(out_path)]


class MjpegPipeWriter:
    """一段录像的 writer；``write`` 只在一个线程里调用，``release`` 可在任意线程调用。"""

    def __init__(self, process: PipeProcess, stderr: IO[bytes], path: Path, *, size: tuple[int, int],
                 rotation: int, quality: int = JPEG_QUALITY) -> None:
        self.path = path
        self._process = process
        self._stderr = stderr
        self._size = (int(size[0]), int(size[1]))
        self._rotation = int(rotation)
        self._params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        self._cond = threading.Condition()
        self._pending: deque[np.ndarray] = deque()
        self._closed = False
        self._error: str | None = None
        self._released = False
        self._last: tuple[np.ndarray, np.ndarray] | None = None   # (源帧, 它的 JPEG)
        self.frames = 0
        self._info: dict[str, Any] = {"backend": "ffmpeg_pipe", "encoder": "cv2.imencode", "quality": int(quality),
                                      "rotation": self._rotation, "feed_peak": 0}
        self._tasks = TaskGroup(f"mjpeg-{path.stem}")
        self._feeder = self._tasks.spawn("feed", self._feed, owner="recording", close_budget_s=RELEASE_TIMEOUT_S)

    @property
    def info(self) -> dict[str, Any]:
        """写进 ``capture.json`` 的 writer 说明：封装方式、编码器、质量、旋转与送入队列的峰值。"""
        with self._cond:
            return dict(self._info)

    # ------------------------------------------------------------------ 写
    def write(self, frame: np.ndarray) -> None:
        """转正、编码一帧并排进送往 ffmpeg 的队列；失败抛 ``RecordingWriterError``。"""
        if self._error is not None:
            raise RecordingWriterError(self._error)
        if self._released:
            raise RecordingWriterError("录像文件已经关闭")
        data = self._encode(frame)
        if self._process.poll() is not None:
            self._fail_with(self._exited_text())
        with self._cond:
            room = self._cond.wait_for(lambda: len(self._pending) < FEED_QUEUE_FRAMES or self._error is not None,
                                       FEED_STALL_S)
            if self._error is None and not room:
                self._error = f"ffmpeg {FEED_STALL_S:g} 秒没有接收画面（已积压 {FEED_QUEUE_FRAMES} 帧）"
            if self._error is not None:
                raise RecordingWriterError(self._error)
            self._pending.append(data)
            self._info["feed_peak"] = max(self._info["feed_peak"], len(self._pending))
            self._cond.notify()
        self.frames += 1

    def _encode(self, frame: np.ndarray) -> np.ndarray:
        last = self._last
        if last is not None and last[0] is frame:
            return last[1]
        image = apply_rotation(frame, self._rotation)
        height, width = image.shape[:2]
        if (width, height) != self._size:
            self._fail_with(f"画面尺寸 {width}×{height} 与本段录像的 {self._size[0]}×{self._size[1]} 不符")
        ok, data = cv2.imencode(".jpg", image, self._params)
        if not ok:
            self._fail_with("JPEG 编码失败")
        self._last = (frame, data)
        return data

    def _fail_with(self, message: str) -> None:
        with self._cond:
            if self._error is None:
                self._error = message
            message = self._error
        raise RecordingWriterError(message)

    def _feed(self, _token: CancelToken) -> None:
        """任务线程：按序把 JPEG 写进 ffmpeg 的 stdin；关闭后写完剩下的再关 stdin。管道出错时记下原因。"""
        stdin = self._process.stdin
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._pending or self._closed)
                    if not self._pending:
                        break
                    data = self._pending[0]
                stdin.write(memoryview(data))
                with self._cond:
                    self._pending.popleft()
                    self._cond.notify_all()
            stdin.close()
        except (OSError, ValueError) as exc:
            try:
                code = self._process.wait(RELEASE_TIMEOUT_S / 4)
            except subprocess.TimeoutExpired:
                code = None
            text = self._exited_text(code) if code is not None else f"写入 ffmpeg 失败：{exc}"
            with self._cond:
                if self._error is None:
                    self._error = text
                self._pending.clear()
                self._cond.notify_all()

    # ------------------------------------------------------------------ 关闭
    def release(self) -> None:
        """送完排队的帧并等 ffmpeg 写完 AVI（最多 ``RELEASE_TIMEOUT_S``）；失败抛 ``RecordingWriterError``。"""
        with self._cond:
            if self._released:
                return
            self._released = True
            self._closed = True
            self._cond.notify_all()
        deadline = time.monotonic() + RELEASE_TIMEOUT_S
        problem = None
        if not self._feeder.join(max(0.0, deadline - time.monotonic())):
            problem = f"ffmpeg 在 {RELEASE_TIMEOUT_S:g} 秒内没有收完画面，录像文件可能不完整"
        code = self._wait_exit(deadline) if problem is None else None
        if problem is None and code is None:
            problem = f"ffmpeg 在 {RELEASE_TIMEOUT_S:g} 秒内没有写完录像文件，录像文件可能不完整"
        if problem is not None:
            self._kill()
        elif code != 0:
            problem = self._error or self._exited_text(code)
        self._stderr.close()
        if self.frames == 0:
            self.path.unlink(missing_ok=True)
        if problem is not None:
            with self._cond:
                self._error = self._error or problem
            raise RecordingWriterError(problem)

    def _wait_exit(self, deadline: float) -> int | None:
        try:
            return self._process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return None

    def _kill(self) -> None:
        try:
            self._process.kill()
            self._process.wait(RELEASE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._feeder.join(RELEASE_TIMEOUT_S)

    def _exited_text(self, code: int | None = None) -> str:
        code = self._process.poll() if code is None else code
        detail = self._stderr_tail()
        text = "ffmpeg 已退出" if code is None else f"ffmpeg 已退出（退出码 {code}）"
        return f"{text}：{detail}" if detail else text

    def _stderr_tail(self) -> str:
        """ffmpeg 错误输出的末尾；进程还在运行时不读（错误输出文件与子进程共用读写位置）。"""
        if self._process.poll() is None:
            return ""
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            text = self._stderr.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""
        return " ".join(text[-_STDERR_TAIL_CHARS:].split())


def open_recording_writer(preferred_path: Path, *, fps: float, size: tuple[int, int], rotation: int = 0,
                          spawn: Spawn | None = None, quality: int = JPEG_QUALITY) -> tuple[MjpegPipeWriter, Path, str]:
    """启动 ffmpeg 封装进程并返回 ``(writer, 实际路径, "MJPG")``；实际路径是 ``preferred_path`` 换成 ``.avi``。
    ``size`` 是转正后的宽高；ffmpeg 启动不了时抛 ``RecordingWriterError``。"""
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size: {size}")
    fps = float(fps) if fps and fps > 1e-6 else 30.0
    out_path = Path(preferred_path).with_suffix(".avi")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    executable = ffmpeg_executable()
    stderr = tempfile.TemporaryFile()
    try:
        process = (spawn or _popen)(mux_command(executable, fps, out_path), stderr)
    except OSError as exc:
        stderr.close()
        raise RecordingWriterError(f"无法启动 ffmpeg（{executable}）：{exc}") from exc
    return MjpegPipeWriter(process, stderr, out_path, size=(width, height), rotation=rotation,
                           quality=quality), out_path, CODEC


__all__ = [
    "CODEC",
    "FEED_QUEUE_FRAMES",
    "FEED_STALL_S",
    "JPEG_QUALITY",
    "MjpegPipeWriter",
    "PipeProcess",
    "RELEASE_TIMEOUT_S",
    "RecordingWriterError",
    "Spawn",
    "mux_command",
    "open_recording_writer",
]
