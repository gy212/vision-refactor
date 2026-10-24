# -*- coding: utf-8 -*-
"""摄像头录像 writer（``core.media.mjpeg_writer``，D-31）：用注入的假 ffmpeg 进程检查写出顺序、转正、补帧复用编码、
尺寸不符、ffmpeg 中途退出、管道断开、ffmpeg 不再接收、关闭（退出码、超时、幂等、没写帧时删文件）与启动失败；
本机找得到 ffmpeg 时再用真 ffmpeg 封装一段，解码检查帧数、帧率与方向（CI 没有 ffmpeg，跳过）。"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.media import RecordingWriterError, ffmpeg_executable, open_recording_writer
from core.media import mjpeg_writer as mw


class FakeStdin:
    """收下写入的字节；``block`` 置位前写入会卡住（ffmpeg 不再读），``broken`` 时抛 BrokenPipeError。"""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False
        self.block = threading.Event()
        self.block.set()
        self.broken = False

    def write(self, data) -> int:
        self.block.wait()
        if self.broken:
            raise BrokenPipeError(32, "Broken pipe")
        self.chunks.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """假 ffmpeg：stdin 关闭后以 ``exit_code`` 退出（``hang`` 时不退出，只能被 kill）；``die`` 立即以给定码退出。"""

    def __init__(self, stderr, *, exit_code: int = 0, hang: bool = False) -> None:
        self.stdin = FakeStdin()
        self.stderr = stderr
        self.exit_code = exit_code
        self.hang = hang
        self.returncode: int | None = None
        self.killed = False

    def die(self, code: int, message: bytes = b"") -> None:
        self.stderr.write(message)
        self.returncode = code
        self.stdin.broken = True

    def poll(self) -> int | None:
        if self.returncode is None and self.stdin.closed and not self.hang:
            self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout=None) -> int:
        deadline = time.monotonic() + (timeout if timeout is not None else 60.0)
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            time.sleep(0.005)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdin.broken = True
        self.stdin.block.set()


class Spawner:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.commands: list[list[str]] = []
        self.process: FakeProcess | None = None

    def __call__(self, command, stderr):
        self.commands.append(command)
        self.process = FakeProcess(stderr, **self.kwargs)
        return self.process


def _frame(value: int, shape=(48, 64)) -> np.ndarray:
    image = np.full((*shape, 3), value, np.uint8)
    image[:8, :8] = 255          # 左上角的白块：转正后应在右上角
    return image


def _jpegs(process: FakeProcess) -> list[np.ndarray]:
    return [cv2.imdecode(np.frombuffer(chunk, np.uint8), cv2.IMREAD_COLOR) for chunk in process.stdin.chunks]


def _open(tmp_path, spawner, *, rotation=90, size=(48, 64)):
    return open_recording_writer(tmp_path / "front.mp4", fps=30.0, size=size, rotation=rotation, spawn=spawner)


def test_frames_are_rotated_encoded_and_fed_in_order(tmp_path):
    spawner = Spawner()
    writer, path, codec = _open(tmp_path, spawner)
    assert path == tmp_path / "front.avi" and codec == "MJPG"
    command = spawner.commands[0]
    assert command[command.index("-f") + 1] == "image2pipe" and command[command.index("-framerate") + 1] == "30"
    assert command[-1] == str(path) and command[command.index("-c:v", command.index("-i")) + 1] == "copy"
    frames = [_frame(value) for value in (20, 60, 100, 140, 180)]
    for frame in frames:
        writer.write(frame)
    writer.release()
    decoded = _jpegs(spawner.process)
    assert len(decoded) == 5 and spawner.process.stdin.closed
    assert [int(round(image[30:, :40].mean())) for image in decoded] == pytest.approx([20, 60, 100, 140, 180], abs=3)
    assert all(image.shape[:2] == (64, 48) for image in decoded)                 # 转正 90°：宽 48、高 64
    assert all(image[:8, -8:].mean() > 200 and image[:8, :8].mean() < 200 for image in decoded)
    assert writer.frames == 5 and writer.info["backend"] == "ffmpeg_pipe" and writer.info["rotation"] == 90


def test_repeated_frame_object_reuses_its_encoding(tmp_path, monkeypatch):
    """时间轴补帧时同一个数组对象连续写两次：只编码一次，两帧字节相同。"""
    spawner = Spawner()
    writer, _path, _codec = _open(tmp_path, spawner, rotation=0, size=(64, 48))
    calls = []
    real = cv2.imencode
    monkeypatch.setattr(mw.cv2, "imencode", lambda *args: calls.append(1) or real(*args))
    frame = _frame(90)
    writer.write(frame)
    writer.write(frame)
    writer.write(_frame(90))     # 内容相同但是另一个对象：重新编码
    writer.release()
    chunks = spawner.process.stdin.chunks
    assert len(calls) == 2 and len(chunks) == 3 and chunks[0] == chunks[1]


def test_wrong_frame_size_fails_and_stays_failed(tmp_path):
    writer, _path, _codec = _open(tmp_path, Spawner(), rotation=0, size=(64, 48))
    with pytest.raises(RecordingWriterError, match="与本段录像的 64×48 不符"):
        writer.write(_frame(10, shape=(64, 48)))
    with pytest.raises(RecordingWriterError, match="不符"):
        writer.write(_frame(10))
    writer.release()


def test_ffmpeg_exiting_mid_segment_fails_the_next_write_with_its_error_output(tmp_path):
    spawner = Spawner()
    writer, _path, _codec = _open(tmp_path, spawner)
    writer.write(_frame(10))
    spawner.process.die(1, b"[avi @ 0x1] No space left on device\n")
    with pytest.raises(RecordingWriterError, match=r"ffmpeg 已退出（退出码 1）：.*No space left on device"):
        writer.write(_frame(20))
    with pytest.raises(RecordingWriterError, match="退出码 1"):
        writer.release()


def test_broken_pipe_is_reported_on_the_following_write(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "RELEASE_TIMEOUT_S", 0.3)
    spawner = Spawner()
    writer, _path, _codec = _open(tmp_path, spawner)
    spawner.process.stdin.broken = True
    writer.write(_frame(10))                 # 排进队列成功；送进管道时断开
    with pytest.raises(RecordingWriterError, match="写入 ffmpeg 失败"):
        for _ in range(200):
            writer.write(_frame(20))
            time.sleep(0.01)
    with pytest.raises(RecordingWriterError):
        writer.release()
    assert spawner.process.killed


def test_ffmpeg_that_stops_reading_fails_writes_and_release_is_bounded(tmp_path, monkeypatch):
    """ffmpeg 不再读（管道写入卡住）：队列满后 ``write`` 最多等 ``FEED_STALL_S`` 就报错；``release`` 在上限内结束进程。"""
    monkeypatch.setattr(mw, "FEED_QUEUE_FRAMES", 3)
    monkeypatch.setattr(mw, "FEED_STALL_S", 0.2)
    monkeypatch.setattr(mw, "RELEASE_TIMEOUT_S", 0.3)
    spawner = Spawner()
    writer, _path, _codec = _open(tmp_path, spawner)
    spawner.process.stdin.block.clear()
    started = time.monotonic()
    with pytest.raises(RecordingWriterError, match="0.2 秒没有接收画面"):
        for value in range(10):
            writer.write(_frame(value))
    assert time.monotonic() - started < 1.5
    started = time.monotonic()
    with pytest.raises(RecordingWriterError, match="录像文件可能不完整"):
        writer.release()
    assert time.monotonic() - started < 1.5 and spawner.process.killed


def test_release_reports_a_nonzero_exit_code(tmp_path):
    spawner = Spawner(exit_code=1)
    writer, _path, _codec = _open(tmp_path, spawner)
    writer.write(_frame(10))
    spawner.process.stderr.write(b"Invalid data found when processing input\n")
    with pytest.raises(RecordingWriterError, match=r"退出码 1.*Invalid data"):
        writer.release()


def test_release_that_never_finishes_kills_ffmpeg_within_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "RELEASE_TIMEOUT_S", 0.3)
    spawner = Spawner(hang=True)
    writer, _path, _codec = _open(tmp_path, spawner)
    writer.write(_frame(10))
    started = time.monotonic()
    with pytest.raises(RecordingWriterError, match="没有写完录像文件"):
        writer.release()
    assert time.monotonic() - started < 1.5 and spawner.process.killed


def test_release_is_idempotent_and_removes_an_empty_output(tmp_path):
    spawner = Spawner()
    writer, path, _codec = _open(tmp_path, spawner)
    path.write_bytes(b"header only")     # ffmpeg 收到 EOF 时可能已写了文件头
    writer.release()
    writer.release()
    assert not path.exists() and spawner.process.stdin.closed
    with pytest.raises(RecordingWriterError, match="已经关闭"):
        writer.write(_frame(10))


def test_ffmpeg_that_cannot_start_is_a_clear_open_error(tmp_path):
    def missing(_command, _stderr):
        raise FileNotFoundError(2, "系统找不到指定的文件。")

    with pytest.raises(RecordingWriterError, match="无法启动 ffmpeg"):
        open_recording_writer(tmp_path / "front.mp4", fps=30.0, size=(64, 48), spawn=missing)
    with pytest.raises(ValueError):
        open_recording_writer(tmp_path / "front.mp4", fps=30.0, size=(0, 48), spawn=Spawner())


def _real_ffmpeg() -> bool:
    exe = ffmpeg_executable()
    return Path(exe).is_file() or shutil.which(exe) is not None


@pytest.mark.skipif(not _real_ffmpeg(), reason="本机没有 ffmpeg（CI 不装 imageio-ffmpeg）")
def test_real_ffmpeg_muxes_a_playable_avi_with_exact_frames_and_orientation(tmp_path):
    writer, path, _codec = open_recording_writer(tmp_path / "front.mp4", fps=30.0, size=(48, 64), rotation=90)
    values = [(10 + 7 * k) % 200 for k in range(40)]
    for value in values:
        writer.write(_frame(value))
    writer.release()
    capture = cv2.VideoCapture(str(path))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    decoded = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        decoded.append(image)
    capture.release()
    assert len(decoded) == 40 and decoded[0].shape[:2] == (64, 48)
    assert [int(round(image[30:, :40].mean())) for image in decoded] == pytest.approx(values, abs=3)
    assert decoded[0][:8, -8:].mean() > 200
