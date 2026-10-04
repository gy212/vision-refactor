from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from core import video_writer


class _FakeWriter:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


def test_open_video_writer_uses_only_avi_for_non_h264_fallbacks(
    tmp_path: Path, monkeypatch
) -> None:
    preferred = tmp_path / "recording.mp4"
    attempts: list[tuple[Path, str, float, tuple[int, int], _FakeWriter]] = []

    monkeypatch.setattr(video_writer, "_fourcc", lambda codec: codec)

    def fake_video_writer(path: str, codec: str, fps: float, size: tuple[int, int]):
        writer = _FakeWriter(opened=codec == "XVID")
        attempts.append((Path(path), codec, fps, size, writer))
        return writer

    monkeypatch.setattr(video_writer.cv2, "VideoWriter", fake_video_writer)

    writer, actual_path, codec = video_writer.open_video_writer(
        preferred, fps=25.0, size=(640, 480)
    )

    assert writer is attempts[-1][-1]
    assert actual_path == preferred.with_suffix(".avi")
    assert codec == "XVID"
    assert [(path, tag) for path, tag, *_rest in attempts] == [
        (preferred, "avc1"),
        (preferred, "H264"),
        (preferred.with_suffix(".avi"), "MJPG"),
        (preferred.with_suffix(".avi"), "XVID"),
    ]
    assert all(attempt[-1].released for attempt in attempts[:-1])
    assert not attempts[-1][-1].released


def test_open_video_writer_normalizes_unknown_container_fallback_to_avi(
    tmp_path: Path, monkeypatch
) -> None:
    preferred = tmp_path / "recording.data"
    attempts: list[tuple[Path, str]] = []

    monkeypatch.setattr(video_writer, "_fourcc", lambda codec: codec)

    def fake_video_writer(path: str, codec: str, _fps: float, _size: tuple[int, int]):
        attempts.append((Path(path), codec))
        return _FakeWriter(opened=True)

    monkeypatch.setattr(video_writer.cv2, "VideoWriter", fake_video_writer)

    _writer, actual_path, codec = video_writer.open_video_writer(
        preferred, fps=30.0, size=(320, 240)
    )

    assert attempts == [(preferred.with_suffix(".avi"), "MJPG")]
    assert actual_path == preferred.with_suffix(".avi")
    assert codec == "MJPG"


def test_transcode_to_h264_runs_ffmpeg_and_removes_avi(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")
    expected_dst = tmp_path / "recording.mp4"
    expected_dst.write_bytes(b"old-mp4")
    calls: list[tuple[list[str], bool, float]] = []

    def fake_run(command: list[str], *, check: bool, timeout: float, **kwargs):
        calls.append((command, check, timeout))
        Path(command[-1]).write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(video_writer.subprocess, "run", fake_run)

    result = video_writer.transcode_to_h264(src)

    assert result == expected_dst
    assert not src.exists()
    assert expected_dst.exists()
    assert expected_dst.read_bytes() == b"mp4"
    assert len(calls) == 1
    command, check, timeout = calls[0]
    temporary_dst = Path(command[-1])
    assert command[:-1] == [
        "ffmpeg",
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
    ]
    assert check is True
    assert timeout == video_writer.FFMPEG_TRANSCODE_TIMEOUT_S
    assert temporary_dst.parent == expected_dst.parent
    assert temporary_dst != expected_dst
    assert temporary_dst.name.startswith(".recording.")
    assert temporary_dst.name.endswith(".tmp.mp4")
    assert not temporary_dst.exists()


def test_transcode_to_h264_passes_through_none_missing_and_non_avi(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("ffmpeg must not run for pass-through inputs")

    monkeypatch.setattr(video_writer.subprocess, "run", unexpected_run)
    missing = tmp_path / "missing.avi"
    mp4 = tmp_path / "recording.mp4"
    mp4.write_bytes(b"mp4")

    assert video_writer.transcode_to_h264(None) is None
    assert video_writer.transcode_to_h264(missing) == missing
    assert video_writer.transcode_to_h264(mp4) == mp4


def test_pre_cancelled_pass_through_inputs_keep_legacy_return_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    stop_evt = threading.Event()
    stop_evt.set()
    missing = tmp_path / "missing.avi"
    mp4 = tmp_path / "recording.mp4"
    mp4.write_bytes(b"mp4")
    monkeypatch.setattr(
        video_writer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ffmpeg must not start for pass-through inputs")
        ),
    )

    assert video_writer.transcode_to_h264(None, stop_evt=stop_evt) is None
    assert video_writer.transcode_to_h264(missing, stop_evt=stop_evt) == missing
    assert video_writer.transcode_to_h264(mp4, stop_evt=stop_evt) == mp4


def test_transcode_to_h264_keeps_avi_when_ffmpeg_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")

    def missing_ffmpeg(*_args, **_kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(video_writer.subprocess, "run", missing_ffmpeg)

    assert video_writer.transcode_to_h264(src) == src
    assert src.exists()


def test_transcode_to_h264_keeps_avi_when_ffmpeg_fails(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")
    existing_dst = src.with_suffix(".mp4")
    existing_dst.write_bytes(b"existing")

    def failed_ffmpeg(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(video_writer.subprocess, "run", failed_ffmpeg)

    assert video_writer.transcode_to_h264(src) == src
    assert src.exists()
    assert existing_dst.read_bytes() == b"existing"
    assert list(tmp_path.glob(".recording.*.tmp.mp4")) == []


def test_transcode_to_h264_keeps_avi_when_ffmpeg_writes_no_output(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")

    def successful_without_output(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(video_writer.subprocess, "run", successful_without_output)

    assert video_writer.transcode_to_h264(src) == src
    assert src.exists()
    assert not src.with_suffix(".mp4").exists()


def test_transcode_to_h264_timeout_cleans_partial_and_keeps_avi(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")

    def timed_out(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(video_writer.subprocess, "run", timed_out)

    assert video_writer.transcode_to_h264(src) == src
    assert src.exists()
    assert not src.with_suffix(".mp4").exists()
    assert list(tmp_path.glob(".recording.*.tmp.mp4")) == []


def test_transcode_to_h264_cancellation_terminates_ffmpeg_and_cleans_partial(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")
    stop_evt = threading.Event()

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.return_code = None

        def poll(self):
            return self.return_code

        def terminate(self) -> None:
            self.terminated = True
            self.return_code = -15

        def kill(self) -> None:
            self.killed = True
            self.return_code = -9

        def wait(self, timeout=None):
            return self.return_code

    process = FakeProcess()

    def fake_popen(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        stop_evt.set()
        return process

    monkeypatch.setattr(video_writer.subprocess, "Popen", fake_popen)

    with pytest.raises(InterruptedError, match="转码已取消"):
        video_writer.transcode_to_h264(src, stop_evt=stop_evt)

    assert process.terminated
    assert not process.killed
    assert src.exists()
    assert not src.with_suffix(".mp4").exists()
    assert list(tmp_path.glob(".recording.*.tmp.mp4")) == []


def test_transcode_to_h264_pre_cancelled_does_not_start_ffmpeg(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "recording.avi"
    src.write_bytes(b"avi")
    stop_evt = threading.Event()
    stop_evt.set()

    monkeypatch.setattr(
        video_writer.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ffmpeg must not start for a cancelled task")
        ),
    )

    with pytest.raises(InterruptedError, match="转码已取消"):
        video_writer.transcode_to_h264(src, stop_evt=stop_evt)
