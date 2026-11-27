# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.media.writer as media_writer
from core.evaluation import estimate_action_range
from core.foundation.tasks import Cancelled, CancelToken
from core.media import (
    first_frame_ok,
    is_mjpeg_avi,
    open_recording_writer,
    open_video,
    transcode_to_h264,
    trim_mjpeg_avi,
)
from core.recording import StepContext, Transcode
from core.workflows import ExamClipStep, align_frame_counts, prepare_exam_pair
from core.workflows.exam.clip import estimate_action_range_from_feature_seq


def test_align_frame_counts_min() -> None:
    n, warnings = align_frame_counts(100, 97)
    assert n == 97
    assert any("frame_align" in w for w in warnings)


def test_estimate_action_range_fallback_time() -> None:
    start, end, warnings = estimate_action_range(
        90, fps=30.0, energy=None, smart_crop=True, trim_head_s=0.5, trim_tail_s=0.8
    )
    assert start == 15  # 0.5*30
    assert end == 90 - 24  # 0.8*30
    assert end > start


def test_exam_default_no_energy_keeps_full_tail() -> None:
    # 考试默认（smart_crop=False）：无 energy → 不裁尾，只固定掐头
    start, end, _ = estimate_action_range(90, fps=30.0, energy=None)
    assert start == 15  # 0.5*30 掐头
    assert end == 90  # 尾部不按时间掐


def test_exam_default_tail_scan_drops_trailing_quiet() -> None:
    # 尾部 2s 空场（能量低）应被回扫切掉，中段动作保留
    fps = 30.0
    n = 300  # 10s
    rng = np.random.default_rng(0)
    energy = np.full(n - 1, 0.001, dtype=np.float32)
    # 1.5s~5s 动作，能量有起伏（真实动作非恒定），末尾 ~5s 空场
    energy[45:150] = 0.3 + 0.2 * rng.random(105).astype(np.float32)
    start, end, _ = estimate_action_range(n, fps=fps, energy=energy)
    assert start == 15
    # 末动作 idx ~149 → +2 +0.3s(9) ≈ 160；尾部 ~140 帧空场被切
    assert 155 <= end <= 170


def test_exam_default_force_finish_keeps_last_action() -> None:
    # force_finish：尾部仍是动作（无空场）→ 回扫停在末帧附近，不吞末动作
    fps = 30.0
    n = 200
    energy = np.full(n - 1, 0.5, dtype=np.float32)  # 全程动作
    _, end, _ = estimate_action_range(n, fps=fps, energy=energy)
    assert end >= n - int(round(0.3 * fps)) - 2  # 末动作不被吞


def test_estimate_action_range_from_motion_features() -> None:
    # 头尾静止、中段运动
    t, j = 120, 4
    feats = np.zeros((t, j, 2), dtype=np.float32)
    for i in range(40, 80):
        feats[i, :, 0] = (i - 40) * 0.05
    start, end, _ = estimate_action_range_from_feature_seq(feats, fps=30.0)
    assert start < 50
    assert end > 70


def test_prepare_exam_pair_aligns_and_writes(tmp_path: Path) -> None:
    import cv2

    def write_video(path: Path, n: int, color: tuple[int, int, int]) -> None:
        w, h = 64, 48
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
        for i in range(n):
            # 中段加噪声形成能量
            fr = np.full((h, w, 3), color, dtype=np.uint8)
            if 20 <= i < 50:
                fr = (
                    (fr + np.random.randint(0, 40, fr.shape, dtype=np.uint8))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            wr.write(fr)
        wr.release()

    front = tmp_path / "front.mp4"
    side = tmp_path / "side.mp4"
    write_video(front, 60, (10, 10, 10))
    write_video(side, 55, (20, 20, 20))
    out = tmp_path / "out"
    result = prepare_exam_pair(
        front,
        side,
        front_frames=60,
        side_frames=55,
        out_dir=out,
        trim_head_s=0.2,
        trim_tail_s=0.2,
    )
    assert result.front_frames == result.side_frames
    assert result.front_frames > 0
    assert result.front_path.is_file()
    assert result.side_path.is_file()
    assert any("frame_align" in w for w in result.warnings)
    assert result.meta.get("streamed") is True
    assert result.meta.get("stream_copy") is False


def _encode_indexed_frame(
    idx: int, *, active: bool = False, base_color: tuple[int, int, int] = (20, 20, 20)
) -> np.ndarray:
    """生成左上角带 8 位二进制帧号块（耐 JPEG 有损压缩）的 64x48 测试帧。"""
    frame = np.full((48, 64, 3), base_color, dtype=np.uint8)
    for bit in range(8):
        val = 255 if (idx >> bit) & 1 else 0
        frame[0:8, bit * 8 : (bit + 1) * 8] = val
    if active:
        shift = (idx * 7) % 40
        frame[20:44, shift : shift + 20] = (220, 180, 40)
    return frame


def _decode_frame_index(frame: np.ndarray) -> int:
    idx = 0
    for bit in range(8):
        block = frame[1:7, bit * 8 + 1 : (bit + 1) * 8 - 1]
        if float(block.mean()) > 127.5:
            idx |= 1 << bit
    return idx


def _read_frame_indices(path: Path) -> list[int]:
    with open_video(path) as source:
        return [_decode_frame_index(frame.bgr) for frame in source]


def _read_all_frames(path: Path) -> list[np.ndarray]:
    with open_video(path) as source:
        return [frame.bgr.copy() for frame in source]


def _write_mjpeg_avi(
    path: Path,
    count: int,
    *,
    fps: float = 30.0,
    active_range: tuple[int, int] = (0, 0),
    base_color: tuple[int, int, int] = (20, 20, 20),
) -> Path:
    writer, actual, _codec = open_recording_writer(path, fps=fps, size=(64, 48))
    for i in range(count):
        active = active_range[0] <= i < active_range[1]
        writer.write(_encode_indexed_frame(i, active=active, base_color=base_color))
    writer.release()
    return actual


def _video_fourcc(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        cap.release()
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


class _FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        self.return_code = -15

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int | None:
        return self.return_code


@pytest.mark.parametrize("fps", [30.0, 29.97])
def test_trim_mjpeg_avi_zero_frame_offset_and_lossless_retrim(tmp_path: Path, fps: float) -> None:
    src = _write_mjpeg_avi(tmp_path / f"src_{fps}.avi", 60, fps=fps, active_range=(15, 45))
    assert is_mjpeg_avi(src)
    source_frames = _read_all_frames(src)
    assert len(source_frames) == 60

    for start, end, expected_end in [
        (0, 12, 12),
        (1, 2, 2),
        (15, 42, 42),
        (45, 60, 60),
        (50, 75, 60),
    ]:
        dst = tmp_path / f"cut_{fps}_{start}_{end}.avi"
        written = trim_mjpeg_avi(src, dst, start_frame=start, end_frame=end)
        assert written == expected_end - start
        assert dst.is_file() and first_frame_ok(dst)
        got_indices = _read_frame_indices(dst)
        assert got_indices == list(range(start, expected_end))
        assert got_indices[0] == start
        assert got_indices[-1] == expected_end - 1
        trimmed_frames = _read_all_frames(dst)
        assert all(
            np.array_equal(a, b) for a, b in zip(trimmed_frames, source_frames[start:expected_end])
        )

    # 连续二次重入裁剪：[15, 42) 再裁 [5, 17) → 原始帧 [20, 32)，逐像素无损
    first_cut = tmp_path / f"cut_{fps}_15_42.avi"
    second_cut = tmp_path / f"retrim_{fps}.avi"
    written2 = trim_mjpeg_avi(first_cut, second_cut, start_frame=5, end_frame=17)
    assert written2 == 12
    assert _read_frame_indices(second_cut) == list(range(20, 32))
    assert all(
        np.array_equal(a, b) for a, b in zip(_read_all_frames(second_cut), source_frames[20:32])
    )
    assert not list(tmp_path.glob(".*.tmp.avi"))


def test_is_mjpeg_avi_and_trim_rejects_invalid_inputs(tmp_path: Path, monkeypatch) -> None:
    mjpeg_avi = _write_mjpeg_avi(tmp_path / "ok.avi", 20)
    assert is_mjpeg_avi(mjpeg_avi)

    cv2_mjpg = tmp_path / "cv2_mjpg.avi"
    vw = cv2.VideoWriter(str(cv2_mjpg), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(10):
        vw.write(_encode_indexed_frame(i))
    vw.release()
    assert is_mjpeg_avi(cv2_mjpg)

    xvid_avi = tmp_path / "xvid.avi"
    vw = cv2.VideoWriter(str(xvid_avi), cv2.VideoWriter_fourcc(*"XVID"), 30.0, (64, 48))
    for i in range(10):
        vw.write(_encode_indexed_frame(i))
    vw.release()
    assert not is_mjpeg_avi(xvid_avi)

    mp4_path = tmp_path / "sample.mp4"
    vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(10):
        vw.write(_encode_indexed_frame(i))
    vw.release()
    assert not is_mjpeg_avi(mp4_path)

    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"not-a-valid-avi")
    assert not is_mjpeg_avi(corrupt)
    assert not is_mjpeg_avi(tmp_path / "missing.avi")

    dst = tmp_path / "out.avi"
    with pytest.raises(TypeError):
        trim_mjpeg_avi(mjpeg_avi, dst, start_frame=True, end_frame=5)
    with pytest.raises(TypeError):
        trim_mjpeg_avi(mjpeg_avi, dst, start_frame=0, end_frame=False)
    for bad_start, bad_end in [(-1, 5), (5, 5), (6, 5), (20, 25)]:
        with pytest.raises(ValueError):
            trim_mjpeg_avi(mjpeg_avi, dst, start_frame=bad_start, end_frame=bad_end)
    with pytest.raises(ValueError):
        trim_mjpeg_avi(mjpeg_avi, tmp_path / "out.mp4", start_frame=0, end_frame=5)
    for bad_src in (xvid_avi, mp4_path, corrupt, tmp_path / "missing.avi"):
        with pytest.raises(ValueError):
            trim_mjpeg_avi(bad_src, dst, start_frame=0, end_frame=5)

    # ffmpeg 失败时保留既有目标文件并清理临时文件
    dst.write_bytes(b"EXISTING_AVI")

    def fail_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(media_writer.subprocess, "run", fail_run)
    with pytest.raises(subprocess.CalledProcessError):
        trim_mjpeg_avi(mjpeg_avi, dst, start_frame=2, end_frame=8)
    assert dst.read_bytes() == b"EXISTING_AVI"
    assert not list(tmp_path.glob(".*.tmp.avi"))


def test_trim_mjpeg_avi_cancellation_terminates_ffmpeg_and_cleans_temp(
    tmp_path: Path, monkeypatch
) -> None:
    src = _write_mjpeg_avi(tmp_path / "src.avi", 30)
    dst = tmp_path / "dst.avi"
    dst.write_bytes(b"KEEP_OLD")

    # 预取消：不启动 ffmpeg、不留临时文件、不破坏既有目标
    monkeypatch.setattr(
        media_writer.subprocess,
        "Popen",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("ffmpeg must not start")),
    )
    with pytest.raises(InterruptedError, match="裁剪已取消"):
        trim_mjpeg_avi(src, dst, start_frame=2, end_frame=10, cancel_check=lambda: True)
    assert dst.read_bytes() == b"KEEP_OLD"
    assert not list(tmp_path.glob(".*.tmp.avi"))

    # 运行中取消：终止进程、清理临时文件、保留既有目标
    cancelled = False
    process = _FakeProcess()

    def fake_popen(command, **_kwargs):
        nonlocal cancelled
        Path(command[-1]).write_bytes(b"partial-avi")
        cancelled = True
        return process

    monkeypatch.setattr(media_writer.subprocess, "Popen", fake_popen)
    with pytest.raises(InterruptedError, match="裁剪已取消"):
        trim_mjpeg_avi(src, dst, start_frame=2, end_frame=10, cancel_check=lambda: cancelled)
    assert process.terminated
    assert not process.killed
    assert dst.read_bytes() == b"KEEP_OLD"
    assert not list(tmp_path.glob(".*.tmp.avi"))

    # stop_evt 取消同样终止进程并清理临时文件
    stop_evt = threading.Event()
    evt_process = _FakeProcess()

    def fake_popen_evt(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial-evt-avi")
        stop_evt.set()
        return evt_process

    monkeypatch.setattr(media_writer.subprocess, "Popen", fake_popen_evt)
    with pytest.raises(InterruptedError, match="裁剪已取消"):
        trim_mjpeg_avi(src, dst, start_frame=2, end_frame=10, stop_evt=stop_evt)
    assert evt_process.terminated
    assert not evt_process.killed
    assert dst.read_bytes() == b"KEEP_OLD"
    assert not list(tmp_path.glob(".*.tmp.avi"))


def test_exam_clip_mjpeg_avi_zero_offset_and_transcodes_to_h264(tmp_path: Path) -> None:
    seg_dir = tmp_path / "seg_exam"
    seg_dir.mkdir()
    front_avi = _write_mjpeg_avi(
        seg_dir / "front.avi",
        90,
        fps=30.0,
        active_range=(25, 65),
        base_color=(15, 15, 15),
    )
    side_avi = _write_mjpeg_avi(
        seg_dir / "side.avi",
        85,
        fps=30.0,
        active_range=(25, 65),
        base_color=(30, 30, 30),
    )

    direct = prepare_exam_pair(
        front_avi,
        side_avi,
        front_frames=90,
        side_frames=85,
        out_dir=tmp_path / "direct_out",
    )
    assert direct.front_path.suffix == ".avi"
    assert direct.side_path.suffix == ".avi"
    assert direct.meta.get("stream_copy") is True
    assert direct.meta.get("streamed") is True
    expected_indices = list(range(direct.trim_start, direct.trim_end))
    assert len(expected_indices) == direct.front_frames == direct.side_frames
    assert _read_frame_indices(direct.front_path) == expected_indices
    assert _read_frame_indices(direct.side_path) == expected_indices

    # 串联 ExamClipStep -> Transcode：输出 .avi 自然转码为真 H.264 .mp4，且 0 帧偏移
    ctx = StepContext(
        segment_id="seg_exam",
        segment_dir=seg_dir,
        views=("front", "side"),
        sources={"front": front_avi, "side": side_avi},
        frames={"front": 90, "side": 85},
        errors={},
        stop_evt=threading.Event(),
        publish=lambda _status, _msg: None,
    )
    cancel = CancelToken()
    ExamClipStep().run(ctx, cancel)

    assert ctx.sources["front"] == seg_dir / "front_exam.avi"
    assert ctx.sources["side"] == seg_dir / "side_exam.avi"
    assert ctx.frames["front"] == ctx.frames["side"] == len(expected_indices)
    assert _read_frame_indices(ctx.sources["front"]) == expected_indices
    assert _read_frame_indices(ctx.sources["side"]) == expected_indices

    Transcode().run(ctx, cancel)
    front_mp4 = ctx.video("front")
    side_mp4 = ctx.video("side")
    assert front_mp4 == seg_dir / "front_exam.mp4"
    assert side_mp4 == seg_dir / "side_exam.mp4"
    assert not (seg_dir / "front_exam.avi").exists()
    assert not (seg_dir / "side_exam.avi").exists()
    assert not list(seg_dir.glob(".*.tmp.*"))
    for mp4 in (front_mp4, side_mp4):
        assert mp4.is_file() and first_frame_ok(mp4)
        fourcc = _video_fourcc(mp4).lower()
        assert fourcc in {"h264", "avc1"}, f"expected H.264, got {fourcc!r}"
        assert fourcc != "mp4v"
        assert _read_frame_indices(mp4) == expected_indices


def test_prepare_exam_pair_mjpeg_cancel_cleans_sibling_and_temp(
    tmp_path: Path, monkeypatch
) -> None:
    front_avi = _write_mjpeg_avi(tmp_path / "front.avi", 60, active_range=(15, 45))
    side_avi = _write_mjpeg_avi(tmp_path / "side.avi", 60, active_range=(15, 45))
    out_dir = tmp_path / "out"
    cancel = CancelToken()
    real_popen = media_writer.subprocess.Popen
    popen_calls = 0

    def cancel_on_second_view(command, *args, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 2:
            Path(command[-1]).write_bytes(b"partial-side")
            cancel.cancel("app_close")
            return _FakeProcess()
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(media_writer.subprocess, "Popen", cancel_on_second_view)
    ctx = StepContext(
        segment_id="seg_cancel",
        segment_dir=out_dir,
        views=("front", "side"),
        sources={"front": front_avi, "side": side_avi},
        frames={"front": 60, "side": 60},
        errors={},
        stop_evt=threading.Event(),
        publish=lambda _status, _msg: None,
    )
    with pytest.raises(Cancelled):
        ExamClipStep().run(ctx, cancel)

    assert popen_calls == 2
    assert not list(out_dir.glob("*_exam.*"))
    assert not list(out_dir.glob(".*.tmp.*"))
    assert not any("exam_clip_failed" in note for note in ctx.notes)


def test_exam_clip_step_fallback_on_xvid_and_keeps_sources_on_corrupt_or_trim_error(
    tmp_path: Path, monkeypatch
) -> None:
    # 1. 非 MJPG 的 XVID .avi：安全走流式回退，且输出 .avi 可被转码为 H.264 .mp4
    xvid_front = tmp_path / "xvid_front.avi"
    xvid_side = tmp_path / "xvid_side.avi"
    for path in (xvid_front, xvid_side):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), 30.0, (64, 48))
        for i in range(60):
            vw.write(_encode_indexed_frame(i, active=(20 <= i < 45)))
        vw.release()
    xvid_out = tmp_path / "xvid_out"
    res = prepare_exam_pair(
        xvid_front, xvid_side, front_frames=60, side_frames=60, out_dir=xvid_out
    )
    assert res.meta.get("stream_copy") is False
    assert res.front_path.suffix == ".avi" and res.side_path.suffix == ".avi"
    h264_front = transcode_to_h264(res.front_path)
    assert h264_front is not None and h264_front.suffix == ".mp4"
    assert _video_fourcc(h264_front).lower() in {"h264", "avc1"}

    # 2. 损坏输入：ExamClipStep 记 exam_clip_failed 且保留原录像引用不炸主流程
    corrupt_dir = tmp_path / "corrupt_seg"
    corrupt_dir.mkdir()
    bad_front = corrupt_dir / "front.avi"
    bad_side = corrupt_dir / "side.avi"
    bad_front.write_bytes(b"corrupt")
    bad_side.write_bytes(b"corrupt")
    corrupt_ctx = StepContext(
        segment_id="seg_corrupt",
        segment_dir=corrupt_dir,
        views=("front", "side"),
        sources={"front": bad_front, "side": bad_side},
        frames={"front": 60, "side": 60},
        errors={},
        stop_evt=threading.Event(),
        publish=lambda _status, _msg: None,
    )
    ExamClipStep().run(corrupt_ctx, CancelToken())
    assert corrupt_ctx.sources["front"] == bad_front
    assert corrupt_ctx.sources["side"] == bad_side
    assert corrupt_ctx.frames == {"front": 60, "side": 60}
    assert any("exam_clip_failed" in note for note in corrupt_ctx.notes)

    # 3. 真实 MJPG .avi 直裁第二路异常：清理已写出的第一路，保留原始录像引用
    err_dir = tmp_path / "err_seg"
    err_dir.mkdir()
    good_front = _write_mjpeg_avi(err_dir / "front.avi", 60, active_range=(15, 45))
    good_side = _write_mjpeg_avi(err_dir / "side.avi", 60, active_range=(15, 45))
    real_popen = media_writer.subprocess.Popen
    calls = 0

    def fail_second_trim(command, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(command[-1]).write_bytes(b"broken")
            proc = _FakeProcess()
            proc.return_code = 1
            return proc
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(media_writer.subprocess, "Popen", fail_second_trim)
    err_ctx = StepContext(
        segment_id="seg_trim_err",
        segment_dir=err_dir,
        views=("front", "side"),
        sources={"front": good_front, "side": good_side},
        frames={"front": 60, "side": 60},
        errors={},
        stop_evt=threading.Event(),
        publish=lambda _status, _msg: None,
    )
    ExamClipStep().run(err_ctx, CancelToken())
    assert err_ctx.sources["front"] == good_front
    assert err_ctx.sources["side"] == good_side
    assert err_ctx.frames == {"front": 60, "side": 60}
    assert any("exam_clip_failed" in note for note in err_ctx.notes)
    assert not list(err_dir.glob("*_exam.*"))
    assert not list(err_dir.glob(".*.tmp.*"))
