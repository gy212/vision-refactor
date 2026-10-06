# -*- coding: utf-8 -*-
"""P2 姿态层：``open_video`` 解码选项、``PoseSequence`` 视图、``extract_pose_sequence`` 与 ``pose_cache``。

假视频用 golden harness 的回放（``(T,33,4)`` 序列当作视频帧）；真实解码选项用小段合成 AVI。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core import action_compare as ac  # noqa: E402
from core.foundation.tasks import Cancelled, CancelToken  # noqa: E402
from core.media import DecodeOptions, UndecodableFrame, VideoOpenError, open_video  # noqa: E402
from core.pose import (  # noqa: E402
    FORMAL,
    GapPolicy,
    extract_pose_raw,
    extract_pose_raw_series,
    extract_pose_sequence,
    formal_engine,
    normalize_pose_xy_v3,
    pose_cache,
)
from core.pose import extract as extract_mod  # noqa: E402
from core.pose.landmarks import RowLandmark  # noqa: E402
from core.pose.view_score import pose_view_score  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
STUDENT = "golden://p2_student.mp4"


def _student_raw() -> np.ndarray:
    return np.asarray(np.load(FIX_DIR / "student_raw.npz", allow_pickle=True)["landmarks"], dtype=np.float32)


@pytest.fixture()
def counted_replay(monkeypatch):
    """golden 回放 + 统计引擎构造次数（每构造一次就是一次完整推理）。"""
    H.clear_registry()
    raw = _student_raw()
    raw[5] = np.nan  # 造一帧归一化失败
    H.register_video(STUDENT, raw, fps=30.0)
    built: list[object] = []
    with H.replay_context():
        fake_create = extract_mod.create_engine

        def counted(spec, **kw):
            built.append(spec)
            return fake_create(spec, **kw)

        monkeypatch.setattr(extract_mod, "create_engine", counted)
        yield built
    H.clear_registry()


def _write_avi(path: Path, n: int, size=(64, 48), fps=10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for i in range(n):
            writer.write(np.full((size[1], size[0], 3), i * 10 % 255, np.uint8))
    finally:
        writer.release()
    return path


# --------------------------------------------------------------------------- #
# open_video
# --------------------------------------------------------------------------- #


def test_open_video_floor_timestamps_and_range(tmp_path):
    video = _write_avi(tmp_path / "a.avi", 12, fps=30.0)
    with open_video(video, DecodeOptions(start=3, end=7)) as source:
        frames = list(source)
    assert [f.index for f in frames] == [3, 4, 5, 6, 7]
    assert [f.t_ms for f in frames] == [int(i * 1000.0 / 30.0) for i in range(3, 8)]


def test_open_video_practice_options_resize_stride_and_monotonic(tmp_path):
    video = _write_avi(tmp_path / "b.avi", 11, size=(200, 100), fps=10.0)
    options = DecodeOptions(sample_fps=4, max_edge=100, timestamps="round_monotonic", strict=True)
    with open_video(video, options) as source:
        frames = list(source)
        scanned = source.frames_scanned
    # stride = ceil(10/4) = 3；末帧总会保留
    assert [f.index for f in frames] == [0, 3, 6, 9, 10]
    assert all(f.bgr.shape[:2] == (50, 100) and f.source_size == (200, 100) for f in frames)
    assert [f.t_ms for f in frames] == [0, 300, 600, 900, 1000]
    assert scanned == 11


def test_open_video_errors_and_cancel(tmp_path):
    with pytest.raises(VideoOpenError, match="无法打开视频"):
        open_video(tmp_path / "missing.avi")
    video = _write_avi(tmp_path / "c.avi", 5)
    token = CancelToken()
    token.cancel("user_stop")
    with pytest.raises(Cancelled):
        with open_video(video, cancel=token) as source:
            list(source)


def test_open_video_strict_raises_on_undecodable_frame(monkeypatch):
    from core.media import video_source

    class BadRetrieve:
        def __init__(self, _path) -> None:
            self.n = 0

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: 3.0}.get(prop, 0.0)

        def grab(self) -> bool:
            self.n += 1
            return self.n <= 3

        def retrieve(self):
            return (self.n == 1, np.zeros((4, 4, 3), np.uint8) if self.n == 1 else None)

        def read(self):
            return (self.grab() and self.retrieve()[0], np.zeros((4, 4, 3), np.uint8))

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_source.cv2, "VideoCapture", BadRetrieve)
    with open_video("x.avi", DecodeOptions(strict=True)) as source:
        with pytest.raises(UndecodableFrame):
            list(source)
    with open_video("x.avi") as source:  # 非严格：read() 失败即结束
        assert [f.index for f in source] == [0]


# --------------------------------------------------------------------------- #
# PoseSequence 视图与旧实现逐元素一致
# --------------------------------------------------------------------------- #


def test_array_inputs_match_landmark_objects():
    raw = _student_raw()
    for row in raw[:20]:
        objs = [RowLandmark(r) for r in row]
        for fn in (normalize_pose_xy_v3, ac.normalize_pose_xy):
            a, b = fn(row.astype(np.float64)), fn(objs)
            assert (a is None) == (b is None)
            if a is not None:
                assert np.array_equal(a, b)
        assert pose_view_score(row.astype(np.float64)) == pose_view_score(objs)


def test_raw_gap_policies(counted_replay):
    seq = extract_pose_sequence(STUDENT, formal_engine("full"), FORMAL)
    seq = replace(seq, detected=seq.detected.copy(), landmarks=seq.landmarks.copy())
    seq.detected[6] = False
    seq.landmarks[6] = 0
    zero = seq.raw(GapPolicy.ZERO)
    legacy = seq.raw(GapPolicy.HOLD_LAST_LEGACY)
    invalid = seq.raw(GapPolicy.HOLD_LAST_INVALID)
    assert zero.dtype == np.float32 and not zero[6].any()
    np.testing.assert_array_equal(legacy[6], legacy[5])
    np.testing.assert_array_equal(invalid[6, :, :3], legacy[5, :, :3])
    assert not invalid[6, :, 3].any()


# --------------------------------------------------------------------------- #
# pose_cache
# --------------------------------------------------------------------------- #


def test_pose_cache_infers_each_video_once(counted_replay):
    with pose_cache():
        feats, _fps, views = ac._extract_pose_features(
            Path(STUDENT), pose_variant="full", normalizer=normalize_pose_xy_v3, compute_view=True)
        series = extract_pose_raw_series(STUDENT, pose_variant="full")
        part, _meta = extract_pose_raw(STUDENT, pose_variant="full", start_frame=3, end_frame=9)
    assert len(counted_replay) == 1
    assert feats.shape[0] == series.landmarks.shape[0] == views.shape[0]
    np.testing.assert_array_equal(part, series.landmarks[3:10])  # 第 5 帧为 NaN

    # 作用域外不缓存：每次重新推理
    extract_pose_raw_series(STUDENT, pose_variant="full")
    extract_pose_raw_series(STUDENT, pose_variant="full")
    assert len(counted_replay) == 3


def test_pose_cache_prefix_reuse_is_identical_to_fresh_extraction(counted_replay):
    fresh, _ = extract_pose_raw(STUDENT, pose_variant="full", start_frame=0, end_frame=9)
    with pose_cache():
        extract_pose_raw_series(STUDENT, pose_variant="full")
        cached, _ = extract_pose_raw(STUDENT, pose_variant="full", start_frame=0, end_frame=9)
    np.testing.assert_array_equal(fresh, cached)


def test_segmented_results_are_not_reused_by_sequential_consumers(counted_replay):
    with pose_cache():
        seg = extract_pose_sequence(STUDENT, formal_engine("full"), FORMAL, workers=3)
        assert seg.meta.tracking == "segmented"
        n_after_segmented = len(counted_replay)
        seq = extract_pose_raw_series(STUDENT, pose_variant="full")   # 规则：顺序跟踪
        assert len(counted_replay) == n_after_segmented + 1
        again = extract_pose_sequence(STUDENT, formal_engine("full"), FORMAL, workers=3)
    assert again is seg
    assert seq.meta["frame_count"] == len(seg)


def test_cache_key_separates_engine_and_profile(counted_replay):
    with pose_cache():
        extract_pose_sequence(STUDENT, formal_engine("full"), FORMAL)
        extract_pose_sequence(STUDENT, formal_engine("heavy"), FORMAL)
        extract_pose_sequence(STUDENT, formal_engine("full"), replace(FORMAL, name="other"))
        extract_pose_sequence(STUDENT, formal_engine("full"), FORMAL)
    assert len(counted_replay) == 3


def test_segmented_features_match_legacy_warmup_semantics(counted_replay):
    """分段特征：段内首个归一化失败的帧沿用预热帧的特征，而非补零。"""
    feats, _fps, views = ac._extract_pose_features(
        Path(STUDENT), pose_variant="full", workers=3, normalizer=normalize_pose_xy_v3, compute_view=True)
    seq_feats, _fps, _ = ac._extract_pose_features(
        Path(STUDENT), pose_variant="full", workers=1, normalizer=normalize_pose_xy_v3)
    # golden 回放无跟踪状态，分段与顺序逐帧结果应一致（第 5 帧 NaN 由上一帧补）
    assert np.array_equal(feats, seq_feats)
    assert views.dtype == np.float32
