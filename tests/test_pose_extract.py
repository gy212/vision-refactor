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

from core.pose.features import extract_features as _extract_pose_features  # noqa: E402
from core.pose import normalize_pose_xy  # noqa: E402
from core.foundation.tasks import Cancelled, CancelToken  # noqa: E402
from core.media import DecodeOptions, UndecodableFrame, VideoOpenError, open_video  # noqa: E402
from core.pose import (  # noqa: E402
    FORMAL,
    Coords,
    GapPolicy,
    ModelMissing,
    extract_pose_raw,
    extract_pose_raw_series,
    extract_pose_sequence,
    formal_engine,
    normalize_pose_xy_v3,
    pose_cache,
)
from core.pose import extract as extract_mod  # noqa: E402
from core.pose.landmarks import RowLandmark  # noqa: E402
from core.pose.validity import derive_valid_mask  # noqa: E402
from core.pose.view_score import pose_view_score  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
STUDENT = "golden://p2_student.mp4"


def _student_raw() -> np.ndarray:
    return np.asarray(np.load(FIX_DIR / "student_raw.npz", allow_pickle=True)["landmarks"], dtype=np.float32)


@pytest.fixture()
def counted_replay():
    """golden 回放 + 统计引擎构造次数（每构造一次就是一次完整推理）。

    monkeypatch 必须在 ``replay_context()`` 内部撤销：fixture 级 ``monkeypatch`` 在
    本 fixture 之后才拆，若那时 ``replay_context`` 已退出并恢复了真 ``create_engine``，
    monkeypatch 的 undo 会把假引擎重新写回去，污染同一会话里之后的测试（问题 3）。
    用 ``monkeypatch.context()`` 让它的撤销发生在 ``with`` 块内，先于 ``replay_context`` 退出。
    """
    H.clear_registry()
    raw = _student_raw()
    raw[5] = np.nan  # 造一帧归一化失败
    H.register_video(STUDENT, raw, fps=30.0)
    built: list[object] = []
    with H.replay_context(), pytest.MonkeyPatch.context() as mp:
        fake_create = extract_mod.create_engine

        def counted(spec, **kw):
            built.append(spec)
            return fake_create(spec, **kw)

        mp.setattr(extract_mod, "create_engine", counted)
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


def test_model_missing_pickle_round_trip_keeps_attrs_and_is_file_not_found_error():
    """问题 6：默认 __reduce__ 只用 self.args（即 hint）重建，缺 model_key 这个必填参数会抛
    TypeError；__reduce__ 显式传回两个构造参数修复此问题。feedback_panel 靠 FileNotFoundError
    子类关系捕获它，往返后仍须保持。"""
    import pickle

    original = ModelMissing("exam_heavy", "缺少 pose_landmarker_heavy.task，请在模型管理中安装。")
    restored = pickle.loads(pickle.dumps(original))

    assert restored.model_key == original.model_key == "exam_heavy"
    assert restored.hint == original.hint
    assert str(restored) == str(original)
    assert isinstance(restored, ModelMissing)
    assert isinstance(restored, FileNotFoundError)


def test_video_open_error_pickle_round_trip_keeps_message_and_path(tmp_path):
    """问题 6：默认 __reduce__ 会用已拼好的提示文案重新当 path 传入，往返后消息里的
    "无法打开视频：" 会出现两次、.path 也不对；__reduce__ 显式传回构造参数修复此问题。"""
    import pickle

    missing = tmp_path / "a.mp4"
    original = VideoOpenError(missing)
    restored = pickle.loads(pickle.dumps(original))

    assert str(restored) == str(original) == f"无法打开视频：{missing}"
    assert restored.path == original.path == missing
    assert isinstance(restored, VideoOpenError)


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
        for fn in (normalize_pose_xy_v3, normalize_pose_xy):
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
    invalid = seq.raw(GapPolicy.HOLD_LAST_INVALID)
    assert zero.dtype == np.float32 and not zero[6].any()
    np.testing.assert_array_equal(invalid[6, :, :3], invalid[5, :, :3])
    assert not invalid[6, :, 3].any()
    # D-2：补帧不得判为有效。
    assert not derive_valid_mask(invalid)[6].any()
    with pytest.raises(ValueError):
        seq.raw(GapPolicy.HOLD_LAST_FEATURE)


# --------------------------------------------------------------------------- #
# pose_cache
# --------------------------------------------------------------------------- #


def test_pose_cache_infers_each_video_once(counted_replay):
    with pose_cache():
        feats, _fps, views = _extract_pose_features(
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



def test_pose_cache_nested_scope_reuses_outer_store(counted_replay):
    """嵌套作用域沿用外层缓存（评分函数自带作用域，不能切断调用方更大范围的复用）。"""
    with pose_cache():
        extract_pose_raw_series(STUDENT, pose_variant="full")
        with pose_cache():
            extract_pose_raw_series(STUDENT, pose_variant="full")
        extract_pose_raw_series(STUDENT, pose_variant="full")
    assert len(counted_replay) == 1
    extract_pose_raw_series(STUDENT, pose_variant="full")
    assert len(counted_replay) == 2

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


# --------------------------------------------------------------------------- #
# 问题 1：缓存命中必须与同参数的全新提取完全相同
# --------------------------------------------------------------------------- #


def _assert_cache_hit_matches_fresh(video, engine, decode_a, decode_b):
    """先按 ``decode_a`` 提取（进入缓存），再按 ``decode_b`` 提取（期望命中缓存或前缀复用）；
    命中结果必须和不经缓存的全新提取逐元素一致，包括 ``frames_scanned``/``stride``/``inference_size``。
    """
    profile_b = replace(FORMAL, decode=decode_b)
    fresh = extract_pose_sequence(video, engine, profile_b)
    with pose_cache():
        extract_pose_sequence(video, engine, replace(FORMAL, decode=decode_a))
        cached = extract_pose_sequence(video, engine, profile_b)
    assert len(cached) == len(fresh)
    np.testing.assert_array_equal(cached.landmarks, fresh.landmarks)
    np.testing.assert_array_equal(cached.detected, fresh.detected)
    np.testing.assert_array_equal(cached.decoded, fresh.decoded)
    np.testing.assert_array_equal(cached.frame_index, fresh.frame_index)
    np.testing.assert_array_equal(cached.t_ms, fresh.t_ms)
    np.testing.assert_array_equal(cached.source_wh, fresh.source_wh)
    assert cached.meta == fresh.meta
    return cached, fresh


def test_cache_prefix_reuse_with_nonzero_start_is_not_corrupted(counted_replay):
    """先 start=5,end=None，再 start=5,end=10：错误实现把 10 行前缀当第 10 帧，返回 11 行而非 6 行。"""
    engine = formal_engine("full")
    cached, fresh = _assert_cache_hit_matches_fresh(
        STUDENT, engine, DecodeOptions(start=5, end=None), DecodeOptions(start=5, end=10))
    assert len(fresh) == 6  # 第 5..10 帧


def test_cache_prefix_reuse_after_segment_end_misjudges_eof(counted_replay):
    """先 start=50,end=100，再 start=50,end=None：错误实现按行数误判为读到结尾，返回 51 行而非 130 行。"""
    engine = formal_engine("full")
    cached, fresh = _assert_cache_hit_matches_fresh(
        STUDENT, engine, DecodeOptions(start=50, end=100), DecodeOptions(start=50, end=None))
    assert len(fresh) == 130  # 第 50..179 帧


def test_cache_prefix_reuse_rejected_when_sampling_changes_row_to_frame_mapping(tmp_path):
    """30fps/60 帧 MJPG，sample_fps=10：先全量再 end=20。错误实现把 21 行前缀当读到第 20 帧，
    应为 7 行（0,3,...,18）。抽帧时行号与帧号不是一一对应，这类请求只认终点相同的命中，
    不能用 ``prefix()`` 截。"""
    video = _write_avi(tmp_path / "sample.avi", 60, size=(32, 24), fps=30.0)
    engine = formal_engine("full")

    class _Fake:
        def __init__(self, **_kw):
            pass

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    import core.pose.extract as extract_mod_local

    old_create = extract_mod_local.create_engine
    extract_mod_local.create_engine = lambda spec, **_kw: _Fake()
    try:
        full_decode = DecodeOptions(sample_fps=10)
        end20_decode = DecodeOptions(sample_fps=10, end=20)
        cached, fresh = _assert_cache_hit_matches_fresh(video, engine, full_decode, end20_decode)
        assert len(fresh) == 7
        assert list(fresh.frame_index) == [0, 3, 6, 9, 12, 15, 18]
    finally:
        extract_mod_local.create_engine = old_create


def test_cache_hit_when_end_exceeds_video_length(counted_replay):
    """end 超过视频长度：命中必须等于读到结尾的全新提取（180 帧）。"""
    engine = formal_engine("full")
    cached, fresh = _assert_cache_hit_matches_fresh(
        STUDENT, engine, DecodeOptions(end=None), DecodeOptions(end=10**6))
    assert len(fresh) == 180


def test_segmented_features_match_legacy_warmup_semantics(counted_replay):
    """分段特征：段内首个归一化失败的帧沿用预热帧的特征，而非补零。"""
    feats, _fps, views = _extract_pose_features(
        Path(STUDENT), pose_variant="full", workers=3, normalizer=normalize_pose_xy_v3, compute_view=True)
    seq_feats, _fps, _ = _extract_pose_features(
        Path(STUDENT), pose_variant="full", workers=1, normalizer=normalize_pose_xy_v3)
    # golden 回放无跟踪状态，分段与顺序逐帧结果应一致（第 5 帧 NaN 由上一帧补）
    assert np.array_equal(feats, seq_feats)
    assert views.dtype == np.float32


# --------------------------------------------------------------------------- #
# 问题 2：workers>1 时非默认的 start/end/sample_fps 退回顺序跟踪
# --------------------------------------------------------------------------- #


def test_segmented_falls_back_to_sequential_when_start_end_given(counted_replay):
    """start=20,end=40,workers=3：错误实现忽略这些选项返回全部 180 帧；应退回顺序跟踪，只读 21 帧。"""
    profile = replace(FORMAL, decode=DecodeOptions(start=20, end=40))
    seq = extract_pose_sequence(STUDENT, formal_engine("full"), profile, workers=3)
    assert seq.meta.tracking == "sequential"  # 退回顺序跟踪
    assert len(seq) == 21
    assert list(seq.frame_index) == list(range(20, 41))


def test_segmented_falls_back_to_sequential_when_sample_fps_given(tmp_path):
    """30fps/60 帧，sample_fps=10,workers=3：评审复现里错误实现分段解码（60 行，仅 21 行 decoded），
    特征遇到首个未解码帧就 break，只剩 1 行非零特征；应退回顺序跟踪，抽帧产出 21 行、全部 decoded、
    每行都有非零特征。
    """
    video = _write_avi(tmp_path / "sample_fps_workers.avi", 60, size=(32, 24), fps=30.0)

    class _Fake:
        def __init__(self, **_kw):
            self.n = 0

        def infer(self, frame, *, timestamp_ms=None):
            self.n += 1
            from types import SimpleNamespace

            return [SimpleNamespace(x=0.1 * (i + 1) + 0.001 * self.n, y=0.9 - 0.02 * i, z=0.01 * i,
                                     visibility=0.9) for i in range(33)], []

        def close(self):
            pass

    import core.pose.extract as extract_mod_local

    old_create = extract_mod_local.create_engine
    extract_mod_local.create_engine = lambda spec, **_kw: _Fake()
    try:
        profile = replace(FORMAL, decode=DecodeOptions(sample_fps=10))
        seq = extract_pose_sequence(video, formal_engine("full"), profile, workers=3)
    finally:
        extract_mod_local.create_engine = old_create
    assert seq.meta.tracking == "sequential"  # 退回顺序跟踪
    assert len(seq) == 21  # stride=ceil(30/10)=3：0,3,...,57 共 20 行，末帧 59 总会保留，合计 21 行
    assert seq.decoded.all()
    feats = seq.features(normalize_pose_xy_v3, (22, 2))
    assert np.count_nonzero(np.any(feats != 0, axis=(1, 2))) == len(seq)


def test_segmented_records_source_wh_for_aspect_corrected_view(tmp_path):
    """分段曾把 source_wh 全填 0，取 raw(ASPECT_CORRECTED) 会除以 0；应记录每帧的真实缩放前尺寸。"""
    video = _write_avi(tmp_path / "seg_wh.avi", 12, size=(40, 20), fps=10.0)

    class _Fake:
        def __init__(self, **_kw):
            pass

        def infer(self, frame, *, timestamp_ms=None):
            return None, []

        def close(self):
            pass

    import core.pose.extract as extract_mod_local

    old_create = extract_mod_local.create_engine
    extract_mod_local.create_engine = lambda spec, **_kw: _Fake()
    try:
        seq = extract_pose_sequence(video, formal_engine("full"), FORMAL, workers=3)
    finally:
        extract_mod_local.create_engine = old_create
    assert seq.meta.tracking == "segmented"
    assert (seq.source_wh == (40, 20)).all()
    with np.errstate(invalid="raise", divide="raise"):
        seq.raw(GapPolicy.ZERO, Coords.ASPECT_CORRECTED)  # 不应抛 FloatingPointError
