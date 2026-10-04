# -*- coding: utf-8 -*-
"""
compare_dual_streams 单测（issue #55，为 #56 打底）。

复用 tests/fixtures/pose33_v3 的已提交 raw fixture + golden_harness 确定性回放，
不新建 fixture。front_src_raw.npz / side_src_raw.npz 本身就是各自单视角的独立源，
正好当作“两路独立同步流”的输入，不需要 frontness 拆分。

覆盖 review 指出的三个回归风险：
1. 正/侧路径调反（channel routing reversed）；
2. 误重新启用 view split（segment 应恒为整段 (0, n-1)）；
3. raw cache 串流（每路 extract_pose_raw_series 只应各自调用一次）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Event

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core.action_compare as ac  # noqa: E402
import core.pose_features as pf  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
FRONT_TPL = FIX_DIR / "front_template.npz"
SIDE_TPL = FIX_DIR / "side_template.npz"

FRONT_SRC = "golden://front_src.mp4"
SIDE_SRC = "golden://side_src.mp4"
FPS = 30.0


class _TrackedCapture:
    instances: list["_TrackedCapture"] = []

    def __init__(self, _path) -> None:
        self.index = 0
        self.released = False
        self.__class__.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, prop) -> float:
        if prop == ac.cv2.CAP_PROP_FPS:
            return 25.0
        if prop == ac.cv2.CAP_PROP_FRAME_COUNT:
            return 3.0
        return 0.0

    def read(self):
        if self.index >= 3:
            return False, None
        self.index += 1
        return True, object()

    def set(self, prop, value) -> bool:
        if prop == ac.cv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)
        return True

    def release(self) -> None:
        self.released = True


class _TrackedPipeline:
    instances: list["_TrackedPipeline"] = []
    fail_on_infer = False
    cancel_evt: Event | None = None

    def __init__(self, *, models_dir=None, cfg=None) -> None:
        self.closed = False
        self.__class__.instances.append(self)

    def infer(self, _frame, *, timestamp_ms=None):
        if self.fail_on_infer:
            raise RuntimeError("inference failed")
        if self.cancel_evt is not None:
            self.cancel_evt.set()
        return None, []

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def tracked_extraction(monkeypatch):
    _TrackedCapture.instances.clear()
    _TrackedPipeline.instances.clear()
    _TrackedPipeline.fail_on_infer = False
    _TrackedPipeline.cancel_evt = None
    monkeypatch.setattr(ac.cv2, "VideoCapture", _TrackedCapture)
    monkeypatch.setattr(ac, "MediaPipePipeline", _TrackedPipeline)
    yield
    _TrackedPipeline.fail_on_infer = False
    _TrackedPipeline.cancel_evt = None


def test_single_thread_extraction_closes_resources_on_success(tracked_extraction):
    features, fps, view_scores = ac._extract_pose_features(
        Path("tracked-success.mp4"),
        pose_variant="full",
        workers=1,
    )

    assert features.shape == (3, *ac.POSE33_V3.shape)
    assert fps == pytest.approx(25.0)
    assert view_scores is None
    assert len(_TrackedCapture.instances) == 1
    assert _TrackedCapture.instances[0].released is True
    assert len(_TrackedPipeline.instances) == 1
    assert _TrackedPipeline.instances[0].closed is True


def test_single_thread_extraction_closes_resources_on_inference_error(tracked_extraction):
    _TrackedPipeline.fail_on_infer = True

    with pytest.raises(RuntimeError, match="inference failed"):
        ac._extract_pose_features(
            Path("tracked-error.mp4"),
            pose_variant="full",
            workers=1,
        )

    assert len(_TrackedCapture.instances) == 1
    assert _TrackedCapture.instances[0].released is True
    assert len(_TrackedPipeline.instances) == 1
    assert _TrackedPipeline.instances[0].closed is True


def test_cancelled_extraction_closes_resources_without_scoring(tracked_extraction, monkeypatch):
    stop_evt = Event()
    _TrackedPipeline.cancel_evt = stop_evt

    def unexpected_score(*_args, **_kwargs):
        pytest.fail("cancelled partial features must not reach scoring")

    monkeypatch.setattr(ac, "_score_view_seq", unexpected_score)

    with pytest.raises(InterruptedError, match="姿态特征提取已取消"):
        ac.compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            Path("tracked-front.mp4"),
            Path("tracked-side.mp4"),
            pose_variant="full",
            stop_evt=stop_evt,
        )

    assert len(_TrackedCapture.instances) == 1
    assert _TrackedCapture.instances[0].released is True
    assert len(_TrackedPipeline.instances) == 1
    assert _TrackedPipeline.instances[0].closed is True


def test_cancel_during_front_scoring_skips_side_and_returns_no_result(monkeypatch):
    stop_evt = Event()
    front_seq = np.zeros((3, *ac.POSE33_V3.shape), dtype=np.float32)
    side_seq = np.ones((3, *ac.POSE33_V3.shape), dtype=np.float32)
    extracted = iter(((front_seq, 30.0, None), (side_seq, 30.0, None)))
    score_calls: list[np.ndarray] = []

    monkeypatch.setattr(ac, "_extract_pose_features", lambda *_args, **_kwargs: next(extracted))

    def score_once_then_cancel(seq, *_args, **_kwargs):
        score_calls.append(seq)
        if len(score_calls) == 1:
            stop_evt.set()
            return 0.9, [], None
        pytest.fail("side scoring must not run after front scoring cancels")

    monkeypatch.setattr(ac, "_score_view_seq", score_once_then_cancel)

    with pytest.raises(InterruptedError, match="动作比对已取消"):
        ac.compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            Path("front-scoring-cancel.mp4"),
            Path("side-scoring-cancel.mp4"),
            pose_variant="full",
            stop_evt=stop_evt,
        )

    assert len(score_calls) == 1
    assert score_calls[0] is front_seq


@pytest.mark.parametrize("dtw", [pf.subsequence_dtw, pf.subsequence_dtw_with_path])
def test_dtw_cancel_callback_is_checked_during_dynamic_programming(dtw):
    checks = 0

    def cancel_on_second_row() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 4

    query = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    seq = np.arange(40, dtype=np.float32).reshape(5, 4, 2)

    with pytest.raises(InterruptedError, match="DTW 评分已取消"):
        dtw(query, seq, stop_evt=cancel_on_second_row)

    assert checks == 4


@pytest.mark.parametrize("dtw", [pf.subsequence_dtw, pf.subsequence_dtw_with_path])
def test_dtw_cancel_event_is_checked_after_local_cost(monkeypatch, dtw):
    stop_evt = Event()
    original_local_cost = pf._dtw_local_cost

    def local_cost_then_cancel(query, seq):
        result = original_local_cost(query, seq)
        stop_evt.set()
        return result

    monkeypatch.setattr(pf, "_dtw_local_cost", local_cost_then_cancel)
    query = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    seq = np.arange(40, dtype=np.float32).reshape(5, 4, 2)

    with pytest.raises(InterruptedError, match="DTW 评分已取消"):
        dtw(query, seq, stop_evt=stop_evt)


def test_multi_worker_extraction_closes_all_resources_on_success(tracked_extraction):
    features, fps, view_scores = ac._extract_pose_features(
        Path("tracked-multi-success.mp4"),
        pose_variant="full",
        workers=2,
    )

    assert features.shape == (3, *ac.POSE33_V3.shape)
    assert fps == pytest.approx(25.0)
    assert view_scores is None
    assert len(_TrackedCapture.instances) == 3
    assert all(item.released for item in _TrackedCapture.instances)
    assert len(_TrackedPipeline.instances) == 2
    assert all(item.closed for item in _TrackedPipeline.instances)


def test_multi_worker_cancellation_closes_resources_and_raises(tracked_extraction):
    stop_evt = Event()
    _TrackedPipeline.cancel_evt = stop_evt

    with pytest.raises(InterruptedError, match="姿态特征提取已取消"):
        ac._extract_pose_features(
            Path("tracked-multi-cancel.mp4"),
            pose_variant="full",
            workers=2,
            stop_evt=stop_evt,
        )

    assert len(_TrackedCapture.instances) >= 2
    assert all(item.released for item in _TrackedCapture.instances)
    assert _TrackedPipeline.instances
    assert all(item.closed for item in _TrackedPipeline.instances)


def test_multi_worker_inference_error_closes_resources_and_propagates(
    tracked_extraction,
):
    _TrackedPipeline.fail_on_infer = True

    with pytest.raises(RuntimeError, match="inference failed"):
        ac._extract_pose_features(
            Path("tracked-multi-error.mp4"),
            pose_variant="full",
            workers=2,
        )

    assert len(_TrackedCapture.instances) >= 2
    assert all(item.released for item in _TrackedCapture.instances)
    assert _TrackedPipeline.instances
    assert all(item.closed for item in _TrackedPipeline.instances)


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


@pytest.fixture()
def registered_videos():
    assert FRONT_TPL.exists() and SIDE_TPL.exists(), (
        "缺少模板 fixture，请先运行 tests/fixtures/regen_pose33_v3_golden.py 生成。"
    )
    H.clear_registry()
    H.register_video(FRONT_SRC, _load_raw("front_src_raw.npz"), fps=FPS)
    H.register_video(SIDE_SRC, _load_raw("side_src_raw.npz"), fps=FPS)
    yield
    H.clear_registry()


def test_no_split_and_high_score_for_matching_streams(registered_videos):
    front_n = int(_load_raw("front_src_raw.npz").shape[0])
    side_n = int(_load_raw("side_src_raw.npz").shape[0])

    with H.replay_context():
        res = ac.compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            FRONT_SRC,
            SIDE_SRC,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )

    # 无 frontness 拆分：segment 恒为整段 (0, n-1)，两路长度可各不相同。
    assert res.front_segment == (0, front_n - 1)
    assert res.side_segment == (0, side_n - 1)
    # 模板正是从这两段源视频生成，同源比对应接近满分。
    assert 0.0 <= res.front_score <= 1.0
    assert 0.0 <= res.side_score <= 1.0
    assert 0 <= res.combined_percent <= 100
    assert res.front_score > 0.9
    assert res.side_score > 0.9
    assert res.front_rule_violations is not None
    assert res.side_rule_violations is not None
    assert res.front_joint_errors is not None
    assert res.side_joint_errors is not None


def test_swapped_streams_score_worse_than_correct_pairing(registered_videos):
    """正确配对应明显优于把 front/side 视频喂反，防止通道接反的回归。"""
    with H.replay_context():
        correct = ac.compare_dual_streams(FRONT_TPL, SIDE_TPL, FRONT_SRC, SIDE_SRC, pose_variant="full")
        swapped = ac.compare_dual_streams(FRONT_TPL, SIDE_TPL, SIDE_SRC, FRONT_SRC, pose_variant="full")

    assert correct.front_score > swapped.front_score
    assert correct.side_score > swapped.side_score


def test_raw_series_extracted_once_per_stream_no_cross_contamination(registered_videos, monkeypatch):
    """独立 raw_series 闭包必须分别绑定各自视频路径：各调用一次、路径不串流。"""
    calls: list[Path] = []
    orig = ac.extract_pose_raw_series

    def spy(video_path, *, pose_variant):
        calls.append(Path(video_path))
        return orig(video_path, pose_variant=pose_variant)

    monkeypatch.setattr(ac, "extract_pose_raw_series", spy)

    with H.replay_context():
        ac.compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            FRONT_SRC,
            SIDE_SRC,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )

    assert calls.count(Path(FRONT_SRC)) == 1
    assert calls.count(Path(SIDE_SRC)) == 1


def test_layout_version_mismatch_raises_value_error(registered_videos, tmp_path):
    """正/侧模板 layout 版本不一致（v3 vs v1）应抛 ValueError，沿用既有守卫。"""
    d = np.load(FRONT_TPL, allow_pickle=True)
    meta = dict(d["meta"].item())
    meta["feature_layout"] = ac.LEGACY_DEFAULT_FEATURE_LAYOUT
    mutated_front_tpl = tmp_path / "front_template_v1.npz"
    np.savez_compressed(mutated_front_tpl, features=d["features"], meta=np.array(meta, dtype=object))

    with H.replay_context():
        with pytest.raises(ValueError):
            ac.compare_dual_streams(mutated_front_tpl, SIDE_TPL, FRONT_SRC, SIDE_SRC, pose_variant="full")
