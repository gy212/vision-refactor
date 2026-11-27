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
import threading
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation import (
    FeedbackConfig,
    LEGACY_DEFAULT_FEATURE_LAYOUT,
    compare_dual_streams,
)  # noqa: E402
from core.pose import POSE33_V3  # noqa: E402
from core.pose.features import extract_features as _extract_pose_features  # noqa: E402
from core.evaluation import subsequence_dtw, subsequence_dtw_with_path  # noqa: E402
from core.evaluation.matching import dtw as dtw_mod  # noqa: E402
from core.evaluation.matching import dual as dual_mod  # noqa: E402
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
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 3.0
        return 0.0

    def read(self):
        if self.index >= 3:
            return False, None
        self.index += 1
        return True, object()

    def set(self, prop, value) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
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
    # P2：解码与引擎构造统一在 core.media.video_source / core.pose.extract。
    from core.media import video_source
    from core.pose import extract

    monkeypatch.setattr(video_source.cv2, "VideoCapture", _TrackedCapture)
    monkeypatch.setattr(extract, "create_engine", lambda spec, **_kw: _TrackedPipeline(cfg=spec))
    yield
    _TrackedPipeline.fail_on_infer = False
    _TrackedPipeline.cancel_evt = None


def test_single_thread_extraction_closes_resources_on_success(tracked_extraction):
    features, fps, view_scores = _extract_pose_features(
        Path("tracked-success.mp4"),
        pose_variant="full",
        workers=1,
    )

    assert features.shape == (3, *POSE33_V3.shape)
    assert fps == pytest.approx(25.0)
    assert view_scores is None
    assert len(_TrackedCapture.instances) == 1
    assert _TrackedCapture.instances[0].released is True
    assert len(_TrackedPipeline.instances) == 1
    assert _TrackedPipeline.instances[0].closed is True


def test_single_thread_extraction_closes_resources_on_inference_error(
    tracked_extraction,
):
    _TrackedPipeline.fail_on_infer = True

    with pytest.raises(RuntimeError, match="inference failed"):
        _extract_pose_features(
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

    monkeypatch.setattr(dual_mod, "score_view", unexpected_score)

    with pytest.raises(InterruptedError, match="姿态特征提取已取消"):
        compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            Path("tracked-front.mp4"),
            Path("tracked-side.mp4"),
            pose_variant="full",
            stop_evt=stop_evt,
        )

    assert 1 <= len(_TrackedCapture.instances) <= 2
    assert all(item.released for item in _TrackedCapture.instances)
    assert 1 <= len(_TrackedPipeline.instances) <= 2
    assert all(item.closed for item in _TrackedPipeline.instances)

    # 显式串行模式取消：首路即停，不打开第二路
    _TrackedCapture.instances.clear()
    _TrackedPipeline.instances.clear()
    serial_stop = Event()
    _TrackedPipeline.cancel_evt = serial_stop
    with pytest.raises(InterruptedError, match="姿态特征提取已取消"):
        compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            Path("tracked-front.mp4"),
            Path("tracked-side.mp4"),
            pose_variant="full",
            stop_evt=serial_stop,
            parallel=False,
        )
    assert len(_TrackedCapture.instances) == 1
    assert _TrackedCapture.instances[0].released is True
    assert len(_TrackedPipeline.instances) == 1
    assert _TrackedPipeline.instances[0].closed is True


def test_cancel_during_front_scoring_skips_side_and_returns_no_result(monkeypatch):
    stop_evt = Event()
    front_seq = np.zeros((3, *POSE33_V3.shape), dtype=np.float32)
    side_seq = np.ones((3, *POSE33_V3.shape), dtype=np.float32)
    extracted = iter(((front_seq, 30.0, None), (side_seq, 30.0, None)))
    score_calls: list[np.ndarray] = []

    monkeypatch.setattr(
        dual_mod, "extract_features", lambda *_args, **_kwargs: next(extracted)
    )  # P3：双流比对迁入 matching.dual

    def score_once_then_cancel(seq, *_args, **_kwargs):
        score_calls.append(seq)
        if len(score_calls) == 1:
            stop_evt.set()
            return 0.9, [], None
        pytest.fail("side scoring must not run after front scoring cancels")

    monkeypatch.setattr(dual_mod, "score_view", score_once_then_cancel)

    with pytest.raises(InterruptedError, match="动作比对已取消"):
        compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            Path("front-scoring-cancel.mp4"),
            Path("side-scoring-cancel.mp4"),
            pose_variant="full",
            stop_evt=stop_evt,
        )

    assert len(score_calls) == 1
    assert score_calls[0] is front_seq


@pytest.mark.parametrize("dtw", [subsequence_dtw, subsequence_dtw_with_path])
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


@pytest.mark.parametrize("dtw", [subsequence_dtw, subsequence_dtw_with_path])
def test_dtw_cancel_event_is_checked_after_local_cost(monkeypatch, dtw):
    stop_evt = Event()
    original_local_cost = dtw_mod._dtw_local_cost  # P3：DTW 迁入 evaluation.matching.dtw

    def local_cost_then_cancel(query, seq):
        result = original_local_cost(query, seq)
        stop_evt.set()
        return result

    monkeypatch.setattr(dtw_mod, "_dtw_local_cost", local_cost_then_cancel)
    query = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    seq = np.arange(40, dtype=np.float32).reshape(5, 4, 2)

    with pytest.raises(InterruptedError, match="DTW 评分已取消"):
        dtw(query, seq, stop_evt=stop_evt)


def test_multi_worker_extraction_closes_all_resources_on_success(tracked_extraction):
    features, fps, view_scores = _extract_pose_features(
        Path("tracked-multi-success.mp4"),
        pose_variant="full",
        workers=2,
    )

    assert features.shape == (3, *POSE33_V3.shape)
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
        _extract_pose_features(
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
        _extract_pose_features(
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


@pytest.mark.parametrize("pose_variant", ["full", "heavy"])
def test_no_split_and_high_score_for_matching_streams(registered_videos, monkeypatch, pose_variant):
    front_n = int(_load_raw("front_src_raw.npz").shape[0])
    side_n = int(_load_raw("side_src_raw.npz").shape[0])
    configured = "heavy" if pose_variant == "full" else "full"
    monkeypatch.setattr(
        dual_mod,
        "load_feedback_config",
        lambda: FeedbackConfig(pose_variant=configured),
    )
    judge_views = dual_mod.judge_views

    def judge_with_actual_model(views, action, stance, cfg):
        assert cfg.pose_variant == pose_variant
        return judge_views(views, action, stance, cfg)

    monkeypatch.setattr(dual_mod, "judge_views", judge_with_actual_model)

    with H.replay_context():
        res = compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            FRONT_SRC,
            SIDE_SRC,
            pose_variant=pose_variant,
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
    assert res.front_rule_checks is not None
    assert res.side_rule_checks is not None
    assert res.front_joint_errors is not None
    assert res.side_joint_errors is not None


def test_swapped_streams_score_worse_than_correct_pairing(registered_videos):
    """正确配对应明显优于把 front/side 视频喂反，防止通道接反的回归。"""
    with H.replay_context():
        correct = compare_dual_streams(
            FRONT_TPL, SIDE_TPL, FRONT_SRC, SIDE_SRC, pose_variant="full"
        )
        swapped = compare_dual_streams(
            FRONT_TPL, SIDE_TPL, SIDE_SRC, FRONT_SRC, pose_variant="full"
        )

    assert correct.front_score > swapped.front_score
    assert correct.side_score > swapped.side_score


def test_raw_series_extracted_once_per_stream_no_cross_contamination(
    registered_videos, monkeypatch
):
    """独立 raw_series 闭包必须分别绑定各自视频路径：各调用一次、路径不串流。"""
    calls: list[Path] = []
    orig = dual_mod.extract_pose_raw_series

    def spy(video_path, *, pose_variant):
        calls.append(Path(video_path))
        return orig(video_path, pose_variant=pose_variant)

    monkeypatch.setattr(dual_mod, "extract_pose_raw_series", spy)

    with H.replay_context():
        compare_dual_streams(
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
    meta["feature_layout"] = LEGACY_DEFAULT_FEATURE_LAYOUT
    mutated_front_tpl = tmp_path / "front_template_v1.npz"
    np.savez_compressed(
        mutated_front_tpl, features=d["features"], meta=np.array(meta, dtype=object)
    )

    with H.replay_context():
        with pytest.raises(ValueError):
            compare_dual_streams(
                mutated_front_tpl, SIDE_TPL, FRONT_SRC, SIDE_SRC, pose_variant="full"
            )


def test_dual_streams_parallel_matches_serial_zero_ulp_and_zero_reextraction(
    registered_videos,
):
    """双流比对并发提点：真双路并发、与串行完全一致、特征与规则/关节误差 0 重复推理。"""
    from core.pose import extract as extract_mod

    with H.replay_context():
        serial = compare_dual_streams(
            FRONT_TPL,
            SIDE_TPL,
            FRONT_SRC,
            SIDE_SRC,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
            parallel=False,
        )
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=3.0)
        created = active = max_active = 0
        orig_create = extract_mod.create_engine

        def tracked(spec, **kw):
            nonlocal created
            with lock:
                created += 1
            pipe = orig_create(spec, **kw)
            orig_infer = pipe.infer
            first = [True]

            def wrapped(frame, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    if first[0]:
                        first[0] = False
                        barrier.wait()
                    return orig_infer(frame, **kwargs)
                finally:
                    with lock:
                        active -= 1

            pipe.infer = wrapped
            return pipe

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(extract_mod, "create_engine", tracked)
            parallel = compare_dual_streams(
                FRONT_TPL,
                SIDE_TPL,
                FRONT_SRC,
                SIDE_SRC,
                pose_variant="full",
                enable_rules=True,
                enable_error_analysis=True,
                parallel=True,
            )

    assert max_active == 2
    assert created == 2
    assert parallel == serial


def test_score_segment_parallel_matches_serial_and_zero_reextraction(registered_videos, tmp_path):
    """离线考试池复算 score_segment 接入共享并发原语：结果与串行完全一致且仅各推理一次。"""
    from core.pose import extract as extract_mod
    from core.services import score_segment

    def make_heavy(out: Path, src: Path) -> Path:
        d = np.load(src, allow_pickle=True)
        meta = {**d["meta"].item(), "pose_variant": "heavy"}
        np.savez_compressed(out, features=d["features"], meta=np.array(meta, dtype=object))
        return out

    front_tpls = [make_heavy(tmp_path / "高鞭腿.npz", FRONT_TPL)]
    side_tpls = [make_heavy(tmp_path / "直拳.npz", SIDE_TPL)]

    with H.replay_context():
        serial = score_segment(FRONT_SRC, SIDE_SRC, front_tpls, side_tpls, parallel=False)
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=3.0)
        created = max_active = active = 0
        orig_create = extract_mod.create_engine

        def tracked(spec, **kw):
            nonlocal created
            with lock:
                created += 1
            pipe = orig_create(spec, **kw)
            orig_infer = pipe.infer
            first = [True]

            def wrapped(frame, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    if first[0]:
                        first[0] = False
                        barrier.wait()
                    return orig_infer(frame, **kwargs)
                finally:
                    with lock:
                        active -= 1

            pipe.infer = wrapped
            return pipe

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(extract_mod, "create_engine", tracked)
            parallel = score_segment(FRONT_SRC, SIDE_SRC, front_tpls, side_tpls, parallel=True)

    assert max_active == 2
    assert created == 2
    assert parallel == serial


def test_exam_pool_skips_parallel_warmup_when_raw_or_windows_injected(tmp_path, monkeypatch):
    """保留默认 compare=match_video 但注入假 raw_series/compare_windows 时不得误调真实预热。"""
    from types import SimpleNamespace
    from core.evaluation import match_video
    from core.evaluation.scoring import exam_pool as exam_pool_mod
    from core.pose import Pose33RawSeries

    tpl = tmp_path / "上步前后手直拳组合.npz"
    np.savez_compressed(
        tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array({"pose_variant": "heavy", "feature_layout": "pose33_v3"}, dtype=object),
    )

    def fail_warmup(*_args, **_kwargs):
        raise AssertionError("注入假依赖时不得触发真实 extract_pose_pair 预热")

    monkeypatch.setattr(exam_pool_mod, "extract_pose_pair", fail_warmup)
    monkeypatch.setattr(exam_pool_mod, "action_side_repetitions", lambda *_a: [(0.5, 1.0)])
    monkeypatch.setattr(
        exam_pool_mod,
        "load_feedback_config",
        lambda: (_ for _ in ()).throw(ValueError("skip rules")),
    )

    fake_raw = Pose33RawSeries(
        landmarks=np.zeros((60, 33, 4), dtype=np.float32),
        meta={
            "fps": 30.0,
            "width": 64,
            "height": 64,
            "frame_count": 60,
            "start_frame": 0,
        },
    )
    fake_windows_calls: list[Path] = []

    def fake_compare_windows(templates, video, _windows, **_kw):
        fake_windows_calls.append(Path(video))
        return [[SimpleNamespace(score=0.82, start_frame=1, end_frame=20)] for _ in templates]

    res = exam_pool_mod.score_exam_pools(
        [tpl],
        [tpl],
        Path("nonexistent_front.mp4"),
        Path("nonexistent_side.mp4"),
        compare=match_video,
        raw_series=lambda *_a, **_k: fake_raw,
        stop_evt=Event(),
        compare_windows=fake_compare_windows,
        parallel=True,
    )
    assert fake_windows_calls == [
        Path("nonexistent_front.mp4"),
        Path("nonexistent_side.mp4"),
    ]
    assert res.front_score == pytest.approx(0.82)
    assert res.side_score == pytest.approx(0.82)


def test_recording_service_exam_guard_switches_parallel_by_foreground_state(tmp_path, monkeypatch):
    """端到端验证 RecordingService + SegmentFinalizer + dual_segment_steps 自动感知前台录制态。"""
    import os
    from types import SimpleNamespace
    from core.evaluation.scoring import exam_pool as exam_pool_mod
    from core.recording import (
        ProcessingPlan,
        RecordingService,
        ResultSpec,
        SegmentFinalizer,
        SegmentMeta,
        SegmentRecorder,
        StepDeps,
        dual_segment_steps,
    )

    tpl = tmp_path / "直拳.npz"
    np.savez_compressed(
        tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array({"pose_variant": "heavy", "feature_layout": "pose33_v3"}, dtype=object),
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 4)  # 弱 CPU 考场机（< 8 核）

    parallel_seen: list[bool] = []
    transcode_gate = Event()
    transcode_waiting = Event()
    completed = [Event(), Event()]
    done_idx = [0]

    def fake_score(*_args, **kwargs):
        parallel_arg = kwargs["parallel"]
        assert callable(parallel_arg)
        parallel_seen.append(bool(parallel_arg()))
        return SimpleNamespace(
            front_score=0.9,
            side_score=0.9,
            combined_score=0.9,
            combined_percent=90,
            front_matches=(),
            side_matches=(),
            front_segment=(0, 4),
            side_segment=(0, 4),
            action_scores=(),
            rule_checks=(),
            rule_error="",
        )

    monkeypatch.setattr(exam_pool_mod, "score_exam_pools", fake_score)

    class StubWriter:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"stub")

        def write(self, _frame) -> None:
            pass

        def release(self) -> None:
            pass

    def writer_factory(path, _fps, _size, _rot):
        w = StubWriter(path)
        return w, w.path, "fake"

    seg_num = [0]

    def gated_transcode(path: Path, _stop: Event) -> Path:
        if seg_num[0] == 2:
            transcode_waiting.set()
            assert transcode_gate.wait(3.0)
        return path

    def on_update(update) -> None:
        if update.status == "completed":
            idx = done_idx[0]
            done_idx[0] += 1
            if idx < len(completed):
                completed[idx].set()

    deps = StepDeps(
        transcode=gated_transcode,
        video_validator=lambda _p: True,
        model_available=lambda: True,
    )

    def plan_for(_segment) -> ProcessingPlan:
        steps = dual_segment_steps(
            record_skeleton=False,
            auto_compare=True,
            front_templates=(tpl,),
            side_templates=(tpl,),
            deps=deps,
        )
        return ProcessingPlan(steps, result=ResultSpec(front_template=tpl, side_template=tpl))

    finalizer = SegmentFinalizer(on_update=on_update)
    service = RecordingService(
        finalizer, plan_for=plan_for, recorder=SegmentRecorder(writer_factory)
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        service.begin_session({"front": (16, 16), "side": (16, 16)}, 30.0)
        # 第 1 段：结束后无后续前台录制（recorder 为 idle）-> 即使 4 核仍允许 parallel=True
        seg_num[0] = 1
        assert service.start(SegmentMeta(tmp_path / "rec"))
        service.write({"front": frame, "side": frame})
        service.finish()
        assert completed[0].wait(3.0)
        assert parallel_seen == [True]

        # 第 2 段：卡在转码阶段，同时前台开启第 3 段录制 -> 放行进入 ScoreExamPool 时自动降为 parallel=False
        seg_num[0] = 2
        assert service.start(SegmentMeta(tmp_path / "rec"))
        service.write({"front": frame, "side": frame})
        service.finish()
        assert transcode_waiting.wait(3.0)
        assert service.start(SegmentMeta(tmp_path / "rec"))
        assert service.state == "recording"
        transcode_gate.set()
        assert completed[1].wait(3.0)
        assert parallel_seen == [True, False]
        service.finish(discard=True)
    finally:
        transcode_gate.set()
        service.close_session()
        finalizer.close(2.0)


def test_can_parallel_pose_pair_and_score_exam_pool_low_cpu_guard(tmp_path, monkeypatch):
    """考试后台资源守卫：前台录制且 CPU < 8 核退回串行；>= 8 核或未录制允许并行。"""
    import os
    from types import SimpleNamespace
    from core.evaluation.scoring import exam_pool as exam_pool_mod
    from core.foundation.tasks import CancelToken
    from core.pose import can_parallel_pose_pair
    from core.recording import ScoreExamPool, StepContext

    assert can_parallel_pose_pair(foreground_recording=False, cpu_count=2) is True
    assert can_parallel_pose_pair(foreground_recording=True, cpu_count=4) is False
    assert can_parallel_pose_pair(foreground_recording=True, cpu_count=7) is False
    assert can_parallel_pose_pair(foreground_recording=True, cpu_count=8) is True
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert can_parallel_pose_pair(foreground_recording=True) is False
    with pytest.raises(TypeError):
        can_parallel_pose_pair(foreground_recording=True, cpu_count=True)  # type: ignore[arg-type]

    tpl = tmp_path / "直拳.npz"
    np.savez_compressed(
        tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array({"pose_variant": "heavy", "feature_layout": "pose33_v3"}, dtype=object),
    )
    recorded_parallel: list[bool] = []
    captured_predicates: list = []

    def fake_score(*_args, **kwargs):
        parallel_arg = kwargs["parallel"]
        assert callable(parallel_arg)
        captured_predicates.append(parallel_arg)
        recorded_parallel.append(bool(parallel_arg()))
        return SimpleNamespace(
            front_score=0.8,
            side_score=0.8,
            combined_score=0.8,
            combined_percent=80,
            front_matches=(),
            side_matches=(),
            front_segment=(0, 4),
            side_segment=(0, 4),
            action_scores=(),
            rule_checks=(),
            rule_error="",
        )

    monkeypatch.setattr(exam_pool_mod, "score_exam_pools", fake_score)
    ctx = StepContext(
        segment_id="seg-guard",
        segment_dir=tmp_path,
        views=("front", "side"),
        sources={"front": tmp_path / "f.mp4", "side": tmp_path / "s.mp4"},
        frames={"front": 10, "side": 10},
        errors={},
        stop_evt=Event(),
        publish=lambda *_a: None,
    )
    for cpus, recording, expected in (
        (4, True, False),
        (8, True, True),
        (4, False, True),
    ):
        monkeypatch.setattr(os, "cpu_count", lambda c=cpus: c)
        state = [recording]
        step = ScoreExamPool(
            (tpl,),
            (tpl,),
            model_available=lambda: True,
            foreground_recording=lambda s=state: s[0],
        )
        step.run(ctx, CancelToken())
        assert recorded_parallel[-1] is expected
        if cpus == 4 and not recording:
            state[0] = True
            assert captured_predicates[-1]() is False
            state[0] = False
            assert captured_predicates[-1]() is True


def _write_tiny_avi(path: Path, frames: int, *, size: tuple[int, int] = (32, 32)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    assert writer.isOpened()
    try:
        for i in range(frames):
            writer.write(np.full((size[1], size[0], 3), (i * 19) % 256, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_score_exam_pool_midflight_recording_throttles_low_cpu_parallel_infer(
    tmp_path, monkeypatch
):
    """< 8 核考场 idle 空窗已并发拉起两路 infer 后，中途开录立即收口至 <= 1，恢复 idle 重新并发。"""
    import os
    import time
    from core.foundation.tasks import CancelToken
    from core.pose import extract as extract_mod
    from core.recording import ScoreExamPool, StepContext

    tpl = tmp_path / "直拳.npz"
    np.savez_compressed(
        tpl,
        features=np.zeros((5, 22, 2), dtype=np.float32),
        meta=np.array({"pose_variant": "heavy", "feature_layout": "pose33_v3"}, dtype=object),
    )
    front_v = _write_tiny_avi(tmp_path / "front.avi", frames=10)
    side_v = _write_tiny_avi(tmp_path / "side.avi", frames=10)

    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    recording = [False]
    lock = threading.Lock()
    idle_barrier = threading.Barrier(2, timeout=3.0)
    rec_start_barrier = threading.Barrier(2, timeout=3.0)
    resume_barrier = threading.Barrier(2, timeout=3.0)
    active = 0
    phase_max = {"idle": 0, "recording": 0, "resumed": 0}
    recording_frames = 0
    recording_switched = [False]
    resumed_switched = [False]

    class TimedPipe:
        def __init__(self) -> None:
            self.idx = 0

        def infer(self, _frame, *, timestamp_ms=None):
            nonlocal active, recording_frames
            idx = self.idx
            self.idx += 1
            if idx == 0:
                with lock:
                    active += 1
                    phase_max["idle"] = max(phase_max["idle"], active)
                try:
                    idle_barrier.wait()
                finally:
                    with lock:
                        active -= 1
                # 两路完成首帧后同步切入前台录制态，避免遗留帧干扰相位归属
                if rec_start_barrier.wait() == 0 and not recording_switched[0]:
                    recording_switched[0] = True
                    recording[0] = True
                return None, []
            if 1 <= idx <= 4:
                with lock:
                    active += 1
                    phase_max["recording"] = max(phase_max["recording"], active)
                    recording_frames += 1
                try:
                    time.sleep(0.015)
                finally:
                    with lock:
                        active -= 1
                with lock:
                    if recording_frames == 8 and not resumed_switched[0]:
                        resumed_switched[0] = True
                        recording[0] = False
                return None, []
            if idx == 5:
                with lock:
                    active += 1
                    phase_max["resumed"] = max(phase_max["resumed"], active)
                try:
                    resume_barrier.wait()
                finally:
                    with lock:
                        active -= 1
            return None, []

        def close(self) -> None:
            pass

    created = [0]
    monkeypatch.setattr(
        extract_mod,
        "create_engine",
        lambda _spec, **_kw: (created.__setitem__(0, created[0] + 1), TimedPipe())[1],
    )
    ctx = StepContext(
        segment_id="seg-midflight",
        segment_dir=tmp_path,
        views=("front", "side"),
        sources={"front": front_v, "side": side_v},
        frames={"front": 10, "side": 10},
        errors={},
        stop_evt=Event(),
        publish=lambda *_a: None,
        foreground_recording=lambda: recording[0],
    )
    step = ScoreExamPool((tpl,), (tpl,), model_available=lambda: True)
    outcome = step.run(ctx, CancelToken())

    assert outcome is not None and outcome.status == "completed"
    assert created[0] == 2  # 两路预热后完全命中 pose_cache，0 重复推理
    assert phase_max["idle"] == 2
    assert recording_frames == 8
    assert phase_max["recording"] == 1
    assert phase_max["resumed"] == 2

    # 8 核下即使前台全程录制也保持并发
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    recording[0] = True
    hi_barrier = threading.Barrier(2, timeout=3.0)
    hi_active = hi_max = 0

    class HiCpuPipe:
        def __init__(self) -> None:
            self.first = True

        def infer(self, _frame, *, timestamp_ms=None):
            nonlocal hi_active, hi_max
            with lock:
                hi_active += 1
                hi_max = max(hi_max, hi_active)
            try:
                if self.first:
                    self.first = False
                    hi_barrier.wait()
                return None, []
            finally:
                with lock:
                    hi_active -= 1

        def close(self) -> None:
            pass

    monkeypatch.setattr(extract_mod, "create_engine", lambda _spec, **_kw: HiCpuPipe())
    step.run(ctx, CancelToken())
    assert hi_max == 2


def test_extract_pose_pair_hardlink_shares_pose_cache_both_directions(tmp_path, monkeypatch):
    """硬链接同物理文件（resolve 路径不同）在 pose_cache 中双键互通，0 重复推理；不同文件仍隔离。"""
    import os
    from core.pose import (
        FORMAL,
        extract_pose_pair,
        extract_pose_sequence,
        formal_engine,
        pose_cache,
    )
    from core.pose import extract as extract_mod

    orig = _write_tiny_avi(tmp_path / "front.avi", frames=5)
    hardlink = tmp_path / "side_hardlink.avi"
    other = _write_tiny_avi(tmp_path / "other.avi", frames=5)
    os.link(orig, hardlink)
    assert orig.samefile(hardlink)
    assert orig.resolve() != hardlink.resolve()

    created = [0]

    class CountingPipe:
        def infer(self, _frame, *, timestamp_ms=None):
            return None, []

        def close(self) -> None:
            pass

    def fake_create(_spec, **_kw):
        created[0] += 1
        return CountingPipe()

    monkeypatch.setattr(extract_mod, "create_engine", fake_create)
    spec = formal_engine("full")

    # 正向：extract_pose_pair 提取 orig 后同步写入 hardlink 键
    with pose_cache():
        elapsed = [-1.0, -1.0]
        s1, s2 = extract_pose_pair(orig, hardlink, spec, FORMAL, parallel=True, elapsed_out=elapsed)
        assert s1 is s2
        assert created[0] == 1
        assert elapsed[0] >= 0.0 and elapsed[1] == 0.0
        assert extract_pose_sequence(orig, spec, FORMAL) is s1
        assert extract_pose_sequence(hardlink, spec, FORMAL) is s1
        assert created[0] == 1

    # 反向：先缓存 hardlink，extract_pose_pair(orig, hardlink) 命中并补齐 orig 键
    with pose_cache():
        pre = extract_pose_sequence(hardlink, spec, FORMAL)
        assert created[0] == 2
        elapsed = [-1.0, -1.0]
        h1, h2 = extract_pose_pair(orig, hardlink, spec, FORMAL, parallel=True, elapsed_out=elapsed)
        assert h1 is pre and h2 is pre
        assert elapsed == [0.0, 0.0]
        assert extract_pose_sequence(orig, spec, FORMAL) is pre
        assert created[0] == 2

        # 负向：不同物理文件仍独立缓存、各自提取
        d1, d2 = extract_pose_pair(orig, other, spec, FORMAL, parallel=True)
        assert d1 is pre and d2 is not pre
        assert created[0] == 3


def test_extract_pose_pair_subthread_cancel_spawn_failure_and_progress_contracts(
    tmp_path, monkeypatch
):
    """覆盖子线程直抛 Cancelled、第二路 spawn 失败清理、双路进度单调有界与注入工厂顺序/独立计时。"""
    import time
    from core.foundation.tasks import Cancelled, TaskGroup
    from core.media import video_source as video_source_mod
    from core.pose import FORMAL, PRACTICE_V4, extract_pose_pair, formal_engine
    from core.pose import extract as extract_mod

    short_v = _write_tiny_avi(tmp_path / "short.avi", frames=5)
    long_v = _write_tiny_avi(tmp_path / "long.avi", frames=30)
    spec = formal_engine("full")

    # 1) 子线程引擎直抛 Cancelled 级联终止另一路并传播 Cancelled（非 AssertionError）
    sibling_frames = [0]

    class CancelFirstEngine:
        def infer(self, _frame, *, timestamp_ms=None):
            raise Cancelled("engine_abort")

        def close(self) -> None:
            pass

    class SlowSiblingEngine:
        def infer(self, _frame, *, timestamp_ms=None):
            sibling_frames[0] += 1
            time.sleep(0.02)
            return None, []

        def close(self) -> None:
            pass

    cancel_pipes = iter([CancelFirstEngine(), SlowSiblingEngine()])
    with pytest.raises(Cancelled, match="engine_abort"):
        extract_pose_pair(
            short_v,
            long_v,
            spec,
            FORMAL,
            parallel=True,
            engine_factory=lambda: next(cancel_pipes),
        )
    assert sibling_frames[0] < 30

    # 2) 第二路 spawn 失败：取消并 join 第一路、释放全部资源且重抛原始异常
    first_handle_ref: list = []
    first_started = threading.Event()
    first_closed = [False]
    orig_spawn = TaskGroup.spawn
    spawn_calls = [0]

    def fail_second_spawn(self, name, fn, **kwargs):
        spawn_calls[0] += 1
        if spawn_calls[0] == 2:
            assert first_started.wait(3.0)
            raise RuntimeError("thread-spawn-failed")
        handle = orig_spawn(self, name, fn, **kwargs)
        first_handle_ref.append(handle)
        return handle

    class SlowPipe:
        def infer(self, _frame, *, timestamp_ms=None):
            first_started.set()
            time.sleep(0.02)
            return None, []

        def close(self) -> None:
            first_closed[0] = True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TaskGroup, "spawn", fail_second_spawn)
        with pytest.raises(RuntimeError, match="thread-spawn-failed"):
            extract_pose_pair(short_v, long_v, spec, FORMAL, parallel=True, engine_factory=SlowPipe)
    assert len(first_handle_ref) == 1 and first_handle_ref[0].done is True
    assert first_closed[0] is True

    # 3) 双路 progress 在长短悬殊（并行与串行）下绝不先报 100% 再倒退，且 done 单调有界
    class NoopPipe:
        def infer(self, _frame, *, timestamp_ms=None):
            return None, []

        def close(self) -> None:
            pass

    for is_parallel in (True, False):
        events: list[tuple[int, int]] = []
        extract_pose_pair(
            short_v,
            long_v,
            spec,
            FORMAL,
            parallel=is_parallel,
            progress=lambda d, t: events.append((d, t)),
            engine_factory=NoopPipe,
        )
        assert events and events[-1] == (35, 35)
        dones = [d for d, _ in events]
        assert dones == sorted(dones)
        ratios = [d / t for d, t in events if t > 0]
        assert ratios == sorted(ratios)
        assert all(0 <= d <= t for d, t in events if t > 0)
        assert all(not (t > 0 and d == t) for d, t in events[:-1])

    # 4) 单路 total==0（未知帧数）时绝不出现 done > total > 0
    class UnknownTotalCap:
        def __init__(self, path) -> None:
            self.path = str(path)
            self.i = 0
            self.known = self.path.endswith("known_total.avi")

        def isOpened(self) -> bool:
            return True

        def get(self, prop) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 20.0 if self.known else 0.0
            return 0.0

        def read(self):
            if self.i >= 20:
                return False, None
            self.i += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    unknown_events: list[tuple[int, int]] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(video_source_mod.cv2, "VideoCapture", UnknownTotalCap)
        extract_pose_pair(
            "no_total.avi",
            "known_total.avi",
            spec,
            FORMAL,
            parallel=True,
            progress=lambda d, t: unknown_events.append((d, t)),
            engine_factory=NoopPipe,
        )
    assert unknown_events
    assert all(t == 0 for _, t in unknown_events)

    # 5) 注入工厂严格先正面后侧面 + 逐路独立 elapsed_out + 异常长度快速失败
    order_log: list[str] = []

    class TagPipe:
        def __init__(self, tag: str, delay_s: float) -> None:
            self.tag = tag
            self.delay_s = delay_s
            self.first = True

        def infer(self, _frame, *, timestamp_ms=None):
            if self.first:
                self.first = False
                order_log.append(self.tag)
                if self.delay_s > 0:
                    time.sleep(self.delay_s)
            return None, []

        def close(self) -> None:
            pass

    for is_parallel in (True, False):
        order_log.clear()
        assigned: list[tuple[str, int]] = []
        front_pipe = TagPipe("front", 0.0)
        side_pipe = TagPipe("side", 0.08)
        pipes = iter([front_pipe, side_pipe])
        elapsed = [0.0, 0.0]

        def tracked_factory():
            pipe = next(pipes)
            assigned.append((pipe.tag, len(assigned)))
            return pipe

        extract_pose_pair(
            short_v,
            short_v.with_name("long.avi"),
            spec,
            PRACTICE_V4,
            parallel=is_parallel,
            engine_factory=tracked_factory,
            elapsed_out=elapsed,
        )
        assert assigned == [("front", 0), ("side", 1)]
        assert elapsed[0] < 0.05
        assert elapsed[1] >= 0.07

    with pytest.raises(ValueError, match="elapsed_out"):
        extract_pose_pair(short_v, long_v, spec, FORMAL, elapsed_out=[0.0])

    # 6) 默认路径单次主线程 default_models_dir 预播种（工作线程不重复调用）
    dir_threads: list[int] = []
    main_tid = threading.get_ident()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            extract_mod,
            "default_models_dir",
            lambda: (dir_threads.append(threading.get_ident()), tmp_path)[1],
        )
        mp.setattr(extract_mod, "create_engine", lambda _s, **_kw: NoopPipe())
        extract_pose_pair(short_v, long_v, spec, FORMAL, parallel=True)
    assert dir_threads == [main_tid]
