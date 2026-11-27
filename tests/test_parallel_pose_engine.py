# -*- coding: utf-8 -*-
"""ParallelPoseEngine 单元/属性测试（不依赖 MediaPipe 模型）。"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation import FeedbackConfig, analyze_practice, extract_series  # noqa: E402
from core.evaluation import practice as practice_mod  # noqa: E402
from core.foundation.tasks import Cancelled, CancelToken  # noqa: E402
from core.media import video_source as video_source_mod  # noqa: E402
from core.pose import (  # noqa: E402
    FORMAL,
    EngineSpec,
    InferResult,
    ParallelPoseEngine,
    PoseSequence,
    SourceMeta,
    extract_pose_pair,
    extract_pose_raw_series,
    extract_pose_sequence,
    formal_engine,
    normalize_pose_xy_v3,
    normalize_pose_xy_v3_batch,
    pose_cache,
)
from core.pose import extract as extract_mod  # noqa: E402
from core.pose.sequence import SegmentWarmup  # noqa: E402
from tests import golden_harness as H  # noqa: E402


class FakePipeline:
    """假管线：把帧的首像素值当作 id 回写到 actions，便于校验帧序对应关系。"""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay
        self.closed = False

    def annotate(self, frame_bgr, *, timestamp_ms=None):
        if self._delay:
            time.sleep(self._delay)
        val = int(frame_bgr[0, 0, 0])
        return frame_bgr, [f"id={val}"]

    def close(self) -> None:
        self.closed = True


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value % 256, dtype=np.uint8)


def _drain_ordered(engine: ParallelPoseEngine, timeout_total: float = 10.0) -> list[InferResult]:
    out: list[InferResult] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_total:
        res = engine.get(timeout=0.1)
        if res is not None:
            out.append(res)
            continue
        if engine.is_drained():
            break
    return out


def test_offline_preserves_order_and_completeness():
    """离线模式（不丢帧）：所有帧都处理，且严格按 index 升序输出。"""
    n = 50
    engine = ParallelPoseEngine(
        pipeline_factory=lambda: FakePipeline(delay=0.001),
        workers=4,
        drop_when_full=False,
    )
    engine.start()

    def produce():
        for i in range(n):
            engine.submit(_frame(i))
        engine.signal_input_done()

    threading.Thread(target=produce, daemon=True).start()
    out = _drain_ordered(engine)
    engine.close()

    indices = [r.index for r in out]
    assert indices == list(range(n)), f"输出未按序或不完整: {indices[:10]}..."
    # 每帧的 annotated 内容应与其 index 对应（首像素 == index % 256）。
    for r in out:
        assert int(r.annotated[0, 0, 0]) == r.index % 256
    assert engine.stats["submitted"] == n
    assert engine.stats["dropped"] == 0
    assert engine.stats["emitted"] == n


def test_ordered_output_is_monotonic_with_many_workers():
    n = 80
    engine = ParallelPoseEngine(
        pipeline_factory=lambda: FakePipeline(delay=0.0),
        workers=8,
        drop_when_full=False,
    )
    engine.start()

    def produce():
        for i in range(n):
            engine.submit(_frame(i))
        engine.signal_input_done()

    threading.Thread(target=produce, daemon=True).start()
    out = _drain_ordered(engine)
    engine.close()

    indices = [r.index for r in out]
    assert indices == sorted(indices)
    assert indices == list(range(n))


def test_realtime_drop_when_full_no_index_gaps_in_output():
    """实时模式（丢帧）：提交数 = 处理数 + 丢弃数；输出 index 子序列严格递增。"""
    engine = ParallelPoseEngine(
        pipeline_factory=lambda: FakePipeline(delay=0.01),
        workers=2,
        drop_when_full=True,
        queue_factor=1,
    )
    engine.start()

    accepted = 0
    for i in range(200):
        idx = engine.submit(_frame(i))
        if idx is not None:
            accepted += 1
        time.sleep(0.001)  # 生产快于消费 → 触发丢帧
    engine.signal_input_done()

    out = _drain_ordered(engine)
    engine.close()

    stats = engine.stats
    assert stats["dropped"] > 0, "高产出速率下应当发生丢帧"
    assert stats["submitted"] + stats["dropped"] == 200
    assert stats["submitted"] == accepted
    indices = [r.index for r in out]
    assert indices == sorted(indices), "输出 index 必须单调递增"
    assert len(set(indices)) == len(indices), "不得有重复 index"


def test_error_in_pipeline_is_propagated():
    class BoomPipeline:
        def annotate(self, frame_bgr, *, timestamp_ms=None):
            raise RuntimeError("boom")

        def close(self):
            pass

    engine = ParallelPoseEngine(
        pipeline_factory=lambda: BoomPipeline(),
        workers=2,
        drop_when_full=False,
    )
    engine.start()
    engine.submit(_frame(1))
    engine.signal_input_done()
    # 给 worker 时间触发异常
    time.sleep(0.3)
    engine.close()
    err = engine.take_error()
    assert isinstance(err, RuntimeError)
    assert "boom" in str(err)


def test_pipelines_are_closed_on_close():
    created: list[FakePipeline] = []

    def factory():
        p = FakePipeline()
        created.append(p)
        return p

    engine = ParallelPoseEngine(pipeline_factory=factory, workers=3, drop_when_full=False)
    engine.start()
    engine.signal_input_done()
    out = _drain_ordered(engine)
    engine.close()
    assert out == []
    assert len(created) == 3
    assert all(p.closed for p in created)


def test_blocking_submit_returns_none_after_close():
    """不丢帧的 submit 阻塞在满队列上时，close 之后放弃该帧返回 None（PR #17 评审 1）。"""
    gate = threading.Event()

    class Blocked:
        def annotate(self, frame_bgr, *, timestamp_ms=None):
            gate.wait(5.0)
            return frame_bgr, []

        def close(self):
            pass

    engine = ParallelPoseEngine(
        pipeline_factory=Blocked, workers=1, drop_when_full=False, queue_factor=1
    )
    engine.start()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    results: list = []

    def produce():
        for _ in range(5):
            results.append(engine.submit(frame))

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    time.sleep(0.3)
    assert producer.is_alive()
    closer = threading.Thread(target=engine.close, daemon=True)
    closer.start()
    producer.join(1.0)
    assert not producer.is_alive()
    assert results[-1] is None
    gate.set()
    closer.join(5.0)


# --------------------------------------------------------------------------- #
# 批量 v3 姿态归一化与 PoseSequence.features（0 ULP + T=0 + 极端边界）
# --------------------------------------------------------------------------- #


def _dummy_meta(n: int, tracking: str = "sequential") -> SourceMeta:
    return SourceMeta(
        video="synthetic.mp4",
        fps=30.0,
        fps_reported=30.0,
        frame_count=n,
        width=640,
        height=480,
        engine=EngineSpec(pose_variant="full", running_mode="video", enable_hands=False),
        profile="formal",
        tracking=tracking,
        validity_policy="mediapipe_visibility",
        valid_conf_thr=0.5,
        frames_scanned=n,
    )


def _make_pose_sequence(
    landmarks: np.ndarray,
    detected: np.ndarray,
    *,
    decoded: np.ndarray | None = None,
    segments: tuple[SegmentWarmup, ...] = (),
) -> PoseSequence:
    n = int(landmarks.shape[0])
    return PoseSequence(
        landmarks=landmarks,
        detected=detected,
        decoded=np.ones((n,), dtype=bool) if decoded is None else decoded,
        frame_index=np.arange(n, dtype=np.int64),
        t_ms=np.arange(n, dtype=np.int64) * 33,
        source_wh=np.tile(np.array([[640, 480]], dtype=np.int64), (n, 1)),
        meta=_dummy_meta(n, "segmented" if segments else "sequential"),
        segments=segments,
    )


def _reference_normalize_v3_frame(row: np.ndarray) -> np.ndarray | None:
    """独立逐帧 v3 数学参考实现（不调用生产归一化函数，防止自调用同谋测试）。"""
    with np.errstate(all="ignore"):
        pts = np.asarray(row[:33, :2], dtype=np.float32)
        ls, rs, lh, rh = pts[11], pts[12], pts[23], pts[24]
        if not (
            np.isfinite(ls).all()
            and np.isfinite(rs).all()
            and np.isfinite(lh).all()
            and np.isfinite(rh).all()
        ):
            return None
        sh_c, hip_c = 0.5 * (ls + rs), 0.5 * (lh + rh)
        center = hip_c if np.isfinite(hip_c).all() else sh_c
        if not np.isfinite(center).all():
            return None
        sh_w, hip_w = float(np.linalg.norm(ls - rs)), float(np.linalg.norm(lh - rh))
        torso_len = float(np.linalg.norm(sh_c - hip_c))
        scale = torso_len if (np.isfinite(torso_len) and torso_len >= 0.02) else max(sh_w, hip_w)
        if (not np.isfinite(scale)) or scale < 0.02:
            return None
        width_ratio = float(sh_w / (torso_len + 1e-6)) if np.isfinite(torso_len) else 0.0
        v, target = (rs - ls, 0.0) if width_ratio >= 0.35 else (sh_c - hip_c, -float(np.pi) / 2.0)
        v_norm = float(np.linalg.norm(v))
        if (not np.isfinite(v_norm)) or v_norm < 1e-6:
            ca, sa = 1.0, 0.0
        else:
            rot = float(target - float(np.arctan2(float(v[1]), float(v[0]))))
            ca, sa = float(np.cos(rot)), float(np.sin(rot))
        rot_m = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        out = np.clip(((pts[11:33] - center) / float(scale)) @ rot_m.T, -5.0, 5.0).astype(
            np.float32, copy=False
        )
        if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
            return None
        return out


def _reference_sequential_v3(landmarks: np.ndarray, detected: np.ndarray) -> np.ndarray:
    n = int(landmarks.shape[0])
    out = np.zeros((n, 22, 2), dtype=np.float32)
    last: np.ndarray | None = None
    for t in range(n):
        f = _reference_normalize_v3_frame(landmarks[t]) if detected[t] else None
        if f is None:
            out[t] = last if last is not None else 0.0
        else:
            last = f
            out[t] = f
    return out


def _reference_segmented_v3(
    landmarks: np.ndarray,
    detected: np.ndarray,
    decoded: np.ndarray,
    segments: tuple[SegmentWarmup, ...],
) -> np.ndarray:
    n = int(landmarks.shape[0])
    out = np.zeros((n, 22, 2), dtype=np.float32)
    for seg in segments:
        last: np.ndarray | None = None
        for w in range(seg.landmarks.shape[0]):
            f = _reference_normalize_v3_frame(seg.landmarks[w]) if seg.detected[w] else None
            if f is not None:
                last = f
        for t in range(seg.seg_start, min(seg.seg_end + 1, n)):
            if not decoded[t]:
                break
            f = _reference_normalize_v3_frame(landmarks[t]) if detected[t] else None
            if f is None:
                out[t] = last if last is not None else 0.0
            else:
                last = f
                out[t] = f
    return out


def test_normalize_pose_xy_v3_batch_empty_and_validation():
    empty_lm = np.zeros((0, 33, 4), dtype=np.float64)
    empty_det = np.zeros((0,), dtype=bool)
    feats, ok = normalize_pose_xy_v3_batch(empty_lm, empty_det)
    assert feats.shape == (0, 22, 2) and feats.dtype == np.float32
    assert ok.shape == (0,) and ok.dtype == bool

    seq_empty = _make_pose_sequence(empty_lm, empty_det)
    out_seq = seq_empty.features(normalize_pose_xy_v3, (22, 2))
    assert out_seq.shape == (0, 22, 2) and out_seq.dtype == np.float32

    seg_empty = _make_pose_sequence(
        empty_lm, empty_det, segments=(SegmentWarmup(0, 0, empty_lm, empty_det),)
    )
    out_seg = seg_empty.features(normalize_pose_xy_v3, (22, 2))
    assert out_seg.shape == (0, 22, 2) and out_seg.dtype == np.float32

    with pytest.raises(ValueError, match="landmarks"):
        normalize_pose_xy_v3_batch(np.zeros((2, 10, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="detected"):
        normalize_pose_xy_v3_batch(
            np.zeros((2, 33, 4), dtype=np.float32), np.ones((3,), dtype=bool)
        )


def test_normalize_pose_xy_v3_batch_extreme_fallbacks_zero_ulp():
    """显式覆盖 lh+rh 溢出回退 sh_c、v_norm<1e-6 不旋转、NaN/Inf 隔离、0.02 临界与只读数组。"""
    base = np.zeros((7, 33, 4), dtype=np.float64)
    for t in range(7):
        base[t, :, 0] = np.linspace(0.3, 0.7, 33)
        base[t, :, 1] = np.linspace(0.2, 0.8, 33)
        base[t, 11, :2] = (0.4, 0.3)
        base[t, 12, :2] = (0.6, 0.32)
        base[t, 23, :2] = (0.43, 0.6)
        base[t, 24, :2] = (0.57, 0.62)
    # 0: 正常正面；1: lh+rh 溢出 float32 -> center 回退 sh_c，scale 回退 shoulder_w=2.0，no_rot 且输出有效
    base[1, 11, :2] = (0.0, 0.0)
    base[1, 12, :2] = (2.0, 0.0)
    base[1, 23, :2] = (2.0e38, 2.0e38)
    base[1, 24, :2] = (2.0e38, 2.0e38)
    # 2: side 模式且 sh_c 与 hip_c 极近但肩/髋足够宽（v_norm < 1e-6 触发不旋转回退）
    base[2, 11, :2] = (0.5, 0.5)
    base[2, 12, :2] = (0.5, 0.5)
    base[2, 23, :2] = (0.4, 0.5)
    base[2, 24, :2] = (0.6, 0.5 + 1e-8)
    # 3: 锚点含 NaN；4: 锚点含 Inf；5: scale 恰好为 np.float32(0.02)（f64 视角 < 0.02）
    base[3, 11, 0] = np.nan
    base[4, 24, 1] = np.inf
    base[5, 11, :2] = (0.0, 0.0)
    base[5, 12, :2] = (float(np.float32(0.02)), 0.0)
    base[5, 23, :2] = (0.0, 0.0)
    base[5, 24, :2] = (float(np.float32(0.02)), 0.0)
    # 6: 未检出帧带 NaN 污染，必须被 cand 隔离
    base[6, :, :2] = np.nan
    detected = np.array([True, True, True, True, True, True, False], dtype=bool)
    base.setflags(write=False)
    detected.setflags(write=False)

    b_out, b_ok = normalize_pose_xy_v3_batch(base, detected)
    assert b_ok[0] and b_ok[1] and b_ok[2]
    assert not b_ok[3] and not b_ok[4] and not b_ok[5] and not b_ok[6]
    assert normalize_pose_xy_v3(None) is None
    for t in range(7):
        ref = _reference_normalize_v3_frame(base[t]) if detected[t] else None
        single_arr = normalize_pose_xy_v3(base[t]) if detected[t] else None
        objs = [SimpleNamespace(x=float(base[t, j, 0]), y=float(base[t, j, 1])) for j in range(33)]
        single_obj = normalize_pose_xy_v3(objs) if detected[t] else None
        if ref is None:
            assert not b_ok[t] and np.all(b_out[t] == 0.0)
            assert single_arr is None and single_obj is None
        else:
            assert b_ok[t] and np.array_equal(b_out[t], ref)
            assert single_arr is not None and np.array_equal(single_arr, ref)
            assert single_obj is not None and np.array_equal(single_obj, ref)

    seq = _make_pose_sequence(base, detected)
    np.testing.assert_array_equal(
        seq.features(normalize_pose_xy_v3, (22, 2)), _reference_sequential_v3(base, detected)
    )


_EXTREME_FLOATS = st.one_of(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.sampled_from(
        [
            0.0,
            -0.0,
            1e-7,
            -1e-7,
            1e-6,
            0.02,
            float(np.float32(0.02)),
            0.020001,
            3e38,
            -3e38,
            1e19,
            -1e19,
            1e-20,
            float("nan"),
            float("inf"),
            float("-inf"),
        ]
    ),
)


@settings(max_examples=200, deadline=None)
@given(
    t=st.integers(min_value=0, max_value=8),
    dtype=st.sampled_from([np.float32, np.float64]),
    data=st.data(),
)
def test_normalize_pose_xy_v3_batch_hypothesis_zero_ulp(t, dtype, data):
    """属性测试：随机极端坐标、只读数组、顺序与分段 PoseSequence 与单帧 v3 逐位 0 ULP 全等。"""
    lm = data.draw(hnp.arrays(dtype=dtype, shape=(t, 33, 4), elements=_EXTREME_FLOATS))
    det = data.draw(hnp.arrays(dtype=bool, shape=(t,)))
    # 额外注入至少一帧合法骨架，提高有效路径覆盖率
    if t >= 2 and data.draw(st.booleans()):
        lm = lm.copy()
        idx = data.draw(st.integers(min_value=0, max_value=t - 1))
        lm[idx, :, :2] = 0.5
        lm[idx, 11, :2] = (0.4, 0.3)
        lm[idx, 12, :2] = (0.6, 0.31)
        lm[idx, 23, :2] = (0.42, 0.62)
        lm[idx, 24, :2] = (0.58, 0.61)
        det = det.copy()
        det[idx] = True
    lm.setflags(write=False)
    det.setflags(write=False)

    b_out, b_ok = normalize_pose_xy_v3_batch(lm, det)
    for i in range(t):
        ref = _reference_normalize_v3_frame(lm[i]) if det[i] else None
        single = normalize_pose_xy_v3(lm[i]) if det[i] else None
        if ref is None:
            assert not b_ok[i] and np.all(b_out[i] == 0.0) and single is None
        else:
            assert b_ok[i] and np.array_equal(b_out[i], ref)
            assert single is not None and np.array_equal(single, ref)

    seq = _make_pose_sequence(lm, det)
    np.testing.assert_array_equal(
        seq.features(normalize_pose_xy_v3, (22, 2)), _reference_sequential_v3(lm, det)
    )

    if t >= 2:
        mid = t // 2
        decoded = data.draw(hnp.arrays(dtype=bool, shape=(t,)))
        decoded.setflags(write=False)
        warm_lm = lm[:mid]
        warm_det = det[:mid]
        segments = (
            SegmentWarmup(
                0, mid - 1, np.zeros((0, 33, 4), dtype=dtype), np.zeros((0,), dtype=bool)
            ),
            SegmentWarmup(mid, t - 1, warm_lm, warm_det),
        )
        seg_seq = _make_pose_sequence(lm, det, decoded=decoded, segments=segments)
        np.testing.assert_array_equal(
            seg_seq.features(normalize_pose_xy_v3, (22, 2)),
            _reference_segmented_v3(lm, det, decoded, segments),
        )


# --------------------------------------------------------------------------- #
# 共享双摄并行提点原语 extract_pose_pair + 父线程 pose_cache + 资源守卫
# --------------------------------------------------------------------------- #


def _write_test_avi(
    path: Path,
    frames: int = 8,
    size=(64, 48),
    fps: float = 10.0,
    *,
    pixel_value: int | None = None,
) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for i in range(frames):
            val = pixel_value if pixel_value is not None else (i * 17) % 256
            writer.write(np.full((size[1], size[0], 3), val, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_extract_pose_pair_parallel_zero_ulp_and_parent_pose_cache_hit():
    """双路并发提点：真 2 路并发、与串行 0 ULP 一致、写回父线程 pose_cache 后下游 0 重复推理。"""
    fix_dir = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
    front_raw = np.asarray(
        np.load(fix_dir / "front_src_raw.npz", allow_pickle=True)["landmarks"], np.float32
    )
    side_raw = np.asarray(
        np.load(fix_dir / "side_src_raw.npz", allow_pickle=True)["landmarks"], np.float32
    )
    front_src, side_src = "golden://pair_front.mp4", "golden://pair_side.mp4"

    H.clear_registry()
    H.register_video(front_src, front_raw, fps=30.0)
    H.register_video(side_src, side_raw, fps=30.0)
    try:
        with H.replay_context():
            spec = formal_engine("full")
            ref_f, ref_s = extract_pose_pair(front_src, side_src, spec, FORMAL, parallel=False)

            lock = threading.Lock()
            barrier = threading.Barrier(2, timeout=3.0)
            active = max_active = created = infer_count = 0
            orig_create = extract_mod.create_engine

            def tracked_create(engine_spec, **kw):
                nonlocal created
                with lock:
                    created += 1
                pipe = orig_create(engine_spec, **kw)
                orig_infer = pipe.infer
                first_frame = [True]

                def wrapped_infer(frame, **kwargs):
                    nonlocal active, max_active, infer_count
                    with lock:
                        active += 1
                        max_active = max(max_active, active)
                        infer_count += 1
                    try:
                        if first_frame[0]:
                            first_frame[0] = False
                            barrier.wait()
                        return orig_infer(frame, **kwargs)
                    finally:
                        with lock:
                            active -= 1

                pipe.infer = wrapped_infer
                return pipe

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(extract_mod, "create_engine", tracked_create)
                with pose_cache():
                    par_f, par_s = extract_pose_pair(
                        front_src, side_src, spec, FORMAL, parallel=True
                    )
                    assert max_active == 2
                    assert created == 2
                    assert infer_count == len(front_raw) + len(side_raw)
                    # 下游在父线程再次提取与取 raw_series 必须 100% 命中缓存、0 次重提
                    hit_f = extract_pose_sequence(front_src, spec, FORMAL)
                    hit_s = extract_pose_sequence(side_src, spec, FORMAL)
                    raw_f = extract_pose_raw_series(front_src, pose_variant="full")
                    raw_s = extract_pose_raw_series(side_src, pose_variant="full")
                    again_f, again_s = extract_pose_pair(
                        front_src, side_src, spec, FORMAL, parallel=True
                    )
                    assert created == 2
                    assert infer_count == len(front_raw) + len(side_raw)
                    assert hit_f is par_f and hit_s is par_s
                    assert again_f is par_f and again_s is par_s
                    assert len(raw_f.landmarks) == len(front_raw)
                    assert len(raw_s.landmarks) == len(side_raw)

                # 退出 pose_cache 作用域后失效，重新触发推理（串行单路测试不用 barrier）
                barrier.abort()
                mp.setattr(extract_mod, "create_engine", orig_create)
                fresh = extract_pose_sequence(front_src, spec, FORMAL)
                np.testing.assert_array_equal(fresh.landmarks, ref_f.landmarks)

            for par, ref in ((par_f, ref_f), (par_s, ref_s)):
                np.testing.assert_array_equal(par.landmarks, ref.landmarks)
                np.testing.assert_array_equal(par.detected, ref.detected)
                np.testing.assert_array_equal(par.frame_index, ref.frame_index)
                np.testing.assert_array_equal(par.t_ms, ref.t_ms)
                np.testing.assert_array_equal(par.source_wh, ref.source_wh)
    finally:
        H.clear_registry()


def test_extract_pose_pair_same_file_dedup_and_partial_cache_hit(tmp_path, monkeypatch):
    """同物理文件（含路径别名）去重仅推理 1 次；单路已缓存时仅提另一路且不起并发线程。"""
    vid_a = _write_test_avi(tmp_path / "a.avi", frames=5)
    vid_b = _write_test_avi(tmp_path / "b.avi", frames=5)
    sub = tmp_path / "sub"
    sub.mkdir()
    alias_a = sub / ".." / "a.avi"
    created_threads: list[int] = []

    class StubPipe:
        def infer(self, _frame, *, timestamp_ms=None):
            return None, []

        def close(self) -> None:
            pass

    def fake_create(_spec, **_kw):
        created_threads.append(threading.get_ident())
        return StubPipe()

    monkeypatch.setattr(extract_mod, "create_engine", fake_create)
    spec = formal_engine("full")
    main_tid = threading.get_ident()

    with pose_cache():
        s1, s2 = extract_pose_pair(vid_a, alias_a, spec, FORMAL, parallel=True)
        assert s1 is s2
        assert created_threads == [main_tid]  # 同物理文件仅在主线程提取 1 次

        # 此时 vid_a 已命中缓存，仅 vid_b 未命中：直接主线程补提 vid_b，不额外起线程
        p1, p2 = extract_pose_pair(vid_a, vid_b, spec, FORMAL, parallel=True)
        assert p1 is s1
        assert len(p2) == 5
        assert created_threads == [main_tid, main_tid]


def test_extract_pose_pair_cancel_and_error_close_resources_without_cache_pollution(monkeypatch):
    """取消传播与单路异常快速失败：全部 capture/engine 必关闭，且不污染父线程 pose_cache。"""
    captures: list[object] = []
    pipes: list[object] = []

    class FakeCap:
        def __init__(self, _path) -> None:
            self.i = 0
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def get(self, prop) -> float:
            return {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: 20.0}.get(prop, 0.0)

        def read(self):
            if self.i >= 20:
                return False, None
            self.i += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    monkeypatch.setattr(video_source_mod.cv2, "VideoCapture", FakeCap)
    spec = formal_engine("full")

    # 1) 外部取消传播
    token = CancelToken()

    class CancelPipe:
        def __init__(self) -> None:
            self.closed = False
            pipes.append(self)

        def infer(self, _frame, *, timestamp_ms=None):
            token.cancel("user_stop")
            time.sleep(0.01)
            return None, []

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(extract_mod, "create_engine", lambda _s, **_kw: CancelPipe())
    with pose_cache():
        with pytest.raises(Cancelled):
            extract_pose_pair("cancel_f.mp4", "cancel_s.mp4", spec, FORMAL, cancel=token)
        assert captures and all(c.released for c in captures)
        assert pipes and all(p.closed for p in pipes)

        # 2) 单路异常级联取消另一路并清理资源，且缓存无残留
        captures.clear()
        pipes.clear()
        created = [0]

        class ErrorOncePipe:
            def __init__(self, idx: int) -> None:
                self.idx = idx
                self.closed = False
                pipes.append(self)

            def infer(self, _frame, *, timestamp_ms=None):
                if self.idx == 1:
                    raise RuntimeError("boom-on-stream")
                time.sleep(0.01)
                return None, []

            def close(self) -> None:
                self.closed = True

        def make_err(_s, **_kw):
            created[0] += 1
            return ErrorOncePipe(created[0])

        monkeypatch.setattr(extract_mod, "create_engine", make_err)
        with pytest.raises(RuntimeError, match="boom-on-stream"):
            extract_pose_pair("err_f.mp4", "err_s.mp4", spec, FORMAL)
        assert len(captures) >= 1 and all(c.released for c in captures)
        assert len(pipes) >= 1 and all(p.closed for p in pipes)

        # 确认失败未污染缓存：换正常引擎后仍会重新提取
        normal_calls = [0]

        class OkPipe:
            def infer(self, _frame, *, timestamp_ms=None):
                normal_calls[0] += 1
                return None, []

            def close(self) -> None:
                pass

        monkeypatch.setattr(extract_mod, "create_engine", lambda _s, **_kw: OkPipe())
        extract_pose_sequence("err_f.mp4", spec, FORMAL)
        assert normal_calls[0] == 20


def _practice_landmarks(offset: float = 0.0):
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
    coords = {
        0: (0.5, 0.15),
        11: (0.4, 0.3),
        12: (0.6, 0.3),
        13: (0.4, 0.45),
        14: (0.6, 0.45),
        15: (0.55 + offset, 0.45),
        16: (0.75, 0.45),
        23: (0.43, 0.6),
        24: (0.57, 0.6),
        25: (0.4, 0.78),
        26: (0.6, 0.78),
        27: (0.35, 0.96),
        28: (0.65, 0.96),
        29: (0.38, 0.98),
        30: (0.68, 0.98),
        31: (0.3, 0.96),
        32: (0.6, 0.96),
    }
    for idx, (x, y) in coords.items():
        pts[idx] = SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)
    return pts


def test_analyze_practice_parallel_runs_concurrently_and_matches_serial(tmp_path):
    """练习分析双摄入口接入共享并发原语：真双路并发且结论与串行完全一致。"""
    front = _write_test_avi(tmp_path / "front.avi", frames=16, size=(96, 96), fps=12.0)
    side = _write_test_avi(tmp_path / "side.avi", frames=16, size=(96, 96), fps=12.0)

    def strip_timing(result: dict) -> dict:
        cleaned = dict(result)
        cleaned["capture"] = {
            view: {k: v for k, v in info.items() if k != "analysisSeconds"}
            for view, info in result["capture"].items()
        }
        return cleaned

    class DeterministicPipe:
        def __init__(self, barrier: threading.Barrier | None = None) -> None:
            self._barrier = barrier
            self._first = True

        def infer(self, frame, *, timestamp_ms=None):
            if self._barrier is not None and self._first:
                self._first = False
                self._barrier.wait()
            return _practice_landmarks(float(frame[0, 0, 0]) * 1e-4), []

        def close(self) -> None:
            pass

    cfg = FeedbackConfig()
    serial = analyze_practice(
        "stance",
        "left",
        front,
        side,
        config=cfg,
        pipeline_factory=lambda: DeterministicPipe(),
        parallel=False,
    )
    barrier = threading.Barrier(2, timeout=3.0)
    parallel = analyze_practice(
        "stance",
        "left",
        front,
        side,
        config=cfg,
        pipeline_factory=lambda: DeterministicPipe(barrier),
        parallel=True,
    )
    for res in (serial, parallel):
        for view in ("front", "side"):
            secs = res["capture"][view]["analysisSeconds"]
            assert isinstance(secs, float) and 0.0 <= secs < 10.0
    assert strip_timing(parallel) == strip_timing(serial)


def test_analyze_practice_default_preseeds_once_and_hits_pose_cache(tmp_path):
    """默认 pipeline_factory=None 时主线程单次预播种、命中 pose_cache，且兼容外部 monkeypatch。"""
    front = _write_test_avi(tmp_path / "front.avi", frames=12, size=(96, 96), fps=12.0)
    side = _write_test_avi(tmp_path / "side.avi", frames=12, size=(96, 96), fps=12.0)
    main_tid = threading.get_ident()
    models_dir_threads: list[int] = []
    created: list[dict] = []
    infer_calls = [0]
    lock = threading.Lock()

    class CountingPipe:
        def infer(self, frame, *, timestamp_ms=None):
            with lock:
                infer_calls[0] += 1
            return _practice_landmarks(float(frame[0, 0, 0]) * 1e-4), []

        def close(self) -> None:
            pass

    def fake_models_dir() -> Path:
        with lock:
            models_dir_threads.append(threading.get_ident())
        return tmp_path

    def fake_create(spec, *, models_dir=None, download=True, **_kw):
        with lock:
            created.append({"spec": spec, "models_dir": models_dir, "download": download})
        return CountingPipe()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(extract_mod, "default_models_dir", fake_models_dir)
        mp.setattr(extract_mod, "create_engine", fake_create)
        cfg = FeedbackConfig()
        with pose_cache():
            res1 = analyze_practice("stance", "left", front, side, config=cfg, parallel=True)
            assert models_dir_threads == [main_tid]
            assert len(created) == 2
            assert all(c["download"] is False and c["models_dir"] == tmp_path for c in created)
            assert infer_calls[0] == 24

            res2 = analyze_practice("stance", "left", front, side, config=cfg, parallel=True)
            assert models_dir_threads == [main_tid]
            assert len(created) == 2
            assert infer_calls[0] == 24
            assert res2["capture"]["front"]["analysisSeconds"] == 0.0
            assert res2["capture"]["side"]["analysisSeconds"] == 0.0
            assert res2["checks"] == res1["checks"]
            assert res2["status"] == res1["status"]

            # 同作用域内单路 extract_series 默认路径亦命中缓存
            extract_series(front, cfg, lambda: False)
            assert infer_calls[0] == 24

    # 兼容外部 monkeypatch practice.create_practice_engine 与 practice.models_dir
    patched_factory_calls = [0]

    def patched_practice_engine(_cfg=None):
        patched_factory_calls[0] += 1
        return CountingPipe()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(practice_mod, "create_practice_engine", patched_practice_engine)
        analyze_practice("stance", "left", front, side, config=FeedbackConfig(), parallel=True)
        assert patched_factory_calls[0] == 2

    empty_models = tmp_path / "empty_models"
    empty_models.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(practice_mod, "models_dir", lambda: empty_models)
        with pytest.raises(FileNotFoundError, match="缺少 MediaPipe heavy"):
            extract_series(front, FeedbackConfig(pose_variant="heavy"), lambda: False)

    patched_create_calls = [0]

    def patched_create_engine(_spec, **_kw):
        patched_create_calls[0] += 1
        return CountingPipe()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(practice_mod, "create_engine", patched_create_engine)
        analyze_practice("stance", "left", front, side, config=FeedbackConfig(), parallel=True)
        assert patched_create_calls[0] == 2


def test_analyze_practice_stateful_pipeline_factory_order_under_parallel(tmp_path, monkeypatch):
    """并行模式下注入有状态迭代器工厂，即使正面首帧更慢，仍严格保证先正面后侧面。"""
    front = _write_test_avi(
        tmp_path / "front.avi", frames=12, size=(80, 64), fps=12.0, pixel_value=20
    )
    side = _write_test_avi(
        tmp_path / "side.avi", frames=12, size=(96, 64), fps=12.0, pixel_value=230
    )

    orig_open_video = extract_mod.open_video

    class DelayedFirstFrameSource:
        def __init__(self, inner, delay_s: float) -> None:
            self._inner = inner
            self._delay_s = delay_s

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def __iter__(self):
            first = True
            for frame in self._inner:
                if first and self._delay_s > 0:
                    first = False
                    time.sleep(self._delay_s)
                yield frame

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    def staggered_open_video(path, options=None, cancel=None):
        src = orig_open_video(path, options, cancel)
        delay = 0.03 if Path(path).name == "front.avi" else 0.0
        return DelayedFirstFrameSource(src, delay)

    monkeypatch.setattr(extract_mod, "open_video", staggered_open_video)

    class TaggedPipe:
        def __init__(self, tag: str, barrier: threading.Barrier) -> None:
            self.tag = tag
            self.barrier = barrier
            self.seen_widths: list[int] = []
            self.seen_luma: list[int] = []
            self.closed = False
            self._first = True

        def infer(self, frame, *, timestamp_ms=None):
            if self._first:
                self._first = False
                self.barrier.wait()
            self.seen_widths.append(int(frame.shape[1]))
            self.seen_luma.append(int(frame[0, 0, 0]))
            return _practice_landmarks(), []

        def close(self) -> None:
            self.closed = True

    for _ in range(3):
        barrier = threading.Barrier(2, timeout=3.0)
        front_pipe = TaggedPipe("front", barrier)
        side_pipe = TaggedPipe("side", barrier)
        pipes = iter([front_pipe, side_pipe])
        analyze_practice(
            "stance",
            "left",
            front,
            side,
            config=FeedbackConfig(),
            pipeline_factory=pipes.__next__,
            parallel=True,
        )
        assert len(front_pipe.seen_widths) == 12 and set(front_pipe.seen_widths) == {80}
        assert all(v < 64 for v in front_pipe.seen_luma)
        assert len(side_pipe.seen_widths) == 12 and set(side_pipe.seen_widths) == {96}
        assert all(v > 192 for v in side_pipe.seen_luma)
        assert front_pipe.closed and side_pipe.closed


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("front_delay,side_delay", [(0.0, 0.15), (0.15, 0.0)])
def test_analyze_practice_per_view_analysis_seconds_isolated(
    tmp_path, parallel: bool, front_delay: float, side_delay: float
):
    """正侧两路 analysisSeconds 严格独立记录，较快一路绝不被较慢一路耗时污染。"""
    front = _write_test_avi(tmp_path / "front.avi", frames=8, size=(64, 64), fps=10.0)
    side = _write_test_avi(tmp_path / "side.avi", frames=8, size=(64, 64), fps=10.0)

    class DelayedPipe:
        def __init__(self, delay: float) -> None:
            self._delay = delay
            self._first = True

        def infer(self, _frame, *, timestamp_ms=None):
            if self._first:
                self._first = False
                if self._delay > 0:
                    time.sleep(self._delay)
            return _practice_landmarks(), []

        def close(self) -> None:
            pass

    pipes = iter([DelayedPipe(front_delay), DelayedPipe(side_delay)])
    res = analyze_practice(
        "stance",
        "left",
        front,
        side,
        config=FeedbackConfig(),
        pipeline_factory=pipes.__next__,
        parallel=parallel,
    )
    front_s = res["capture"]["front"]["analysisSeconds"]
    side_s = res["capture"]["side"]["analysisSeconds"]
    fast_s, slow_s = (front_s, side_s) if front_delay == 0.0 else (side_s, front_s)
    assert fast_s < 0.08, f"较快一路耗时被污染: front={front_s}, side={side_s}"
    assert slow_s >= 0.12, f"较慢一路耗时未如实记录: front={front_s}, side={side_s}"
    assert slow_s - fast_s >= 0.08


def test_extract_pose_pair_engine_cancelled_cascades_and_practice_raises_interrupted(tmp_path):
    """子线程引擎直抛 Cancelled 时级联终止另一路并向上传播 Cancelled，练习入口转为 InterruptedError。"""
    v1 = _write_test_avi(tmp_path / "v1.avi", frames=5, size=(32, 32), fps=10.0)
    v2 = _write_test_avi(tmp_path / "v2.avi", frames=60, size=(32, 32), fps=10.0)

    class CancelEngine:
        def __init__(self) -> None:
            self.closed = False

        def infer(self, _frame, *, timestamp_ms=None):
            raise Cancelled("engine_abort")

        def close(self) -> None:
            self.closed = True

    class SlowSiblingEngine:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def infer(self, _frame, *, timestamp_ms=None):
            self.calls += 1
            time.sleep(0.01)
            return None, []

        def close(self) -> None:
            self.closed = True

    cancel_pipe = CancelEngine()
    slow_pipe = SlowSiblingEngine()
    with pytest.raises(Cancelled) as exc_info:
        extract_pose_pair(
            v1,
            v2,
            formal_engine("full"),
            FORMAL,
            parallel=True,
            engine_factory=iter([cancel_pipe, slow_pipe]).__next__,
        )
    assert exc_info.value.reason == "engine_abort"
    assert cancel_pipe.closed and slow_pipe.closed
    assert slow_pipe.calls < 60

    for parallel in (False, True):
        with pytest.raises(InterruptedError, match="分析已取消"):
            analyze_practice(
                "stance",
                "left",
                v1,
                v2,
                config=FeedbackConfig(),
                pipeline_factory=CancelEngine,
                parallel=parallel,
            )


def test_extract_pose_pair_second_spawn_failure_cancels_and_joins_first_worker(
    tmp_path, monkeypatch
):
    """第二路 TaskGroup.spawn 抛 RuntimeError 时，已启动的第一路 worker 被取消并 join 干净无残留。"""
    v1 = _write_test_avi(tmp_path / "v1.avi", frames=80, size=(32, 32), fps=10.0)
    v2 = _write_test_avi(tmp_path / "v2.avi", frames=80, size=(32, 32), fps=10.0)
    first_started = threading.Event()
    first_closed = threading.Event()
    first_calls = [0]
    spawned_handles = []

    class SlowFirstPipe:
        def infer(self, _frame, *, timestamp_ms=None):
            first_started.set()
            first_calls[0] += 1
            time.sleep(0.01)
            return None, []

        def close(self) -> None:
            first_closed.set()

    real_group_cls = extract_mod.TaskGroup
    spawn_count = [0]

    class FailingSecondSpawnGroup(real_group_cls):
        def spawn(self, name, fn, *, owner="", close_budget_s=2.0, daemon=True):
            spawn_count[0] += 1
            if spawn_count[0] == 2:
                assert first_started.wait(2.0), "第一路 worker 未启动"
                raise RuntimeError("spawn-second-boom")
            handle = super().spawn(
                name, fn, owner=owner, close_budget_s=close_budget_s, daemon=daemon
            )
            spawned_handles.append(handle)
            return handle

    monkeypatch.setattr(extract_mod, "TaskGroup", FailingSecondSpawnGroup)
    with pytest.raises(RuntimeError, match="spawn-second-boom"):
        extract_pose_pair(
            v1,
            v2,
            formal_engine("full"),
            FORMAL,
            parallel=True,
            engine_factory=SlowFirstPipe,
        )
    assert len(spawned_handles) == 1 and spawned_handles[0].done is True
    assert first_closed.is_set() is True
    calls_after_exit = first_calls[0]
    assert calls_after_exit < 80
    time.sleep(0.05)
    assert first_calls[0] == calls_after_exit


def test_extract_pose_pair_progress_monotonic_and_bounded(tmp_path, monkeypatch):
    """双路 progress 在长短悬殊（5 帧 vs 50 帧）与未知总帧数（total==0）下绝不先 100% 再回退或越界。"""
    short_vid = _write_test_avi(tmp_path / "short.avi", frames=5, size=(16, 16), fps=10.0)
    long_vid = _write_test_avi(tmp_path / "long.avi", frames=50, size=(16, 16), fps=10.0)

    class PacedPipe:
        def __init__(self, delay_s: float) -> None:
            self._delay_s = delay_s

        def infer(self, _frame, *, timestamp_ms=None):
            if self._delay_s > 0:
                time.sleep(self._delay_s)
            return None, []

        def close(self) -> None:
            pass

    spec = formal_engine("full")
    for parallel in (True, False):
        for pair, delays in (
            ((short_vid, long_vid), (0.0, 0.002)),
            ((long_vid, short_vid), (0.002, 0.0)),
        ):
            events: list[tuple[int, int]] = []
            pipes = iter([PacedPipe(delays[0]), PacedPipe(delays[1])])
            extract_pose_pair(
                pair[0],
                pair[1],
                spec,
                FORMAL,
                parallel=parallel,
                progress=lambda d, t: events.append((d, t)),
                engine_factory=pipes.__next__,
            )
            assert events, f"未收到进度回调: parallel={parallel}"
            known_ratios: list[float] = []
            for done, total in events:
                assert done >= 0 and total >= 0
                if total > 0:
                    assert done <= total, f"进度越界 done>total: {events}"
                    known_ratios.append(done / total)
            assert known_ratios == sorted(known_ratios), f"进度出现倒退: {events}"
            assert not any(t > 0 and d == t and t < 55 for d, t in events), (
                f"提前报满 100%: {events}"
            )
            assert known_ratios and known_ratios[-1] == 1.0

    # 未知总帧数（total == 0）场景：绝不出现 done > total > 0 或提前满 100%
    class FakeCap:
        def __init__(self, path: str) -> None:
            self._name = Path(path).name
            self._i = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0 if "unknown" in self._name else 20.0
            return 0.0

        def read(self):
            if self._i >= 20:
                return False, None
            self._i += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_source_mod.cv2, "VideoCapture", FakeCap)
    for parallel in (True, False):
        for pair in (("unknown_a.mp4", "known_b.mp4"), ("unknown_a.mp4", "unknown_b.mp4")):
            events = []
            extract_pose_pair(
                pair[0],
                pair[1],
                spec,
                FORMAL,
                parallel=parallel,
                progress=lambda d, t: events.append((d, t)),
                engine_factory=lambda: PacedPipe(0.0),
            )
            assert events
            for done, total in events:
                assert done >= 0 and total >= 0
                assert total == 0, f"含未知帧数流时不应虚报已知总量: {events}"
