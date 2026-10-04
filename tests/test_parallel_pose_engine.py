# -*- coding: utf-8 -*-
"""ParallelPoseEngine 单元/属性测试（不依赖 MediaPipe 模型）。"""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.parallel_pose_engine import ParallelPoseEngine, InferResult  # noqa: E402


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
