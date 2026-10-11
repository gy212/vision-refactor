# -*- coding: utf-8 -*-
"""issue #10 回归：慢考试裁剪不得在关窗预算后拖住进程退出，单摄转码仍在关窗后做完，取消终态唯一，裁剪可取消。"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.foundation.tasks import Cancelled, CancelToken
from core.recording import ProcessingPlan, ResultSpec, SegmentFinalizer, SegmentUpdate
from core.workflows import ExamClipStep, prepare_exam_pair
from tests.segment_doubles import dual_segment

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXIT_LIMIT_S = 20.0  # 子进程退出上限：远小于阻塞替身的等待，只要没被线程拖住就能在这之内退出


def _run_child(code: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    started = time.monotonic()
    out = subprocess.run(
        [sys.executable, "-B", str(script), str(_REPO_ROOT), str(tmp_path)],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=_EXIT_LIMIT_S + 30.0,
    )
    return out, time.monotonic() - started


_CHILD_PRELUDE = """
import sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
tmp = Path(sys.argv[2])
from core.recording import ProcessingPlan, ResultSpec, SegmentFinalizer
from core.workflows import ExamClipStep
from tests.segment_doubles import dual_segment, single_segment
"""


def test_slow_exam_clip_does_not_keep_process_alive_after_close_budget(tmp_path):
    """只有一个慢考试裁剪（完全不响应取消的最坏情况）：close 超时后主线程返回，进程必须退出。"""
    out, elapsed = _run_child(_CHILD_PRELUDE + """
entered = threading.Event()

def stuck_clip(*_args, **_kwargs):
    entered.set()
    threading.Event().wait(600.0)  # 不响应取消

finalizer = SegmentFinalizer()
segment_dir = tmp / "record_exam"
segment_dir.mkdir()
segment = dual_segment("record_exam", segment_dir, segment_dir / "front.mp4", segment_dir / "side.mp4")
plan = ProcessingPlan((ExamClipStep(clip=stuck_clip),), result=ResultSpec(), close_budget_s=0.05)
assert finalizer.submit(segment, plan)
assert entered.wait(5.0)
print("closed", finalizer.close(0.05))
io_threads = [t for t in threading.enumerate() if t.name.endswith("/io")]
print("io_daemon", bool(io_threads) and all(t.daemon for t in io_threads))
""", tmp_path)

    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["closed", "False", "io_daemon", "True"]
    assert elapsed < _EXIT_LIMIT_S


def test_single_camera_transcode_still_finishes_after_main_thread_returns(tmp_path):
    """V-4：单摄转码在关窗后做完，进程等它结束才退出。"""
    marker = tmp_path / "transcoded.txt"
    out, _elapsed = _run_child(_CHILD_PRELUDE + """
class SlowTranscode:
    lane = "io"

    def run(self, ctx, cancel):
        time.sleep(0.5)
        (tmp / "transcoded.txt").write_text(threading.current_thread().name, encoding="utf-8")

finalizer = SegmentFinalizer()
segment = single_segment("record_single", tmp / "record_single.mp4")
assert finalizer.submit(segment, ProcessingPlan((SlowTranscode(),), finish_on_close=True))
print("closed", finalizer.close(0.0))
""", tmp_path)

    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["closed", "False"]
    assert marker.read_text(encoding="utf-8") == "finalizer/segment-finalizer/io-finish"


class _Compare:
    """裁剪之后的步骤：若执行到这里会发布 completed。"""

    lane = "analysis"

    def __init__(self) -> None:
        self.ran = threading.Event()

    def run(self, ctx, cancel):
        from core.recording.steps import Terminal

        self.ran.set()
        return Terminal("completed", "done")


def test_cancellable_exam_clip_publishes_single_cancelled_terminal_on_close(tmp_path):
    updates: list[SegmentUpdate] = []
    finalizer = SegmentFinalizer(on_update=updates.append)
    entered = threading.Event()
    threads: list[tuple[str, bool]] = []

    def cancellable_clip(*_args, cancel: CancelToken, **_kwargs):
        current = threading.current_thread()
        threads.append((current.name, current.daemon))
        entered.set()
        assert cancel.wait(5.0)
        cancel.raise_if_cancelled()

    segment_dir = tmp_path / "record_exam"
    segment_dir.mkdir()
    segment = dual_segment("record_exam", segment_dir, segment_dir / "front.mp4", segment_dir / "side.mp4")
    compare = _Compare()
    plan = ProcessingPlan((ExamClipStep(clip=cancellable_clip), compare), result=ResultSpec(), close_budget_s=30.0)

    assert finalizer.submit(segment, plan)
    assert entered.wait(1.0)
    assert finalizer.close(2.0)
    finalizer.cancel_all()

    assert threads == [("finalizer/segment-finalizer/io", True)]
    assert not compare.ran.is_set()
    assert [(u.status, u.error_code) for u in updates] == [("cancelled", "app_closing")]
    payload = json.loads((segment_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["error"]["code"] == "app_closing"
    assert not any("exam_clip_failed" in w for w in payload["warnings"])


def _write_video(path: Path, n: int, color: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(n):
        frame = np.full((48, 64, 3), color + (i % 7) * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _CancelAfter(CancelToken):
    """第 ``n`` 次检查时自行取消，模拟裁剪写到一半时关窗。"""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.remaining = n

    def raise_if_cancelled(self) -> None:
        self.remaining -= 1
        if self.remaining <= 0:
            self.cancel("app_close")
        super().raise_if_cancelled()


@pytest.mark.parametrize("checks", [1, 40, 140])
def test_prepare_exam_pair_stops_on_cancel_and_removes_partial_output(tmp_path, checks):
    # 两路各 60 帧：能量扫描 60 次 + 侧路计数 60 次检查后进入写出阶段；140 落在正面写出途中
    front, side = tmp_path / "front.mp4", tmp_path / "side.mp4"
    _write_video(front, 60, 10)
    _write_video(side, 60, 20)
    out = tmp_path / "out"

    with pytest.raises(Cancelled):
        prepare_exam_pair(front, side, front_frames=60, side_frames=60, out_dir=out, cancel=_CancelAfter(checks))

    assert not list(out.glob("*_exam.*"))
