# -*- coding: utf-8 -*-
"""学生练习的后台分析与个人历史：在后台任务里分析并保存记录，按代次丢弃作废的结果；历史的读取、导出与教师复核。

分析任务的代次由本控制器的 ``Epoch`` 持有：改学号、点取消都推进代次，之前发起、带工作流 token 的分析结果
回来时只恢复控件，不展示也不回调。结果经 dispatcher 回到 Tk 线程。
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.action_feedback import analyze_feedback, format_report
from core.feedback_geometry import load_feedback_config
from core.foundation import Epoch, TaskGroup
from core.reporting import practice_report_renderers
from core.services import HistoryStore

Post = Callable[[Callable[[], Any]], Any]

MODEL_CHOICES = {"Full（精细分析）": "full", "Lite（快速对比）": "lite", "Heavy（较慢）": "heavy"}
DEFAULT_MODEL = "Full（精细分析）"
ANALYZING_TEXT = "正在用MediaPipe CPU提取关键点并检查规则，完成后保存到个人历史…"


def analysis_error_text(exc: Exception) -> str:
    """后台分析失败时给用户看的文字。"""
    if isinstance(exc, InterruptedError):
        return "分析已取消"
    if isinstance(exc, (ValueError, FileNotFoundError, PermissionError)):
        return str(exc)
    return "分析或保存失败，请检查MediaPipe模型、视频和保存目录后重试。"


def saved_status(record: dict) -> str:
    return "问题说明已保存" + ("；部分证据不足，请重新录制" if record["result"]["needsRerecord"] else "")


class PracticeAnalysis:
    """同一时刻只跑一个分析任务；``closed`` 后一切结果都被丢弃。

    回调都在 Tk 线程调用：``on_busy(bool)``、``on_result(text)``、``on_status(text)``、``on_finished(applied)``
    （任务结束、控件需要按新状态刷新时；``applied`` 表示结果已展示，历史窗口据此刷新）。
    """

    def __init__(
        self,
        post: Post,
        *,
        on_busy: Callable[[bool], None],
        on_result: Callable[[str], None],
        on_status: Callable[[str], None],
        on_finished: Callable[[bool], None],
    ) -> None:
        self._post = post
        self._on_busy = on_busy
        self._on_result = on_result
        self._on_status = on_status
        self._on_finished = on_finished
        self.store: HistoryStore | None = None
        self.current: dict | None = None
        self.busy = False
        self.closed = False
        self.epoch = Epoch()
        self.stop_event = threading.Event()
        self._tasks = TaskGroup("practice-analysis")

    def history(self) -> HistoryStore:
        if self.store is None:
            self.store = HistoryStore(renderers=practice_report_renderers())
        return self.store

    def analyzer(self, model_label: str) -> Callable[..., Any]:
        """在 Tk 线程按所选模型读一次配置，后台任务与本次选择绑定；配置损坏时抛 ``ValueError``/``OSError``。"""
        config = replace(load_feedback_config(), pose_variant=MODEL_CHOICES[model_label])
        return lambda *args, **kwargs: analyze_feedback(*args, config=config, **kwargs)

    def start(self, work: Callable[[], dict], *, on_saved: Callable[[dict], None] | None = None,
              token: Any = None) -> bool:
        """启动后台分析；已有任务时返回 False。"""
        if self.busy:
            return False
        stamp = self.epoch.advance()
        self.busy = True
        self.stop_event.clear()
        self._on_busy(True)
        self._on_status(ANALYZING_TEXT)

        def run(_token) -> None:
            try:
                outcome = (work(), None)
            except Exception as exc:
                outcome = (None, analysis_error_text(exc))
            self._post(lambda: self._finish(*outcome, stale=token is not None and not self.epoch.is_current(stamp),
                                            on_saved=on_saved))

        self._tasks.spawn("student-action-feedback", run, owner="practice")
        return True

    def analyze(self, identity: dict, front: Path | None, side: Path | None, *, analyzer: Callable[..., Any],
                record_id: str | None = None, on_saved: Callable[[dict], None] | None = None,
                token: Any = None) -> bool:
        """分析新录像并保存，或对已有记录补充分析（``record_id``）。"""
        stopped = self.stop_event.is_set
        if record_id:
            work = lambda: self.history().reanalyze(record_id, stopped=stopped, analyzer=analyzer)  # noqa: E731
        else:
            work = lambda: self.history().add(identity, front, side, stopped=stopped, analyzer=analyzer)  # noqa: E731
        return self.start(work, on_saved=on_saved, token=token)

    def invalidate(self) -> None:
        """学号变了：作废进行中的分析代次与当前记录。"""
        self.epoch.advance()
        self.current = None

    def cancel(self) -> None:
        self.epoch.advance()
        self.stop_event.set()
        self._on_status("正在取消；当前帧处理结束后释放任务…")

    def close(self) -> None:
        self.closed = True
        self.epoch.advance()
        self.stop_event.set()

    def export_text(self, record_id: str, destination: Path) -> None:
        self.history().export(record_id, destination)

    def _finish(self, record: dict | None, error: str | None, *, stale: bool,
                on_saved: Callable[[dict], None] | None) -> None:
        """Tk 线程：应用分析结果；面板已关、代次过期或已取消时只恢复控件。"""
        if self.closed:
            return
        self.busy = False
        self._on_busy(False)
        if stale or self.stop_event.is_set():
            self._on_finished(False)
            return
        if error:
            self._on_status(error)
            self._on_result(error)
        else:
            self.current = record
            self._on_result(format_report(record))
            self._on_status(saved_status(record))
            if on_saved:
                on_saved(record)
        self._on_finished(True)
