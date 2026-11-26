# -*- coding: utf-8 -*-
"""动作分析窗口的控制器：在后台任务里对模板池逐个比对，结果经 dispatcher 交回视图。

比对是后端黑盒链路，统一 heavy，与主界面预览档位无关。视图只需实现 ``CompareView``。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from core.foundation import TaskGroup, to_jsonable
from core.recording import template_action_name, try_load_heavy_template
from core.services import compare_templates

POSE_VARIANT = "heavy"

Post = Callable[[Callable[[], Any]], Any]


class CompareView(Protocol):
    def show_progress(self, stage: str, done: int, total: int) -> None: ...
    def set_status(self, text: str) -> None: ...
    def set_result(self, text: str) -> None: ...
    def set_raw(self, text: str) -> None: ...
    def set_running(self, running: bool) -> None: ...
    def reset_scores(self) -> None: ...
    def show_scores(
        self, actions: list[tuple[str, float]], match_info: str
    ) -> None: ...


def usable_template(path: str, existing: Sequence[str]) -> tuple[str | None, bool]:
    """校验能否加入模板池：返回 (可用路径或 None, 是否无效)。已在池内时路径为 None 且不算无效。"""
    usable = try_load_heavy_template(path)
    if usable is None:
        return None, True
    if str(usable) in {str(p) for p in existing}:
        return None, False
    return str(usable), False


def compare_report(
    results: Sequence[Any], video: str, workers: int
) -> tuple[list[tuple[str, float]], str, dict]:
    """比对结果 → (逐动作分, 说明文字, 详细数据)。动作名 = 模板文件 stem，综合分为各动作平均。"""
    actions = [
        (template_action_name(res.template_path), float(res.score)) for res in results
    ]
    payload: dict = {"video_path": video}
    payload["actions"] = [
        {
            "name": template_action_name(res.template_path),
            "score": res.score,
            "avg_cost": res.avg_cost,
            "cost": res.cost,
            "start_frame": res.start_frame,
            "end_frame": res.end_frame,
            "fps": res.fps,
            "pose_variant": res.pose_variant,
            "workers_used": res.workers_used,
            "template_path": str(res.template_path),
        }
        for res in results
    ]
    if actions:
        combined = sum(s for _, s in actions) / len(actions)
        payload["combined_score"] = combined
        payload["combined_percent"] = int(round(combined * 100))
    match_info = f"共 {len(actions)} 个动作已评分"
    if any(r.workers_used and r.workers_used != workers for r in results):
        match_info += "\n说明：模板为 VIDEO 模式，已自动使用单线程以保证准确。"
    return actions, match_info, payload


class CompareController:
    """同一时刻只跑一个后台任务；``stop``/``close`` 置停止标志，任务在下一帧检查时退出。"""

    def __init__(
        self,
        view: CompareView,
        post: Post,
        *,
        compare: Callable[..., Any] = compare_templates,
    ) -> None:
        self._view = view
        self._post = post
        self._compare = compare
        self._tasks = TaskGroup("compare-dialog")
        self._stop_evt = threading.Event()

    @property
    def busy(self) -> bool:
        return bool(self._tasks.pending())

    def compare(self, templates: Sequence[str], video: str, workers: int) -> bool:
        """目标视频对模板池逐个比对。已有任务在跑时返回 False。"""
        if self.busy:
            return False
        self._begin("正在分析…")
        templates = list(templates)

        def run(_token) -> None:
            try:
                results = self._compare(
                    templates,
                    video,
                    pose_variant=POSE_VARIANT,
                    workers=workers,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                actions, match_info, payload = compare_report(results, video, workers)
                self._post(lambda: self._view.show_scores(actions, match_info))
                raw = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2)
                self._post(
                    lambda: (self._view.set_raw(raw), self._view.set_status("分析完成"))
                )
            except InterruptedError:
                self._finish("已停止", "动作比对已停止", "")
            except Exception as exc:
                self._finish("分析失败", str(exc), "")
            finally:
                self._post(lambda: self._view.set_running(False))

        self._tasks.spawn("compare", run, owner="compare")
        return True

    def stop(self) -> None:
        self._stop_evt.set()
        self._view.set_status("正在停止…")

    def close(self) -> None:
        self._stop_evt.set()

    def _begin(self, status: str) -> None:
        self._stop_evt.clear()
        self._view.set_running(True)
        self._view.set_status(status)
        self._view.reset_scores()
        self._view.set_raw("")
        self._view.show_progress("准备中", 0, 0)

    def _progress(self, stage: str, done: int, total: int) -> None:
        self._post(lambda: self._view.show_progress(stage, done, total))

    def _finish(self, status: str, result: str, raw: str) -> None:
        def apply() -> None:
            self._view.set_status(status)
            self._view.set_result(result)
            self._view.set_raw(raw)

        self._post(apply)
