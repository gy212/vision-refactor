# -*- coding: utf-8 -*-
"""录制性能测试"按当前设置录像"一段的处理循环负载：用独立的 ``LiveSessionManager`` 在测试已接管的摄像头上运行正式会话的同一个循环。

会话配置取主界面此刻的设置（与点「开始」时相同：``session_config(modes.session_settings(collect_state()))``），
只改三处：摄像头换成测试接管的那几路（``ClaimedCameraInput``，测试负责接管与关闭）、没有录制端口（写盘由测试自己的
写盘线程按生产录像线程的写法做）、没有加载期裸帧预览。双摄骨架关时站位检测一直开着（考试录像期间它一直在跑，
是骨架关时最重的情形）；双摄的在线识别按 D-7 关闭，与"实时动作识别"勾不勾无关。
预览由 Tk 主线程定时调用 ``render`` 画到主窗口的预览面（正式会话也这样画）。
测试不下载模型：缺 Lite 模型时不开站位检测（结果记 ``presence_skipped``），缺骨架识别模型而设置要识别骨架时不运行循环、
写明原因（缺模型的提示由测试的总结论统一给出）。
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from core.foundation import CancelToken
from core.live import ClaimedCameraInput, LiveControls, LiveSessionManager, SessionConfig, SessionEvent
from core.media import SECONDARY, CameraStream
from core.workflows import hip_midpoint_in_roi

STOP_WAIT_S = 10.0
OCCUPANCY_OWNER = "benchmark"


class BenchmarkSessionLoad:
    """一次测试一个实例；``__call__`` 在测试的任务线程里运行（``BenchmarkSessionLoad`` 契约），``render`` 在 Tk 主线程。

    ``preview`` 是主窗口的 ``PreviewSurface``（``size_of``、``show``、``dual_visible``、``set_dual_visible``）；
    ``missing`` 是本机缺的模型（键 ``lite``、``skeleton``，与 ``BenchmarkSetup.missing_models`` 相同）。"""

    def __init__(self, config: SessionConfig, *, rotation_of: Callable[[str], int], preview: Any,
                 missing: Mapping[str, str] | None = None) -> None:
        self._config = config
        self._missing = dict(missing or {})
        self._rotation_of = rotation_of
        self._preview = preview
        self._lock = threading.Lock()
        self._manager: LiveSessionManager | None = None
        self._store: Any = None
        self._shown = 0
        self._rendered = 0
        self._dual_before: bool | None = None

    # ------------------------------------------------------------------ 测试线程
    def __call__(self, stream: CameraStream, token: CancelToken, on_running: Callable[[], None]) -> dict[str, Any]:
        config = replace(self._config, cameras=dict(stream.specs), loading_raw_preview=False)
        state: dict[str, Any] = {"presence": 0}
        if config.inference and "skeleton" in self._missing:
            state["refused"] = "缺少骨架识别模型，没有运行处理循环"
            return self._result(config, state)
        controls = LiveControls()
        for role in stream.specs:
            controls.rotation[role].set(int(self._rotation_of(role)))
        if config.occupancy and "lite" not in self._missing:
            controls.arm_occupancy(OCCUPANCY_OWNER, True)
        changed = threading.Condition()

        def on_event(event: SessionEvent) -> None:
            with changed:
                if event.kind in ("running", "ended"):
                    state[event.kind] = (time.monotonic(), event.payload)
                elif event.kind == "presence":
                    state["presence"] += 1
                changed.notify_all()

        manager = LiveSessionManager(make_input=lambda _config: ClaimedCameraInput(stream), on_event=on_event,
                                     controls=controls, probe=hip_midpoint_in_roi,
                                     preview_target=self._preview.size_of)
        with self._lock:
            self._manager = manager
        try:
            manager.start(config)
            with changed:
                while not ({"running", "ended"} & state.keys()) and not token.cancelled:
                    changed.wait(0.1)
            if "running" in state and "ended" not in state:
                on_running()
                token.wait()
        finally:
            manager.stop(OCCUPANCY_OWNER)
            manager.wait_idle(STOP_WAIT_S)
            with self._lock:
                self._manager = None
        return self._result(config, state)

    def _result(self, config: SessionConfig, state: dict[str, Any]) -> dict[str, Any]:
        settings = {"dual": config.dual, "skeleton": config.inference, "pose_variant": config.pose_variant,
                    "enable_hands": config.enable_hands, "parallel": config.parallel, "workers": config.workers,
                    "online_match": config.online_match, "presence": config.occupancy}
        ended = state.get("ended")
        running = state.get("running")
        outcome = None if ended is None else ended[1]
        error = state.get("refused")
        if error is None and running is None:
            error = "处理循环没有开始" + (f"：{outcome.status}" if outcome is not None and outcome.status else "")
        elif error is None and outcome is not None and outcome.failed:
            error = f"处理循环失败：{outcome.status}"
        seconds = (ended[0] - running[0]) if running is not None and ended is not None else 0.0
        frames = 0 if outcome is None else int(outcome.frames)
        return {
            "settings": settings,
            "frames": frames,
            "fps": round(frames / seconds, 2) if seconds > 0 else None,
            "rendered": self._rendered,
            "presence_samples": state["presence"],
            "presence_skipped": bool(config.occupancy and "lite" in self._missing),
            "error": error,
        }

    # ------------------------------------------------------------------ Tk 主线程
    def render(self) -> None:
        """把测试会话比已显示更新的一帧画到主窗口预览；测试会话结束后恢复主窗口原来的单/双路布局。"""
        with self._lock:
            manager = self._manager
        store = None if manager is None else manager.preview
        if store is None:
            if self._dual_before is not None:
                self._preview.set_dual_visible(self._dual_before)
                self._dual_before = None
            return
        if store is not self._store:
            self._store, self._shown = store, 0
        frame = store.take_newer(self._shown)
        if frame is None:
            return
        self._shown = frame.seq
        dual = SECONDARY in frame.images
        if self._dual_before is None:
            self._dual_before = bool(self._preview.dual_visible)
        if dual != self._preview.dual_visible:
            self._preview.set_dual_visible(dual)
        self._preview.show(frame.images, frame.sizes)
        store.mark_rendered(frame)
        self._rendered += 1


__all__ = ["BenchmarkSessionLoad"]
