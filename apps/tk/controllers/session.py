# -*- coding: utf-8 -*-
"""实时会话的界面编排：开始时冻结界面设置为会话配置、开始/停止、会话事件回到 Tk 线程后的处理、预览渲染与运行态控件联动。

唯一逐帧循环在 ``core.live`` 的会话线程里；会话事件经 dispatcher 回到 Tk 线程，过期会话的事件丢弃。
每路旋转只存在 ``LiveControls.rotation``：点击预览顺时针 +90°，会话线程每帧读取。
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import messagebox
from typing import Any

from core import foundation
from core.live import (
    RUNNING,
    STARTING,
    LiveControls,
    LiveSessionManager,
    PresenceSample,
    RunningInfo,
    SessionConfig,
    SessionEvent,
    SessionOutcome,
    camera_session,
    video_session,
)
from core.media import PRIMARY, SECONDARY, CaptureSpec
from core.workflows import NORMAL, ModeManager, WorkflowHost, hip_midpoint_in_roi

from apps.tk.controllers.flags import LiveFlags

ROTATIONS = (0, 90, 180, 270)


def clamp_workers(n: int) -> int:
    """将离线线程数钳制到闭区间 [1, os.cpu_count()]；无法确定核心数时上界回退为 1。"""
    cpu = os.cpu_count()
    upper = cpu if cpu and cpu >= 1 else 1
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = 1
    return min(max(value, 1), upper)


def default_workers() -> int:
    """主预览默认并行 worker 数：留 2 个核给 UI/采集/系统，上限 8（6 worker 已达 ~57fps，再多只换来内存占用）。"""
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu - 2))


def next_rotation_cw(degrees: int) -> int:
    """顺时针旋转 90°（0→90→180→270→0）；非法值按 0 处理。"""
    return ((degrees if degrees in ROTATIONS else 0) + 90) % 360


@dataclass(frozen=True)
class UiState:
    """开始时冻结的界面设置；模式要求经 ``ModeManager.session_settings`` 叠加在它上面。"""

    source: str
    pose_variant: str
    workers: int
    enable_hands: bool
    source2: str | None = None
    online_match_enabled: bool = True
    rotate: int = 0
    rotate2: int = 0
    record_skeleton: bool = False
    auto_compare: bool = True                # 双摄录制结束后是否自动比对；False 为仅录制
    capture_size: tuple[int, int] = NORMAL.requirements.fmt.size
    capture_policy: str = NORMAL.requirements.fmt.policy   # 驱动给不出请求尺寸时的处理（D-6、D-17）
    camera_names: tuple[tuple[int, str], ...] = ()   # 按编号记下的设备名（V-3）


def session_spec(state: UiState, source: str) -> CaptureSpec:
    width, height = state.capture_size
    return CaptureSpec(int(source), int(width), int(height), expected_name=dict(state.camera_names).get(int(source)),
                       policy=state.capture_policy)


def session_config(state: UiState) -> SessionConfig:
    """把冻结的界面状态换成会话配置；四种循环的差异由设计文档 §8.2 的表决定。"""
    if not state.source.isdigit():
        return video_session(state.source, pose_variant=state.pose_variant, enable_hands=state.enable_hands,
                             workers=state.workers)
    cameras = {PRIMARY: session_spec(state, state.source)}
    if state.source2:
        cameras[SECONDARY] = session_spec(state, state.source2)
    return camera_session(cameras, pose_variant=state.pose_variant, enable_hands=state.enable_hands,
                          workers=state.workers, skeleton=bool(state.record_skeleton),
                          online_match=bool(state.online_match_enabled))


class OccupancyPort:
    """工作流的站位检测端口（``OccupancyControl``）：按所有者武装会话内的站位检测。"""

    def __init__(self, controls: LiveControls, owner: str) -> None:
        self._controls = controls
        self._owner = owner

    def arm(self, armed: bool) -> None:
        self._controls.arm_occupancy(self._owner, bool(armed))

    def set_roi(self, roi) -> None:
        self._controls.set_roi(roi)


class SessionController:
    """``view`` 提供 ``capture``、``status``、``preview`` 三个面板与 ``compare_btn``；``live`` 由组合根接上。"""

    def __init__(self, view: Any, *, controls: LiveControls, cameras: Any, recording: Any, modes: ModeManager,
                 workflows: WorkflowHost, flags: LiveFlags, post: Callable[..., Any],
                 set_exam_mode: Callable[[bool], None]) -> None:
        self.view = view
        self.controls = controls
        self.cameras = cameras
        self.recording = recording
        self.modes = modes
        self.workflows = workflows
        self.flags = flags
        self._post = post
        self._set_exam_mode = set_exam_mode
        self.live: LiveSessionManager | None = None
        self._shown_seq = 0
        self._pending = 0
        self._pending_lock = threading.Lock()

    def build_live(self, *, sink: Any) -> LiveSessionManager:
        """组装会话管理器：摄像头经 ``CameraService``，录制经 ``sink``，站位判定用髋中点 ROI。"""
        self.live = LiveSessionManager(
            make_input=self.cameras.make_input, on_event=self.on_event, controls=self.controls, sink=sink,
            probe=hip_midpoint_in_roi, preview_target=self.view.preview.size_of,
        )
        return self.live

    # ------------------------------------------------------------------ 状态
    def active(self) -> bool:
        """有会话（含启动中与收尾中）。"""
        return self.live.active()

    def live_now(self) -> bool:
        """会话已开始且没有在停止：录制、站位检测只在这时接受操作。"""
        return self.live.state in (STARTING, RUNNING)

    def finished(self) -> bool:
        """会话已结束且投递的事件都已在 Tk 线程处理完。"""
        with self._pending_lock:
            return not self.live.active() and self._pending == 0

    def status(self, text: str) -> None:
        self.view.status.status_var.set(text)

    # ------------------------------------------------------------------ 旋转
    def rotation(self, role: str) -> int:
        return self.controls.rotation_of(role)

    def set_rotation(self, degrees: int, role: str = PRIMARY, *, announce: bool = True) -> int:
        value = int(degrees) if int(degrees) in ROTATIONS else 0
        self.controls.rotation[role].set(value)
        if announce:
            side = "侧摄" if role == SECONDARY else "主摄"
            message = f"{side}旋转 → {value}°（点击预览可继续顺时针 90°）"
            if role != SECONDARY and self.modes.is_active("exam"):
                message += "；考试 ROI 可能失效，请重新标定"
            self.status(message)
        return value

    def rotate_clockwise(self, role: str) -> int:
        return self.set_rotation(next_rotation_cw(self.rotation(role)), role)

    # ------------------------------------------------------------------ 开始与停止
    def collect_state(self) -> UiState:
        """读取界面设置；第二摄像头只在主输入源也是摄像头时生效（唯一收敛点）。"""
        source_state = self.cameras.source
        source = source_state.value.strip()
        if source_state.kind == "none" or not source:
            raise ValueError("请先选择摄像头或视频")
        capture, status = self.view.capture, self.view.status
        source2 = None
        label_2 = self.cameras.secondary_label()
        if source.isdigit() and label_2:
            index_2 = self.cameras.index_for(label_2)
            if index_2 is not None:
                if int(source) == index_2:
                    raise ValueError("两个摄像头不能选同一个")
                source2 = str(index_2)
        recording = self.view.recording
        return UiState(
            source=source,
            pose_variant=capture.pose_var.get().strip() or "full",
            workers=clamp_workers(capture.workers_var.get() or 1),
            enable_hands=bool(capture.enable_hands_var.get()),
            source2=source2,
            online_match_enabled=bool(status.online_match_var.get()),
            rotate=self.rotation(PRIMARY),
            rotate2=self.rotation(SECONDARY),
            record_skeleton=bool(recording.record_skeleton_var.get()),
            auto_compare=bool(recording.auto_compare_var.get()) if source2 is not None else True,
            capture_size=self.cameras.capture_size(),
            capture_policy=self.cameras.capture_policy(),
            camera_names=self.cameras.camera_names(),
        )

    def start(self) -> None:
        if self.active():
            self.status("上一会话仍在收尾，请稍候再点「开始」")
            return
        start_click = time.monotonic()
        try:
            # 会话配置 = 用户设置快照叠加当前模式的要求（§8.5），在开始时计算一次
            state = self.modes.session_settings(self.collect_state())
            config = session_config(state)
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        if config.is_file:
            self.cameras.release_warmups()   # 视频文件会话先放弃选择阶段持有的摄像头预热
        capture = self.view.capture
        capture.start_btn.configure(state="disabled")
        capture.stop_btn.configure(state="normal")
        self.status("启动中…（首次运行可能需要下载模型）")
        self.view.status.reset_for_start()
        self.view.preview.set_dual_visible(config.dual)
        self.set_rotation(state.rotate, PRIMARY, announce=False)
        self.set_rotation(state.rotate2, SECONDARY, announce=False)
        self.flags.auto_compare = self.recording.current_auto_compare()
        self.flags.recording_ready = not config.dual
        self.flags.record_skeleton = bool(state.record_skeleton) and not config.is_file
        self.flags.pose_variant, self.flags.enable_hands = config.pose_variant, config.enable_hands
        self._shown_seq = 0
        token = self.live.start(config, start_click=start_click)
        self.workflows.session_started(token)
        foundation.diag.event(
            "session.start", session=token, primary=state.source, secondary=state.source2,
            requested="x".join(str(v) for v in state.capture_size), student=self.modes.is_active("practice"),
        )
        self.set_running_controls(True)

    def stop(self) -> None:
        # 先同步工作流：否则站位检测退出后考试仍停在 recording，手动锁不解除；练习停止等待到位
        try:
            self.workflows.session_stopped(time.monotonic())
        except Exception:
            traceback.print_exc()
        self._set_exam_mode(False)
        foundation.diag.event("session.stop_requested", session=self.live.token, cancel="user_stop")
        self.live.stop("user_stop")
        self.flags.recording_ready = False
        self.view.recording.record_btn.configure(state="disabled")
        self.view.capture.stop_btn.configure(state="disabled")
        self.status("正在停止…")

    # ------------------------------------------------------------------ 会话事件
    def on_event(self, event: SessionEvent) -> None:
        """会话线程回调：只投递到 Tk 线程；会话终态同时写诊断行。"""
        if event.kind == "ended":
            outcome = event.payload
            foundation.diag.event("session.end", session=event.token, status=outcome.status,
                                  rewarm=not outcome.failed)
        with self._pending_lock:
            self._pending += 1
        if not self._post(lambda: self.handle(event)):
            self._done_one()

    def _done_one(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def handle(self, event: SessionEvent) -> None:
        """Tk 线程：关窗后或过期会话的事件丢弃。"""
        try:
            if self.flags.closing or not self.live.is_current(event.token):
                return
            handler = {
                "status": self.status,
                "progress": lambda progress: self.view.status.set_progress(*progress),
                "match": self._on_match,
                "presence": self._on_presence,
                "running": self._on_running,
                "ended": self._on_ended,
            }[event.kind]
            handler(event.payload)
        finally:
            self._done_one()

    def _on_match(self, text: str) -> None:
        if self.live_now():
            self.view.status.match_var.set(text)

    def _on_presence(self, sample: PresenceSample) -> None:
        """站位采样：会话未在停止时交给当前激活的工作流（考试或学生练习）。"""
        if self.live_now():
            self.workflows.presence(self.live.token, bool(sample.present), float(sample.sampled_at))

    def _on_running(self, _info: RunningInfo) -> None:
        """会话已接管设备并登记录制：双摄此时才允许录制，学生练习据此启用站位检测。"""
        if not self.live.config.dual or not self.live_now():
            return
        self.flags.dual_active = True
        self.flags.auto_compare = self.recording.current_auto_compare()
        self.flags.recording_ready = True
        if not self.modes.is_active("practice"):
            self.view.recording.record_btn.configure(state="normal")
            self.view.recording.set_record_text("idle")
        self.workflows.session_ready(self.live.token, time.monotonic())

    def _on_ended(self, outcome: SessionOutcome) -> None:
        """会话终态（设备已归还或已隔离）：复位界面；未失败时重新声明预热。"""
        config = self.live.config
        self.flags.recording_ready = False
        self.flags.dual_active = False
        # 开考后主会话结束：考试中止；学生练习回到可重新「开始」
        self.workflows.session_ended(self.live.token, time.monotonic(), status=outcome.status or "")
        self._set_exam_mode(False)
        self.set_running_controls(False)
        if outcome.status:
            self.status(outcome.status)
        # 新 reader 若遇到仍在退出的旧进程，会在后台等待真实释放后再打开，Tk 线程不等待驱动。
        if not outcome.failed or (config is not None and config.is_file):
            try:
                self.cameras.sync_warmup(allow_running=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ 逐帧
    def render_preview(self) -> None:
        """每 tick 显示比已显示更新的一帧预览；双摄首个裸帧与首个标注帧各提示一次启动耗时。"""
        store = self.live.preview
        if store is None or not self.live_now():
            return
        frame = store.take_newer(self._shown_seq)
        if frame is None:
            return
        self._shown_seq = frame.seq
        self.view.status.actions_var.set(frame.actions_text)
        self.view.preview.show(frame.images, frame.sizes)
        first = store.mark_rendered(frame)
        metrics = store.metrics
        if SECONDARY in frame.images and first and metrics is not None:
            elapsed_ms = metrics.elapsed_ms()
            if frame.stage == "raw":
                loading = "，模型加载中…" if metrics.inference else ""
                self.status(f"双摄画面已显示（{elapsed_ms:.0f} ms）{loading}")
            else:
                self.status(f"运行中…（双摄首个标注帧 {elapsed_ms:.0f} ms）")

    def set_running_controls(self, running: bool) -> None:
        """运行态控件联动：运行中禁用摄像头与模型选择、开始键置「停止」；未运行恢复并显示「就绪」。

        录制控件由录制控制器按 ``flags.recording_ready`` 联动；动作分析按钮始终可用。
        """
        capture = self.view.capture
        for combo in (capture.camera_combo, capture.camera_combo_2):
            if running:
                combo.configure(state="disabled")
            elif self.cameras.entries:
                combo.configure(state="readonly")
        self.cameras.set_refresh_enabled()
        capture.model_combo.configure(state="disabled" if running else "readonly")
        self.recording.session_running(running)
        self.view.compare_btn.configure(state="normal")
        capture.start_btn.configure(state="disabled" if running else "normal", text="停止" if running else "开始")
        capture.stop_btn.configure(state="normal" if running else "disabled")
        if not running:
            self.status("就绪")
