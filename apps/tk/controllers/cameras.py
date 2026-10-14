# -*- coding: utf-8 -*-
"""摄像头选择：后台枚举、正/侧下拉与偏好设备的恢复和持久化、按模式冻结采集规格，并经 ``CameraService`` 声明预热。

所有摄像头打开都经 ``CameraService`` 的池；枚举结果经 dispatcher 回到 Tk 线程，关窗后不再触碰控件。
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from core.foundation import load_camera_selection, save_camera_selection
from core.live import CameraInput, FrameInput, SessionConfig, VideoFileInput
from core.media import PRIMARY, SECONDARY, CameraService, CaptureSpec, ProcessCamera
from core.media import CameraDevice as CameraEntry
from core.workflows import ModeManager

from apps.tk.input_source import InputSourceState
from apps.tk.controllers.flags import LiveFlags
from apps.tk.views.capture_panel import NO_SECOND_CAMERA

JOIN_TIMEOUT_S = 0.05     # 取消预热时每次 join 的片段，避免 Tk 主线程长时间无响应
RELEASE_WAIT_S = 3.0      # 关窗时等待采集进程真实退出的预算


def open_camera_process(index: int, *, width: int, height: int, stop_event: threading.Event,
                        expected_name: str | None = None, policy: str = "prefer") -> Any:
    """池的唯一打开入口：每路一个 spawn 采集进程（调用时解析模块级 ``ProcessCamera``）。"""
    return ProcessCamera(index, width=width, height=height, stop_event=stop_event,
                         expected_name=expected_name, policy=policy)


def make_camera_service() -> CameraService:
    return CameraService(capture_factory=open_camera_process, join_timeout=JOIN_TIMEOUT_S)


class CameraController:
    """``view`` 是 ``CapturePanel``（下拉、刷新按钮、输入源提示）；``set_status`` 写状态栏。"""

    def __init__(self, view: Any, service: CameraService, *, modes: ModeManager, flags: LiveFlags,
                 post: Callable[..., Any], session_active: Callable[[], bool],
                 set_status: Callable[[str], None]) -> None:
        self.view = view
        self.service = service
        self.modes = modes
        self.flags = flags
        self._post = post
        self._session_active = session_active
        self._set_status = set_status
        self.entries: list[CameraEntry] = []
        self.source = InputSourceState()
        self.preferred_primary, self.preferred_secondary = load_camera_selection()
        self.enumerating = threading.Event()

    # ------------------------------------------------------------------ 枚举
    def start_enumeration(self) -> None:
        """后台枚举可用摄像头；结果经 dispatcher 在 Tk 线程应用（兜底探测跳过池占用的设备，V-6）。"""
        if self.enumerating.is_set():
            return
        self.enumerating.set()
        self.set_refresh_enabled()
        self.view.camera_combo.configure(state="disabled")
        self.service.refresh(
            on_done=lambda snapshot: self._post(lambda: self.apply_entries(list(snapshot.devices), snapshot.ok)))

    def refresh(self) -> None:
        """刷新列表；会话中或枚举中忽略。刷新不放弃预热（V-6）。"""
        if self._session_active() or self.enumerating.is_set():
            return
        self.start_enumeration()

    def set_refresh_enabled(self) -> None:
        """仅当非枚举中且没有会话时启用刷新。"""
        enabled = not self.enumerating.is_set() and not self._session_active()
        self.view.refresh_btn.configure(state="normal" if enabled else "disabled")

    def apply_entries(self, entries: list[CameraEntry], ok: bool) -> None:
        """重建两个下拉并恢复偏好设备；偏好设备暂时缺失只做界面回退，不覆盖持久化值。"""
        try:
            if not ok:
                self._set_status("刷新摄像头失败，已保留原有列表。")
                if self.entries:
                    self.view.camera_combo.configure(state="readonly")
                self.sync_warmup()
                return
            self.entries = list(entries)
            if not entries:
                self._show_no_cameras()
                return
            labels = [entry.label for entry in entries]
            self.view.camera_combo.configure(values=labels, state="readonly")
            current = self.label_for(self.preferred_primary) or self.view.camera_choice.get()
            self.view.camera_choice.set(current if current in labels else labels[0])
            labels_2 = [NO_SECOND_CAMERA] + labels
            self.view.camera_combo_2.configure(values=labels_2, state="readonly")
            current_2 = self.label_for(self.preferred_secondary) or self.view.camera_choice_2.get()
            self.view.camera_choice_2.set(current_2 if current_2 in labels_2 else NO_SECOND_CAMERA)
            # 两个选择都稳定后再统一同步预热，避免先按旧的第二路选择启动一次无效预热。
            self._select(self.view.camera_choice.get())
        finally:
            self.enumerating.clear()
            self.set_refresh_enabled()

    def _show_no_cameras(self) -> None:
        self.view.camera_combo.configure(values=[], state="disabled")
        self.view.camera_choice.set("")
        self.view.source_hint_var.set("未检测到可用摄像头")
        if self.source.kind == "camera":
            self.source.clear()
        self.release_warmups()
        self.view.camera_combo_2.configure(values=[NO_SECOND_CAMERA], state="disabled")
        self.view.camera_choice_2.set(NO_SECOND_CAMERA)

    # ------------------------------------------------------------------ 选择
    def index_for(self, label: str) -> int | None:
        return next((entry.index for entry in self.entries if entry.label == label), None)

    def label_for(self, index: int | None) -> str | None:
        return next((entry.label for entry in self.entries if entry.index == index), None)

    def _select(self, label: str) -> None:
        index = self.index_for(label)
        if index is not None:
            self.source.select_camera(index)
            self.view.source_hint_var.set(self.source.hint_text())
            self.sync_warmup()

    def primary_selected(self) -> None:
        index = self.index_for(self.view.camera_choice.get())
        if index is None:
            return
        self.preferred_primary = index
        self._select(self.view.camera_choice.get())
        save_camera_selection(self.preferred_primary, self.preferred_secondary)

    def secondary_selected(self) -> None:
        label = self.view.camera_choice_2.get()
        index = self.index_for(label)
        if label != NO_SECOND_CAMERA and index is None:
            return
        self.preferred_secondary = index
        save_camera_selection(self.preferred_primary, self.preferred_secondary)
        self.sync_warmup()

    def secondary_label(self) -> str:
        """第二路下拉的当前文本；未选为空串。"""
        label = (self.view.camera_choice_2.get() or "").strip()
        return "" if label == NO_SECOND_CAMERA else label

    def dual_selection_problem(self) -> str | None:
        """学生练习需要正/侧双摄；返回缺少的选择，齐全返回 None。"""
        if not self.secondary_label():
            return "学生练习需要正/侧双摄像头，请选择第二摄像头（侧面）。"
        if not (self.view.camera_choice.get() or "").strip():
            return "请选择第一摄像头（正面）。"
        return None

    def selected_indices(self) -> tuple[int | None, int | None]:
        primary = int(self.source.value) if self.source.kind == "camera" and self.source.value.isdigit() else None
        label = self.secondary_label()
        secondary = self.index_for(label) if label else None
        if primary is None or secondary == primary:
            secondary = None
        return primary, secondary

    # ------------------------------------------------------------------ 采集规格与预热
    def capture_size(self) -> tuple[int, int]:
        """当前模式的请求尺寸；只在主线程构造 spec 时读取一次，随后冻结。"""
        return self.modes.current.requirements.fmt.size

    def capture_policy(self) -> str:
        """学生练习严格要求 1080p，驱动给不出时失败并提示（D-6）；其余模式接受实际尺寸。"""
        return self.modes.current.requirements.fmt.policy

    def camera_names(self) -> tuple[tuple[int, str], ...]:
        """枚举得到的设备名，供打开时复核（V-3）；没有名称的设备不复核。"""
        return tuple((entry.index, entry.name) for entry in self.entries if getattr(entry, "name", None))

    def spec(self, index: int) -> CaptureSpec:
        width, height = self.capture_size()
        return CaptureSpec(int(index), width, height, expected_name=dict(self.camera_names()).get(int(index)),
                           policy=self.capture_policy())

    def sync_warmup(self, *, allow_running: bool = False) -> None:
        """按两个下拉和当前模式声明预热需求；配置一致则复用，不一致由池重新协商。"""
        if self.flags.closing or (self._session_active() and not allow_running):
            return
        primary, secondary = self.selected_indices()
        specs = {role: self.spec(index) for role, index in ((PRIMARY, primary), (SECONDARY, secondary))
                 if index is not None}
        try:
            self.service.prepare(specs)
        except Exception as exc:
            self.release_warmups()
            self._set_status(f"摄像头预热失败：{exc}")

    def release_warmups(self) -> None:
        """放弃所有预热；释放在后台完成，重开同编号时由池等待真实释放。"""
        try:
            self.service.prepare({})
        except Exception:
            pass

    def make_input(self, config: SessionConfig) -> FrameInput:
        """会话输入：视频文件直接读；摄像头只登记设备句柄，等待与接管在会话线程上进行。"""
        if config.is_file:
            return VideoFileInput(config.video_path)
        return CameraInput(self.service.open_stream(config.cameras))

    def close(self) -> None:
        """关窗：关闭池并在预算内等待采集进程真实退出；超时由诊断日志记录。"""
        try:
            self.service.close(RELEASE_WAIT_S)
        except Exception:
            pass
