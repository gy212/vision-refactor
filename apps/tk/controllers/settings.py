# -*- coding: utf-8 -*-
"""设置窗口的控制器：MediaPipe 模型的安装状态文字、当前所用模型汇总，以及在后台任务里逐个下载模型。

下载源为 Google 官方直链；网络不通时只在状态区提示用户自行配置代理。视图只需实现 ``SettingsView``。
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from core.foundation import TaskGroup
from core.pose import MEDIAPIPE_MODELS, ModelSpec, download_model, installed_size_mb, is_installed

Post = Callable[[Callable[[], Any]], Any]

_POSE_KEYS = {"lite": "pose_lite", "full": "pose_full", "heavy": "pose_heavy"}


class SettingsView(Protocol):
    def set_status(self, text: str) -> None: ...
    def set_progress(self, downloaded: int, total: int | None) -> None: ...
    def refresh_states(self) -> None: ...
    def set_buttons_enabled(self, enabled: bool) -> None: ...
    def show_download_errors(self, detail: str) -> None: ...


def active_model_keys(current_pose: str, hands_enabled: bool) -> set[str]:
    """主界面当前实际会用到的模型 key：所选姿态档位，加上开启时的手部模型。"""
    keys = {_POSE_KEYS.get((current_pose or "full").strip().lower(), "pose_full")}
    if hands_enabled:
        keys.add("hand")
    return keys


def model_state(spec: ModelSpec, active: bool) -> tuple[str, str]:
    """模型一行的 (状态文字, 按钮文字)；当前使用的模型前加「● 使用中」。"""
    mark = "● 使用中  " if active else ""
    if is_installed(spec):
        size = installed_size_mb(spec)
        return mark + (f"已安装 ({size} MB)" if size else "已安装"), "重新下载"
    approx = f"约 {spec.approx_mb} MB" if spec.approx_mb else ""
    return (mark + f"未安装 {approx}").strip(), "下载"


def current_summary(current_pose: str, hands_enabled: bool, active_keys: set[str],
                    specs: Iterable[ModelSpec] = MEDIAPIPE_MODELS) -> str:
    """「当前模型」区的汇总：所选姿态档位、手部开关，以及所用模型是否就绪。"""
    by_key = {spec.key: spec for spec in specs}
    parts: list[str] = []
    missing: list[str] = []
    for key in ("pose_lite", "pose_full", "pose_heavy", "hand"):
        spec = by_key.get(key)
        if key not in active_keys or spec is None:
            continue
        ready = is_installed(spec)
        parts.append(f"{spec.label}（{'已就绪' if ready else '缺失'}）")
        if not ready:
            missing.append(spec.label)
    hands_txt = "开启" if hands_enabled else "关闭"
    pose = (current_pose or "full").strip().lower()
    summary = f"姿态模型：{pose}　手部检测：{hands_txt}\n" + "\n".join(parts)
    if missing:
        summary += "\n\n⚠ 缺失模型会导致无法开始识别，请在下方下载后再使用。"
    return summary


class SettingsController:
    """同一时刻只跑一个下载任务；``close`` 让正在下载的模型在下一块数据后中断。"""

    def __init__(self, view: SettingsView, post: Post, *, download: Callable[..., Any] = download_model) -> None:
        self._view = view
        self._post = post
        self._download = download
        self._tasks = TaskGroup("settings-dialog")
        self._stop_evt = threading.Event()

    @property
    def busy(self) -> bool:
        return bool(self._tasks.pending())

    def download(self, specs: Sequence[ModelSpec]) -> bool:
        """后台依次下载；每装好一个刷新状态，结束后汇总失败项。已有任务在跑时返回 False。"""
        if self.busy:
            return False
        self._stop_evt.clear()
        self._view.set_buttons_enabled(False)
        self._tasks.spawn("download", lambda _token: self._run(list(specs)), owner="settings")
        return True

    def close(self) -> None:
        self._stop_evt.set()

    def _status(self, text: str) -> None:
        self._post(lambda: self._view.set_status(text))

    def _run(self, specs: list[ModelSpec]) -> None:
        ok, failed = 0, []
        try:
            for spec in specs:
                if self._stop_evt.is_set():
                    break
                self._status(f"正在下载：{spec.label} …")
                try:
                    self._download(
                        spec,
                        progress_cb=lambda done, total: self._post(lambda: self._view.set_progress(done, total)),
                        should_stop=self._stop_evt.is_set,
                    )
                    ok += 1
                    self._post(self._view.refresh_states)
                except InterruptedError:
                    self._status("下载已取消。")
                    return
                except Exception as e:  # noqa: BLE001
                    failed.append((spec.label, str(e)))
            if failed:
                detail = "\n".join(f"- {name}: {err}" for name, err in failed)
                self._status(f"完成 {ok} 个，{len(failed)} 个失败。")
                self._post(lambda: self._view.show_download_errors(detail))
            else:
                self._status(f"全部完成，共下载 {ok} 个模型。")
        finally:
            self._post(self._view.refresh_states)
            self._post(lambda: self._view.set_buttons_enabled(True))
