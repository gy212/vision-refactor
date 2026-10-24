# -*- coding: utf-8 -*-
"""录制性能测试的桌面流程：检查能否开始、在后台运行 ``run_recording_benchmark``、把报告存到桌面并把结果交给对话框。

入口是主窗口的隐藏快捷键 Ctrl+Shift+B（``DesktopApp``）。测试用主界面当前选中的摄像头、旋转与骨架模型设置；
"按当前设置录像"一段另用点「开始」时的会话配置跑正式的处理循环（``BenchmarkSessionLoad``），预览画到主窗口。
测试不下载模型：开始前查好本机缺哪些模型（``missing_models``），缺的那部分负载不测，结论最前面提醒去模型管理安装。
临时录像写在当前录制目录下、测完删除；报告固定保存为桌面上的 ``录制性能测试_<日期>_<时间>.json``，
桌面不可写时退到数据目录下的 ``录制性能测试`` 文件夹，实际路径都显示在对话框里。
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.foundation import CancelToken, TaskGroup, desktop_dir, models_dir, repo_root
from core.live import SessionConfig
from core.media import PRIMARY, SECONDARY
from core.pose import create_engine, missing_model_message
from core.services import BENCHMARK_PHASES, BenchmarkSetup, run_recording_benchmark, save_benchmark_report

from apps.tk.build_info import load_build_info
from apps.tk.controllers.benchmark_load import BenchmarkSessionLoad
from apps.tk.controllers.recording import model_installed

MIN_FREE_GB = 3.0          # 每段写盘约 1 GB（两路 1080p × 30 s），段末删除
FALLBACK_FOLDER = "录制性能测试"


def report_directory() -> Path:
    """报告目录：桌面；取不到桌面时为数据目录下的 ``录制性能测试``。"""
    return desktop_dir() or repo_root() / FALLBACK_FOLDER


def total_seconds() -> int:
    return int(sum(phase.seconds for phase in BENCHMARK_PHASES))


def missing_models(pose_variant: str, enable_hands: bool,
                   installed: Callable[[str], bool] = model_installed) -> dict[str, str]:
    """本机缺的、测试要用的模型：``lite``（站位检测；考试与练习也都离不开）与 ``skeleton``（按当前设置的骨架识别），
    值是给用户看的提示。"""
    missing: dict[str, str] = {}
    if not installed("pose_lite"):
        missing["lite"] = missing_model_message("benchmark_lite")
    absent = [name for key, name in ((f"pose_{pose_variant}", f"pose_landmarker_{pose_variant}.task"),
                                     ("hand", "hand_landmarker.task"))
              if (key != "hand" or enable_hands) and not installed(key)]
    if absent:
        missing["skeleton"] = missing_model_message("benchmark_skeleton", models="、".join(absent))
    return missing


@dataclass(frozen=True)
class BenchmarkContext:
    """测试需要从主界面读取的状态（都在 Tk 线程读取）。"""

    cameras: Any                                  # CameraController
    session_active: Callable[[], bool]
    mode_name: Callable[[], str]
    rotation_of: Callable[[str], int]
    pose_variant: Callable[[], str]
    enable_hands: Callable[[], bool]
    record_dir: Callable[[], Path]
    post: Callable[[Callable[[], Any]], Any]
    session_config: Callable[[], SessionConfig]   # 此刻点「开始」会用的会话配置
    preview: Any                                  # 主窗口的 PreviewSurface


class BenchmarkController:
    """``view`` 是 ``BenchmarkDialog``：``show_progress``、``show_result``、``show_problem``、``set_running``。"""

    def __init__(self, ctx: BenchmarkContext, view: Any) -> None:
        self.ctx = ctx
        self.view = view
        self._tasks = TaskGroup("benchmark")
        self._token: CancelToken | None = None
        self._load: BenchmarkSessionLoad | None = None
        self.last_report_path: Path | None = None

    @property
    def running(self) -> bool:
        return self._token is not None

    def problem(self) -> str | None:
        """不能开始测试的原因；可以开始时返回 None。"""
        if self.ctx.session_active():
            return "请先在主界面点「停止」结束预览，再开始测试。"
        if self.ctx.mode_name() != "normal":
            return "请先退出考试或学生练习，回到普通模式后再测试。"
        primary, _secondary = self.ctx.cameras.selected_indices()
        if primary is None:
            return "请先在主界面选择摄像头（正面；有侧面摄像头时也选上）。"
        directory = self.ctx.record_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(directory).free / 2**30
        except OSError as exc:
            return f"录制目录不可用：{directory}（{exc}）"
        if free_gb < MIN_FREE_GB:
            return f"录制目录所在磁盘只剩 {free_gb:.1f} GB，测试需要至少 {MIN_FREE_GB:.0f} GB 空闲空间。"
        return None

    def _setup(self) -> BenchmarkSetup:
        cameras = self.ctx.cameras
        primary, secondary = cameras.selected_indices()
        indices = {PRIMARY: primary, SECONDARY: secondary}
        specs = {role: cameras.spec(index) for role, index in indices.items() if index is not None}
        labels = {role: cameras.label_for(index) or str(index) for role, index in indices.items() if index is not None}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            build = load_build_info()
        except (OSError, ValueError):
            build = None
        missing = missing_models(self.ctx.pose_variant(), self.ctx.enable_hands())
        self._load = BenchmarkSessionLoad(self.ctx.session_config(), rotation_of=self.ctx.rotation_of,
                                          preview=self.ctx.preview, missing=missing)
        return BenchmarkSetup(
            stream=cameras.service.open_stream(specs),
            rotations={role: int(self.ctx.rotation_of(role)) for role in specs},
            labels=labels,
            work_dir=self.ctx.record_dir() / f"_录制性能测试_临时_{stamp}",
            engine_factory=lambda spec: create_engine(spec, models_dir=models_dir(), download=False),
            pose_variant=self.ctx.pose_variant(),
            enable_hands=self.ctx.enable_hands(),
            extra={"build": build, "record_dir": str(self.ctx.record_dir())},
            session_load=self._load,
            missing_models=missing,
        )

    def start(self) -> None:
        if self.running:
            return
        problem = self.problem()
        if problem is not None:
            self.view.show_problem(problem)
            return
        try:
            setup = self._setup()
        except Exception as exc:
            self.view.show_problem(f"准备测试失败：{exc}")
            return
        token = self._token = CancelToken()
        self.view.set_running(True)
        post = self.ctx.post

        def progress(text: str, done: float) -> None:
            post(lambda: self.view.show_progress(text, done))

        def run(_task_token: CancelToken) -> None:
            report = run_recording_benchmark(setup, token, progress)
            path, error = self._save(report)
            post(lambda: self._finished(report, path, error))

        self._tasks.spawn("run", run, owner="benchmark", close_budget_s=3.0)

    def _save(self, report: dict) -> tuple[Path | None, str | None]:
        """先存桌面，失败时存到数据目录；都失败时返回错误文字。"""
        last_error = None
        for directory in dict.fromkeys((report_directory(), repo_root() / FALLBACK_FOLDER)):
            try:
                return save_benchmark_report(report, directory), None
            except OSError as exc:
                last_error = f"{directory}：{exc}"
        return None, f"报告保存失败（{last_error}）"

    def _finished(self, report: dict, path: Path | None, error: str | None) -> None:
        self._token = None
        self.last_report_path = path
        self.view.set_running(False)
        self.view.show_result(dict(report.get("conclusion") or {}), list(report.get("summary", [])), path, error)

    def render_preview(self) -> None:
        """Tk 主线程定时调用：把"按当前设置录像"一段的预览画到主窗口（没有在测这一段时只恢复布局）。"""
        if self._load is not None:
            self._load.render()

    def cancel(self) -> None:
        if self._token is not None:
            self._token.cancel("user_cancel")

    def close(self) -> None:
        """关窗或主程序退出：取消仍在跑的测试（后台线程仍会收尾、交回摄像头并保存已测到的部分）。"""
        self.cancel()


__all__ = ["BenchmarkContext", "BenchmarkController", "report_directory", "total_seconds"]
