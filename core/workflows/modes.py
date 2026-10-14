# -*- coding: utf-8 -*-
"""运行模式与会话要求：普通、考试、学生练习三种模式各自声明摄像头、采集格式、锁定参数、片段自动比对与预检。

会话配置 = 用户设置快照叠加当前模式的要求，在开始时计算一次（``ModeManager.session_settings``）。
学生练习只在会话配置里锁定骨架、手部、自动比对、在线识别，不改写用户的勾选（V-5）。表格见设计文档 §8.5。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from core.pose import missing_model_message
from core.recording import validate_auto_compare_template_lists

ModeName = Literal["normal", "exam", "practice"]
_T = TypeVar("_T")


@dataclass(frozen=True)
class CaptureFormat:
    """模式要求的采集尺寸与给不出时的处理（D-6、D-17：``require`` 直接失败，``prefer`` 接受实际尺寸）。"""

    width: int
    height: int
    policy: Literal["prefer", "require"] = "prefer"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class PreflightContext:
    """预检用到的事实，由入口层在点击时采集。``model_installed`` 按模型键（如 ``pose_lite``）查询。"""

    dual_active: bool = False
    recording_ready: bool = False
    skeleton_on: bool = False
    front_templates: tuple[Path, ...] = ()
    side_templates: tuple[Path, ...] = ()
    model_installed: Callable[[str], bool] = lambda _key: True
    identity_error: str | None = None
    cameras_error: str | None = None


def _no_preflight(_ctx: PreflightContext) -> list[str]:
    return []


@dataclass(frozen=True)
class SessionRequirements:
    cameras: Literal["any", "dual"]
    fmt: CaptureFormat
    overrides: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))  # 锁定的会话参数
    segment_auto_compare: bool | None = None  # None 表示跟随用户开关
    preflight: Callable[[PreflightContext], list[str]] = _no_preflight  # 返回不满足的条件（按检查顺序）


@dataclass(frozen=True)
class Mode:
    name: ModeName
    requirements: SessionRequirements


def exam_preflight(ctx: PreflightContext) -> list[str]:
    """开考条件：双摄会话已就绪、骨架关、模板池至少一侧可用、heavy 与 lite 模型已安装。"""
    if not ctx.dual_active:
        return ["请先选择双摄像头并点击「开始」进入双摄会话。"]
    if not ctx.recording_ready:
        return ["双摄尚未就绪，请稍候再试。"]
    problems: list[str] = []
    # 骨架会话跳过站位检测，首位会永久卡在 wait_enter
    if ctx.skeleton_on:
        problems.append(
            "考试模式不支持「开启骨架」会话（占用检测会被跳过）。请取消骨架后重新「开始」双摄，再开考。"
        )
    # 多模板池：正对正、侧对侧，至少一侧非空且模板可用（heavy+pose33_v3）即可开考。
    try:
        validate_auto_compare_template_lists(list(ctx.front_templates), list(ctx.side_templates))
    except Exception as exc:
        problems.append(
            f"正/侧模板池不可用（须至少一侧含可用 heavy+pose33_v3 模板，请在主界面录制区添加）：{exc}"
        )
    if not ctx.model_installed("pose_heavy"):
        problems.append(missing_model_message("exam_heavy"))
    if not ctx.model_installed("pose_lite"):
        problems.append(missing_model_message("exam_occupancy_lite"))
    return problems


def practice_preflight(ctx: PreflightContext) -> list[str]:
    """练习开始条件：学号、动作、左右式有效；正/侧双摄已选；lite 模型已安装。"""
    problems = [msg for msg in (ctx.identity_error, ctx.cameras_error) if msg]
    if not ctx.model_installed("pose_lite"):
        problems.append(missing_model_message("student_lite"))
    return problems


# D-17：三种模式统一按 1080p 采集，给不出即失败；正面竖屏、侧面横屏由各路旋转设置得到
CAPTURE_FORMAT = CaptureFormat(1920, 1080, "require")

NORMAL = Mode("normal", SessionRequirements(cameras="any", fmt=CAPTURE_FORMAT))
EXAM = Mode(
    "exam",
    SessionRequirements(
        cameras="dual",
        fmt=CAPTURE_FORMAT,
        overrides=MappingProxyType({"record_skeleton": False}),
        segment_auto_compare=True,
        preflight=exam_preflight,
    ),
)
PRACTICE = Mode(
    "practice",
    SessionRequirements(
        cameras="dual",
        fmt=CAPTURE_FORMAT,
        overrides=MappingProxyType(
            {"record_skeleton": False, "enable_hands": False, "online_match_enabled": False, "auto_compare": False}
        ),
        segment_auto_compare=False,
        preflight=practice_preflight,
    ),
)


class ModeManager:
    """当前模式。考试模式在开考时进入、主会话停止或换名单时退出；学生练习在进入/退出练习界面时切换。"""

    def __init__(self) -> None:
        self._mode = NORMAL

    @property
    def current(self) -> Mode:
        return self._mode

    @property
    def name(self) -> ModeName:
        return self._mode.name

    def is_active(self, name: ModeName) -> bool:
        return self._mode.name == name

    def enter(self, mode: Mode) -> None:
        self._mode = mode

    def leave(self, name: ModeName | None = None) -> None:
        """回到普通模式；给出 ``name`` 时只在当前正是该模式时才退出。"""
        if name is None or self._mode.name == name:
            self._mode = NORMAL

    @property
    def segment_auto_compare(self) -> bool | None:
        return self._mode.requirements.segment_auto_compare

    def preflight(self, mode: Mode, ctx: PreflightContext) -> list[str]:
        return mode.requirements.preflight(ctx)

    def session_settings(self, snapshot: _T) -> _T:
        """用户设置快照叠加模式要求：锁定参数覆盖同名字段，采集尺寸与策略按模式声明。"""
        req = self._mode.requirements
        names = {f.name for f in fields(snapshot)}  # type: ignore[arg-type]
        changes = {k: v for k, v in req.overrides.items() if k in names}
        if "capture_size" in names:
            changes["capture_size"] = req.fmt.size
        if "capture_policy" in names:
            changes["capture_policy"] = req.fmt.policy
        return replace(snapshot, **changes)  # type: ignore[type-var]
