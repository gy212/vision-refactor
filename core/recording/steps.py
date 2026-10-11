# -*- coding: utf-8 -*-
"""录后处理步骤：``Step`` 协议与通用步骤（录像检查、转码、可读性校验、比对开关、考试池评分），以及常用处理计划。

每个步骤声明跑在 ``io`` 线还是 ``analysis`` 线（设计文档 §8.4）。步骤返回 ``None`` 表示继续，
返回 ``Terminal`` 表示片段到达终态；失败抛 ``PostprocessError``（稳定的 ``code``），取消抛 ``Cancelled``。
考试裁剪步骤在 ``workflows/exam/clip.py``，由构建计划的一方放进计划，本模块不导入它。
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from core.evaluation.matching.engine import match_video
from core.evaluation.scoring import exam_pool
from core.evaluation.templates import TemplateError
from core.foundation.jsonable import to_jsonable
from core.foundation.paths import templates_dir
from core.foundation.tasks import Cancelled, CancelToken
from core.media.video_source import first_frame_ok
from core.media.writer import transcode_to_h264
from core.pose.engine import missing_model_message
from core.pose.models import MEDIAPIPE_MODELS, is_installed
from core.pose.raw_series import extract_pose_raw_series

Lane = Literal["io", "analysis"]
SegmentStatus = Literal[
    "queued", "transcoding", "validating", "comparing", "completed", "failed", "skipped", "cancelled",
]
TERMINAL_STATUSES = frozenset({"completed", "failed", "skipped", "cancelled"})
VIEW_LABELS = {"front": "正面", "side": "侧面"}

TranscodeFn = Callable[[Path, threading.Event], Path | None]
VideoValidator = Callable[[Path], bool]
ModelAvailable = Callable[[], bool]
PoolCompare = Callable[..., Any]


class PostprocessError(RuntimeError):
    """步骤失败；``code`` 写进结果 JSON 的 ``error.code``。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Terminal:
    status: Literal["completed", "skipped"]
    message: str
    error_code: str | None = None
    result: dict[str, Any] | None = None


@dataclass
class StepContext:
    """一个片段在各步骤间传递的可变状态。``publish`` 由 finalizer 提供，发布进度状态。"""

    segment_id: str
    segment_dir: Path | None
    views: tuple[str, ...]
    sources: dict[str, Path]
    frames: dict[str, int]
    errors: dict[str, str]
    stop_evt: threading.Event
    publish: Callable[[SegmentStatus, str], None]
    videos: dict[str, Path | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # 计划附带的说明（如考试裁剪的提示），在结果 JSON 里排在步骤警告之后
    notes: list[str] = field(default_factory=list)

    def video(self, view: str) -> Path:
        """该路当前最新的视频：转码后的产物，否则是录像源。"""
        return self.videos.get(view) or self.sources[view]


class Step(Protocol):
    lane: Lane

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None: ...


def _raise_if_cancelled(ctx: StepContext, cancel: CancelToken) -> None:
    """任务 token 或本片段的 ``stop_evt`` 任一置位即取消（转码等旧接口只认 ``stop_evt``）。"""
    if cancel.cancelled or ctx.stop_evt.is_set():
        raise Cancelled(cancel.reason or "后台比对已取消")


# --------------------------------------------------------------------------- #
# 默认依赖
# --------------------------------------------------------------------------- #


def default_transcode(path: Path, stop_evt: threading.Event) -> Path | None:
    return transcode_to_h264(path, stop_evt=stop_evt)


def default_video_validator(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    return first_frame_ok(path)


def default_model_available() -> bool:
    spec = next((item for item in MEDIAPIPE_MODELS if item.key == "pose_heavy"), None)
    return bool(spec is not None and is_installed(spec))


# --------------------------------------------------------------------------- #
# 模板校验（考试池评分与 Tk 模板池共用）
# --------------------------------------------------------------------------- #


def default_template_paths() -> tuple[Path, Path]:
    root = templates_dir()
    return root / "standard_front_heavy.npz", root / "standard_side_heavy.npz"


def try_load_heavy_template(path: Path | str | None) -> Path | None:
    """模板路径可用则返回规范化 Path，否则 None（缺文件/无效均视为未填入）。"""
    return exam_pool.usable_template(path)


def validate_template_pair(front_path: Path, side_path: Path) -> None:
    """考试等正式路径：正/侧模板必须同时可用且布局一致。"""
    try:
        exam_pool.check_pair(Path(front_path), Path(side_path))
    except TemplateError as exc:
        raise PostprocessError(exc.code, str(exc)) from exc


def validate_auto_compare_templates(
    front_path: Path | str | None,
    side_path: Path | str | None,
) -> tuple[Path | None, Path | None]:
    """一条龙自动比对：至少一侧可用即可；两侧都缺才 ``template_missing``，两侧都可用时校验布局一致。"""
    front = try_load_heavy_template(front_path)
    side = try_load_heavy_template(side_path)
    if front is None and side is None:
        raise PostprocessError(
            "template_missing",
            "正/侧标准模板均未填入或不可用（heavy + pose33_v3）。"
            "请在主界面「录制」区填入至少一侧模板。",
        )
    if front is not None and side is not None:
        validate_template_pair(front, side)
    return front, side


def template_action_name(path: Path | str) -> str:
    """动作名 = 模板文件 stem（如 直拳.npz → "直拳"）。"""
    return Path(path).stem


def validate_auto_compare_template_lists(
    front_list: list[Path | str] | None,
    side_list: list[Path | str] | None,
) -> tuple[list[Path], list[Path]]:
    """多模板池校验：各自过滤出可用的 heavy+pose33_v3 模板（单个坏模板跳过）；两池都空才报错。"""
    try:
        return exam_pool.usable_pools(front_list, side_list)
    except TemplateError as exc:
        raise PostprocessError(exc.code, str(exc)) from exc


# --------------------------------------------------------------------------- #
# 通用步骤
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckRecording:
    """录像快照检查：写错误、路径属于片段目录、帧数有效且各路一致；通过后发布 ``queued``。"""

    lane: Lane = "io"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        for view in ctx.views:
            if ctx.errors.get(view):
                raise PostprocessError(f"{view}_recording_failed", f"{VIEW_LABELS.get(view, view)}录像失败：{ctx.errors[view]}")
        if not all(ctx.sources.get(view) for view in ctx.views):
            raise PostprocessError("recording_missing", "正面或侧面录像路径缺失")
        if ctx.segment_dir is not None:
            segment_dir = Path(ctx.segment_dir).resolve(strict=False)
            for view in ctx.views:
                source = ctx.sources[view]
                try:
                    Path(source).resolve(strict=False).relative_to(segment_dir)
                except (OSError, ValueError) as exc:
                    raise PostprocessError(
                        "recording_path_mismatch",
                        f"{VIEW_LABELS.get(view, view)}录像路径不属于当前片段目录：{source}",
                    ) from exc
        counts = [int(ctx.frames.get(view, 0)) for view in ctx.views]
        if any(count <= 0 for count in counts):
            raise PostprocessError("recording_empty", "正面或侧面录像没有有效帧")
        if len(set(counts)) > 1:
            detail = " ".join(f"{view}={ctx.frames.get(view, 0)}" for view in ctx.views)
            raise PostprocessError("frame_count_mismatch", f"正侧录像帧数不一致：{detail}")
        ctx.publish("queued", "已进入后台比对队列")
        return None


@dataclass(frozen=True)
class Transcode:
    """逐路把回退产物（AVI）转成 H.264 MP4；转码失败回退使用 AVI 时记警告。"""

    transcode: TranscodeFn = default_transcode
    message: str = "正在顺序完成正侧录像转码"
    lane: Lane = "io"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        ctx.publish("transcoding", self.message)
        for view in ctx.views:
            _raise_if_cancelled(ctx, cancel)
            ctx.videos[view] = self._one(ctx, view)
        return None

    def _one(self, ctx: StepContext, view: str) -> Path | None:
        source = ctx.sources.get(view)
        if source is None:
            return None
        label = VIEW_LABELS.get(view, view)
        source = Path(source)
        try:
            final_path = self.transcode(source, ctx.stop_evt)
        except InterruptedError:
            raise
        except Exception as exc:
            raise PostprocessError("transcode_failed", f"{label}录像转码失败：{exc}") from exc
        if final_path is None:
            raise PostprocessError("transcode_failed", f"{label}录像转码未返回文件路径")
        final_path = Path(final_path)
        if source.suffix.lower() == ".avi" and final_path.suffix.lower() == ".avi":
            ctx.warnings.append(f"{label}录像 H.264 转码失败，已回退使用 AVI")
        return final_path


@dataclass(frozen=True)
class ValidateVideos:
    """各路最终视频存在、非空且首帧可解码。"""

    validator: VideoValidator = default_video_validator
    lane: Lane = "io"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        _raise_if_cancelled(ctx, cancel)
        ctx.publish("validating", "正在校验录像、模板和模型")
        for view in ctx.views:
            path = ctx.video(view)
            label = VIEW_LABELS.get(view, view)
            try:
                valid = self.validator(Path(path))
            except Exception as exc:
                raise PostprocessError("video_unreadable", f"{label}录像无法读取：{exc}") from exc
            if not valid:
                raise PostprocessError("video_unreadable", f"{label}录像不存在、为空或无法读取：{path}")
        return None


@dataclass(frozen=True)
class CompareGate:
    """带骨架的录像、或关闭自动比对时，片段到此以 ``skipped`` 结束。"""

    record_skeleton: bool
    auto_compare: bool
    lane: Lane = "io"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        _raise_if_cancelled(ctx, cancel)
        if self.record_skeleton:
            return Terminal("skipped", "带骨架录像未自动比对，请关闭骨架后重新录制", "annotated_recording")
        if not self.auto_compare:
            return Terminal("skipped", "仅录制模式，已保存视频未自动比对", "auto_compare_disabled")
        return None


@dataclass(frozen=True)
class ScoreExamPool:
    """正对正池、侧对侧池逐模板比对，按 ``EXAM_POOL`` 方案合并（评分直接用 evaluation.scoring）。"""

    front_templates: tuple[Path, ...]
    side_templates: tuple[Path, ...]
    compare: PoolCompare = match_video
    model_available: ModelAvailable = default_model_available
    lane: Lane = "analysis"

    def run(self, ctx: StepContext, cancel: CancelToken) -> Terminal | None:
        front_tpls, side_tpls = validate_auto_compare_template_lists(
            list(self.front_templates), list(self.side_templates)
        )
        if not self.model_available():
            raise PostprocessError("model_missing", missing_model_message("postprocess_heavy"))
        _raise_if_cancelled(ctx, cancel)
        if front_tpls and side_tpls:
            message = f"正在执行正侧多模板 DTW 比对（正{len(front_tpls)}/侧{len(side_tpls)}）"
        elif front_tpls:
            message = f"正在执行正面多模板 DTW 比对（{len(front_tpls)} 个，侧面池为空）"
            ctx.warnings.append("侧面模板池为空，仅比对正面")
        else:
            message = f"正在执行侧面多模板 DTW 比对（{len(side_tpls)} 个，正面池为空）"
            ctx.warnings.append("正面模板池为空，仅比对侧面")
        ctx.publish("comparing", message)
        result = exam_pool.score_exam_pools(
            front_tpls, side_tpls, ctx.video("front"), ctx.video("side"),
            compare=self.compare, raw_series=extract_pose_raw_series, stop_evt=ctx.stop_evt,
        )
        _raise_if_cancelled(ctx, cancel)
        message = "后台多模板比对完成" if front_tpls and side_tpls else "后台单侧多模板比对完成"
        return Terminal("completed", message, result=result_payload(result))


def result_payload(result: Any) -> dict[str, Any]:
    """考试池评分结果 → 结果 JSON 的 ``result`` 字段。"""

    def _segment(value: Any) -> dict[str, int] | None:
        return None if value is None else {"start": int(value[0]), "end": int(value[1])}

    def _opt_float(value: Any) -> float | None:
        return None if value is None else float(value)

    return {
        "front_score": _opt_float(getattr(result, "front_score", None)),
        "side_score": _opt_float(getattr(result, "side_score", None)),
        "combined_score": float(result.combined_score),
        "combined_percent": int(result.combined_percent),
        "front_matches": to_jsonable(getattr(result, "front_matches", ())),
        "side_matches": to_jsonable(getattr(result, "side_matches", ())),
        "front_segment": _segment(getattr(result, "front_segment", None)),
        "side_segment": _segment(getattr(result, "side_segment", None)),
        "action_scores": to_jsonable(getattr(result, "action_scores", ())),
    }


# --------------------------------------------------------------------------- #
# 常用处理计划的步骤（设计文档 §8.4 表）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepDeps:
    """通用步骤的可替换依赖（测试注入假实现）。"""

    transcode: TranscodeFn = default_transcode
    video_validator: VideoValidator = default_video_validator
    compare: PoolCompare = match_video
    model_available: ModelAvailable = default_model_available


def dual_segment_steps(
    *,
    record_skeleton: bool,
    auto_compare: bool,
    front_templates: tuple[Path, ...] = (),
    side_templates: tuple[Path, ...] = (),
    deps: StepDeps = StepDeps(),
) -> tuple[Step, ...]:
    """双路片段：检查 → 转码 → 校验 →（自动比对开时）考试池评分。考试计划在前面加裁剪步骤。"""
    return (
        CheckRecording(),
        Transcode(deps.transcode),
        ValidateVideos(deps.video_validator),
        CompareGate(record_skeleton=record_skeleton, auto_compare=auto_compare),
        ScoreExamPool(
            tuple(front_templates), tuple(side_templates),
            compare=deps.compare, model_available=deps.model_available,
        ),
    )


def single_segment_steps(deps: StepDeps = StepDeps()) -> tuple[Step, ...]:
    """单路片段：只转码（V-4）。"""
    return (Transcode(deps.transcode),)

