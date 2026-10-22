# -*- coding: utf-8 -*-
"""片段处理计划的构建：开录时由工作流给出 ``SegmentIntent``，结束时按它与用户设置组装 ``ProcessingPlan``。

开录方把 ``SegmentIntent`` 放进 ``SegmentMeta.context``，片段结束时 ``build_segment_plan`` 读回它：
考试片段在步骤前加裁剪并强制比对，学生练习强制不比对，普通片段跟随结束时的自动比对开关（设计文档 §8.4、§8.5）。
单路片段只转码（V-4）。录像本身是原始画面，开了"录制骨架"的片段在录后生成骨架成片（D-26）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.recording import (
    FinishedSegment,
    ProcessingPlan,
    RenderSkeleton,
    ResultSpec,
    dual_segment_steps,
    single_segment_steps,
    skeleton_engine,
)
from core.workflows.exam.clip import EXAM_CLOSE_BUDGET_S, ExamClipStep


@dataclass(frozen=True)
class SegmentIntent:
    """开录时冻结的片段意图。``auto_compare`` 为 None 时跟随用户开关；``exam`` 是考试片段的考生信息。"""

    auto_compare: bool | None = None
    exam: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PlanSettings:
    """片段结束时读取的用户设置与模板池。"""

    record_skeleton: bool
    auto_compare: bool
    front_templates: tuple[Path, ...] = ()
    side_templates: tuple[Path, ...] = ()
    pose_variant: str = "full"        # 骨架成片用会话的模型与手部设置
    enable_hands: bool = True


def segment_intent(segment: FinishedSegment) -> SegmentIntent:
    """片段开录时登记的意图；没有登记（如旧路径开录）时按普通片段处理。"""
    context = segment.ref.meta.context
    return context if isinstance(context, SegmentIntent) else SegmentIntent()


def build_segment_plan(segment: FinishedSegment, settings: PlanSettings) -> ProcessingPlan:
    """按片段意图与结束时的设置组装处理计划。"""
    skeleton = None
    if settings.record_skeleton:
        skeleton = RenderSkeleton(skeleton_engine(settings.pose_variant, settings.enable_hands))
    if not segment.multi:
        # 骨架成片可能很久：关窗时取消（保留原始录像），只转码的计划关窗后仍做完（V-4）
        return ProcessingPlan(single_segment_steps(skeleton=skeleton), finish_on_close=skeleton is None)
    intent = segment_intent(segment)
    auto_compare = settings.auto_compare if intent.auto_compare is None else intent.auto_compare
    steps = dual_segment_steps(
        record_skeleton=settings.record_skeleton,
        auto_compare=auto_compare,
        front_templates=settings.front_templates,
        side_templates=settings.side_templates,
        skeleton=skeleton,
    )
    exam = dict(intent.exam) if intent.exam is not None else None
    if exam is not None:
        steps = (ExamClipStep(), *steps)
    return ProcessingPlan(
        steps,
        result=ResultSpec(
            record_skeleton=settings.record_skeleton,
            auto_compare=auto_compare,
            front_template=settings.front_templates[0] if settings.front_templates else Path(""),
            side_template=settings.side_templates[0] if settings.side_templates else Path(""),
            exam=exam,
        ),
        close_budget_s=EXAM_CLOSE_BUDGET_S if exam is not None else 0.0,
    )
