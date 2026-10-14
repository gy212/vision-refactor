# -*- coding: utf-8 -*-
"""规则的声明结构：``RuleSpec``（一条规则）、``RuleProfile``（一套规则目录）、判定上下文、聚合方式与结构化结论（设计文档 §7.2）。

正式规则（``formal_v1``）用 ``RatioDeduction``：违规帧占比 ≥ 触发比例即扣一次分。结构化三态（YOLO 迁移 S4 /
Issue #11）：``state ∈ {evaluated, skipped}``，``skip_reason ∈ {missing_landmarks, low_confidence,
insufficient_valid_frames}``；中文 ``detail`` 只供界面显示，下游判断只读结构化字段。

学生练习（``practice_v4``，D-16）用 ``PhaseVerdict``：在 ``RuleContext`` 给出的相位窗口里取帧，以起始护位为基线逐帧测量，
单视角结论为 ``violation``/``pass``/``unable``，被拦截的项为 ``pending_rule``；正侧融合后的 ``candidate``/``not_observed``
由 ``evaluation/practice.py`` 写入检查项。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RuleState = Literal[
    "evaluated", "skipped",  # formal_v1
    "pass", "violation", "not_observed", "unable", "candidate", "pending_rule",  # practice_v4
]
SkipReason = Literal["missing_landmarks", "low_confidence", "insufficient_valid_frames", "phase_not_found"]

RULE_STATE_EVALUATED: RuleState = "evaluated"
RULE_STATE_SKIPPED: RuleState = "skipped"
RULE_STATE_PASS: RuleState = "pass"
RULE_STATE_VIOLATION: RuleState = "violation"
RULE_STATE_NOT_OBSERVED: RuleState = "not_observed"
RULE_STATE_UNABLE: RuleState = "unable"
RULE_STATE_CANDIDATE: RuleState = "candidate"
RULE_STATE_PENDING_RULE: RuleState = "pending_rule"

SKIP_MISSING_LANDMARKS: SkipReason = "missing_landmarks"
SKIP_LOW_CONFIDENCE: SkipReason = "low_confidence"
SKIP_INSUFFICIENT_VALID_FRAMES: SkipReason = "insufficient_valid_frames"
SKIP_PHASE_NOT_FOUND: SkipReason = "phase_not_found"

# 逐帧判定：输入 (T,33,4) 关键点与 (T,33) 有效掩码，返回 (违规帧, 参与判定的帧, 规则说明)。
FrameCheck = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, str]]


@dataclass(frozen=True)
class RuleContext:
    """一次判定的上下文：视角、动作范围、左右式（正式 profile 不使用），以及练习 profile 的相位窗口与基线。

    ``phase_windows`` 以 ``"segment:phase"`` 为键给出序列帧下标；``baseline`` 是起始护位的帧下标；
    ``params`` 是 profile 的阈值配置（练习为 ``FeedbackConfig``）。
    """

    view: str
    action_scope: str
    stance: str = ""
    phase_windows: Mapping[str, np.ndarray] = field(default_factory=dict)
    baseline: np.ndarray | None = None
    params: Any = None


# 相位判定：输入窗口内关键点与掩码、基线关键点与掩码、上下文，返回 (问题帧, 有效帧, {测量名: (逐帧值, 单位)})。
PhaseCheck = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, RuleContext],
    tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, str]]],
]


@dataclass(frozen=True)
class RatioDeduction:
    """违规帧占比 ≥ ``trigger_ratio`` 时扣 ``points`` 分（一次）。"""

    points: int
    trigger_ratio: float


@dataclass(frozen=True)
class PhaseVerdict:
    """相位窗口内的单视角判定。

    窗口短于 ``min_valid`` 帧为未找到阶段；有效帧少于 ``min_valid`` 或占比低于 ``valid_ratio`` 为 ``unable``；
    问题帧占有效帧比例 ≥ ``trigger_ratio`` 为 ``violation``。``absence=True`` 的项（"未做某动作"）
    只在全部有效帧都有问题时才算 ``violation``。
    """

    min_valid: int
    valid_ratio: float
    trigger_ratio: float
    absence: bool = False


Aggregation = RatioDeduction | PhaseVerdict


@dataclass(frozen=True)
class RuleSpec:
    code: str
    title: str
    check: FrameCheck | PhaseCheck  # RatioDeduction 用 FrameCheck，PhaseVerdict 用 PhaseCheck
    required: tuple[int, ...]  # 所需关键点（BlazePose33），声明能力契约与缺点判定；须与 check 内的取点一致
    view: Literal["front", "side", "any"]
    scope: str  # stance / punch，或练习的 "segment:phase"
    aggregation: Aggregation
    authorization: Literal["scored", "candidate", "blocked"] = "scored"
    blocked_reason: str = ""
    source: str = ""  # 依据：文档与行号


@dataclass(frozen=True)
class RuleProfile:
    name: str
    rules: tuple[RuleSpec, ...]


@dataclass(frozen=True)
class RuleOutcome:
    """一条规则的结论；``detail`` 已带"（合格）""（有效帧不足，未评估）"等后缀。"""

    spec: RuleSpec
    state: RuleState
    skip_reason: SkipReason | None
    deduction: int
    violation_ratio: float
    valid_frames: int
    total_frames: int
    detail: str
    missing_landmarks: tuple[str, ...] = ()
    # 练习 profile：证据帧（序列帧下标，首/中/末代表帧）与测量值摘要（名称、单位、中位数、最小、最大）。
    evidence_frames: tuple[int, ...] = ()
    measurements: tuple[dict, ...] = ()
