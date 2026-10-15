# -*- coding: utf-8 -*-
"""规则的声明结构：``RuleSpec``（一条规则）、``RuleProfile``（一套规则目录）、判定上下文、聚合方式与结构化结论（设计文档 §7.2）。

唯一的规则目录 ``sanda_v5``（D-3）用 ``PhaseVerdict``：在 ``RuleContext`` 给出的相位窗口里取帧，以起始护位为基线逐帧测量，
单视角结论为 ``violation``/``pass``/``unable``，被拦截的项为 ``pending_rule``；练习正侧融合后的 ``candidate``/``not_observed``
由 ``evaluation/practice.py`` 写入检查项。``unable`` 带结构化原因 ``skip_reason``（后端缺点、有效帧不足、未找到阶段），
下游判断只读结构化字段（YOLO 迁移 S4 / Issue #11）。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RuleState = Literal["pass", "violation", "not_observed", "unable", "candidate", "pending_rule"]
SkipReason = Literal["missing_landmarks", "insufficient_valid_frames", "phase_not_found"]

RULE_STATE_PASS: RuleState = "pass"
RULE_STATE_VIOLATION: RuleState = "violation"
RULE_STATE_NOT_OBSERVED: RuleState = "not_observed"
RULE_STATE_UNABLE: RuleState = "unable"
RULE_STATE_CANDIDATE: RuleState = "candidate"
RULE_STATE_PENDING_RULE: RuleState = "pending_rule"

SKIP_MISSING_LANDMARKS: SkipReason = "missing_landmarks"
SKIP_INSUFFICIENT_VALID_FRAMES: SkipReason = "insufficient_valid_frames"
SKIP_PHASE_NOT_FOUND: SkipReason = "phase_not_found"

@dataclass(frozen=True)
class RuleContext:
    """一次判定的上下文：视角、动作、左右式、相位窗口与基线。

    ``phase_windows`` 以 ``"segment:phase"`` 为键给出序列帧下标；``baseline`` 是起始护位的帧下标；
    ``params`` 是阈值配置（``FeedbackConfig``）。
    """

    view: str
    action: str
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


@dataclass(frozen=True)
class RuleSpec:
    code: str
    title: str
    check: PhaseCheck
    required: tuple[int, ...]  # 所需关键点（BlazePose33），声明能力契约与缺点判定；须与 check 内的取点一致
    view: Literal["front", "side", "any"]
    scope: str  # 相位窗口 "segment:phase"
    aggregation: PhaseVerdict
    authorization: Literal["scored", "candidate", "blocked"] = "scored"
    blocked_reason: str = ""
    source: str = ""  # 依据：文档与行号


@dataclass(frozen=True)
class RuleProfile:
    name: str
    rules: tuple[RuleSpec, ...]


@dataclass(frozen=True)
class RuleOutcome:
    """一条规则在一路视角上的结论；``detail`` 只在 ``pending_rule`` 时写拦截原因。"""

    spec: RuleSpec
    state: RuleState
    skip_reason: SkipReason | None
    violation_ratio: float
    valid_frames: int
    total_frames: int
    detail: str = ""
    missing_landmarks: tuple[str, ...] = ()
    # 证据帧（序列帧下标，首/中/末代表帧）与测量值摘要（名称、单位、中位数、最小、最大）。
    evidence_frames: tuple[int, ...] = ()
    measurements: tuple[dict, ...] = ()
