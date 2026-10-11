# -*- coding: utf-8 -*-
"""规则的声明结构：``RuleSpec``（一条规则）、``RuleProfile``（一套规则目录）、聚合方式与结构化结论（设计文档 §7.2）。

正式规则（``formal_v1``）用 ``RatioDeduction``：违规帧占比 ≥ 触发比例即扣一次分。结构化三态（YOLO 迁移 S4 /
Issue #11）：``state ∈ {evaluated, skipped}``，``skip_reason ∈ {missing_landmarks, low_confidence,
insufficient_valid_frames}``；中文 ``detail`` 只供界面显示，下游判断只读结构化字段。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

RuleState = Literal["evaluated", "skipped"]
SkipReason = Literal["missing_landmarks", "low_confidence", "insufficient_valid_frames"]

RULE_STATE_EVALUATED: RuleState = "evaluated"
RULE_STATE_SKIPPED: RuleState = "skipped"

SKIP_MISSING_LANDMARKS: SkipReason = "missing_landmarks"
SKIP_LOW_CONFIDENCE: SkipReason = "low_confidence"
SKIP_INSUFFICIENT_VALID_FRAMES: SkipReason = "insufficient_valid_frames"

# 逐帧判定：输入 (T,33,4) 关键点与 (T,33) 有效掩码，返回 (违规帧, 参与判定的帧, 规则说明)。
FrameCheck = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, str]]


@dataclass(frozen=True)
class RatioDeduction:
    """违规帧占比 ≥ ``trigger_ratio`` 时扣 ``points`` 分（一次）。"""

    points: int
    trigger_ratio: float


@dataclass(frozen=True)
class RuleSpec:
    code: str
    title: str
    check: FrameCheck
    required: tuple[int, ...]  # 所需关键点（BlazePose33），声明能力契约与缺点判定；须与 check 内的取点一致
    view: Literal["front", "side", "any"]
    scope: str  # stance / punch
    aggregation: RatioDeduction


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
