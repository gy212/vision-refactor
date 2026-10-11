# -*- coding: utf-8 -*-
"""DTW 成本 → 相似度分数的唯一换算：``score = baseline / (baseline + avg_cost)``，baseline 只来自布局注册表（设计文档 §7.6）。

``avg_cost = 0`` 时为 1.0，``avg_cost = baseline`` 时为 0.5。pose33_v3 的 baseline 3.0 于 2026-07-11 标定
（同人自复现 avg_cost≈0.333 锚到约 0.90 分）；body_core_v1 的 1.2826 仍是 ``calibration_status=unvalidated``。
"""
from __future__ import annotations

from core.pose.layouts import POSE33_V3, FeatureLayoutSpec, get_layout


def baseline_for(layout: str | FeatureLayoutSpec = POSE33_V3) -> float:
    """布局登记的 baseline；未登记 baseline 的布局直接报错，不回退到别的布局的值。"""
    spec = get_layout(layout) if isinstance(layout, str) else layout
    if spec.default_baseline is None:
        raise ValueError(f"布局 {spec.name} 没有登记 baseline")
    return float(spec.default_baseline)


POSE33_BASELINE = baseline_for(POSE33_V3)


def avg_cost_of(cost: float, query_frames: int) -> float:
    """按查询帧数平均的 DTW 成本（查询为空时按 1 帧算）。"""
    return cost / max(1, int(query_frames))


def score_from_avg_cost(avg_cost: float, baseline: float) -> float:
    return float(baseline / (baseline + avg_cost))
