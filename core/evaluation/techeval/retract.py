# -*- coding: utf-8 -*-
"""收拳速度指标：侧面出拳事件的回收时长判定。"""
from __future__ import annotations

import numpy as np

from core.evaluation.techeval.contract import IndicatorResult, resolve_mask
from core.evaluation.techeval.events import detect_retract_events_side


def eval_retract_speed_side(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    ext_angle_thr: float = 160.0,
    retract_angle_thr: float = 45.0,
    ext_forward_dx: float = 0.02,
    retract_sec_thr: float = 0.2,
    fatal_slow_ratio: float = 0.75,
    min_punches: int = 4,
    max_window_sec: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "total_punches": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    events = detect_retract_events_side(
        landmarks,
        fps=fps,
        vis_thr=vis_thr,
        ext_angle_thr=ext_angle_thr,
        retract_angle_thr=retract_angle_thr,
        ext_forward_dx=ext_forward_dx,
        retract_sec_thr=retract_sec_thr,
        max_window_sec=max_window_sec,
        valid_mask=mask,
    )
    events = sorted(events, key=lambda d: int(d.get("start") or 0))

    total = int(len(events))
    slow_cnt = int(sum(1 for e in events if bool(e.get("slow"))))

    detail = {
        "total_punches": int(total),
        "slow_punches": int(slow_cnt),
        "min_punches": int(min_punches),
        "fatal_slow_ratio": float(fatal_slow_ratio),
        "retract_sec_thr": float(retract_sec_thr),
        "events": events,
    }

    if total < int(min_punches):
        detail["primary_cause"] = "出拳次数不足"
        return IndicatorResult(status="无法判定", reason="侧面：出拳次数不足", detail=detail)

    slow_ratio = float(slow_cnt / max(1, total))
    detail["slow_ratio"] = float(slow_ratio)

    if slow_ratio >= float(fatal_slow_ratio):
        detail["primary_cause"] = "回收超时"
        return IndicatorResult(status="不合格", reason="侧面：多数出拳回收超时", detail=detail)
    detail["primary_cause"] = "达标"
    return IndicatorResult(status="合格", reason="侧面：回收速度达标", detail=detail)
