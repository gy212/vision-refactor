# -*- coding: utf-8 -*-
"""腕角指标：出拳伸展时拳面与前臂的夹角判定。"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.evaluation.techeval.contract import IndicatorResult, all_valid, resolve_mask
from core.evaluation.techeval.events import detect_extension_events
from core.foundation import angle_deg, landmark_xy
from core.pose.landmarks import L_ELBOW, L_INDEX, L_PINKY, L_WRIST, R_ELBOW, R_INDEX, R_PINKY, R_WRIST


def eval_wrist_angle(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    ext_angle_thr: float = 160.0,
    wrist_align_thr: float = 175.0,
    wrist_ok_ratio: float = 0.7,
    min_events: int = 2,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "events_total": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    events = detect_extension_events(landmarks, fps=fps, vis_thr=vis_thr, ext_angle_thr=ext_angle_thr, valid_mask=mask)
    if len(events) < int(min_events):
        detail = {
            "primary_cause": "出拳次数不足",
            "events": events,
            "events_total": int(len(events)),
        }
        return IndicatorResult(status="无法判定", reason="拳面角度：出拳次数不足", detail=detail)

    ok = 0
    valid = 0
    per_event: list[dict[str, Any]] = []
    n = int(landmarks.shape[0])
    for e in events:
        arm = str(e.get("arm") or "")
        peak = int(e.get("peak") or e.get("start") or 0)
        s = max(0, peak - 2)
        t = min(n - 1, peak + 2)
        angles: list[float] = []
        for i in range(s, t + 1):
            lm = landmarks[i]
            if arm.upper() == "L":
                el, wr, idx, pk = L_ELBOW, L_WRIST, L_INDEX, L_PINKY
            else:
                el, wr, idx, pk = R_ELBOW, R_WRIST, R_INDEX, R_PINKY
            if not all_valid(mask[i], (el, wr, idx, pk)):
                continue
            mid = (landmark_xy(lm, idx) + landmark_xy(lm, pk)) * 0.5
            a = angle_deg(landmark_xy(lm, el), landmark_xy(lm, wr), mid)
            if np.isfinite(a):
                angles.append(float(a))
        if not angles:
            per_event.append({"arm": arm, "peak": int(peak), "ok": None, "angle_deg": None})
            continue
        ang = float(np.median(np.asarray(angles, dtype=np.float32)))
        is_ok = bool(ang >= float(wrist_align_thr))
        valid += 1
        ok += 1 if is_ok else 0
        per_event.append({"arm": arm, "peak": int(peak), "ok": bool(is_ok), "angle_deg": float(ang)})

    detail = {
        "events_total": int(len(events)),
        "events_valid": int(valid),
        "events_ok": int(ok),
        "wrist_align_thr": float(wrist_align_thr),
        "wrist_ok_ratio": float(wrist_ok_ratio),
        "events": per_event,
    }

    if valid < int(min_events):
        detail["primary_cause"] = "关键点不可见"
        return IndicatorResult(status="无法判定", reason="拳面角度：关键点不可见", detail=detail)

    ratio = float(ok / max(1, valid))
    detail["ok_ratio"] = float(ratio)
    if ratio >= float(wrist_ok_ratio):
        detail["primary_cause"] = "达标"
        return IndicatorResult(status="合格", reason="拳面角度：手腕对齐达标", detail=detail)
    detail["primary_cause"] = "折腕"
    return IndicatorResult(status="不合格", reason="拳面角度：存在折腕/塌腕", detail=detail)
