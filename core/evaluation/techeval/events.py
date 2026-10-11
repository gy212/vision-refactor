# -*- coding: utf-8 -*-
"""侧面出拳事件检测：伸展—回收事件、伸展事件与区间合并/取子集，供回收速度、重心、腕角与发力顺序共用。"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.evaluation.body_sides import facing_direction
from core.evaluation.techeval.contract import all_valid, resolve_mask
from core.foundation import angle_deg, landmark_xy
from core.pose.landmarks import L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER, R_WRIST


def detect_retract_events_side(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float,
    ext_angle_thr: float,
    retract_angle_thr: float,
    ext_forward_dx: float,
    retract_sec_thr: float,
    max_window_sec: float,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mask = resolve_mask(landmarks, valid_mask, vis_thr)
    max_window_frames = int(max(1, round(float(max_window_sec) * float(fps))))

    def arm_series(arm: str) -> tuple[np.ndarray, np.ndarray]:
        if arm.upper() == "L":
            sh, el, wr = L_SHOULDER, L_ELBOW, L_WRIST
        else:
            sh, el, wr = R_SHOULDER, R_ELBOW, R_WRIST

        ang = np.full((landmarks.shape[0],), np.nan, dtype=np.float32)
        dx = np.full_like(ang, np.nan)

        for i, lm in enumerate(landmarks):
            mask_row = mask[i]
            d = facing_direction(lm, mask_row)
            if d is None:
                continue
            if not all_valid(mask_row, (sh, el, wr)):
                continue
            a = angle_deg(landmark_xy(lm, sh), landmark_xy(lm, el), landmark_xy(lm, wr))
            ang[i] = float(a)
            dx[i] = (float(landmark_xy(lm, wr)[0]) - float(landmark_xy(lm, sh)[0])) * float(d)

        return ang, dx

    def detect_events(arm: str) -> list[dict[str, Any]]:
        ang, dx = arm_series(arm)

        events: list[dict[str, Any]] = []
        state = "idle"
        ext_start: int | None = None
        ext_hold = 0
        ret_hold = 0

        last_i = int(ang.size - 1)
        for i in range(int(ang.size)):
            a = float(ang[i])
            if not np.isfinite(a):
                continue

            forward_ok = np.isfinite(dx[i]) and (float(dx[i]) >= float(ext_forward_dx))
            ext_ok = (a >= float(ext_angle_thr)) and bool(forward_ok)
            ret_ok = (a <= float(retract_angle_thr))

            if state == "idle":
                if ext_ok:
                    ext_hold += 1
                    if ext_hold >= 2:
                        ext_start = int(i - 1)
                        state = "extended"
                        ret_hold = 0
                else:
                    ext_hold = 0
                continue

            if state == "extended":
                if ext_start is not None and (i - ext_start) > max_window_frames:
                    events.append({"arm": arm, "start": int(ext_start), "end": None, "duration_sec": None, "slow": True})
                    state = "cooldown"
                    ext_start = None
                    ext_hold = 0
                    ret_hold = 0
                    continue

                if ret_ok:
                    ret_hold += 1
                    if ret_hold >= 2 and ext_start is not None:
                        end = int(i - 1)
                        dur = float((end - ext_start) / max(1e-6, float(fps)))
                        slow = bool(dur > float(retract_sec_thr))
                        events.append({"arm": arm, "start": int(ext_start), "end": int(end), "duration_sec": dur, "slow": slow})
                        state = "cooldown"
                        ext_start = None
                        ext_hold = 0
                        ret_hold = 0
                else:
                    ret_hold = 0
                continue

            if a < float(ext_angle_thr) - 20.0:
                state = "idle"

        if state == "extended" and ext_start is not None:
            end = int(last_i)
            dur = float((end - int(ext_start)) / max(1e-6, float(fps)))
            slow = bool(dur > float(retract_sec_thr))
            events.append({"arm": arm, "start": int(ext_start), "end": int(end), "duration_sec": dur, "slow": slow})

        return events

    return sorted(detect_events("L") + detect_events("R"), key=lambda d: int(d.get("start") or 0))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(((int(s), int(e)) for s, e in intervals), key=lambda t: t[0])
    out: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe + 1:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def subset_by_intervals(
    landmarks: np.ndarray,
    intervals: list[tuple[int, int]],
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray | None]:
    n = int(landmarks.shape[0])
    clipped: list[tuple[int, int]] = []
    for s, e in intervals:
        s2 = max(0, int(s))
        e2 = min(n - 1, int(e))
        if e2 >= s2:
            clipped.append((s2, e2))
    clipped = merge_intervals(clipped)
    if not clipped:
        empty_mask = None if valid_mask is None else np.zeros((0, 33), dtype=bool)
        return np.zeros((0, 33, 4), dtype=np.float32), [], empty_mask
    idxs: list[int] = []
    for s, e in clipped:
        idxs.extend(list(range(int(s), int(e) + 1)))
    sel = np.asarray(idxs, dtype=np.int32)
    sub_mask = None if valid_mask is None else valid_mask[sel]
    return landmarks[sel], clipped, sub_mask


def detect_extension_events(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    ext_angle_thr: float = 160.0,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mask = resolve_mask(landmarks, valid_mask, vis_thr)

    def series(arm: str) -> np.ndarray:
        if arm.upper() == "L":
            sh, el, wr = L_SHOULDER, L_ELBOW, L_WRIST
        else:
            sh, el, wr = R_SHOULDER, R_ELBOW, R_WRIST
        ang = np.full((landmarks.shape[0],), np.nan, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            if not all_valid(mask[i], (sh, el, wr)):
                continue
            ang[i] = float(angle_deg(landmark_xy(lm, sh), landmark_xy(lm, el), landmark_xy(lm, wr)))
        return ang

    def detect(arm: str) -> list[dict[str, Any]]:
        ang = series(arm)
        out: list[dict[str, Any]] = []
        state = "idle"
        hold = 0
        start_i: int | None = None
        peak_i: int | None = None
        peak_a = -1.0

        last_i = int(ang.size - 1)
        for i in range(int(ang.size)):
            a = float(ang[i])
            if not np.isfinite(a):
                continue

            ext_ok = a >= float(ext_angle_thr)
            if state == "idle":
                if ext_ok:
                    hold += 1
                    if hold >= 2:
                        start_i = int(i - 1)
                        peak_i = int(i)
                        peak_a = float(a)
                        state = "extended"
                else:
                    hold = 0
                continue

            if state == "extended":
                if a >= peak_a:
                    peak_a = float(a)
                    peak_i = int(i)
                if a < float(ext_angle_thr) - 20.0:
                    if start_i is not None and peak_i is not None:
                        out.append({"arm": arm, "start": int(start_i), "peak": int(peak_i), "end": int(i)})
                    state = "cooldown"
                    hold = 0
                    start_i = None
                    peak_i = None
                    peak_a = -1.0
                continue

            if a < float(ext_angle_thr) - 30.0:
                state = "idle"

        if state == "extended" and start_i is not None and peak_i is not None:
            out.append({"arm": arm, "start": int(start_i), "peak": int(peak_i), "end": int(last_i)})

        return out

    return sorted(detect("L") + detect("R"), key=lambda d: int(d.get("start") or 0))
