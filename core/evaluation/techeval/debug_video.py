# -*- coding: utf-8 -*-
"""技术指标调试视频导出：骨架、各指标状态与出拳窗口叠加到原视频。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.evaluation.techeval.contract import TechEvalResult
from core.evaluation.techeval.evaluate import evaluate_video_assets
from core.evaluation.techeval.events import merge_intervals
from core.media import DecodeOptions, open_video, open_video_writer
from core.pose import BODY_CONNECTIONS, draw_skeleton, draw_status_lines


def status_tag(s: str) -> str:
    if s == "合格":
        return "PASS"
    if s == "不合格":
        return "FAIL"
    return "NA"


def export_debug_video(
    video_path: Path,
    out_path: Path,
    *,
    pose_variant: str = "full",
    stance: str = "left",
    view_hint: str = "auto",
    res: TechEvalResult | None = None,
    landmarks: np.ndarray | None = None,
    view_scores: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    if res is None or landmarks is None or view_scores is None or meta is None:
        res, landmarks, view_scores, meta = evaluate_video_assets(video_path, pose_variant=pose_variant, stance=stance, view_hint=view_hint)
    fps = float((meta or {}).get("fps") or res.fps or 30.0)

    source = open_video(video_path, DecodeOptions())
    w, h = source.meta.width, source.meta.height
    vw, actual_path, _codec = open_video_writer(Path(out_path), fps=fps, size=(w, h))

    windows: list[tuple[int, int]] = []
    if res.retract_speed.detail is not None:
        for e in (res.retract_speed.detail.get("events") or []):
            s = e.get("start")
            ed = e.get("end")
            if s is None:
                continue
            if ed is None:
                ed = int(s) + int(round(1.0 * float(fps)))
            windows.append((int(s) - int(round(0.2 * float(fps))), int(ed) + int(round(0.1 * float(fps)))))
    windows = merge_intervals(windows)

    def in_window(i: int) -> bool:
        for s, e in windows:
            if int(s) <= int(i) <= int(e):
                return True
        return False

    for item in source:
        frame, i = item.bgr, item.index
        if i < int(landmarks.shape[0]):
            draw_skeleton(frame, landmarks[i], connections=BODY_CONNECTIONS)
        lines = [
            f"COG:{status_tag(res.cog_final.status)}  Retract:{status_tag(res.retract_speed.status)}",
            f"ForceSeq:{status_tag(res.force_sequence.status)}  Wrist:{status_tag(res.wrist_angle.status)}",
        ]
        # 显示CoM评估结果（方案3）
        if res.cog_com is not None:
            lines.append(f"CoM(方案3):{status_tag(res.cog_com.status)}")
        if np.isfinite(view_scores[i]) if i < int(view_scores.size) else False:
            lines.append(f"ViewScore:{float(view_scores[i]):.2f}")
        if in_window(i):
            lines.append("PUNCH_WINDOW")
        draw_status_lines(frame, lines)
        vw.write(frame)

    vw.release()
    source.close()
    return actual_path
