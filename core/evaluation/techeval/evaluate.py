# -*- coding: utf-8 -*-
"""技术指标的视频级编排：提取姿态与视角分数、切正侧段、逐项评估并补齐契约，产出 ``TechEvalResult`` 或 JSON 字典。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from core.evaluation.segmentation import classify_single_view, split_front_side
from core.evaluation.techeval.cog import eval_cog_front, eval_cog_side_prefer_punch_windows
from core.evaluation.techeval.com import eval_cog_com
from core.evaluation.techeval.contract import (
    COG_COM_REQUIRED_INDICES,
    COG_REQUIRED_INDICES,
    FORCE_REQUIRED_INDICES,
    RETRACT_REQUIRED_INDICES,
    WRIST_REQUIRED_INDICES,
    IndicatorResult,
    TechEvalResult,
    attach_contract,
    resolve_mask,
)
from core.evaluation.techeval.force_sequence import eval_force_sequence
from core.evaluation.techeval.retract import eval_retract_speed_side
from core.evaluation.techeval.wrist import eval_wrist_angle
from core.foundation import to_jsonable
from core.pose import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY, GapPolicy, formal_end_frame, formal_sequence


def extract_pose_and_view_scores(
    video_path: Path,
    *,
    pose_variant: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """``FORMAL`` 从第 0 帧顺序跟踪到 ``end_frame``，取 ``[start, end]`` 的 ``raw(ZERO)`` 与 ``view_scores(NAN)``。"""
    video = Path(video_path)
    start_i = max(0, int(start_frame) if start_frame is not None else 0)
    end_i = max(start_i, formal_end_frame(video, end_frame))
    seq = formal_sequence(video, pose_variant, end_i)
    lms = seq.raw(GapPolicy.ZERO)[start_i : end_i + 1].copy()
    vs = seq.view_scores(GapPolicy.NAN)[start_i : end_i + 1].copy()
    if lms.shape[0] == 0:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    m = seq.meta
    meta = {
        "video": str(video_path),
        "fps": float(m.fps),
        "frame_count": int(m.frame_count),
        "width": int(m.width),
        "height": int(m.height),
        "pose_variant": str(pose_variant),
        "start_frame": int(start_i),
        "end_frame": int(start_i + lms.shape[0] - 1),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        # valid_mask 契约（YOLO 迁移 Issue #5）：声明有效性策略与阈值，便于下游区分
        # 可信/待标定。本入口的 meta 会被序列化进 JSON 报告，故只落地标量字段；
        # valid_mask 由下游按同一 `valid_conf_thr` 现场推导（逐位等价）。
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
    }
    return lms, vs, meta


def split_front_side_segments(view_scores: np.ndarray, fps: float, **thresholds) -> tuple:
    """按逐帧视角分数切出 ``(正面段, 侧面段, view_mode)``，保守口径（``segmentation.split_front_side``）。"""
    return split_front_side(view_scores, fps, mode="conservative", **thresholds)


def _segments(n: int, view_scores: np.ndarray, fps: float, view_hint: str):
    """``(正面段, 侧面段, view_mode)``：指定视角时整段归该视角，否则自动切分，单视角时再按分数判归属。"""
    if view_hint in ("front", "side"):
        if view_hint == "front":
            front_seg, side_seg = (0, int(n - 1)), None
        else:
            front_seg, side_seg = None, (0, int(n - 1))
        return front_seg, side_seg, "single"
    front_seg, side_seg, view_mode = split_front_side_segments(view_scores, fps)
    if view_mode == "single":
        single_view = classify_single_view(view_scores, side_thr=0.9, front_thr=0.95)
        if single_view == "side":
            front_seg, side_seg = None, (0, int(n - 1))
        elif single_view == "front":
            front_seg, side_seg = (0, int(n - 1)), None
    return front_seg, side_seg, view_mode


def _slice(arr: np.ndarray, seg, empty_shape: tuple, dtype) -> np.ndarray:
    return arr[slice(seg[0], seg[1] + 1)] if seg is not None else np.zeros(empty_shape, dtype=dtype)


def _indicators(landmarks, full_mask, front_seg, side_seg, fps: float, stance: str) -> dict[str, Any]:
    """各项指标（未补契约）；缺正面或侧面段的项给出"无法判定"。"""
    front_lm = _slice(landmarks, front_seg, (0, 33, 4), np.float32)
    side_lm = _slice(landmarks, side_seg, (0, 33, 4), np.float32)
    front_mask = _slice(full_mask, front_seg, (0, 33), bool)
    side_mask = _slice(full_mask, side_seg, (0, 33), bool)
    # 重心评估：方案1+2（侧面优先）+ 方案3（CoM分段质心）
    if side_seg is not None:
        cog_side, cog_com = eval_cog_side_prefer_punch_windows(side_lm, fps=fps, use_com=True, valid_mask=side_mask)
    else:
        cog_side = IndicatorResult(status="无法判定", reason="侧面段缺失")
        # 无侧面段时，全段使用CoM评估作为参考
        cog_com = eval_cog_com(landmarks, fps=fps, valid_mask=full_mask)
    if front_seg is not None:
        cog_front = eval_cog_front(front_lm, fps=fps, stance=str(stance), valid_mask=front_mask)
    else:
        cog_front = IndicatorResult(status="无法判定", reason="正面段缺失")
    retract = eval_retract_speed_side(side_lm, fps=fps, valid_mask=side_mask) if side_seg is not None else IndicatorResult(status="无法判定", reason="侧面段缺失")
    if side_seg is not None:
        wrist_src, wrist_mask = side_lm, side_mask
    else:
        wrist_src, wrist_mask = front_lm, front_mask
    wrist = eval_wrist_angle(wrist_src, fps=fps, valid_mask=wrist_mask) if wrist_src.size != 0 else IndicatorResult(status="无法判定", reason="无有效帧")
    force = eval_force_sequence(front_lm, side_lm, fps=fps, stance=str(stance),
                                front_valid_mask=front_mask, side_valid_mask=side_mask)
    return {"cog_side": cog_side, "cog_front": cog_front, "cog_final": cog_side if cog_side.status != "无法判定" else cog_front,
            "cog_com": cog_com, "retract": retract, "wrist": wrist, "force": force,
            "front_mask": front_mask, "side_mask": side_mask, "wrist_mask": wrist_mask, "wrist_empty": wrist_src.size == 0}


def _with_contracts(r: dict[str, Any], full_mask, front_seg, side_seg) -> dict[str, Any]:
    """装配层统一补齐结构化契约字段（YOLO 迁移 S4 / Issue #11），不改各指标内部判定。"""
    front_mask, side_mask = r["front_mask"], r["side_mask"]
    masks = [m for m in (front_mask, side_mask) if m is not None and m.shape[0] > 0]
    force_mask = np.concatenate(masks, axis=0) if masks else None
    cog_side = attach_contract(r["cog_side"], COG_REQUIRED_INDICES, mask=side_mask if side_seg is not None else None)
    out = {
        "cog_side": cog_side,
        "cog_front": attach_contract(r["cog_front"], COG_REQUIRED_INDICES, mask=front_mask if front_seg is not None else None),
        # cog_final 跟随被采用的视角口径。
        "cog_final": attach_contract(
            r["cog_final"], COG_REQUIRED_INDICES,
            mask=(side_mask if (side_seg is not None and cog_side.status != "无法判定") else (front_mask if front_seg is not None else None)),
        ),
        "cog_com": None if r["cog_com"] is None else attach_contract(
            r["cog_com"], COG_COM_REQUIRED_INDICES, mask=(side_mask if side_seg is not None else full_mask)),
        "retract": attach_contract(r["retract"], RETRACT_REQUIRED_INDICES, mask=side_mask if side_seg is not None else None),
        "wrist": attach_contract(r["wrist"], WRIST_REQUIRED_INDICES, mask=(None if r["wrist_empty"] else r["wrist_mask"])),
        "force": attach_contract(r["force"], FORCE_REQUIRED_INDICES, mask=force_mask),
    }
    return out


def _strip(r: IndicatorResult | None) -> IndicatorResult | None:
    # 用 replace 仅清空 detail，保留 Issue #11 结构化契约字段。
    return None if r is None else replace(r, detail=None)


def evaluate_arrays(
    landmarks: np.ndarray,
    view_scores: np.ndarray,
    meta: dict[str, Any],
    *,
    pose_variant: str,
    stance: str,
    view_hint: str,
    keep_detail: bool,
    valid_mask: np.ndarray | None = None,
) -> TechEvalResult:
    """已提取的关键点与视角分数上的全部技术指标。"""
    fps = float(meta.get("fps") or 30.0)
    view_hint = (view_hint or "auto").lower()
    # valid_mask 契约（YOLO 迁移 Issue #5）：在视频级入口解析一次，按段切片下发。
    # `None` 时按 meta 的 `valid_conf_thr`（缺省 0.5）现场推导，逐位等价旧行为。
    mask_thr = float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR)
    full_mask = resolve_mask(landmarks, valid_mask, mask_thr)
    front_seg, side_seg, view_mode = _segments(int(landmarks.shape[0]), view_scores, fps, view_hint)
    r = _with_contracts(_indicators(landmarks, full_mask, front_seg, side_seg, fps, stance), full_mask, front_seg, side_seg)
    if not keep_detail:
        r = {k: _strip(v) for k, v in r.items()}
    return TechEvalResult(
        video_path=str(meta.get("video") or ""),
        pose_variant=str(pose_variant),
        fps=float(fps),
        view_mode=str(view_mode),
        front_segment=None if front_seg is None else (int(front_seg[0]), int(front_seg[1])),
        side_segment=None if side_seg is None else (int(side_seg[0]), int(side_seg[1])),
        cog_final=r["cog_final"],
        cog_side=r["cog_side"],
        cog_front=r["cog_front"],
        cog_com=r["cog_com"],
        retract_speed=r["retract"],
        force_sequence=r["force"],
        wrist_angle=r["wrist"],
    )


def evaluate_video_assets(
    video_path: Path,
    *,
    pose_variant: str = "full",
    stance: str = "left",
    view_hint: str = "auto",
    valid_mask: np.ndarray | None = None,
) -> tuple[TechEvalResult, np.ndarray, np.ndarray, dict[str, Any]]:
    landmarks, view_scores, meta = extract_pose_and_view_scores(video_path, pose_variant=str(pose_variant))
    res = evaluate_arrays(
        landmarks,
        view_scores,
        meta,
        pose_variant=str(pose_variant),
        stance=str(stance),
        view_hint=str(view_hint),
        keep_detail=True,
        valid_mask=valid_mask,
    )
    return res, landmarks, view_scores, meta


def evaluate_video_detail(
    video_path: Path,
    *,
    pose_variant: str = "full",
    stance: str = "left",
    view_hint: str = "auto",
    valid_mask: np.ndarray | None = None,
) -> TechEvalResult:
    res, _lm, _vs, _meta = evaluate_video_assets(
        video_path, pose_variant=pose_variant, stance=stance, view_hint=view_hint, valid_mask=valid_mask
    )
    return res


def evaluate_video(
    video_path: Path,
    *,
    pose_variant: str = "full",
    stance: str = "left",
    view_hint: str = "auto",
    valid_mask: np.ndarray | None = None,
) -> TechEvalResult:
    detail = evaluate_video_detail(
        video_path, pose_variant=pose_variant, stance=stance, view_hint=view_hint, valid_mask=valid_mask
    )
    return TechEvalResult(
        video_path=str(video_path),
        pose_variant=str(pose_variant),
        fps=float(detail.fps),
        view_mode=str(detail.view_mode),
        front_segment=detail.front_segment,
        side_segment=detail.side_segment,
        cog_final=_strip(detail.cog_final),
        cog_side=_strip(detail.cog_side),
        cog_front=_strip(detail.cog_front),
        cog_com=_strip(detail.cog_com),
        retract_speed=_strip(detail.retract_speed),
        force_sequence=_strip(detail.force_sequence),
        wrist_angle=_strip(detail.wrist_angle),
    )


def evaluate_video_full(
    video_path: Path,
    *,
    pose_variant: str = "full",
    stance: str = "left",
    view_hint: str = "auto",
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    res, _landmarks, _view_scores, meta = evaluate_video_assets(
        video_path, pose_variant=pose_variant, stance=stance, view_hint=view_hint, valid_mask=valid_mask
    )
    return {
        "video_path": str(video_path),
        "pose_variant": str(pose_variant),
        "fps": float(res.fps),
        "view_mode": str(res.view_mode),
        "front_segment": None if res.front_segment is None else [int(res.front_segment[0]), int(res.front_segment[1])],
        "side_segment": None if res.side_segment is None else [int(res.side_segment[0]), int(res.side_segment[1])],
        "meta": meta,
        "cog_side": to_jsonable(res.cog_side),
        "cog_front": to_jsonable(res.cog_front),
        "cog_final": to_jsonable(res.cog_final),
        "cog_com": to_jsonable(res.cog_com),
        "retract_speed": to_jsonable(res.retract_speed),
        "force_sequence": to_jsonable(res.force_sequence),
        "wrist_angle": to_jsonable(res.wrist_angle),
    }
