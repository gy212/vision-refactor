# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from core.pose_features import (
    DEFAULT_VALID_CONF_THR,
    MEDIAPIPE_VALIDITY_POLICY,
    derive_valid_mask,
    landmark_names,
    landmarks_missing_for_capabilities,
    pose_view_score,
)
from core.paths import models_dir
from core.vision_pipeline import MediaPipePipeline, PipelineConfig

Status = Literal["合格", "不合格", "无法判定"]

# 后端标识（YOLO 迁移 S4 / Issue #11）。本期 #10 结论为「仅预览」，tech_eval 主链路
# 仍是 MediaPipe；backend 字段让结果诚实声明数据来源，为后续 YOLO 路径预留。
BACKEND_MEDIAPIPE = "mediapipe"


# BlazePose 33 landmark indices
NOSE = 0
L_EAR = 7
R_EAR = 8
MOUTH_L = 9
MOUTH_R = 10
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_PINKY = 17
R_PINKY = 18
L_INDEX = 19
R_INDEX = 20
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28
L_HEEL = 29
R_HEEL = 30
L_FOOT_INDEX = 31
R_FOOT_INDEX = 32


@dataclass(frozen=True)
class IndicatorResult:
    status: Status
    reason: str
    detail: dict[str, Any] | None = None
    # YOLO 迁移 S4 / Issue #11：结构化契约字段。
    # - required_landmarks：该指标依赖的 BlazePose33 关键点名（声明）。
    # - missing_landmarks：运行时缺失/后端不支持的关键点名。
    # - backend：数据来源后端（本期主链路恒为 mediapipe）。
    # 这些字段在指标计算内默认空，由 _evaluate_from_arrays 装配层统一补齐，
    # 既不改动各 eval_* 内部判定逻辑（golden 不漂移），又让结果可被下游稳定解析。
    required_landmarks: tuple[str, ...] = ()
    missing_landmarks: tuple[str, ...] = ()
    backend: str = BACKEND_MEDIAPIPE


@dataclass(frozen=True)
class TechEvalResult:
    video_path: str
    pose_variant: str
    fps: float
    view_mode: str  # mixed|single
    front_segment: tuple[int, int] | None
    side_segment: tuple[int, int] | None

    cog_final: IndicatorResult
    cog_side: IndicatorResult
    cog_front: IndicatorResult
    cog_com: IndicatorResult | None  # 方案3：分段质心评估结果
    retract_speed: IndicatorResult
    force_sequence: IndicatorResult
    wrist_angle: IndicatorResult


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def _lm_xy(lm: np.ndarray, idx: int) -> np.ndarray:
    return lm[idx, :2].astype(np.float32)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (2D points)."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _valid(mask_row: np.ndarray, idxs: tuple[int, ...] | list[int]) -> bool:
    """该帧给定关键点是否全部可用。

    读取由 ``derive_valid_mask`` 产出的单帧 ``(33,)`` bool 切片，不再现场比较
    ``lm[idx, 3] >= thr``（阈值只存在于 ``derive_valid_mask``，YOLO 迁移 Issue #5）。
    """
    return all(bool(mask_row[i]) for i in idxs)


def _resolve_mask(landmarks: np.ndarray, valid_mask: np.ndarray | None, vis_thr: float) -> np.ndarray:
    """统一解析 ``valid_mask``。

    ``None`` 时由集中式 ``derive_valid_mask`` 按 ``vis_thr`` 现场推导，等价旧式
    ``visibility >= vis_thr``，保证旧调用方（不传 mask）行为逐位不变。
    """
    if valid_mask is None:
        return derive_valid_mask(landmarks, vis_thr)
    return np.asarray(valid_mask, dtype=bool)


# ---------------------------------------------------------------------------
# 指标关键点契约声明（YOLO 迁移 S4 / Issue #11）
# ---------------------------------------------------------------------------
# 每个技术指标声明其依赖的 BlazePose33 关键点索引。装配层据此补齐
# required_landmarks（声明）与 missing_landmarks（运行时/后端缺失），让结果诚实表达
# “依赖什么、缺了什么”，下游无需解析中文 reason。
COG_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
COG_COM_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_PINKY, R_PINKY, L_INDEX, R_INDEX,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
RETRACT_REQUIRED_INDICES: tuple[int, ...] = (
    NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
)
FORCE_REQUIRED_INDICES: tuple[int, ...] = (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_HEEL, R_HEEL, L_FOOT_INDEX, R_FOOT_INDEX,
)
WRIST_REQUIRED_INDICES: tuple[int, ...] = (
    L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_INDEX, R_INDEX, L_PINKY, R_PINKY,
)


def _runtime_missing_landmarks(indices: tuple[int, ...], mask: np.ndarray | None) -> tuple[str, ...]:
    """返回 ``indices`` 中在整段 ``mask`` 内从未有效的关键点名（运行时缺失）。

    “从未有效”= 该点在所有帧的 valid_mask 都为 False（被遮挡/低置信/段缺失），据此
    诚实声明该指标本次缺了哪些点。空段（无帧）视为全部缺失。
    """
    if mask is None:
        return landmark_names(indices)
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or arr.shape[0] == 0:
        return landmark_names(indices)
    ever_valid = np.any(arr, axis=0)  # (33,)
    missing_idx = [int(i) for i in indices if not bool(ever_valid[int(i)])]
    return landmark_names(missing_idx)


def _attach_contract(
    result: "IndicatorResult",
    required_indices: tuple[int, ...],
    *,
    mask: np.ndarray | None,
    supported_capabilities=None,
    backend: str = BACKEND_MEDIAPIPE,
) -> "IndicatorResult":
    """给指标结果补齐 Issue #11 结构化契约字段（不改 status/reason/detail 判定）。

    missing_landmarks = 运行时从未有效的点 ∪ 后端结构性不支持的点（按能力分组）。
    本期 MediaPipe 路径 ``supported_capabilities=None``（全部支持），故只反映运行时缺失。
    """
    runtime_missing = _runtime_missing_landmarks(required_indices, mask)
    backend_missing = landmarks_missing_for_capabilities(required_indices, supported_capabilities)
    # 合并去重，保持声明顺序。
    missing: list[str] = list(runtime_missing)
    for n in backend_missing:
        if n not in missing:
            missing.append(n)
    return replace(
        result,
        required_landmarks=landmark_names(required_indices),
        missing_landmarks=tuple(missing),
        backend=str(backend),
    )


def extract_pose_and_view_scores(
    video_path: Path,
    *,
    pose_variant: str,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    start_i = int(start_frame) if start_frame is not None else 0
    end_i = int(end_frame) if end_frame is not None else (n_frames - 1 if n_frames > 0 else 10**9)
    start_i = max(0, start_i)
    end_i = max(start_i, end_i)

    models_dir_path = models_dir()
    pipe = MediaPipePipeline(
        models_dir=models_dir_path,
        cfg=PipelineConfig(pose_variant=pose_variant, running_mode="video", enable_hands=False),
    )

    lms: list[np.ndarray] = []
    vs: list[float] = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = int(i * 1000.0 / fps)
        pose_landmarks, _hands = pipe.infer(frame, timestamp_ms=ts)

        if start_i <= i <= end_i:
            if pose_landmarks is None:
                arr = np.zeros((33, 4), dtype=np.float32)
                lms.append(arr)
                vs.append(float("nan"))
            else:
                arr = np.zeros((33, 4), dtype=np.float32)
                for j in range(min(33, len(pose_landmarks))):
                    lm = pose_landmarks[j]
                    arr[j, 0] = float(getattr(lm, "x", 0.0))
                    arr[j, 1] = float(getattr(lm, "y", 0.0))
                    arr[j, 2] = float(getattr(lm, "z", 0.0))
                    arr[j, 3] = float(getattr(lm, "visibility", 0.0))
                lms.append(arr)
                s = pose_view_score(pose_landmarks)
                vs.append(float("nan") if (s is None or not np.isfinite(s)) else float(s))

        if i >= end_i:
            break
        i += 1

    cap.release()
    if not lms:
        raise RuntimeError(f"未能从视频提取姿态：{video_path}")

    meta = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(n_frames),
        "width": int(w),
        "height": int(h),
        "pose_variant": str(pose_variant),
        "start_frame": int(start_i),
        "end_frame": int(start_i + len(lms) - 1),
        "landmark_layout": "pose33_normalized_xyzw(visibility)",
        # valid_mask 契约（YOLO 迁移 Issue #5）：声明有效性策略与阈值，便于下游区分
        # 可信/待标定。本入口的 meta 会被序列化进 JSON 报告，故只落地标量字段；
        # valid_mask 由下游按同一 `valid_conf_thr` 现场推导（逐位等价）。
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
    }
    return np.stack(lms, axis=0), np.asarray(vs, dtype=np.float32), meta


def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    k = int(k)
    if k % 2 == 0:
        k += 1
    if x.size < k:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), kernel, mode="same").astype(np.float32)


def _longest_segment(mask: np.ndarray) -> tuple[int, int] | None:
    if mask.size == 0:
        return None

    best_s: int | None = None
    best_e: int | None = None
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        s = i
        i += 1
        while i < n and bool(mask[i]):
            i += 1
        e = i - 1
        if best_s is None or (e - s) > (best_e - best_s):  # type: ignore[operator]
            best_s, best_e = s, e

    if best_s is None or best_e is None:
        return None
    return int(best_s), int(best_e)


def split_front_side_segments(
    view_scores: np.ndarray,
    fps: float,
    *,
    min_len_sec: float = 1.0,
    min_mean_diff: float = 0.12,
    side_thr: float = 0.9,
    front_thr: float = 0.95,
    max_transitions: int = 3,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, str]:
    """Split by per-frame view score into (front_seg, side_seg, view_mode).

    Conservative strategy:
    - Try to find two stable segments (high vs low) by median threshold.
    - Only accept "mixed" when both segments are long enough and differ enough.
    - Otherwise fall back to "single": treat entire video as front (side unavailable).

    NOTE: side-only videos are hard to detect robustly from a single heuristic, so we prefer
    returning side_seg=None over mislabeling.
    """
    if view_scores.size == 0:
        return None, None, "single"

    vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    k = int(max(5, min(51, round(float(fps) * 0.5))))
    vs_s = _smooth_1d(vs, k=k)

    thr = float(np.median(vs_s))
    high_mask = vs_s >= thr
    high_seg = _longest_segment(high_mask)
    low_seg = _longest_segment(~high_mask)

    margin = int(max(5, min(20, round(float(fps) * 0.3))))
    min_len = int(max(1, round(float(fps) * float(min_len_sec))))
    transitions = int(np.sum(high_mask[1:] != high_mask[:-1]))

    def _shrink(seg: tuple[int, int] | None) -> tuple[int, int] | None:
        if seg is None:
            return None
        s, e = int(seg[0]), int(seg[1])
        if e - s + 1 > (2 * margin + 10):
            s += margin
            e -= margin
        if (e - s + 1) < min_len:
            return None
        return s, e

    high_seg = _shrink(high_seg)
    low_seg = _shrink(low_seg)

    if transitions <= int(max_transitions) and high_seg is not None and low_seg is not None:
        hs, he = high_seg
        ls, le = low_seg
        mean_high = float(np.mean(vs_s[hs : he + 1]))
        mean_low = float(np.mean(vs_s[ls : le + 1]))
        if abs(mean_high - mean_low) >= float(min_mean_diff):
            if mean_high >= mean_low:
                if mean_high >= float(front_thr) and mean_low <= float(side_thr):
                    return high_seg, low_seg, "mixed"
            else:
                if mean_low >= float(front_thr) and mean_high <= float(side_thr):
                    return low_seg, high_seg, "mixed"

    # Fallback: keep front on the full clip, side unavailable.
    return (0, int(vs_s.size - 1)), None, "single"


def _classify_single_view(view_scores: np.ndarray, *, side_thr: float, front_thr: float) -> str:
    vs = np.nan_to_num(view_scores.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vs.size == 0:
        return "unknown"
    med = float(np.median(vs))
    if med <= float(side_thr):
        return "side"
    if med >= float(front_thr):
        return "front"
    return "unknown"


def _center_x(lm: np.ndarray, mask_row: np.ndarray, idx_a: int, idx_b: int) -> float | None:
    va, vb = bool(mask_row[idx_a]), bool(mask_row[idx_b])
    xa, xb = float(_lm_xy(lm, idx_a)[0]), float(_lm_xy(lm, idx_b)[0])
    if va and vb:
        return 0.5 * (xa + xb)
    if va:
        return xa
    if vb:
        return xb
    return None


def _frame_dir(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
    ref = _center_x(lm, mask_row, L_HIP, R_HIP)
    if ref is None:
        ref = _center_x(lm, mask_row, L_SHOULDER, R_SHOULDER)
    if ref is None:
        return None

    face_x = None
    if bool(mask_row[NOSE]):
        face_x = float(_lm_xy(lm, NOSE)[0])
    else:
        cand: list[float] = []
        for idx in (MOUTH_L, MOUTH_R, L_EAR, R_EAR):
            if bool(mask_row[idx]):
                cand.append(float(_lm_xy(lm, idx)[0]))
        if cand:
            face_x = float(np.mean(np.asarray(cand, dtype=np.float32)))
    if face_x is None:
        return None

    dx = float(face_x) - float(ref)
    if not np.isfinite(dx):
        return None
    return 1.0 if dx >= 0.0 else -1.0


def _foot_edges_x(lm: np.ndarray, mask_row: np.ndarray, *, side: str, dir_x: float) -> tuple[float, float] | None:
    if side.upper() == "L":
        heel, toe, ankle = L_HEEL, L_FOOT_INDEX, L_ANKLE
    else:
        heel, toe, ankle = R_HEEL, R_FOOT_INDEX, R_ANKLE

    if _valid(mask_row, (heel, toe)):
        hx = float(_lm_xy(lm, heel)[0]) * float(dir_x)
        tx = float(_lm_xy(lm, toe)[0]) * float(dir_x)
        return (min(hx, tx), max(hx, tx))

    if bool(mask_row[ankle]):
        ax = float(_lm_xy(lm, ankle)[0]) * float(dir_x)
        return (ax, ax)

    return None


def _infer_front_leg_side(landmarks: np.ndarray, valid_mask: np.ndarray, *, fallback: str = "L") -> str:
    # Determine which ankle tends to be more "forward" (+x) after applying per-frame dir.
    cnt_l = 0
    cnt_r = 0
    for i, lm in enumerate(landmarks):
        mask_row = valid_mask[i]
        d = _frame_dir(lm, mask_row)
        if d is None:
            continue
        if not _valid(mask_row, (L_ANKLE, R_ANKLE)):
            continue
        xl = float(_lm_xy(lm, L_ANKLE)[0]) * float(d)
        xr = float(_lm_xy(lm, R_ANKLE)[0]) * float(d)
        if xl >= xr:
            cnt_l += 1
        else:
            cnt_r += 1

    if (cnt_l + cnt_r) < 10:
        return fallback.upper()
    return "L" if cnt_l >= cnt_r else "R"


def eval_cog_side(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    knee_straight_thr: float = 175.0,
    hip_r_back: float = 0.35,
    hip_r_front: float = 0.65,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    stance_fallback: str = "L",
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
    front_leg = _infer_front_leg_side(landmarks, mask, fallback=stance_fallback)
    back_leg = "R" if front_leg == "L" else "L"

    forward = backward = center = unknown = 0

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        d = _frame_dir(lm, mask_row)
        if d is None:
            unknown += 1
            continue

        def knee_angle(leg: str) -> float | None:
            if leg == "L":
                idxs = (L_HIP, L_KNEE, L_ANKLE)
                hip, knee, ankle = L_HIP, L_KNEE, L_ANKLE
            else:
                idxs = (R_HIP, R_KNEE, R_ANKLE)
                hip, knee, ankle = R_HIP, R_KNEE, R_ANKLE
            if not _valid(mask_row, idxs):
                return None
            return _angle_deg(_lm_xy(lm, hip), _lm_xy(lm, knee), _lm_xy(lm, ankle))

        ang_front = knee_angle(front_leg)
        ang_back = knee_angle(back_leg)

        # Scheme 2 (side): straight back-leg => forward; straight front-leg => backward.
        if ang_back is not None and float(ang_back) >= float(knee_straight_thr):
            forward += 1
            continue
        if ang_front is not None and float(ang_front) >= float(knee_straight_thr):
            backward += 1
            continue

        # Center by knee projection on the front foot.
        if front_leg == "L":
            knee_idx = L_KNEE
            foot_edges = _foot_edges_x(lm, mask_row, side="L", dir_x=d)
        else:
            knee_idx = R_KNEE
            foot_edges = _foot_edges_x(lm, mask_row, side="R", dir_x=d)

        if foot_edges is not None and bool(mask_row[knee_idx]):
            fb, ff = foot_edges
            foot_len = float(ff - fb)
            if foot_len >= 0.02:
                kx = float(_lm_xy(lm, knee_idx)[0]) * float(d)
                t = (kx - fb) / (foot_len + 1e-9)
                if 0.25 <= float(t) <= 0.75:
                    center += 1
                    continue

        # Scheme 1: hip projection ratio in the overall support area.
        hip_x = _center_x(lm, mask_row, L_HIP, R_HIP)
        if hip_x is None:
            unknown += 1
            continue

        left_edges = _foot_edges_x(lm, mask_row, side="L", dir_x=d)
        right_edges = _foot_edges_x(lm, mask_row, side="R", dir_x=d)
        if left_edges is None or right_edges is None:
            unknown += 1
            continue

        support_back = float(min(left_edges[0], right_edges[0]))
        support_front = float(max(left_edges[1], right_edges[1]))
        denom = float(support_front - support_back)
        if denom < 0.02:
            unknown += 1
            continue

        r = ((float(hip_x) * float(d)) - support_back) / (denom + 1e-9)
        if float(r) <= float(hip_r_back):
            backward += 1
        elif float(r) >= float(hip_r_front):
            forward += 1
        else:
            center += 1

    valid = forward + backward + center
    detail = {
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "front_leg": str(front_leg),
        "knee_straight_thr": float(knee_straight_thr),
        "hip_r_back": float(hip_r_back),
        "hip_r_front": float(hip_r_front),
        "trigger_ratio": float(trigger_ratio),
    }

    f_ratio = float(forward / max(1, valid))
    b_ratio = float(backward / max(1, valid))
    c_ratio = float(center / max(1, valid))
    detail["forward_ratio"] = float(f_ratio)
    detail["backward_ratio"] = float(b_ratio)
    detail["center_ratio"] = float(c_ratio)
    detail["unknown_ratio"] = float(unknown / max(1, int(landmarks.shape[0])))

    if valid < int(min_valid_frames):
        detail["primary_cause"] = "有效帧不足"
        return IndicatorResult(status="无法判定", reason="侧面：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="侧面：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="侧面：重心偏前", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="侧面：重心偏后", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="侧面：重心居中", detail=detail)


def eval_cog_front(
    landmarks: np.ndarray,
    *,
    fps: float,
    stance: str = "left",
    vis_thr: float = 0.5,
    knee_straight_thr: float = 175.0,
    ankle_acute_thr: float = 90.0,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
    stance = (stance or "left").lower()
    front_leg = "L" if stance == "left" else "R"

    forward = backward = center = unknown = 0

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        if front_leg == "L":
            knee_idxs = (L_HIP, L_KNEE, L_ANKLE)
            ankle_idxs = (L_KNEE, L_ANKLE, L_FOOT_INDEX)
            hip, knee, ankle, toe = L_HIP, L_KNEE, L_ANKLE, L_FOOT_INDEX
        else:
            knee_idxs = (R_HIP, R_KNEE, R_ANKLE)
            ankle_idxs = (R_KNEE, R_ANKLE, R_FOOT_INDEX)
            hip, knee, ankle, toe = R_HIP, R_KNEE, R_ANKLE, R_FOOT_INDEX

        knee_ang = None
        ankle_ang = None
        if _valid(mask_row, knee_idxs):
            knee_ang = _angle_deg(_lm_xy(lm, hip), _lm_xy(lm, knee), _lm_xy(lm, ankle))
        if _valid(mask_row, ankle_idxs):
            ankle_ang = _angle_deg(_lm_xy(lm, knee), _lm_xy(lm, ankle), _lm_xy(lm, toe))

        if knee_ang is None and ankle_ang is None:
            unknown += 1
            continue

        # Front rules (from requirements):
        # - front knee ~180 => backward
        # - front ankle acute (<90) => forward
        k_back = (knee_ang is not None) and (float(knee_ang) >= float(knee_straight_thr))
        a_fwd = (ankle_ang is not None) and (float(ankle_ang) < float(ankle_acute_thr))

        if k_back and a_fwd:
            unknown += 1
            continue
        if k_back:
            backward += 1
        elif a_fwd:
            forward += 1
        else:
            center += 1

    valid = forward + backward + center
    detail = {
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "front_leg": str(front_leg),
        "knee_straight_thr": float(knee_straight_thr),
        "ankle_acute_thr": float(ankle_acute_thr),
        "trigger_ratio": float(trigger_ratio),
    }

    f_ratio = float(forward / max(1, valid))
    b_ratio = float(backward / max(1, valid))
    c_ratio = float(center / max(1, valid))
    detail["forward_ratio"] = float(f_ratio)
    detail["backward_ratio"] = float(b_ratio)
    detail["center_ratio"] = float(c_ratio)
    detail["unknown_ratio"] = float(unknown / max(1, int(landmarks.shape[0])))

    if valid < int(min_valid_frames):
        detail["primary_cause"] = "有效帧不足"
        return IndicatorResult(status="无法判定", reason="正面：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="正面：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="正面：重心偏前", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="正面：重心偏后", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="正面：重心居中", detail=detail)


# ============================================================================
# 方案3：分段质心法 (Center of Mass - CoM)
# 基于人体解剖学质量分布计算真实质心，替代髋部代理
# ============================================================================

# 人体分段质量比例（基于Winter DA. Biomechanics of Human Movement）
# 躯干包含头部，四肢对称
BODY_SEGMENT_MASS_RATIOS = {
    "head": 0.081,           # 头部
    "trunk": 0.497,          # 躯干
    "upper_arm_l": 0.028,    # 左上臂
    "upper_arm_r": 0.028,    # 右上臂
    "forearm_l": 0.015,      # 左前臂
    "forearm_r": 0.015,      # 右前臂
    "hand_l": 0.006,         # 左手
    "hand_r": 0.006,         # 右手
    "thigh_l": 0.100,        # 左大腿
    "thigh_r": 0.100,        # 右大腿
    "shank_l": 0.047,        # 左小腿
    "shank_r": 0.047,        # 右小腿
    "foot_l": 0.014,         # 左脚
    "foot_r": 0.014,         # 右脚
}


def _compute_segment_center(lm: np.ndarray, mask_row: np.ndarray, indices: list[int]) -> np.ndarray | None:
    """计算身体段的中心点坐标（基于可见关键点）"""
    points = []
    for idx in indices:
        if bool(mask_row[idx]):
            points.append(_lm_xy(lm, idx))
    if len(points) == 0:
        return None
    return np.mean(points, axis=0)


def _compute_body_com_single(lm: np.ndarray, mask_row: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """
    计算单帧人体质心（CoM）
    返回: (com_coords, debug_info)
    """
    weighted_sum = np.zeros(2)
    total_mass = 0.0
    segment_status = {}

    # 头部：使用鼻子和双耳
    head_center = _compute_segment_center(lm, mask_row, [NOSE, L_EAR, R_EAR])
    if head_center is not None:
        mass = BODY_SEGMENT_MASS_RATIOS["head"]
        weighted_sum += head_center * mass
        total_mass += mass
        segment_status["head"] = {"mass": mass, "center": head_center.tolist()}
    else:
        segment_status["head"] = None

    # 躯干：质心约在剑突-肚脐之间（偏下约40%，即肩:髋 = 0.6:0.4）
    if _valid(mask_row, [L_SHOULDER, R_SHOULDER]) and _valid(mask_row, [L_HIP, R_HIP]):
        shoulder_center = (_lm_xy(lm, L_SHOULDER) + _lm_xy(lm, R_SHOULDER)) / 2
        hip_center = (_lm_xy(lm, L_HIP) + _lm_xy(lm, R_HIP)) / 2
        trunk_com = shoulder_center * 0.6 + hip_center * 0.4
        mass = BODY_SEGMENT_MASS_RATIOS["trunk"]
        weighted_sum += trunk_com * mass
        total_mass += mass
        segment_status["trunk"] = {"mass": mass, "center": trunk_com.tolist()}
    else:
        segment_status["trunk"] = None

    # 左上臂
    if _valid(mask_row, [L_SHOULDER, L_ELBOW]):
        shoulder = _lm_xy(lm, L_SHOULDER)
        elbow = _lm_xy(lm, L_ELBOW)
        com = (shoulder + elbow) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["upper_arm_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["upper_arm_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["upper_arm_l"] = None

    # 右上臂
    if _valid(mask_row, [R_SHOULDER, R_ELBOW]):
        shoulder = _lm_xy(lm, R_SHOULDER)
        elbow = _lm_xy(lm, R_ELBOW)
        com = (shoulder + elbow) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["upper_arm_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["upper_arm_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["upper_arm_r"] = None

    # 左前臂
    if _valid(mask_row, [L_ELBOW, L_WRIST]):
        elbow = _lm_xy(lm, L_ELBOW)
        wrist = _lm_xy(lm, L_WRIST)
        com = (elbow + wrist) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["forearm_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["forearm_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["forearm_l"] = None

    # 右前臂
    if _valid(mask_row, [R_ELBOW, R_WRIST]):
        elbow = _lm_xy(lm, R_ELBOW)
        wrist = _lm_xy(lm, R_WRIST)
        com = (elbow + wrist) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["forearm_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["forearm_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["forearm_r"] = None

    # 左手
    if _valid(mask_row, [L_WRIST, L_INDEX, L_PINKY]):
        com = _compute_segment_center(lm, mask_row, [L_WRIST, L_INDEX, L_PINKY])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["hand_l"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["hand_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["hand_l"] = None

    # 右手
    if _valid(mask_row, [R_WRIST, R_INDEX, R_PINKY]):
        com = _compute_segment_center(lm, mask_row, [R_WRIST, R_INDEX, R_PINKY])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["hand_r"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["hand_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["hand_r"] = None

    # 左大腿
    if _valid(mask_row, [L_HIP, L_KNEE]):
        hip = _lm_xy(lm, L_HIP)
        knee = _lm_xy(lm, L_KNEE)
        com = (hip + knee) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["thigh_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["thigh_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["thigh_l"] = None

    # 右大腿
    if _valid(mask_row, [R_HIP, R_KNEE]):
        hip = _lm_xy(lm, R_HIP)
        knee = _lm_xy(lm, R_KNEE)
        com = (hip + knee) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["thigh_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["thigh_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["thigh_r"] = None

    # 左小腿
    if _valid(mask_row, [L_KNEE, L_ANKLE]):
        knee = _lm_xy(lm, L_KNEE)
        ankle = _lm_xy(lm, L_ANKLE)
        com = (knee + ankle) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["shank_l"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["shank_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["shank_l"] = None

    # 右小腿
    if _valid(mask_row, [R_KNEE, R_ANKLE]):
        knee = _lm_xy(lm, R_KNEE)
        ankle = _lm_xy(lm, R_ANKLE)
        com = (knee + ankle) / 2
        mass = BODY_SEGMENT_MASS_RATIOS["shank_r"]
        weighted_sum += com * mass
        total_mass += mass
        segment_status["shank_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["shank_r"] = None

    # 左脚
    if _valid(mask_row, [L_ANKLE, L_HEEL, L_FOOT_INDEX]):
        com = _compute_segment_center(lm, mask_row, [L_ANKLE, L_HEEL, L_FOOT_INDEX])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["foot_l"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["foot_l"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["foot_l"] = None

    # 右脚
    if _valid(mask_row, [R_ANKLE, R_HEEL, R_FOOT_INDEX]):
        com = _compute_segment_center(lm, mask_row, [R_ANKLE, R_HEEL, R_FOOT_INDEX])
        if com is not None:
            mass = BODY_SEGMENT_MASS_RATIOS["foot_r"]
            weighted_sum += com * mass
            total_mass += mass
            segment_status["foot_r"] = {"mass": mass, "center": com.tolist()}
    else:
        segment_status["foot_r"] = None

    if total_mass < 0.5:  # 至少50%的身体质量可见
        return None, {"error": "insufficient_visible_mass", "total_mass": total_mass, "segments": segment_status}

    com = weighted_sum / total_mass
    return com, {"total_mass": total_mass, "segments": segment_status}


def eval_cog_com(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    com_r_back: float = 0.38,
    com_r_front: float = 0.62,
    trigger_ratio: float = 0.3,
    min_valid_frames: int = 15,
    valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    """
    基于分段质心(CoM)的重心判定（方案3）
    
    优势：
    - 更准确反映全身重心分布
    - 对"上身前倾/后仰"场景更鲁棒
    - 减少单点（髋部）代理带来的偏差
    
    参数说明：
    - com_r_back/com_r_front: CoM投影比例阈值（需重新标定，比髋部阈值更宽松）
    """
    if landmarks.size == 0:
        detail = {
            "primary_cause": "无有效帧",
            "valid_frames": 0,
            "total_frames": 0,
        }
        return IndicatorResult(status="无法判定", reason="无有效帧", detail=detail)

    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
    forward = backward = center = unknown = 0
    com_ratios = []

    for i, lm in enumerate(landmarks):
        mask_row = mask[i]
        # 计算该帧CoM
        com, _info = _compute_body_com_single(lm, mask_row)

        if com is None:
            unknown += 1
            continue

        d = _frame_dir(lm, mask_row)
        if d is None:
            unknown += 1
            continue

        com_x = float(com[0]) * float(d)

        # 计算支撑面边界（两脚heel/toe）
        left_edges = _foot_edges_x(lm, mask_row, side="L", dir_x=d)
        right_edges = _foot_edges_x(lm, mask_row, side="R", dir_x=d)
        
        if left_edges is None or right_edges is None:
            unknown += 1
            continue

        support_back = float(min(left_edges[0], right_edges[0]))
        support_front = float(max(left_edges[1], right_edges[1]))
        denom = float(support_front - support_back)
        
        if denom < 0.02:
            unknown += 1
            continue

        # CoM投影比例
        r = (float(com_x) - support_back) / (denom + 1e-9)
        com_ratios.append(r)

        if float(r) <= float(com_r_back):
            backward += 1
        elif float(r) >= float(com_r_front):
            forward += 1
        else:
            center += 1

    valid = forward + backward + center
    
    # 计算com_ratio统计信息
    com_stats = {}
    if com_ratios:
        com_arr = np.array(com_ratios)
        com_stats = {
            "mean": float(np.mean(com_arr)),
            "std": float(np.std(com_arr)),
            "median": float(np.median(com_arr)),
            "min": float(np.min(com_arr)),
            "max": float(np.max(com_arr)),
        }

    detail = {
        "method": "CoM_segment_based",
        "valid_frames": int(valid),
        "total_frames": int(landmarks.shape[0]),
        "forward_frames": int(forward),
        "backward_frames": int(backward),
        "center_frames": int(center),
        "unknown_frames": int(unknown),
        "com_r_back": float(com_r_back),
        "com_r_front": float(com_r_front),
        "com_ratio_stats": com_stats,
        "trigger_ratio": float(trigger_ratio),
    }

    f_ratio = float(forward / max(1, valid))
    b_ratio = float(backward / max(1, valid))
    c_ratio = float(center / max(1, valid))
    detail["forward_ratio"] = float(f_ratio)
    detail["backward_ratio"] = float(b_ratio)
    detail["center_ratio"] = float(c_ratio)
    detail["unknown_ratio"] = float(unknown / max(1, int(landmarks.shape[0])))

    if valid < int(min_valid_frames):
        detail["primary_cause"] = "有效帧不足"
        return IndicatorResult(status="无法判定", reason="CoM：有效帧不足/遮挡", detail=detail)

    if f_ratio >= float(trigger_ratio) and b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "冲突"
        return IndicatorResult(status="无法判定", reason="CoM：前后判断冲突（可能斜拍/遮挡）", detail=detail)
    if f_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏前"
        return IndicatorResult(status="不合格", reason="重心偏前（CoM）", detail=detail)
    if b_ratio >= float(trigger_ratio):
        detail["primary_cause"] = "偏后"
        return IndicatorResult(status="不合格", reason="重心偏后（CoM）", detail=detail)
    detail["primary_cause"] = "居中"
    return IndicatorResult(status="合格", reason="重心居中（CoM）", detail=detail)


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

    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
    events = _detect_retract_events_side(
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


def _detect_retract_events_side(
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
    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
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
            d = _frame_dir(lm, mask_row)
            if d is None:
                continue
            if not _valid(mask_row, (sh, el, wr)):
                continue
            a = _angle_deg(_lm_xy(lm, sh), _lm_xy(lm, el), _lm_xy(lm, wr))
            ang[i] = float(a)
            dx[i] = (float(_lm_xy(lm, wr)[0]) - float(_lm_xy(lm, sh)[0])) * float(d)

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


_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (L_ANKLE, L_HEEL),
    (L_HEEL, L_FOOT_INDEX),
    (L_ANKLE, L_FOOT_INDEX),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (R_ANKLE, R_HEEL),
    (R_HEEL, R_FOOT_INDEX),
    (R_ANKLE, R_FOOT_INDEX),
)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
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


def _subset_by_intervals(
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
    clipped = _merge_intervals(clipped)
    if not clipped:
        empty_mask = None if valid_mask is None else np.zeros((0, 33), dtype=bool)
        return np.zeros((0, 33, 4), dtype=np.float32), [], empty_mask
    idxs: list[int] = []
    for s, e in clipped:
        idxs.extend(list(range(int(s), int(e) + 1)))
    sel = np.asarray(idxs, dtype=np.int32)
    sub_mask = None if valid_mask is None else valid_mask[sel]
    return landmarks[sel], clipped, sub_mask


def _detect_extension_events(
    landmarks: np.ndarray,
    *,
    fps: float,
    vis_thr: float = 0.5,
    ext_angle_thr: float = 160.0,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    mask = _resolve_mask(landmarks, valid_mask, vis_thr)

    def series(arm: str) -> np.ndarray:
        if arm.upper() == "L":
            sh, el, wr = L_SHOULDER, L_ELBOW, L_WRIST
        else:
            sh, el, wr = R_SHOULDER, R_ELBOW, R_WRIST
        ang = np.full((landmarks.shape[0],), np.nan, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            if not _valid(mask[i], (sh, el, wr)):
                continue
            ang[i] = float(_angle_deg(_lm_xy(lm, sh), _lm_xy(lm, el), _lm_xy(lm, wr)))
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


def _eval_cog_side_prefer_punch_windows(
    landmarks: np.ndarray, 
    *, 
    fps: float,
    use_com: bool = False,
    valid_mask: np.ndarray | None = None,
) -> tuple[IndicatorResult, IndicatorResult | None]:
    """
    侧面重心评估（优先出拳窗口）
    
    Args:
        use_com: 是否同时返回CoM（方案3）评估结果
        valid_mask: ``(T,33)`` 有效性掩码，``None`` 时现场推导（行为不变）。
    
    Returns:
        (主评估结果, CoM评估结果或None)
    """
    mask = _resolve_mask(landmarks, valid_mask, DEFAULT_VALID_CONF_THR)
    seg_res = eval_cog_side(landmarks, fps=fps, stance_fallback="L", valid_mask=mask)
    events = _detect_retract_events_side(
        landmarks,
        fps=fps,
        vis_thr=0.5,
        ext_angle_thr=160.0,
        retract_angle_thr=45.0,
        ext_forward_dx=0.02,
        retract_sec_thr=0.2,
        max_window_sec=1.0,
        valid_mask=mask,
    )
    
    com_res: IndicatorResult | None = None
    
    if not events:
        if seg_res.detail is not None:
            seg_res.detail["eval_scope"] = "segment"
        # 无出拳事件时，全段使用CoM评估
        if use_com:
            com_res = eval_cog_com(landmarks, fps=fps, valid_mask=mask)
        return seg_res, com_res

    pre = int(round(0.3 * float(fps)))
    post = int(round(0.1 * float(fps)))
    n = int(landmarks.shape[0])

    intervals: list[tuple[int, int]] = []
    for e in events:
        s = int(e.get("start") or 0)
        end = e.get("end")
        if end is None:
            end = min(n - 1, s + int(round(1.0 * float(fps))))
        intervals.append((s - pre, int(end) + post))

    subset, used, subset_mask = _subset_by_intervals(landmarks, intervals, mask)
    
    # 方案1+2：基于髋部/膝角的评估
    win_res = eval_cog_side(subset, fps=fps, stance_fallback="L", valid_mask=subset_mask)
    if win_res.detail is not None:
        win_res.detail["eval_scope"] = "punch_windows"
        win_res.detail["punch_windows"] = used
        win_res.detail["punch_events"] = int(len(events))
    
    # 方案3：基于分段质心的评估（出拳窗口）
    if use_com:
        com_res = eval_cog_com(subset, fps=fps, valid_mask=subset_mask)
        if com_res.detail is not None:
            com_res.detail["eval_scope"] = "punch_windows"
            com_res.detail["punch_windows"] = used
            com_res.detail["punch_events"] = int(len(events))
    
    if win_res.status != "无法判定":
        return win_res, com_res

    # 出拳窗口无法判定时，回退到全段评估
    if seg_res.detail is not None:
        seg_res.detail["eval_scope"] = "segment_fallback"
        seg_res.detail["punch_windows"] = used
        seg_res.detail["punch_events"] = int(len(events))
    
    return seg_res, com_res


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

    mask = _resolve_mask(landmarks, valid_mask, vis_thr)
    events = _detect_extension_events(landmarks, fps=fps, vis_thr=vis_thr, ext_angle_thr=ext_angle_thr, valid_mask=mask)
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
            if not _valid(mask[i], (el, wr, idx, pk)):
                continue
            mid = (_lm_xy(lm, idx) + _lm_xy(lm, pk)) * 0.5
            a = _angle_deg(_lm_xy(lm, el), _lm_xy(lm, wr), mid)
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


def eval_force_sequence(
    front_landmarks: np.ndarray,
    side_landmarks: np.ndarray,
    *,
    fps: float,
    stance: str = "left",
    vis_thr: float = 0.5,
    twist_min_deg: float = 10.0,
    shoulder_drive_min: float = 0.03,
    step_front_min: float = 0.01,
    step_back_min: float = 0.01,
    step_ok_ratio: float = 0.5,
    step_sync_allow: int = 3,
    step_post_sec: float = 0.4,
    rotation_var_thr: float = 1820.0,
    rotation_min_frames: int = 12,
    front_valid_mask: np.ndarray | None = None,
    side_valid_mask: np.ndarray | None = None,
) -> IndicatorResult:
    front_mask = _resolve_mask(front_landmarks, front_valid_mask, vis_thr)
    side_mask = _resolve_mask(side_landmarks, side_valid_mask, vis_thr)

    def _foot_center_x(lm: np.ndarray, mask_row: np.ndarray, side: str) -> float | None:
        if side.upper() == "L":
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        if not _valid(mask_row, (heel, toe)):
            return None
        return float((_lm_xy(lm, heel)[0] + _lm_xy(lm, toe)[0]) * 0.5)

    def _foot_angle_deg(lm: np.ndarray, mask_row: np.ndarray, side: str) -> float | None:
        if side.upper() == "L":
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        if not _valid(mask_row, (heel, toe)):
            return None
        a = _lm_xy(lm, heel)
        b = _lm_xy(lm, toe)
        return float(np.degrees(np.arctan2(float(b[1] - a[1]), float(b[0] - a[0]))))

    def _rotation_fail_front(lms: np.ndarray, lms_mask: np.ndarray) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {
            "rotation_fail": False,
            "rotation_var_l": None,
            "rotation_var_r": None,
            "rotation_var_thr": float(rotation_var_thr),
            "rotation_frames_l": 0,
            "rotation_frames_r": 0,
        }
        if lms.size == 0:
            return False, detail

        def _series(side: str) -> np.ndarray:
            vals: list[float] = []
            for i in range(int(lms.shape[0])):
                v = _foot_angle_deg(lms[i], lms_mask[i], side)
                if v is None or not np.isfinite(v):
                    continue
                vals.append(float(v))
            return np.asarray(vals, dtype=np.float32)

        def _var_deg(angles: np.ndarray) -> float | None:
            if angles.size < int(rotation_min_frames):
                return None
            ang = np.deg2rad(angles.astype(np.float32))
            ang = np.unwrap(ang)
            ang_deg = np.rad2deg(ang)
            return float(np.var(ang_deg))

        angles_l = _series("L")
        angles_r = _series("R")
        detail["rotation_frames_l"] = int(angles_l.size)
        detail["rotation_frames_r"] = int(angles_r.size)
        var_l = _var_deg(angles_l)
        var_r = _var_deg(angles_r)
        detail["rotation_var_l"] = None if var_l is None else float(var_l)
        detail["rotation_var_r"] = None if var_r is None else float(var_r)

        fail = False
        if var_l is not None and float(var_l) > float(rotation_var_thr):
            fail = True
        if var_r is not None and float(var_r) > float(rotation_var_thr):
            fail = True
        detail["rotation_fail"] = bool(fail)
        return bool(fail), detail

    rotation_fail, rotation_detail = _rotation_fail_front(front_landmarks, front_mask)
    if rotation_fail:
        detail = {
            "events_total": 0,
            "events_used": 0,
            "events_ok": 0,
            "events_bad": 0,
            "push_detected": 0,
            "twist_detected": 0,
            "drive_detected": 0,
            "step_front_detected": 0,
            "step_back_detected": 0,
            "step_ok": 0,
            "step_ok_ratio": None,
            "rotation_fail": True,
            "rotation": rotation_detail,
            "events": [],
        }
        detail["primary_cause"] = "脚部旋转异常"
        detail["failed_stage"] = "rotation"
        return IndicatorResult(status="不合格", reason="发力顺序：正面脚部旋转异常，判定蹬地与顺序不合格", detail=detail)

    if side_landmarks.size == 0:
        detail = {"primary_cause": "侧面段缺失"}
        return IndicatorResult(status="无法判定", reason="发力顺序：侧面段缺失", detail=detail)

    events = _detect_retract_events_side(
        side_landmarks,
        fps=fps,
        vis_thr=vis_thr,
        ext_angle_thr=160.0,
        retract_angle_thr=45.0,
        ext_forward_dx=0.02,
        retract_sec_thr=0.2,
        max_window_sec=1.0,
        valid_mask=side_mask,
    )
    if not events:
        detail = {"primary_cause": "未检测到出拳", "events": []}
        return IndicatorResult(status="无法判定", reason="发力顺序：未检测到出拳", detail=detail)

    front_leg = _infer_front_leg_side(side_landmarks, side_mask, fallback=("L" if (stance or "left").lower() == "left" else "R"))
    back_leg = "R" if front_leg == "L" else "L"

    def _wrap_deg(x: float) -> float:
        y = (float(x) + 180.0) % 360.0 - 180.0
        return float(y)

    def _twist_deg(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
        if not _valid(mask_row, (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)):
            return None
        ls = _lm_xy(lm, L_SHOULDER)
        rs = _lm_xy(lm, R_SHOULDER)
        lh = _lm_xy(lm, L_HIP)
        rh = _lm_xy(lm, R_HIP)
        sa = float(np.degrees(np.arctan2(float(rs[1] - ls[1]), float(rs[0] - ls[0]))))
        ha = float(np.degrees(np.arctan2(float(rh[1] - lh[1]), float(rh[0] - lh[0]))))
        return abs(_wrap_deg(sa - ha))

    def _shoulder_drive(lm: np.ndarray, mask_row: np.ndarray, arm: str) -> float | None:
        d = _frame_dir(lm, mask_row)
        if d is None:
            return None
        if arm.upper() == "L":
            sh = L_SHOULDER
        else:
            sh = R_SHOULDER
        if not _valid(mask_row, (sh, L_HIP, R_HIP)):
            return None
        hip_x = float((_lm_xy(lm, L_HIP)[0] + _lm_xy(lm, R_HIP)[0]) * 0.5)
        return (float(_lm_xy(lm, sh)[0]) - hip_x) * float(d)

    def _push_off_ok(lm: np.ndarray, mask_row: np.ndarray) -> bool | None:
        """检测蹬地动作：后脚脚跟明显抬高 + 膝盖参与伸展发力。
        
        真正的蹬地特征：
        1. 脚跟明显抬高（pitch 在合理范围，且垂直位移足够）
        2. 后腿膝盖有一定伸展（说明腿部在发力推地）
        """
        # 获取后脚关键点索引
        if back_leg == "L":
            heel, toe, hip, knee, ankle = L_HEEL, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE
        else:
            heel, toe, hip, knee, ankle = R_HEEL, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE
        
        # 检查基本可见性
        if not _valid(mask_row, (heel, toe, knee, ankle)):
            return None
        
        # 1. 脚跟明显抬高检测
        heel_xy = _lm_xy(lm, heel)
        toe_xy = _lm_xy(lm, toe)
        dy = float(toe_xy[1] - heel_xy[1])  # 脚尖y - 脚跟y（y向下为正）
        if dy <= 0.0:
            return False  # 脚跟必须高于脚尖（抬脚跟）
        
        dx = float(toe_xy[0] - heel_xy[0])
        pitch = float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9)))
        
        # 基于标准动作NPZ数据分析优化的阈值（侧面视角）
        # - Pitch P80 范围: 31-38°, 取 20-45° 作为有效范围
        # - dy_ratio P80 范围: 0.6-0.8, 取 ≥0.5 作为阈值
        # - 膝盖角度 P20-P90: 125-165°, 取 120-165° 作为有效范围
        pitch_ok = 20.0 <= pitch <= 45.0
        dy_ratio = dy / (abs(dx) + 1e-6)
        heel_lift_ok = pitch_ok and (dy >= 0.03) and (dy_ratio >= 0.5)
        
        # 2. 后腿膝盖角度检测（基于标准动作数据：P20=125°, P90=165°）
        knee_ang = _angle_deg(_lm_xy(lm, hip), _lm_xy(lm, knee), _lm_xy(lm, ankle))
        # 蹬地时后腿膝盖通常在 120°-165° 之间
        knee_ok = 120.0 <= knee_ang <= 165.0
        
        return heel_lift_ok and knee_ok

    def _calc_heel_lift(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
        """计算后脚脚跟抬高程度（dy / |dx| 的比例，范围 0~1+）。"""
        if back_leg == "L":
            heel, toe = L_HEEL, L_FOOT_INDEX
        else:
            heel, toe = R_HEEL, R_FOOT_INDEX
        if not _valid(mask_row, (heel, toe)):
            return None
        heel_xy = _lm_xy(lm, heel)
        toe_xy = _lm_xy(lm, toe)
        dy = float(toe_xy[1] - heel_xy[1])  # y向下为正，脚尖在脚跟下方时dy>0
        dx = float(toe_xy[0] - heel_xy[0])
        if dy <= 0:
            return 0.0
        return dy / (abs(dx) + 1e-6)
    
    def _calc_knee_angle(lm: np.ndarray, mask_row: np.ndarray) -> float | None:
        """计算后腿膝盖角度。"""
        if back_leg == "L":
            hip, knee, ankle = L_HIP, L_KNEE, L_ANKLE
        else:
            hip, knee, ankle = R_HIP, R_KNEE, R_ANKLE
        if not _valid(mask_row, (hip, knee, ankle)):
            return None
        return _angle_deg(_lm_xy(lm, hip), _lm_xy(lm, knee), _lm_xy(lm, ankle))

    pre = int(round(0.3 * float(fps)))
    post_step = int(round(float(step_post_sec) * float(fps)))
    allow_overlap = 3
    ok = 0
    bad = 0
    used = 0
    per_event: list[dict[str, Any]] = []

    for e in events:
        arm = str(e.get("arm") or "")
        t_punch = int(e.get("start") or 0)
        s = max(0, t_punch - pre)
        lm_pre = side_landmarks[s : t_punch + 1]
        mask_pre = side_mask[s : t_punch + 1]
        if lm_pre.size == 0:
            continue

        # 1. 首先尝试静态检测（单帧是否符合蹬地姿态）
        t_push = None
        for j in range(lm_pre.shape[0]):
            ok_push = _push_off_ok(lm_pre[j], mask_pre[j])
            if ok_push is None:
                continue
            if bool(ok_push):
                t_push = int(s + j)
                break
        
        # 2. 如果静态检测失败，尝试动态检测（脚跟是否有明显抬高趋势）
        push_dynamic_score = 0.0
        if t_push is None and lm_pre.shape[0] >= 5:
            heel_lifts = [_calc_heel_lift(lm_pre[j], mask_pre[j]) for j in range(lm_pre.shape[0])]
            knee_angles = [_calc_knee_angle(lm_pre[j], mask_pre[j]) for j in range(lm_pre.shape[0])]
            
            # 寻找脚跟抬高的趋势：早期低 -> 晚期高
            valid_lifts = [(j, v) for j, v in enumerate(heel_lifts) if v is not None]
            valid_knees = [(j, v) for j, v in enumerate(knee_angles) if v is not None]
            
            if len(valid_lifts) >= 4 and len(valid_knees) >= 4:
                # 分前后两半比较
                mid = len(valid_lifts) // 2
                early_lifts = [v for _, v in valid_lifts[:mid]]
                late_lifts = [v for _, v in valid_lifts[mid:]]
                early_knees = [v for _, v in valid_knees[:mid]]
                late_knees = [v for _, v in valid_knees[mid:]]
                
                lift_increase = float(np.median(late_lifts)) - float(np.median(early_lifts))
                knee_change = float(np.median(late_knees)) - float(np.median(early_knees))
                
                # 动态蹬地特征（基于标准动作NPZ数据分析优化）：
                # - 脚跟明显抬高（后半段比前半段高至少 0.10，且后半段中位数≥0.40）
                # - 膝盖角度变化不大或略微伸展（变化在 -15°~+20° 之间）
                lift_ok = lift_increase >= 0.10 and float(np.median(late_lifts)) >= 0.40
                knee_stable = -15.0 <= knee_change <= 20.0
                
                if lift_ok and knee_stable:
                    # 找到后半段第一个lift达标的位置作为t_push
                    for j, lift in valid_lifts[mid:]:
                        if lift >= 0.40:
                            t_push = int(s + j)
                            push_dynamic_score = float(lift_increase)
                            break

        twists: list[tuple[int, float]] = []
        for j in range(lm_pre.shape[0]):
            v = _twist_deg(lm_pre[j], mask_pre[j])
            if v is None:
                continue
            twists.append((int(s + j), float(v)))
        if twists:
            t_twist, twist_v = max(twists, key=lambda t: t[1])
            if float(twist_v) < float(twist_min_deg):
                t_twist = None
        else:
            t_twist = None

        drives: list[tuple[int, float]] = []
        for j in range(lm_pre.shape[0]):
            v = _shoulder_drive(lm_pre[j], mask_pre[j], arm)
            if v is None:
                continue
            drives.append((int(s + j), float(v)))
        if drives:
            base = float(np.median(np.asarray([v for _t, v in drives[: max(1, len(drives) // 2)]], dtype=np.float32)))
            t_drive = None
            for t, v in drives:
                if float(v - base) >= float(shoulder_drive_min):
                    t_drive = int(t)
                    break
        else:
            t_drive = None

        # 上步检测（前脚上步 -> 后脚跟进）
        front_step_detected = None
        back_step_detected = None
        t_front_step_50 = None
        t_front_step_100 = None
        t_back_step_start = None
        step_ok = None
        def _foot_series(lms: np.ndarray, lms_mask: np.ndarray, side: str) -> np.ndarray:
            xs = np.full((lms.shape[0],), np.nan, dtype=np.float32)
            for i in range(lms.shape[0]):
                lm = lms[i]
                mask_row = lms_mask[i]
                d = _frame_dir(lm, mask_row)
                if d is None:
                    continue
                cx = _foot_center_x(lm, mask_row, side)
                if cx is None:
                    continue
                xs[i] = float(cx) * float(d)
            return xs

        post_end = min(int(side_landmarks.shape[0]), int(t_punch + post_step + 1))
        lm_post = side_landmarks[t_punch : post_end]
        mask_post = side_mask[t_punch : post_end]
        front_x_pre = _foot_series(lm_pre, mask_pre, front_leg)
        back_x_post = _foot_series(lm_post, mask_post, back_leg) if lm_post.size > 0 else np.zeros((0,), dtype=np.float32)

        def _first_idx(vals: np.ndarray, thr: float) -> int | None:
            for i in range(int(vals.size)):
                v = float(vals[i])
                if np.isfinite(v) and v >= float(thr):
                    return int(i)
            return None

        def _baseline(vals: np.ndarray) -> float | None:
            if vals.size == 0:
                return None
            n0 = max(1, int(round(float(vals.size) * 0.2)))
            base = vals[:n0]
            base = base[np.isfinite(base)]
            if base.size == 0:
                return None
            return float(np.median(base))

        base_front = _baseline(front_x_pre)
        if base_front is not None:
            front_disp = front_x_pre - float(base_front)
            t_front_step_100 = _first_idx(front_disp, float(step_front_min))
            t_front_step_50 = _first_idx(front_disp, float(step_front_min) * 0.5)
            front_step_detected = t_front_step_100 is not None

        base_back = None
        if back_x_post.size > 0:
            base_back = _baseline(back_x_post)
        if base_back is not None:
            back_disp = back_x_post - float(base_back)
            t_back_step_start = _first_idx(back_disp, float(step_back_min))
            back_step_detected = t_back_step_start is not None

        if front_step_detected is None or back_step_detected is None:
            step_ok = None
        elif not front_step_detected or not back_step_detected:
            step_ok = False
        else:
            t_front_step_50_abs = None if t_front_step_50 is None else int(s + t_front_step_50)
            t_front_step_100_abs = None if t_front_step_100 is None else int(s + t_front_step_100)
            t_back_step_abs = None if t_back_step_start is None else int(t_punch + t_back_step_start)
            front_before_punch = (t_front_step_50_abs is not None) and (t_front_step_50_abs <= int(t_punch))
            front_sync = (t_front_step_100_abs is not None) and (abs(int(t_front_step_100_abs) - int(t_punch)) <= int(step_sync_allow))
            back_after_punch = (t_back_step_abs is not None) and (int(t_back_step_abs) >= int(t_punch - 1))
            step_ok = bool(front_before_punch and front_sync and back_after_punch)

        used += 1
        if t_push is None or t_twist is None or t_drive is None:
            per_event.append({
                "arm": arm, "t_push": t_push, "t_twist": t_twist, "t_drive": t_drive,
                "t_punch": int(t_punch), "ok": None,
                "push_detected": t_push is not None,
                "twist_detected": t_twist is not None,
                "drive_detected": t_drive is not None,
                "push_dynamic_score": float(push_dynamic_score) if push_dynamic_score > 0 else None,
                "front_step_detected": front_step_detected,
                "back_step_detected": back_step_detected,
                "t_front_step_50": None if t_front_step_50 is None else int(s + t_front_step_50),
                "t_front_step_100": None if t_front_step_100 is None else int(s + t_front_step_100),
                "t_back_step": None if t_back_step_start is None else int(t_punch + t_back_step_start),
                "step_ok": step_ok,
            })
            continue

        # 判定顺序是否正确：
        # - 蹬地应在转腰之前或同时（允许蹬地略晚于转腰3帧，考虑到侧面视角检测误差）
        # - 转腰应在送肩之前或同时（允许3帧重叠）
        # - 送肩应在出拳之前或同时
        push_before_twist = (t_push <= t_twist + 3)  # 允许蹬地略晚于转腰3帧（约0.1秒@30fps）
        twist_before_drive = (t_twist <= t_drive + allow_overlap)
        drive_before_punch = (t_drive <= t_punch)
        is_ok = bool(push_before_twist and twist_before_drive and drive_before_punch)
        ok += 1 if is_ok else 0
        bad += 0 if is_ok else 1
        per_event.append({
            "arm": arm, "t_push": int(t_push), "t_twist": int(t_twist), "t_drive": int(t_drive),
            "t_punch": int(t_punch), "ok": bool(is_ok),
            "push_dynamic_score": float(push_dynamic_score) if push_dynamic_score > 0 else None,
            "front_step_detected": front_step_detected,
            "back_step_detected": back_step_detected,
            "t_front_step_50": None if t_front_step_50 is None else int(s + t_front_step_50),
            "t_front_step_100": None if t_front_step_100 is None else int(s + t_front_step_100),
            "t_back_step": None if t_back_step_start is None else int(t_punch + t_back_step_start),
            "step_ok": step_ok,
        })

    # 统计各信号检测情况
    push_detected = sum(1 for e in per_event if e.get("t_push") is not None)
    twist_detected = sum(1 for e in per_event if e.get("t_twist") is not None)
    drive_detected = sum(1 for e in per_event if e.get("t_drive") is not None)
    step_front_detected = sum(1 for e in per_event if e.get("front_step_detected") is True)
    step_back_detected = sum(1 for e in per_event if e.get("back_step_detected") is True)
    step_eval = [e for e in per_event if e.get("step_ok") is not None]
    step_ok = sum(1 for e in step_eval if e.get("step_ok") is True)
    
    detail = {
        "events_total": int(len(events)),
        "events_used": int(used),
        "events_ok": int(ok),
        "events_bad": int(bad),
        "push_detected": int(push_detected),
        "twist_detected": int(twist_detected),
        "drive_detected": int(drive_detected),
        "step_front_detected": int(step_front_detected),
        "step_back_detected": int(step_back_detected),
        "step_ok": int(step_ok),
        "step_ok_ratio": None if len(step_eval) == 0 else float(step_ok / max(1, len(step_eval))),
        "front_leg": str(front_leg),
        "back_leg": str(back_leg),
        "twist_min_deg": float(twist_min_deg),
        "shoulder_drive_min": float(shoulder_drive_min),
        "step_front_min": float(step_front_min),
        "step_back_min": float(step_back_min),
        "step_ok_ratio_thr": float(step_ok_ratio),
        "rotation": rotation_detail,
        "events": per_event,
    }

    # 判定逻辑：
    # 1. 首先要求有足够的事件用于评估
    if used < 2:
        detail["primary_cause"] = "出拳次数不足"
        return IndicatorResult(status="无法判定", reason="发力顺序：出拳次数不足", detail=detail)

    # 2. 计算蹬地检测率
    push_detection_rate = float(push_detected / max(1, used))
    
    # 3. 在有效检测到蹬地的事件中，按顺序正确率判定
    evaluated = ok + bad
    accuracy = float(ok / max(1, evaluated)) if evaluated > 0 else 0.0
    
    # 4. 综合判定（改进版）：
    # 关键逻辑：蹬地检测率过低说明大量出拳没有标准蹬地动作
    # - 如果蹬地检测率 < 40%：视为"不合格"（大量出拳蹬地动作不标准）
    # - 如果检测率 >= 40% 且正确率 >= 70%：合格
    # - 如果检测率 >= 40% 且正确率 < 70%：不合格
    # 
    # 理由：侧面视角下虽然后腿可能部分遮挡，但40%以上的检测率是基本要求
    # 如果检测率<40%，说明多数出拳的蹬地动作不规范或没有蹬地
    
    if push_detection_rate < 0.40:
        detail["primary_cause"] = "蹬地检测率低"
        detail["failed_stage"] = "push_off"
        return IndicatorResult(
            status="不合格", 
            reason="发力顺序：蹬地动作检测率偏低，多数出拳蹬地动作不标准", 
            detail=detail
        )

    if len(step_eval) == 0:
        detail["primary_cause"] = "上步信号不足"
        detail["failed_stage"] = "step"
        return IndicatorResult(status="无法判定", reason="发力顺序：上步信号不足", detail=detail)
    step_ratio = float(step_ok / max(1, len(step_eval)))
    if step_ratio < float(step_ok_ratio):
        detail["primary_cause"] = "上步不规范"
        detail["failed_stage"] = "step"
        return IndicatorResult(status="不合格", reason="发力顺序：上步不规范", detail=detail)
    
    if evaluated < 2:
        detail["primary_cause"] = "有效评估事件不足"
        return IndicatorResult(status="无法判定", reason="发力顺序：有效评估事件不足", detail=detail)
    
    if accuracy >= 0.7:  # 至少70%的正确率才算合格
        detail["primary_cause"] = "达标"
        return IndicatorResult(status="合格", reason="发力顺序：顺序一致性良好", detail=detail)
    detail["primary_cause"] = "顺序一致性不足"
    detail["failed_stage"] = "sequence"
    return IndicatorResult(status="不合格", reason="发力顺序：顺序一致性不足", detail=detail)


def _draw_pose33(frame_bgr: np.ndarray, lm: np.ndarray) -> None:
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    for i, j in _POSE_CONNECTIONS:
        if float(lm[i, 3]) < 0.3 or float(lm[j, 3]) < 0.3:
            continue
        ax, ay = int(float(lm[i, 0]) * w), int(float(lm[i, 1]) * h)
        bx, by = int(float(lm[j, 0]) * w), int(float(lm[j, 1]) * h)
        cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 140, 255), 2, cv2.LINE_AA)
    for idx in range(33):
        if float(lm[idx, 3]) < 0.3:
            continue
        x, y = int(float(lm[idx, 0]) * w), int(float(lm[idx, 1]) * h)
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)


def _status_tag(s: str) -> str:
    if s == "合格":
        return "PASS"
    if s == "不合格":
        return "FAIL"
    return "NA"


def _evaluate_from_arrays(
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
    fps = float(meta.get("fps") or 30.0)
    view_hint = (view_hint or "auto").lower()
    # valid_mask 契约（YOLO 迁移 Issue #5）：在视频级入口解析一次，按段切片下发。
    # `None` 时按 meta 的 `valid_conf_thr`（缺省 0.5）现场推导，逐位等价旧行为。
    mask_thr = float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR)
    full_mask = _resolve_mask(landmarks, valid_mask, mask_thr)

    if view_hint in ("front", "side"):
        if view_hint == "front":
            front_seg, side_seg = (0, int(landmarks.shape[0] - 1)), None
        else:
            front_seg, side_seg = None, (0, int(landmarks.shape[0] - 1))
        view_mode = "single"
    else:
        front_seg, side_seg, view_mode = split_front_side_segments(view_scores, fps)
        if view_mode == "single":
            single_view = _classify_single_view(view_scores, side_thr=0.9, front_thr=0.95)
            if single_view == "side":
                front_seg, side_seg = None, (0, int(landmarks.shape[0] - 1))
            elif single_view == "front":
                front_seg, side_seg = (0, int(landmarks.shape[0] - 1)), None

    front_lm = landmarks[slice(front_seg[0], front_seg[1] + 1)] if front_seg is not None else np.zeros((0, 33, 4), dtype=np.float32)
    side_lm = landmarks[slice(side_seg[0], side_seg[1] + 1)] if side_seg is not None else np.zeros((0, 33, 4), dtype=np.float32)
    front_mask = full_mask[slice(front_seg[0], front_seg[1] + 1)] if front_seg is not None else np.zeros((0, 33), dtype=bool)
    side_mask = full_mask[slice(side_seg[0], side_seg[1] + 1)] if side_seg is not None else np.zeros((0, 33), dtype=bool)

    # 重心评估：方案1+2（侧面优先）+ 方案3（CoM分段质心）
    cog_com: IndicatorResult | None = None
    if side_seg is not None:
        cog_side, cog_com = _eval_cog_side_prefer_punch_windows(side_lm, fps=fps, use_com=True, valid_mask=side_mask)
    else:
        cog_side = IndicatorResult(status="无法判定", reason="侧面段缺失")
        # 无侧面段时，全段使用CoM评估作为参考
        cog_com = eval_cog_com(landmarks, fps=fps, valid_mask=full_mask)
    if front_seg is not None:
        cog_front = eval_cog_front(front_lm, fps=fps, stance=str(stance), valid_mask=front_mask)
    else:
        cog_front = IndicatorResult(status="无法判定", reason="正面段缺失")
    cog_final = cog_side if cog_side.status != "无法判定" else cog_front

    retract = eval_retract_speed_side(side_lm, fps=fps, valid_mask=side_mask) if side_seg is not None else IndicatorResult(status="无法判定", reason="侧面段缺失")

    if side_seg is not None:
        wrist_src, wrist_mask = side_lm, side_mask
    else:
        wrist_src, wrist_mask = front_lm, front_mask
    wrist = eval_wrist_angle(wrist_src, fps=fps, valid_mask=wrist_mask) if wrist_src.size != 0 else IndicatorResult(status="无法判定", reason="无有效帧")

    force = eval_force_sequence(
        front_lm,
        side_lm,
        fps=fps,
        stance=str(stance),
        front_valid_mask=front_mask,
        side_valid_mask=side_mask,
    )

    # YOLO 迁移 S4 / Issue #11：装配层统一补齐结构化契约字段
    # （required_landmarks / missing_landmarks / backend），不改各指标内部判定。
    def _force_union_mask() -> np.ndarray | None:
        masks = [m for m in (front_mask, side_mask) if m is not None and m.shape[0] > 0]
        if not masks:
            return None
        return np.concatenate(masks, axis=0)

    cog_side = _attach_contract(cog_side, COG_REQUIRED_INDICES, mask=side_mask if side_seg is not None else None)
    cog_front = _attach_contract(cog_front, COG_REQUIRED_INDICES, mask=front_mask if front_seg is not None else None)
    # cog_final 跟随被采用的视角口径。
    cog_final = _attach_contract(
        cog_final,
        COG_REQUIRED_INDICES,
        mask=(side_mask if (side_seg is not None and cog_side.status != "无法判定") else (front_mask if front_seg is not None else None)),
    )
    if cog_com is not None:
        cog_com = _attach_contract(
            cog_com,
            COG_COM_REQUIRED_INDICES,
            mask=(side_mask if side_seg is not None else full_mask),
        )
    retract = _attach_contract(retract, RETRACT_REQUIRED_INDICES, mask=side_mask if side_seg is not None else None)
    wrist = _attach_contract(wrist, WRIST_REQUIRED_INDICES, mask=(wrist_mask if wrist_src.size != 0 else None))
    force = _attach_contract(force, FORCE_REQUIRED_INDICES, mask=_force_union_mask())

    if not keep_detail:
        def strip_detail(r: IndicatorResult) -> IndicatorResult:
            # 用 replace 仅清空 detail，保留 Issue #11 结构化契约字段。
            return replace(r, detail=None)

        cog_final = strip_detail(cog_final)
        cog_side = strip_detail(cog_side)
        cog_front = strip_detail(cog_front)
        if cog_com is not None:
            cog_com = strip_detail(cog_com)
        retract = strip_detail(retract)
        wrist = strip_detail(wrist)
        force = strip_detail(force)

    return TechEvalResult(
        video_path=str(meta.get("video") or ""),
        pose_variant=str(pose_variant),
        fps=float(fps),
        view_mode=str(view_mode),
        front_segment=None if front_seg is None else (int(front_seg[0]), int(front_seg[1])),
        side_segment=None if side_seg is None else (int(side_seg[0]), int(side_seg[1])),
        cog_final=cog_final,
        cog_side=cog_side,
        cog_front=cog_front,
        cog_com=cog_com,
        retract_speed=retract,
        force_sequence=force,
        wrist_angle=wrist,
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
    res = _evaluate_from_arrays(
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

    def strip(r: IndicatorResult) -> IndicatorResult:
        return replace(r, detail=None)

    return TechEvalResult(
        video_path=str(video_path),
        pose_variant=str(pose_variant),
        fps=float(detail.fps),
        view_mode=str(detail.view_mode),
        front_segment=detail.front_segment,
        side_segment=detail.side_segment,
        cog_final=strip(detail.cog_final),
        cog_side=strip(detail.cog_side),
        cog_front=strip(detail.cog_front),
        cog_com=None if detail.cog_com is None else strip(detail.cog_com),
        retract_speed=strip(detail.retract_speed),
        force_sequence=strip(detail.force_sequence),
        wrist_angle=strip(detail.wrist_angle),
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
    from video_writer import open_video_writer

    if res is None or landmarks is None or view_scores is None or meta is None:
        res, landmarks, view_scores, meta = evaluate_video_assets(video_path, pose_variant=pose_variant, stance=stance, view_hint=view_hint)
    fps = float((meta or {}).get("fps") or res.fps or 30.0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
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
    windows = _merge_intervals(windows)

    def in_window(i: int) -> bool:
        for s, e in windows:
            if int(s) <= int(i) <= int(e):
                return True
        return False

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i < int(landmarks.shape[0]):
            _draw_pose33(frame, landmarks[i])
        y0 = 22
        dy = 22
        lines = [
            f"COG:{_status_tag(res.cog_final.status)}  Retract:{_status_tag(res.retract_speed.status)}",
            f"ForceSeq:{_status_tag(res.force_sequence.status)}  Wrist:{_status_tag(res.wrist_angle.status)}",
        ]
        # 显示CoM评估结果（方案3）
        if res.cog_com is not None:
            lines.append(f"CoM(方案3):{_status_tag(res.cog_com.status)}")
        if np.isfinite(view_scores[i]) if i < int(view_scores.size) else False:
            lines.append(f"ViewScore:{float(view_scores[i]):.2f}")
        if in_window(i):
            lines.append("PUNCH_WINDOW")
        for k, t in enumerate(lines):
            cv2.putText(frame, t, (10, y0 + k * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, t, (10, y0 + k * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        vw.write(frame)
        i += 1

    vw.release()
    cap.release()
    return actual_path
