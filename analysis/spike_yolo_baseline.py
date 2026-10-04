# -*- coding: utf-8 -*-
r"""S0b 决策 Spike：YOLO vs MediaPipe 基线对照脚本（临时脚本，不接主链路）。

用途
====
对照 issue #2（S0b）：用一份独立脚本跑通 YOLO-pose 与 MediaPipe-pose 在同一批
样本上的关键点提取对照，导出 keypoints、track_id、FPS、漏检帧率、关键点抖动、
初始化耗时等客观数据，
供 ``docs/yolo_baseline_report.md`` 的预注册阈值表对照。

强约束（来自迁移计划 / AGENTS.md）
================================
- 本脚本 **不得被主链路 import**，**不得修改主代码**。它只读视频、跑两套后端、
  写 JSON/CSV 到 ``outputs/spike/``。
- YOLO COCO17 → BlazePose33 的映射严格遵循 ``docs/yolo_migration_plan_optimized.md``
  的映射表：缺失点（嘴角 9/10、手指 17-22、脚跟脚尖 29-32、眼细分 1/3/4/6）一律
  置 ``valid=False``，**绝不伪造成有效点**。
- 抖动/相关性等"对照"指标只在 **两套后端都判定有效** 的关键点上计算，避免拿合成
  点污染统计。

运行示例
========
::

    .\.venv\Scripts\python.exe -m analysis.spike_yolo_baseline \
        --samples docs/yolo_eval_samples.json \
        --yolo-model models/yolo11n-pose.pt \
        --pose-variant full \
        --out outputs/spike

也可直接对单个视频快速验证：

::

    .\.venv\Scripts\python.exe -m analysis.spike_yolo_baseline \
        --video "标准样本/正面.mp4" --yolo-model models/yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# 降低原生日志噪声，保持与主链路一致的安静度。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2  # noqa: E402


# ---------------------------------------------------------------------------
# COCO17 → BlazePose33 映射（与迁移计划映射表一致）
# ---------------------------------------------------------------------------
# 键：BlazePose33 索引；值：COCO17 索引。仅列出有真实对应的点。
COCO17_TO_BLAZE33: dict[int, int] = {
    0: 0,    # nose
    2: 1,    # left_eye  → BlazePose 2（近似）
    5: 2,    # right_eye → BlazePose 5（近似）
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

# COCO17 在 BlazePose33 中结构性缺失、不可用的点（绝不伪造）。
BLAZE33_MISSING_IN_COCO17: tuple[int, ...] = (
    1, 3, 4, 6,           # 眼睛细分点
    9, 10,                # 嘴角
    17, 18, 19, 20, 21, 22,  # 手指
    29, 30, 31, 32,       # 脚跟、脚尖
)

# body_core_v1 关心的 12 个躯干四肢核心点（用于"对照"统计聚焦）。
BODY_CORE_V1_INDICES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
YOLO_ARTIFACT_METADATA: dict[str, Any] = {
    "backend": "yolo",
    "raw_layout": "pose33_like_coco17",
    "feature_layout": "body_core_v1",
    "capability": "body_only",
    "score_authorized": False,
    "calibration_status": "unvalidated",
    "display_scope": "internal",
}


@dataclass
class BackendRun:
    """单后端在单个视频上的提取结果与统计。"""

    backend: str
    model_name: str
    n_frames: int = 0
    n_detected: int = 0          # 至少检出 1 人/1 套 pose 的帧数
    init_sec: float = 0.0
    infer_sec: float = 0.0       # 纯推理累计耗时（不含解码）
    wall_sec: float = 0.0        # 端到端累计耗时（含解码+推理+映射）
    # landmarks[T,33,4] = (x_norm, y_norm, z, conf/visibility)；valid_mask[T,33]
    landmarks: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    multi_person_frames: int = 0
    max_persons: int = 0
    person_counts: list[int] = field(default_factory=list)
    track_ids: list[int | None] = field(default_factory=list)
    selected_indices: list[int | None] = field(default_factory=list)
    boxes_xywh_norm: list[list[float] | None] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fps_infer(self) -> float:
        return float(self.n_frames / self.infer_sec) if self.infer_sec > 0 else 0.0

    @property
    def fps_wall(self) -> float:
        return float(self.n_frames / self.wall_sec) if self.wall_sec > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        if self.n_frames <= 0:
            return 1.0
        return float(1.0 - self.n_detected / self.n_frames)


# ---------------------------------------------------------------------------
# MediaPipe 基线提取
# ---------------------------------------------------------------------------
def run_mediapipe(video_path: Path, *, pose_variant: str, models_dir: Path) -> BackendRun:
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    pose_path = models_dir / f"pose_landmarker_{pose_variant}.task"
    if not pose_path.exists():
        cap.release()
        raise FileNotFoundError(f"缺少 MediaPipe pose 模型：{pose_path}")

    t_init = time.perf_counter()
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    init_sec = time.perf_counter() - t_init

    run = BackendRun(backend="mediapipe", model_name=f"pose_landmarker_{pose_variant}", init_sec=init_sec)
    lms: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_wall = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(i * 1000.0 / fps)

        t_inf = time.perf_counter()
        res = landmarker.detect_for_video(mp_image, ts)
        run.infer_sec += time.perf_counter() - t_inf

        arr = np.zeros((33, 4), dtype=np.float32)
        mask = np.zeros((33,), dtype=bool)
        pose_landmarks = res.pose_landmarks[0] if getattr(res, "pose_landmarks", None) else None
        if pose_landmarks is not None:
            run.n_detected += 1
            for j in range(min(33, len(pose_landmarks))):
                lm = pose_landmarks[j]
                vis = float(getattr(lm, "visibility", 0.0))
                arr[j] = (float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0)), vis)
                mask[j] = vis >= 0.5
        run.person_counts.append(1 if pose_landmarks is not None else 0)
        run.track_ids.append(None)
        run.selected_indices.append(0 if pose_landmarks is not None else None)
        run.boxes_xywh_norm.append(None)
        lms.append(arr)
        masks.append(mask)
        run.wall_sec += time.perf_counter() - t_wall
        i += 1

    cap.release()
    landmarker.close()
    run.n_frames = i
    run.landmarks = np.stack(lms, axis=0) if lms else np.zeros((0, 33, 4), np.float32)
    run.valid_mask = np.stack(masks, axis=0) if masks else np.zeros((0, 33), bool)
    run.max_persons = 1
    run.extra["confidence_kind"] = "mp_visibility"
    run.extra["validity_policy"] = "visibility_thr"
    run.extra["valid_conf_thr"] = 0.5
    run.extra["fps"] = fps
    return run


@dataclass
class SimpleTrackAssigner:
    """Spike-only selected-target tracker for recording a reproducible track_id."""

    next_id: int = 1
    active_id: int | None = None
    active_box: np.ndarray | None = None
    missed: int = 0
    max_missed: int = 5
    max_center_dist: float = 0.18
    min_iou: float = 0.05

    def update(self, box_xywh_norm: list[float] | None) -> int | None:
        if box_xywh_norm is None:
            self.missed += 1
            if self.missed > self.max_missed:
                self.active_id = None
                self.active_box = None
            return None

        box = np.asarray(box_xywh_norm, dtype=np.float32)
        if self.active_id is None or self.active_box is None:
            return self._start_track(box)

        center_dist = float(np.linalg.norm(box[:2] - self.active_box[:2]))
        iou = _xywh_iou(self.active_box, box)
        if center_dist > self.max_center_dist and iou < self.min_iou:
            return self._start_track(box)

        self.active_box = box
        self.missed = 0
        return self.active_id

    def _start_track(self, box: np.ndarray) -> int:
        self.active_id = self.next_id
        self.next_id += 1
        self.active_box = box
        self.missed = 0
        return self.active_id


# ---------------------------------------------------------------------------
# YOLO 基线提取
# ---------------------------------------------------------------------------
def run_yolo(
    video_path: Path,
    *,
    yolo_model: str,
    valid_conf_thr: float,
    imgsz: int = 640,
) -> BackendRun:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    t_init = time.perf_counter()
    model = YOLO(yolo_model)
    init_sec = time.perf_counter() - t_init

    run = BackendRun(backend="yolo", model_name=Path(yolo_model).name, init_sec=init_sec)
    lms: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    track_assigner = SimpleTrackAssigner()

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_wall = time.perf_counter()
        t_inf = time.perf_counter()
        results = model.predict(frame, imgsz=imgsz, verbose=False, device="cpu")
        run.infer_sec += time.perf_counter() - t_inf

        arr = np.zeros((33, 4), dtype=np.float32)
        mask = np.zeros((33,), dtype=bool)
        n_persons = 0
        selected_idx: int | None = None
        selected_box_xywh_norm: list[float] | None = None
        res = results[0] if results else None
        kpts = getattr(res, "keypoints", None) if res is not None else None
        if kpts is not None and kpts.data is not None and len(kpts.data) > 0:
            n_persons = int(len(kpts.data))
            # 单人 MVP：取面积最大的框对应实例（多人闸门统计另记）。
            best = _select_main_person(res)
            selected_idx = best
            selected_box_xywh_norm = _box_xywh_norm(res, best, w, h)
            xy = kpts.xyn[best].cpu().numpy() if kpts.xyn is not None else None   # (17,2) 归一化
            conf = kpts.conf[best].cpu().numpy() if kpts.conf is not None else None  # (17,)
            if xy is not None and xy.shape[0] >= 17:
                run.n_detected += 1
                for blaze_idx, coco_idx in COCO17_TO_BLAZE33.items():
                    c = float(conf[coco_idx]) if conf is not None else 0.0
                    arr[blaze_idx] = (float(xy[coco_idx, 0]), float(xy[coco_idx, 1]), 0.0, c)
                    mask[blaze_idx] = c >= valid_conf_thr
                # 缺失点：坐标保持 0，valid 强制 False（不伪造）。
                for miss in BLAZE33_MISSING_IN_COCO17:
                    mask[miss] = False
        if n_persons > 1:
            run.multi_person_frames += 1
        run.max_persons = max(run.max_persons, n_persons)
        run.person_counts.append(n_persons)
        run.selected_indices.append(selected_idx)
        run.boxes_xywh_norm.append(selected_box_xywh_norm)
        run.track_ids.append(track_assigner.update(selected_box_xywh_norm))
        lms.append(arr)
        masks.append(mask)
        run.wall_sec += time.perf_counter() - t_wall
        i += 1

    cap.release()
    run.n_frames = i
    run.landmarks = np.stack(lms, axis=0) if lms else np.zeros((0, 33, 4), np.float32)
    run.valid_mask = np.stack(masks, axis=0) if masks else np.zeros((0, 33), bool)
    run.extra["confidence_kind"] = "yolo_conf"
    run.extra["validity_policy"] = "confidence_thr"
    run.extra["valid_conf_thr"] = valid_conf_thr  # 待标定占位值（S3 标定前不得对外评分）
    run.extra["calibration_status"] = "unvalidated"
    run.extra["fps"] = fps
    run.extra["width"] = w
    run.extra["height"] = h
    run.extra["track_policy"] = "spike_selected_largest_box_iou_center"
    run.extra["track_note"] = "Spike-only track_id for data audit; production multi-person gate is S2."
    return run


def _select_main_person(res: Any) -> int:
    """单人 MVP：从多实例里选主目标（按检测框面积最大）。多人正确性风险在 S2 处理。"""
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xywh is None or len(boxes.xywh) == 0:
        return 0
    xywh = boxes.xywh.cpu().numpy()
    areas = xywh[:, 2] * xywh[:, 3]
    return int(np.argmax(areas))


def _box_xywh_norm(res: Any, idx: int, width: int, height: int) -> list[float] | None:
    boxes = getattr(res, "boxes", None)
    if boxes is None or boxes.xywh is None or len(boxes.xywh) <= idx or width <= 0 or height <= 0:
        return None
    xywh = boxes.xywh[idx].cpu().numpy().astype(np.float32)
    return [
        round(float(xywh[0] / width), 6),
        round(float(xywh[1] / height), 6),
        round(float(xywh[2] / width), 6),
        round(float(xywh[3] / height), 6),
    ]


def _xywh_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _xywh_to_xyxy(box: np.ndarray) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box[:4]]
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


# ---------------------------------------------------------------------------
# 对照统计：抖动 / 漏检 / 共同有效点相关性
# ---------------------------------------------------------------------------
def temporal_jitter(run: BackendRun, indices: tuple[int, ...]) -> float | None:
    """逐帧位移（归一化坐标）的中位数，仅在该点连续两帧都有效时计入。

    抖动越小代表关键点越稳定。返回 None 表示样本不足。
    """
    if run.landmarks is None or run.valid_mask is None or run.n_frames < 2:
        return None
    lm = run.landmarks
    mask = run.valid_mask
    diffs: list[float] = []
    for idx in indices:
        for t in range(1, lm.shape[0]):
            if mask[t, idx] and mask[t - 1, idx]:
                d = float(np.linalg.norm(lm[t, idx, :2] - lm[t - 1, idx, :2]))
                if np.isfinite(d):
                    diffs.append(d)
    if not diffs:
        return None
    return float(np.median(diffs))


def cross_backend_position_diff(
    mp_run: BackendRun, yolo_run: BackendRun, indices: tuple[int, ...]
) -> dict[str, Any]:
    """两后端在 **共同有效** 关键点上的逐帧位置差（归一化坐标）。"""
    out: dict[str, Any] = {"compared_points": 0, "median_diff": None, "p90_diff": None}
    if mp_run.landmarks is None or yolo_run.landmarks is None:
        return out
    n = min(mp_run.landmarks.shape[0], yolo_run.landmarks.shape[0])
    if n <= 0:
        return out
    diffs: list[float] = []
    for t in range(n):
        for idx in indices:
            if bool(mp_run.valid_mask[t, idx]) and bool(yolo_run.valid_mask[t, idx]):
                d = float(np.linalg.norm(mp_run.landmarks[t, idx, :2] - yolo_run.landmarks[t, idx, :2]))
                if np.isfinite(d):
                    diffs.append(d)
    if diffs:
        out["compared_points"] = len(diffs)
        out["median_diff"] = float(np.median(diffs))
        out["p90_diff"] = float(np.percentile(diffs, 90))
    return out


# ---------------------------------------------------------------------------
# 环境信息
# ---------------------------------------------------------------------------
def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    for pkg in ("numpy", "cv2", "mediapipe", "ultralytics", "torch"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            env[pkg] = f"<not available: {exc}>"
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
        env["torch_device"] = "cpu"
    return env


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def process_video(
    video_path: Path,
    *,
    pose_variant: str,
    yolo_model: str,
    models_dir: Path,
    yolo_conf_thr: float,
    keypoints_jsonl: Path | None = None,
) -> dict[str, Any]:
    print(f"[spike] 处理：{video_path}", flush=True)
    mp_run = run_mediapipe(video_path, pose_variant=pose_variant, models_dir=models_dir)
    yolo_run = run_yolo(video_path, yolo_model=yolo_model, valid_conf_thr=yolo_conf_thr)

    body_core = BODY_CORE_V1_INDICES
    yolo_meta = {
        **YOLO_ARTIFACT_METADATA,
        "model": yolo_run.model_name,
        "fps_infer": round(yolo_run.fps_infer, 3),
        "fps_wall": round(yolo_run.fps_wall, 3),
        "miss_rate": round(yolo_run.miss_rate, 4),
        "init_sec": round(yolo_run.init_sec, 3),
        "jitter_body_core": _round_or_none(temporal_jitter(yolo_run, body_core)),
        "multi_person_frames": yolo_run.multi_person_frames,
        "max_persons": yolo_run.max_persons,
        "track_policy": yolo_run.extra.get("track_policy"),
    }
    rec: dict[str, Any] = {
        "video": str(video_path),
        "frames": mp_run.n_frames,
        "mediapipe": {
            "model": mp_run.model_name,
            "fps_infer": round(mp_run.fps_infer, 3),
            "fps_wall": round(mp_run.fps_wall, 3),
            "miss_rate": round(mp_run.miss_rate, 4),
            "init_sec": round(mp_run.init_sec, 3),
            "jitter_body_core": _round_or_none(temporal_jitter(mp_run, body_core)),
        },
        "yolo": yolo_meta,
        "cross_backend_body_core_diff": cross_backend_position_diff(mp_run, yolo_run, body_core),
        "fps_speedup_infer": _round_or_none(
            yolo_run.fps_infer / mp_run.fps_infer if mp_run.fps_infer > 0 else None
        ),
    }
    if keypoints_jsonl is not None:
        _append_keypoints_jsonl(keypoints_jsonl, video_path, mp_run, yolo_run)
        rec["keypoints_export"] = str(keypoints_jsonl)
    return rec


def _round_or_none(x: float | None, ndigits: int = 4) -> float | None:
    return None if x is None else round(float(x), ndigits)


def _append_keypoints_jsonl(path: Path, video_path: Path, mp_run: BackendRun, yolo_run: BackendRun) -> None:
    """Append one JSON line per frame for S0 audit and later calibration replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(mp_run.n_frames, yolo_run.n_frames)
    video_id = _stable_video_id(video_path)
    with path.open("a", encoding="utf-8") as fh:
        for t in range(n):
            rec = {
                "schema_version": 1,
                "video_id": video_id,
                "video": str(video_path),
                "frame_index": t,
                "mediapipe": _frame_export(mp_run, t),
                "yolo": _frame_export(yolo_run, t),
            }
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stable_video_id(video_path: Path) -> str:
    return hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:12]


def _frame_export(run: BackendRun, frame_index: int) -> dict[str, Any]:
    has_frame = run.landmarks is not None and frame_index < run.landmarks.shape[0]
    person_count = run.person_counts[frame_index] if frame_index < len(run.person_counts) else 0
    rec = {
        "backend": run.backend,
        "model": run.model_name,
        "detected": bool(person_count > 0),
        "person_count": int(person_count),
        "selected_person_index": run.selected_indices[frame_index]
        if frame_index < len(run.selected_indices) else None,
        "track_id": run.track_ids[frame_index] if frame_index < len(run.track_ids) else None,
        "bbox_xywh_norm": run.boxes_xywh_norm[frame_index]
        if frame_index < len(run.boxes_xywh_norm) else None,
        "keypoints_blaze33": _rounded_landmarks(run.landmarks[frame_index]) if has_frame else [],
        "valid_mask": run.valid_mask[frame_index].astype(bool).tolist()
        if run.valid_mask is not None and frame_index < run.valid_mask.shape[0] else [],
        "confidence_kind": run.extra.get("confidence_kind"),
        "validity_policy": run.extra.get("validity_policy"),
        "valid_conf_thr": run.extra.get("valid_conf_thr"),
    }
    if run.backend == "yolo":
        rec.update(YOLO_ARTIFACT_METADATA)
    else:
        rec["calibration_status"] = run.extra.get("calibration_status")
    return rec


def _rounded_landmarks(arr: np.ndarray) -> list[list[float]]:
    return np.round(arr.astype(np.float32), 6).tolist()


def load_samples(samples_json: Path) -> list[Path]:
    data = json.loads(samples_json.read_text(encoding="utf-8"))
    items = data.get("samples", []) if isinstance(data, dict) else data
    paths: list[Path] = []
    base = samples_json.resolve().parent.parent  # 仓库根（docs/ 的上一级）
    for it in items:
        p = it["path"] if isinstance(it, dict) else str(it)
        pp = Path(p)
        if not pp.is_absolute():
            pp = base / pp
        paths.append(pp)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S0b YOLO vs MediaPipe 基线对照 spike（不接主链路）")
    ap.add_argument("--samples", type=str, default=None, help="样本清单 JSON（docs/yolo_eval_samples.json）")
    ap.add_argument("--video", type=str, default=None, help="单个视频路径（与 --samples 二选一）")
    ap.add_argument("--yolo-model", type=str, default="models/yolo11n-pose.pt")
    ap.add_argument("--pose-variant", type=str, default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--yolo-conf-thr", type=float, default=0.5, help="YOLO 有效性占位阈值（待 S3 标定）")
    ap.add_argument("--out", type=str, default="outputs/spike")
    ap.add_argument(
        "--no-keypoints-export",
        action="store_true",
        help="不写出 spike_keypoints.jsonl（默认写出 YOLO/MediaPipe keypoints 与 track_id）",
    )
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    models_dir = (repo_root / args.models_dir).resolve()

    if args.video:
        videos = [Path(args.video)]
    elif args.samples:
        videos = load_samples(Path(args.samples))
    else:
        ap.error("必须提供 --samples 或 --video 之一")
        return 2

    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keypoints_jsonl = None if args.no_keypoints_export else out_dir / "spike_keypoints.jsonl"
    if keypoints_jsonl is not None and keypoints_jsonl.exists():
        keypoints_jsonl.unlink()

    env = collect_env()
    print(f"[spike] 环境：{json.dumps(env, ensure_ascii=False)}", flush=True)

    records: list[dict[str, Any]] = []
    for v in videos:
        if not v.exists():
            print(f"[spike] 跳过（文件不存在）：{v}", flush=True)
            continue
        try:
            records.append(
                process_video(
                    v,
                    pose_variant=args.pose_variant,
                    yolo_model=str((repo_root / args.yolo_model).resolve()),
                    models_dir=models_dir,
                    yolo_conf_thr=args.yolo_conf_thr,
                    keypoints_jsonl=keypoints_jsonl,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[spike] 处理失败：{v} -> {exc}", flush=True)
            records.append({"video": str(v), "error": str(exc)})

    summary = _summarize(records)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": env,
        "yolo_model": args.yolo_model,
        "pose_variant": args.pose_variant,
        "yolo_conf_thr": args.yolo_conf_thr,
        "note": "YOLO valid_conf_thr 为待标定占位值；YOLO 分数 calibration_status=unvalidated，不得对外评分。",
        "summary": summary,
        "records": records,
    }

    json_path = out_dir / "spike_baseline.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "spike_baseline.csv", records)
    print(f"[spike] 已写出：{json_path}", flush=True)
    if keypoints_jsonl is not None:
        print(f"[spike] 已写出 keypoints：{keypoints_jsonl}", flush=True)
    print(f"[spike] 汇总：{json.dumps(summary, ensure_ascii=False, indent=2)}", flush=True)
    return 0


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"valid_videos": 0}

    def _avg(getter) -> float | None:
        vals = [getter(r) for r in ok if getter(r) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "valid_videos": len(ok),
        "mp_fps_infer_avg": _avg(lambda r: r["mediapipe"]["fps_infer"]),
        "yolo_fps_infer_avg": _avg(lambda r: r["yolo"]["fps_infer"]),
        "fps_speedup_infer_avg": _avg(lambda r: r.get("fps_speedup_infer")),
        "mp_miss_rate_avg": _avg(lambda r: r["mediapipe"]["miss_rate"]),
        "yolo_miss_rate_avg": _avg(lambda r: r["yolo"]["miss_rate"]),
        "mp_jitter_body_core_avg": _avg(lambda r: r["mediapipe"]["jitter_body_core"]),
        "yolo_jitter_body_core_avg": _avg(lambda r: r["yolo"]["jitter_body_core"]),
        "cross_diff_median_avg": _avg(lambda r: r["cross_backend_body_core_diff"]["median_diff"]),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    cols = [
        "video", "frames",
        "mp_fps_infer", "yolo_fps_infer", "fps_speedup_infer",
        "mp_miss_rate", "yolo_miss_rate",
        "mp_jitter_body_core", "yolo_jitter_body_core",
        "cross_diff_median", "cross_diff_p90",
        "yolo_multi_person_frames", "yolo_max_persons",
        "yolo_backend", "yolo_raw_layout", "yolo_feature_layout", "yolo_capability",
        "yolo_score_authorized", "yolo_calibration_status", "yolo_display_scope",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            if "error" in r:
                w.writerow([r.get("video", ""), "ERROR", r["error"]])
                continue
            cb = r["cross_backend_body_core_diff"]
            w.writerow([
                r["video"], r["frames"],
                r["mediapipe"]["fps_infer"], r["yolo"]["fps_infer"], r.get("fps_speedup_infer"),
                r["mediapipe"]["miss_rate"], r["yolo"]["miss_rate"],
                r["mediapipe"]["jitter_body_core"], r["yolo"]["jitter_body_core"],
                cb["median_diff"], cb["p90_diff"],
                r["yolo"]["multi_person_frames"], r["yolo"]["max_persons"],
                r["yolo"]["backend"], r["yolo"]["raw_layout"], r["yolo"]["feature_layout"],
                r["yolo"]["capability"], r["yolo"]["score_authorized"],
                r["yolo"]["calibration_status"], r["yolo"]["display_scope"],
            ])


if __name__ == "__main__":
    raise SystemExit(main())
