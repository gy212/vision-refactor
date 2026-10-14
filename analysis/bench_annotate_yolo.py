# -*- coding: utf-8 -*-
"""S5 标注帧率基准的 YOLO 被测对象：逐帧推理、绘制 body_core 骨架与可选 MediaPipe 手部，并统计漏检、有效率与抖动。

YOLO 只作基准对照（``body_core_v1``、未标定），不进入正式评分。只供 ``analysis/bench_annotate_fps.py`` 使用。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analysis.bench_annotate_metrics import round_or_none
from core.pose import draw_hands
from core.pose.backends.yolo import (
    BODY_CORE_V1_VALID_INDICES,
    FrameResult,
    YoloPoseAdapter,
    evaluate_multi_person_gate,
)

BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
)


class YoloPreviewAnnotator:
    def __init__(
        self,
        *,
        model_path: Path,
        models_dir: Path,
        device: str,
        enable_hands: bool,
        imgsz: int,
        half: bool,
        adapter_warmup: bool,
        warmup_shape: tuple[int, int, int] | None,
        valid_conf_thr: float,
    ) -> None:
        self.adapter = YoloPoseAdapter(
            model_path=model_path,
            valid_conf_thr=valid_conf_thr,
            device=device,
            imgsz=imgsz,
            half=half,
            warmup=adapter_warmup,
            warmup_shape=warmup_shape,
        )
        self.enable_hands = bool(enable_hands)
        self.hand_landmarker: Any = None
        self.reset_counters()
        if self.enable_hands:
            import mediapipe as mp

            hand_path = models_dir / "hand_landmarker.task"
            if not hand_path.exists():
                raise FileNotFoundError(f"Missing hand landmarker model: {hand_path}")
            self._mp = mp
            self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=2,
                )
            )

    def reset_counters(self) -> None:
        self.frames = 0
        self.yolo_raw_infer_sec = 0.0
        self.yolo_raw_infer_latencies_sec: list[float] = []
        self.last_yolo_raw_infer_sec: float | None = None
        self.detected_frames = 0
        self.body_core_full_valid_frames = 0
        self._body_core_rows: list[np.ndarray | None] = []
        self.num_persons_per_frame: list[int] = []

    def _timed_infer(self, frame_bgr: np.ndarray) -> FrameResult:
        t_infer = time.perf_counter()
        frame_result = self.adapter.infer_frame(frame_bgr)
        raw_sec = time.perf_counter() - t_infer
        self.last_yolo_raw_infer_sec = raw_sec
        self.yolo_raw_infer_sec += raw_sec
        self.yolo_raw_infer_latencies_sec.append(raw_sec)
        self.frames += 1
        return frame_result

    def annotate(self, frame_bgr: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        out = frame_bgr.copy()
        frame_result = self._timed_infer(frame_bgr)
        meta = frame_result.meta or {}
        num_persons = int(meta.get("num_persons") or 0)
        self.num_persons_per_frame.append(num_persons)
        if frame_result.pose33 is not None and num_persons > 0:
            self.detected_frames += 1
        body_core = body_core_xy_from_frame(frame_result, self.adapter.valid_conf_thr)
        if body_core is not None:
            self.body_core_full_valid_frames += 1
        self._body_core_rows.append(body_core)

        draw_yolo_body(out, frame_result, self.adapter.valid_conf_thr)
        if self.hand_landmarker is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            hand_res = self.hand_landmarker.detect_for_video(mp_image, int(timestamp_ms))
            hands = hand_res.hand_landmarks if getattr(hand_res, "hand_landmarks", None) else []
            draw_hands(out, hands)
        return out

    def reset_metrics(self) -> None:
        self.reset_counters()
        self.adapter.reset_tracker()

    def warmup_raw_infer(self, frame_bgr: np.ndarray) -> float:
        self._timed_infer(frame_bgr)
        return float(self.last_yolo_raw_infer_sec)

    def close(self) -> None:
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()

    def metrics(self) -> dict[str, Any]:
        miss_rate = 1.0 - (self.detected_frames / self.frames) if self.frames > 0 else None
        full_valid_rate = (
            self.body_core_full_valid_frames / self.frames if self.frames > 0 else None
        )
        gate = evaluate_multi_person_gate(self.num_persons_per_frame)
        return {
            "yolo_raw_infer_sec": round(self.yolo_raw_infer_sec, 4),
            "yolo_raw_infer_fps": (
                round(self.frames / self.yolo_raw_infer_sec, 3)
                if self.yolo_raw_infer_sec > 0 else 0.0
            ),
            "yolo_miss_rate": round_or_none(miss_rate),
            "yolo_body_core_full_valid_rate": round_or_none(full_valid_rate),
            "yolo_body_core_missing_rate": round_or_none(
                1.0 - full_valid_rate if full_valid_rate is not None else None
            ),
            "yolo_body_core_jitter_median": round_or_none(
                body_core_jitter_median(self._body_core_rows)
            ),
            "max_persons": max(self.num_persons_per_frame) if self.num_persons_per_frame else 0,
            "multi_person_frames": int(sum(1 for n in self.num_persons_per_frame if n > 1)),
            "review_required": bool(gate["review_required"]),
            "gate_status": gate["gate_status"],
        }


def _valid_body_core_indices(frame_result: FrameResult, valid_conf_thr: float) -> list[int]:
    """body_core 12 点中非合成且置信度达标的索引（按 ``BODY_CORE_V1_VALID_INDICES`` 顺序）。"""
    pose = frame_result.pose33
    valid: list[int] = []
    for idx in BODY_CORE_V1_VALID_INDICES:
        lm = pose[idx]
        conf = lm.confidence if lm.confidence is not None else lm.visibility
        if not (lm.synthetic or conf < valid_conf_thr):
            valid.append(idx)
    return valid


def body_core_xy_from_frame(frame_result: FrameResult, valid_conf_thr: float) -> np.ndarray | None:
    """12 点全部有效时返回 ``(12, 2)`` 坐标，否则 None。"""
    pose = frame_result.pose33
    if not pose:
        return None
    valid = _valid_body_core_indices(frame_result, valid_conf_thr)
    if len(valid) != len(BODY_CORE_V1_VALID_INDICES):
        return None
    return np.asarray([(float(pose[i].x), float(pose[i].y)) for i in valid], dtype=np.float32)


def body_core_jitter_median(rows: list[np.ndarray | None]) -> float | None:
    diffs: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for row in rows:
        if row is None:
            prev = None
            continue
        if prev is not None and prev.shape == row.shape:
            diffs.append(np.linalg.norm(row - prev, axis=1))
        prev = row
    if not diffs:
        return None
    return float(np.median(np.concatenate(diffs)))


def empty_yolo_metrics() -> dict[str, Any]:
    return {
        "yolo_raw_infer_sec": None,
        "yolo_raw_infer_fps": None,
        "yolo_miss_rate": None,
        "yolo_body_core_full_valid_rate": None,
        "yolo_body_core_missing_rate": None,
        "yolo_body_core_jitter_median": None,
        "max_persons": None,
        "multi_person_frames": None,
        "review_required": None,
        "gate_status": None,
    }


def draw_yolo_body(out_bgr: np.ndarray, frame_result: FrameResult, valid_conf_thr: float) -> None:
    pose = frame_result.pose33
    if not pose:
        return
    h, w = out_bgr.shape[:2]
    valid_indices = set(_valid_body_core_indices(frame_result, valid_conf_thr))
    for i, j in BODY_CONNECTIONS:
        if i not in valid_indices or j not in valid_indices:
            continue
        a, b = pose[i], pose[j]
        cv2.line(
            out_bgr,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
    for idx in valid_indices:
        lm = pose[idx]
        cv2.circle(out_bgr, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1, cv2.LINE_AA)
