# -*- coding: utf-8 -*-
"""
YOLO adapter + COCO17 → Pose33-like 映射（YOLO 迁移 Issue #7 / S2）。

定位
----
本模块是 YOLO 进入主链路的**唯一入口**，把 ultralytics YOLO-pose 的 COCO17
关键点映射到 BlazePose33-like 容器。核心铁律（来自 AGENTS.md / 迁移计划）：

- **缺失点不得伪造成有效点**：COCO17 结构性缺失的点（嘴角 9/10、手指 17-22、
  脚跟脚尖 29-32、眼细分 1/3/4/6）在序列层一律 ``valid_mask=False``，在边界层
  一律 ``synthetic=True`` / ``visibility=0.0``。
- **不改变 MediaPipe 旧默认路径**：本模块独立存在，懒加载 ultralytics
  （``from ultralytics import YOLO`` 只在方法内部执行），未安装 ultralytics 时
  ``import core.yolo_adapter`` 不受影响，MediaPipe 路径与既有测试照常工作。

synthetic 信息的存放位置（写死契约，解决「序列热路径只返回 numpy」冲突）
----------------------------------------------------------------------
- **序列层**（``extract_yolo_landmark_series``）只返回 ``landmarks[T,33,4]`` +
  ``valid_mask[T,33]`` 的 numpy 数组；缺失点在序列层的唯一可见判据是
  ``valid_mask=False``。**不**把逐点 ``synthetic`` 标量塞进 ``(T,33,4)``。
- ``synthetic=True`` 属于**边界容器语义**，只存在于 adapter 的
  ``FrameResult`` / ``Landmark`` 层。
- 因此「缺失点是合成点」的断言限定在边界层测试；序列层只断言 ``valid_mask=False``。

confidence 语义
---------------
YOLO 的 keypoint confidence 写入第 4 通道，但标 ``confidence_kind="yolo_conf"``，
分布与 MediaPipe visibility 不同，**不可**共用同一阈值。

标定时序（重要）
----------------
YOLO 侧 ``valid_conf_thr`` 已在 **S3（#10）定为 0.6**（替换 #7 占位值 0.5）。
标定依据见 ``docs/yolo_body_core_calibration.md``：在该阈值下 YOLO body_core_v1
与 MediaPipe body_core_v1 的跨视频一致性最好，且 body_core 有效帧率仍 >= 0.87。
但跨视频 go/no-go 未通过，结论为「仅预览 / 内部标定参考」，不进 full tech_eval。
输出 meta 仍标 ``calibration_status="unvalidated"``，不得作为对外评分。

tracker 段边界语义
------------------
单视频 / 单 segment 内 tracker 状态不跨任务复用。``extract_yolo_landmark_series``
在每段开始处调用 ``adapter.reset_tracker()``，因此 **track_id 在段边界必然重置**。
完整跨段 track 延续 / 中心最近 / tie-break 等多人鲁棒策略本期不做。

多人场景闸门（Issue #9 / S2）
----------------------------
``extract_yolo_landmark_series`` 暴露每帧检出人数并据此判定多人闸门：
检出 ``num_persons>1`` 时 meta 标 ``multi_person_detected=True`` /
``review_required=True`` / ``gate_status="multi_person_review_required"``。
判定逻辑集中在纯函数 ``evaluate_multi_person_gate``。**闸门本身只判定 + 写 meta，
拒绝/降级出分由下游评分入口执行**（``body_core_compare`` 闭环遇 ``review_required``
直接拒绝出分）。单人（``max_persons<=1``）才走「最大框/最高分取单人」。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.foundation.paths import models_dir
from core.pose.backends.yolo_mapping import (  # noqa: F401  （映射契约，经本模块转出以保持旧导入路径）
    DEFAULT_YOLO_VALID_CONF_THR,
    YOLO_CONFIDENCE_KIND,
    YOLO_VALIDITY_POLICY,
    YOLO_CALIBRATION_STATUS,
    YOLO_RAW_LAYOUT,
    YOLO_FEATURE_LAYOUT,
    YOLO_CAPABILITY,
    YOLO_DISPLAY_SCOPE_LIMITED,
    YOLO_DISPLAY_SCOPE_INTERNAL,
    YOLO_REALTIME_TARGET_POLICY,
    DEFAULT_YOLO_MODEL_NAME,
    TRACK_RESET_NOTE,
    GATE_STATUS_OK,
    GATE_STATUS_MULTI_PERSON,
    MULTI_PERSON_GATE_NOTE,
    CALIBRATION_NOTE,
    COCO17_TO_BLAZE33,
    BLAZE33_MISSING_IN_COCO17,
    NUM_BLAZE33,
    NUM_COCO17,
    BODY_CORE_V1_VALID_INDICES,
    Landmark,
    FrameResult,
    map_coco17_to_blaze33,
    coco17_to_landmarks,
    map_coco17_person,
    boundary_meta,
    yolo_authorization_meta,
    empty_frame_result,
    empty_sequence_row,
    evaluate_multi_person_gate,
    extract_persons,
    select_main_person,
    yolo_result_to_arrays,
    yolo_result_to_frame,
    SingleTargetTracker,
)


def _device_supports_half(device: str) -> bool:
    value = str(device).strip().lower()
    if value in {"cpu", "mps"}:
        return False
    return value.startswith("cuda") or value.isdigit()


def _normalize_warmup_shape(
    warmup_shape: tuple[int, int] | tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    if warmup_shape is None:
        return None
    shape = tuple(int(v) for v in warmup_shape)
    if len(shape) == 2:
        h, w = shape
        return (max(1, h), max(1, w), 3)
    if len(shape) == 3:
        h, w, c = shape
        return (max(1, h), max(1, w), max(1, c))
    raise ValueError("warmup_shape must be (height, width) or (height, width, channels)")


# --------------------------------------------------------------------------- #
# YOLO adapter（懒加载 ultralytics）
# --------------------------------------------------------------------------- #
class YoloPoseAdapter:
    """轻量 YOLO-pose adapter：懒加载 ultralytics，逐帧产出 BlazePose33-like 结果。

    懒加载约束
    ----------
    ``from ultralytics import YOLO`` 只在 ``_load()`` 内执行，因此
    ``import core.yolo_adapter`` 不需要 ultralytics，也不会影响 MediaPipe 路径
    或既有测试（ultralytics 缺失时本模块仍可正常 import）。

    单人选择
    --------
    本期 = 最大框 / 最高分取单人（``select_main_person``）。``num_persons`` 暴露给
    每帧结果，多人闸门由序列层 ``evaluate_multi_person_gate`` 判定并要求下游拒绝/降级。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
        confidence_kind: str = YOLO_CONFIDENCE_KIND,
        imgsz: int = 640,
        device: str = "cpu",
        half: bool = False,
        warmup: bool = False,
        warmup_shape: tuple[int, int] | tuple[int, int, int] | None = None,
        max_missed: int = 5,
    ) -> None:
        if model_path is None:
            model_path = models_dir() / DEFAULT_YOLO_MODEL_NAME
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        # S3（#10）落库 valid_conf_thr=0.6，但仍不授权对外评分。
        self.valid_conf_thr = float(valid_conf_thr)
        self.confidence_kind = str(confidence_kind)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self.requested_half = bool(half)
        self.half = bool(half) and _device_supports_half(self.device)
        self.warmup = bool(warmup)
        self.warmup_shape = _normalize_warmup_shape(warmup_shape)
        self._warmup_done = False
        self._model: Any = None
        self._tracker = SingleTargetTracker(max_missed=max_missed)
        # 最近一帧检出人数（供调试 / Issue #9 闸门读取）。
        self.last_num_persons: int = 0

    # -- 懒加载 ---------------------------------------------------------------
    def _load(self) -> Any:
        """懒加载 YOLO 模型。``from ultralytics import YOLO`` 仅在此执行。"""
        if self._model is None:
            from ultralytics import YOLO  # noqa: WPS433 (intentional lazy import)

            self._model = YOLO(str(self.model_path))
        return self._model

    # -- tracker 段边界控制 ---------------------------------------------------
    def reset_tracker(self) -> None:
        """重置 tracker 状态（段边界 / 新视频）。track_id 不跨段复用。"""
        self._tracker.reset()
        self.last_num_persons = 0

    # -- 推理 -----------------------------------------------------------------
    def _predict_raw(self, frame: np.ndarray) -> Any:
        model = self._load()
        results = model.predict(
            frame,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        return results[0] if results else None

    def _ensure_warmup(self, frame: np.ndarray) -> None:
        if not self.warmup or self._warmup_done:
            return
        if self.warmup_shape is None:
            shape = tuple(int(v) for v in frame.shape[:3])
        else:
            shape = self.warmup_shape
        dummy = np.zeros(shape, dtype=frame.dtype if hasattr(frame, "dtype") else np.uint8)
        self._predict_raw(dummy)
        self._warmup_done = True

    def _predict(self, frame: np.ndarray) -> Any:
        self._ensure_warmup(frame)
        return self._predict_raw(frame)

    def infer_arrays(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int | None]:
        """序列热路径：返回 numpy ``(row[33,4], valid[33], num_persons, track_id)``。

        不构造 33 个冻结对象，直接摊平成 numpy（遵守「序列层只返回 numpy」契约）。
        """
        res = self._predict(frame)
        row, valid, num_persons, _idx = yolo_result_to_arrays(
            res, valid_conf_thr=self.valid_conf_thr
        )
        self.last_num_persons = num_persons
        track_id = self._tracker.update(num_persons >= 1)
        return row, valid, num_persons, track_id

    def infer_frame(self, frame: np.ndarray) -> FrameResult:
        """边界层：返回单帧 ``FrameResult``（含 ``num_persons`` / ``track_id``）。"""
        res = self._predict(frame)
        persons = extract_persons(res)
        num_persons = len(persons)
        self.last_num_persons = num_persons
        track_id = self._tracker.update(num_persons >= 1)
        if num_persons == 0:
            return empty_frame_result(
                valid_conf_thr=self.valid_conf_thr, num_persons=0, track_id=track_id
            )
        idx = select_main_person(persons)
        _row, _valid, frame_result = map_coco17_person(
            persons[idx]["xy"],
            persons[idx]["conf"],
            valid_conf_thr=self.valid_conf_thr,
            track_id=track_id,
            num_persons=num_persons,
            selected_index=idx,
        )
        return frame_result

    def annotate(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """实时预览路径：YOLO body-only 推理 + 轻量骨架绘制。

        ``timestamp_ms`` 只为兼容 MediaPipe pipeline 接口；YOLO 单帧推理不依赖它。
        """
        del timestamp_ms
        result = self.infer_frame(frame_bgr)
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        pose33 = result.pose33
        actions = _classify_yolo_body_actions(pose33, w, h)
        _draw_yolo_body(out, pose33, w, h)

        meta = dict(result.meta or {})
        person_count = int(meta.get("num_persons", self.last_num_persons) or 0)
        gate = evaluate_multi_person_gate([person_count])
        meta.update(
            {
                "backend": "yolo",
                "model_name": self.model_name,
                "running_mode": "realtime_preview",
                "multi_person_detected": bool(gate["multi_person_detected"]),
                "multi_person_frames": int(gate["multi_person_frames"]),
                "person_count": person_count,
                "review_required": bool(gate["review_required"]),
                "gate_status": str(gate["gate_status"]),
                "gate_note": str(gate["gate_note"]),
                "target_policy": YOLO_REALTIME_TARGET_POLICY,
                "track_id": result.track_id,
            }
        )
        return out, actions, meta


def _landmark_confidence(lm: Landmark) -> float:
    if lm.synthetic:
        return 0.0
    if lm.confidence is not None:
        return float(lm.confidence)
    return float(lm.visibility)


def _yolo_pt2d(lm: Landmark, w: int, h: int) -> np.ndarray:
    return np.array([float(lm.x) * w, float(lm.y) * h], dtype=np.float32)


def _yolo_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _classify_yolo_body_actions(
    landmarks: tuple[Landmark, ...] | None,
    w: int,
    h: int,
) -> list[str]:
    if landmarks is None:
        return []

    l_sh = landmarks[11]
    r_sh = landmarks[12]
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    actions: list[str] = []
    min_conf = DEFAULT_YOLO_VALID_CONF_THR

    left_hand_up = (
        _landmark_confidence(l_wrist) >= min_conf
        and _landmark_confidence(l_sh) >= min_conf
        and l_wrist.y < l_sh.y - 0.05
    )
    right_hand_up = (
        _landmark_confidence(r_wrist) >= min_conf
        and _landmark_confidence(r_sh) >= min_conf
        and r_wrist.y < r_sh.y - 0.05
    )
    if left_hand_up and right_hand_up:
        actions.append("HANDS_UP")
    elif left_hand_up:
        actions.append("LEFT_HAND_UP")
    elif right_hand_up:
        actions.append("RIGHT_HAND_UP")

    def knee_ok(hip_idx: int, knee_idx: int, ankle_idx: int) -> bool:
        hip = landmarks[hip_idx]
        knee = landmarks[knee_idx]
        ankle = landmarks[ankle_idx]
        if min(_landmark_confidence(hip), _landmark_confidence(knee), _landmark_confidence(ankle)) < min_conf:
            return False
        hip_pt = _yolo_pt2d(hip, w, h)
        knee_pt = _yolo_pt2d(knee, w, h)
        ankle_pt = _yolo_pt2d(ankle, w, h)
        angle = _yolo_angle_deg(hip_pt, knee_pt, ankle_pt)
        hip_knee_dist = abs((hip_pt[1] - knee_pt[1]) / max(h, 1))
        return angle < 125.0 and hip_knee_dist < 0.18

    if knee_ok(23, 25, 27) and knee_ok(24, 26, 28):
        actions.append("SQUAT")
    return actions


def _draw_yolo_body(
    out_bgr: np.ndarray,
    landmarks: tuple[Landmark, ...] | None,
    w: int,
    h: int,
) -> None:
    if landmarks is None:
        return
    import cv2

    body_edges = (
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
    min_conf = DEFAULT_YOLO_VALID_CONF_THR
    for i, j in body_edges:
        a, b = landmarks[i], landmarks[j]
        if min(_landmark_confidence(a), _landmark_confidence(b)) < min_conf:
            continue
        ax, ay = int(a.x * w), int(a.y * h)
        bx, by = int(b.x * w), int(b.y * h)
        cv2.line(out_bgr, (ax, ay), (bx, by), (255, 170, 0), 2, cv2.LINE_AA)
    for idx in BODY_CORE_V1_VALID_INDICES:
        lm = landmarks[idx]
        if _landmark_confidence(lm) < min_conf:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(out_bgr, (x, y), 3, (0, 220, 255), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# 序列提取层（只返回 numpy 数组 + meta）
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _cancelled_as_interrupted():
    """解码层的 ``Cancelled`` 转回本函数对外约定的 ``InterruptedError``。"""
    from core.foundation.tasks import Cancelled

    try:
        yield
    except Cancelled:
        raise InterruptedError("YOLO landmark extraction cancelled") from None


def _resolve_adapter(yolo_model, **options) -> Any:
    """已构造的 adapter（鸭子类型：暴露 infer_arrays/reset_tracker）直接复用；否则按路径/None 构造。"""
    if hasattr(yolo_model, "infer_arrays") and hasattr(yolo_model, "reset_tracker"):
        return yolo_model
    return YoloPoseAdapter(model_path=yolo_model, **options)


@dataclass
class _SeriesRows:
    rows: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    num_persons: list = field(default_factory=list)
    track_ids: list = field(default_factory=list)
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0


def _read_series(video_path, adapter, start_i: int, end_frame: int | None, should_stop) -> _SeriesRows:
    """逐帧推理 ``[start_i, end]``；取消抛 ``InterruptedError``。"""
    from core.foundation.tasks import from_stopped
    from core.media.video_source import NO_END, DecodeOptions, open_video

    def check_stop() -> None:
        if should_stop is not None and should_stop():
            raise InterruptedError("YOLO landmark extraction cancelled")

    out = _SeriesRows()
    check_stop()
    source = open_video(video_path, DecodeOptions(start=start_i), from_stopped(should_stop))
    with source, _cancelled_as_interrupted():
        m = source.meta
        out.fps, out.frame_count, out.width, out.height = m.fps, m.frame_count, m.width, m.height
        end_i = int(end_frame) if end_frame is not None else (m.frame_count - 1 if m.frame_count > 0 else NO_END)
        end_i = max(start_i, end_i)
        for frame in source:
            if frame.index > end_i:
                break
            check_stop()
            row, valid, num_persons, track_id = adapter.infer_arrays(frame.bgr)
            check_stop()
            out.rows.append(row)
            out.masks.append(valid)
            out.num_persons.append(int(num_persons))
            out.track_ids.append(track_id)
            if frame.index >= end_i:
                break
    return out


def _series_meta(video_path, adapter, data: _SeriesRows, *, start_i: int, running_mode: str,
                 imgsz: int, device: str) -> dict:
    # 多人闸门判定：max_persons>1 → multi_person_detected + review_required。
    # 本期只判定并写 meta；拒绝/降级出分由下游评分入口（body_core 闭环 / batch）执行。
    gate = evaluate_multi_person_gate(data.num_persons)
    return {
        **yolo_authorization_meta(display_scope=YOLO_DISPLAY_SCOPE_INTERNAL),
        "video": str(video_path),
        "backend": "yolo",
        "model_name": adapter.model_name,
        "running_mode": str(running_mode),
        "imgsz": int(getattr(adapter, "imgsz", imgsz)),
        "device": str(getattr(adapter, "device", device)),
        "half": bool(getattr(adapter, "half", False)),
        "requested_half": bool(getattr(adapter, "requested_half", False)),
        "warmup": bool(getattr(adapter, "warmup", False)),
        "warmup_shape": getattr(adapter, "warmup_shape", None),
        "confidence_kind": adapter.confidence_kind,
        "validity_policy": YOLO_VALIDITY_POLICY,
        # S3（#10）落库 valid_conf_thr=0.6，但未授权对外评分。
        "valid_conf_thr": float(adapter.valid_conf_thr),
        "calibration_note": CALIBRATION_NOTE,
        "frame_count": int(data.frame_count),
        "fps": float(data.fps),
        "width": int(data.width),
        "height": int(data.height),
        "start_frame": int(start_i),
        "end_frame": int(start_i + len(data.rows) - 1) if data.rows else int(start_i),
        "layout": "pose33_normalized_xyzw(yolo_conf)",
        # 多人闸门：检出 num_persons>1 即标记并要求下游拒绝/降级，不静默选最大框。
        "num_persons_per_frame": data.num_persons,
        "max_persons": int(max(data.num_persons)) if data.num_persons else 0,
        "multi_person_frames": int(sum(1 for n in data.num_persons if n > 1)),
        "multi_person_detected": gate["multi_person_detected"],
        "gate_status": gate["gate_status"],
        "review_required": gate["review_required"],
        "gate_note": gate["gate_note"],
        "track_ids": data.track_ids,
        "track_policy": "single_target_largest_box; per-segment reset",
        "track_reset_note": TRACK_RESET_NOTE,
    }


def extract_yolo_landmark_series(
    video_path: str | Path,
    *,
    yolo_model: str | Path | YoloPoseAdapter | None = None,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
    start_frame: int | None = None,
    end_frame: int | None = None,
    imgsz: int = 640,
    device: str = "cpu",
    half: bool = False,
    warmup: bool = False,
    warmup_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    running_mode: str = "video",
    should_stop=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """提取 YOLO body-only 关键点序列。

    返回 ``(landmarks[T,33,4], valid_mask[T,33], meta)``：
      - **只返回 numpy 数组**，不逐帧返回冻结 ``Landmark``/``FrameResult`` 列表；
        缺失点在序列层的唯一信号是 ``valid_mask=False``（不携带逐点 synthetic 标量）。
      - 无人帧 / 空结果优雅降级为零行 + 全 False mask（不抛异常）。

    参数
    ----
    yolo_model:
        模型路径、或已构造的 ``YoloPoseAdapter``（便于测试注入 fake adapter）。
        ``None`` 时按默认路径 ``models/yolo11n-pose.pt`` 构造 adapter。
    valid_conf_thr:
        S3（#10）落库阈值 0.6；结论仍为仅预览 / 内部标定参考。YOLO-only 不进 full tech_eval。
    """
    adapter = _resolve_adapter(yolo_model, valid_conf_thr=valid_conf_thr, imgsz=imgsz, device=device,
                               half=half, warmup=warmup, warmup_shape=warmup_shape)
    # 段边界：tracker 状态不跨任务/段复用，每段开始处显式重置。
    adapter.reset_tracker()
    start_i = max(0, int(start_frame) if start_frame is not None else 0)
    data = _read_series(video_path, adapter, start_i, end_frame, should_stop)
    if data.rows:
        landmarks = np.stack(data.rows, axis=0).astype(np.float32)
        valid_mask = np.stack(data.masks, axis=0).astype(bool)
    else:
        # 空视频 / 全部超出范围：优雅返回空序列，不抛异常。
        landmarks = np.zeros((0, NUM_BLAZE33, 4), dtype=np.float32)
        valid_mask = np.zeros((0, NUM_BLAZE33), dtype=bool)
    meta = _series_meta(video_path, adapter, data, start_i=start_i, running_mode=running_mode,
                        imgsz=imgsz, device=device)
    return landmarks, valid_mask, meta
