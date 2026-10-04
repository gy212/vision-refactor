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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .paths import models_dir

# --------------------------------------------------------------------------- #
# 常量：YOLO 侧 confidence / validity 策略
# --------------------------------------------------------------------------- #
# YOLO 侧 valid_conf_thr 已在 S3（#10）定为 0.6（替换 #7 占位值 0.5）。
# 标定依据：docs/yolo_body_core_calibration.md（thr=0.6 时 YOLO↔MediaPipe body_core
# 跨视频相关性 0.98、MAE 0.047、有效帧率 >= 0.87）。刻意与 MediaPipe 的
# DEFAULT_VALID_CONF_THR 分开定义，避免把 YOLO keypoint confidence 当成 MediaPipe
# visibility 复用同一阈值。S3 结论仍为仅预览 / 内部标定参考，YOLO-only 不进 full tech_eval。
DEFAULT_YOLO_VALID_CONF_THR: float = 0.6

YOLO_CONFIDENCE_KIND: str = "yolo_conf"
YOLO_VALIDITY_POLICY: str = "confidence_thr"
# S3（#10）已落 valid_conf_thr=0.6，但跨视频 go/no-go 未通过；仍不得对外评分。
YOLO_CALIBRATION_STATUS: str = "unvalidated"
YOLO_RAW_LAYOUT: str = "pose33_like_coco17"
YOLO_FEATURE_LAYOUT: str = "body_core_v1"
YOLO_CAPABILITY: str = "body_only"
YOLO_DISPLAY_SCOPE_LIMITED: str = "limited"
YOLO_DISPLAY_SCOPE_INTERNAL: str = "internal"
YOLO_REALTIME_TARGET_POLICY: str = "select_main_person_largest_box_highest_score"

DEFAULT_YOLO_MODEL_NAME: str = "yolo11n-pose.pt"

# 段边界 track 重置说明（写入 meta，供下游与 #9 多人扩展参考）。
TRACK_RESET_NOTE: str = (
    "tracker state is NOT reused across videos/segments; track_id resets at each "
    "segment boundary (extract_yolo_landmark_series calls reset_tracker() per call). "
    "Cross-segment track continuation (center-nearest / tie-break) is deferred."
)

# --------------------------------------------------------------------------- #
# 多人场景闸门（YOLO 迁移 Issue #9 / S2）
# --------------------------------------------------------------------------- #
# 背景：batch_tech_eval / batch_dual_compare 跑的是学员视频，教练、镜面反射、路人入镜
# 常见（S0 spike 已实证：学员样本最多检出 8 人、单视频 274 帧多人）。YOLO「最大框/最高
# 分取单人」可能稳定选错实例——这是**正确性风险**，不是鲁棒性优化。MVP 必须能识别并
# 拒绝/降级，**不允许静默选最大框**。
#
# 本期策略（MVP）：
#   - 单人（max_persons<=1）：才走 select_main_person 的「最大框/最高分取单人」。
#   - 多人（max_persons>1）：meta 标 multi_person_detected=True，闸门判 review_required，
#     下游（body_core 闭环 / batch）必须拒绝出分或降级「需人工复核」，不混入正常评分。
#   - 完整多人鲁棒策略（中心最近、tie-break、track 延续）本期不做。
GATE_STATUS_OK: str = "ok"
GATE_STATUS_MULTI_PERSON: str = "multi_person_review_required"
MULTI_PERSON_GATE_NOTE: str = (
    "multi-person frames detected (num_persons>1); 'largest box' single-person "
    "selection may stably pick the wrong instance. This is a CORRECTNESS risk: the "
    "video is flagged review_required and MUST NOT enter outward-facing scoring. "
    "Full multi-person robustness (center-nearest / tie-break / track continuation) "
    "is out of scope this round (Issue #9 MVP)."
)
# 标定状态说明（写入 meta）。
CALIBRATION_NOTE: str = (
    "YOLO valid_conf_thr is set to 0.6 in S3 (#10), but cross-video go/no-go "
    "criteria did not pass. Scores remain preview/internal calibration only and "
    "MUST NOT enter outward-facing scoring. YOLO-only does NOT enter full tech_eval "
    "(COCO17 lacks mouth/heel/foot-index/fingers; see S0 degradation list)."
)


# --------------------------------------------------------------------------- #
# COCO17 → BlazePose33 映射（与 analysis/spike_yolo_baseline.py 已验证映射一致）
# --------------------------------------------------------------------------- #
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

# COCO17 在 BlazePose33 中结构性缺失、不可用的点（绝不伪造成有效点）。
BLAZE33_MISSING_IN_COCO17: tuple[int, ...] = (
    1, 3, 4, 6,              # 眼睛细分点
    9, 10,                   # 嘴角
    17, 18, 19, 20, 21, 22,  # 手指
    29, 30, 31, 32,          # 脚跟、脚尖
)

NUM_BLAZE33: int = 33
NUM_COCO17: int = 17

# body_core_v1（12,2）布局在 BlazePose33 中的源索引（肩/肘/腕 11..16 + 髋/膝/踝 23..28）。
# 这 12 个点**全部落在 COCO17 可映射点**（见 COCO17_TO_BLAZE33），因此 YOLO 路径下
# body_core_v1 是真实点（非合成缺失点）。供 body_core 闭环按 valid_mask 统计有效帧。
# 与 core.feature_layout.BODY_CORE_V1.source_indices 保持一致（此处冗余定义仅为避免
# yolo_adapter 反向依赖 feature_layout / 触发循环 import；定义不一致由测试守卫）。
BODY_CORE_V1_VALID_INDICES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


# --------------------------------------------------------------------------- #
# 边界层结果容器（frozen dataclasses）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Landmark:
    """边界层单点容器。

    ``synthetic=True`` 仅在此边界层出现，绝不进入序列层 ``(T,33,4)`` 数组。
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    confidence: float | None = None
    synthetic: bool = False


@dataclass(frozen=True)
class FrameResult:
    """边界层单帧容器（服务实时/单帧调试/后端适配输出，不进序列热路径）。"""

    pose33: tuple[Landmark, ...] | None
    hands: tuple[tuple[Landmark, ...], ...] = ()
    track_id: int | None = None
    meta: dict | None = None


# --------------------------------------------------------------------------- #
# 纯映射：单个 YOLO person 的 COCO17 → BlazePose33
# --------------------------------------------------------------------------- #
def map_coco17_to_blaze33(
    xy: np.ndarray,
    conf: np.ndarray | None,
    *,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
) -> tuple[np.ndarray, np.ndarray]:
    """把单人 COCO17（归一化 xy + conf）映射成序列层 numpy 行。

    参数
    ----
    xy:
        ``(17, 2)`` 归一化坐标（YOLO ``keypoints.xyn``）。
    conf:
        ``(17,)`` keypoint confidence（YOLO ``keypoints.conf``）；``None`` 视作全 0。
    valid_conf_thr:
        YOLO 有效性阈值（S3 #10 当前落库值为 0.6；仍不授权对外评分）。

    返回
    ----
    ``(row, valid)``：
      - ``row``  : ``(33, 4)`` float32 ``(x_norm, y_norm, z=0.0, conf)``，
                   只填映射到的 COCO 点，其余保持 0。
      - ``valid``: ``(33,)`` bool，映射点 ``conf >= valid_conf_thr`` 才为 True；
                   所有 ``BLAZE33_MISSING_IN_COCO17`` 强制 False（不伪造）。
    """
    xy_arr = np.asarray(xy, dtype=np.float32)
    if xy_arr.ndim != 2 or xy_arr.shape[0] < NUM_COCO17 or xy_arr.shape[1] < 2:
        raise ValueError(
            f"map_coco17_to_blaze33 期望 xy 形状 (17,2)，实际 {xy_arr.shape}"
        )
    conf_arr = None if conf is None else np.asarray(conf, dtype=np.float32).reshape(-1)

    row = np.zeros((NUM_BLAZE33, 4), dtype=np.float32)
    valid = np.zeros((NUM_BLAZE33,), dtype=bool)

    for blaze_idx, coco_idx in COCO17_TO_BLAZE33.items():
        c = float(conf_arr[coco_idx]) if conf_arr is not None else 0.0
        row[blaze_idx, 0] = float(xy_arr[coco_idx, 0])
        row[blaze_idx, 1] = float(xy_arr[coco_idx, 1])
        row[blaze_idx, 2] = 0.0
        row[blaze_idx, 3] = c
        valid[blaze_idx] = c >= float(valid_conf_thr)

    # 缺失点：坐标保持 0，valid 强制 False（不伪造）。
    for miss in BLAZE33_MISSING_IN_COCO17:
        valid[miss] = False

    return row, valid


def coco17_to_landmarks(
    xy: np.ndarray,
    conf: np.ndarray | None,
) -> tuple[Landmark, ...]:
    """把单人 COCO17 映射成边界层 ``(33,)`` Landmark 元组。

    - 映射点：``confidence=conf``、``synthetic=False``、``visibility=0.0``。
    - 缺失点：``synthetic=True``、``visibility=0.0``、``confidence=None``，坐标置 0。
    """
    xy_arr = np.asarray(xy, dtype=np.float32)
    conf_arr = None if conf is None else np.asarray(conf, dtype=np.float32).reshape(-1)

    landmarks: list[Landmark | None] = [None] * NUM_BLAZE33

    for blaze_idx, coco_idx in COCO17_TO_BLAZE33.items():
        c = float(conf_arr[coco_idx]) if conf_arr is not None else 0.0
        landmarks[blaze_idx] = Landmark(
            x=float(xy_arr[coco_idx, 0]),
            y=float(xy_arr[coco_idx, 1]),
            z=0.0,
            visibility=0.0,
            confidence=c,
            synthetic=False,
        )

    for miss in BLAZE33_MISSING_IN_COCO17:
        landmarks[miss] = Landmark(
            x=0.0, y=0.0, z=0.0, visibility=0.0, confidence=None, synthetic=True
        )

    # 17 映射点 + 16 缺失点 = 33，全部覆盖；防御性校验。
    assert all(lm is not None for lm in landmarks), "BlazePose33 未被完整填充"
    return tuple(landmarks)  # type: ignore[arg-type]


def map_coco17_person(
    xy: np.ndarray,
    conf: np.ndarray | None,
    *,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
    track_id: int | None = None,
    num_persons: int = 1,
    selected_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, FrameResult]:
    """单人 COCO17 → (序列行, 有效行, 边界容器) 三件套。

    返回 ``(row[33,4], valid[33], FrameResult)``：
      - ``row`` / ``valid`` 是序列层 numpy 信号；
      - ``FrameResult`` 是边界层容器，缺失点 ``synthetic=True``、``visibility=0.0``，
        映射点 ``confidence=conf``、``synthetic=False``。
    """
    row, valid = map_coco17_to_blaze33(xy, conf, valid_conf_thr=valid_conf_thr)
    pose33 = coco17_to_landmarks(xy, conf)
    frame = FrameResult(
        pose33=pose33,
        hands=(),
        track_id=track_id,
        meta=_boundary_meta(
            valid_conf_thr=valid_conf_thr,
            num_persons=num_persons,
            selected_index=selected_index,
        ),
    )
    return row, valid, frame


def _boundary_meta(
    *,
    valid_conf_thr: float,
    num_persons: int,
    selected_index: int | None,
    display_scope: str = YOLO_DISPLAY_SCOPE_LIMITED,
) -> dict:
    return {
        **_yolo_authorization_meta(display_scope=display_scope),
        "backend": "yolo",
        "confidence_kind": YOLO_CONFIDENCE_KIND,
        "validity_policy": YOLO_VALIDITY_POLICY,
        "valid_conf_thr": float(valid_conf_thr),
        "calibration_note": CALIBRATION_NOTE,
        "num_persons": int(num_persons),
        "selected_index": selected_index,
        "track_reset_note": TRACK_RESET_NOTE,
    }


def _yolo_authorization_meta(*, display_scope: str) -> dict:
    scope = str(display_scope or YOLO_DISPLAY_SCOPE_LIMITED)
    if scope not in {YOLO_DISPLAY_SCOPE_LIMITED, YOLO_DISPLAY_SCOPE_INTERNAL}:
        scope = YOLO_DISPLAY_SCOPE_LIMITED
    return {
        "raw_layout": YOLO_RAW_LAYOUT,
        "feature_layout": YOLO_FEATURE_LAYOUT,
        "capability": YOLO_CAPABILITY,
        "score_authorized": False,
        "calibration_status": YOLO_CALIBRATION_STATUS,
        "display_scope": scope,
    }


def empty_frame_result(
    *,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
    num_persons: int = 0,
    track_id: int | None = None,
) -> FrameResult:
    """无人帧 / 空结果的边界容器：``pose33=None``。"""
    return FrameResult(
        pose33=None,
        hands=(),
        track_id=track_id,
        meta=_boundary_meta(
            valid_conf_thr=valid_conf_thr,
            num_persons=num_persons,
            selected_index=None,
        ),
    )


def empty_sequence_row() -> tuple[np.ndarray, np.ndarray]:
    """无人帧 / 空结果的序列行：零行 + 全 False mask。"""
    return np.zeros((NUM_BLAZE33, 4), dtype=np.float32), np.zeros((NUM_BLAZE33,), dtype=bool)


# --------------------------------------------------------------------------- #
# 多人场景闸门判定（纯函数，便于下游与测试直接调用）
# --------------------------------------------------------------------------- #
def evaluate_multi_person_gate(
    num_persons_per_frame: "list[int] | tuple[int, ...] | np.ndarray",
) -> dict:
    """根据每帧检出人数判定多人闸门状态。

    参数
    ----
    num_persons_per_frame:
        每帧检出人数序列（``extract_yolo_landmark_series`` 的 ``num_persons_per_frame``）。

    返回
    ----
    dict，含：
      - ``multi_person_detected`` (bool)：是否存在 ``num_persons>1`` 的帧。
      - ``max_persons`` (int)：整段最大检出人数。
      - ``multi_person_frames`` (int)：``num_persons>1`` 的帧数。
      - ``gate_status`` (str)：``GATE_STATUS_OK`` 或 ``GATE_STATUS_MULTI_PERSON``。
      - ``review_required`` (bool)：是否必须降级人工复核 / 拒绝出分。
      - ``gate_note`` (str)：多人时附说明，单人时为空串。

    语义铁律：``review_required=True`` 的视频**不得进入对外评分**，下游必须拒绝出分
    或降级「需人工复核」，**不允许静默选最大框当唯一目标**。
    """
    counts = [int(n) for n in num_persons_per_frame] if num_persons_per_frame is not None else []
    max_persons = int(max(counts)) if counts else 0
    multi_person_frames = int(sum(1 for n in counts if n > 1))
    multi_person_detected = max_persons > 1
    gate_status = GATE_STATUS_MULTI_PERSON if multi_person_detected else GATE_STATUS_OK
    return {
        "multi_person_detected": multi_person_detected,
        "max_persons": max_persons,
        "multi_person_frames": multi_person_frames,
        "gate_status": gate_status,
        "review_required": bool(multi_person_detected),
        "gate_note": MULTI_PERSON_GATE_NOTE if multi_person_detected else "",
    }


# --------------------------------------------------------------------------- #
# YOLO 结果对象解析（容忍 torch tensor 或 numpy；便于测试注入 fake result）
# --------------------------------------------------------------------------- #
def _to_numpy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def extract_persons(res: Any) -> list[dict[str, Any]]:
    """从单帧 YOLO 结果对象解析出每个 person 的 COCO17 keypoints 与选择评分。

    返回 person 列表，每项含：``xy``(17,2)、``conf``(17,)、``area``(框面积)、
    ``score``(框置信度或关键点均值置信度)。无检出返回空列表。
    """
    if res is None:
        return []
    kpts = getattr(res, "keypoints", None)
    if kpts is None:
        return []
    xyn = _to_numpy(getattr(kpts, "xyn", None))
    if xyn is None or len(xyn) == 0:
        return []
    kconf = _to_numpy(getattr(kpts, "conf", None))

    boxes = getattr(res, "boxes", None)
    box_xywh = _to_numpy(getattr(boxes, "xywh", None)) if boxes is not None else None
    box_conf = _to_numpy(getattr(boxes, "conf", None)) if boxes is not None else None

    persons: list[dict[str, Any]] = []
    n = int(xyn.shape[0])
    for i in range(n):
        xy_i = np.asarray(xyn[i], dtype=np.float32)
        conf_i = (
            np.asarray(kconf[i], dtype=np.float32).reshape(-1)
            if kconf is not None and i < len(kconf)
            else None
        )
        if box_xywh is not None and i < len(box_xywh):
            w = float(box_xywh[i][2])
            h = float(box_xywh[i][3])
            area = w * h
        else:
            area = 0.0
        if box_conf is not None and i < len(box_conf):
            score = float(box_conf[i])
        elif conf_i is not None:
            score = float(np.mean(conf_i))
        else:
            score = 0.0
        persons.append({"xy": xy_i, "conf": conf_i, "area": area, "score": score})
    return persons


def select_main_person(persons: list[dict[str, Any]]) -> int:
    """单人 MVP：取面积最大的框（无框时取置信度最高）。

    **仅在单人场景（``num_persons<=1``）才应被信任**——多人时由多人闸门
    （``evaluate_multi_person_gate``）拒绝/降级出分，不允许静默用本函数选最大框当
    唯一目标。完整多人鲁棒选择（中心最近 / tie-break / track 延续）本期不做。
    """
    if not persons:
        raise ValueError("persons 为空，无法选择主目标")
    areas = np.array([p.get("area", 0.0) for p in persons], dtype=np.float64)
    if float(np.max(areas)) > 0.0:
        return int(np.argmax(areas))
    scores = np.array([p.get("score", 0.0) for p in persons], dtype=np.float64)
    return int(np.argmax(scores))


def yolo_result_to_arrays(
    res: Any,
    *,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
) -> tuple[np.ndarray, np.ndarray, int, int | None]:
    """单帧 YOLO 结果 → ``(row[33,4], valid[33], num_persons, selected_index)``。

    无检出（无人帧 / 空结果）→ 零行 + 全 False mask + ``num_persons=0`` + ``None``。
    """
    persons = extract_persons(res)
    num_persons = len(persons)
    if num_persons == 0:
        row, valid = empty_sequence_row()
        return row, valid, 0, None
    idx = select_main_person(persons)
    row, valid = map_coco17_to_blaze33(
        persons[idx]["xy"], persons[idx]["conf"], valid_conf_thr=valid_conf_thr
    )
    return row, valid, num_persons, idx


def yolo_result_to_frame(
    res: Any,
    *,
    valid_conf_thr: float = DEFAULT_YOLO_VALID_CONF_THR,
    track_id: int | None = None,
) -> FrameResult:
    """单帧 YOLO 结果 → 边界层 ``FrameResult``（无检出时 ``pose33=None``）。"""
    persons = extract_persons(res)
    num_persons = len(persons)
    if num_persons == 0:
        return empty_frame_result(valid_conf_thr=valid_conf_thr, num_persons=0, track_id=track_id)
    idx = select_main_person(persons)
    _row, _valid, frame = map_coco17_person(
        persons[idx]["xy"],
        persons[idx]["conf"],
        valid_conf_thr=valid_conf_thr,
        track_id=track_id,
        num_persons=num_persons,
        selected_index=idx,
    )
    return frame


# --------------------------------------------------------------------------- #
# 单目标 tracker（段内连续；段边界重置）
# --------------------------------------------------------------------------- #
@dataclass
class SingleTargetTracker:
    """段内单目标 track_id 分配器。

    语义（本期单人 MVP，足够且可测）：
      - 检出主目标时：若当前无 active track，则分配新 id（``next_id``++）；否则沿用。
      - 连续未检出超过 ``max_missed`` 帧：丢弃 active track（下次检出会分配新 id）。
      - 未检出帧本身 track_id 记为 ``None``（该帧无人可跟）。

    **段边界语义**：``reset()`` 把 ``next_id`` 归 1、清空 active 状态。
    ``extract_yolo_landmark_series`` 每段开始处调用它，因此 track_id 不跨段复用、
    必然在段边界重置。完整跨段 track 延续是 Issue #9。
    """

    next_id: int = 1
    active_id: int | None = None
    missed: int = 0
    max_missed: int = 5

    def reset(self) -> None:
        self.next_id = 1
        self.active_id = None
        self.missed = 0

    def update(self, detected: bool) -> int | None:
        if detected:
            if self.active_id is None:
                self.active_id = self.next_id
                self.next_id += 1
            self.missed = 0
            return self.active_id
        # 未检出
        self.missed += 1
        if self.active_id is not None and self.missed > self.max_missed:
            self.active_id = None
        return None


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
    import cv2  # 局部 import 与既有 extract_pose_raw 风格保持一致，避免顶层耦合。

    # 已构造的 adapter（鸭子类型：暴露 infer_arrays/reset_tracker 即可，便于测试注入
    # fake adapter）直接复用；否则按路径/None 构造真实 YoloPoseAdapter。
    if hasattr(yolo_model, "infer_arrays") and hasattr(yolo_model, "reset_tracker"):
        adapter = yolo_model
    else:
        adapter = YoloPoseAdapter(
            model_path=yolo_model,
            valid_conf_thr=valid_conf_thr,
            imgsz=imgsz,
            device=device,
            half=half,
            warmup=warmup,
            warmup_shape=warmup_shape,
        )

    # 段边界：tracker 状态不跨任务/段复用，每段开始处显式重置。
    adapter.reset_tracker()

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

    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    num_persons_per_frame: list[int] = []
    track_ids: list[int | None] = []

    i = 0
    try:
        while True:
            if should_stop is not None and should_stop():
                raise InterruptedError("YOLO landmark extraction cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            if start_i <= i <= end_i:
                if should_stop is not None and should_stop():
                    raise InterruptedError("YOLO landmark extraction cancelled")
                row, valid, num_persons, track_id = adapter.infer_arrays(frame)
                if should_stop is not None and should_stop():
                    raise InterruptedError("YOLO landmark extraction cancelled")
                rows.append(row)
                masks.append(valid)
                num_persons_per_frame.append(int(num_persons))
                track_ids.append(track_id)
            if i >= end_i:
                break
            i += 1
    finally:
        cap.release()

    if rows:
        landmarks = np.stack(rows, axis=0).astype(np.float32)
        valid_mask = np.stack(masks, axis=0).astype(bool)
    else:
        # 空视频 / 全部超出范围：优雅返回空序列，不抛异常。
        landmarks = np.zeros((0, NUM_BLAZE33, 4), dtype=np.float32)
        valid_mask = np.zeros((0, NUM_BLAZE33), dtype=bool)

    multi_person_frames = int(sum(1 for n in num_persons_per_frame if n > 1))
    max_persons = int(max(num_persons_per_frame)) if num_persons_per_frame else 0

    # 多人闸门判定：max_persons>1 → multi_person_detected + review_required。
    # 本期只判定并写 meta；拒绝/降级出分由下游评分入口（body_core 闭环 / batch）执行。
    gate = evaluate_multi_person_gate(num_persons_per_frame)

    meta = {
        **_yolo_authorization_meta(display_scope=YOLO_DISPLAY_SCOPE_INTERNAL),
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
        "frame_count": int(n_frames),
        "fps": float(fps),
        "width": int(w),
        "height": int(h),
        "start_frame": int(start_i),
        "end_frame": int(start_i + len(rows) - 1) if rows else int(start_i),
        "layout": "pose33_normalized_xyzw(yolo_conf)",
        # 多人闸门：检出 num_persons>1 即标记并要求下游拒绝/降级，不静默选最大框。
        "num_persons_per_frame": num_persons_per_frame,
        "max_persons": max_persons,
        "multi_person_frames": multi_person_frames,
        "multi_person_detected": gate["multi_person_detected"],
        "gate_status": gate["gate_status"],
        "review_required": gate["review_required"],
        "gate_note": gate["gate_note"],
        "track_ids": track_ids,
        "track_policy": "single_target_largest_box; per-segment reset",
        "track_reset_note": TRACK_RESET_NOTE,
    }
    return landmarks, valid_mask, meta
