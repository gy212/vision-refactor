# -*- coding: utf-8 -*-
"""YOLO COCO17 → BlazePose33-like 的映射契约：常量、边界容器、逐人映射、多人闸门、结果解析与单目标 tracker。

纯 numpy、不导入 ultralytics。缺失点（嘴角、手指、脚跟脚尖、眼细分）一律 ``valid_mask=False`` /
``synthetic=True``，不得伪造成有效点。适配器与序列提取见 ``core.pose.backends.yolo.adapter``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
# 与 core.pose.layouts.BODY_CORE_V1.source_indices 保持一致（此处冗余定义仅为避免
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
        meta=boundary_meta(
            valid_conf_thr=valid_conf_thr,
            num_persons=num_persons,
            selected_index=selected_index,
        ),
    )
    return row, valid, frame


def boundary_meta(
    *,
    valid_conf_thr: float,
    num_persons: int,
    selected_index: int | None,
    display_scope: str = YOLO_DISPLAY_SCOPE_LIMITED,
) -> dict:
    return {
        **yolo_authorization_meta(display_scope=display_scope),
        "backend": "yolo",
        "confidence_kind": YOLO_CONFIDENCE_KIND,
        "validity_policy": YOLO_VALIDITY_POLICY,
        "valid_conf_thr": float(valid_conf_thr),
        "calibration_note": CALIBRATION_NOTE,
        "num_persons": int(num_persons),
        "selected_index": selected_index,
        "track_reset_note": TRACK_RESET_NOTE,
    }


def yolo_authorization_meta(*, display_scope: str) -> dict:
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
        meta=boundary_meta(
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
