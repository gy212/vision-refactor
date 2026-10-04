from __future__ import annotations

from collections.abc import Callable
from threading import Event

import numpy as np

from .feature_layout import (
    FeatureLayoutSpec,
    get_layout,
    resolve_layout_by_shape,
)

# ---------------------------------------------------------------------------
# valid_mask 契约（YOLO 迁移 S1 / Issue #5）
# ---------------------------------------------------------------------------
# “此点是否可用”的唯一判据集中在此，不再散落 `lm[idx, 3] >= thr` / `_lm_vis(...) >= thr`。
# - MediaPipe 路径：第 4 通道存 visibility，按 `visibility >= 0.5` 灌入，
#   故 validity_policy=visibility_thr、valid_conf_thr=0.5（视为已标定）。
# - 阈值只允许出现在 `derive_valid_mask`，避免再次散落。
DEFAULT_VALID_CONF_THR: float = 0.5
# MediaPipe 路径的有效性策略名（写入 meta/config，便于下游区分可信/待标定）。
MEDIAPIPE_VALIDITY_POLICY: str = "visibility_thr"


def derive_valid_mask(landmarks: np.ndarray, thr: float = DEFAULT_VALID_CONF_THR) -> np.ndarray:
    """集中式有效性判据：``landmarks[..., 3] >= thr``。

    这是把散落在 ``analysis/tech_eval.py`` 与 ``core/rule_scoring.py`` 的
    ``lm[idx, 3] >= thr`` 风格判据统一收口的唯一入口（YOLO 迁移 Issue #5）。
    下游一律读返回的 mask，不再各自比较第 4 通道。

    参数
    ----
    landmarks:
        ``(33, 4)`` 单帧或 ``(T, 33, 4)`` 序列。第 4 通道按 ``confidence_kind``
        语义存储（MediaPipe 为 visibility）。
    thr:
        有效阈值。MediaPipe 侧默认 ``0.5``（已标定）。

    返回
    ----
    与输入对应的 bool mask：单帧返回 ``(33,)``，序列返回 ``(T, 33)``。
    其逐元素结果与旧式 ``float(lm[idx, 3]) >= thr`` 完全一致（float32→float64
    拓宽无精度损失，``0.5`` 在两种精度下均可精确表示）。
    """
    arr = np.asarray(landmarks)
    if arr.ndim not in (2, 3) or arr.shape[-1] < 4:
        raise ValueError(
            f"derive_valid_mask 期望 (33,4) 或 (T,33,4) 的关键点数组，实际 shape={arr.shape}"
        )
    return arr[..., 3] >= float(thr)


# ---------------------------------------------------------------------------
# 关键点命名与能力分组（YOLO 迁移 S4 / Issue #11）
# ---------------------------------------------------------------------------
# 评估链路要能诚实声明「依赖哪些关键点」「运行时缺了哪些点」。本节提供集中的
# BlazePose33 名称表与「后端可提供性」能力分组，供 `rule_scoring`（规则）与
# `tech_eval`（技术指标）统一声明 ``required_landmarks`` / ``required_capabilities``
# 与运行时 ``missing_landmarks``，下游用稳定字段判断而非靠中文字符串。
#
# 能力分组按 COCO17（YOLO）可映射性划分：COCO17 缺眼细分(1,3,4,6)/
# 嘴角(9,10)/手指(17-22)/脚跟脚尖(29-32)，故这些分组在 YOLO-only 下
# 结构性不可用（缺点指标）。本期 #10 标定结论为「仅预览」，YOLO partial eval
# 未启用，这些声明先服务 MediaPipe 侧的结构化状态，并为后续 YOLO 路径预留判据。
BLAZE33_LANDMARK_NAMES: tuple[str, ...] = (
    "nose",            # 0
    "left_eye_inner",  # 1
    "left_eye",        # 2
    "left_eye_outer",  # 3
    "right_eye_inner",  # 4
    "right_eye",       # 5
    "right_eye_outer",  # 6
    "left_ear",        # 7
    "right_ear",       # 8
    "mouth_left",      # 9
    "mouth_right",     # 10
    "left_shoulder",   # 11
    "right_shoulder",  # 12
    "left_elbow",      # 13
    "right_elbow",     # 14
    "left_wrist",      # 15
    "right_wrist",     # 16
    "left_pinky",      # 17
    "right_pinky",     # 18
    "left_index",      # 19
    "right_index",     # 20
    "left_thumb",      # 21
    "right_thumb",     # 22
    "left_hip",        # 23
    "right_hip",       # 24
    "left_knee",       # 25
    "right_knee",      # 26
    "left_ankle",      # 27
    "right_ankle",     # 28
    "left_heel",       # 29
    "right_heel",      # 30
    "left_foot_index",  # 31
    "right_foot_index",  # 32
)

# 能力分组标签。
CAP_FACE_CENTER = "face_center"  # 鼻 0（COCO17 有）
CAP_EYES = "eyes"                # 左/右眼中心 2,5（COCO17 有近似映射）
CAP_EYE_DETAILS = "eye_details"  # 眼细分 1,3,4,6（COCO17 无）
CAP_EARS = "ears"                # 耳 7,8（COCO17 有）
CAP_MOUTH = "mouth"              # 嘴角 9,10（COCO17 无）
CAP_ARMS = "arms"                # 肩/肘/腕 11-16（COCO17 有）
CAP_HANDS = "hands"              # 手指 17-22（COCO17 无）
CAP_LEGS = "legs"                # 髋/膝/踝 23-28（COCO17 有）
CAP_FEET = "feet"                # 脚跟/脚尖 29-32（COCO17 无）

# COCO17（YOLO）结构性支持的能力集合；缺 eye_details/mouth/hands/feet。
# 仅作声明用途，本期不接 YOLO partial eval。
COCO17_SUPPORTED_CAPABILITIES: frozenset[str] = frozenset(
    {CAP_FACE_CENTER, CAP_EYES, CAP_EARS, CAP_ARMS, CAP_LEGS}
)

_LANDMARK_CAPABILITY: dict[int, str] = {
    0: CAP_FACE_CENTER,
    1: CAP_EYE_DETAILS, 2: CAP_EYES, 3: CAP_EYE_DETAILS,
    4: CAP_EYE_DETAILS, 5: CAP_EYES, 6: CAP_EYE_DETAILS,
    7: CAP_EARS, 8: CAP_EARS,
    9: CAP_MOUTH, 10: CAP_MOUTH,
    11: CAP_ARMS, 12: CAP_ARMS, 13: CAP_ARMS, 14: CAP_ARMS, 15: CAP_ARMS, 16: CAP_ARMS,
    17: CAP_HANDS, 18: CAP_HANDS, 19: CAP_HANDS, 20: CAP_HANDS, 21: CAP_HANDS, 22: CAP_HANDS,
    23: CAP_LEGS, 24: CAP_LEGS, 25: CAP_LEGS, 26: CAP_LEGS, 27: CAP_LEGS, 28: CAP_LEGS,
    29: CAP_FEET, 30: CAP_FEET, 31: CAP_FEET, 32: CAP_FEET,
}


def landmark_names(indices) -> tuple[str, ...]:
    """把 BlazePose33 索引序列映射为稳定的英文名称元组（顺序保持输入顺序）。"""
    return tuple(BLAZE33_LANDMARK_NAMES[int(i)] for i in indices)


def landmark_capabilities(indices) -> tuple[str, ...]:
    """把索引序列归并为去重后的能力分组元组（按首次出现顺序）。

    供规则 / 指标声明 ``required_capabilities``。例如依赖嘴角的规则会带 ``mouth``，
    依赖脚跟脚尖的指标会带 ``feet``——这些在 COCO17 下结构性缺失。
    """
    caps: list[str] = []
    for i in indices:
        c = _LANDMARK_CAPABILITY.get(int(i))
        if c is not None and c not in caps:
            caps.append(c)
    return tuple(caps)


def landmarks_missing_for_capabilities(indices, supported_capabilities) -> tuple[str, ...]:
    """返回 ``indices`` 中其能力分组不在 ``supported_capabilities`` 内的关键点名称。

    用于「后端结构性缺点」判定：例如 YOLO（COCO17）不支持 ``feet`` 能力时，依赖
    脚跟/脚尖的规则会被判为 ``skipped`` + ``skip_reason=missing_landmarks``，并由本
    函数列出具体缺失的关键点名（``left_heel`` 等）。``supported_capabilities=None``
    视为「全部支持」（MediaPipe full），返回空元组。
    """
    if supported_capabilities is None:
        return ()
    supported = frozenset(supported_capabilities)
    return tuple(
        BLAZE33_LANDMARK_NAMES[int(i)]
        for i in indices
        if _LANDMARK_CAPABILITY.get(int(i)) not in supported
    )


def pose_view_score(pose_landmarks) -> float | None:
    """
    Heuristic "frontness" score for a single frame.

    Higher => more likely facing the camera (front view).
    Lower  => more likely side view.

    This uses only a few stable BlazePose landmarks (shoulders/hips) and combines:
      - normalized shoulder width (relative to torso length)
      - left/right visibility balance (front tends to be more symmetric)
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    def xy(i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    def vis(i: int) -> float:
        return float(getattr(lm[i], "visibility", 1.0))

    ls, rs = xy(L_SHOULDER), xy(R_SHOULDER)
    lh, rh = xy(L_HIP), xy(R_HIP)

    if (not np.isfinite(ls).all()) or (not np.isfinite(rs).all()) or (not np.isfinite(lh).all()) or (not np.isfinite(rh).all()):
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    torso = float(np.linalg.norm((0.5 * (ls + rs)) - (0.5 * (lh + rh))))
    if (not np.isfinite(shoulder_w)) or (not np.isfinite(torso)) or torso < 1e-6:
        return None

    width_ratio = shoulder_w / (torso + 1e-6)

    # Visibility balance helps when one side is occluded (typical in side view).
    v_ls, v_rs = vis(L_SHOULDER), vis(R_SHOULDER)
    v_lh, v_rh = vis(L_HIP), vis(R_HIP)

    def balance(a: float, b: float) -> float:
        denom = max(a, b, 1e-6)
        return 1.0 - (abs(a - b) / denom)

    vis_balance = 0.5 * (balance(v_ls, v_rs) + balance(v_lh, v_rh))

    s = float(width_ratio * vis_balance)
    if not np.isfinite(s):
        return None
    return s


def mirror_pose_features(
    features: np.ndarray,
    *,
    layout: FeatureLayoutSpec | str | None = None,
) -> np.ndarray:
    """
    Mirror normalized pose features along the X axis and swap left/right joints.

    Input: ``(T, J, 2)`` or ``(J, 2)``. 左右互换按 layout 的 ``mirror_pairs`` 进行，
    不再写死 22 点配对。

    layout 解析顺序：
      1. 显式传入的 ``layout``（``FeatureLayoutSpec`` 或已注册布局名）。
      2. 未传时按单帧 shape 反查唯一匹配的已注册布局
         （``pose33_v3`` 的 ``(22, 2)`` 即走此路径，行为与旧实现一致）。
      3. 仍无法解析则报清晰错误，不静默按 22 点处理。
    """
    if features.ndim not in (2, 3):
        raise ValueError(f"Unsupported features shape: {features.shape}")

    frame_shape = features.shape if features.ndim == 2 else features.shape[1:]

    spec: FeatureLayoutSpec | None
    if isinstance(layout, FeatureLayoutSpec):
        spec = layout
    elif isinstance(layout, str):
        spec = get_layout(layout)
    else:
        spec = resolve_layout_by_shape(tuple(int(x) for x in frame_shape))

    if spec is None:
        raise ValueError(
            f"无法为单帧 shape {tuple(int(x) for x in frame_shape)} 解析 feature layout；"
            "请显式传入 layout 名或注册对应布局。"
        )
    if tuple(int(x) for x in frame_shape) != tuple(int(x) for x in spec.shape):
        raise ValueError(
            f"features 单帧 shape {tuple(int(x) for x in frame_shape)} 与 layout "
            f"{spec.name!r} 的 shape {tuple(spec.shape)} 不一致"
        )

    if features.ndim == 2:
        x = features.copy()
        x[:, 0] *= -1.0
        y = x.copy()
        for a, b in spec.mirror_pairs:
            y[a] = x[b]
            y[b] = x[a]
        return y

    x = features.copy()
    x[:, :, 0] *= -1.0
    y = x.copy()
    for a, b in spec.mirror_pairs:
        y[:, a] = x[:, b]
        y[:, b] = x[:, a]
    return y


def normalize_pose_xy_v1(pose_landmarks) -> np.ndarray | None:
    """
    Legacy normalization (v1) kept for backward compatibility with older templates.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    pts = _landmark_xy_array(lm, range(33))
    ls, rs = pts[L_SHOULDER], pts[R_SHOULDER]
    lh, rh = pts[L_HIP], pts[R_HIP]

    center = 0.5 * (lh + rh)
    if not np.isfinite(center).all():
        center = 0.5 * (ls + rs)

    scale = float(np.linalg.norm(ls - rs))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.linalg.norm(lh - rh))
    if not np.isfinite(scale) or scale < 1e-6:
        return None

    v = rs - ls
    ang = float(np.arctan2(v[1], v[0]))
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    out = _transform_xy_points(pts[11:33], center=center, scale=scale, rot=R, clip=None)
    if not np.isfinite(out).all():
        return None
    return out


def normalize_pose_xy(pose_landmarks) -> np.ndarray | None:
    """
    Convert pose landmarks into a normalized feature tensor.

    Output shape: (22, 2) for landmark indices 11..32 (face removed).
    Normalization:
      - translate by hip center (or shoulder center fallback)
      - scale by shoulder width (or hip width fallback)
      - rotate to make shoulders horizontal (when available)
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    pts = _landmark_xy_array(lm, range(33))
    ls, rs = pts[L_SHOULDER], pts[R_SHOULDER]
    lh, rh = pts[L_HIP], pts[R_HIP]

    center = 0.5 * (lh + rh)
    if not np.isfinite(center).all():
        center = 0.5 * (ls + rs)

    # In normalized image coords, shoulder/hip width should not be extremely tiny.
    # A very small scale usually means a bad detection and causes numeric blow-ups.
    min_scale = 0.02
    scale = float(np.linalg.norm(ls - rs))
    if (not np.isfinite(scale)) or (scale < min_scale):
        scale = float(np.linalg.norm(lh - rh))
    if (not np.isfinite(scale)) or (scale < min_scale):
        return None

    # rotation (align shoulders horizontally)
    v = rs - ls
    ang = float(np.arctan2(v[1], v[0]))  # radians
    ca, sa = float(np.cos(-ang)), float(np.sin(-ang))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    out = _transform_xy_points(pts[11:33], center=center, scale=scale, rot=R, clip=5.0)
    if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
        return None
    return out


def normalize_pose_xy_v3(pose_landmarks) -> np.ndarray | None:
    """
    Normalization v3: more view-robust for side angles.

    Compared to v2, this prefers torso length as the scale (less dependent on yaw),
    and uses a hybrid rotation strategy:
      - front-ish: align shoulders to horizontal
      - side-ish:  align torso to vertical

    Output shape: (22, 2) for landmark indices 11..32.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks

    # BlazePose indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    pts = _landmark_xy_array(lm, range(33))
    ls, rs = pts[L_SHOULDER], pts[R_SHOULDER]
    lh, rh = pts[L_HIP], pts[R_HIP]

    if (not np.isfinite(ls).all()) or (not np.isfinite(rs).all()) or (not np.isfinite(lh).all()) or (not np.isfinite(rh).all()):
        return None

    sh_c = 0.5 * (ls + rs)
    hip_c = 0.5 * (lh + rh)
    center = hip_c if np.isfinite(hip_c).all() else sh_c
    if not np.isfinite(center).all():
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    hip_w = float(np.linalg.norm(lh - rh))
    torso_len = float(np.linalg.norm(sh_c - hip_c))

    # Prefer torso length as scale; fall back to widths if needed.
    min_scale = 0.02
    scale = torso_len if (np.isfinite(torso_len) and torso_len >= min_scale) else max(shoulder_w, hip_w)
    if (not np.isfinite(scale)) or (scale < min_scale):
        return None

    # Decide rotation axis by "front-ish" heuristic.
    width_ratio = float(shoulder_w / (torso_len + 1e-6)) if np.isfinite(torso_len) else 0.0
    use_shoulders = bool(width_ratio >= 0.35)
    if use_shoulders:
        v = rs - ls
        target = 0.0  # align to +X axis
    else:
        v = sh_c - hip_c
        target = -float(np.pi) / 2.0  # align upwards (-Y)

    v_norm = float(np.linalg.norm(v))
    if (not np.isfinite(v_norm)) or v_norm < 1e-6:
        # Fallback: no rotation (still scaled/centered).
        ca, sa = 1.0, 0.0
    else:
        ang = float(np.arctan2(float(v[1]), float(v[0])))
        rot = float(target - ang)
        ca, sa = float(np.cos(rot)), float(np.sin(rot))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    out = _transform_xy_points(pts[11:33], center=center, scale=float(scale), rot=R, clip=5.0)
    if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
        return None
    return out


def _blaze33_xy_getter(pose_landmarks):
    """返回一个 ``getter(i) -> (x, y) float32`` ，统一 MediaPipe landmark 对象与
    YOLO ``(33,4)`` numpy 行两种输入，供 ``body_core_v1`` 共享 normalizer 使用。

    - MediaPipe：``pose_landmarks`` 是一串带 ``.x/.y`` 属性的 landmark 对象。
    - YOLO：``pose_landmarks`` 是 ``(33,4)`` 的 ``(x, y, z, conf)`` numpy 行
      （由 ``core.yolo_adapter.map_coco17_to_blaze33`` 产出）。

    返回 ``None`` 表示输入无法识别。
    """
    if pose_landmarks is None:
        return None
    arr = None
    if isinstance(pose_landmarks, np.ndarray):
        arr = pose_landmarks
    elif hasattr(pose_landmarks, "shape") and not hasattr(pose_landmarks, "__len__"):
        arr = np.asarray(pose_landmarks)
    if arr is not None:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] < 33 or a.shape[1] < 2:
            return None

        def _get(i: int) -> np.ndarray:
            return np.array([a[i, 0], a[i, 1]], dtype=np.float32)

        return _get

    # MediaPipe landmark 对象序列。
    lm = pose_landmarks

    def _get_obj(i: int) -> np.ndarray:
        return np.array([float(lm[i].x), float(lm[i].y)], dtype=np.float32)

    return _get_obj


def _landmark_xy_array(pose_landmarks, indices) -> np.ndarray:
    """Return landmark ``(x, y)`` rows as float32 for vectorized normalizers."""
    return np.asarray(
        [[float(pose_landmarks[int(i)].x), float(pose_landmarks[int(i)].y)] for i in indices],
        dtype=np.float32,
    )


def _transform_xy_points(
    points: np.ndarray,
    *,
    center: np.ndarray,
    scale: float,
    rot: np.ndarray,
    clip: float | None,
) -> np.ndarray:
    out = ((points.astype(np.float32, copy=False) - center) / float(scale)) @ rot.T
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    return out.astype(np.float32, copy=False)


def normalize_pose_body_core_v1(pose_landmarks) -> np.ndarray | None:
    """``body_core_v1`` 共享 normalizer（YOLO 迁移 Issue #8 / S2）。

    只使用躯干四肢核心 12 点（肩/肘/腕 + 髋/膝/踝，BlazePose 11..16 / 23..28），
    输出 ``(12, 2)``。归一化策略与 ``normalize_pose_xy_v3`` 同构（躯干长度为主尺度、
    front/side 自适应旋转），但**索引、输出 shape 与关节顺序都来自 ``body_core_v1``
    layout**，因此 YOLO（COCO17 映射）与 MediaPipe 都能产出同一布局，供 #10 三方对比。

    输入兼容
    --------
    - MediaPipe：landmark 对象序列（带 ``.x/.y``）。
    - YOLO：``(33,4)`` numpy 行（``map_coco17_to_blaze33`` 的输出）。

    body_core_v1 的 12 点全部落在 COCO17 可映射点，因此 YOLO 路径下这些点是真实点
    （非合成缺失点）；缺失点不参与该布局，不会污染归一化。
    """
    from .feature_layout import BODY_CORE_V1

    get_xy = _blaze33_xy_getter(pose_landmarks)
    if get_xy is None:
        return None

    # BlazePose 索引（与 BODY_CORE_V1.source_indices 对应）。
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24

    ls, rs = get_xy(L_SHOULDER), get_xy(R_SHOULDER)
    lh, rh = get_xy(L_HIP), get_xy(R_HIP)

    if (
        (not np.isfinite(ls).all())
        or (not np.isfinite(rs).all())
        or (not np.isfinite(lh).all())
        or (not np.isfinite(rh).all())
    ):
        return None

    sh_c = 0.5 * (ls + rs)
    hip_c = 0.5 * (lh + rh)
    center = hip_c if np.isfinite(hip_c).all() else sh_c
    if not np.isfinite(center).all():
        return None

    shoulder_w = float(np.linalg.norm(ls - rs))
    hip_w = float(np.linalg.norm(lh - rh))
    torso_len = float(np.linalg.norm(sh_c - hip_c))

    # 与 v3 同构：优先用躯干长度作为尺度（对 yaw 更稳），回退到肩宽/髋宽。
    min_scale = 0.02
    scale = (
        torso_len
        if (np.isfinite(torso_len) and torso_len >= min_scale)
        else max(shoulder_w, hip_w)
    )
    if (not np.isfinite(scale)) or (scale < min_scale):
        return None

    # 与 v3 同构：按“正面程度”决定旋转对齐轴。
    width_ratio = float(shoulder_w / (torso_len + 1e-6)) if np.isfinite(torso_len) else 0.0
    use_shoulders = bool(width_ratio >= 0.35)
    if use_shoulders:
        v = rs - ls
        target = 0.0  # 对齐到 +X 轴
    else:
        v = sh_c - hip_c
        target = -float(np.pi) / 2.0  # 对齐到上方（-Y）

    v_norm = float(np.linalg.norm(v))
    if (not np.isfinite(v_norm)) or v_norm < 1e-6:
        ca, sa = 1.0, 0.0
    else:
        ang = float(np.arctan2(float(v[1]), float(v[0])))
        rot = float(target - ang)
        ca, sa = float(np.cos(rot)), float(np.sin(rot))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    pts = np.stack([get_xy(int(src)) for src in BODY_CORE_V1.source_indices], axis=0)
    out = _transform_xy_points(pts, center=center, scale=float(scale), rot=R, clip=5.0)
    if (not np.isfinite(out).all()) or float(np.max(np.abs(out))) > 5.0:
        return None
    return out


def motion_energy(seq: np.ndarray) -> np.ndarray:
    # seq: (T, D) with finite values
    d = np.diff(seq, axis=0)
    return np.sqrt(np.mean(d * d, axis=1))


def find_active_range(energy: np.ndarray, pad: int = 10) -> tuple[int, int]:
    """
    Pick the most "active" contiguous segment based on motion energy.
    Returns (start_frame, end_frame) inclusive, in original frame indices.
    """
    if energy.size == 0:
        return 0, 0

    # Smooth (simple moving average)
    k = 9
    if energy.size >= k:
        kernel = np.ones(k, dtype=np.float32) / k
        e = np.convolve(energy, kernel, mode="same")
    else:
        e = energy

    thr = float(np.percentile(e, 70))
    active = e > thr
    if not active.any():
        return 0, int(energy.size)  # energy is T-1

    best_s = best_e = 0
    cur_s = None
    for i, on in enumerate(active.tolist()):
        if on and cur_s is None:
            cur_s = i
        if (not on) and cur_s is not None:
            cur_e = i - 1
            if (cur_e - cur_s) > (best_e - best_s):
                best_s, best_e = cur_s, cur_e
            cur_s = None
    if cur_s is not None:
        cur_e = int(active.size - 1)
        if (cur_e - cur_s) > (best_e - best_s):
            best_s, best_e = cur_s, cur_e

    # energy index i corresponds to transition frame i->i+1, so map back to frames
    start = max(0, best_s - pad)
    end = min(int(active.size), best_e + 1 + pad)  # end frame index
    return start, end


_DtwCancelSignal = Event | Callable[[], bool]


def _raise_if_dtw_cancelled(stop_evt: _DtwCancelSignal | None) -> None:
    if stop_evt is None:
        return
    should_stop = stop_evt if callable(stop_evt) else stop_evt.is_set
    if should_stop():
        raise InterruptedError("DTW 评分已取消")


def subsequence_dtw(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    stop_evt: _DtwCancelSignal | None = None,
) -> tuple[float, int, int]:
    """
    Subsequence DTW: find best matching subsequence of `seq` for `query`.
    Returns (cost, start_index, end_index) in seq indices (inclusive).
    """
    _raise_if_dtw_cancelled(stop_evt)
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    local_cost = _dtw_local_cost(q, s)
    _raise_if_dtw_cancelled(stop_evt)
    for i in range(1, n + 1):
        _raise_if_dtw_cancelled(stop_evt)
        for j in range(1, m + 1):
            cost = float(local_cost[i - 1, j - 1])
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    _raise_if_dtw_cancelled(stop_evt)
    end = int(np.argmin(dp[n, 1:]) + 1)  # dp-space
    best_cost = float(dp[n, end])

    i, j = n, end
    while i > 0:
        _raise_if_dtw_cancelled(stop_evt)
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
        if j <= 0:
            break
    start = max(1, j)  # dp-space

    _raise_if_dtw_cancelled(stop_evt)
    return best_cost, start - 1, end - 1


def subsequence_dtw_with_path(
    query: np.ndarray,
    seq: np.ndarray,
    *,
    stop_evt: _DtwCancelSignal | None = None,
) -> tuple[float, int, int, list[tuple[int, int]]]:
    _raise_if_dtw_cancelled(stop_evt)
    q = query.astype(np.float32).reshape(query.shape[0], -1)
    s = seq.astype(np.float32).reshape(seq.shape[0], -1)

    n, m = q.shape[0], s.shape[0]
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    prev = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, :] = 0.0

    local_cost = _dtw_local_cost(q, s)
    _raise_if_dtw_cancelled(stop_evt)
    for i in range(1, n + 1):
        _raise_if_dtw_cancelled(stop_evt)
        for j in range(1, m + 1):
            cost = float(local_cost[i - 1, j - 1])
            a = dp[i - 1, j - 1]
            b = dp[i - 1, j]
            c = dp[i, j - 1]
            if a <= b and a <= c:
                dp[i, j] = cost + a
                prev[i, j] = 0
            elif b <= c:
                dp[i, j] = cost + b
                prev[i, j] = 1
            else:
                dp[i, j] = cost + c
                prev[i, j] = 2

    _raise_if_dtw_cancelled(stop_evt)
    end = int(np.argmin(dp[n, 1:]) + 1)
    best_cost = float(dp[n, end])

    path_rev: list[tuple[int, int]] = []
    i, j = n, end
    while i > 0 and j > 0:
        _raise_if_dtw_cancelled(stop_evt)
        path_rev.append((int(i - 1), int(j - 1)))
        p = int(prev[i, j])
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1

    start = max(1, int(j))
    path = list(reversed(path_rev))
    _raise_if_dtw_cancelled(stop_evt)
    return best_cost, start - 1, end - 1, path


def _dtw_local_cost(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    diff = q[:, None, :] - s[None, :, :]
    sq = np.sum(diff * diff, axis=2, dtype=np.float32)
    return np.sqrt(sq).astype(np.float32, copy=False)
