# -*- coding: utf-8 -*-
"""姿态归一化：v1/v2/v3 与 body_core_v1 归一化、左右镜像。输入为 landmark 对象序列或 ``(33, 4)`` 数组行。"""
from __future__ import annotations

import numpy as np

from core.pose.layouts import FeatureLayoutSpec, get_layout, resolve_layout_by_shape

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
    """Return landmark ``(x, y)`` rows as float32 for vectorized normalizers.

    ``(33, 4)`` 数组与 landmark 对象给出同一组 float32 值（对象属性是 float32 精度，数组以 float64 无损保存）。
    """
    if isinstance(pose_landmarks, np.ndarray):
        idx = np.fromiter((int(i) for i in indices), dtype=np.intp)
        return np.asarray(pose_landmarks[idx, :2], dtype=np.float32)
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
    from core.pose.layouts import BODY_CORE_V1

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
