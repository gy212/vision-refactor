# -*- coding: utf-8 -*-
"""Pose33 骨架叠加的唯一实现：骨架、关节角、手部、ROI 框、FPS/提示/标签文字（设计文档 §6.4、§3 第 19 项）。

``render_pose`` 是"骨架 + 关节角 + 手部 + 实时标签"的无状态组合，``PoseEngine.annotate`` 与各实时循环
都调用它。文字叠加按样式常量绘制，样式逐项复现迁移前各循环里的 ``cv2.putText`` 参数。骨架输入可以是
landmark 对象序列，也可以是 ``PoseSequence`` 的 ``(33, 4)`` 数组行。

"唯一"的范围是 §3 第 19 项取代的 Pose33 叠加。以下绘制不在其内、各自保留：YOLO 调试框
（``backends/yolo.py``，YOLO 不进默认路径）、``analysis/bench_annotate_fps`` 的基准对照、练习报告的
证据帧插图（``feedback_report``，P8 随报告迁入 reporting）、练习面板的 Tk 画布（``apps/feedback_panel``）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from core.foundation.geometry import angle_deg
from core.pose.mediapipe_runtime import mediapipe
from core.pose.landmarks import as_landmark_objects, visibility as _visibility
from core.pose.live_labels import classify_pose_actions, is_v_sign

SKELETON_COLOR = (0, 140, 255)
POINT_COLOR = (0, 255, 0)
DRAW_VIS_THR = 0.3
FACE_MAX = 10  # BlazePose 0..10 是面部点；手挡脸时画出来容易混淆

# BlazePose 33 点的身体连线（不含面部与手指），用于离线导出（原 tech_eval._POSE_CONNECTIONS）。
BODY_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


@lru_cache(maxsize=1)
def _mediapipe_connections() -> tuple[tuple[int, int], ...]:
    connections = mediapipe().tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    return tuple((int(c.start), int(c.end)) for c in connections)


def draw_skeleton(out_bgr: np.ndarray, landmarks, *, draw_face: bool = False,
                  connections: tuple[tuple[int, int], ...] | None = None) -> None:
    """画骨架连线与关节点；visibility < 0.3 的点与连线不画。``landmarks`` 可为对象序列或 ``(33, 4)`` 数组。

    ``connections=None`` 用 MediaPipe 的完整 33 点连线并按 ``draw_face`` 过滤面部；离线导出传
    ``BODY_CONNECTIONS``（此时面部点照常画点）。
    """
    landmarks = as_landmark_objects(landmarks)
    if landmarks is None:
        return
    h, w = out_bgr.shape[:2]
    skip_face = connections is None and not draw_face
    pairs = _mediapipe_connections() if connections is None else connections

    # Draw connections first for nicer layering.
    for i, j in pairs:
        if skip_face and (i <= FACE_MAX or j <= FACE_MAX):
            continue
        a, b = landmarks[i], landmarks[j]
        if _visibility(a) < DRAW_VIS_THR or _visibility(b) < DRAW_VIS_THR:
            continue
        cv2.line(out_bgr, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)),
                 SKELETON_COLOR, 2, cv2.LINE_AA)

    for idx, p in enumerate(landmarks):
        if skip_face and idx <= FACE_MAX:
            continue
        if _visibility(p) < DRAW_VIS_THR:
            continue
        cv2.circle(out_bgr, (int(p.x * w), int(p.y * h)), 3, POINT_COLOR, -1, cv2.LINE_AA)


def _pt2d(lm, w: int, h: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def draw_joint_angles(out_bgr: np.ndarray, landmarks) -> None:
    """Overlay knee (hip-knee-ankle) and elbow (shoulder-elbow-wrist) angles. ``landmarks`` 可为对象序列或 ``(33, 4)`` 数组。"""
    landmarks = as_landmark_objects(landmarks)
    if landmarks is None:
        return
    h, w = out_bgr.shape[:2]

    # (label_anchor_idx, a, b, c): angle is at vertex b.
    joints = [
        (25, 23, 25, 27),  # left knee
        (26, 24, 26, 28),  # right knee
        (13, 11, 13, 15),  # left elbow
        (14, 12, 14, 16),  # right elbow
    ]
    for vtx, ia, ib, ic in joints:
        a, b, c = landmarks[ia], landmarks[ib], landmarks[ic]
        if _visibility(a) < DRAW_VIS_THR or _visibility(b) < DRAW_VIS_THR or _visibility(c) < DRAW_VIS_THR:
            continue
        pa, pb, pc = _pt2d(a, w, h), _pt2d(b, w, h), _pt2d(c, w, h)
        ang = angle_deg(pa, pb, pc)

        # 加粗重画构成夹角的两条边与三个节点（覆盖在骨架细线之上），用与骨架明显不同的高对比色。
        ipa = (int(pa[0]), int(pa[1]))
        ipb = (int(pb[0]), int(pb[1]))
        ipc = (int(pc[0]), int(pc[1]))
        cv2.line(out_bgr, ipb, ipa, (255, 0, 255), 5, cv2.LINE_AA)  # 品红边
        cv2.line(out_bgr, ipb, ipc, (255, 0, 255), 5, cv2.LINE_AA)
        for pt in (ipa, ipb, ipc):
            cv2.circle(out_bgr, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)  # 红节点

        # 在顶点 b 处用一段圆弧连接 b→a 与 b→c 两条边（取两向量间的内角短弧）。
        start = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
        end = math.degrees(math.atan2(pc[1] - pb[1], pc[0] - pb[0]))
        sweep = (end - start + 180.0) % 360.0 - 180.0  # 归一化到 (-180, 180] 走短边
        radius = max(8, int(0.018 * math.hypot(w, h)))
        cv2.ellipse(out_bgr, (int(pb[0]), int(pb[1])), (radius, radius), 0.0,
                    start, start + sweep, (255, 255, 0), 4, cv2.LINE_AA)

        vx, vy = int(landmarks[vtx].x * w), int(landmarks[vtx].y * h)
        cv2.putText(out_bgr, f"{ang:.0f}", (vx + 6, vy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_hands(out_bgr: np.ndarray, hands) -> None:
    """画手部 21 点连线与关节点。"""
    if not hands:
        return
    h, w = out_bgr.shape[:2]
    connections = mediapipe().tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    for hand in hands:
        for c in connections:
            a, b = hand[int(c.start)], hand[int(c.end)]
            cv2.line(out_bgr, (int(a.x * w), int(a.y * h)), (int(b.x * w), int(b.y * h)),
                     (200, 255, 0), 2, cv2.LINE_AA)
        for p in hand:
            cv2.circle(out_bgr, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1, cv2.LINE_AA)


UNSET_ACTION_POSE = object()


def render_pose(frame_bgr: np.ndarray, pose_landmarks, hands, *, draw_face: bool = False,
                draw_joint_angles_overlay: bool = False,
                action_pose_landmarks=UNSET_ACTION_POSE) -> tuple[np.ndarray, list[str]]:
    """无状态绘制：在帧副本上画 pose/hands 并返回 ``(annotated, actions)``；pose 可为对象序列或 ``(33, 4)`` 数组。

    ``action_pose_landmarks`` 可单独指定用于动作分类的 raw landmarks，使平滑结果只影响骨架绘制；
    省略时用 ``pose_landmarks`` 分类。
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    pose_landmarks = as_landmark_objects(pose_landmarks)
    action_pose = pose_landmarks if action_pose_landmarks is UNSET_ACTION_POSE else as_landmark_objects(action_pose_landmarks)
    actions = classify_pose_actions(action_pose, w, h)

    draw_skeleton(out, pose_landmarks, draw_face=draw_face)
    if draw_joint_angles_overlay:
        draw_joint_angles(out, pose_landmarks)

    draw_hands(out, hands)
    if any(is_v_sign(hand, w, h) for hand in hands):
        actions.insert(0, "V_SIGN")
    return out, actions


# --------------------------------------------------------------------------- #
# 文字与框
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextStyle:
    scale: float
    color: tuple[int, int, int]
    thickness: int
    line_type: int = cv2.LINE_8


TK_FPS = TextStyle(0.8, (255, 255, 255), 2, cv2.LINE_AA)
TK_HINT = TextStyle(0.6, (255, 255, 255), 1, cv2.LINE_AA)
CLI_LABEL = TextStyle(0.9, (20, 20, 240), 2, cv2.LINE_AA)
CLI_FPS = TextStyle(0.7, (255, 255, 255), 1)
CLI_HINT = TextStyle(0.6, (255, 255, 255), 1)
STATUS_OUTER = TextStyle(0.6, (255, 255, 255), 2, cv2.LINE_AA)
STATUS_INNER = TextStyle(0.6, (0, 0, 0), 1, cv2.LINE_AA)
TK_HINT_TEXT = "点击“停止”结束"
CLI_HINT_TEXT = "Press 'q' to quit"
ROI_COLOR = (0, 255, 255)


def draw_text(out_bgr: np.ndarray, text: str, origin: tuple[int, int], style: TextStyle) -> None:
    cv2.putText(out_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, style.scale, style.color,
                style.thickness, style.line_type)


def draw_tk_hud(out_bgr: np.ndarray, fps_text: str, *, hint: bool) -> None:
    """Tk 预览：左上角 FPS，可选底部"点击停止结束"提示。"""
    draw_text(out_bgr, fps_text, (10, 30), TK_FPS)
    if hint:
        draw_text(out_bgr, TK_HINT_TEXT, (10, out_bgr.shape[0] - 12), TK_HINT)


def draw_cli_hud(out_bgr: np.ndarray, labels: list[str], fps_text: str, *, hint: bool = True) -> None:
    """CLI 预览：左上角最多 5 个英文标签，其下 FPS，可选底部"按 q 退出"提示。"""
    y = 30
    for label in labels[:5]:
        draw_text(out_bgr, label, (10, y), CLI_LABEL)
        y += 30
    draw_text(out_bgr, fps_text, (10, y + 5), CLI_FPS)
    if hint:
        draw_text(out_bgr, CLI_HINT_TEXT, (10, out_bgr.shape[0] - 15), CLI_HINT)


def draw_status_lines(out_bgr: np.ndarray, lines: list[str], *, top: int = 22, step: int = 22) -> None:
    """离线导出：左上角逐行白底黑字的状态文字。"""
    for k, text in enumerate(lines):
        draw_text(out_bgr, text, (10, top + k * step), STATUS_OUTER)
        draw_text(out_bgr, text, (10, top + k * step), STATUS_INNER)


def draw_roi(out_bgr: np.ndarray, roi: tuple[float, float, float, float]) -> None:
    """按归一化 ``(x0, y0, x1, y1)`` 画考试站位 ROI 框。"""
    h0, w0 = out_bgr.shape[:2]
    cv2.rectangle(out_bgr, (int(roi[0] * w0), int(roi[1] * h0)), (int(roi[2] * w0), int(roi[3] * h0)),
                  ROI_COLOR, 2)
