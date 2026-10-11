"""学生练习的合成输入：护位基线 + 前/后手出拳或低鞭腿的位移，写成真实 AVI，由脚本化管线回放关键点。"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

W, H, FPS, N = 320, 180, 30.0, 150
_A = W / H  # 宽高比：练习用 ASPECT_CORRECTED 坐标，x 单位是图像高度

# 正面实战式（左式），坐标为宽高比修正后的 (x, y)；最后换回归一化图像坐标。
_BASE = {
    0: (0.86, 0.15), 1: (0.85, 0.13), 2: (0.85, 0.13), 3: (0.84, 0.13), 4: (0.87, 0.13), 5: (0.87, 0.13),
    6: (0.88, 0.13), 7: (0.82, 0.14), 8: (0.90, 0.14), 9: (0.85, 0.19), 10: (0.87, 0.19),
    11: (0.76, 0.30), 12: (0.96, 0.30), 13: (0.73, 0.40), 14: (0.99, 0.40), 15: (0.80, 0.26), 16: (0.92, 0.26),
    17: (0.80, 0.25), 18: (0.92, 0.25), 19: (0.80, 0.25), 20: (0.92, 0.25), 21: (0.80, 0.25), 22: (0.92, 0.25),
    23: (0.80, 0.55), 24: (0.92, 0.55), 25: (0.76, 0.70), 26: (0.96, 0.70), 27: (0.72, 0.85), 28: (1.00, 0.85),
    29: (0.73, 0.87), 30: (0.99, 0.87), 31: (0.68, 0.88), 32: (1.04, 0.88),
}


def _bump(t: int, start: int, length: int = 12) -> float:
    if not start <= t < start + length:
        return 0.0
    return math.sin(math.pi * (t - start) / (length - 1))


def _events(action: str) -> list[tuple[str, int]]:
    seq = {"straight_combo": ["front_straight", "rear_straight"], "hook_combo": ["front_hook", "rear_hook"]}
    names = seq.get(action, [] if action == "stance" else [action])
    if action == "broken_combo":
        names = ["rear_straight", "front_straight"]
    return [(name, 30 + 40 * i) for i, name in enumerate(names)]


def landmarks(action: str, stance: str, view: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 33, 4)`` 归一化图像坐标与 ``(N,)`` 是否检出。"""
    rng = np.random.default_rng(seed)
    base = np.zeros((33, 4))
    for i, (x, y) in _BASE.items():
        base[i] = (x, y, 0.0, 0.97)
    front_left = stance == "left"
    out = np.repeat(base[None], N, axis=0)
    for name, start in _events(action):
        front = name.startswith("front_")
        left = front == front_left
        wr, el, ankle, toe = (15, 13, 27, 31) if left else (16, 14, 28, 32)
        sign = -1.0 if left else 1.0
        for t in range(N):
            b = _bump(t, start)
            if "kick" in name:
                out[t, ankle, :2] += (sign * 0.10 * b, -0.18 * b)
                out[t, toe, :2] += (sign * 0.12 * b, -0.20 * b)
                out[t, 23 if left else 24, 2] += 0.08 * b
            elif "hook" in name:
                out[t, wr, :2] += (sign * 0.12 * b, 0.02 * b)
                out[t, el, :2] += (sign * 0.02 * b, -0.10 * b)
            else:
                out[t, wr, :2] += (sign * 0.16 * b, 0.04 * b)
                out[t, el, :2] += (sign * 0.06 * b, -0.04 * b)
    if view == "side":
        cx = 0.86
        out[:, :, 0] = cx + 0.35 * (out[:, :, 0] - cx) + 0.04 * out[:, :, 1]
    out[:, :, :2] += rng.normal(0.0, 0.002, size=(N, 33, 2))
    out[:, :, 3] = np.clip(out[:, :, 3] - np.abs(rng.normal(0.0, 0.05, size=(N, 33))), 0.0, 1.0)
    out[115:118, 15:17, 3] = 0.3
    out[:, :, 0] /= _A
    detected = np.ones(N, dtype=bool)
    detected[[100, 101, 102, 140]] = False
    return out, detected


def write_avi(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(N):
        writer.write(np.full((H, W, 3), (i * 7) % 255, np.uint8))
    writer.release()
    return path


def scripted_pipeline(lm: np.ndarray, detected: np.ndarray):
    """按推理时间戳回放第 i 帧关键点（抽帧时也对得上帧号）。"""

    class Pipeline:
        def infer(self, frame, *, timestamp_ms):
            i = int(round(float(timestamp_ms) * FPS / 1000.0))
            if not detected[i]:
                return None, []
            return [SimpleNamespace(x=float(r[0]), y=float(r[1]), z=float(r[2]), visibility=float(r[3]))
                    for r in lm[i]], []

        def close(self):
            pass

    return Pipeline
