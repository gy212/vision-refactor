# -*- coding: utf-8 -*-
"""踢腿高度几何质量评分（低鞭腿 / 高鞭腿）。

与 rule_scoring 的"违规扣分"不同,这里输出**连续质量分 0..1**,并入动作 DTW 子分。

输入用 raw Pose33（归一化图像坐标 (T,33,4)，与 rule_scoring.extract_pose_raw 同源）,
**不能用 DTW 的归一化特征**：后者排除鼻子（idx0）、且旋转对齐会扭曲踢腿高度。
高度只用 y 比值（同除画面高），不受横拍长宽比影响。

评分取整段（DTW 窗口）内踢腿脚**峰值帧**（踝最高）。踢腿脚 = 窗口内帧间位移大的那条腿。

校准旋钮（真实数据可调）：Q_BOUND / Q_PASS / 合格角。
"""
from __future__ import annotations

import numpy as np

# BlazePose33 索引（与 core/rule_scoring.py 一致）
NOSE = 0
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 校准旋钮
Q_PASS = 0.6     # 高鞭：踢到腰（髋）高/夹角90°的合格分；踢到头（鼻）高=1.0
MIN_VALID_FRAMES = 3
PASS_ANGLE_DEG = 90.0  # 高鞭合格线：髋为顶点两踝张开角

# 低鞭腿夹角评分旋钮（髋中点为顶点、站立腿↔踢腿两踝张开角，单位度）。
# 实测三张到位低鞭腿 = 64/73/84°，用户口径：64~84 满分，超区间递减，≥90 直接 0。
LOW_WHIP_FULL_MIN = 64.0   # 满分平台下界
LOW_WHIP_FULL_MAX = 84.0   # 满分平台上界
LOW_WHIP_ZERO_HIGH = 90.0  # 上侧归零角（劈叉过头/踢太高）
LOW_WHIP_ZERO_LOW = 40.0   # 下侧归零角（踢得太低）


def _y_up(lm_row: np.ndarray, idx: int) -> float:
    """up-positive 高度：图像 y 向下增大,取负使"越高越大"。"""
    return -float(lm_row[idx, 1])


def _kick_stand_sides(
    landmarks: np.ndarray, valid: np.ndarray
) -> tuple[int, int, int, int]:
    """按窗口内踝帧间位移判定踢腿腿；返回 (kick_ankle, kick_knee, stand_hip, stand_knee)。"""
    def _travel(ankle_idx: int) -> float:
        ys = np.array(
            [float(landmarks[i, ankle_idx, 1]) for i in range(len(landmarks)) if valid[i]],
            dtype=np.float32,
        )
        xs = np.array(
            [float(landmarks[i, ankle_idx, 0]) for i in range(len(landmarks)) if valid[i]],
            dtype=np.float32,
        )
        if ys.size < 2:
            return 0.0
        return float(np.abs(np.diff(ys)).sum() + np.abs(np.diff(xs)).sum())

    left_travel = _travel(L_ANKLE)
    right_travel = _travel(R_ANKLE)
    if left_travel >= right_travel:
        return L_ANKLE, L_KNEE, R_HIP, R_KNEE
    return R_ANKLE, R_KNEE, L_HIP, L_KNEE


def _peak_frame(landmarks: np.ndarray, valid: np.ndarray, kick_ankle: int) -> int:
    """踢腿踝最高（y 最小）的有效帧索引。"""
    best_i, best_y = -1, np.inf
    for i in range(len(landmarks)):
        if not bool(valid[i]):
            continue
        y = float(landmarks[i, kick_ankle, 1])
        if y < best_y:
            best_y, best_i = y, i
    return best_i


def _xy_px(lm: np.ndarray, idx: int, wh: tuple[float, float]) -> np.ndarray:
    """取真实像素坐标：归一化 x/y 乘回画面宽/高，抵消横竖画幅长宽比畸变。"""
    w, h = wh
    return np.array([float(lm[idx, 0]) * w, float(lm[idx, 1]) * h], dtype=np.float32)


def _ankle_spread_angle(
    lm: np.ndarray, kick_ankle: int, wh: tuple[float, float] = (1.0, 1.0)
) -> float:
    """髋中点为顶点、站立腿↔踢腿两踝张开角（度）。wh=(宽,高) 反归一化算真实几何。"""
    hip = (_xy_px(lm, L_HIP, wh) + _xy_px(lm, R_HIP, wh)) / 2.0
    ka = _xy_px(lm, kick_ankle, wh)
    other_ankle = R_ANKLE if kick_ankle == L_ANKLE else L_ANKLE
    sa = _xy_px(lm, other_ankle, wh)
    v1, v2 = ka - hip, sa - hip
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _score_low_whip(lm, kick_ankle, kick_knee, stand_hip, stand_knee, wh) -> tuple[float, str]:
    """夹角梯形：两腿张开角在 [FULL_MIN,FULL_MAX] 满分1.0,超区间线性降到归零角。

    比旧的"高度三角峰"稳——夹角只用髋中点+两踝三个点,不依赖站立腿膝/髋基准,
    也不受踢腿/站立腿判定影响（张开角与哪条腿是踢腿无关）。
    """
    angle = _ankle_spread_angle(lm, kick_ankle, wh)
    if LOW_WHIP_FULL_MIN <= angle <= LOW_WHIP_FULL_MAX:
        q = 1.0
    elif angle > LOW_WHIP_FULL_MAX:
        # 上侧：FULL_MAX(1.0) → ZERO_HIGH(0)
        span = LOW_WHIP_ZERO_HIGH - LOW_WHIP_FULL_MAX
        q = 1.0 - (angle - LOW_WHIP_FULL_MAX) / span if span > 1e-6 else 0.0
    else:
        # 下侧：FULL_MIN(1.0) → ZERO_LOW(0)
        span = LOW_WHIP_FULL_MIN - LOW_WHIP_ZERO_LOW
        q = 1.0 - (LOW_WHIP_FULL_MIN - angle) / span if span > 1e-6 else 0.0
    q = float(np.clip(q, 0.0, 1.0))
    return q, f"低鞭两腿张开角{angle:.0f}°({LOW_WHIP_FULL_MIN:.0f}~{LOW_WHIP_FULL_MAX:.0f}满分),质量{q:.2f}"


def _score_high_whip(lm, kick_ankle, kick_knee, stand_hip, stand_knee, wh) -> tuple[float, str]:
    """高度映射:膝高→0,髋高(腰平/合格)→Q_PASS,鼻高(头平/优秀)→1.0,更高 clamp 1.0。"""
    q, level = _height_map_score(lm, kick_ankle, stand_hip, stand_knee)
    angle = _ankle_spread_angle(lm, kick_ankle, wh)
    gate = "达标" if angle >= PASS_ANGLE_DEG else f"夹角{angle:.0f}°<90°"
    return q, f"高鞭峰值{level},{gate},质量{q:.2f}"


def _height_map_score(lm, kick_ankle, stand_hip, stand_knee) -> tuple[float, str]:
    """踢腿踝高度映射:膝→0、髋(腰平/合格)→Q_PASS、鼻(头平/优秀)→1.0,更高 clamp 1.0。

    高鞭/侧踹/正蹬共用（"提到髋及格、到头满分"同一口径）。返回 (质量分, 档位文案)。
    """
    ankle_h = _y_up(lm, kick_ankle)
    knee_h = _y_up(lm, stand_knee)
    hip_h = _y_up(lm, stand_hip)
    nose_h = _y_up(lm, NOSE)
    if ankle_h >= hip_h:
        span = nose_h - hip_h
        frac = 0.0 if abs(span) < 1e-6 else (ankle_h - hip_h) / span
        q = Q_PASS + (1.0 - Q_PASS) * float(np.clip(frac, 0.0, 1.0))
    else:
        span = hip_h - knee_h
        frac = 0.0 if abs(span) < 1e-6 else (ankle_h - knee_h) / span
        q = Q_PASS * float(np.clip(frac, 0.0, 1.0))
    q = float(np.clip(q, 0.0, 1.0))
    level = "优秀(头平)" if ankle_h >= nose_h else ("合格(腰平)" if ankle_h >= hip_h else "未达腰平")
    return q, level


def _score_kick_up(lm, kick_ankle, kick_knee, stand_hip, stand_knee, wh) -> tuple[float, str]:
    """侧踹/正蹬:踢腿踝高度映射(提到髋及格、到头满分)。无分腿夹角门。"""
    q, level = _height_map_score(lm, kick_ankle, stand_hip, stand_knee)
    return q, f"踢腿峰值{level},质量{q:.2f}"


_SCORERS = {
    "low_whip": _score_low_whip,
    "high_whip": _score_high_whip,
    "kick_up": _score_kick_up,
}


def score_kick_quality(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    kind: str,
    *,
    width: float | None = None,
    height: float | None = None,
) -> tuple[float | None, str]:
    """踢腿几何质量分 0..1。

    landmarks: (T,33,4) raw Pose33 窗口切片；valid_mask: (T,33) bool。
    kind ∈ {"low_whip","high_whip","kick_up"}。有效帧不足或无法测返回 (None, 原因)。
    width/height: 画面真实宽高（来自 raw meta），用于低鞭腿夹角反归一化抵消横竖画幅
    畸变；缺省时按 1.0 退化（正方形假设，横拍侧面会有长宽比误差）。
    """
    if kind not in _SCORERS:
        raise ValueError(f"未知踢腿类型：{kind}")
    if landmarks.ndim != 3 or landmarks.shape[1:] != (33, 4):
        return None, "关键点形状无效"
    # 踝/膝/髋齐全的帧才有效
    key_idx = (L_ANKLE, R_ANKLE, L_KNEE, R_KNEE, L_HIP, R_HIP)
    valid = np.asarray(valid_mask[:, key_idx], dtype=bool).all(axis=1)
    if int(valid.sum()) < MIN_VALID_FRAMES:
        return None, "有效帧不足,未评估几何质量"
    kick_ankle, kick_knee, stand_hip, stand_knee = _kick_stand_sides(landmarks, valid)
    peak = _peak_frame(landmarks, valid, kick_ankle)
    if peak < 0:
        return None, "无法定位峰值帧"
    wh = (float(width) if width else 1.0, float(height) if height else 1.0)
    return _SCORERS[kind](landmarks[peak], kick_ankle, kick_knee, stand_hip, stand_knee, wh)


def _demo() -> None:
    """合成峰值帧自检：只需踝/膝/髋/鼻的 y，其余填 0，valid 全 1。"""
    def frame(*, kick_ankle_y, stand_knee_y, stand_hip_y, nose_y, kick_side="L"):
        lm = np.zeros((33, 4), dtype=np.float32)
        ka, kk = (L_ANKLE, L_KNEE) if kick_side == "L" else (R_ANKLE, R_KNEE)
        sh, sk = (R_HIP, R_KNEE) if kick_side == "L" else (L_HIP, L_KNEE)
        lm[ka, 1] = kick_ankle_y
        lm[sk, 1] = stand_knee_y
        lm[sh, 1] = stand_hip_y
        lm[NOSE, 1] = nose_y
        # 踢腿踝张开一点 x，让夹角>90（自检 high 合格门）
        lm[ka, 0] = 0.4
        lm[R_ANKLE if kick_side == "L" else L_ANKLE, 0] = -0.4
        return lm

    # y 向下增大：nose 最小(最高)、hip 中、knee 大(低)、ankle 抬高则 y 变小
    # 站立腿：膝 y=0.7, 髋 y=0.5（髋比膝高）；鼻 y=0.1
    def series(kick_ankle_y, n=5, **kw):
        # 制造位移：其余帧踝在低位(y大)，仅一帧到目标高度；peak 取最高(y最小)=目标帧。
        low_y = 0.95  # 比任何目标都低(y更大)，确保不被选为峰值
        frames = [frame(kick_ankle_y=low_y, **kw) for _ in range(n - 1)]
        frames.append(frame(kick_ankle_y=kick_ankle_y, **kw))
        return np.stack(frames), np.ones((n, 33), dtype=bool)

    base = dict(stand_knee_y=0.7, stand_hip_y=0.5, nose_y=0.1)

    # —— low_whip：夹角梯形（髋中点为顶点、站立踝正下方、踢腿踝按目标角摆放）——
    def angle_series(deg, n=5):
        """构造两腿张开角=deg 的峰值帧：髋中点(0.5,0.4)，站立踝正下方，踢腿踝按角度。
        用 width=height=1（正方形）自检，角度即几何张开角。"""
        import math
        hx, hy = 0.5, 0.4
        r = 0.3
        ang = math.radians(deg)
        # 峰值帧：踢腿踝在目标角度位置（此帧角度=deg，被 peak 选中）
        def mk(*, peak):
            lm = np.zeros((33, 4), dtype=np.float32)
            lm[L_HIP] = [hx - 0.02, hy, 0, 0]
            lm[R_HIP] = [hx + 0.02, hy, 0, 0]
            lm[R_ANKLE] = [hx, hy + r, 0, 0]  # 站立腿：正下方（垂直）
            if peak:
                lm[L_ANKLE] = [hx - r * math.sin(ang), hy + r * math.cos(ang), 0, 0]
            else:
                # 非峰值帧：踢腿踝垂放在最低处（y 最大），不被选为峰值，且制造 x/y 位移
                lm[L_ANKLE] = [hx - 0.02, hy + r + 0.2, 0, 0]
            return lm
        frames = [mk(peak=False) for _ in range(n - 1)]
        frames.append(mk(peak=True))
        return np.stack(frames), np.ones((n, 33), dtype=bool)

    # 实测三张到位低鞭 = 64/73/84° → 满分区，全 1.0
    for deg in (64, 73, 84):
        lm, vm = angle_series(deg)
        s, d = score_kick_quality(lm, vm, "low_whip")
        assert s is not None and s > 0.95, (deg, s, d)
    # ≥90° 直接 0（劈叉过头）
    lm, vm = angle_series(90)
    s, d = score_kick_quality(lm, vm, "low_whip")
    assert s < 0.05, (s, d)
    # 87° 在上侧递减区间（84→90 线性）→ 0.5 附近
    lm, vm = angle_series(87)
    s, _ = score_kick_quality(lm, vm, "low_whip")
    assert 0.3 < s < 0.7, s
    # 40° 踢太低 → 0
    lm, vm = angle_series(40)
    s, _ = score_kick_quality(lm, vm, "low_whip")
    assert s < 0.05, s

    # high：踝到鼻高 y=0.1 → ~1.0
    lm, vm = series(0.1, **base)
    s, d = score_kick_quality(lm, vm, "high_whip")
    assert s > 0.95, (s, d)
    # high：踝到髋高 y=0.5（腰平/合格）→ ~Q_PASS
    lm, vm = series(0.5, **base)
    s, _ = score_kick_quality(lm, vm, "high_whip")
    assert abs(s - Q_PASS) < 0.05, s
    # high：踝在膝高 y=0.7（未达腰平）→ ~0
    lm, vm = series(0.7, **base)
    s, _ = score_kick_quality(lm, vm, "high_whip")
    assert s < 0.1, s

    # kick_up（侧踹/正蹬）：与 high 同高度映射。髋高→Q_PASS、鼻高→1.0
    lm, vm = series(0.5, **base)
    s, _ = score_kick_quality(lm, vm, "kick_up")
    assert abs(s - Q_PASS) < 0.05, s
    lm, vm = series(0.1, **base)
    s, _ = score_kick_quality(lm, vm, "kick_up")
    assert s > 0.95, s

    # 有效帧不足 → None
    lm = np.zeros((2, 33, 4), dtype=np.float32)
    vm = np.ones((2, 33), dtype=bool)
    s, d = score_kick_quality(lm, vm, "low_whip")
    assert s is None, (s, d)

    print("kick_quality self-check OK")


if __name__ == "__main__":
    _demo()
