"""准备动作识别共用的时间、身体尺度与原始护位参照窗口。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.stance_windows import stable_guard_windows
from core.foundation.geometry import torso_length


@dataclass(frozen=True)
class MotionInput:
    times: np.ndarray
    torso: np.ndarray
    stable: list[np.ndarray]


def prepare_motion_input(series: dict, cfg: FeedbackConfig) -> MotionInput:
    """选源与执行共享同一原始参照，不运行任何动作事件识别。"""
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    torso = torso_length(lm, mask, "visible_side_chain")
    if len(times) < cfg.min_valid * 2 + 1:
        return MotionInput(times, torso, [])
    dt = float(np.median(np.diff(times)))
    if not math.isfinite(dt) or dt <= 0:
        return MotionInput(times, torso, [])
    edge = max(cfg.min_valid, int(math.ceil(cfg.stance_window_seconds / dt)))
    return MotionInput(times, torso, stable_guard_windows(lm, mask, torso, edge, cfg))
