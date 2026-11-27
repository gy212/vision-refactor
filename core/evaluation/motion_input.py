"""准备动作识别共用的时间、身体尺度与原始护位参照窗口。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.rules.feedback_config import FeedbackConfig
from core.evaluation.stance_windows import stable_guard_windows, stance_window_edge
from core.foundation.geometry import torso_length


@dataclass(frozen=True)
class MotionInput:
    times: np.ndarray
    torso: np.ndarray
    stable: tuple[np.ndarray, ...]


def prepare_motion_input(series: dict, cfg: FeedbackConfig) -> MotionInput:
    """选源与执行共享同一原始参照，不运行任何动作事件识别。"""
    lm, mask = series["landmarks"], series["valid_mask"]
    times = np.asarray(series["times"], dtype=float)
    torso = torso_length(lm, mask, "visible_side_chain")
    edge = stance_window_edge(times, cfg)
    if edge is None:
        return MotionInput(times, torso, ())
    return MotionInput(times, torso, stable_guard_windows(lm, mask, torso, edge, cfg))
