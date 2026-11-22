"""按基础动作所用肢体与方向约束组织公共运动事件识别。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.hook_events import hook_motion_landmarks, observe_hook
from core.evaluation.motion_events import MotionEvent
from core.evaluation.punch_events import observe_straight, straight_motion_landmarks


@dataclass(frozen=True)
class ActionEvent:
    segment: str
    motion: MotionEvent

    def to_record(self, times: np.ndarray) -> dict:
        """保留真实观测的动作及其起止时间，不用连续帧段猜动作次数。"""
        event = self.motion
        return {
            "segment": self.segment,
            "start": float(times[event.start]),
            "peak": float(times[event.peak]),
            "end": float(times[event.end]),
            "ending": event.ending,
        }


def motion_landmarks(segment: str, stance: str) -> tuple[int, int]:
    """基础动作位移使用的固定点与运动点，供识别和机位可见性选择共用。"""
    role = "front" if segment.startswith("front_") else "rear"
    if segment.endswith("_straight"):
        return straight_motion_landmarks(role, stance)
    if segment.endswith("_hook"):
        return hook_motion_landmarks(role, stance)
    shoulder, _, wrist, hip, _, ankle, *_ = sides_for_stance(role, stance)
    return (hip, ankle) if ("kick" in segment or segment == "pad_step") else (shoulder, wrist)


def action_events(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    segments: tuple[str, ...],
    stance: str,
    cfg,
    stable: list[np.ndarray],
    diagnostics: dict,
    view: str,
) -> list[ActionEvent]:
    """按基础动作分派唯一识别方法，直拳与摆拳在单拳和组合中共用各自观测策略。"""
    events: list[ActionEvent] = []
    for segment in segments:
        role = "front" if segment.startswith("front_") else "rear"
        if segment.endswith("_straight"):
            observation = observe_straight(
                lm, mask, times, torso, baseline, stable, role, stance, cfg, view
            )
        elif segment.endswith("_hook"):
            observation = observe_hook(
                lm, mask, times, torso, baseline, stable, role, stance, cfg, view
            )
        else:
            raise ValueError(f"未注册事件观测器的动作段：{segment}")
        if observation.reason:
            diagnostics["reason"] = observation.reason
        events.extend(ActionEvent(segment, event) for event in observation.events)
    return sorted(events, key=lambda event: event.motion.peak)
