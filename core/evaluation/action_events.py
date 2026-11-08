"""按基础动作所用肢体与方向约束组织公共运动事件识别。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.motion_events import EventPolicy, MotionEvent, excursion_events
from core.evaluation.measures.motion import normalized_motion
from core.evaluation.punch_events import observe_straight


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
    shoulder, _, wrist, hip, _, ankle, *_ = sides_for_stance(role, stance)
    return (hip, ankle) if "kick" in segment else (shoulder, wrist)


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
    """按基础动作分派唯一识别方法，直拳单独练习和编入组合时使用同一观测策略。"""
    events: list[ActionEvent] = []
    for segment in segments:
        if segment.endswith("_straight"):
            role = "front" if segment.startswith("front_") else "rear"
            observation = observe_straight(
                lm, mask, times, torso, baseline, stable, role, stance, cfg, view
            )
            if observation.reason:
                diagnostics["reason"] = observation.reason
            events.extend(ActionEvent(segment, event) for event in observation.events)
            continue
        anchor, moving = motion_landmarks(segment, stance)
        relative = normalized_motion(lm, mask, torso, anchor, moving)
        if not np.isfinite(relative[baseline]).all():
            continue
        displacement = relative - np.median(relative[baseline], axis=0)
        distance = np.linalg.norm(displacement, axis=1)
        found = excursion_events(
            distance,
            times,
            baseline,
            cfg,
            rise=cfg.motion_distance,
            fall=cfg.return_distance,
            policy=EventPolicy(start_at_rest=True, inclusive_return=True, allow_held=False),
        )
        events.extend(ActionEvent(segment, event) for event in found)
    return sorted(events, key=lambda event: event.motion.peak)
