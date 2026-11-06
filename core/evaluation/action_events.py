"""按基础动作所用肢体与方向约束组织公共运动事件识别。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.evaluation.body_sides import sides_for_stance
from core.evaluation.motion_events import EventPolicy, MotionEvent, excursion_events
from core.evaluation.measures.motion import normalized_motion


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


def action_events(
    lm: np.ndarray,
    mask: np.ndarray,
    times: np.ndarray,
    torso: np.ndarray,
    baseline: np.ndarray,
    segments: tuple[str, ...],
    stance: str,
    cfg,
) -> list[ActionEvent]:
    """普通动作按护位位移识别；直拳共用方向判据，腿法和摆拳保留各自要求。"""
    events: list[ActionEvent] = []
    for segment in segments:
        role = "front" if segment.startswith("front_") else "rear"
        shoulder, _, wrist, hip, _, ankle, *_ = sides_for_stance(role, stance)
        anchor, moving = (hip, ankle) if "kick" in segment else (shoulder, wrist)
        relative = normalized_motion(lm, mask, torso, anchor, moving)
        if not np.isfinite(relative[baseline]).all():
            continue
        displacement = relative - np.median(relative[baseline], axis=0)
        distance = np.linalg.norm(displacement, axis=1)
        straight = "straight" in segment
        found = excursion_events(
            distance,
            times,
            baseline,
            cfg,
            rise=cfg.motion_distance,
            fall=cfg.return_distance,
            policy=EventPolicy(start_at_rest=True, inclusive_return=True, allow_held=straight),
            forward=np.abs(displacement[:, 0]) if straight else None,
            downward=displacement[:, 1] if straight else None,
        )
        events.extend(ActionEvent(segment, event) for event in found)
    return sorted(events, key=lambda event: event.motion.peak)
