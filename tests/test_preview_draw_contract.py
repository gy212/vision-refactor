from __future__ import annotations

import numpy as np

from core import vision_pipeline


def test_draw_pose_frame_classifies_raw_and_draws_smoothed(monkeypatch) -> None:
    raw_pose = object()
    smoothed_pose = object()
    seen: dict[str, object] = {}

    def classify(landmarks, _width, _height):
        seen["classified"] = landmarks
        return ["RAW_ACTION"]

    def draw_pose(_frame, landmarks, _width, _height, *, draw_face):
        seen["drawn"] = landmarks
        seen["draw_face"] = draw_face

    monkeypatch.setattr(vision_pipeline, "_classify_pose_actions", classify)
    monkeypatch.setattr(vision_pipeline.MediaPipePipeline, "_draw_pose", draw_pose)
    monkeypatch.setattr(
        vision_pipeline.MediaPipePipeline,
        "_draw_joint_angles",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        vision_pipeline.MediaPipePipeline,
        "_draw_hands",
        lambda *_args, **_kwargs: None,
    )

    _frame, actions = vision_pipeline.draw_pose_frame(
        np.zeros((8, 8, 3), dtype=np.uint8),
        smoothed_pose,
        [],
        draw_face=False,
        draw_joint_angles=False,
        action_pose_landmarks=raw_pose,
    )

    assert actions == ["RAW_ACTION"]
    assert seen == {
        "classified": raw_pose,
        "drawn": smoothed_pose,
        "draw_face": False,
    }
