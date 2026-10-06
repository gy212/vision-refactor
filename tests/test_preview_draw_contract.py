from __future__ import annotations

import numpy as np

from core.pose import render  # P2：绘制迁入 core/pose/render.py


def test_render_pose_classifies_raw_and_draws_smoothed(monkeypatch) -> None:
    raw_pose = object()
    smoothed_pose = object()
    seen: dict[str, object] = {}

    def classify(landmarks, _width, _height):
        seen["classified"] = landmarks
        return ["RAW_ACTION"]

    def draw_pose(_frame, landmarks, *, draw_face):
        seen["drawn"] = landmarks
        seen["draw_face"] = draw_face

    monkeypatch.setattr(render, "classify_pose_actions", classify)
    monkeypatch.setattr(render, "draw_skeleton", draw_pose)
    monkeypatch.setattr(render, "draw_joint_angles", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(render, "draw_hands", lambda *_args, **_kwargs: None)

    _frame, actions = render.render_pose(
        np.zeros((8, 8, 3), dtype=np.uint8),
        smoothed_pose,
        [],
        draw_face=False,
        draw_joint_angles_overlay=False,
        action_pose_landmarks=raw_pose,
    )

    assert actions == ["RAW_ACTION"]
    assert seen == {
        "classified": raw_pose,
        "drawn": smoothed_pose,
        "draw_face": False,
    }
