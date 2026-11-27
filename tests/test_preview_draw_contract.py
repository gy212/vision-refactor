from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from core.live import fit_preview_rgb
from core.pose import render  # P2：绘制迁入 core/pose/render.py


def _reference_fit_preview_rgb(frame_bgr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """参考实现：先 BGR→RGB 再按目标区域等比缩放。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pw, ph = target
    if pw > 1 and ph > 1:
        ih, iw = rgb.shape[:2]
        scale = min(pw / iw, ph / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return rgb


def _assert_preview_invariants(bgr: np.ndarray, target: tuple[int, int]) -> None:
    expected = _reference_fit_preview_rgb(bgr, target)
    snapshot = bgr.copy()
    out = fit_preview_rgb(bgr, target)

    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
    assert out.flags.c_contiguous
    assert out.flags.owndata
    assert out.flags.writeable
    assert not np.shares_memory(out, bgr)

    out[0, 0, 0] ^= 0xFF
    assert np.array_equal(bgr, snapshot)


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


@pytest.mark.parametrize(
    ("shape", "target", "strided", "readonly"),
    [
        ((1, 1, 3), (0, 0), False, False),
        ((1, 1, 3), (1, 1), False, True),
        ((1, 1, 3), (17, 13), False, True),
        ((13, 19, 3), (0, 0), False, False),
        ((13, 19, 3), (1, 100), True, True),
        ((13, 19, 3), (100, 1), True, False),
        ((13, 19, 3), (-4, 50), False, True),
        ((13, 19, 3), (19, 13), False, True),
        ((13, 19, 3), (19, 13), True, True),
        ((13, 19, 3), (7, 5), True, True),
        ((7, 5, 3), (19, 13), True, True),
        ((32, 48, 3), (48, 32), True, False),
    ],
)
def test_fit_preview_rgb_edge_cases_zero_diff_and_independent_memory(
    shape: tuple[int, int, int],
    target: tuple[int, int],
    strided: bool,
    readonly: bool,
) -> None:
    rng = np.random.default_rng(12345)
    if strided:
        base = rng.integers(0, 256, size=(shape[0] * 2, shape[1] * 2, 3), dtype=np.uint8)
        bgr = base[::2, ::2, :]
        assert not bgr.flags.c_contiguous
    else:
        bgr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    if readonly:
        bgr.setflags(write=False)
    _assert_preview_invariants(bgr, target)


@settings(max_examples=250, deadline=None)
@given(
    ih=st.integers(min_value=1, max_value=83),
    iw=st.integers(min_value=1, max_value=83),
    target_mode=st.sampled_from(["unlayout", "same", "random"]),
    pw=st.integers(min_value=-2, max_value=120),
    ph=st.integers(min_value=-2, max_value=120),
    strided=st.booleans(),
    readonly=st.booleans(),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fit_preview_rgb_hypothesis_matches_cvt_then_resize_byte_for_byte(
    ih: int,
    iw: int,
    target_mode: str,
    pw: int,
    ph: int,
    strided: bool,
    readonly: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    if strided:
        base = rng.integers(0, 256, size=(ih * 2, iw * 2, 3), dtype=np.uint8)
        bgr = base[::2, ::2, :]
    else:
        bgr = rng.integers(0, 256, size=(ih, iw, 3), dtype=np.uint8)
    if readonly:
        bgr.setflags(write=False)

    if target_mode == "unlayout":
        target = (min(pw, 1), ph) if seed % 2 == 0 else (pw, min(ph, 1))
    elif target_mode == "same":
        target = (iw, ih)
    else:
        target = (pw, ph)

    _assert_preview_invariants(bgr, target)
