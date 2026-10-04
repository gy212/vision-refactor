from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import action_compare


def _features_from_energy(energy: np.ndarray) -> np.ndarray:
    values = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(energy, dtype=np.float32)]
    )
    return np.broadcast_to(values[:, None, None], (values.size, 22, 2)).copy()


def _multi_activity_energy(*, seconds: int = 120, fps: int = 30) -> np.ndarray:
    energy = np.full(seconds * fps - 1, 0.002, dtype=np.float32)
    for start_s in range(5, seconds - 5, 6):
        start = start_s * fps
        energy[start : start + int(1.5 * fps)] = 0.2
    return energy


def test_long_multi_activity_video_requires_manual_review() -> None:
    analysis = action_compare.analyze_template_auto_crop(
        _multi_activity_energy(),
        fps=30.0,
    )

    assert analysis.requires_review is True
    assert analysis.duration_s == pytest.approx(120.0)
    assert analysis.selected_duration_s < 5.0
    assert len(analysis.activity_groups) > 1


def test_short_single_action_keeps_existing_auto_crop_behavior() -> None:
    fps = 30
    energy = np.full(12 * fps - 1, 0.002, dtype=np.float32)
    energy[4 * fps : 6 * fps] = 0.2

    analysis = action_compare.analyze_template_auto_crop(energy, fps=float(fps))

    assert analysis.requires_review is False
    assert analysis.start_frame < 4 * fps
    assert analysis.end_frame > 6 * fps - 1


def test_under_30_seconds_is_not_blocked_even_with_multiple_groups() -> None:
    fps = 30
    energy = np.full(20 * fps - 1, 0.002, dtype=np.float32)
    energy[2 * fps : 3 * fps] = 0.2
    energy[10 * fps : 11 * fps] = 0.2

    analysis = action_compare.analyze_template_auto_crop(energy, fps=float(fps))

    assert len(analysis.activity_groups) == 2
    assert analysis.selected_duration_s < 5.0
    assert analysis.requires_review is False


def test_create_template_rejects_suspicious_long_auto_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy = _multi_activity_energy()
    features = _features_from_energy(energy)
    out_path = tmp_path / "unexpected.npz"
    monkeypatch.setattr(
        action_compare,
        "_extract_pose_features",
        lambda *_args, **_kwargs: (features, 30.0, None),
    )

    with pytest.raises(action_compare.TemplateAutoCropReviewRequired) as exc_info:
        action_compare.create_template_from_video("long.mp4", out_path=out_path)

    assert "startFrame/endFrame" in str(exc_info.value)
    assert not out_path.exists()


def test_explicit_range_overrides_review_and_records_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy = _multi_activity_energy()
    features = _features_from_energy(energy)
    out_path = tmp_path / "manual.npz"
    monkeypatch.setattr(
        action_compare,
        "_extract_pose_features",
        lambda *_args, **_kwargs: (features, 30.0, None),
    )

    result = action_compare.create_template_from_video(
        "long.mp4",
        start=90,
        end=300,
        out_path=out_path,
    )

    with np.load(result, allow_pickle=True) as data:
        meta = dict(data["meta"].item())
        assert data["features"].shape == (211, 22, 2)
    assert meta["crop_selection"] == "manual"
    assert meta["auto_review_required"] is True
    assert meta["selected_retained_ratio"] == pytest.approx(211 / features.shape[0])
