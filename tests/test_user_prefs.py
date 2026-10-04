# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

from core import paths


def _use_prefs_path(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(paths, "_prefs_path", lambda: path)


def test_camera_selection_roundtrip_preserves_other_preferences(
    monkeypatch, tmp_path: Path
) -> None:
    prefs_path = tmp_path / "user_prefs.json"
    prefs_path.write_text(
        json.dumps(
            {
                "record_dir": "D:/recordings",
                "exam_roster_path": "D:/roster.xlsx",
            }
        ),
        encoding="utf-8",
    )
    _use_prefs_path(monkeypatch, prefs_path)

    paths.save_camera_selection(2, 1)

    assert paths.load_camera_selection() == (2, 1)
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["record_dir"] == "D:/recordings"
    assert payload["exam_roster_path"] == "D:/roster.xlsx"
    assert payload["camera_selection"] == {
        "primary_index": 2,
        "secondary_index": 1,
    }


def test_camera_selection_rejects_invalid_indices(
    monkeypatch, tmp_path: Path
) -> None:
    prefs_path = tmp_path / "user_prefs.json"
    _use_prefs_path(monkeypatch, prefs_path)

    for primary, secondary in (
        (True, -1),
        ("1", 1.5),
        (None, None),
    ):
        prefs_path.write_text(
            json.dumps(
                {
                    "camera_selection": {
                        "primary_index": primary,
                        "secondary_index": secondary,
                    }
                }
            ),
            encoding="utf-8",
        )
        assert paths.load_camera_selection() == (None, None)


def test_camera_selection_write_failure_is_best_effort(
    monkeypatch, tmp_path: Path
) -> None:
    _use_prefs_path(monkeypatch, tmp_path)

    paths.save_camera_selection(1, None)

    assert paths.load_camera_selection() == (None, None)
