# -*- coding: utf-8 -*-
"""core.foundation.prefs：保留未知键、非法值回退默认、旧键回退、并发写不丢键、偏好隔离。"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from core import paths as paths_facade
from core.foundation import paths, prefs
from core.foundation.prefs import (
    CAMERA_SELECTION,
    DEFAULT_EXAM_ROI,
    EXAM_MIN_RECORD_S,
    EXAM_ROI,
    EXAM_ROSTER_PATH,
    PreferenceStore,
    PrefKey,
    load_camera_selection,
    load_template_pools,
    save_camera_selection,
    save_template_pools,
)


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_update_preserves_unknown_keys(tmp_path):
    target = tmp_path / "user_prefs.json"
    _write(target, {"未知键": {"嵌套": [1, 2]}, "record_dir": "D:/rec"})
    store = PreferenceStore(lambda: target)

    store.set(EXAM_ROSTER_PATH, "D:/名单.xlsx")

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {"未知键": {"嵌套": [1, 2]}, "record_dir": "D:/rec", "exam_roster_path": "D:/名单.xlsx"}


@pytest.mark.parametrize(
    "raw",
    [
        [0.5, 0.1, 0.4, 0.9],  # x0 >= x1
        [0.2, 0.1, 0.8, 1.5],  # 越界
        [0.2, 0.1, 0.8],  # 少一个分量
        "0.2,0.1,0.8,0.9",
        None,
        {"x0": 0.2},
    ],
)
def test_invalid_roi_falls_back_to_default(tmp_path, raw):
    target = tmp_path / "user_prefs.json"
    _write(target, {"exam_roi_norm": raw})
    assert PreferenceStore(lambda: target).get(EXAM_ROI) == DEFAULT_EXAM_ROI


def test_valid_roi_roundtrip_is_written_as_list(tmp_path):
    target = tmp_path / "user_prefs.json"
    store = PreferenceStore(lambda: target)
    store.set(EXAM_ROI, (0.3, 0.2, 0.7, 0.9))
    assert json.loads(target.read_text(encoding="utf-8"))["exam_roi_norm"] == [0.3, 0.2, 0.7, 0.9]
    assert store.get(EXAM_ROI) == (0.3, 0.2, 0.7, 0.9)


def test_invalid_float_and_corrupt_file_fall_back(tmp_path):
    target = tmp_path / "user_prefs.json"
    _write(target, {"exam_min_record_s": "abc"})
    assert PreferenceStore(lambda: target).get(EXAM_MIN_RECORD_S) == 3.0
    target.write_text("{not json", encoding="utf-8")
    assert PreferenceStore(lambda: target).raw() == {}
    _write(target, [1, 2, 3])
    assert PreferenceStore(lambda: target).get(EXAM_MIN_RECORD_S) == 3.0


def test_template_pools_fall_back_to_legacy_single_keys(tmp_path, monkeypatch):
    target = tmp_path / "user_prefs.json"
    monkeypatch.setattr(prefs, "_prefs_path", lambda: target)
    _write(target, {
        "auto_compare_front_template": " D:/front.npz ",
        "auto_compare_side_template": "",
        "auto_compare_side_templates": ["", "  "],
    })
    assert load_template_pools() == ([Path("D:/front.npz")], [])

    save_template_pools(side=["D:/s1.npz", Path("D:/s2.npz")])
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["auto_compare_side_templates"] == ["D:/s1.npz", str(Path("D:/s2.npz"))]
    assert "auto_compare_front_templates" not in data  # 只写传入的一侧
    assert data["auto_compare_front_template"] == " D:/front.npz "  # 旧键只读不写，保留原值
    assert load_template_pools() == ([Path("D:/front.npz")], [Path("D:/s1.npz"), Path("D:/s2.npz")])


def test_concurrent_writes_do_not_lose_keys(tmp_path):
    target = tmp_path / "user_prefs.json"
    keys = [PrefKey(f"k{i}", 0, int) for i in range(16)]
    start = threading.Barrier(len(keys))

    def worker(key: PrefKey[int], value: int) -> None:
        start.wait()
        for _ in range(5):
            PreferenceStore(lambda: target).set(key, value)

    threads = [threading.Thread(target=worker, args=(k, i + 1)) for i, k in enumerate(keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {f"k{i}": i + 1 for i in range(16)}


def test_unwritable_path_is_best_effort(tmp_path):
    store = PreferenceStore(lambda: tmp_path)  # 目标是目录，写不进去
    store.set(CAMERA_SELECTION, (1, None))
    assert store.get(CAMERA_SELECTION) == (None, None)


def test_prefs_path_is_resolved_on_every_call(tmp_path, monkeypatch):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    current = [first]
    monkeypatch.setattr(prefs, "_prefs_path", lambda: current[0])
    save_camera_selection(1, 0)
    current[0] = second
    assert load_camera_selection() == (None, None)
    save_camera_selection(2, None)
    assert json.loads(first.read_text(encoding="utf-8"))["camera_selection"]["primary_index"] == 1
    assert json.loads(second.read_text(encoding="utf-8"))["camera_selection"]["primary_index"] == 2


def test_facade_only_forwards():
    assert paths_facade.load_camera_selection is prefs.load_camera_selection
    assert paths_facade.save_record_dir is prefs.save_record_dir
    assert paths_facade.repo_root is paths.repo_root
    assert not hasattr(paths_facade, "_prefs_path")


def test_isolation_covers_all_keys_and_leaves_repo_prefs_untouched():
    """conftest 隔离下读写摄像头选择、模板池、考试 ROI，仓库根的 user_prefs.json 不被改动。"""
    real = paths.repo_root() / "user_prefs.json"
    before = real.read_bytes() if real.exists() else None

    save_camera_selection(3, 1)
    save_template_pools(front=["D:/f.npz"], side=["D:/s.npz"])
    PreferenceStore().update({EXAM_ROI: (0.1, 0.1, 0.9, 0.9), EXAM_ROSTER_PATH: "D:/r.xlsx"})
    prefs.save_record_dir("D:/rec")

    assert load_camera_selection() == (3, 1)
    assert load_template_pools() == ([Path("D:/f.npz")], [Path("D:/s.npz")])
    assert PreferenceStore().get(EXAM_ROI) == (0.1, 0.1, 0.9, 0.9)
    assert prefs._prefs_path() != real
    after = real.read_bytes() if real.exists() else None
    assert after == before
