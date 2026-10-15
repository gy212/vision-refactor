# -*- coding: utf-8 -*-
"""摄像头控制器：预热规格声明、枚举结果回到 Tk 线程、偏好设备的恢复与持久化（取代 test_app_ui_lifecycle 的同名用例）。"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from apps.tk.controllers import cameras as cameras_mod
from apps.tk.views.capture_panel import NO_SECOND_CAMERA
from core.media import CameraDevice as CameraEntry
from core.media import CameraWarmupError, CatalogSnapshot, CaptureSpec
from core.workflows import EXAM, PRACTICE
from tests.tk_app_doubles import FakeLive, SyncPool, build_app, root  # noqa: F401


def _spec(index: int, **kwargs) -> CaptureSpec:
    """各模式统一的 1080p ``require`` 规格（D-17）。"""
    return CaptureSpec(index, 1920, 1080, policy="require", **kwargs)


def _warm_app(root, *, second_label: str = "摄像头 1", entries=None):
    pool = SyncPool()
    app = build_app(root, pool=pool, live=FakeLive())
    cams = app.cameras
    cams.entries = entries or [CameraEntry(label="摄像头 0", index=0), CameraEntry(label="摄像头 1", index=1)]
    cams.source.select_camera(0)
    app.window.capture.camera_choice_2.set(second_label)
    return app, pool


def test_single_camera_selection_declares_primary_spec_only(root):
    app, pool = _warm_app(root, second_label=NO_SECOND_CAMERA)
    app.cameras.sync_warmup()
    assert pool.sync_calls == [(_spec(0), None)]


def test_two_camera_selection_declares_both_specs(root):
    app, pool = _warm_app(root)
    app.cameras.sync_warmup()
    assert pool.sync_calls == [(_spec(0), _spec(1))]


def test_same_camera_selection_never_warms_second_role(root):
    app, pool = _warm_app(root, second_label="摄像头 0")
    app.cameras.sync_warmup()
    assert pool.sync_calls == [(_spec(0), None)]


@pytest.mark.parametrize("mode", [None, EXAM, PRACTICE])
def test_every_mode_declares_1080p_require_specs(root, mode):
    app, pool = _warm_app(root)
    if mode is not None:
        app.modes.enter(mode)
    app.cameras.sync_warmup()
    app.modes.leave()
    # D-6、D-17：普通、考试、学生练习都严格要求 1080p，规格在声明时冻结。
    assert pool.sync_calls == [
        (CaptureSpec(0, 1920, 1080, policy="require"), CaptureSpec(1, 1920, 1080, policy="require"))
    ]


def test_enumerated_device_names_are_frozen_into_specs_for_open_check(root):
    app, pool = _warm_app(root, second_label="摄像头 1: Side", entries=[
        CameraEntry(label="摄像头 0: Front", index=0, name="Front"),
        CameraEntry(label="摄像头 1: Side", index=1, name="Side"),
    ])
    app.cameras.sync_warmup()
    # V-3：打开时在采集进程内复核设备名。
    assert pool.sync_calls == [
        (_spec(0, expected_name="Front"), _spec(1, expected_name="Side"))
    ]


def test_running_session_blocks_warmup_changes_unless_allowed(root):
    app, pool = _warm_app(root)
    app.session.live.state = "running"
    app.cameras.sync_warmup()
    assert pool.sync_calls == []
    app.cameras.sync_warmup(allow_running=True)
    assert len(pool.sync_calls) == 1


def test_warmup_rejection_releases_and_reports_status(root):
    app, pool = _warm_app(root)

    def reject(_primary, _secondary):
        raise CameraWarmupError("pool closed")

    pool.sync = reject
    app.cameras.sync_warmup()
    assert pool.release_all_calls == 1
    assert app.window.status.status_var.get() == "摄像头预热失败：pool closed"


def test_process_factory_forwards_frozen_spec(monkeypatch):
    calls = []
    monkeypatch.setattr(cameras_mod, "ProcessCamera", lambda index, **kwargs: calls.append((index, kwargs)))
    stop = threading.Event()
    cameras_mod.open_camera_process(0, width=1280, height=720, stop_event=stop)
    assert calls == [(0, {"width": 1280, "height": 720, "stop_event": stop, "expected_name": None,
                          "policy": "prefer"})]


def test_camera_enumeration_result_is_applied_only_on_tk_thread(root):
    app, _pool = _warm_app(root)
    pending: list[object] = []
    app.cameras.service = SimpleNamespace(refresh=lambda on_done: pending.append(on_done),
                                          prepare=lambda _specs: None)
    entries = [CameraEntry(label="摄像头 9", index=9)]
    app.cameras.start_enumeration()
    assert app.cameras.enumerating.is_set()
    pending[0](CatalogSnapshot(tuple(entries), ok=True))   # 后台线程只投递
    assert app.cameras.entries != entries
    app.ui.pump()
    assert app.cameras.entries == entries
    assert not app.cameras.enumerating.is_set()


def test_camera_enumeration_return_after_close_does_not_touch_tk(root):
    app, _pool = _warm_app(root)
    pending: list[object] = []
    app.cameras.service = SimpleNamespace(refresh=lambda on_done: pending.append(on_done))
    app.cameras.start_enumeration()
    app.ui.close()   # 关窗：dispatcher 进入关闭态
    pending[0](CatalogSnapshot((CameraEntry(label="摄像头 9", index=9),), ok=True))
    app.ui.pump()
    assert [entry.index for entry in app.cameras.entries] == [0, 1]


def _selection_app(root, *, preferred_primary, preferred_secondary):
    app = build_app(root, live=FakeLive())
    app.cameras.preferred_primary = preferred_primary
    app.cameras.preferred_secondary = preferred_secondary
    return app


def test_camera_enumeration_restores_persisted_primary_and_secondary(root):
    app = _selection_app(root, preferred_primary=2, preferred_secondary=1)
    app.cameras.apply_entries([
        CameraEntry(label="摄像头 0", index=0),
        CameraEntry(label="摄像头 1: Side", index=1),
        CameraEntry(label="摄像头 2: Front", index=2),
    ], True)
    capture = app.window.capture
    assert capture.camera_choice.get() == "摄像头 2: Front"
    assert capture.camera_choice_2.get() == "摄像头 1: Side"
    assert app.cameras.source.value == "2"


def test_missing_preferred_cameras_fallback_without_forgetting_then_restore(root):
    app = _selection_app(root, preferred_primary=2, preferred_secondary=1)
    capture = app.window.capture
    app.cameras.apply_entries([CameraEntry(label="摄像头 0", index=0)], True)
    assert capture.camera_choice.get() == "摄像头 0"
    assert capture.camera_choice_2.get() == NO_SECOND_CAMERA
    assert (app.cameras.preferred_primary, app.cameras.preferred_secondary) == (2, 1)

    app.cameras.apply_entries([CameraEntry(label=f"摄像头 {i}", index=i) for i in range(3)], True)
    assert capture.camera_choice.get() == "摄像头 2"
    assert capture.camera_choice_2.get() == "摄像头 1"


def test_explicit_camera_selections_are_persisted(root, monkeypatch):
    app = _selection_app(root, preferred_primary=0, preferred_secondary=None)
    app.cameras.entries = [CameraEntry(label="摄像头 0", index=0), CameraEntry(label="摄像头 1", index=1)]
    saved: list[tuple[int | None, int | None]] = []
    monkeypatch.setattr(cameras_mod, "save_camera_selection", lambda p, s: saved.append((p, s)))
    capture = app.window.capture
    capture.camera_choice.set("摄像头 1")
    app.cameras.primary_selected()
    capture.camera_choice_2.set("摄像头 0")
    app.cameras.secondary_selected()
    capture.camera_choice_2.set(NO_SECOND_CAMERA)
    app.cameras.secondary_selected()
    assert saved == [(1, None), (1, 0), (1, None)]
    assert app.cameras.source.value == "1"


def test_failed_enumeration_keeps_previous_list(root):
    app, pool = _warm_app(root)
    app.cameras.apply_entries([], False)
    assert len(app.cameras.entries) == 2
    assert app.window.status.status_var.get() == "刷新摄像头失败，已保留原有列表。"
    assert len(pool.sync_calls) == 1   # 仍按原选择声明预热


def test_empty_enumeration_clears_camera_source(root):
    app, pool = _warm_app(root)
    app.cameras.apply_entries([], True)
    assert app.cameras.source.kind == "none"
    assert app.window.capture.source_hint_var.get() == "未检测到可用摄像头"
    assert pool.release_all_calls == 1


def test_refresh_keeps_camera_warmups_while_enumerating(root):
    # V-6：刷新不再先放弃预热、等设备释放；兜底探测由 DeviceCatalog 跳过占用的设备。
    app, pool = _warm_app(root)
    refreshed: list[object] = []
    app.cameras.service = SimpleNamespace(refresh=lambda on_done: refreshed.append(on_done), prepare=lambda _s: _never_release())
    app.cameras.refresh()
    assert len(refreshed) == 1 and pool.release_all_calls == 0
    app.cameras.refresh()   # 枚举中忽略
    assert len(refreshed) == 1


def _never_release():
    raise AssertionError("刷新不得放弃预热")
