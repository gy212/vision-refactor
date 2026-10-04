# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

from core.presence_gate import PresenceGate, PresenceGateConfig, hip_midpoint_in_roi


def test_enter_stable_requires_duration() -> None:
    gate = PresenceGate(PresenceGateConfig(enter_stable_s=0.8))
    gate.begin_wait_enter()
    assert gate.update(True, 0.0, mode="wait_enter") == []
    assert gate.update(True, 0.5, mode="wait_enter") == []
    ev = gate.update(True, 0.8, mode="wait_enter")
    assert len(ev) == 1 and ev[0].kind == "enter_stable"
    # 不重复触发
    assert gate.update(True, 1.5, mode="wait_enter") == []


def test_enter_resets_on_absence() -> None:
    gate = PresenceGate(PresenceGateConfig(enter_stable_s=0.8))
    gate.begin_wait_enter()
    gate.update(True, 0.0, mode="wait_enter")
    gate.update(False, 0.5, mode="wait_enter")
    assert gate.update(True, 1.0, mode="wait_enter") == []
    ev = gate.update(True, 1.8, mode="wait_enter")
    assert len(ev) == 1


def test_empty_stable_respects_min_record() -> None:
    gate = PresenceGate(
        PresenceGateConfig(empty_hold_s=2.0, min_record_s=3.0)
    )
    gate.begin_recording(0.0)
    # 无人但录制不足 3s
    assert gate.update(False, 1.0, mode="recording") == []
    assert gate.update(False, 2.5, mode="recording") == []
    # 录制够了但空场不足 2s（从 2.5 起算空场的话已经... wait empty since 1.0)
    # empty since 1.0, at 3.0: empty=2.0, record=3.0 → fire
    ev = gate.update(False, 3.0, mode="recording")
    assert len(ev) == 1 and ev[0].kind == "empty_stable"


def test_presence_resets_empty_timer() -> None:
    gate = PresenceGate(
        PresenceGateConfig(empty_hold_s=2.0, min_record_s=1.0)
    )
    gate.begin_recording(0.0)
    gate.update(False, 0.5, mode="recording")
    gate.update(True, 1.0, mode="recording")  # reset empty
    assert gate.update(False, 2.5, mode="recording") == []  # empty 1.5s
    assert gate.update(False, 3.5, mode="recording") == []  # empty 2.0s? from 2.5 → 1.0s
    ev = gate.update(False, 4.5, mode="recording")  # empty from 2.5 = 2.0s
    assert len(ev) == 1


def test_hip_midpoint_in_roi() -> None:
    roi = (0.2, 0.1, 0.8, 0.9)

    def lm(x, y, v=0.9):
        return SimpleNamespace(x=x, y=y, visibility=v)

    # 33-like list with hips at 23/24
    landmarks = [lm(0, 0, 0)] * 33
    landmarks[23] = lm(0.4, 0.5)
    landmarks[24] = lm(0.6, 0.5)
    assert hip_midpoint_in_roi(landmarks, roi) is True
    landmarks[23] = lm(0.05, 0.5)
    landmarks[24] = lm(0.05, 0.5)
    assert hip_midpoint_in_roi(landmarks, roi) is False
    landmarks[23] = lm(0.4, 0.5, 0.1)
    landmarks[24] = lm(0.6, 0.5, 0.1)
    assert hip_midpoint_in_roi(landmarks, roi, vis_thr=0.3) is False
