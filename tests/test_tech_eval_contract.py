# -*- coding: utf-8 -*-
"""
tech_eval 指标结构化契约回归（YOLO 迁移 S4 / Issue #11）。

为什么需要它
------------
评估链路要能诚实输出「能评估什么、不能评估什么」。本期给每个 ``IndicatorResult`` 补
结构化契约字段：

  - ``status`` / ``reason``（原有）
  - ``required_landmarks``：该指标依赖的 BlazePose33 关键点名（声明）
  - ``missing_landmarks``：运行时缺失/后端不支持的关键点名
  - ``backend``：数据来源后端（本期主链路恒为 mediapipe）

本测试验证：① ``evaluate_video_full`` 输出的每个指标都含上述五字段；② 单指标函数
（``eval_*``）经装配后带契约字段；③ 运行时缺失能被如实声明（人为遮挡脚部后，依赖脚部的
指标 ``missing_landmarks`` 含脚跟/脚尖）。

确定性
------
通过 ``tests/golden_harness.py`` 回放已提交的 ``(T,33,4)`` fixture，不依赖 MediaPipe 推理。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_tech_eval_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.tech_eval import (  # noqa: E402
    BACKEND_MEDIAPIPE,
    COG_REQUIRED_INDICES,
    _attach_contract,
    eval_cog_side,
    evaluate_video_full,
    to_jsonable,
)
from core.pose_features import derive_valid_mask  # noqa: E402
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
STUDENT = "golden://student.mp4"
FPS = 30.0

# evaluate_video_full 输出里属于 IndicatorResult 的键。
_INDICATORS = ("cog_side", "cog_front", "cog_final", "cog_com", "retract_speed", "force_sequence", "wrist_angle")
# 每个指标必须带的契约字段。
_CONTRACT_KEYS = ("status", "reason", "required_landmarks", "missing_landmarks", "backend")


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


@pytest.fixture(scope="module")
def student() -> np.ndarray:
    return _load_raw("student_raw.npz")


@pytest.fixture(scope="module")
def registered_student():
    H.clear_registry()
    H.register_video(STUDENT, _load_raw("student_raw.npz"), fps=FPS)
    yield
    H.clear_registry()


@pytest.fixture(scope="module")
def full_result(registered_student) -> dict:
    with H.replay_context():
        return evaluate_video_full(STUDENT, pose_variant="full", stance="left", view_hint="auto")


# --------------------------------------------------------------------------- #
# 1) evaluate_video_full：每个指标都含 status/reason/required_landmarks/
#    missing_landmarks/backend
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("indicator", _INDICATORS)
def test_indicator_has_contract_fields(full_result, indicator):
    got = full_result.get(indicator)
    if got is None:
        # cog_com 在某些段缺失时可能为 None；其余指标必须存在。
        assert indicator == "cog_com"
        return
    for key in _CONTRACT_KEYS:
        assert key in got, f"指标 {indicator} 缺少契约字段 {key}"
    assert got["backend"] == BACKEND_MEDIAPIPE
    assert isinstance(got["required_landmarks"], list)
    assert len(got["required_landmarks"]) > 0
    assert isinstance(got["missing_landmarks"], list)
    # missing 必须是 required 的子集。
    assert set(got["missing_landmarks"]).issubset(set(got["required_landmarks"]))


# --------------------------------------------------------------------------- #
# 2) 单指标经 _attach_contract 后带契约字段（不改 status/reason/detail）
# --------------------------------------------------------------------------- #
def test_attach_contract_preserves_status_and_adds_fields(student):
    mask = derive_valid_mask(student)
    raw = eval_cog_side(student, fps=FPS, valid_mask=mask)
    attached = _attach_contract(raw, COG_REQUIRED_INDICES, mask=mask)

    # status/reason/detail 不变。
    assert attached.status == raw.status
    assert attached.reason == raw.reason
    assert to_jsonable(attached.detail) == to_jsonable(raw.detail)
    # 契约字段补齐。
    assert attached.backend == BACKEND_MEDIAPIPE
    assert len(attached.required_landmarks) == len(COG_REQUIRED_INDICES)
    assert "left_heel" in attached.required_landmarks


# --------------------------------------------------------------------------- #
# 3) 运行时缺失被如实声明：人为遮挡脚部 → 依赖脚部的指标 missing 含脚跟/脚尖
# --------------------------------------------------------------------------- #
def test_runtime_missing_landmarks_reported(student):
    occluded = student.copy()
    # 把脚跟/脚尖（29-32）的 visibility 置 0（全程不可见）。
    occluded[:, 29:33, 3] = 0.0
    mask = derive_valid_mask(occluded)
    raw = eval_cog_side(occluded, fps=FPS, valid_mask=mask)
    attached = _attach_contract(raw, COG_REQUIRED_INDICES, mask=mask)

    for name in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
        assert name in attached.missing_landmarks
    # 未遮挡的核心点不应出现在 missing 里。
    assert "left_shoulder" not in attached.missing_landmarks


# --------------------------------------------------------------------------- #
# 4) JSON 可序列化：契约字段（tuple）经 to_jsonable 后是 list，可写 CSV/JSONL
# --------------------------------------------------------------------------- #
def test_contract_fields_jsonable(full_result):
    cog = full_result.get("cog_final")
    assert cog is not None
    j = to_jsonable(cog)
    assert isinstance(j["required_landmarks"], list)
    assert isinstance(j["missing_landmarks"], list)
    assert isinstance(j["backend"], str)
