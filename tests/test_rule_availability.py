# -*- coding: utf-8 -*-
"""
规则可用性 / 结构化三态回归（YOLO 迁移 S4 / Issue #11）。

为什么需要它
------------
此前 ``RuleViolation`` 的三态（已评估 / 合格 / 未评估）只靠 ``detail`` 中文字符串拼出
（ （合格） / （有效帧不足，未评估） ），下游无法用稳定字段判断。本期给
``RuleViolation`` 补结构化字段：

  - ``state ∈ {evaluated, skipped}``
  - ``skip_reason ∈ {missing_landmarks, low_confidence, insufficient_valid_frames}``
  - ``required_landmarks`` / ``required_capabilities`` / ``missing_landmarks``

本测试覆盖每种 ``skip_reason`` 与「后端能力受限 → 缺点规则 skipped」的判定，并守住
向后兼容：不传 ``supported_capabilities``（MediaPipe full）时不产生 backend 级缺点。

确定性
------
用合成 ``(T,33,4)`` 关键点（直接控制 visibility 通道来控制有效性），不依赖 MediaPipe 推理。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_rule_availability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pose_features import (  # noqa: E402
    COCO17_SUPPORTED_CAPABILITIES,
    landmarks_missing_for_capabilities,
)
from core.rule_scoring import (  # noqa: E402
    RULE_STATE_EVALUATED,
    RULE_STATE_SKIPPED,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_LOW_CONFIDENCE,
    SKIP_MISSING_LANDMARKS,
    NOSE,
    MOUTH_L,
    MOUTH_R,
    L_SHOULDER,
    R_SHOULDER,
    L_ELBOW,
    R_ELBOW,
    L_WRIST,
    R_WRIST,
    L_HIP,
    R_HIP,
    L_KNEE,
    R_KNEE,
    L_ANKLE,
    R_ANKLE,
    L_HEEL,
    R_HEEL,
    L_FOOT_INDEX,
    R_FOOT_INDEX,
    score_rules,
)

# 站立姿态的合成坐标（仅需 finite + 大致合理，状态分类不依赖具体几何）。
_STANCE_COORDS: dict[int, tuple[float, float]] = {
    NOSE: (0.50, 0.10),
    MOUTH_L: (0.48, 0.13),
    MOUTH_R: (0.52, 0.13),
    L_SHOULDER: (0.42, 0.28),
    R_SHOULDER: (0.58, 0.28),
    L_ELBOW: (0.40, 0.42),
    R_ELBOW: (0.60, 0.42),
    L_WRIST: (0.44, 0.12),
    R_WRIST: (0.56, 0.12),
    L_HIP: (0.45, 0.55),
    R_HIP: (0.55, 0.55),
    L_KNEE: (0.45, 0.75),
    R_KNEE: (0.55, 0.75),
    L_ANKLE: (0.45, 0.93),
    R_ANKLE: (0.55, 0.93),
    L_HEEL: (0.44, 0.95),
    R_HEEL: (0.56, 0.95),
    L_FOOT_INDEX: (0.47, 0.96),
    R_FOOT_INDEX: (0.53, 0.96),
}


def _make_frame(*, valid: bool) -> np.ndarray:
    arr = np.zeros((33, 4), dtype=np.float32)
    for idx, (x, y) in _STANCE_COORDS.items():
        arr[idx, 0] = float(x)
        arr[idx, 1] = float(y)
        arr[idx, 2] = 0.0
        arr[idx, 3] = 0.9 if valid else 0.0
    return arr


def _make_seq(n_valid: int, n_invalid: int) -> np.ndarray:
    frames = [_make_frame(valid=True) for _ in range(int(n_valid))]
    frames += [_make_frame(valid=False) for _ in range(int(n_invalid))]
    return np.stack(frames, axis=0)


def _by_id(rs) -> dict:
    return {v.rule_id: v for v in rs.violations}


# --------------------------------------------------------------------------- #
# 1) state=evaluated：有效帧充足时为已评估，skip_reason=None
# --------------------------------------------------------------------------- #
def test_state_evaluated_when_enough_valid_frames():
    seq = _make_seq(n_valid=10, n_invalid=0)
    rs = score_rules(seq, view="front", action_scope="stance")
    assert len(rs.violations) > 0
    for v in rs.violations:
        assert v.state == RULE_STATE_EVALUATED
        assert v.skip_reason is None
        # 声明字段必须非空，且 missing 在 MediaPipe full（默认）下为空。
        assert len(v.required_landmarks) > 0
        assert len(v.required_capabilities) > 0
        assert v.missing_landmarks == ()


# --------------------------------------------------------------------------- #
# 2) skip_reason=low_confidence：所需点全程不可见（valid_cnt==0）
# --------------------------------------------------------------------------- #
def test_skip_low_confidence_when_no_valid_frames():
    seq = _make_seq(n_valid=0, n_invalid=10)
    rs = score_rules(seq, view="front", action_scope="stance")
    assert len(rs.violations) > 0
    for v in rs.violations:
        assert v.state == RULE_STATE_SKIPPED
        assert v.skip_reason == SKIP_LOW_CONFIDENCE
        assert v.valid_frames == 0
        assert v.penalty == 0
        # detail 中文保留供 UI。
        assert "未评估" in v.detail
    # 整体不应扣分（全部 skipped）。
    assert rs.total_deduction == 0


# --------------------------------------------------------------------------- #
# 3) skip_reason=insufficient_valid_frames：有效帧 0<cnt<min_valid
# --------------------------------------------------------------------------- #
def test_skip_insufficient_valid_frames():
    # 3 个有效帧 < 默认 min_valid=5，但 >0 → insufficient_valid_frames。
    seq = _make_seq(n_valid=3, n_invalid=7)
    rs = score_rules(seq, view="front", action_scope="stance")
    assert len(rs.violations) > 0
    for v in rs.violations:
        assert v.state == RULE_STATE_SKIPPED
        assert v.skip_reason == SKIP_INSUFFICIENT_VALID_FRAMES
        assert 0 < v.valid_frames < 5


# --------------------------------------------------------------------------- #
# 4) skip_reason=missing_landmarks：后端能力受限（COCO17）→ 缺点规则跳过
# --------------------------------------------------------------------------- #
def test_skip_missing_landmarks_under_coco17():
    seq = _make_seq(n_valid=10, n_invalid=0)
    rs = score_rules(
        seq,
        view="front",
        action_scope="both",
        supported_capabilities=COCO17_SUPPORTED_CAPABILITIES,
    )
    by_id = _by_id(rs)

    # 依赖 feet（脚跟/脚尖）的规则在 COCO17 下结构性缺失。
    feet_rule = by_id["stance_feet"]
    assert feet_rule.state == RULE_STATE_SKIPPED
    assert feet_rule.skip_reason == SKIP_MISSING_LANDMARKS
    assert set(feet_rule.missing_landmarks) == {
        "left_heel",
        "left_foot_index",
        "right_heel",
        "right_foot_index",
    }

    # 依赖 mouth（嘴角）的规则也缺点。
    for rid in ("stance_back_arm", "punch_guard"):
        v = by_id[rid]
        assert v.state == RULE_STATE_SKIPPED
        assert v.skip_reason == SKIP_MISSING_LANDMARKS
        assert "mouth_left" in v.missing_landmarks
        assert "mouth_right" in v.missing_landmarks

    # 仅依赖 arms/legs/face_center 的规则在 COCO17 下仍可评估。
    for rid in ("stance_elbow", "stance_knee", "stance_width", "stance_fist_height"):
        v = by_id[rid]
        assert v.state == RULE_STATE_EVALUATED
        assert v.skip_reason is None
        assert v.missing_landmarks == ()

    # 缺点规则不得贡献扣分（不当合格也不当不合格）。
    for rid in ("stance_feet", "stance_back_arm", "punch_guard"):
        assert by_id[rid].penalty == 0


# --------------------------------------------------------------------------- #
# 5) required_landmarks / required_capabilities 声明正确
# --------------------------------------------------------------------------- #
def test_required_landmarks_and_capabilities_declared():
    seq = _make_seq(n_valid=10, n_invalid=0)
    rs = score_rules(seq, view="front", action_scope="both")
    by_id = _by_id(rs)

    feet = by_id["stance_feet"]
    assert set(feet.required_landmarks) == {
        "left_heel",
        "left_foot_index",
        "right_heel",
        "right_foot_index",
    }
    assert feet.required_capabilities == ("feet",)

    elbow = by_id["stance_elbow"]
    assert set(elbow.required_landmarks) == {
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    }
    assert elbow.required_capabilities == ("arms",)


# --------------------------------------------------------------------------- #
# 6) 向后兼容：不传 supported_capabilities 时不产生 backend 级缺点
# --------------------------------------------------------------------------- #
def test_backward_compatible_no_backend_gating_by_default():
    seq = _make_seq(n_valid=10, n_invalid=0)
    rs = score_rules(seq, view="front", action_scope="both")
    for v in rs.violations:
        # 默认（MediaPipe full）不应出现 missing_landmarks 级跳过。
        assert v.skip_reason != SKIP_MISSING_LANDMARKS
        assert v.missing_landmarks == ()


# --------------------------------------------------------------------------- #
# 7) COCO17 能力分组必须和 adapter 映射契约一致：左右眼中心可用，眼细分缺失
# --------------------------------------------------------------------------- #
def test_coco17_eye_capabilities_match_adapter_contract():
    missing = set(
        landmarks_missing_for_capabilities(
            (1, 2, 3, 4, 5, 6),
            COCO17_SUPPORTED_CAPABILITIES,
        )
    )
    assert "left_eye" not in missing
    assert "right_eye" not in missing
    assert {
        "left_eye_inner",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye_outer",
    }.issubset(missing)


def test_coco17_missing_groups_remain_structural_capability_gaps():
    missing = set(
        landmarks_missing_for_capabilities(
            (
                MOUTH_L,
                MOUTH_R,
                L_HEEL,
                R_HEEL,
                L_FOOT_INDEX,
                R_FOOT_INDEX,
                17,
                18,
                19,
                20,
                21,
                22,
            ),
            COCO17_SUPPORTED_CAPABILITIES,
        )
    )
    assert {
        "mouth_left",
        "mouth_right",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
    }.issubset(missing)
