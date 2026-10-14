# -*- coding: utf-8 -*-
"""
规则可用性 / 结构化结论回归（YOLO 迁移 S4 / Issue #11；D-3 起针对唯一的规则目录 ``sanda_v5``）。

为什么需要它
------------
下游（考试结果、batch 报告）只读结构化字段判断一个检查项的结论，不解析中文 ``detail``：

  - ``state ∈ {pass, violation, unable, pending_rule}``
  - ``unable`` 时 ``skip_reason ∈ {missing_landmarks, insufficient_valid_frames, phase_not_found}``
  - ``required_landmarks`` / ``required_capabilities`` / ``missing_landmarks``

本测试覆盖每种 ``skip_reason`` 与「后端能力受限 → 缺点检查项 unable」的判定，并守住：不传
``supported_capabilities``（MediaPipe full）时不产生 backend 级缺点；取手按左右式，只声明前手的点。

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pose.validity import COCO17_SUPPORTED_CAPABILITIES, landmarks_missing_for_capabilities  # noqa: E402
from core.evaluation import FeedbackConfig, judge_view, series_from_raw  # noqa: E402
from core.evaluation.rules.spec import (  # noqa: E402
    RULE_STATE_UNABLE,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_MISSING_LANDMARKS,
    SKIP_PHASE_NOT_FOUND,
)
from core.pose import (  # noqa: E402
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


def _judge(seq: np.ndarray, *, caps=None) -> dict:
    """左式实战式在正面的开始阶段各项结论（以检查项编码为键）。"""
    series = series_from_raw(seq, {"width": 100, "height": 100, "fps": 30.0, "start_frame": 0})
    checks, _diag = judge_view(series, "front", "stance", "left", FeedbackConfig(), supported_capabilities=caps)
    return {c.code: c for c in checks if c.phase == "start"}


def _measured(by_code: dict) -> dict:
    return {code: c for code, c in by_code.items() if c.state != "pending_rule"}


# --------------------------------------------------------------------------- #
# 1) 有效帧充足：给出 pass/violation，skip_reason=None
# --------------------------------------------------------------------------- #
def test_judged_when_enough_valid_frames():
    by_code = _measured(_judge(_make_seq(n_valid=10, n_invalid=0)))
    assert {"stance_elbow", "stance_height", "stance_width", "stance_knee", "stance_toes"} <= set(by_code)
    for c in by_code.values():
        assert c.state in {"pass", "violation"}
        assert c.skip_reason is None
        # 声明字段必须非空，且 missing 在 MediaPipe full（默认）下为空。
        assert len(c.required_landmarks) > 0
        assert len(c.required_capabilities) > 0
        assert c.missing_landmarks == ()


# --------------------------------------------------------------------------- #
# 2) skip_reason=phase_not_found：全程不可见，找不到实战式阶段
# --------------------------------------------------------------------------- #
def test_unable_phase_not_found_when_no_valid_frames():
    by_code = _measured(_judge(_make_seq(n_valid=0, n_invalid=10)))
    assert by_code
    for c in by_code.values():
        assert c.state == RULE_STATE_UNABLE
        assert c.skip_reason == SKIP_PHASE_NOT_FOUND
        assert c.valid_frames == 0
        assert "未找到该阶段" in c.detail  # 中文保留供 UI。


# --------------------------------------------------------------------------- #
# 3) skip_reason=insufficient_valid_frames：阶段找到了，但该项所需的点不可见
# --------------------------------------------------------------------------- #
def test_unable_insufficient_valid_frames():
    seq = _make_seq(n_valid=10, n_invalid=0)
    seq[:, [MOUTH_L, MOUTH_R], 3] = 0.0
    by_code = _judge(seq)
    rear = by_code["stance_rear_guard"]
    assert rear.state == RULE_STATE_UNABLE
    assert rear.skip_reason == SKIP_INSUFFICIENT_VALID_FRAMES
    assert rear.valid_frames == 0 < rear.total_frames
    assert by_code["stance_elbow"].state in {"pass", "violation"}


# --------------------------------------------------------------------------- #
# 4) skip_reason=missing_landmarks：后端能力受限（COCO17）→ 缺点检查项 unable
# --------------------------------------------------------------------------- #
def test_unable_missing_landmarks_under_coco17():
    by_code = _judge(_make_seq(n_valid=10, n_invalid=0), caps=COCO17_SUPPORTED_CAPABILITIES)

    # 依赖 feet（脚跟/脚尖）的检查项在 COCO17 下结构性缺失。
    toes = by_code["stance_toes"]
    assert toes.state == RULE_STATE_UNABLE
    assert toes.skip_reason == SKIP_MISSING_LANDMARKS
    assert set(toes.missing_landmarks) == {"left_heel", "left_foot_index", "right_heel", "right_foot_index"}

    # 依赖 mouth（嘴角）的检查项也缺点。
    rear = by_code["stance_rear_guard"]
    assert rear.state == RULE_STATE_UNABLE and rear.skip_reason == SKIP_MISSING_LANDMARKS
    assert {"mouth_left", "mouth_right"} <= set(rear.missing_landmarks)

    # 仅依赖 arms/legs/face_center 的检查项在 COCO17 下仍可判定。
    for code in ("stance_elbow", "stance_knee", "stance_width", "stance_height"):
        c = by_code[code]
        assert c.state in {"pass", "violation"}
        assert c.skip_reason is None and c.missing_landmarks == ()


# --------------------------------------------------------------------------- #
# 5) required_landmarks / required_capabilities 声明正确；取手按左右式（D-3）
# --------------------------------------------------------------------------- #
def test_required_landmarks_and_capabilities_declared():
    by_code = _judge(_make_seq(n_valid=10, n_invalid=0))

    toes = by_code["stance_toes"]
    assert set(toes.required_landmarks) == {"left_heel", "left_foot_index", "right_heel", "right_foot_index"}
    assert toes.required_capabilities == ("feet",)

    # 左式只看前手（左手），不再"任一手满足"。
    elbow = by_code["stance_elbow"]
    assert set(elbow.required_landmarks) == {"left_shoulder", "left_elbow", "left_wrist"}
    assert elbow.required_capabilities == ("arms",)


# --------------------------------------------------------------------------- #
# 6) 向后兼容：不传 supported_capabilities 时不产生 backend 级缺点
# --------------------------------------------------------------------------- #
def test_backward_compatible_no_backend_gating_by_default():
    for c in _judge(_make_seq(n_valid=10, n_invalid=0)).values():
        # 默认（MediaPipe full）不应出现 missing_landmarks 级 unable。
        assert c.skip_reason != SKIP_MISSING_LANDMARKS
        assert c.missing_landmarks == ()


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
