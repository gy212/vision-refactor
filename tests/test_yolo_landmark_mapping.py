# -*- coding: utf-8 -*-
"""
YOLO COCO17 → BlazePose33 映射回归（YOLO 迁移 Issue #7 / S2）。

验收目标（对应 Issue #7）
-------------------------
- **边界层**：缺失点（嘴角 9/10、手指 17-22、脚跟脚尖 29-32、眼细分 1/3/4/6）
  在 ``FrameResult.pose33`` 中 ``synthetic=True`` 且 ``visibility=0.0``。
- **序列层**：同样的索引 ``valid_mask=False``；输出长度恰为 33；COCO 对应点的
  confidence 正确传入第 4 通道。
- 序列层输出是 numpy 数组（不是冻结对象列表）。

确定性 / 无网络
--------------
不下载模型、不读真实视频：直接用合成 COCO17 数组喂纯映射函数，
并用 fake YOLO result 对象驱动结果解析。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_yolo_landmark_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.yolo_adapter import (  # noqa: E402
    BLAZE33_MISSING_IN_COCO17,
    COCO17_TO_BLAZE33,
    DEFAULT_YOLO_VALID_CONF_THR,
    FrameResult,
    Landmark,
    coco17_to_landmarks,
    map_coco17_person,
    map_coco17_to_blaze33,
    yolo_result_to_arrays,
)
from tests.yolo_fakes import FakeYoloResult, make_coco17  # noqa: E402


# --------------------------------------------------------------------------- #
# 纯映射：序列层 numpy 行
# --------------------------------------------------------------------------- #
def test_map_returns_numpy_arrays_length_33():
    xy, conf = make_coco17(conf_value=0.9)
    row, valid = map_coco17_to_blaze33(xy, conf, valid_conf_thr=0.5)
    assert isinstance(row, np.ndarray)
    assert isinstance(valid, np.ndarray)
    assert row.shape == (33, 4)
    assert valid.shape == (33,)
    assert row.dtype == np.float32
    assert valid.dtype == bool


def test_mapped_points_confidence_passthrough():
    # 给每个 COCO 点一个独特置信度，验证 channel-3 正确传递。
    xy, conf = make_coco17()
    for coco_idx in range(17):
        conf[coco_idx] = 0.10 + 0.05 * coco_idx  # 0.10 .. 0.90
        xy[coco_idx] = (0.01 * coco_idx, 0.02 * coco_idx)
    row, valid = map_coco17_to_blaze33(xy, conf, valid_conf_thr=0.5)

    for blaze_idx, coco_idx in COCO17_TO_BLAZE33.items():
        assert row[blaze_idx, 0] == pytest.approx(0.01 * coco_idx, abs=1e-6)
        assert row[blaze_idx, 1] == pytest.approx(0.02 * coco_idx, abs=1e-6)
        assert row[blaze_idx, 2] == 0.0
        assert row[blaze_idx, 3] == pytest.approx(0.10 + 0.05 * coco_idx, abs=1e-6)


def test_sequence_layer_missing_points_valid_false():
    # 即使给所有点高置信度，缺失点仍必须 valid_mask=False（不伪造）。
    xy, conf = make_coco17(conf_value=1.0)
    _row, valid = map_coco17_to_blaze33(xy, conf, valid_conf_thr=0.5)
    for miss in BLAZE33_MISSING_IN_COCO17:
        assert valid[miss] == False, f"缺失点 {miss} 不得被标记为有效"  # noqa: E712
    # 具体清单：嘴角 9/10、手指 17-22、脚跟脚尖 29-32、眼细分 1/3/4/6
    for idx in (9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32, 1, 3, 4, 6):
        assert valid[idx] == False  # noqa: E712


def test_sequence_layer_valid_threshold_applied():
    xy, conf = make_coco17(conf_value=0.4)  # 低于 0.5 阈值
    _row, valid = map_coco17_to_blaze33(xy, conf, valid_conf_thr=0.5)
    # 映射点都低于阈值 → 全部无效
    for blaze_idx in COCO17_TO_BLAZE33:
        assert valid[blaze_idx] == False  # noqa: E712

    xy2, conf2 = make_coco17(conf_value=0.6)  # 高于阈值
    _row2, valid2 = map_coco17_to_blaze33(xy2, conf2, valid_conf_thr=0.5)
    for blaze_idx in COCO17_TO_BLAZE33:
        assert valid2[blaze_idx] == True  # noqa: E712
    # 但缺失点无论如何都无效
    for miss in BLAZE33_MISSING_IN_COCO17:
        assert valid2[miss] == False  # noqa: E712


def test_missing_count_is_16_and_mapped_is_17():
    assert len(BLAZE33_MISSING_IN_COCO17) == 16
    assert len(COCO17_TO_BLAZE33) == 17
    # 互不重叠且并集覆盖 0..32
    mapped = set(COCO17_TO_BLAZE33.keys())
    missing = set(BLAZE33_MISSING_IN_COCO17)
    assert mapped.isdisjoint(missing)
    assert mapped | missing == set(range(33))


# --------------------------------------------------------------------------- #
# 边界层：synthetic / visibility
# --------------------------------------------------------------------------- #
def test_boundary_layer_missing_points_synthetic():
    xy, conf = make_coco17(conf_value=0.9)
    landmarks = coco17_to_landmarks(xy, conf)
    assert isinstance(landmarks, tuple)
    assert len(landmarks) == 33
    assert all(isinstance(lm, Landmark) for lm in landmarks)

    for miss in BLAZE33_MISSING_IN_COCO17:
        lm = landmarks[miss]
        assert lm.synthetic is True, f"缺失点 {miss} 必须 synthetic=True"
        assert lm.visibility == 0.0
        assert lm.confidence is None


def test_boundary_layer_mapped_points_not_synthetic_carry_confidence():
    xy, conf = make_coco17()
    for coco_idx in range(17):
        conf[coco_idx] = 0.3 + 0.04 * coco_idx
    landmarks = coco17_to_landmarks(xy, conf)
    for blaze_idx, coco_idx in COCO17_TO_BLAZE33.items():
        lm = landmarks[blaze_idx]
        assert lm.synthetic is False, f"映射点 {blaze_idx} 不应是 synthetic"
        assert lm.confidence == pytest.approx(0.3 + 0.04 * coco_idx, abs=1e-6)
        assert lm.visibility == 0.0


def test_boundary_specific_missing_groups_synthetic():
    """显式覆盖 Issue #7 列举的缺失组：嘴角/手指/脚跟脚尖/眼细分。"""
    xy, conf = make_coco17(conf_value=1.0)
    landmarks = coco17_to_landmarks(xy, conf)
    groups = {
        "mouth": (9, 10),
        "fingers": (17, 18, 19, 20, 21, 22),
        "heel_foot": (29, 30, 31, 32),
        "eye_subdiv": (1, 3, 4, 6),
    }
    for _name, idxs in groups.items():
        for idx in idxs:
            assert landmarks[idx].synthetic is True
            assert landmarks[idx].visibility == 0.0


# --------------------------------------------------------------------------- #
# map_coco17_person：序列三件套 + 边界容器一致性
# --------------------------------------------------------------------------- #
def test_map_coco17_person_returns_arrays_and_frameresult():
    xy, conf = make_coco17(conf_value=0.8)
    row, valid, frame = map_coco17_person(xy, conf, valid_conf_thr=0.5, track_id=7)
    assert isinstance(row, np.ndarray) and row.shape == (33, 4)
    assert isinstance(valid, np.ndarray) and valid.shape == (33,)
    assert isinstance(frame, FrameResult)
    assert frame.pose33 is not None and len(frame.pose33) == 33
    assert frame.track_id == 7
    assert frame.meta is not None
    assert frame.meta["backend"] == "yolo"
    assert frame.meta["raw_layout"] == "pose33_like_coco17"
    assert frame.meta["feature_layout"] == "body_core_v1"
    assert frame.meta["capability"] == "body_only"
    assert frame.meta["confidence_kind"] == "yolo_conf"
    assert frame.meta["validity_policy"] == "confidence_thr"
    assert frame.meta["calibration_status"] == "unvalidated"
    assert frame.meta["score_authorized"] is False
    assert frame.meta["display_scope"] == "limited"
    assert "eval_scope" not in frame.meta
    assert "internal_use_only" not in frame.meta


# --------------------------------------------------------------------------- #
# yolo_result_to_arrays：从 fake YOLO result 解析（仍只返回 numpy）
# --------------------------------------------------------------------------- #
def test_yolo_result_to_arrays_numpy_only():
    xy, conf = make_coco17(conf_value=0.9)
    res = FakeYoloResult.single(xy, conf)
    row, valid, num_persons, sel = yolo_result_to_arrays(res, valid_conf_thr=0.5)
    assert isinstance(row, np.ndarray) and row.shape == (33, 4)
    assert isinstance(valid, np.ndarray) and valid.shape == (33,)
    # 不是冻结对象列表
    assert not isinstance(row, (list, tuple))
    assert not isinstance(valid, (list, tuple))
    assert num_persons == 1
    assert sel == 0
    # 缺失点序列层仍无效
    for miss in BLAZE33_MISSING_IN_COCO17:
        assert valid[miss] == False  # noqa: E712


def test_default_thr_is_calibrated_constant():
    # S3（#10）落库阈值 0.6，刻意与 MediaPipe（0.5）分开；仍不授权对外评分。
    assert DEFAULT_YOLO_VALID_CONF_THR == 0.6
