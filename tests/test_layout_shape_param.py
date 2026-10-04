# -*- coding: utf-8 -*-
"""
layout shape 参数化回归（YOLO 迁移 Issue #4 / S1）。

验收目标（对应 Issue #4）
-------------------------
1. 断言 `_extract_pose_features` 缺帧补零、`mirror_pose_features`、双模板关节误差统计
   在 `pose33_v3` 下行为不变（缺帧补零 shape、mirror 的 L/R 交换、误差关节名顺序）。
2. 断言热路径（缺帧补零、mirror、误差统计、周期裁切）**源码层面不再硬编码**
   `22` / `(22,2)` 作为 shape 魔法值——
   `FeatureLayoutSpec.pose33_v3` 的注册定义、测试期望值、注释中出现 `22` 属合法，不计入。
3. `FeatureLayoutSpec` 注册表只注册生产用 `pose33_v3`，不提前引入 `body_core_v1`，
   且不引入 S4 才消费的 `required_landmarks` 字段。

确定性
------
本测试不依赖 MediaPipe 推理：mirror / 周期裁切用纯 numpy 输入；缺帧补零通过 golden
harness 回放「整段 normalizer 返回 None」的退化序列来触发 `np.zeros(layout.shape)` 分支。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_layout_shape_param.py
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import core.action_compare as ac  # noqa: E402
from core.action_compare import _extract_pose_features, _select_representative_cycle  # noqa: E402
from core.action_compare import _assert_feature_layout_match  # noqa: E402
from core.feature_layout import (  # noqa: E402
    POSE33_V3,
    FeatureLayoutSpec,
    get_layout,
    has_layout,
)
from core.pose_features import mirror_pose_features  # noqa: E402
from tests import golden_harness as H  # noqa: E402

ABS_TOL = 1e-6


# --------------------------------------------------------------------------- #
# 注册表范围：只注册 pose33_v3，不提前引入 body_core_v1，不引入 required_landmarks
# --------------------------------------------------------------------------- #
def test_registry_scope_pose33_v3_only():
    spec = get_layout("pose33_v3")
    assert spec is POSE33_V3
    assert spec.shape == (22, 2)
    assert spec.num_joints == 22
    assert len(spec.source_indices) == 22
    assert spec.source_indices == tuple(range(11, 33))
    assert len(spec.joint_names) == 22
    # body_core_v1 在 #8（S2）随 YOLO 离线闭环注册；此处确认其已注册且与 pose33_v3 并存。
    assert has_layout("body_core_v1")


def test_layout_spec_has_no_required_landmarks_field():
    # S4 才消费的 required_landmarks 字段本期不得引入
    field_names = {f.name for f in dataclasses.fields(FeatureLayoutSpec)}
    assert "required_landmarks" not in field_names
    expected = {
        "name",
        "source_indices",
        "shape",
        "mirror_pairs",
        "joint_names",
        "default_baseline",
    }
    assert field_names == expected


def test_pose33_v3_mirror_pairs_are_adjacent():
    # pose33_v3 的 mirror_pairs 必须等价旧实现的相邻对 (0,1),(2,3),...,(20,21)
    assert POSE33_V3.mirror_pairs == tuple((i, i + 1) for i in range(0, 22, 2))


# --------------------------------------------------------------------------- #
# mirror_pose_features 在 pose33_v3 下行为不变
# --------------------------------------------------------------------------- #
def _legacy_mirror(features: np.ndarray) -> np.ndarray:
    """旧实现（写死 22 点相邻对），仅用于本测试做等价对照。"""
    if features.ndim == 2:
        x = features.copy()
        x[:, 0] *= -1.0
        y = x.copy()
        for a in range(0, 22, 2):
            y[a] = x[a + 1]
            y[a + 1] = x[a]
        return y
    x = features.copy()
    x[:, :, 0] *= -1.0
    y = x.copy()
    for a in range(0, 22, 2):
        y[:, a] = x[:, a + 1]
        y[:, a + 1] = x[:, a]
    return y


@pytest.mark.parametrize("ndim", [2, 3])
def test_mirror_equivalent_to_legacy_for_pose33_v3(ndim):
    rng = np.random.default_rng(1234)
    if ndim == 2:
        feats = rng.standard_normal((22, 2)).astype(np.float32)
    else:
        feats = rng.standard_normal((17, 22, 2)).astype(np.float32)

    got = mirror_pose_features(feats)
    exp = _legacy_mirror(feats)
    assert got.shape == exp.shape
    assert np.allclose(got, exp, atol=ABS_TOL)


def test_mirror_accepts_explicit_layout_name():
    rng = np.random.default_rng(7)
    feats = rng.standard_normal((22, 2)).astype(np.float32)
    by_name = mirror_pose_features(feats, layout="pose33_v3")
    by_shape = mirror_pose_features(feats)
    assert np.allclose(by_name, by_shape, atol=ABS_TOL)


def test_mirror_unresolvable_shape_raises_not_silent():
    # 无法解析布局（未注册的 (J,2)）时必须报错，不得静默按 22 点处理。
    # 用 (15,2)：既非 pose33_v3 (22,2) 也非 body_core_v1 (12,2)，注册表无法反查。
    rng = np.random.default_rng(9)
    feats = rng.standard_normal((15, 2)).astype(np.float32)
    with pytest.raises(ValueError):
        mirror_pose_features(feats)


def test_mirror_with_dummy_layout_uses_its_pairs():
    # test-only dummy 布局：mirror 必须按其 mirror_pairs 工作，与 22 点无关
    dummy = FeatureLayoutSpec(
        name="_dummy_mirror_test",
        source_indices=(11, 12, 23, 24),
        shape=(4, 2),
        mirror_pairs=((0, 1), (2, 3)),
        joint_names=("L_A", "R_A", "L_B", "R_B"),
        default_baseline=None,
    )
    feats = np.array(
        [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]], dtype=np.float32
    )
    out = mirror_pose_features(feats, layout=dummy)
    # x 取反 + 交换 (0,1) 与 (2,3)
    exp = np.array(
        [[2.0, 6.0], [1.0, 5.0], [4.0, 8.0], [3.0, 7.0]], dtype=np.float32
    )
    exp[:, 0] *= -1.0
    assert np.allclose(out, exp, atol=ABS_TOL)


# --------------------------------------------------------------------------- #
# 缺帧补零按 layout shape 生成（不再写死 (22,2)）
# --------------------------------------------------------------------------- #
def test_missing_frame_padding_uses_layout_shape():
    # 构造一段「所有帧 normalizer 都返回 None」的退化序列：
    # landmark 全 NaN -> normalize_pose_xy_v3 返回 None -> 走 np.zeros(layout.shape) 分支。
    T = 6
    raw = np.full((T, 33, 4), np.nan, dtype=np.float32)
    key = "golden://layout_param_empty.mp4"
    H.clear_registry()
    H.register_video(key, raw, fps=30.0)
    try:
        with H.replay_context():
            feats, fps, _view = _extract_pose_features(
                Path(key),
                pose_variant="full",
                workers=1,
                normalizer=ac.normalize_pose_xy_v3,
            )
    finally:
        H.clear_registry()

    # 缺帧补零形状必须等于 layout.shape，且全为 0
    assert feats.shape == (T, *POSE33_V3.shape)
    assert np.allclose(feats, 0.0, atol=ABS_TOL)
    assert float(fps) == pytest.approx(30.0, abs=ABS_TOL)


# --------------------------------------------------------------------------- #
# 不同 layout / shape 不允许静默比对
# --------------------------------------------------------------------------- #
def test_feature_layout_mismatch_raises_clear_error():
    query = np.zeros((8, 22, 2), dtype=np.float32)
    seq = np.zeros((12, 12, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Feature layout mismatch"):
        _assert_feature_layout_match(
            query,
            seq,
            left_label="template",
            right_label="video",
            left_layout="pose33_v3",
            right_layout="body_core_v1",
        )


def test_feature_layout_name_mismatch_raises_even_with_same_shape():
    query = np.zeros((8, *POSE33_V3.shape), dtype=np.float32)
    seq = np.zeros((12, *POSE33_V3.shape), dtype=np.float32)

    with pytest.raises(ValueError, match="Feature layout mismatch"):
        _assert_feature_layout_match(
            query,
            seq,
            left_label="front template",
            right_label="side template",
            left_layout="pose33_v3",
            right_layout="legacy_pose33_alias",
        )


# --------------------------------------------------------------------------- #
# 源码层面：热路径不得出现 (22, 2) 字面量或裸 range(0, 22, 2) / range(22) 魔法值
# --------------------------------------------------------------------------- #
def _tuple_22_2_literals(tree: ast.AST) -> list[ast.AST]:
    """收集源码 AST 中所有 (22, 2) 元组字面量节点。"""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            vals = []
            ok = True
            for e in node.elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, int):
                    vals.append(e.value)
                else:
                    ok = False
                    break
            if ok and vals == [22, 2]:
                found.append(node)
    return found


def _int_literal_22_calls(tree: ast.AST) -> list[ast.AST]:
    """收集裸用整数 22 作为 range(...) 参数的调用（mirror/误差统计旧风格）。"""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == 22:
                    found.append(node)
    return found


def test_no_22_2_literal_in_action_compare_hotpath():
    src = (_REPO_ROOT / "core" / "action_compare.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert _tuple_22_2_literals(tree) == [], "action_compare 热路径不得出现 (22,2) 字面量"
    assert _int_literal_22_calls(tree) == [], "action_compare 热路径不得出现 range(...,22,...) 魔法值"


def test_no_22_2_literal_in_pose_features_hotpath():
    src = (_REPO_ROOT / "core" / "pose_features.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # mirror_pose_features 不得再写死 range(0,22,2)
    assert _int_literal_22_calls(tree) == [], "pose_features mirror 不得出现 range(0,22,2) 魔法值"


def test_no_22_2_literal_in_feature_layout_is_allowed():
    # 反向确认：注册定义里出现 (22,2) 是合法的（shape 注册），不应被上面规则误伤。
    src = (_REPO_ROOT / "core" / "feature_layout.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert len(_tuple_22_2_literals(tree)) >= 1
