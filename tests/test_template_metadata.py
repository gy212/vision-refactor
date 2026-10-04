# -*- coding: utf-8 -*-
"""模板 metadata 扩展 + 向后兼容回归（YOLO 迁移 S1 / Issue #6 Part B/C）。

覆盖
----
1. 新模板：通过 ``create_template_from_video``（确定性回放）生成模板，断言 Issue #6
   metadata 字段齐全且取值正确，且 ``feature_layout`` 写入注册表布局名。
2. 向后兼容：构造“缺新字段”的 legacy 模板（取已提交模板，剥掉新键后另存），
   断言 ``compare_video_to_template`` / ``compare_video_to_dual_templates`` 仍能加载并
   运行不报错，且 ``normalize_template_meta`` 能补齐默认值。

确定性
------
与 golden 回归一致，通过 ``tests/golden_harness.py`` 回放已提交的 ``(T,33,4)`` landmark
fixture，不依赖 MediaPipe 实际推理。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_template_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.action_compare import (  # noqa: E402
    LEGACY_DEFAULT_FEATURE_LAYOUT,
    compare_video_to_dual_templates,
    compare_video_to_template,
    create_template_from_video,
    normalize_template_meta,
    template_meta_defaults,
)
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"

FRONT_SRC = "golden://front_src.mp4"
SIDE_SRC = "golden://side_src.mp4"
STUDENT = "golden://student.mp4"
FPS = 30.0

# 历史模板使用的 feature_layout 字符串（loader 需继续兼容）。
LEGACY_FEATURE_LAYOUT = "pose_indices_11_32_xy_rot_scale_norm_v3"
EXPECTED_FEATURE_LAYOUT = "pose33_v3"

# Issue #6 metadata 字段（含取值）。
_ISSUE6_FIELDS = {
    "backend": "mediapipe",
    "model_name": "pose_landmarker_full",
    "feature_layout": EXPECTED_FEATURE_LAYOUT,
    "normalizer_version": "v3",
    "confidence_kind": "visibility",
    "validity_policy": "visibility_thr",
    "valid_conf_thr": 0.5,
}


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


def _load_meta(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return dict(d["meta"].item())


# --------------------------------------------------------------------------- #
# 1) 新模板：Issue #6 字段齐全 + feature_layout 使用注册表布局名
# --------------------------------------------------------------------------- #
def test_new_template_has_all_metadata_fields(tmp_path):
    front_raw = _load_raw("front_src_raw.npz")
    out_path = tmp_path / "front_template_new.npz"

    H.clear_registry()
    H.register_video(FRONT_SRC, front_raw, fps=FPS)
    with H.replay_context():
        create_template_from_video(FRONT_SRC, pose_variant="full", out_path=out_path)
    H.clear_registry()

    meta = _load_meta(out_path)

    # Issue #6 字段齐全且取值正确
    assert meta["backend"] == "mediapipe"
    assert meta["model_name"] == "pose_landmarker_full"
    assert meta["feature_layout"] == EXPECTED_FEATURE_LAYOUT
    assert meta["normalizer_version"] == "v3"
    assert meta["confidence_kind"] == "visibility"
    assert meta["validity_policy"] == "visibility_thr"
    assert float(meta["valid_conf_thr"]) == pytest.approx(0.5, abs=1e-9)
    assert "feature_layout_name" not in meta

    # feature_layout 使用 #4 注册表布局名，供 #7/#8 后续统一读取
    assert meta["feature_layout"] == EXPECTED_FEATURE_LAYOUT
    # 既有字段仍在
    assert meta["pose_variant"] == "full"
    assert meta["running_mode"] == "video"


def test_template_meta_defaults_and_normalize():
    defaults = template_meta_defaults()
    # 默认值与字段集合一致
    for key, value in _ISSUE6_FIELDS.items():
        assert key in defaults
        if isinstance(value, float):
            assert float(defaults[key]) == pytest.approx(value, abs=1e-9)
        else:
            assert defaults[key] == value
    assert "feature_layout_name" not in defaults

    # normalize 只补缺失键，不覆盖已存在的值
    legacy = {"feature_layout": LEGACY_FEATURE_LAYOUT, "backend": "custom"}
    normalized = normalize_template_meta(dict(legacy))
    assert normalized["feature_layout"] == LEGACY_FEATURE_LAYOUT  # 旧模板已有值不覆盖
    assert normalized["backend"] == "custom"  # 已存在不覆盖
    assert normalized["model_name"] == "pose_landmarker_full"  # 缺失补默认
    assert normalized["normalizer_version"] == "v3"
    assert normalized["validity_policy"] == "visibility_thr"
    assert float(normalized["valid_conf_thr"]) == pytest.approx(0.5, abs=1e-9)


def test_normalize_accepts_transitional_feature_layout_name():
    transitional = {"feature_layout_name": EXPECTED_FEATURE_LAYOUT}
    normalized = normalize_template_meta(dict(transitional))
    assert normalized["feature_layout"] == EXPECTED_FEATURE_LAYOUT
    assert "feature_layout_name" not in normalized


# --------------------------------------------------------------------------- #
# 2) 向后兼容：缺新字段的 legacy 模板仍能加载并比对
# --------------------------------------------------------------------------- #
def _make_legacy_template(src: Path, dst: Path) -> dict:
    """复制已提交模板，剥掉 Issue #6 新增的全部新字段，另存为 legacy 模板。"""
    d = np.load(src, allow_pickle=True)
    features = d["features"]
    meta = dict(d["meta"].item())
    for key in _ISSUE6_FIELDS:
        meta.pop(key, None)
    meta["feature_layout"] = LEGACY_FEATURE_LAYOUT
    np.savez_compressed(dst, features=features, meta=np.array(meta, dtype=object))
    return meta


def test_legacy_template_missing_fields_normalized():
    # 直接验证 normalize 能把“剥光新字段”的 meta 补齐。
    stripped = {
        k: v
        for k, v in _load_meta(FIX_DIR / "front_template.npz").items()
        if k not in _ISSUE6_FIELDS
    }
    for key in _ISSUE6_FIELDS:
        assert key not in stripped
    stripped["feature_layout"] = LEGACY_FEATURE_LAYOUT
    normalized = normalize_template_meta(dict(stripped))
    for key in _ISSUE6_FIELDS:
        assert key in normalized, f"normalize 未补齐 {key}"
    assert normalized["feature_layout"] == LEGACY_FEATURE_LAYOUT


def test_missing_feature_layout_metadata_preserves_legacy_runtime_fallback(tmp_path, monkeypatch):
    legacy_tpl = tmp_path / "missing_feature_layout.npz"
    d = np.load(FIX_DIR / "front_template.npz", allow_pickle=True)
    meta = dict(d["meta"].item())
    for key in _ISSUE6_FIELDS:
        meta.pop(key, None)
    meta.pop("feature_layout_name", None)
    np.savez_compressed(legacy_tpl, features=d["features"], meta=np.array(meta, dtype=object))

    captured: dict[str, str] = {}

    def fake_extract(video_path, *, normalizer, **kwargs):
        captured["normalizer"] = normalizer.__name__
        return np.zeros((40, 22, 2), dtype=np.float32), FPS, None

    monkeypatch.setattr("core.action_compare._extract_pose_features", fake_extract)
    compare_video_to_template(legacy_tpl, STUDENT, pose_variant="full")

    # The normalized metadata is filled for downstream visibility, but runtime matching must still
    # preserve the pre-Issue #6 fallback for templates that had no feature_layout key at all.
    assert normalize_template_meta(dict(meta))["feature_layout"] == EXPECTED_FEATURE_LAYOUT
    assert LEGACY_DEFAULT_FEATURE_LAYOUT == "pose_indices_11_32_xy_rot_scale_norm"
    assert captured["normalizer"] == "normalize_pose_xy_v1"


def test_legacy_single_template_still_loads(tmp_path):
    legacy_tpl = tmp_path / "legacy_front_template.npz"
    legacy_meta = _make_legacy_template(FIX_DIR / "front_template.npz", legacy_tpl)
    # 确认 legacy 模板确实缺新字段
    for key in _ISSUE6_FIELDS:
        if key == "feature_layout":
            assert legacy_meta[key] == LEGACY_FEATURE_LAYOUT
        else:
            assert key not in legacy_meta

    student_raw = _load_raw("student_raw.npz")
    H.clear_registry()
    H.register_video(STUDENT, student_raw, fps=FPS)
    with H.replay_context():
        res = compare_video_to_template(legacy_tpl, STUDENT, pose_variant="full")
    H.clear_registry()

    # 数值断言保持宽松：只要返回结果对象、分数有限即可。
    assert res is not None
    assert np.isfinite(float(res.score))
    assert np.isfinite(float(res.avg_cost))


def test_legacy_dual_templates_still_load(tmp_path):
    legacy_front = tmp_path / "legacy_front.npz"
    legacy_side = tmp_path / "legacy_side.npz"
    _make_legacy_template(FIX_DIR / "front_template.npz", legacy_front)
    _make_legacy_template(FIX_DIR / "side_template.npz", legacy_side)

    student_raw = _load_raw("student_raw.npz")
    H.clear_registry()
    H.register_video(STUDENT, student_raw, fps=FPS)
    with H.replay_context():
        res = compare_video_to_dual_templates(
            legacy_front,
            legacy_side,
            STUDENT,
            pose_variant="full",
            enable_rules=True,
            enable_error_analysis=True,
        )
    H.clear_registry()

    assert res is not None
    assert 0 <= int(res.combined_percent) <= 100
    assert np.isfinite(float(res.front_score))
    assert np.isfinite(float(res.side_score))


def test_mixed_legacy_and_new_layout_aliases_match(tmp_path):
    legacy_front = tmp_path / "legacy_front.npz"
    _make_legacy_template(FIX_DIR / "front_template.npz", legacy_front)

    side_raw = _load_raw("side_src_raw.npz")
    new_side = tmp_path / "new_side.npz"
    H.clear_registry()
    H.register_video(SIDE_SRC, side_raw, fps=FPS)
    with H.replay_context():
        create_template_from_video(SIDE_SRC, pose_variant="full", out_path=new_side)
    H.clear_registry()

    assert _load_meta(legacy_front)["feature_layout"] == LEGACY_FEATURE_LAYOUT
    assert _load_meta(new_side)["feature_layout"] == EXPECTED_FEATURE_LAYOUT

    student_raw = _load_raw("student_raw.npz")
    H.clear_registry()
    H.register_video(STUDENT, student_raw, fps=FPS)
    with H.replay_context():
        res = compare_video_to_dual_templates(
            legacy_front,
            new_side,
            STUDENT,
            pose_variant="full",
            enable_rules=False,
            enable_error_analysis=False,
        )
    H.clear_registry()

    assert res is not None
    assert 0 <= int(res.combined_percent) <= 100
