# -*- coding: utf-8 -*-
"""模板读取与契约校验的唯一实现：``Template``、``TemplateContract``、``load_template``、``TemplateStore``（设计文档 §7.4）。

三种契约预设复现迁移前三处各自的读取与校验（§3 第 26 项）：
- ``COMPARE``：比对入口（单/多/双模板、CLI、body_core）。不做契约校验、读取失败原样抛出，
  布局与特征形状不符留给匹配步骤报错（``matching.engine.assert_feature_layout_match``）。
- ``EXAM_POOL``：考试与录后自动比对的 heavy + pose33_v3 模板（原 ``_load_heavy_template``）。
- ``ONLINE``：在线识别的模板库，要求 pose33_v3 / v3（原 ``load_template_library``）。
契约违规抛 ``TemplateError(code, message)``；提示文案逐字沿用迁移前的版本。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.pose.layouts import POSE33_V3
from core.pose.normalize import normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3
from core.pose.validity import DEFAULT_VALID_CONF_THR, MEDIAPIPE_VALIDITY_POLICY

# 历史模板曾把 feature_layout 写成 normalizer 描述字符串；Issue #6 后统一写注册表布局名。
LEGACY_DEFAULT_FEATURE_LAYOUT = "pose_indices_11_32_xy_rot_scale_norm"
LEGACY_POSE33_V3_FEATURE_LAYOUT = "pose_indices_11_32_xy_rot_scale_norm_v3"
_NORMALIZERS = {"v1": normalize_pose_xy_v1, "v2": normalize_pose_xy, "v3": normalize_pose_xy_v3}


class TemplateError(ValueError):
    """模板不满足契约。``code`` 与录后处理的错误码一致（template_missing/invalid/incompatible）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __reduce__(self):
        return (self.__class__, (self.code, self.message))


def template_meta_defaults() -> dict:
    """Issue #6 模板 metadata 字段及默认值：写模板时补齐、读旧模板（缺字段）时填默认。"""
    return {
        "backend": "mediapipe",
        "model_name": "pose_landmarker_full",
        "feature_layout": POSE33_V3.name,
        "normalizer_version": "v3",
        "confidence_kind": "visibility",
        "validity_policy": MEDIAPIPE_VALIDITY_POLICY,
        "valid_conf_thr": float(DEFAULT_VALID_CONF_THR),
    }


def normalize_template_meta(meta: dict | None) -> dict:
    """原地用默认值补齐缺失字段（不覆盖已有键）。旧模板已有的历史 ``feature_layout`` 字符串保留，
    读取方仍可按后缀识别 v1/v2/v3；只有 ``feature_layout_name`` 的旧模板把它挪到 ``feature_layout``。"""
    if meta is None:
        meta = {}
    legacy_layout_name = meta.pop("feature_layout_name", None)
    if "feature_layout" not in meta and legacy_layout_name:
        meta["feature_layout"] = legacy_layout_name
    for key, value in template_meta_defaults().items():
        meta.setdefault(key, value)
    return meta


def runtime_feature_layout(raw_meta: dict, normalized_meta: dict) -> str:
    """驱动匹配的布局名：有 ``feature_layout`` 用它，只有 ``feature_layout_name`` 用补齐后的值，都没有按最旧的 v1。"""
    if "feature_layout" in raw_meta:
        return str(raw_meta["feature_layout"])
    if "feature_layout_name" in raw_meta:
        return str(normalized_meta.get("feature_layout", raw_meta["feature_layout_name"]))
    return LEGACY_DEFAULT_FEATURE_LAYOUT


def canonical_feature_layout(layout: str | None) -> str | None:
    """把历史别名映射到注册表布局名（新 metadata 写的名字）。"""
    if layout is None:
        return None
    layout_s = str(layout)
    if layout_s == POSE33_V3.name or layout_s == LEGACY_POSE33_V3_FEATURE_LAYOUT:
        return POSE33_V3.name
    return layout_s


def normalizer_version_from_layout(layout: str) -> str:
    """由布局名（含历史别名）判归一化版本 v1/v2/v3。"""
    layout = str(layout)
    if canonical_feature_layout(layout) == POSE33_V3.name or layout.endswith("_v3"):
        return "v3"
    if layout.endswith("_v2"):
        return "v2"
    return "v1"


def normalizer_for(version: str) -> Callable:
    """归一化版本对应的单帧归一化函数。"""
    return _NORMALIZERS[version]


@dataclass(frozen=True)
class Template:
    """读入的模板。``declared_meta`` 是文件里的原值（可能为 None），``raw_meta`` 是它的 dict 副本，
    ``meta`` 是补齐默认字段后的 dict；``layout``/``normalizer_version`` 按 ``runtime_feature_layout`` 推断。"""

    path: Path
    features: np.ndarray
    declared_meta: Any
    raw_meta: dict
    meta: dict
    layout: str
    normalizer_version: str

    @property
    def name(self) -> str:
        """动作名 = 文件 stem（如 直拳.npz → "直拳"）。"""
        return self.path.stem

    @property
    def running_mode(self) -> str:
        meta = self.meta
        return str(meta.get("running_mode") or (meta.get("cfg") or {}).get("running_mode") or "video").lower()


Check = Callable[["TemplateContract", Path, np.ndarray, dict], None]


@dataclass(frozen=True)
class TemplateContract:
    """模板契约：允许的布局/归一化/模型规格，以及读取方式与按顺序执行的校验。``None`` 表示不限。"""

    name: str
    layouts: frozenset[str] | None
    normalizers: frozenset[str] | None
    variants: frozenset[str] | None
    allow_missing_layout_meta: bool  # 旧模板无 feature_layout 字段时放行（按形状/默认推断）
    label: str = "模板"
    dtype: Any = None  # 特征转换的 dtype；None 保留文件里的 dtype
    require_file: bool = False  # 读之前先要求文件存在且非空，读取失败包装成 template_invalid
    optional_meta: bool = False  # 文件缺 meta 键时按空 meta 处理
    checks: tuple[Check, ...] = field(default=())


def _exam_shape(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    if features.ndim != 3 or features.shape[0] <= 0 or features.shape[1:] != POSE33_V3.shape:
        raise TemplateError("template_invalid", f"{c.label}特征形状无效：{path} -> {tuple(features.shape)}")


def _exam_finite(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    if not np.isfinite(features).all():
        raise TemplateError("template_invalid", f"{c.label}包含非有限特征：{path}")


def _exam_variant(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    if str(meta.get("pose_variant") or "") not in c.variants:
        raise TemplateError("template_invalid", f"{c.label}必须使用 {'/'.join(sorted(c.variants))} 模型：{path}")


def _exam_layout(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    layout = str(meta.get("feature_layout") or "")
    if (layout or not c.allow_missing_layout_meta) and layout not in c.layouts:
        raise TemplateError("template_incompatible",
                            f"{c.label}布局必须为 {'/'.join(sorted(c.layouts))}：{path} -> {layout}")


def _online_layout(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    layout = str(meta.get("feature_layout", "")) or "?"
    normalizer = str(meta.get("normalizer_version", "")) or "?"
    if layout not in c.layouts or normalizer not in c.normalizers:
        raise TemplateError(
            "template_incompatible",
            f"在线模板 {path.name} 布局/归一化不符：feature_layout={layout} normalizer_version={normalizer}，"
            f"在线识别要求 {POSE33_V3.name}/v3（与现场归一化逐字一致）",
        )


def _online_shape(c: TemplateContract, path: Path, features: np.ndarray, meta: dict) -> None:
    if features.ndim != 3 or features.shape[1:] != POSE33_V3.shape:
        raise TemplateError("template_invalid", f"在线模板 {path.name} 特征形状 {features.shape} 非 (M,{POSE33_V3.shape})")


COMPARE = TemplateContract("compare", None, None, None, True)
EXAM_POOL = TemplateContract(
    "exam_pool", frozenset({POSE33_V3.name}), None, frozenset({"heavy"}), True, label="固定模板",
    dtype=np.float32, require_file=True, checks=(_exam_shape, _exam_finite, _exam_variant, _exam_layout),
)
ONLINE = TemplateContract(
    "online", frozenset({POSE33_V3.name}), frozenset({"v3"}), None, False, dtype=np.float32,
    optional_meta=True, checks=(_online_layout, _online_shape),
)


def _read(path: Path, contract: TemplateContract) -> tuple[np.ndarray, Any, dict]:
    with np.load(path, allow_pickle=True) as data:
        if contract.require_file and ("features" not in data or "meta" not in data):
            raise ValueError("缺少 features/meta")
        raw = data["features"]
        features = raw if contract.dtype is None else np.asarray(raw, dtype=contract.dtype)
        declared = data["meta"].item() if (not contract.optional_meta or "meta" in data) else None
    return features, declared, dict(declared or {})


def load_template(path: str | Path, contract: TemplateContract = COMPARE) -> Template:
    """读模板并按契约校验；违规抛 ``TemplateError``。``COMPARE`` 的读取错误（缺文件、坏文件）原样抛出。"""
    path = Path(path)
    if contract.require_file:
        if not path.is_file() or path.stat().st_size <= 0:
            raise TemplateError("template_missing", f"{contract.label}不存在：{path}")
        try:
            features, declared, raw_meta = _read(path, contract)
        except Exception as exc:
            raise TemplateError("template_invalid", f"无法读取{contract.label} {path}：{exc}") from exc
    else:
        features, declared, raw_meta = _read(path, contract)
    for check in contract.checks:
        check(contract, path, features, raw_meta)
    meta = normalize_template_meta(dict(raw_meta))
    layout = runtime_feature_layout(raw_meta, meta)
    return Template(path, features, declared, raw_meta, meta, layout, normalizer_version_from_layout(layout))


class TemplateStore:
    """按契约读模板：同一文件（路径、大小、修改时间不变）只读一次；过滤可用模板；按目录读模板库。"""

    def __init__(self, contract: TemplateContract) -> None:
        self.contract = contract
        self._cache: dict[tuple, Template] = {}

    def load(self, path: str | Path) -> Template:
        path = Path(path)
        try:
            st = path.stat()
            key = (str(path.resolve(strict=False)), st.st_size, st.st_mtime_ns)
        except OSError:
            return load_template(path, self.contract)
        if key not in self._cache:
            self._cache[key] = load_template(path, self.contract)
        return self._cache[key]

    def usable(self, paths: Iterable[str | Path | None]) -> list[Path]:
        """可用模板的规范化路径，保序去重；空串、None 与不满足契约的跳过，不整体失败。"""
        out: list[Path] = []
        seen: set[str] = set()
        for raw in paths:
            if raw is None or not str(raw).strip():
                continue
            path = Path(str(raw).strip())
            try:
                self.load(path)
            except TemplateError:
                continue
            key = str(path.resolve(strict=False))
            if key not in seen:
                seen.add(key)
                out.append(path)
        return out

    def library(self, directory: str | Path) -> list[Template]:
        """目录下全部 ``*.npz``（按文件名排序）；目录不存在时为空。"""
        directory = Path(directory)
        if not directory.is_dir():
            return []
        return [self.load(p) for p in sorted(directory.glob("*.npz"))]
