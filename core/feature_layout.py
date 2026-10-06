"""特征布局门面：转发到 ``core.pose.layouts``（P9 删除）。"""
from core.pose.layouts import (  # noqa: F401
    FeatureLayoutSpec,
    POSE33_V3,
    BODY_CORE_V1,
    MIN_LAYOUT_JOINTS,
    register_layout,
    get_layout,
    has_layout,
    resolve_layout_by_shape,
    is_valid_feature_shape,
)

__all__ = [
    "FeatureLayoutSpec",
    "POSE33_V3",
    "BODY_CORE_V1",
    "MIN_LAYOUT_JOINTS",
    "register_layout",
    "get_layout",
    "has_layout",
    "resolve_layout_by_shape",
    "is_valid_feature_shape",
]
