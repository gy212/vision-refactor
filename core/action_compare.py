"""动作比对门面：实现已迁入 ``core.evaluation``（模板、匹配、切分）、``core.pose.features`` 与 ``core.services``（P9 删除）。

旧名与旧签名逐个转发，门面里不含逻辑。
"""
from __future__ import annotations

from .evaluation.matching.dual import DualCompareResult  # noqa: F401
from .evaluation.matching.dual import compare_dual_streams  # noqa: F401
from .evaluation.matching.dual import compare_dual_templates as compare_video_to_dual_templates  # noqa: F401
from .evaluation.matching.engine import CompareResult  # noqa: F401
from .evaluation.matching.engine import assert_feature_layout_match as _assert_feature_layout_match  # noqa: F401
from .evaluation.matching.engine import match_video as compare_video_to_templates  # noqa: F401
from .evaluation.matching.repetition import JointErrorStat, RepetitionMatch  # noqa: F401
from .evaluation.matching.repetition import multi_subsequence_matches as _multi_subsequence_matches  # noqa: F401
from .evaluation.matching.repetition import score_view as _score_view_seq  # noqa: F401
from .evaluation.matching.repetition import valid_mask_from_raw as _valid_mask_from_raw  # noqa: F401
from .evaluation.segmentation import estimate_period_frames as _estimate_period_frames  # noqa: F401
from .evaluation.segmentation import select_representative_cycle as _select_representative_cycle  # noqa: F401
from .evaluation.templates import LEGACY_DEFAULT_FEATURE_LAYOUT, LEGACY_POSE33_V3_FEATURE_LAYOUT  # noqa: F401
from .evaluation.templates import canonical_feature_layout as _canonical_feature_layout  # noqa: F401
from .evaluation.templates import normalize_template_meta, template_meta_defaults  # noqa: F401
from .evaluation.templates import normalizer_version_from_layout as _normalizer_version_from_layout  # noqa: F401
from .evaluation.templates import runtime_feature_layout as _runtime_feature_layout  # noqa: F401
from .pose.features import ProgressCb  # noqa: F401
from .pose.features import extract_features as _extract_pose_features  # noqa: F401
from .pose.layouts import POSE33_V3  # noqa: F401
from .pose.normalize import normalize_pose_xy, normalize_pose_xy_v1, normalize_pose_xy_v3  # noqa: F401
from .pose.raw_series import extract_pose_raw_series  # noqa: F401
from .services.analyze import compare_video as compare_video_to_template  # noqa: F401
from .services.preview import export_match_preview, write_pose_preview  # noqa: F401
from .services.templates import (  # noqa: F401
    TemplateAutoCropAnalysis,
    TemplateAutoCropReviewRequired,
    analyze_template_auto_crop,
    create_template_from_video,
)
from .foundation.paths import templates_dir  # noqa: F401
