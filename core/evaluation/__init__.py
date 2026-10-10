# -*- coding: utf-8 -*-
"""评估层（L3）的公共 API：模板、匹配、计分、侧别、时间轴切分（入口层只从这里导入）。"""
from core.evaluation.body_sides import (
    center_x,
    facing_direction,
    foot_edges_x,
    infer_front_leg,
    infer_kick_leg,
    side_label,
    sides_for_stance,
)
from core.evaluation.matching.dtw import subsequence_dtw, subsequence_dtw_with_path
from core.evaluation.matching.dual import DualCompareResult, compare_dual_streams, compare_dual_templates
from core.evaluation.matching.engine import (
    CompareResult,
    TemplateMatch,
    assert_feature_layout_match,
    match_templates,
    match_video,
    template_features,
)
from core.evaluation.matching.online import (
    ActionTemplate,
    MatchResult,
    MatcherConfig,
    OnlineActionMatcher,
    load_template_library,
    match_window,
)
from core.evaluation.matching.repetition import (
    JointErrorStat,
    RepetitionMatch,
    multi_subsequence_matches,
    score_view,
    valid_mask_from_raw,
)
from core.evaluation.scoring import exam_pool, schemes
from core.evaluation.scoring.cost import POSE33_BASELINE, avg_cost_of, baseline_for, score_from_avg_cost
from core.evaluation.scoring.kick_quality import score_kick_quality
from core.evaluation.scoring.schemes import DUAL_STREAM, DUAL_TEMPLATE, EXAM_POOL, SCHEMES, Scheme, scheme
from core.evaluation.segmentation import (
    active_window,
    classify_single_view,
    estimate_action_range,
    estimate_action_range_from_features,
    estimate_period_frames,
    select_representative_cycle,
    split_front_side,
)
from core.evaluation.templates import (
    COMPARE,
    ONLINE,
    Template,
    TemplateContract,
    TemplateError,
    TemplateStore,
    load_template,
    normalize_template_meta,
    normalizer_for,
)
from core.evaluation.templates import EXAM_POOL as EXAM_POOL_TEMPLATES

__all__ = [
    "ActionTemplate",
    "COMPARE",
    "CompareResult",
    "DUAL_STREAM",
    "DUAL_TEMPLATE",
    "DualCompareResult",
    "EXAM_POOL",
    "EXAM_POOL_TEMPLATES",
    "JointErrorStat",
    "MatchResult",
    "MatcherConfig",
    "ONLINE",
    "OnlineActionMatcher",
    "POSE33_BASELINE",
    "RepetitionMatch",
    "SCHEMES",
    "Scheme",
    "Template",
    "TemplateContract",
    "TemplateError",
    "TemplateMatch",
    "TemplateStore",
    "active_window",
    "assert_feature_layout_match",
    "avg_cost_of",
    "baseline_for",
    "center_x",
    "classify_single_view",
    "compare_dual_streams",
    "compare_dual_templates",
    "estimate_action_range",
    "estimate_action_range_from_features",
    "estimate_period_frames",
    "exam_pool",
    "facing_direction",
    "foot_edges_x",
    "infer_front_leg",
    "infer_kick_leg",
    "load_template",
    "load_template_library",
    "match_templates",
    "match_video",
    "match_window",
    "multi_subsequence_matches",
    "normalize_template_meta",
    "normalizer_for",
    "schemes",
    "scheme",
    "score_from_avg_cost",
    "score_kick_quality",
    "score_view",
    "select_representative_cycle",
    "side_label",
    "sides_for_stance",
    "split_front_side",
    "subsequence_dtw",
    "subsequence_dtw_with_path",
    "template_features",
    "valid_mask_from_raw",
]
