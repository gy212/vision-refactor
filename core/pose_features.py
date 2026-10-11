"""姿态特征门面：有效性、归一化、视角分数转发到 ``core.pose``，子序列 DTW 转发到 ``evaluation.matching.dtw``（P9 删除）。"""
from __future__ import annotations

from core.pose.normalize import (  # noqa: F401
    mirror_pose_features,
    normalize_pose_body_core_v1,
    normalize_pose_xy,
    normalize_pose_xy_v1,
    normalize_pose_xy_v3,
)
from core.pose.validity import (  # noqa: F401
    BLAZE33_LANDMARK_NAMES,
    CAP_ARMS,
    CAP_EARS,
    CAP_EYE_DETAILS,
    CAP_EYES,
    CAP_FACE_CENTER,
    CAP_FEET,
    CAP_HANDS,
    CAP_LEGS,
    CAP_MOUTH,
    COCO17_SUPPORTED_CAPABILITIES,
    DEFAULT_VALID_CONF_THR,
    MEDIAPIPE_VALIDITY_POLICY,
    derive_valid_mask,
    landmark_capabilities,
    landmark_names,
    landmarks_missing_for_capabilities,
)
from core.pose.view_score import pose_view_score  # noqa: F401
from core.evaluation.matching.dtw import subsequence_dtw, subsequence_dtw_with_path  # noqa: F401,E402
