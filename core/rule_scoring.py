"""正式规则门面：规则在 ``core.evaluation.rules.formal_v1``，原始序列在 ``core.pose.raw_series``（P9 删除）。"""
from __future__ import annotations

from core.evaluation.rules.formal_v1 import (  # noqa: F401
    FORMAL_V1,
    RuleScore,
    RuleViolation,
    score_rules,
)
from core.evaluation.rules.formal_v1 import feet_parallel as _rule_feet_parallel  # noqa: F401
from core.evaluation.rules.formal_v1 import knee_slight_bend as _rule_knee_slight_bend  # noqa: F401
from core.evaluation.rules.spec import (  # noqa: F401
    RULE_STATE_EVALUATED,
    RULE_STATE_SKIPPED,
    SKIP_INSUFFICIENT_VALID_FRAMES,
    SKIP_LOW_CONFIDENCE,
    SKIP_MISSING_LANDMARKS,
    RuleState,
    SkipReason,
)
from core.pose.landmarks import (  # noqa: F401
    L_ANKLE,
    L_ELBOW,
    L_FOOT_INDEX,
    L_HEEL,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_FOOT_INDEX,
    R_HEEL,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from core.pose.raw_series import (  # noqa: F401
    Pose33RawSeries,
    extract_pose_raw,
    extract_pose_raw_series,
    slice_pose_raw_series,
)
