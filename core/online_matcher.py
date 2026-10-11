# -*- coding: utf-8 -*-
"""在线动作识别门面：转发到 ``core.evaluation.matching.online``（P9 删除）。"""
from core.evaluation.matching.online import (  # noqa: F401
    ActionTemplate,
    MatchResult,
    MatcherConfig,
    OnlineActionMatcher,
    load_template_library,
    match_window,
)
