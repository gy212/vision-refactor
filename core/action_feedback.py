"""学生练习门面：规则目录在 ``core.evaluation.rules.practice_v4``，分析在 ``core.evaluation.practice``，
文本报告在 ``core.reporting.text``（P9 删除）。"""
from core.evaluation.practice import analyze_practice as analyze_feedback  # noqa: F401
from core.evaluation.rules.practice_v4 import (  # noqa: F401
    ACTIONS,
    DOC,
    PARTS,
    PHASES,
    RULE_VERSION,
    RULES,
    STANCE_RULES,
    STANCES,
    STATE_LABELS,
    SUMMARY,
    build_checks,
)
from core.reporting.text import format_fusion, format_report  # noqa: F401
