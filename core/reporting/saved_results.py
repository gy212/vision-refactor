"""历史报告只读取保存时的检查清单，缺失信息不由当前规则回填。"""

MISSING_CHECKS_NOTE = (
    "该记录未保存详细检查清单，无法还原当时各项目的标准、启用状态和判断结果。"
)


def saved_checks(result: dict) -> list[dict]:
    """无清单和空清单都保留为空；只接受保存的检查对象。"""
    checks = result.get("checks")
    return (
        [check for check in checks if isinstance(check, dict)]
        if isinstance(checks, list)
        else []
    )


def skipped_checks_note(result: dict, enabled_count: int) -> str:
    """只读保存记录中的跳过项清单，生成检查范围说明；未跳过时返回空串。"""
    disabled_ids = result.get("disabledCheckIds")
    if not isinstance(disabled_ids, list) or not disabled_ids:
        return ""
    skipped_n = len(disabled_ids)
    raw_total = result.get("totalCatalogChecks")
    total_n = (
        raw_total
        if isinstance(raw_total, int)
        and not isinstance(raw_total, bool)
        and raw_total > 0
        else enabled_count + skipped_n
    )
    return (
        f"本次已按设置跳过 {skipped_n} 个检查项（启用 {enabled_count}/{total_n} 项）。"
    )
