"""历史报告只读取保存时的检查清单，缺失信息不由当前规则回填。"""

MISSING_CHECKS_NOTE = "该记录未保存详细检查清单，无法还原当时各项目的标准、启用状态和判断结果。"


def saved_checks(result: dict) -> list[dict]:
    """无清单和空清单都保留为空；只接受保存的检查对象。"""
    checks = result.get("checks")
    return (
        [check for check in checks if isinstance(check, dict)] if isinstance(checks, list) else []
    )
