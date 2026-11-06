"""规则启用状态：区分待验证、验证未通过与模型能力限制。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

AvailabilityState = Literal[
    "enabled",
    "trial",
    "validation_pending",
    "calibration_failed",
    "unsupported",
]

LABELS = {
    "enabled": "已启用",
    "trial": "试用中",
    "validation_pending": "待样本验证",
    "calibration_failed": "样本验证未通过",
    "unsupported": "模型能力不支持",
}


@dataclass(frozen=True)
class RuleAvailability:
    """启用依据随结果保存，历史报告不依赖后来修改的规则目录。"""

    state: AvailabilityState
    reason: str = ""
    evidence: str = ""

    def to_dict(self) -> dict:
        return {**asdict(self), "label": LABELS[self.state]}
