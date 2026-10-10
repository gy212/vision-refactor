# -*- coding: utf-8 -*-
"""正侧视角权重表：类型、校验与归一化（设计文档 §3 第 41 项）。

考试分数合并（``scoring.exam_pool.VIEW_WEIGHTS``，按动作）与练习证据融合（``FeedbackConfig.front_weights``，
按动作族与检查项）共用这里的权重读取；合并方式各自保留：考试按分数加权，练习按证据支持度（``fusion_support``）。
"""
from __future__ import annotations

import math
from collections.abc import Mapping

ViewWeights = Mapping[str, float]  # {"front": w, "side": w}
FrontWeightTable = Mapping[str, Mapping[str, float]]  # 动作族 → {"default": w, 检查项: w}；值为正面权重
FAMILIES = ("stance", "straight", "hook", "kick")


def normalized_pair(weights: ViewWeights) -> tuple[float, float] | None:
    """``{"front", "side"}`` 归一成和为 1 的 ``(正面, 侧面)``；两者之和不为正时为 None（不合并）。"""
    w_front = float(weights.get("front", 0.0))
    w_side = float(weights.get("side", 0.0))
    total = w_front + w_side
    if total <= 0:
        return None
    return w_front / total, w_side / total


def family_of(segment: str) -> str:
    """练习动作段所属的权重族：实战式、踢腿、摆拳、其余按直拳。"""
    return "stance" if segment == "stance" else "kick" if "kick" in segment else "hook" if "hook" in segment else "straight"


def check_front_weights(table, known_codes) -> None:
    """练习的正面权重表：四个动作族齐全，每族有 ``default`` 且只用已知检查项，值在 (0, 1) 内（侧面 = 1 - 正面）。"""
    if not isinstance(table, dict) or set(table) != set(FAMILIES):
        raise ValueError("视角权重必须包含 stance/straight/hook/kick 四类")
    for weights in table.values():
        if not isinstance(weights, dict) or "default" not in weights or set(weights) - {"default", *known_codes}:
            raise ValueError("视角权重必须包含default且使用已知检查项名称")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not 0 < v < 1 for v in weights.values()):
            raise ValueError("双视角的正面权重必须大于0且小于1，侧面权重为1减正面权重")


def practice_view_weights(code: str, segment: str, table: FrontWeightTable,
                          primary_view: Mapping[str, str]) -> dict[str, float]:
    """练习检查项的正侧权重：只适用单视角公式的项为 1/0，其余取动作族里该项（缺省 ``default``）的正面权重。"""
    if code in primary_view:
        return {view: float(view == primary_view[code]) for view in ("front", "side")}
    weights = table[family_of(segment)]
    front = weights.get(code, weights["default"])
    return {"front": front, "side": 1 - front}
