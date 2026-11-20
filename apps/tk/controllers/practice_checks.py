# -*- coding: utf-8 -*-
"""学生练习检查项全局开关的分组整理、状态文案计算与保存校验。"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.evaluation import (
    ACTIONS,
    DEFAULT_STANCE,
    catalog_checks,
    select_practice_checks,
)
from core.foundation import load_practice_disabled_checks, save_practice_disabled_checks


@dataclass(frozen=True)
class CheckGroup:
    """按基础动作段统一归组的检查项集合。"""

    key: str
    title: str
    items: tuple[dict[str, Any], ...]


def _group_meta(check: Mapping[str, Any]) -> tuple[str, str]:
    segment = str(check["segment"])
    if segment == "stance":
        phase = str(check["phase"])
        if phase == "start":
            return "stance:start", "开始实战式"
        if phase == "end":
            return "stance:end", "结束实战式"
        return f"stance:{phase}", str(check.get("phaseLabel") or phase)
    return segment, str(check.get("segmentLabel") or segment)


def build_check_groups(stance: str = DEFAULT_STANCE) -> list[CheckGroup]:
    """调用目录接口将全系统全部检查项按基础动作段统一归组。"""
    grouped: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for item in catalog_checks(stance):
        key, title = _group_meta(item)
        if key not in grouped:
            grouped[key] = (title, [])
        grouped[key][1].append(item)
    return [
        CheckGroup(key=key, title=title, items=tuple(items))
        for key, (title, items) in grouped.items()
    ]


def check_item_label(check: Mapping[str, Any]) -> str:
    """格式化单个检查项的复选框展示文案。"""
    return f"[{check['phaseLabel']}] {check['bodyPart']}：{check['name']}"


def count_enabled_checks(
    all_ids: Sequence[str], disabled_ids: Collection[str] | None = None
) -> tuple[int, int]:
    """统计给定全量检查项中当前已开启的数量与总数量。"""
    current_disabled = (
        load_practice_disabled_checks() if disabled_ids is None else disabled_ids
    )
    disabled_set = {
        cid.strip() for cid in current_disabled if isinstance(cid, str) and cid.strip()
    }
    enabled = sum(1 for cid in all_ids if cid not in disabled_set)
    return enabled, len(all_ids)


def checks_button_text(disabled_ids: Collection[str] | None = None) -> str:
    """生成控制面板上的检查项设置按钮文案。"""
    all_ids = [item["id"] for item in catalog_checks()]
    enabled, total = count_enabled_checks(all_ids, disabled_ids)
    return f"检查项设置（已开 {enabled}/{total} 项）"


def checks_summary_text(enabled: int, total: int) -> str:
    """生成设置弹窗顶部的全局生效摘要文案。"""
    return f"已启用 {enabled} / {total} 项（全局统一生效，自动作用于所有包含对应阶段的练习动作）"


def validate_and_save_checks(
    all_ids: Sequence[str], enabled_ids: Collection[str]
) -> tuple[str, ...]:
    """校验至少保留 1 个开启项后，将关闭项列表持久化到用户偏好。"""
    enabled_set = {
        cid.strip() for cid in enabled_ids if isinstance(cid, str) and cid.strip()
    }
    if not any(cid in enabled_set for cid in all_ids):
        raise ValueError("至少需要保留 1 个开启的检查项")
    disabled = [cid for cid in all_ids if cid not in enabled_set]
    return save_practice_disabled_checks(disabled)


def zero_enabled_actions(disabled_ids: Collection[str]) -> tuple[str, ...]:
    """返回在给定关闭列表下已无任何开启检查项的动作中文名元组。"""
    zero_actions: list[str] = []
    for action, label in ACTIONS.items():
        try:
            select_practice_checks(action, DEFAULT_STANCE, disabled_checks=disabled_ids)
        except ValueError:
            zero_actions.append(label)
    return tuple(zero_actions)


def validate_action_checks(
    action: str,
    stance: str = DEFAULT_STANCE,
    disabled_ids: Collection[str] | None = None,
    *,
    pose_variant: str = "full",
) -> None:
    """复用核心筛选逻辑校验指定动作在当前全局开关下至少有 1 个开启项。"""
    current_disabled = (
        load_practice_disabled_checks() if disabled_ids is None else disabled_ids
    )
    select_practice_checks(
        action,
        stance,
        pose_variant=pose_variant,
        disabled_checks=current_disabled,
    )
