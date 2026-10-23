# -*- coding: utf-8 -*-
"""样本清单 CSV 的读取与校验：逐行标注（可按"第几次"分行）归并成逐对样本，并给出每对的标注口径。

一行是一对视频（正面 + 侧面）的一次标注。第一批没有"第几次"，一对一行；第二批起每段做 2 次，
教师对两次分别判定，同一对写成两行（``repetition`` 为 1、2），视频、类别、动作等对级字段必须一致。
逐次切分落地（规则修复计划步骤 8）之前，评估按整段进行，"第几次"只参与标注汇总。

列（表头可用英文键，也可用括号里的中文名）：

- ``pair_id``（样本编号）：对的唯一编号，必填；
- ``repetition``（第几次）：空或正整数，同一对内不重复；
- ``segment_dir``（片段目录）：相对样本根目录，内含 ``front.mp4``、``side.mp4``；
- ``category``（类别）：A 标准示范 / B 标定 / C 错误样本 / D 边界；
- ``person``（人员编号）、``action``（动作）、``note``（备注）：原样记录；
- ``rule_actions``（规则动作）：v5 目录动作键，多个用 ``|`` 分隔，空 = 不评估；
- ``error_item``（故意做错的项）：首个空白前是项编号（如 ``C7``），空或"无" = 没有故意做错；
- ``expected_codes``（预期检查项）：该错误应触发的检查项 code，多个用 ``|`` 分隔；
- ``teacher_verdict``（教师判定）：空 = 未判定；``对``/``错``/``看不清`` 是整体判定；
  也可逐项写 ``C7=错;C1=对``（逐项优先，``*=对`` 表示其余项）；
- ``front_rotation``（正面旋转）：推理前把正面画面顺时针转的度数（0/90/180/270），用于录反方向的样本。
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("pair_id", "repetition", "segment_dir", "category", "person", "action", "rule_actions",
           "error_item", "expected_codes", "teacher_verdict", "front_rotation", "note")
REQUIRED = ("pair_id", "segment_dir", "category")
ALIASES = {
    "样本编号": "pair_id", "第几次": "repetition", "片段目录": "segment_dir", "类别": "category",
    "人员编号": "person", "动作": "action", "规则动作": "rule_actions", "故意做错的项": "error_item",
    "预期检查项": "expected_codes", "教师判定": "teacher_verdict", "正面旋转": "front_rotation", "备注": "note",
}
PAIR_FIELDS = ("segment_dir", "category", "person", "action", "rule_actions", "front_rotation")
CATEGORIES = ("A", "B", "C", "D")
VERDICTS = ("对", "错", "看不清")
ROTATIONS = (0, 90, 180, 270)
NO_ERROR = ("", "无")

# 每对的标注口径
ROLE_CORRECT = "correct"    # 正确样本：用于误报率
ROLE_ERROR = "error"        # 错误样本：用于检出率
ROLE_EXCLUDED = "excluded"  # 教师判"看不清"或判错但不是故意项等，不计入两类指标


class ManifestError(ValueError):
    """清单格式错误；消息带 CSV 行号。"""


@dataclass(frozen=True)
class Label:
    """一行标注（一对视频的一次）。"""

    line: int
    repetition: int | None
    error_item: str
    expected_codes: tuple[str, ...]
    verdicts: dict[str, str]
    note: str

    @property
    def item_id(self) -> str:
        return self.error_item.split()[0] if self.error_item else ""

    def verdict_for(self, item: str) -> str:
        return self.verdicts.get(item, self.verdicts.get("*", ""))


@dataclass(frozen=True)
class SamplePair:
    """一对视频及其全部标注行。"""

    pair_id: str
    segment_dir: str
    category: str
    person: str
    action: str
    rule_actions: tuple[str, ...]
    front_rotation: int
    labels: tuple[Label, ...]

    def video(self, root: Path, view: str) -> Path:
        return Path(root) / self.segment_dir / f"{view}.mp4"

    @property
    def repetitions(self) -> str:
        return "|".join(str(lb.repetition) for lb in self.labels if lb.repetition is not None)


@dataclass(frozen=True)
class PairRole:
    """一对样本在指标中的口径：正确 / 错误 / 不计入，错误样本带预期检查项与错误项编号。"""

    role: str
    error_item: str
    expected_codes: tuple[str, ...]
    teacher_confirmed: bool


def split_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split("|") if part.strip())


def parse_verdict(text: str, line: int = 0) -> dict[str, str]:
    """教师判定 → ``{项编号或 "*": 对/错/看不清}``；空串得空字典。"""
    text = text.strip()
    if not text:
        return {}
    if "=" not in text:
        if text not in VERDICTS:
            raise ManifestError(f"第 {line} 行：教师判定只能是 {'/'.join(VERDICTS)} 或 项=判定 列表，得到 {text!r}")
        return {"*": text}
    out = {}
    for part in (p.strip() for p in text.replace("；", ";").split(";")):
        if not part:
            continue
        item, _, value = (s.strip() for s in part.partition("="))
        if not item or value not in VERDICTS:
            raise ManifestError(f"第 {line} 行：逐项判定 {part!r} 格式应为 项=对/错/看不清")
        out[item] = value
    return out


def _canonical_header(header: Iterable[str]) -> list[str]:
    names = [ALIASES.get(h.strip().lstrip("﻿"), h.strip().lstrip("﻿")) for h in header]
    missing = [c for c in REQUIRED if c not in names]
    if missing:
        raise ManifestError(f"清单缺少列：{', '.join(missing)}")
    return names


def _int_field(value: str, line: int, name: str, allowed: tuple[int, ...] | None = None) -> int | None:
    if not value:
        return None
    try:
        number = int(value)
    except ValueError:
        raise ManifestError(f"第 {line} 行：{name} 应为整数，得到 {value!r}") from None
    if (allowed is not None and number not in allowed) or (allowed is None and number < 1):
        raise ManifestError(f"第 {line} 行：{name} 取值 {number} 不合法")
    return number


def _label(row: dict, line: int) -> Label:
    error_item = "" if row.get("error_item", "") in NO_ERROR else row["error_item"]
    expected = split_list(row.get("expected_codes", ""))
    if expected and not error_item:
        raise ManifestError(f"第 {line} 行：填了预期检查项却没有故意做错的项")
    return Label(line=line, repetition=_int_field(row.get("repetition", ""), line, "第几次"),
                 error_item=error_item, expected_codes=expected,
                 verdicts=parse_verdict(row.get("teacher_verdict", ""), line), note=row.get("note", ""))


def _pair(rows: list[tuple[int, dict]]) -> SamplePair:
    first_line, first = rows[0]
    for line, row in rows[1:]:
        for name in PAIR_FIELDS:
            if row.get(name, "") != first.get(name, ""):
                raise ManifestError(f"第 {line} 行：同一对 {first['pair_id']} 的 {name} 与第 {first_line} 行不一致")
    if first["category"] not in CATEGORIES:
        raise ManifestError(f"第 {first_line} 行：类别只能是 {'/'.join(CATEGORIES)}，得到 {first['category']!r}")
    labels = tuple(_label(row, line) for line, row in rows)
    reps = [lb.repetition for lb in labels]
    if len(labels) > 1 and (None in reps or len(set(reps)) != len(reps)):
        raise ManifestError(f"对 {first['pair_id']} 有多行时，每行的第几次必须填写且不重复")
    rotation = _int_field(first.get("front_rotation", ""), first_line, "正面旋转", ROTATIONS) or 0
    return SamplePair(pair_id=first["pair_id"], segment_dir=first["segment_dir"], category=first["category"],
                      person=first.get("person", ""), action=first.get("action", ""),
                      rule_actions=split_list(first.get("rule_actions", "")), front_rotation=rotation,
                      labels=tuple(sorted(labels, key=lambda lb: lb.repetition or 0)))


def read_manifest(path: Path | str) -> list[SamplePair]:
    """读清单 CSV（UTF-8，可带 BOM），按 ``pair_id`` 归并，保持首次出现的顺序。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = _canonical_header(next(reader))
        except StopIteration:
            raise ManifestError("清单是空文件") from None
        grouped: dict[str, list[tuple[int, dict]]] = {}
        for line, cells in enumerate(reader, start=2):
            if not any(c.strip() for c in cells):
                continue
            row = {name: (cells[i].strip() if i < len(cells) else "") for i, name in enumerate(header)}
            for name in REQUIRED:
                if not row.get(name):
                    raise ManifestError(f"第 {line} 行：{name} 不能为空")
            grouped.setdefault(row["pair_id"], []).append((line, row))
    return [_pair(rows) for rows in grouped.values()]


def pair_role(pair: SamplePair) -> PairRole:
    """整段评估时一对样本的口径。

    - 错误样本：有故意做错的项，且教师没有对该项判"对"或"看不清"（未判定时按推定错误计）；
    - 正确样本：没有故意做错的项，教师没有判任何项为"错"或"看不清"（未判定时按推定正确计）；
    - 其余不计入。多行（多次）时任一次满足错误样本即为错误样本，正确样本要求每一次都正确。
    """
    errors = [lb for lb in pair.labels if lb.error_item and lb.verdict_for(lb.item_id) not in ("对", "看不清")]
    if errors:
        codes = tuple(dict.fromkeys(code for lb in errors for code in lb.expected_codes))
        confirmed = all(lb.verdict_for(lb.item_id) == "错" for lb in errors)
        return PairRole(ROLE_ERROR, errors[0].error_item, codes, confirmed)
    clean = all(not lb.error_item and not ({"错", "看不清"} & set(lb.verdicts.values())) for lb in pair.labels)
    confirmed = all(lb.verdicts.get("*") == "对" for lb in pair.labels)
    return PairRole(ROLE_CORRECT if clean else ROLE_EXCLUDED, "", (), clean and confirmed)


__all__ = [
    "COLUMNS", "Label", "ManifestError", "PairRole", "ROLE_CORRECT", "ROLE_ERROR", "ROLE_EXCLUDED", "SamplePair",
    "pair_role", "parse_verdict", "read_manifest", "split_list",
]
