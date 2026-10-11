# -*- coding: utf-8 -*-
"""考试名单的 Excel 导入与导入模板；成绩台账已迁入 ``core.workflows.exam.scorebook``，这里保留旧名转发（P7 迁入 workflows）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.workflows.exam.scorebook import (  # noqa: F401  台账部分的旧导入路径
    EXPORT_HEADERS,
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    RowStatus,
    collect_action_names,
    local_timestamp,
    row_to_export_values,
    write_scorebook_xlsx,
)

IMPORT_HEADERS = ("序号", "学号", "姓名", "班级", "备注")


class RosterError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        # 科学计数法风险：不自动“还原”学号，交给上层校验
        text = format(value, ".15g")
        return text
    return str(value).strip()


def _excel_format_is_plain_number(number_format: str | None) -> bool:
    """General / @ / 0 等视为「无自定义显示」，可安全回退为裸数字。"""
    import re

    fmt = (number_format or "").strip()
    if not fmt or fmt in {"General", "@", "0"}:
        return True
    first = fmt.split(";")[0].strip()
    first = re.sub(r"\[[^\]]*\]", "", first).strip()
    return first in {"", "General", "@", "0"}


def _tokenize_excel_number_format(number_format: str | None) -> list[tuple[str, str]] | None:
    """解析 Excel 数字格式第一段为 token 列表：('d','0'|'#') 或 ('lit', ch)。

    支持反斜杠转义（``000\\-000``）与双引号字面量（``\"ID-\"000000``）。
    不支持小数/千分位/科学计数/条件格式表达式时返回 None。
    """
    import re

    fmt = (number_format or "").strip()
    if not fmt:
        return None
    first = fmt.split(";")[0]
    first = re.sub(r"\[[^\]]*\]", "", first)
    if not first.strip() or first.strip() in {"General", "@"}:
        return None

    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(first)
    while i < n:
        ch = first[i]
        if ch == "\\":
            if i + 1 >= n:
                return None
            tokens.append(("lit", first[i + 1]))
            i += 2
            continue
        if ch == '"':
            i += 1
            buf: list[str] = []
            while i < n:
                if first[i] == '"':
                    if i + 1 < n and first[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(first[i])
                i += 1
            else:
                return None  # 未闭合引号
            for c in buf:
                tokens.append(("lit", c))
            continue
        if ch in "0#":
            tokens.append(("d", ch))
            i += 1
            continue
        if ch in "- /()_:.":
            # 字面分隔；小数点单独拒绝（与科学/会计格式混淆）
            if ch == ".":
                return None
            tokens.append(("lit", ch))
            i += 1
            continue
        if ch in {",", "E", "e", "%", "*", "?", "_"}:
            return None
        # 其他字符（含中文）视为字面量，便于少量自定义
        if ch.isalpha() or ord(ch) > 127:
            tokens.append(("lit", ch))
            i += 1
            continue
        return None
    if not any(t[0] == "d" for t in tokens):
        return None
    return tokens


def _display_text_from_number_format(value: int | float, number_format: str | None) -> str | None:
    """尽量按 Excel 显示文本还原学号。

    支持：
    - 纯补零：``000000`` + 123 → ``000123``
    - ``0`` 必选位 / ``#`` 可选位：``###000`` + 123 → ``123``（不按 # 强制补零）
    - 分隔：``000-000`` / ``000\\-000`` + 123 → ``000-123``
    - 字面量前缀：``\"ID-\"000000`` + 123 → ``ID-000123``
    - 超出占位宽度时保留字面量：``\"ID-\"000`` + 12345 → ``ID-12345``

    无法安全解析时返回 None（调用方对自定义格式应整表拒绝，不得静默裸数字）。
    """
    tokens = _tokenize_excel_number_format(number_format)
    if tokens is None:
        if _excel_format_is_plain_number(number_format):
            return str(int(value)) if float(value).is_integer() else None
        return None
    ph_types = [t[1] for t in tokens if t[0] == "d"]
    if not ph_types:
        return None
    n = int(value)
    if n < 0:
        return None
    # 仅按必选 0 位补零；# 不强制加宽
    min_len = sum(1 for p in ph_types if p == "0")
    # Excel：纯 #（无必选 0 位）对数值 0 显示为空，不得导入成 "0"
    if n == 0 and min_len == 0:
        digit_str = ""
    else:
        digit_str = str(n)
        if len(digit_str) < min_len:
            digit_str = digit_str.zfill(min_len)

    # 从右向左填入占位；多余数字保留为 excess（仍挂在字面量之后）
    slots: list[str | None] = [None] * len(ph_types)
    di = len(digit_str) - 1
    for si in range(len(ph_types) - 1, -1, -1):
        if di >= 0:
            slots[si] = digit_str[di]
            di -= 1
        elif ph_types[si] == "0":
            slots[si] = "0"
        else:
            slots[si] = None  # 未使用的 #
    excess = digit_str[: di + 1] if di >= 0 else ""

    out: list[str] = []
    si = 0
    excess_pending = excess
    for kind, payload in tokens:
        if kind == "d":
            ch = slots[si]
            si += 1
            if ch is None:
                continue
            if excess_pending:
                out.append(excess_pending)
                excess_pending = ""
            out.append(ch)
        else:
            out.append(payload)
    if excess_pending:
        # 无数字位可挂时（极端），前缀 excess
        out.insert(0, excess_pending)
    result = "".join(out)
    # 零值 + 仅 # 位：显示无数字 → 空串（调用方按空学号拒绝）
    if n == 0 and min_len == 0 and not any(c.isdigit() for c in result):
        return ""
    return result


def _student_id_from_cell(value: Any, *, row_no: int, number_format: str | None = None) -> str:
    """强制按文本语义读取学号；数值单元格优先用 number_format 显示文本。

    自定义格式无法还原时 **拒绝**（不静默变成裸数字），迫使将学号列设为文本。
    """
    if value is None:
        raise RosterError("empty_student_id", f"第 {row_no} 行学号为空")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise RosterError("empty_student_id", f"第 {row_no} 行学号为空")
        return text
    if isinstance(value, bool):
        raise RosterError(
            "student_id_not_text",
            f"第 {row_no} 行学号类型无效：{value!r}",
        )
    if isinstance(value, int):
        display = _display_text_from_number_format(value, number_format)
        if display is not None:
            if not str(display).strip():
                raise RosterError(
                    "empty_student_id",
                    f"第 {row_no} 行学号为空（显示格式下无有效文本，如 0+###）",
                )
            return display
        if _excel_format_is_plain_number(number_format):
            return str(value)
        raise RosterError(
            "student_id_not_text",
            f"第 {row_no} 行学号无法按显示格式 {number_format!r} 还原，"
            "请将该列设为文本格式后重试",
        )
    if isinstance(value, float):
        # 允许整数值 float（Excel 有时如此），但拒绝非整数 / 过大科学计数
        if not value.is_integer():
            raise RosterError(
                "student_id_not_text",
                f"第 {row_no} 行学号疑似被 Excel 读成小数/科学计数法，请将该列设为文本后重试",
            )
        if abs(value) >= 1e15:
            raise RosterError(
                "student_id_not_text",
                f"第 {row_no} 行学号过大，请将该列设为文本格式后重试",
            )
        ival = int(value)
        display = _display_text_from_number_format(ival, number_format)
        if display is not None:
            if not str(display).strip():
                raise RosterError(
                    "empty_student_id",
                    f"第 {row_no} 行学号为空（显示格式下无有效文本，如 0+###）",
                )
            return display
        if _excel_format_is_plain_number(number_format):
            return str(ival)
        raise RosterError(
            "student_id_not_text",
            f"第 {row_no} 行学号无法按显示格式 {number_format!r} 还原，"
            "请将该列设为文本格式后重试",
        )
    text = _cell_text(value)
    if not text:
        raise RosterError("empty_student_id", f"第 {row_no} 行学号为空")
    return text


def import_roster_xlsx(path: Path | str) -> list[ExamCandidate]:
    """导入第一个 sheet；失败整表拒绝。"""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RosterError("openpyxl_missing", "缺少 openpyxl，请先安装依赖") from exc

    path = Path(path)
    if not path.is_file():
        raise RosterError("file_missing", f"名单文件不存在：{path}")

    try:
        # data_only=False：保留 number_format，便于学号按「显示文本」读取（如 000123）
        wb = load_workbook(path, read_only=False, data_only=False)
    except Exception as exc:
        raise RosterError("file_unreadable", f"无法读取 Excel：{exc}") from exc

    try:
        ws = wb[wb.sheetnames[0]]
        if getattr(ws, "merged_cells", None) and ws.merged_cells.ranges:
            raise RosterError(
                "merged_cells",
                "导入表存在合并单元格，请拆分后再导入",
            )

        rows_iter = ws.iter_rows()
        try:
            header_cells = next(rows_iter)
        except StopIteration as exc:
            raise RosterError("empty_sheet", "Excel 为空") from exc

        headers = [_cell_text(c.value) for c in header_cells]
        # 去掉尾部空表头
        while headers and not headers[-1]:
            headers.pop()
        index = {name: i for i, name in enumerate(headers) if name}
        for required in ("序号", "学号", "姓名"):
            if required not in index:
                raise RosterError("missing_column", f"缺少必填列：{required}")

        candidates: list[ExamCandidate] = []
        seen_orders: set[int] = set()
        for excel_row_no, row_cells in enumerate(rows_iter, start=2):
            cells = list(row_cells)
            while len(cells) < len(headers):
                cells.append(None)  # type: ignore[arg-type]
            values = [c.value if c is not None else None for c in cells[: len(headers)]]
            # 全空行跳过
            if all(v is None or _cell_text(v) == "" for v in values):
                continue

            order_raw = values[index["序号"]]
            try:
                if isinstance(order_raw, float) and order_raw.is_integer():
                    order = int(order_raw)
                else:
                    order = int(str(order_raw).strip())
            except (TypeError, ValueError) as exc:
                raise RosterError(
                    "bad_order", f"第 {excel_row_no} 行序号无效：{order_raw!r}"
                ) from exc
            if order <= 0:
                raise RosterError("bad_order", f"第 {excel_row_no} 行序号须为正整数")
            if order in seen_orders:
                raise RosterError("duplicate_order", f"序号重复：{order}")
            seen_orders.add(order)

            sid_cell = cells[index["学号"]]
            sid_fmt = getattr(sid_cell, "number_format", None) if sid_cell is not None else None
            student_id = _student_id_from_cell(
                values[index["学号"]],
                row_no=excel_row_no,
                number_format=sid_fmt,
            )
            name = _cell_text(values[index["姓名"]])
            if not name:
                raise RosterError("empty_name", f"第 {excel_row_no} 行姓名为空")

            class_name = (
                _cell_text(values[index["班级"]]) if "班级" in index else ""
            )
            note = _cell_text(values[index["备注"]]) if "备注" in index else ""
            candidates.append(
                ExamCandidate(
                    order=order,
                    student_id=student_id,
                    name=name,
                    class_name=class_name,
                    note=note,
                )
            )

        if not candidates:
            raise RosterError("no_rows", "没有有效考生行")
        candidates.sort(key=lambda c: c.order)
        return candidates
    finally:
        wb.close()


def write_import_template(path: Path | str) -> Path:
    """写出导入模板（表头 + 示例行，学号列文本格式）。"""
    from openpyxl import Workbook
    from openpyxl.styles import numbers

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "名单"
    ws.append(list(IMPORT_HEADERS))
    ws.append([1, "2026001", "张三", "一年级1班", ""])
    ws.append([2, "2026002", "李四", "一年级1班", ""])
    for row in range(2, 4):
        ws.cell(row=row, column=2).number_format = numbers.FORMAT_TEXT
    wb.save(path)
    wb.close()
    return path
