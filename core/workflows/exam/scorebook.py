# -*- coding: utf-8 -*-
"""考试成绩台账：考生行模型、计分行策略、异步落盘与 Excel 导出，以及按 ``exam_run_id`` 回填录后结果的 ``ScorebookRegistry``。

分数口径：内部 front/side/combined_score 为 0..1；导出时正侧 ×100（一位小数），
综合用 combined_percent（0–100 整数）。P4 从 ``core/exam_roster.py`` 的台账部分迁入。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from core.foundation.atomic_io import atomic_target
from core.foundation.tasks import CancelToken, TaskGroup

# 各台账的后台落盘线程
_WRITERS = TaskGroup("scorebook")

EXPORT_HEADERS = (
    "序号",
    "学号",
    "姓名",
    "班级",
    "综合分",
    "正面分",
    "侧面分",
    "状态",
    "错误信息",
    "片段ID",
    "录制目录",
    "叫号时间",
    "开录时间",
    "结束时间",
    "备注",
    "尝试次数",
    "是否计分行",
)

RowStatus = Literal[
    "pending",
    "recording",
    "processing",
    "completed",
    "failed",
    "skipped",
    "superseded",
]

@dataclass(frozen=True)
class ExamCandidate:
    order: int
    student_id: str
    name: str
    class_name: str = ""
    note: str = ""


@dataclass
class ExamResultRow:
    candidate: ExamCandidate
    status: RowStatus = "pending"
    segment_id: str | None = None
    segment_dir: str | None = None
    front_score: float | None = None  # 0..1
    side_score: float | None = None  # 0..1
    combined_score: float | None = None  # 0..1
    combined_percent: int | None = None  # 0..100
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)
    # 逐动作子分：动作名 → 0..1 相似度分（宽表导出为「<动作>分」列，×100 一位小数）
    action_scores: dict[str, float] = field(default_factory=dict)
    called_at: str | None = None
    record_started_at: str | None = None
    record_ended_at: str | None = None
    attempt_index: int = 1
    is_scoring_row: bool = True
    row_id: str = ""

    def __post_init__(self) -> None:
        if not self.row_id:
            self.row_id = (
                f"{self.candidate.student_id}#{self.attempt_index}"
                f"@{self.candidate.order}"
            )


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_front_side_export(score: float | None) -> str | float:
    if score is None:
        return ""
    return round(float(score) * 100.0, 1)


def collect_action_names(rows: Iterable[ExamResultRow]) -> list[str]:
    """按各行首次出现顺序收集所有动作名（去重保序）→ 宽表动作列顺序。"""
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row.action_scores:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def row_to_export_values(
    row: ExamResultRow, action_names: list[str] | None = None
) -> list[Any]:
    err = ""
    if row.error_code or row.error_message:
        err = f"{row.error_code or ''}: {row.error_message or ''}".strip(": ")
    values: list[Any] = [
        row.candidate.order,
        row.candidate.student_id,
        row.candidate.name,
        row.candidate.class_name,
        row.combined_percent if row.combined_percent is not None else "",
        _format_front_side_export(row.front_score),
        _format_front_side_export(row.side_score),
        row.status,
        err,
        row.segment_id or "",
        row.segment_dir or "",
        row.called_at or "",
        row.record_started_at or "",
        row.record_ended_at or "",
        row.candidate.note,
        row.attempt_index,
        "是" if row.is_scoring_row else "否",
    ]
    # 动作分列追加表尾：×100 一位小数，缺失动作留空
    for name in action_names or []:
        values.append(_format_front_side_export(row.action_scores.get(name)))
    return values


def write_scorebook_xlsx(path: Path | str, rows: Iterable[ExamResultRow]) -> None:
    """同步写出成绩表（原子 replace）。调用方应处理 PermissionError。"""
    from openpyxl import Workbook

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    action_names = collect_action_names(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "成绩"
    ws.append(list(EXPORT_HEADERS) + [f"{n}分" for n in action_names])
    for row in rows:
        ws.append(row_to_export_values(row, action_names))

    # 临时文件用 ASCII 名，避免部分环境下中文文件名 + 前导点的怪异行为
    with atomic_target(path) as tmp:
        wb.save(tmp)
        wb.close()


class ExamScorebook:
    """考试成绩台账：行列表 + 可选异步落盘。"""

    def __init__(
        self,
        *,
        path: Path | None = None,
        write_fn: Callable[[Path, list[ExamResultRow]], None] | None = None,
        throttle_s: float = 2.0,
    ) -> None:
        self.path = Path(path) if path else None
        self._write_fn = write_fn or write_scorebook_xlsx
        self._throttle_s = float(throttle_s)
        self._lock = threading.RLock()
        # 串行化落盘，配合 generation 防止旧快照覆盖新快照
        self._write_lock = threading.Lock()
        self._rows: list[ExamResultRow] = []
        self._by_segment: dict[str, ExamResultRow] = {}
        self._dirty = False
        self._generation = 0
        self._written_generation = 0
        self._last_write = 0.0
        self._write_error: str | None = None
        self._closed = False
        self._worker = None
        self._wake = threading.Event()
        if self.path is not None:
            self._worker = _WRITERS.spawn(
                "exam-scorebook-writer", self._writer_loop, owner="exam-scorebook", close_budget_s=3.0
            )

    @property
    def rows(self) -> list[ExamResultRow]:
        with self._lock:
            return list(self._rows)

    @property
    def write_error(self) -> str | None:
        with self._lock:
            return self._write_error

    def load_candidates(self, candidates: list[ExamCandidate]) -> None:
        with self._lock:
            self._rows = [
                ExamResultRow(candidate=c, attempt_index=1, is_scoring_row=True)
                for c in candidates
            ]
            self._by_segment.clear()
            self._mark_dirty_unlocked()

    def append_retest(self, candidate: ExamCandidate) -> ExamResultRow:
        """队尾追加同学号重考行；计分行策略见 ``_apply_scoring_policy_unlocked``。"""
        with self._lock:
            attempts = [
                r.attempt_index
                for r in self._rows
                if r.candidate.student_id == candidate.student_id
            ]
            next_attempt = (max(attempts) if attempts else 0) + 1
            # 已有成功分时，新补考暂不计分，直至其 completed
            has_completed = any(
                r.candidate.student_id == candidate.student_id
                and r.status in {"completed", "superseded"}
                and r.combined_percent is not None
                for r in self._rows
            )
            row = ExamResultRow(
                candidate=candidate,
                attempt_index=next_attempt,
                is_scoring_row=not has_completed,
            )
            self._rows.append(row)
            self._apply_scoring_policy_unlocked(candidate.student_id)
            self._mark_dirty_unlocked()
            return row

    def get_by_segment(self, segment_id: str) -> ExamResultRow | None:
        with self._lock:
            return self._by_segment.get(segment_id)

    def bind_segment(
        self, row: ExamResultRow, *, segment_id: str, segment_dir: str | None
    ) -> None:
        with self._lock:
            if row.segment_id and row.segment_id in self._by_segment:
                self._by_segment.pop(row.segment_id, None)
            row.segment_id = segment_id
            row.segment_dir = segment_dir
            self._by_segment[segment_id] = row
            self._mark_dirty_unlocked()

    def update_row(self, row: ExamResultRow, **fields: Any) -> None:
        with self._lock:
            # 终态保护：
            # - completed/skipped/superseded 不得被 recording/processing 等弱状态盖回
            # - failed 允许回到 recording（BEGIN 失败后同人重试，见 begin_failed 契约）
            _hard_terminal = frozenset(
                {"completed", "skipped", "superseded", "cancelled"}
            )
            new_status = fields.get("status")
            if (
                new_status is not None
                and row.status in _hard_terminal
                and new_status not in _hard_terminal
                and new_status != "failed"
                and new_status != row.status
            ):
                fields = {k: v for k, v in fields.items() if k != "status"}
            for key, value in fields.items():
                if not hasattr(row, key):
                    raise AttributeError(key)
                setattr(row, key, value)
            if row.status in {
                "completed",
                "failed",
                "skipped",
                "superseded",
                "processing",
                "pending",
            }:
                self._apply_scoring_policy_unlocked(row.candidate.student_id)
            self._mark_dirty_unlocked(force=row.status in {
                "completed", "failed", "skipped", "superseded", "processing"
            })

    def _apply_scoring_policy_unlocked(self, student_id: str) -> None:
        """同学号计分行策略：

        - 有成功分：最后一次 completed（含可恢复的 superseded）为唯一计分行；
          更早成功 → superseded 且 is_scoring_row=False。
        - 无成功分：仅 attempt_index 最大的一行 is_scoring_row=True
          （避免「首次成功 + 补考失败」双行同时计分，也避免双失败双计分）。
        """
        rows = [r for r in self._rows if r.candidate.student_id == student_id]
        if not rows:
            return
        completed = [
            r
            for r in rows
            if r.combined_percent is not None
            and r.status in {"completed", "superseded"}
        ]
        if completed:
            completed.sort(key=lambda r: r.attempt_index)
            winner = completed[-1]
            for r in rows:
                if r is winner:
                    r.status = "completed"
                    r.is_scoring_row = True
                else:
                    if r.combined_percent is not None and r.status in {
                        "completed",
                        "superseded",
                    }:
                        r.status = "superseded"
                    r.is_scoring_row = False
            return
        last = max(rows, key=lambda r: r.attempt_index)
        for r in rows:
            r.is_scoring_row = r is last

    def scoring_rows(self) -> list[ExamResultRow]:
        """导出用：每学号优先最后成功，否则最后一次尝试。"""
        with self._lock:
            by_sid: dict[str, list[ExamResultRow]] = {}
            for r in self._rows:
                by_sid.setdefault(r.candidate.student_id, []).append(r)
            out: list[ExamResultRow] = []
            # 保持首次出现顺序
            seen: list[str] = []
            for r in self._rows:
                if r.candidate.student_id not in seen:
                    seen.append(r.candidate.student_id)
            for sid in seen:
                group = by_sid[sid]
                completed = [x for x in group if x.status == "completed"]
                if completed:
                    out.append(max(completed, key=lambda x: x.attempt_index))
                else:
                    out.append(max(group, key=lambda x: x.attempt_index))
            return out

    def flush(self, *, timeout: float = 5.0) -> None:
        """请求立即写盘并等待；同步落盘且不让旧快照覆盖更新代。"""
        with self._lock:
            self._dirty = True
            self._last_write = 0.0
            path = self.path
        self._wake.set()
        if path is None:
            return
        self._persist_latest(force=True)
        _ = timeout  # API 兼容

    def close(self) -> None:
        """关闭台账：有限次终刷后停止后台 writer（文件被锁时不永久重试）。"""
        with self._lock:
            self._closed = True
            self._dirty = True
        self._wake.set()
        # 同步有限次终刷（例如 Excel 占用时最多 2 次）
        if self.path is not None:
            for _ in range(2):
                self._persist_latest(force=True)
                with self._lock:
                    if not self._dirty:
                        break
            # 仍失败则放弃后台重试，保留 write_error 供 UI 提示
            with self._lock:
                self._dirty = False
        worker = self._worker
        if worker is not None:
            worker.join(timeout=3.0)

    def _mark_dirty_unlocked(self, *, force: bool = False) -> None:
        self._dirty = True
        self._generation += 1
        if force:
            self._last_write = 0.0
        self._wake.set()

    def _persist_latest(self, *, force: bool = False) -> None:
        """在写锁内取最新快照写盘；仅当 generation 未前进时清 dirty。"""
        with self._write_lock:
            with self._lock:
                if self.path is None:
                    return
                if not self._dirty and self._generation == self._written_generation:
                    return
                if (
                    not force
                    and not self._closed
                    and (time.monotonic() - self._last_write) < self._throttle_s
                    and self._dirty
                ):
                    # 节流：后台循环会再试；force/flush/close 跳过
                    return
                rows_snapshot = list(self._rows)
                path = self.path
                gen = self._generation
                closed = self._closed
            try:
                self._write_fn(path, rows_snapshot)
                with self._lock:
                    if gen >= self._written_generation:
                        self._written_generation = gen
                    # 写盘期间若又有更新，保留 dirty 让后续再刷
                    if gen == self._generation:
                        self._dirty = False
                    self._write_error = None
                    self._last_write = time.monotonic()
            except PermissionError as exc:
                with self._lock:
                    self._write_error = (
                        f"scorebook_locked: 成绩文件被占用，请关闭 Excel 后重试（{exc}）"
                    )
                    # close 之后不再靠 dirty 驱动无限重试
                    self._dirty = not closed
            except OSError as exc:
                with self._lock:
                    self._write_error = f"scorebook_write_failed: {exc}"
                    self._dirty = not closed

    def _writer_loop(self, _token: CancelToken) -> None:
        while True:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                if self._closed and not self._dirty:
                    return
                if self.path is None:
                    if self._closed:
                        return
                    continue
                if not self._dirty and self._generation == self._written_generation:
                    if self._closed:
                        return
                    continue
                now = time.monotonic()
                if (
                    (now - self._last_write) < self._throttle_s
                    and not self._closed
                    and self._dirty
                ):
                    continue
                closed = self._closed
            self._persist_latest(force=closed)
            with self._lock:
                if self._closed:
                    # close 周期只做有限尝试；避免 PermissionError 下永久循环
                    self._dirty = False
                    return


# --------------------------------------------------------------------------- #
# 录后结果回填
# --------------------------------------------------------------------------- #

_PROCESSING_STATUSES = frozenset({"queued", "transcoding", "validating", "comparing"})


def apply_segment_update(scorebook: ExamScorebook, update: Any) -> ExamResultRow | None:
    """把一条片段更新写进台账；台账里没有该片段时返回 None。"""
    row = scorebook.get_by_segment(str(update.segment_id))
    if row is None:
        return None
    if update.status == "completed":
        # 逐动作子分：{name: 0..1}（同名后者覆盖前者，正常不重名）
        action_scores = {
            str(d.get("name")): float(d.get("score") or 0.0)
            for d in getattr(update, "action_scores", ()) or ()
            if d.get("name")
        }
        scorebook.update_row(
            row,
            status="completed",
            front_score=float(update.front_score or 0.0),
            side_score=float(update.side_score or 0.0),
            combined_percent=int(update.combined_percent or 0),
            combined_score=float(update.combined_percent or 0) / 100.0,
            action_scores=action_scores,
            error_code=None,
            error_message=None,
        )
    elif update.status in {"failed", "skipped", "cancelled"}:
        scorebook.update_row(
            row,
            status="failed" if update.status == "failed" else update.status,
            error_code=update.error_code,
            error_message=update.message,
        )
    elif update.status in _PROCESSING_STATUSES:
        scorebook.update_row(row, status="processing")
    return row


class ScorebookRegistry:
    """按 ``exam_run_id`` 持有各场考试的台账，录后结果按 run_id 回填；考试面板关闭后台账仍在这里接收回填。

    同一台账可以登记在多个 run_id 下（整场结束后追加重考会开新场次但沿用台账）。面板持有台账时回填后由面板刷新；
    面板放手后 ``detach``，此后每次回填若已无处理中的行就落盘一次。``close_all`` 在关窗最后一步落盘并关闭全部台账。
    更新没带 run_id、或 run_id 未登记时，按片段 id 在全部台账里找。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, ExamScorebook] = {}
        self._books: list[ExamScorebook] = []
        self._detached: list[ExamScorebook] = []

    def register(self, run_id: str, scorebook: ExamScorebook) -> None:
        with self._lock:
            self._by_run[str(run_id)] = scorebook
            if not any(book is scorebook for book in self._books):
                self._books.append(scorebook)
            self._detached = [book for book in self._detached if book is not scorebook]

    def detach(self, scorebook: ExamScorebook) -> None:
        """面板不再持有该台账（面板关闭、替换名单或关窗）：落盘一次，留在登记表里继续接收回填。"""
        with self._lock:
            if not any(book is scorebook for book in self._books):
                self._books.append(scorebook)
            if not any(book is scorebook for book in self._detached):
                self._detached.append(scorebook)
        try:
            scorebook.flush()
        except Exception:
            pass

    def apply(self, update: Any) -> tuple[ExamScorebook, ExamResultRow] | None:
        """回填一条片段更新；返回 (台账, 行)，没有台账认领该片段时返回 None。"""
        run_id = getattr(update, "exam_run_id", None)
        with self._lock:
            book = self._by_run.get(str(run_id)) if run_id else None
            books = [book] if book is not None else list(self._books)
        for candidate in books:
            row = apply_segment_update(candidate, update)
            if row is None:
                continue
            self._flush_if_detached(candidate)
            return candidate, row
        return None

    @property
    def has_detached(self) -> bool:
        with self._lock:
            return bool(self._detached)

    @property
    def has_processing(self) -> bool:
        with self._lock:
            books = list(self._books)
        return any(any(r.status == "processing" for r in book.rows) for book in books)

    def close_all(self) -> None:
        with self._lock:
            books = list(self._books)
            self._books.clear()
            self._by_run.clear()
            self._detached.clear()
        for book in books:
            try:
                book.flush()
                book.close()
            except Exception:
                pass

    def _flush_if_detached(self, scorebook: ExamScorebook) -> None:
        with self._lock:
            detached = any(book is scorebook for book in self._detached)
        if detached and not any(r.status == "processing" for r in scorebook.rows):
            try:
                scorebook.flush()
            except Exception:
                pass
