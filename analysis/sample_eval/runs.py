# -*- coding: utf-8 -*-
"""样本评估运行目录的读写：``checks.csv``（逐对逐检查项）、``phases.csv``（逐对逐动作）、``run.json``（运行信息）。

CSV 用带 BOM 的 UTF-8，WPS/Excel 可直接打开；读回时把布尔列还原成 ``bool``/``None``，
这样指标只依赖运行目录，修改前后的两次运行可以在任何时候重新对比。
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

from core.foundation import atomic_write_json, atomic_write_text, repo_root

CHECKS_CSV = "checks.csv"
PHASES_CSV = "phases.csv"
RUN_JSON = "run.json"
REPORT_MD = "report.md"
BOOL_COLUMNS = ("expected", "alignment_reliable", "front_found", "side_found", "front_window", "side_window", "fusible")


def _columns(rows: list[dict]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    return keys


def write_csv(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_columns(rows), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "﻿" + buf.getvalue())


def _typed(row: dict) -> dict:
    for key in BOOL_COLUMNS:
        if key in row:
            row[key] = {"True": True, "False": False}.get(row[key])
    return row


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return [_typed(row) for row in csv.DictReader(fh)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    """当前提交（短）与工作区是否有改动；不在 git 仓库里时返回 ``unknown``。"""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, encoding="utf-8",
                              errors="replace", check=True).stdout.strip()

    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("（工作区有未提交改动）" if dirty else "")


def write_run(out_dir: Path, meta: dict, check_rows: list[dict], phase_rows: list[dict]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / CHECKS_CSV, check_rows)
    write_csv(out_dir / PHASES_CSV, phase_rows)
    atomic_write_json(out_dir / RUN_JSON, meta, trailing_newline=True)


def read_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """``(运行信息, 逐检查项行, 相位行)``。"""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / RUN_JSON).read_text(encoding="utf-8"))
    return meta, read_csv(run_dir / CHECKS_CSV), read_csv(run_dir / PHASES_CSV)
