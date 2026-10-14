"""结果页拆分的差分基准：遍历控件树，记下每个控件的类型、文字、配色、字体、图片尺寸与布局参数。

P8 拆分前用旧 ``FeedbackResultPanel`` 走一遍固定的交互（加载记录、切换阶段、切换状态筛选、
逐项选中），把每一步快照的摘要写进 ``tests/fixtures/result_page/``；拆分后的面板走同样的交互，逐项比对。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from tests.report_html_cases import _evidence, _with_video
from tests.test_feedback_report import make_synthetic_check, make_synthetic_record

FIXTURE = Path(__file__).parent / "fixtures" / "result_page" / "snapshots.json"
_OPTIONS = ("text", "bg", "fg", "font", "relief", "bd", "padx", "pady", "width", "height",
            "wraplength", "anchor", "justify", "highlightbackground", "highlightthickness", "state")


def _widget_state(widget: tk.Misc) -> dict:
    state: dict = {"class": widget.winfo_class()}
    for option in _OPTIONS:
        try:
            value = widget.cget(option)
        except tk.TclError:
            continue
        state[option] = str(value)
    try:
        variable = widget.cget("textvariable")
        if str(variable):
            state["textvariable"] = widget.getvar(str(variable))
    except tk.TclError:
        pass
    try:
        image = str(widget.cget("image"))
        if image:
            state["image"] = [int(widget.tk.call("image", "width", image)), int(widget.tk.call("image", "height", image))]
    except tk.TclError:
        pass
    manager = widget.winfo_manager()
    if manager:
        info = widget.pack_info() if manager == "pack" else widget.grid_info() if manager == "grid" else {}
        state["layout"] = {manager: {k: str(v) for k, v in info.items() if k != "in"}}
    if isinstance(widget, ttk.Treeview):
        state["rows"] = [[iid, list(widget.item(iid, "values")), list(widget.item(iid, "tags"))]
                         for iid in widget.get_children()]
        state["selection"] = list(widget.selection())
    if isinstance(widget, tk.Text):
        state["content"] = widget.get("1.0", "end")
    return state


def snapshot(widget: tk.Misc) -> list:
    """控件树的先序遍历快照（不含当前未摆放的子树，也记下其存在）。"""
    widget.update_idletasks()
    return [_widget_state(widget), [snapshot(child) for child in widget.winfo_children()]]


def build_records(root: Path) -> dict[str, dict]:
    checks = [
        make_synthetic_check(code="stance_elbow", phase="start", review="confirmed", evidence=["front:0", "side:1"]),
        make_synthetic_check(code="guard_low", segment="front_straight", phase="motion", evidence=["front:2"]),
        make_synthetic_check(code="guard_back", segment="rear_straight", phase="motion", status="unable", evidence=[]),
        make_synthetic_check(code="hip_turn", segment="rear_straight", phase="finish", review="revoked", evidence=["side:3"]),
        make_synthetic_check(code="stance_knee", phase="end", status="not_observed", evidence=["front:4"]),
        make_synthetic_check(code="vertical_fist", segment="front_straight", phase="finish", status="pending_rule"),
    ]
    evidence = [_evidence(v, f) for f in range(6) for v in ("front", "side")]
    for ev in evidence:
        ev["phase"] = "motion"
    with_video = _with_video(make_synthetic_record(checks=checks, evidence=evidence), root)
    clean = make_synthetic_record(checks=[make_synthetic_check(status="not_observed")], action="stance")
    broken = make_synthetic_record()
    broken["result"]["checks"] = [{"id": "x", "status": "candidate", "review": "revoked"}]
    return {"with_video": with_video, "clean": clean, "revoked_only": broken}


def drive(panel, record: dict, *, select_phase, select_status, select_check) -> list:
    """固定交互序列，返回每一步之后的快照。"""
    steps = []
    panel.set_record(record)
    steps.append(["loaded", snapshot(panel)])
    for phase in ["start", "motion", "finish", "end", "segment", None]:
        select_phase(phase)
        steps.append([f"phase:{phase}", snapshot(panel)])
    for status in ["candidate", "not_observed", "unable", "pending_rule", "revoked", "all"]:
        select_status(status)
        steps.append([f"status:{status}", snapshot(panel)])
    for iid in list(panel.check_tree.get_children()):
        select_check(iid)
        steps.append([f"check:{iid}", snapshot(panel)])
    panel.clear()
    steps.append(["cleared", snapshot(panel)])
    return steps


def digest(snap: list) -> str:
    """快照的摘要；基准只存摘要，失败时报告第一处不一致的步骤。"""
    return hashlib.sha256(json.dumps(snap, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
