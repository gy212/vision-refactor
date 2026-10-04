# -*- coding: utf-8 -*-
"""控制区布局结构断言单元测试（任务 9.4）。

设计来源：``.kiro/specs/ui-layout-redesign/design.md``
（Architecture「控制区布局结构」图、Components 2「App._build_ui() 重构」）。

本文件在**真实 Tk 控件树**上断言左侧控制区的布局结构：
  1. ``controls_inner`` 顶层分区顺序为
     [Primary Labelframe(主要操作), Separator, Secondary Labelframe(次要选项),
      Status Labelframe(状态)]（需求 1.2、1.4）。
  2. Primary_Controls 恰好按垂直顺序排列五项核心控件：
     camera_combo → model_combo → start_btn → record_btn → compare_btn，
     且均为 Primary Labelframe 的后代（需求 1.1、1.3）。
  3. Separator 位于 Primary 与 Secondary 之间（需求 1.2）。
  4. Secondary 组位于 Primary 组之后，workers_spin 为 Secondary 的后代（需求 1.4）。
  5. Sanity：root.minsize() == (800, 600)。

无显示环境（无 DISPLAY / TclError）时整体 ``pytest.skip``。

_Requirements: 1.1, 1.2, 1.3, 1.4_
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 夹具：构造真实 Tk root + App，无显示环境跳过
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """构造隐藏的 Tk root 与 App；无显示环境时 skip。

    teardown 调用 root.destroy()；不调用 mainloop。
    """
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - 环境缺 tkinter
        pytest.skip(f"tkinter 不可用：{exc}")

    from apps.app_ui import App

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"无显示环境，无法创建 Tk 窗口：{exc}")

    try:
        root.withdraw()  # 不显示窗口
        application = App(root)
    except tk.TclError as exc:  # pragma: no cover - 构建期窗口错误
        root.destroy()
        pytest.skip(f"构建 App 失败（无显示）：{exc}")

    root.update_idletasks()  # 计算几何后再读取 winfo_*
    try:
        yield application
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _is_descendant(widget, ancestor) -> bool:
    """判断 widget 是否为 ancestor 的后代（含自身）。"""
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        parent_name = w.winfo_parent()
        if not parent_name:
            return False
        w = w.nametowidget(parent_name)
    return False


def _find_labelframe(slaves, text):
    """在 pack_slaves 列表中查找 cget('text') 匹配的 ttk.Labelframe。"""
    from tkinter import ttk

    for child in slaves:
        if isinstance(child, ttk.Labelframe):
            try:
                if child.cget("text") == text:
                    return child
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


def test_controls_inner_section_order(app):
    """controls_inner 顶层分区顺序：Primary → Separator → Secondary → Status
    （需求 1.2、1.4）。"""
    from tkinter import ttk

    slaves = app.controls_inner.pack_slaves()

    # 至少包含四个顶层分区。
    assert len(slaves) >= 4, f"顶层分区数量不足：{slaves}"

    # index 0：Primary Labelframe（主要操作）。
    assert isinstance(slaves[0], ttk.Labelframe)
    assert slaves[0].cget("text") == "主要操作"

    # index 1：Separator。
    assert isinstance(slaves[1], ttk.Separator)

    # index 2：Secondary Labelframe（次要选项）。
    assert isinstance(slaves[2], ttk.Labelframe)
    assert slaves[2].cget("text") == "次要选项"

    # index 3：Status Labelframe（状态）。
    assert isinstance(slaves[3], ttk.Labelframe)
    assert slaves[3].cget("text") == "状态"


def test_primary_controls_five_in_vertical_order(app):
    """Primary_Controls 恰好五项核心控件，按垂直顺序排列且均为 Primary 组后代
    （需求 1.1、1.3）。"""
    app.root.update_idletasks()

    ordered = [
        app.camera_combo,
        app.model_combo,
        app.start_btn,
        app.record_btn,
        app.compare_btn,
    ]

    # 定位 Primary Labelframe。
    primary = _find_labelframe(app.controls_inner.pack_slaves(), "主要操作")
    assert primary is not None, "未找到 Primary Labelframe（主要操作）"

    # 每项均为 Primary 组的后代（需求 1.1）。
    for widget in ordered:
        assert _is_descendant(widget, primary), f"{widget} 不是 Primary 组后代"

    # 垂直顺序：winfo_rooty 严格递增（需求 1.3）。
    ys = [w.winfo_rooty() for w in ordered]
    for prev, cur in zip(ys, ys[1:]):
        assert prev < cur, f"五项核心控件垂直顺序不满足严格递增：{ys}"


def test_separator_between_primary_and_secondary(app):
    """分隔线存在且位于 Primary 与 Secondary 之间（index 1）（需求 1.2）。"""
    from tkinter import ttk

    assert isinstance(app.primary_secondary_separator, ttk.Separator)

    slaves = app.controls_inner.pack_slaves()
    assert slaves[1] is app.primary_secondary_separator
    # Primary 在前（index 0），Secondary 在后（index 2）。
    assert app.primary_secondary_separator is slaves[1]


def test_secondary_below_primary_and_contains_workers(app):
    """Secondary 组位于 Primary 组之后，workers_spin 为 Secondary 后代（需求 1.4）。"""
    app.root.update_idletasks()

    slaves = app.controls_inner.pack_slaves()
    primary = _find_labelframe(slaves, "主要操作")
    secondary = _find_labelframe(slaves, "次要选项")
    assert primary is not None and secondary is not None

    # Secondary 组的 y 位置在 Primary 组之后（需求 1.4）。
    assert secondary.winfo_rooty() > primary.winfo_rooty()

    # workers_spin 为 Secondary 组后代。
    assert _is_descendant(app.workers_spin, secondary)


def test_minsize_sanity(app):
    """Sanity：root.minsize() == (800, 600)。"""
    assert tuple(app.root.minsize()) == (800, 600)
