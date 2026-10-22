# -*- coding: utf-8 -*-
"""录制控件组：开始/暂停/继续与结束录制、"仅录制 ↔ 自动比对"滑块、正/侧模板池、骨架开关与保存目录。

录制按钮是三态，结束写盘期间另显示"正在结束…"（``RECORD_BTN_TEXT``）；会话未运行时整组禁用。命令在控制器建好后经 ``connect`` 接上。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, Listbox, StringVar, filedialog, ttk
from typing import Any

RECORD_BTN_TEXT = {"idle": "开始录制", "recording": "暂停录制", "paused": "继续录制", "stopping": "正在结束…"}
TEMPLATE_VIEWS = (("front", "正面模板池"), ("side", "侧面模板池"))


class RecordingPanel:
    def __init__(self, parent: Any, *, root: Any, record_dir: Path, front: Iterable[Path],
                 side: Iterable[Path]) -> None:
        self.root = root
        # 双摄默认「录制+检测一条龙」；骨架默认关（预览与录制都不叠骨架、不跑姿态推理）。
        self.auto_compare_var = BooleanVar(root, value=True)
        self.auto_compare_scale_var = DoubleVar(root, value=1.0)
        self.auto_compare_hint_var = StringVar(root, value="录制后自动比对（检测一条龙）")
        self.record_skeleton_var = BooleanVar(root, value=False)
        self.record_dir_var = StringVar(root, value=str(record_dir))
        self.template_status_var = StringVar(root, value="")
        self.pools: dict[str, Listbox] = {}
        group = self.frame = ttk.Labelframe(parent, text="录制", padding=8)
        group.pack(fill="x", pady=(10, 0))
        self.record_btn = ttk.Button(group, text=RECORD_BTN_TEXT["idle"], state="disabled")
        self.record_btn.pack(fill="x")
        self.record_stop_btn = ttk.Button(group, text="结束录制", state="disabled")
        self.record_stop_btn.pack(fill="x", pady=(6, 0))
        self._build_auto_compare(group)
        self._build_pools(group, {"front": list(front), "side": list(side)})
        self.record_skeleton_check = ttk.Checkbutton(
            group, text="开启骨架（预览与录制；双摄开启后不自动比对）", variable=self.record_skeleton_var)
        self.record_skeleton_check.pack(anchor="w", pady=(4, 0))
        ttk.Label(group, text="保存目录：").pack(anchor="w", pady=(8, 0))
        row = ttk.Frame(group)
        row.pack(fill="x", pady=(2, 0))
        self.record_dir_entry = ttk.Entry(row, textvariable=self.record_dir_var)
        self.record_dir_entry.pack(side="left", fill="x", expand=True)
        self.record_dir_btn = ttk.Button(row, text="选择…")
        self.record_dir_btn.pack(side="left", padx=(8, 0))

    def _build_auto_compare(self, group: Any) -> None:
        row = ttk.Frame(group)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="仅录制").pack(side="left")
        self.auto_compare_scale = ttk.Scale(row, from_=0, to=1, orient="horizontal", length=96,
                                            variable=self.auto_compare_scale_var)
        self.auto_compare_scale.pack(side="left", padx=(6, 6), fill="x", expand=True)
        ttk.Label(row, text="自动比对").pack(side="left")
        ttk.Label(group, textvariable=self.auto_compare_hint_var).pack(anchor="w", pady=(2, 0))

    def _build_pools(self, group: Any, seeds: dict[str, list[Path]]) -> None:
        """正/侧两个模板池（正对正、侧对侧，每边可多个动作模板；动作名 = 文件 stem）。"""
        self._pool_buttons: list[tuple[str, Any, Any]] = []
        for view, label in TEMPLATE_VIEWS:
            frame = ttk.Frame(group)
            frame.pack(fill="x", pady=(6, 0))
            head = ttk.Frame(frame)
            head.pack(fill="x")
            ttk.Label(head, text=label).pack(side="left")
            add = ttk.Button(head, text="添加…", width=6)
            add.pack(side="right")
            remove = ttk.Button(head, text="移除选中", width=8)
            remove.pack(side="right", padx=(0, 4))
            self._pool_buttons.append((view, add, remove))
            box = self.pools[view] = Listbox(frame, height=4, selectmode="extended", exportselection=False)
            box.pack(fill="x", pady=(2, 0))
            for path in seeds[view]:
                box.insert("end", str(path))
        ttk.Label(group, textvariable=self.template_status_var, wraplength=320, foreground="#555").pack(
            anchor="w", pady=(2, 0))

    def connect(self, *, toggle: Callable[[], None], stop: Callable[[], None],
                add_templates: Callable[[str], None], remove_templates: Callable[[str], None],
                auto_scale: Callable[[str], None], auto_changed: Callable[[], None],
                choose_dir: Callable[[], None]) -> None:
        self.record_btn.configure(command=toggle)
        self.record_stop_btn.configure(command=stop)
        self.auto_compare_scale.configure(command=auto_scale)
        self.auto_compare_var.trace_add("write", lambda *_args: auto_changed())
        self.record_dir_btn.configure(command=choose_dir)
        for view, add, remove in self._pool_buttons:
            add.configure(command=lambda v=view: add_templates(v))
            remove.configure(command=lambda v=view: remove_templates(v))

    # ------------------------------------------------------------------ 模板池
    def pool_paths(self, view: str) -> list[Path]:
        box = self.pools[view]
        return [Path(str(box.get(i)).strip()) for i in range(box.size()) if str(box.get(i)).strip()]

    def add_pool_path(self, view: str, path: Path) -> None:
        self.pools[view].insert("end", str(path))

    def remove_selected(self, view: str) -> None:
        box = self.pools[view]
        for index in reversed(box.curselection()):
            box.delete(index)

    def ask_templates(self, view: str) -> tuple[str, ...]:
        return filedialog.askopenfilenames(
            title="添加正面模板" if view == "front" else "添加侧面模板",
            filetypes=[("模板文件", "*.npz"), ("所有文件", "*.*")],
        )

    def ask_record_dir(self, initial: str) -> str:
        return filedialog.askdirectory(title="选择录制视频保存目录", initialdir=initial)

    # ------------------------------------------------------------------ 状态联动
    def set_record_text(self, state: str) -> None:
        self.record_btn.configure(text=RECORD_BTN_TEXT[state])

    def set_controls_state(self, state: str) -> None:
        """关窗时禁用录制相关的全部控件。"""
        for widget in (self.record_btn, self.record_stop_btn, self.auto_compare_scale,
                       self.record_skeleton_check, self.record_dir_entry, self.record_dir_btn):
            widget.configure(state=state)
