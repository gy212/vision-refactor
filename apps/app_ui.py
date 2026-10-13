"""Tk 桌面主程序：实时预览、双摄录制、考试与学生练习入口（P8 拆为 apps/tk/ 的组合根、控制器和视图）。"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from tkinter import BooleanVar, Canvas, DoubleVar, IntVar, Listbox, Scrollbar, StringVar, TclError, Text, Tk, Toplevel, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.evaluation import COMPARE, load_template
from core.services import compare_templates, create_template_from_video
from core.pose import missing_model_message
from core.foundation import (
    EXAM_ROI,
    PreferenceStore,
    to_jsonable,
    load_camera_selection,
    load_record_dir,
    load_template_pools,
    save_camera_selection,
    save_record_dir,
    save_template_pools,
)
from core.paths import models_dir, outputs_dir
from core import model_manager, paths, video_writer
from core.live import (
    APP_CLOSE,
    RUNNING,
    STARTING,
    CameraInput,
    FrameInput,
    LiveControls,
    LiveSessionManager,
    PresenceSample,
    RunningInfo,
    SessionConfig,
    SessionEvent,
    SessionOutcome,
    VideoFileInput,
    camera_session,
    video_session,
)
from core.media import SegmentLayout
from core.recording import (
    BeginError,
    FinishedSegment,
    ProcessingPlan,
    RecordingService,
    RecordingState,
    SegmentFinalizer,
    SegmentMeta,
    SegmentUpdate,
    default_template_paths,
    template_action_name,
    try_load_heavy_template,
    validate_template_pair,
)
from core.workflows import (
    EXAM,
    NORMAL,
    PRACTICE,
    Announcer,
    CommandExecutor,
    ModeManager,
    PlanSettings,
    PracticeSession,
    PreflightContext,
    ScorebookRegistry,
    SegmentIntent,
    WorkflowHost,
    WorkflowPorts,
    build_segment_plan,
    hip_midpoint_in_roi,
)
from apps.input_source import InputSourceState
from core import foundation
from core.media import PRIMARY, SECONDARY, CameraService, CaptureSpec, ProcessCamera
from core.media import CameraDevice as CameraEntry
from apps.feedback_panel import FeedbackControls


# Record_Toggle 三态按钮文本映射，供录制回调（任务 7.1/7.2/7.3）与 UI 重构（任务 9.x）复用。
RECORD_BTN_TEXT = {
    "idle": "开始录制",
    "recording": "暂停录制",
    "paused": "继续录制",
}

# 第二摄像头下拉的「不选」sentinel（双摄像头双面视图，issue #57）。
NO_SECOND_CAMERA = "无（单摄像头）"

# 画面旋转：摄像头竖起来拍时手动转正（USB 摄像头一般不给方向传感器，无法可靠自动判断）。
# 旋转在 cap.read() 之后立即应用，姿态检测 / 预览 / 录制统一用转正后的帧。
ROTATE_CHOICES = ("0°", "90°", "180°", "270°")
_VALID_ROTATIONS = frozenset((0, 90, 180, 270))


def _parse_rotate(text: str) -> int:
    """把下拉文本（"90°"）解析成度数；非法值回退 0。"""
    try:
        degrees = int(str(text).rstrip("°").strip())
    except (TypeError, ValueError):
        return 0
    return degrees if degrees in _VALID_ROTATIONS else 0


def _degrees_to_rotate_label(degrees: int) -> str:
    """度数 → 下拉文案；非法值回退 0°。"""
    d = int(degrees) if int(degrees) in _VALID_ROTATIONS else 0
    return f"{d}°"


def _next_rotate_cw(degrees: int) -> int:
    """顺时针旋转 90°（0→90→180→270→0）。"""
    d = int(degrees) if int(degrees) in _VALID_ROTATIONS else 0
    return (d + 90) % 360


def _choose_dual_preview_layout(
    size: tuple[int, int], size2: tuple[int, int]
) -> str:
    """两路都为竖画面时左右并排；横向、方形或混合方向时上下堆叠。"""
    w, h = size
    w2, h2 = size2
    return "side_by_side" if h > w and h2 > w2 else "stacked"

# 关窗时给采集线程收尾并等待 H.264 转码的最长时间。等待拆成短 join 片段，
# 避免 Tk 主线程在单次回调里长时间无响应。
_CLOSE_JOIN_TIMEOUT_S = 3.0
_CLOSE_JOIN_SLICE_S = 0.05
_CLOSE_POLL_MS = 50
_CAMERA_CANCEL_JOIN_TIMEOUT_S = 0.05
_CAMERA_RELEASE_WAIT_S = 3.0
# 普通模式的请求尺寸（各模式的采集格式由 core.workflows.modes 声明）
_PREVIEW_CAPTURE_SIZE = NORMAL.requirements.fmt.size


def clamp_workers(n: int) -> int:
    """将离线线程数钳制到闭区间 [1, os.cpu_count()]。

    - 小于 1 的输入钳制为 1
    - 大于主机 CPU 逻辑核心数的输入钳制为核心数
    - 当 ``os.cpu_count()`` 返回 None（无法确定核心数）时，上界回退为 1

    Validates: Requirements 7.2
    """
    cpu = os.cpu_count()
    upper = cpu if cpu and cpu >= 1 else 1
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = 1
    if value < 1:
        return 1
    if value > upper:
        return upper
    return value


def default_workers() -> int:
    """主预览默认并行 worker 数：按 CPU 逻辑核数自适应，尽量榨满多核吞吐。

    单条 VIDEO 管线在多核机上只用到少数核（32 核机整机 ~10% 利用率）；多 worker 并行
    IMAGE 推理可近线性扩展。留 2 个核给 UI/采集/系统；上限 8——实测 6 worker 已达
    ~57fps，再多是拿内存（每 worker 一份模型）换不到额外 fps。想更激进可在「线程数」
    里手调（上限 16）。
    """
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu - 2))


class CollapsibleSection:
    """A collapsible section with a toggle header."""

    def __init__(self, parent: ttk.Frame, title: str, expanded: bool = False) -> None:
        self.parent = parent
        self._expanded = BooleanVar(value=expanded)

        self.header = ttk.Frame(parent)
        self._toggle_btn = ttk.Button(
            self.header,
            text=self._get_toggle_text(),
            command=self._toggle,
            width=len(title) + 4,
        )
        self._toggle_btn.pack(side="left", anchor="w")

        self.content = ttk.Frame(parent)
        if expanded:
            self.content.pack(fill="x", pady=(4, 0))

    def _get_toggle_text(self) -> str:
        arrow = "▾" if self._expanded.get() else "▸"
        return f"{arrow} {self._toggle_btn.cget('text').lstrip('▸▾ ') if hasattr(self, '_toggle_btn') else ''}"

    def _toggle(self) -> None:
        self._expanded.set(not self._expanded.get())
        current_text = self._toggle_btn.cget("text")
        base_text = current_text.lstrip("▸▾ ")
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {base_text}")

        if self._expanded.get():
            self.content.pack(fill="x", pady=(4, 0))
        else:
            self.content.pack_forget()

    def set_title(self, title: str) -> None:
        arrow = "▾" if self._expanded.get() else "▸"
        self._toggle_btn.configure(text=f"{arrow} {title}", width=len(title) + 4)

    def pack_header(self, **kwargs) -> None:
        self.header.pack(**kwargs)

    def is_expanded(self) -> bool:
        return self._expanded.get()


class CompareWindow:
    def __init__(self, parent: Tk) -> None:
        self._win = Toplevel(parent)
        self._win.title("动作分析（模板比对）")
        self._win.geometry("720x860")
        self._win.minsize(640, 720)

        # 从视频生成模板用的基准视频路径。
        self.base_video_var = StringVar(value="")
        self.target_video_var = StringVar(value="")

        # 动作分析为后端黑盒链路：统一 heavy，与主界面「预览姿态模型」分离。
        self.pose_var = StringVar(value="heavy")
        self.workers_var = IntVar(value=1)
        self.start_var = StringVar(value="")
        self.end_var = StringVar(value="")

        self.status_var = StringVar(value="就绪")
        self.result_var = StringVar(value="")
        self.progress_text_var = StringVar(value="")

        # 模板池 Listbox（懒创建于 _build）：池内每个 .npz 模板各出一个动作分。
        self.template_listbox: Listbox | None = None

        # 上次结果缓存（综合分 / 匹配信息）。
        self._last_score: float | None = None
        self._last_match_info: str = ""

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None

        self._build()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)

        # ===== Step 1: 模板池（池内每个 .npz 各出一个动作分，动作名 = 文件 stem）=====
        step1 = ttk.Labelframe(outer, text="① 模板池（每个模板对应一个动作，各出一个成绩）", padding=10)
        step1.pack(fill="x")

        self._template_body = ttk.Frame(step1)
        self._template_body.pack(fill="x")

        # 模板池 Listbox + 添加/移除按钮。
        pool_row = ttk.Frame(self._template_body)
        pool_row.pack(fill="x")
        lb_box = ttk.Frame(pool_row)
        lb_box.pack(side="left", fill="both", expand=True)
        self.template_listbox = Listbox(lb_box, height=5, selectmode="extended", exportselection=False)
        self.template_listbox.pack(side="left", fill="both", expand=True)
        pool_sb = Scrollbar(lb_box, command=self.template_listbox.yview)
        pool_sb.pack(side="right", fill="y")
        self.template_listbox.configure(yscrollcommand=pool_sb.set)

        pool_btns = ttk.Frame(pool_row)
        pool_btns.pack(side="left", fill="y", padx=(8, 0))
        ttk.Button(pool_btns, text="添加模板…", command=self._add_templates).pack(fill="x")
        ttk.Button(pool_btns, text="移除选中", command=self._remove_selected_templates).pack(fill="x", pady=(6, 0))
        ttk.Button(pool_btns, text="清空", command=self._clear_templates).pack(fill="x", pady=(6, 0))

        # 从视频生成模板（生成后自动加入池）。
        gen_wrap = CollapsibleSection(self._template_body, "从视频生成模板", expanded=False)
        gen_wrap.set_title("从视频生成模板")
        gen_wrap.pack_header(fill="x", pady=(10, 0))
        self._generate_frame = gen_wrap.content

        gen_row1 = ttk.Frame(self._generate_frame)
        gen_row1.pack(fill="x", pady=(4, 0))
        ttk.Label(gen_row1, text="基准视频：").grid(row=0, column=0, sticky="w")
        ttk.Entry(gen_row1, textvariable=self.base_video_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(gen_row1, text="选择…", command=self._browse_base).grid(row=0, column=2, padx=(8, 0))
        gen_row1.columnconfigure(1, weight=1)

        adv_row2 = ttk.Frame(self._generate_frame)
        adv_row2.pack(fill="x", pady=(6, 0))
        ttk.Label(adv_row2, text="起始帧：").pack(side="left")
        ttk.Entry(adv_row2, textvariable=self.start_var, width=8).pack(side="left", padx=(6, 14))
        ttk.Label(adv_row2, text="结束帧：").pack(side="left")
        ttk.Entry(adv_row2, textvariable=self.end_var, width=8).pack(side="left", padx=(6, 0))

        self._gen_btn = ttk.Button(self._generate_frame, text="生成并加入池", command=self._gen_template)
        self._gen_btn.pack(fill="x", pady=(10, 0))

        # ===== Step 2: Target Video =====
        step2 = ttk.Labelframe(outer, text="② 选择目标视频", padding=10)
        step2.pack(fill="x", pady=(12, 0))

        tgt_row = ttk.Frame(step2)
        tgt_row.pack(fill="x")
        ttk.Label(tgt_row, text="目标视频：").grid(row=0, column=0, sticky="w")
        ttk.Entry(tgt_row, textvariable=self.target_video_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(tgt_row, text="选择…", command=self._browse_target).grid(row=0, column=2, padx=(8, 0))
        tgt_row.columnconfigure(1, weight=1)
        ttk.Label(tgt_row, text="线程数：").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(tgt_row, from_=1, to=16, textvariable=self.workers_var, width=6).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        # ===== Step 3: Execute =====
        step3 = ttk.Labelframe(outer, text="③ 执行分析", padding=10)
        step3.pack(fill="x", pady=(12, 0))

        btn_row = ttk.Frame(step3)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, text="═══ 开始分析 ═══", command=self._start_compare)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))

        # Progress bar (hidden initially)
        self._progress_frame = ttk.Frame(step3)
        self._progress_frame.pack(fill="x", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(self._progress_frame, orient="horizontal", mode="determinate", maximum=100.0)
        self.progress_bar.pack(fill="x")
        ttk.Label(self._progress_frame, textvariable=self.progress_text_var).pack(anchor="w", pady=(4, 0))
        self._progress_frame.pack_forget()  # Hide initially

        # ===== Results =====
        result_frame = ttk.Labelframe(outer, text="结果", padding=10)
        result_frame.pack(fill="both", expand=True, pady=(12, 0))

        # Status line
        ttk.Label(result_frame, textvariable=self.status_var).pack(anchor="w")

        # 综合分（各动作分平均）大号显示。
        self._score_frame = ttk.Frame(result_frame)
        self._score_frame.pack(fill="x", pady=(8, 0))
        self._score_label = ttk.Label(
            self._score_frame, text="—", font=("Helvetica", 36, "bold"), anchor="center"
        )
        self._score_label.pack()
        self._score_hint = ttk.Label(self._score_frame, text="综合分（各动作平均）", anchor="center")
        self._score_hint.pack()

        # 逐动作成绩列表：每个模板一行「动作名 …… 百分比」，颜色随分数。
        self._action_frame = ttk.Labelframe(result_frame, text="各动作成绩", padding=8)
        self._action_frame.pack(fill="x", pady=(8, 0))
        self._action_rows: list[tuple[ttk.Label, ttk.Label]] = []

        # Match info
        self._match_label = ttk.Label(result_frame, textvariable=self.result_var, wraplength=600)
        self._match_label.pack(anchor="w", pady=(8, 0))

        # Collapsible JSON section
        self._json_section = CollapsibleSection(result_frame, "查看详细数据", expanded=False)
        self._json_section.set_title("查看详细数据")
        self._json_section.pack_header(fill="x", pady=(10, 0))

        json_content = self._json_section.content
        raw_box = ttk.Frame(json_content)
        raw_box.pack(fill="both", expand=True, pady=(4, 0))
        self.raw_text = Text(raw_box, height=10, wrap="none")
        self.raw_text.pack(side="left", fill="both", expand=True)
        sb = Scrollbar(raw_box, command=self.raw_text.yview)
        sb.pack(side="right", fill="y")
        self.raw_text.configure(yscrollcommand=sb.set)
        ttk.Button(json_content, text="复制数据", command=self._copy_raw).pack(anchor="e", pady=(6, 0))

        # Initialize display state
        self._reset_score_display()

    @staticmethod
    def _score_color(score: float) -> str:
        """分数 → 红黄绿。≥0.8 绿 / ≥0.5 黄 / 否则红。"""
        if score >= 0.8:
            return "#2e7d32"
        if score >= 0.5:
            return "#f9a825"
        return "#c62828"

    def _reset_score_display(self) -> None:
        """Reset score display to initial state."""
        self._last_score = None
        self._last_match_info = ""
        self._score_label.configure(text="—", foreground="")
        self.result_var.set("")
        self._clear_action_rows()

    def _clear_action_rows(self) -> None:
        for name_lbl, score_lbl in self._action_rows:
            name_lbl.destroy()
            score_lbl.destroy()
        self._action_rows = []

    def _update_score_display(self, actions: list[tuple[str, float]], match_info: str = "") -> None:
        """刷新综合分 + 逐动作成绩列表。actions = [(动作名, 0..1 分), ...]。"""
        self._last_match_info = match_info
        self._clear_action_rows()

        if not actions:
            self._score_label.configure(text="—", foreground="")
            self.result_var.set(match_info)
            return

        # 综合分 = 各动作分平均。
        combined = sum(s for _, s in actions) / len(actions)
        self._last_score = combined
        self._score_label.configure(
            text=f"{int(combined * 100)}%", foreground=self._score_color(combined)
        )

        # 逐动作一行。
        for name, score in actions:
            row = ttk.Frame(self._action_frame)
            row.pack(fill="x", pady=1)
            name_lbl = ttk.Label(row, text=name, anchor="w")
            name_lbl.pack(side="left")
            score_lbl = ttk.Label(
                row, text=f"{int(score * 100)}%", anchor="e", foreground=self._score_color(score)
            )
            score_lbl.pack(side="right")
            self._action_rows.append((name_lbl, score_lbl))

        self.result_var.set(match_info)

    def _show_progress(self, show: bool = True) -> None:
        """Show or hide the progress frame."""
        if show:
            self._progress_frame.pack(fill="x", pady=(8, 0))
        else:
            self._progress_frame.pack_forget()

    def _browse_base(self) -> None:
        p = filedialog.askopenfilename(
            title="选择基准视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.base_video_var.set(p)

    def _browse_target(self) -> None:
        p = filedialog.askopenfilename(
            title="选择目标视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        if p:
            self.target_video_var.set(p)

    def _pool_paths(self) -> list[str]:
        """当前模板池中的所有路径（去空）。"""
        if self.template_listbox is None:
            return []
        return [s for s in self.template_listbox.get(0, "end") if str(s).strip()]

    def _add_template_path(self, path: str) -> None:
        """校验后加入池（heavy + pose33_v3），去重。返回值仅用于内部判断由调用方处理。"""
        if self.template_listbox is None:
            return
        usable = try_load_heavy_template(path)
        if usable is None:
            messagebox.showerror(
                "模板无效",
                f"{Path(path).name} 不是可用的 heavy + pose33_v3 模板，已跳过。\n"
                "请用 make_template.py --pose heavy 生成后再添加。",
                parent=self._win,
            )
            return
        if str(usable) in {str(p) for p in self._pool_paths()}:
            return
        self.template_listbox.insert("end", str(usable))

    def _add_templates(self) -> None:
        paths = filedialog.askopenfilenames(
            title="添加模板（可多选）",
            filetypes=[("模板文件", "*.npz"), ("所有文件", "*.*")],
            parent=self._win,
        )
        self._win.lift()
        self._win.focus_force()
        for p in paths:
            self._add_template_path(p)

    def _remove_selected_templates(self) -> None:
        if self.template_listbox is None:
            return
        for i in reversed(self.template_listbox.curselection()):
            self.template_listbox.delete(i)

    def _clear_templates(self) -> None:
        if self.template_listbox is not None:
            self.template_listbox.delete(0, "end")

    def _parse_int_or_none(self, s: str) -> int | None:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"请输入整数帧号：{s!r}")

    def _progress(self, stage: str, done: int, total: int) -> None:
        def _set() -> None:
            self._show_progress(True)
            if total > 0:
                self.progress_bar.configure(mode="determinate", maximum=float(total))
                self.progress_bar["value"] = float(done)
                pct = (done / total) * 100.0
                self.progress_text_var.set(f"{stage}：{done}/{total}（{pct:.1f}%）")
            else:
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set(f"{stage}…")

        self._win.after(0, _set)

    def _set_status(self, text: str) -> None:
        self._win.after(0, lambda: self.status_var.set(text))

    def _set_result(self, text: str) -> None:
        self._win.after(0, lambda: self.result_var.set(text))

    def _set_raw(self, text: str) -> None:
        def _set() -> None:
            self.raw_text.delete("1.0", "end")
            self.raw_text.insert("1.0", text)

        self._win.after(0, _set)

    def _copy_raw(self) -> None:
        try:
            s = self.raw_text.get("1.0", "end").strip()
            self._win.clipboard_clear()
            self._win.clipboard_append(s)
            self._set_status("已复制原始数据到剪贴板")
        except Exception as e:
            messagebox.showerror("复制失败", str(e), parent=self._win)

    def _set_buttons(self, running: bool) -> None:
        def _set() -> None:
            self.start_btn.configure(state="disabled" if running else "normal")
            self.stop_btn.configure(state="normal" if running else "disabled")
            if not running:
                self._show_progress(False)

        self._win.after(0, _set)

    def _gen_template(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        base = self.base_video_var.get().strip()
        if not base:
            messagebox.showerror("配置错误", "请先选择基准视频。", parent=self._win)
            return

        try:
            start = self._parse_int_or_none(self.start_var.get())
            end = self._parse_int_or_none(self.end_var.get())
        except Exception as e:
            messagebox.showerror("配置错误", str(e), parent=self._win)
            return

        self._stop_evt.clear()
        self._set_buttons(True)
        self._set_status("正在生成模板…")
        self._reset_score_display()
        self._set_raw("")
        self._progress("准备中", 0, 0)

        def _run() -> None:
            try:
                # 模板生成属后端黑盒链路，固定 heavy。
                tpl_path = create_template_from_video(
                    base,
                    pose_variant="heavy",
                    start=start,
                    end=end,
                    workers=int(self.workers_var.get() or 1),
                    preview=False,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                self._win.after(0, lambda p=str(tpl_path): self._add_template_path(p))
                template_meta = load_template(tpl_path, COMPARE).raw_meta
                start_frame = int(template_meta.get("start_frame") or 0)
                end_frame = int(template_meta.get("end_frame") or start_frame)
                fps = float(template_meta.get("fps") or 30.0)
                kept_frames = max(1, end_frame - start_frame + 1)
                retained = float(template_meta.get("selected_retained_ratio") or 0.0)
                summary = (
                    f"模板：{tpl_path}\n"
                    f"区间：{start_frame}..{end_frame}（{kept_frames / fps:.2f}s，"
                    f"保留 {retained * 100.0:.1f}%）"
                )
                self._set_status("模板生成完成")
                self._set_result(summary)
                payload = {
                    "template_path": str(tpl_path),
                    "template_meta": template_meta,
                }
                self._set_raw(json.dumps(payload, ensure_ascii=False, indent=2))
            except Exception as e:
                self._set_status("模板生成失败")
                self._set_result(str(e))
                self._set_raw("")
            finally:
                self._set_buttons(False)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _start_compare(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        tpls = self._pool_paths()
        vid = self.target_video_var.get().strip()
        if not vid:
            messagebox.showerror("配置错误", "请先选择目标视频。", parent=self._win)
            return
        if not tpls:
            messagebox.showerror(
                "配置错误", "模板池为空，请先添加模板（.npz），或从基准视频生成。", parent=self._win
            )
            return

        # 后端黑盒分析一律 heavy，与主界面预览档位无关。
        pose_variant = "heavy"
        self.pose_var.set("heavy")
        workers = int(self.workers_var.get() or 1)

        self._stop_evt.clear()
        self._set_buttons(True)
        self._set_status("正在分析…")
        self._reset_score_display()
        self._set_raw("")
        self._progress("准备中", 0, 0)

        def _run() -> None:
            payload: dict = {"video_path": vid}
            try:
                results = compare_templates(
                    tpls,
                    vid,
                    pose_variant=pose_variant,
                    workers=workers,
                    progress_cb=self._progress,
                    stop_evt=self._stop_evt,
                )
                # 动作名 = 模板文件 stem；每个模板一个动作分。
                actions = [
                    (template_action_name(res.template_path), float(res.score))
                    for res in results
                ]
                payload["actions"] = [
                    {
                        "name": template_action_name(res.template_path),
                        "score": res.score,
                        "avg_cost": res.avg_cost,
                        "cost": res.cost,
                        "start_frame": res.start_frame,
                        "end_frame": res.end_frame,
                        "fps": res.fps,
                        "pose_variant": res.pose_variant,
                        "workers_used": res.workers_used,
                        "template_path": str(res.template_path),
                    }
                    for res in results
                ]
                if actions:
                    combined = sum(s for _, s in actions) / len(actions)
                    payload["combined_score"] = combined
                    payload["combined_percent"] = int(round(combined * 100))

                match_info = f"共 {len(actions)} 个动作已评分"
                downgraded = any(
                    r.workers_used and r.workers_used != workers for r in results
                )
                if downgraded:
                    match_info += "\n说明：模板为 VIDEO 模式，已自动使用单线程以保证准确。"

                self._win.after(0, lambda: self._update_score_display(actions, match_info))
                self._set_raw(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))
                self._set_status("分析完成")
            except Exception as e:
                self._set_status("分析失败")
                self._set_result(str(e))
                self._set_raw("")
            finally:
                self._set_buttons(False)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _stop(self) -> None:
        self._stop_evt.set()
        self._set_status("正在停止…")

    def _on_close(self) -> None:
        self._stop_evt.set()
        self._win.destroy()


@dataclass(frozen=True)
class UiState:
    source: str
    pose_variant: str
    workers: int
    enable_hands: bool
    out_path: str | None = None
    source2: str | None = None
    online_match_enabled: bool = True
    rotate: int = 0
    rotate2: int = 0
    record_skeleton: bool = False
    # 双摄录制结束后是否自动进入黑盒比对（检测一条龙）；False 为仅录制。
    auto_compare: bool = True
    # 会话启动时冻结的请求尺寸；打开途中切换模式不得改变本次采集配置。
    capture_size: tuple[int, int] = _PREVIEW_CAPTURE_SIZE
    # 驱动给不出请求尺寸时的处理（D-6）与按编号记下的设备名（V-3），同样在启动时冻结。
    capture_policy: str = "prefer"
    camera_names: tuple[tuple[int, str], ...] = ()


class SettingsWindow:
    """设置窗口：模型管理（查看当前模型 / 下载缺失模型）。

    本期仅提供 MediaPipe 模型下载。下载源为 Google 官方直链；若用户网络无法访问，
    会在状态区提示自行解决网络问题（代理 / 加速等）。
    """

    def __init__(self, parent: Tk, *, current_pose: str = "full", hands_enabled: bool = True) -> None:
        self._win = Toplevel(parent)
        self._win.title("设置")
        self._win.geometry("560x560")
        self._win.minsize(520, 500)

        # 主界面当前选择：用于在模型列表中标注「当前使用」并在顶部汇总展示。
        self._current_pose = (current_pose or "full").strip().lower()
        self._hands_enabled = bool(hands_enabled)
        # 当前实际会用到的模型 key 集合（pose 选档 + 可选 hand）。
        self._active_keys = {
            {"lite": "pose_lite", "full": "pose_full", "heavy": "pose_heavy"}.get(
                self._current_pose, "pose_full"
            )
        }
        if self._hands_enabled:
            self._active_keys.add("hand")

        self.status_var = StringVar(value="就绪")
        self.progress_text_var = StringVar(value="")
        self.current_model_var = StringVar(value="")

        # 每个模型一行的控件引用：key -> dict(state_var, btn)
        self._rows: dict[str, dict] = {}

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None

        self._build()
        self._refresh_states()
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def is_open(self) -> bool:
        return bool(self._win.winfo_exists())

    def focus(self) -> None:
        try:
            self._win.deiconify()
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)

        # ===== 当前模型 =====
        current_frame = ttk.Labelframe(outer, text="当前模型", padding=10)
        current_frame.pack(fill="x")
        ttk.Label(
            current_frame,
            textvariable=self.current_model_var,
            wraplength=510,
        ).pack(anchor="w")

        # ===== 模型管理 =====
        mp_frame = ttk.Labelframe(outer, text="MediaPipe 模型", padding=10)
        mp_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(
            mp_frame,
            text="模型目录：" + str(models_dir()),
            wraplength=500,
        ).pack(anchor="w", pady=(0, 8))

        for spec in model_manager.MEDIAPIPE_MODELS:
            row = ttk.Frame(mp_frame)
            row.pack(fill="x", pady=(4, 0))

            name = ttk.Label(row, text=spec.label, width=34, anchor="w")
            name.grid(row=0, column=0, sticky="w")

            state_var = StringVar(value="检查中…")
            state_lbl = ttk.Label(row, textvariable=state_var, width=14, anchor="w")
            state_lbl.grid(row=0, column=1, sticky="w", padx=(8, 0))

            btn = ttk.Button(
                row, text="下载", width=8,
                command=lambda s=spec: self._download(s),
            )
            btn.grid(row=0, column=2, sticky="e", padx=(8, 0))
            row.columnconfigure(0, weight=1)

            self._rows[spec.key] = {"spec": spec, "state_var": state_var, "btn": btn}

        # 一键下载缺失模型
        bulk_row = ttk.Frame(mp_frame)
        bulk_row.pack(fill="x", pady=(12, 0))
        self.download_all_btn = ttk.Button(
            bulk_row, text="下载全部缺失模型", command=self._download_missing
        )
        self.download_all_btn.pack(side="left")
        self.refresh_btn = ttk.Button(bulk_row, text="刷新状态", command=self._refresh_states)
        self.refresh_btn.pack(side="left", padx=(8, 0))

        # ===== 进度 / 状态 =====
        status_frame = ttk.Labelframe(outer, text="状态", padding=10)
        status_frame.pack(fill="both", expand=True, pady=(12, 0))

        ttk.Label(status_frame, textvariable=self.status_var, wraplength=510).pack(anchor="w")
        self.progress_bar = ttk.Progressbar(
            status_frame, orient="horizontal", mode="determinate", maximum=100.0
        )
        self.progress_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.progress_text_var, wraplength=510).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Label(
            status_frame,
            text="说明：模型从 Google 官方源下载。若长时间无进度或失败，通常是网络无法访问"
            " storage.googleapis.com，请自行配置代理后重试。",
            wraplength=510,
            foreground="#666",
        ).pack(anchor="w", pady=(10, 0))

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _refresh_states(self) -> None:
        """刷新每个模型的「已安装 / 未安装」状态显示，并标注当前使用的模型。"""
        for key, info in self._rows.items():
            spec = info["spec"]
            active = key in self._active_keys
            mark = "● 使用中  " if active else ""
            if model_manager.is_installed(spec):
                size = model_manager.installed_size_mb(spec)
                info["state_var"].set(mark + (f"已安装 ({size} MB)" if size else "已安装"))
                info["btn"].configure(text="重新下载")
            else:
                approx = f"约 {spec.approx_mb} MB" if spec.approx_mb else ""
                info["state_var"].set((mark + f"未安装 {approx}").strip())
                info["btn"].configure(text="下载")

        self._refresh_current_summary()

    def _refresh_current_summary(self) -> None:
        """汇总当前主界面所选模型及其就绪情况，写入「当前模型」区。"""
        parts: list[str] = []
        missing: list[str] = []
        for key in ("pose_lite", "pose_full", "pose_heavy", "hand"):
            if key not in self._active_keys:
                continue
            info = self._rows.get(key)
            if info is None:
                continue
            spec = info["spec"]
            ready = model_manager.is_installed(spec)
            parts.append(f"{spec.label}（{'已就绪' if ready else '缺失'}）")
            if not ready:
                missing.append(spec.label)
        hands_txt = "开启" if self._hands_enabled else "关闭"
        summary = f"姿态模型：{self._current_pose}　手部检测：{hands_txt}\n" + "\n".join(parts)
        if missing:
            summary += "\n\n⚠ 缺失模型会导致无法开始识别，请在下方下载后再使用。"
        self.current_model_var.set(summary)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for info in self._rows.values():
            info["btn"].configure(state=state)
        self.download_all_btn.configure(state=state)
        self.refresh_btn.configure(state=state)

    def _set_status(self, text: str) -> None:
        self._win.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, downloaded: int, total: int | None) -> None:
        def _set() -> None:
            if total and total > 0:
                pct = (downloaded / total) * 100.0
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = pct
                self.progress_text_var.set(
                    f"{downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB（{pct:.0f}%）"
                )
            else:
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set(f"{downloaded / 1024 / 1024:.1f} MB…")

        self._win.after(0, _set)

    def _download(self, spec: "model_manager.ModelSpec") -> None:
        self._start_download([spec])

    def _download_missing(self) -> None:
        missing = [info["spec"] for info in self._rows.values()
                   if not model_manager.is_installed(info["spec"])]
        if not missing:
            messagebox.showinfo("模型管理", "所有模型均已安装。", parent=self._win)
            return
        self._start_download(missing)

    def _start_download(self, specs: list) -> None:
        if self._busy():
            return
        self._stop_evt.clear()
        self._set_buttons_enabled(False)

        def _run() -> None:
            ok, failed = 0, []
            try:
                for spec in specs:
                    if self._stop_evt.is_set():
                        break
                    self._set_status(f"正在下载：{spec.label} …")
                    try:
                        model_manager.download_model(
                            spec,
                            progress_cb=self._set_progress,
                            should_stop=self._stop_evt.is_set,
                        )
                        ok += 1
                        self._win.after(0, self._refresh_states)
                    except InterruptedError:
                        self._set_status("下载已取消。")
                        return
                    except Exception as e:  # noqa: BLE001
                        failed.append((spec.label, str(e)))
                if failed:
                    detail = "\n".join(f"- {name}: {err}" for name, err in failed)
                    self._set_status(f"完成 {ok} 个，{len(failed)} 个失败。")
                    self._win.after(
                        0,
                        lambda: messagebox.showerror(
                            "下载失败",
                            "以下模型下载失败（多为网络无法访问 Google 源，请配置代理后重试）：\n\n"
                            + detail,
                            parent=self._win,
                        ),
                    )
                else:
                    self._set_status(f"全部完成，共下载 {ok} 个模型。")
            finally:
                self._win.after(0, self._refresh_states)
                self._win.after(0, lambda: self._set_buttons_enabled(True))

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def _on_close(self) -> None:
        self._stop_evt.set()
        self._win.destroy()


class App:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("MediaPipe 动作识别（人体姿态 + 手部）")
        self.root.geometry("1100x720")
        # 控制区改为可滚动容器后，限制窗口最小尺寸，保证滚动条与预览区可用（需求 1.5/1.6）。
        self.root.minsize(800, 600)

        (
            self._preferred_primary_camera_index,
            self._preferred_secondary_camera_index,
        ) = load_camera_selection()
        self.source_var = StringVar(value="")
        self.camera_choice_var = StringVar(value="")
        self.camera_choice_var_2 = StringVar(value=NO_SECOND_CAMERA)
        self.rotate_var = StringVar(value=ROTATE_CHOICES[0])
        self.rotate_var_2 = StringVar(value=ROTATE_CHOICES[0])
        # 会话运行中可调的参数（每路旋转、站位检测）：主线程写，会话线程每帧读（§8.1）。
        self._live_controls = LiveControls()
        self.source_hint_var = StringVar(value="当前输入源：未选择")
        self.pose_var = StringVar(value="full")
        self.workers_var = IntVar(value=default_workers())
        self.enable_hands_var = BooleanVar(value=True)
        # 全局骨架开关（单/双摄共用）：默认关 → 预览与录制均不叠骨架、不跑姿态推理。
        self.record_skeleton_var = BooleanVar(value=False)
        # 双摄默认「录制+检测一条龙」；取消勾选后仅录制落盘/转码，不自动比对。
        # 运行中可通过滑块快速切换；片段开录/提交时读取当前值。
        self.auto_compare_var = BooleanVar(value=True)
        self.auto_compare_scale_var = DoubleVar(value=1.0)
        # 录制视频保存目录（默认上次持久化的目录，无记录时回退 outputs_dir()）。
        # 录制文件名仍由控制器按时间戳生成。
        _initial_record_dir = load_record_dir()
        self.record_dir_var = StringVar(value=str(_initial_record_dir))
        # worker 的 path_provider 不得读取 Tk StringVar；主线程在初始化、选目录和开录时
        # 把值缓存为普通 Path，避免录制锁与 Tk 主线程形成互等。
        self._record_base_dir = _initial_record_dir
        self.status_var = StringVar(value="就绪")
        self.actions_var = StringVar(value="-")
        self.match_var = StringVar(value="识别：待机")
        self.online_match_var = BooleanVar(value=True)
        self.progress_var = DoubleVar(value=0.0)
        self.progress_text_var = StringVar(value="")
        # 录制状态文本与 Result_Video 完整路径（需求 5.9/5.10）。对应的 Label 控件
        # 由任务 9.3 在 _build_ui 中加入，此处先维护变量。
        self.recording_status_var = StringVar(value="")
        self.compare_segment_var = StringVar(value="片段：-")
        self.compare_status_var = StringVar(value="自动比对：待机")
        self.compare_score_var = StringVar(value="正面：-　侧面：-　综合：-")
        self.compare_error_var = StringVar(value="")
        # 守卫标志：避免每 30ms 重复弹出同一录制错误对话框（需求 5.11）。
        self._record_error_shown = False

        # 输入源状态模型（camera / video / none 三态互斥）。
        self._source_state = InputSourceState()
        self._camera_entries: list[CameraEntry] = []
        self._enum_busy = threading.Event()
        self._camera_enum_result_queue: Queue[tuple[list[CameraEntry], bool]] = Queue(
            maxsize=1
        )

        # 摄像头唯一入口：单摄/双摄预热、会话接管和释放隔离都经 CameraService 的池。
        # 采集配置（编号+请求尺寸）在请求时冻结为 CaptureSpec，不在打开途中读模式标志。
        self._camera_service = CameraService(
            capture_factory=self._open_camera_process,
            join_timeout=_CAMERA_CANCEL_JOIN_TIMEOUT_S,
        )

        self._closing = False
        self._close_deadline: float | None = None
        self._close_prepare_worker: threading.Thread | None = None
        self._latest_compare_segment_id: str | None = None
        self._compare_update_queue: Queue[SegmentUpdate] = Queue()
        # 录制与录后处理：结束片段并提交处理计划在 RecordingService 的同一把锁内完成；
        # 计划由 _recording_plan 按模式构建（P7 起由工作流构建）。
        self._recording = RecordingService(
            SegmentFinalizer(on_update=self._post_recording_compare_update),
            plan_for=self._recording_plan,
        )
        # 运行模式（普通/考试/学生练习，§8.5）与工作流事件路由（§8.6）。考试面板开考时进入考试模式。
        self._exam_manual_locked = False
        self.modes = ModeManager()
        # 各场考试的成绩台账：按 exam_run_id 回填录后结果，考试面板关闭后仍可回填
        self.exam_scorebooks = ScorebookRegistry()
        self.workflows = WorkflowHost(self.exam_scorebooks)
        self.exam_panel = None
        # 学生练习（MediaPipe 关键点规则检查）：进入练习模式时创建流程状态机与播报器
        self._practice: PracticeSession | None = None
        self._student_announcer: Announcer | None = None

        # 实时会话：唯一逐帧循环在 core.live 的会话线程里；事件经 _live_events 回到 Tk 线程（_tick 排空）。
        self._live_events: Queue[SessionEvent] = Queue()
        self._live = App._build_live_manager(self)
        self._shown_preview_seq = 0
        self._dual_recording_ready = False
        self._photo: ImageTk.PhotoImage | None = None
        self._photo2: ImageTk.PhotoImage | None = None
        # 预览区当前尺寸（主线程 Configure 回调写、会话线程按它缩放预览帧；tuple 读写原子）。
        self._preview_wh: tuple[int, int] = (0, 0)
        self._preview_wh2: tuple[int, int] = (0, 0)
        self._compare_win: CompareWindow | None = None
        self._settings_win: SettingsWindow | None = None
        self._settings_win: SettingsWindow | None = None

        # worker 关闭会话前转存尚未被 Tk tick 消费的单摄 writer 错误。
        self._pending_record_errors: list[str] = []
        self._dual_active = False
        self._dual_record_skeleton = False
        self._dual_auto_compare = True

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()
        # 启动时后台枚举可用摄像头（需求 1.1）。
        self._start_enumeration()

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Left: scrollable controls container（可滚动 Canvas + 垂直 Scrollbar + 内嵌 inner Frame，需求 1.5/1.6）。
        # outer 列 0 固定宽度容器；列 1 预览区可伸展。
        left_container = ttk.Frame(outer)
        self.left_container = left_container
        left_container.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        left_container.rowconfigure(0, weight=1)

        self.left_canvas = Canvas(left_container, width=360, highlightthickness=0, borderwidth=0)
        self.left_canvas.grid(row=0, column=0, sticky="ns")
        left_scroll = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        left_scroll.grid(row=0, column=1, sticky="ns")
        self.left_canvas.configure(yscrollcommand=left_scroll.set)

        # 内嵌 inner Frame 作为所有控制卡片的父容器。
        self.controls_inner = ttk.Frame(self.left_canvas)
        self._controls_window = self.left_canvas.create_window(
            (0, 0), window=self.controls_inner, anchor="nw"
        )

        # inner 内容尺寸变化时更新 scrollregion；同时让 inner 宽度跟随 canvas 宽度。
        def _on_inner_configure(event: object) -> None:
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        self.controls_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event: "Event") -> None:
            self.left_canvas.itemconfigure(self._controls_window, width=event.width)

        self.left_canvas.bind("<Configure>", _on_canvas_configure)

        # 鼠标滚轮滚动（Windows: <MouseWheel> + event.delta）。指针进入控制区时绑定，离开时解绑。
        def _on_mousewheel(event: "Event") -> None:
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event: object) -> None:
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event: object) -> None:
            self.left_canvas.unbind_all("<MouseWheel>")

        self.left_canvas.bind("<Enter>", _bind_mousewheel)
        self.left_canvas.bind("<Leave>", _unbind_mousewheel)
        self.controls_inner.bind("<Enter>", _bind_mousewheel)
        self.controls_inner.bind("<Leave>", _unbind_mousewheel)

        # 所有左侧控制卡片改为放入可滚动的 inner Frame。
        left = self.controls_inner

        # ===== Primary_Controls 分组（核心五项，固定顺序，需求 1.1/1.3）=====
        # 自上而下恰好五项：Camera_Selector → Model_Selector → Start_Control
        # → Record_Toggle → Compare_Control，相邻控件之间不插入任何 Secondary_Option。
        primary = ttk.Labelframe(left, text="主要操作", padding=10)
        primary.pack(fill="x", pady=(0, 10))

        # 0) 顶部工具行：右上角「设置」按钮（模型管理等）。
        top_row = ttk.Frame(primary)
        top_row.pack(fill="x")
        self.settings_btn = ttk.Button(top_row, text="⚙ 设置", width=8, command=self._open_settings)
        self.settings_btn.pack(side="right")

        # 1) Camera_Selector：摄像头下拉 + 刷新按钮（刷新为摄像头选择的附属操作，允许同行）。
        ttk.Label(primary, text="第一摄像头（正面）：").pack(anchor="w")
        cam_row = ttk.Frame(primary)
        cam_row.pack(fill="x", pady=(6, 0))
        self.camera_combo = ttk.Combobox(
            cam_row, textvariable=self.camera_choice_var, state="readonly"
        )
        self.camera_combo.pack(side="left", fill="x", expand=True)
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)
        self.refresh_btn = ttk.Button(cam_row, text="刷新", command=self._refresh_cameras)
        self.refresh_btn.pack(side="left", padx=(8, 0))
        # 画面旋转：不再用下拉；运行中点击对应预览窗顺时针 +90°（两路隔离）。

        # 1b) 第二摄像头（可选，双摄双面视图预览，issue #57）：默认「无」，不影响单摄路径。
        ttk.Label(primary, text="第二摄像头（侧面，可选）：").pack(anchor="w", pady=(10, 0))
        cam_row_2 = ttk.Frame(primary)
        cam_row_2.pack(fill="x", pady=(6, 0))
        self.camera_combo_2 = ttk.Combobox(
            cam_row_2,
            textvariable=self.camera_choice_var_2,
            values=[NO_SECOND_CAMERA],
            state="readonly",
        )
        self.camera_combo_2.pack(side="left", fill="x", expand=True)
        self.camera_combo_2.bind("<<ComboboxSelected>>", self._on_camera_2_selected)

        # 教师主操作块（学生练习模式整体 pack_forget）
        self._teacher_block = ttk.Frame(primary)
        self._teacher_block.pack(fill="x")

        # 2) Model_Selector：仅驱动实时预览/骨架推理；黑盒评分固定 heavy。
        ttk.Label(self._teacher_block, text="预览姿态模型：").pack(anchor="w", pady=(10, 0))
        self.model_combo = ttk.Combobox(
            self._teacher_block,
            textvariable=self.pose_var,
            values=["lite", "full", "heavy"],
            state="readonly",
        )
        self.model_combo.pack(fill="x", pady=(6, 0))
        ttk.Label(
            self._teacher_block,
            text="仅影响实时预览；录后黑盒评分固定 heavy",
            foreground="#555555",
        ).pack(anchor="w", pady=(2, 0))

        # 3) Start_Control：开始/停止（保留分离的 start/stop 双按钮与既有 _start/_stop 接线）。
        start_row = ttk.Frame(self._teacher_block)
        start_row.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(start_row, text="开始", command=self._start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(start_row, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # 4) Record_Controls：录制相关操作独立成组，与会话级 Start_Control（开始/停止）
        #    在视觉上区分开。组内含两个动作（需求 5.x）：
        #      - record_btn（三态切换）：开始录制 / 暂停录制 / 继续录制。
        #      - record_stop_btn（结束录制）：结束当前录制片段并落盘，但不结束识别会话，
        #        随后可在同一会话内再次「开始录制」生成新文件。
        #    会话未运行时整组禁用（需求 5.1）。
        record_group = ttk.Labelframe(self._teacher_block, text="录制", padding=8)
        record_group.pack(fill="x", pady=(10, 0))
        self.record_btn = ttk.Button(
            record_group, text=RECORD_BTN_TEXT["idle"], command=self._on_record_toggle, state="disabled"
        )
        self.record_btn.pack(fill="x")
        self.record_stop_btn = ttk.Button(
            record_group, text="结束录制", command=self._on_record_stop, state="disabled"
        )
        self.record_stop_btn.pack(fill="x", pady=(6, 0))
        # 自动比对滑块：运行中可快速切换；片段开录/提交时读取。
        auto_row = ttk.Frame(record_group)
        auto_row.pack(fill="x", pady=(8, 0))
        ttk.Label(auto_row, text="仅录制").pack(side="left")
        self.auto_compare_scale = ttk.Scale(
            auto_row,
            from_=0,
            to=1,
            orient="horizontal",
            length=96,
            variable=self.auto_compare_scale_var,
            command=self._on_auto_compare_scale,
        )
        self.auto_compare_scale.pack(side="left", padx=(6, 6), fill="x", expand=True)
        # 兼容既有控件名（考试锁/运行态联动仍引用 auto_compare_check）。
        self.auto_compare_check = self.auto_compare_scale
        ttk.Label(auto_row, text="自动比对").pack(side="left")
        self.auto_compare_hint_var = StringVar(value="录制后自动比对（检测一条龙）")
        ttk.Label(record_group, textvariable=self.auto_compare_hint_var).pack(
            anchor="w", pady=(2, 0)
        )
        # 一条龙/考试标准模板：正/侧各一个多模板池（正对正、侧对侧，每边可多个动作模板）。
        # 至少一侧非空即可跑比对。旧单值 StringVar 保留（兼容既有引用），运行时以 Listbox 为准。
        _front_list, _side_list = load_template_pools()
        self.front_template_var = StringVar(
            value=str(_front_list[0]) if _front_list else ""
        )
        self.side_template_var = StringVar(
            value=str(_side_list[0]) if _side_list else ""
        )
        self.template_status_var = StringVar(value="")
        self._build_template_pool_ui(record_group, _front_list, _side_list)
        self._sync_template_status()
        self.record_skeleton_check = ttk.Checkbutton(
            record_group,
            text="开启骨架（预览与录制；双摄开启后不自动比对）",
            variable=self.record_skeleton_var,
        )
        self.record_skeleton_check.pack(anchor="w", pady=(4, 0))
        self._sync_auto_compare_hint()
        try:
            self.auto_compare_var.trace_add("write", self._on_auto_compare_var_write)
        except Exception:
            pass

        # 录制视频保存目录选择行：默认 outputs_dir()，可改到任意目录。
        ttk.Label(record_group, text="保存目录：").pack(anchor="w", pady=(8, 0))
        record_dir_row = ttk.Frame(record_group)
        record_dir_row.pack(fill="x", pady=(2, 0))
        self.record_dir_entry = ttk.Entry(record_dir_row, textvariable=self.record_dir_var)
        self.record_dir_entry.pack(side="left", fill="x", expand=True)
        self.record_dir_btn = ttk.Button(record_dir_row, text="选择…", command=self._choose_record_dir)
        self.record_dir_btn.pack(side="left", padx=(8, 0))

        # 5) Compare_Control：动作分析（模板比对，绑定 self.compare_btn）。
        self.compare_btn = ttk.Button(
            self._teacher_block, text="动作分析…", command=self._open_compare
        )
        self.compare_btn.pack(fill="x", pady=(10, 0))

        # 考试模式：叫号 / ROI 占用 / 自动录制比对（设计见 docs/exam_system_design.md）
        self.exam_btn = ttk.Button(
            self._teacher_block, text="考试模式…", command=self._open_exam_panel
        )
        self.exam_btn.pack(fill="x", pady=(6, 0))

        # 学生练习：MediaPipe 关键点规则检查（无分数，见 docs/error_analysis_plan.md §C）
        self.student_practice_btn = ttk.Button(
            self._teacher_block,
            text="学生练习…",
            command=self._enter_student_practice,
        )
        self.student_practice_btn.pack(fill="x", pady=(6, 0))

        # 学生练习块（默认不显示；进入模式后替换 teacher_block）
        self._student_block = ttk.Labelframe(primary, text="学生练习（动作问题说明）", padding=8)
        self.student_exit_btn = ttk.Button(
            self._student_block,
            text="退出学生练习",
            command=self._exit_student_practice,
        )
        self.student_exit_btn.pack(fill="x")
        ttk.Label(
            self._student_block,
            text="仅指出动作问题，不打分；MediaPipe Lite 本机CPU运行。骨架、手部、自动比对、在线识别由学生练习模式锁定为关闭，不改你的设置",
            foreground="#555555",
            wraplength=320,
        ).pack(anchor="w", pady=(6, 0))
        self.feedback_controls = FeedbackControls(
            self._student_block, root=self.root,
            set_busy=self._set_feedback_busy,
            show_result=self._set_student_result_text,
            set_status=self.student_feedback_status,
            can_analyze=lambda: self._practice is not None
            and not (self._practice.busy or self._practice.waiting)
            and self._recording.state not in {"recording", "paused"},
        )
        student_btn_row = ttk.Frame(self._student_block)
        student_btn_row.pack(fill="x", pady=(10, 0))
        self.student_start_btn = ttk.Button(
            student_btn_row, text="开始", command=self._on_practice_start
        )
        self.student_start_btn.pack(side="left", fill="x", expand=True)
        self.student_end_btn = ttk.Button(
            student_btn_row,
            text="结束",
            command=self._on_practice_end,
            state="disabled",
        )
        self.student_end_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.student_judge_btn = ttk.Button(
            self._student_block,
            text="动作评判",
            command=self._on_practice_judge,
            state="disabled",
        )
        self.student_judge_btn.pack(fill="x", pady=(8, 0))
        self.student_status_var = StringVar(value="请选择正/侧双摄后点「开始」")
        ttk.Label(
            self._student_block,
            textvariable=self.student_status_var,
            wraplength=320,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(self._student_block, text="点评结果：").pack(anchor="w", pady=(8, 0))
        student_result_frame = ttk.Frame(self._student_block)
        student_result_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.student_result_text = Text(
            student_result_frame,
            height=9,
            width=36,
            wrap="word",
            state="disabled",
            font=("Microsoft YaHei UI", 9),
        )
        student_result_scroll = ttk.Scrollbar(
            student_result_frame,
            orient="vertical",
            command=self.student_result_text.yview,
        )
        self.student_result_text.configure(yscrollcommand=student_result_scroll.set)
        self.student_result_text.pack(side="left", fill="both", expand=True)
        student_result_scroll.pack(side="right", fill="y")

        # ===== 可见分隔线：在 Primary_Controls 与 Secondary_Options 之间插入显式
        # 水平分隔，强化主/次分区（需求 1.2）。布局测试可通过该属性定位。=====
        self.primary_secondary_separator = ttk.Separator(left, orient="horizontal")
        self.primary_secondary_separator.pack(fill="x", pady=8)

        # ===== Secondary_Options 分组（次要选项，统一置于 Compare_Control 之后，
        # 需求 1.4/7.1/7.2/7.3/7.4）。=====
        self._secondary_block = ttk.Labelframe(left, text="次要选项", padding=10)
        self._secondary_block.pack(fill="x", pady=(0, 10))
        secondary = self._secondary_block

        # 输入源提示（显示当前选中的摄像头）。
        ttk.Label(secondary, textvariable=self.source_hint_var, wraplength=320).pack(
            anchor="w"
        )

        # 线程数（视频文件离线处理 / 摄像头实时并行；需求 7.2）。
        ttk.Label(secondary, text="线程数（>1：多核并行，关闭时序平滑）：").pack(anchor="w", pady=(10, 0))
        self.workers_spin = ttk.Spinbox(secondary, from_=1, to=16, textvariable=self.workers_var, width=6)
        self.workers_spin.pack(anchor="w", pady=(6, 0))

        # 启用手部检测（需求 7.3）。
        ttk.Checkbutton(secondary, text="启用手部检测（V 手势 / 手部骨架）", variable=self.enable_hands_var).pack(
            anchor="w", pady=(10, 0)
        )

        # ===== Status_Area（状态区，固定置于控制区底部，需求 7.6/7.7/7.8）。=====
        self._info_block = ttk.Labelframe(left, text="状态", padding=10)
        self._info_block.pack(fill="x")
        info = self._info_block
        ttk.Label(info, textvariable=self.status_var, wraplength=320).pack(anchor="w")
        # 录制状态文本与 Result_Video 完整路径（需求 5.9/5.10）。
        ttk.Label(info, textvariable=self.recording_status_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        self._info_teacher_extra = ttk.Frame(info)
        self._info_teacher_extra.pack(fill="x")
        ttk.Separator(self._info_teacher_extra, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(self._info_teacher_extra, text="双摄自动比对：").pack(anchor="w")
        ttk.Label(self._info_teacher_extra, textvariable=self.compare_segment_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        ttk.Label(self._info_teacher_extra, textvariable=self.compare_status_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        ttk.Label(self._info_teacher_extra, textvariable=self.compare_score_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        ttk.Label(self._info_teacher_extra, textvariable=self.compare_error_var, wraplength=320).pack(anchor="w", pady=(4, 0))
        ttk.Label(self._info_teacher_extra, text="识别结果：").pack(anchor="w", pady=(8, 0))
        ttk.Label(self._info_teacher_extra, textvariable=self.actions_var, wraplength=320).pack(anchor="w")
        online_match_row = ttk.Frame(self._info_teacher_extra)
        online_match_row.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            online_match_row,
            text="实时动作识别",
            variable=self.online_match_var,
        ).pack(side="left")
        ttk.Label(online_match_row, textvariable=self.match_var, wraplength=210).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(self._info_teacher_extra, textvariable=self.progress_text_var, wraplength=320).pack(anchor="w", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(
            self._info_teacher_extra, orient="horizontal", mode="determinate", maximum=100.0
        )
        self.progress_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(self._info_teacher_extra, text="提示：点击“停止”结束识别。").pack(
            anchor="w", pady=(8, 0)
        )

        # Right: preview
        right = ttk.Labelframe(outer, text="预览", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        # 第二行/列默认权重 0。双摄时根据旋转后的画面比例切换上下或左右排布；
        # 单摄隐藏第二路时必须同时清空第二条 grid 轨道的权重，避免主预览被空白挤压。
        right.rowconfigure(1, weight=0)
        right.columnconfigure(1, weight=0)

        self.preview = ttk.Label(right, cursor="hand2")
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.bind("<Configure>", self._on_preview_configure)
        self.preview.bind("<Button-1>", self._on_preview_click_rotate)

        # 第二路预览（双摄像头双面视图，issue #58）：默认隐藏，仅 source2 选中真实摄像头时显示。
        self.right = right
        self._preview_right = right
        self._dual_preview_visible = False
        self._dual_preview_layout = "stacked"
        self.preview2 = ttk.Label(right, cursor="hand2")
        self.preview2.grid(row=1, column=0, sticky="nsew")
        self.preview2.bind("<Configure>", self._on_preview2_configure)
        self.preview2.bind("<Button-1>", self._on_preview2_click_rotate)
        self._set_dual_preview_visible(False)

        # 学生练习分析结果主视图 (Milestone 2)；任务代次在 PracticeSession
        self.current_student_id = self.feedback_controls.student_id

        from apps.tk.views.practice_result import FeedbackResultPanel

        self.feedback_result_panel = FeedbackResultPanel(
            outer,
            on_practice_again=self._on_student_practice_again,
            on_export_report=self._on_student_export_report,
            on_view_history=self._on_student_view_history,
            on_review_toggle=self._on_student_review_toggle,
            on_switch_to_preview=self._show_student_preview_view,
            history_root=self.feedback_controls.history().root,
        )
        self.feedback_result_panel.grid_remove()

        self.feedback_controls.on_identity_changed = self._on_student_identity_changed
        self.feedback_controls.on_cancelled = self._on_student_cancelled

    def _on_preview_configure(self, event) -> None:
        """主预览 Label 尺寸变化（主线程）：缓存供会话线程按它缩放预览帧。"""
        self._preview_wh = (event.width, event.height)

    def _on_preview2_configure(self, event) -> None:
        """第二路预览 Label 尺寸变化（主线程），与 _on_preview_configure 同构。"""
        self._preview_wh2 = (event.width, event.height)

    def _get_runtime_rotate(self, which: str = "primary") -> int:
        return self._live_controls.rotation_of(SECONDARY if which == "secondary" else PRIMARY)

    def _set_runtime_rotate(
        self,
        degrees: int,
        *,
        which: str = "primary",
        sync_combo: bool = True,
        announce: bool = True,
    ) -> int:
        """设置一路运行时旋转（0/90/180/270），可选同步内部角度缓存与状态提示。"""
        d = int(degrees) if int(degrees) in _VALID_ROTATIONS else 0
        self._live_controls.rotation[SECONDARY if which == "secondary" else PRIMARY].set(d)
        if sync_combo:
            # 无下拉控件；仍同步内部 StringVar，供 _collect_state / 考试 ROI 读数。
            label = _degrees_to_rotate_label(d)
            try:
                if which == "secondary" and hasattr(self, "rotate_var_2"):
                    if self.rotate_var_2.get() != label:
                        self.rotate_var_2.set(label)
                elif which != "secondary" and hasattr(self, "rotate_var"):
                    if self.rotate_var.get() != label:
                        self.rotate_var.set(label)
            except Exception:
                pass
        if announce:
            side = "侧摄" if which == "secondary" else "主摄"
            msg = f"{side}旋转 → {d}°（点击预览可继续顺时针 90°）"
            if which != "secondary" and self.modes.is_active("exam"):
                msg += "；考试 ROI 可能失效，请重新标定"
            try:
                self.status_var.set(msg)
            except Exception:
                pass
        return d

    def _bump_runtime_rotate_cw(self, which: str = "primary") -> int:
        cur = self._get_runtime_rotate(which)
        return self._set_runtime_rotate(_next_rotate_cw(cur), which=which)

    def _on_preview_click_rotate(self, _event=None) -> None:
        self._bump_runtime_rotate_cw("primary")

    def _on_preview2_click_rotate(self, _event=None) -> None:
        if not bool(getattr(self, "_dual_preview_visible", False)):
            return
        self._bump_runtime_rotate_cw("secondary")

    def _current_template_paths(self) -> tuple[Path, Path]:
        """主线程读取 UI 填入的正/侧模板路径；空串回落默认 fixed 路径。"""
        front_default, side_default = default_template_paths()
        try:
            front_raw = (
                str(self.front_template_var.get()).strip()
                if hasattr(self, "front_template_var")
                else ""
            )
        except Exception:
            front_raw = ""
        try:
            side_raw = (
                str(self.side_template_var.get()).strip()
                if hasattr(self, "side_template_var")
                else ""
            )
        except Exception:
            side_raw = ""
        front = Path(front_raw) if front_raw else front_default
        side = Path(side_raw) if side_raw else side_default
        return front, side

    def _build_template_pool_ui(
        self, parent, front_list: list[Path], side_list: list[Path]
    ) -> None:
        """正/侧两个模板池 Listbox + 添加/移除按钮。动作名 = 文件 stem。"""
        for view, label, seed in (
            ("front", "正面模板池", front_list),
            ("side", "侧面模板池", side_list),
        ):
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=(6, 0))
            head = ttk.Frame(frame)
            head.pack(fill="x")
            ttk.Label(head, text=label).pack(side="left")
            ttk.Button(
                head, text="添加…", width=6,
                command=lambda v=view: self._add_pool_templates(v),
            ).pack(side="right")
            ttk.Button(
                head, text="移除选中", width=8,
                command=lambda v=view: self._remove_pool_selected(v),
            ).pack(side="right", padx=(0, 4))
            lb = Listbox(frame, height=4, selectmode="extended", exportselection=False)
            lb.pack(fill="x", pady=(2, 0))
            for p in seed:
                lb.insert("end", str(p))
            setattr(self, f"_{view}_template_listbox", lb)
        self.template_status_var = getattr(self, "template_status_var", StringVar(value=""))
        ttk.Label(
            parent,
            textvariable=self.template_status_var,
            wraplength=320,
            foreground="#555",
        ).pack(anchor="w", pady=(2, 0))

    def _pool_listbox(self, view: str):
        return getattr(self, f"_{view}_template_listbox", None)

    def _pool_paths(self, view: str) -> list[Path]:
        lb = self._pool_listbox(view)
        if lb is None:
            return []
        return [Path(str(lb.get(i)).strip()) for i in range(lb.size()) if str(lb.get(i)).strip()]

    def _current_template_lists(self) -> tuple[list[Path], list[Path]]:
        """主线程读取正/侧模板池（正对正、侧对侧）。"""
        return self._pool_paths("front"), self._pool_paths("side")

    def _add_pool_templates(self, view: str) -> None:
        lb = self._pool_listbox(view)
        if lb is None:
            return
        paths = filedialog.askopenfilenames(
            title="添加正面模板" if view == "front" else "添加侧面模板",
            filetypes=[("模板文件", "*.npz"), ("所有文件", "*.*")],
        )
        if not paths:
            return
        existing = {str(p) for p in self._pool_paths(view)}
        rejected: list[str] = []
        for path in paths:
            usable = try_load_heavy_template(path)
            if usable is None:
                rejected.append(Path(path).name)
                continue
            if str(usable) in existing:
                continue
            lb.insert("end", str(usable))
            existing.add(str(usable))
        self._persist_template_pools()
        self._sync_template_status()
        if rejected:
            messagebox.showerror(
                "部分模板无效",
                "以下文件不是可用的 heavy + pose33_v3 模板，已跳过：\n"
                + "\n".join(rejected)
                + "\n请用 make_template.py --pose heavy 生成后再添加。",
                parent=self.root,
            )

    def _remove_pool_selected(self, view: str) -> None:
        lb = self._pool_listbox(view)
        if lb is None:
            return
        for i in reversed(lb.curselection()):
            lb.delete(i)
        self._persist_template_pools()
        self._sync_template_status()

    def _persist_template_pools(self) -> None:
        front, side = self._current_template_lists()
        # 同步旧单值 StringVar（兼容既有引用），运行时以池为准
        try:
            self.front_template_var.set(str(front[0]) if front else "")
            self.side_template_var.set(str(side[0]) if side else "")
        except Exception:
            pass
        try:
            save_template_pools(front=front, side=side)
        except Exception:
            pass

    def _sync_template_status(self) -> None:
        var = getattr(self, "template_status_var", None)
        if var is None:
            return
        front, side = self._current_template_lists()
        fn = len(front)
        sn = len(side)
        if fn and sn:
            text = f"模板池：正面 {fn} 个 · 侧面 {sn} 个（正对正、侧对侧，逐动作出分）"
        elif fn:
            text = f"模板池：正面 {fn} 个 · 侧面 0 个（仅比对正面；两侧都空才报错）"
        elif sn:
            text = f"模板池：正面 0 个 · 侧面 {sn} 个（仅比对侧面；两侧都空才报错）"
        else:
            text = "模板池：正/侧均为空 — 开启自动比对将失败"
        try:
            var.set(text)
        except Exception:
            pass

    def _sync_auto_compare_hint(self) -> None:
        on = bool(self.auto_compare_var.get()) if hasattr(self, "auto_compare_var") else True
        hint = (
            "录制后自动比对（检测一条龙）"
            if on
            else "仅录制落盘，不自动比对"
        )
        var = getattr(self, "auto_compare_hint_var", None)
        if var is not None:
            try:
                var.set(hint)
            except Exception:
                pass

    def _on_auto_compare_scale(self, value) -> None:
        try:
            on = float(value) >= 0.5
        except (TypeError, ValueError):
            on = True
        # 吸附到两端
        try:
            self.auto_compare_scale_var.set(1.0 if on else 0.0)
        except Exception:
            pass
        if hasattr(self, "auto_compare_var"):
            try:
                if bool(self.auto_compare_var.get()) != on:
                    self.auto_compare_var.set(on)
            except Exception:
                pass
        self._dual_auto_compare = bool(on)
        self._sync_auto_compare_hint()

    def _on_auto_compare_var_write(self, *_args) -> None:
        on = bool(self.auto_compare_var.get())
        try:
            target = 1.0 if on else 0.0
            if abs(float(self.auto_compare_scale_var.get()) - target) > 1e-6:
                self.auto_compare_scale_var.set(target)
        except Exception:
            pass
        self._dual_auto_compare = on
        self._sync_auto_compare_hint()

    def _current_auto_compare(self, *, prefer_ui: bool = False) -> bool:
        """片段级自动比对开关。

        - 当前模式锁定时按模式（考试强制开、学生练习强制关）。
        - 默认读主线程维护的 ``_dual_auto_compare`` 缓存（worker 可安全读取）。
        - ``prefer_ui=True`` 时（仅主线程）从滑块/BooleanVar 刷新缓存。
        """
        forced = self.modes.segment_auto_compare
        if forced is not None:
            return forced
        if prefer_ui and hasattr(self, "auto_compare_var"):
            try:
                on = bool(self.auto_compare_var.get())
                self._dual_auto_compare = on
                return on
            except Exception:
                pass
        return bool(getattr(self, "_dual_auto_compare", True))

    def _set_dual_preview_visible(self, visible: bool) -> None:
        """显示/隐藏第二预览，并同步清理不再使用的 grid 轨道。"""
        self._dual_preview_visible = bool(visible)
        if visible:
            self.preview2.grid()
            self._set_dual_preview_layout(self._dual_preview_layout)
            return

        # 隐藏时复位到单路布局。仅 grid_remove() 不会释放旧行/列权重，空轨道仍会
        # 挤占主预览；因此坐标和四个权重必须一起复位。
        self._dual_preview_layout = "stacked"
        self.preview.grid_configure(row=0, column=0, sticky="nsew")
        self.preview2.grid_configure(row=1, column=0, sticky="nsew")
        self.preview2.grid_remove()
        self._preview_right.rowconfigure(0, weight=1)
        self._preview_right.rowconfigure(1, weight=0)
        self._preview_right.columnconfigure(0, weight=1)
        self._preview_right.columnconfigure(1, weight=0)

    def _set_dual_preview_layout(self, layout: str) -> None:
        """在 Tk 主线程切换双摄布局：竖画面左右、其他情况上下。"""
        if layout not in ("stacked", "side_by_side"):
            raise ValueError(f"未知双摄预览布局：{layout}")

        self._dual_preview_layout = layout
        self.preview.grid_configure(row=0, column=0, sticky="nsew")
        if not getattr(self, "_dual_preview_visible", False):
            self._preview_right.rowconfigure(0, weight=1)
            self._preview_right.rowconfigure(1, weight=0)
            self._preview_right.columnconfigure(0, weight=1)
            self._preview_right.columnconfigure(1, weight=0)
            return

        if layout == "side_by_side":
            self.preview2.grid_configure(row=0, column=1, sticky="nsew")
            self._preview_right.rowconfigure(0, weight=1)
            self._preview_right.rowconfigure(1, weight=0)
            self._preview_right.columnconfigure(0, weight=1)
            self._preview_right.columnconfigure(
                1, weight=1 if self._dual_preview_visible else 0
            )
        else:
            self.preview2.grid_configure(row=1, column=0, sticky="nsew")
            self._preview_right.rowconfigure(0, weight=1)
            self._preview_right.rowconfigure(
                1, weight=1 if self._dual_preview_visible else 0
            )
            self._preview_right.columnconfigure(0, weight=1)
            self._preview_right.columnconfigure(1, weight=0)

    def _open_compare(self) -> None:
        if self._compare_win and self._compare_win.is_open():
            self._compare_win.focus()
            return
        # 比对窗口创建失败时弹框并保持主窗口状态不变（需求 6.3）：
        # 不修改 self._compare_win（保留先前值/None），主窗口控件与录制状态均不受影响。
        try:
            win = CompareWindow(self.root)
        except Exception as e:
            messagebox.showerror("打开失败", f"动作比对窗口创建失败：{e}")
            return
        self._compare_win = win

    def _open_settings(self) -> None:
        if self._settings_win and self._settings_win.is_open():
            self._settings_win.focus()
            return
        try:
            win = SettingsWindow(
                self.root,
                current_pose=self.pose_var.get(),
                hands_enabled=bool(self.enable_hands_var.get()),
            )
        except Exception as e:
            messagebox.showerror("打开失败", f"设置窗口创建失败：{e}")
            return
        self._settings_win = win

    def _choose_record_dir(self) -> None:
        """选择录制视频保存目录。"""
        current = self.record_dir_var.get().strip()
        d = filedialog.askdirectory(
            title="选择录制视频保存目录",
            initialdir=current or str(outputs_dir()),
        )
        if d:
            self.record_dir_var.set(d)
            save_record_dir(d)

    def _on_record_toggle(self) -> None:
        """Record_Toggle 点击回调：请求录制状态机切换，并据返回状态刷新按钮文本。

        idle→开始录制、recording→暂停录制、paused→继续录制（见 RECORD_BTN_TEXT）。
        考试模式下手动录制锁定，直接返回。双摄片段已分叉（一路写失败或状态不一致）时
        先结束并提交该片段，不在分叉状态上开新片段。
        """
        if bool(getattr(self, "_exam_manual_locked", False)):
            return
        if getattr(self, "_closing", False) or not App._session_live(self):
            return
        if not bool(getattr(self, "_dual_recording_ready", True)):
            return

        if self._recording.finish_if_broken() is not None:
            new_state: RecordingState = "idle"
        elif self._recording.state == "idle":
            App._begin_recording_segment(self)
            return
        else:
            new_state = self._recording.toggle_pause()
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None:
            record_btn.configure(text=RECORD_BTN_TEXT[new_state])
        sync_record_stop = getattr(self, "_sync_record_stop_enabled", None)
        if sync_record_stop is not None:
            sync_record_stop(new_state)

    def _begin_recording_segment(self, intent: SegmentIntent | None = None):
        """从 idle 开录一段（工作流与手动共用）；成功返回片段引用，失败返回 None。

        前置：会话已 begin_session；各路同时 idle→recording，任一路失败则全部复位。片段意图
        （考生信息；自动比对未指定时按当前模式）随片段登记，结束时据此构建处理计划。
        """
        if getattr(self, "_closing", False) or not App._session_live(self):
            return None
        if not bool(getattr(self, "_dual_recording_ready", True)):
            return None

        record_dir_var = getattr(self, "record_dir_var", None)
        next_base_dir = getattr(self, "_record_base_dir", outputs_dir())
        if record_dir_var is not None:
            base = record_dir_var.get().strip()
            # 空串多为 Tk 读取时序异常，非"用户要默认目录"。保持上次已缓存的目录，
            # 不回退 outputs_dir()，避免下方 save_record_dir 把持久化路径覆盖成默认。
            if base:
                next_base_dir = Path(base)
        self._record_base_dir = Path(next_base_dir)
        save_record_dir(next_base_dir)
        intent = intent or SegmentIntent()
        if intent.auto_compare is None:
            intent = replace(intent, auto_compare=self.modes.segment_auto_compare)
        segment = self._recording.start(SegmentMeta(Path(next_base_dir), context=intent))
        if isinstance(segment, BeginError):
            return None

        if segment.multi:
            App._show_new_dual_segment(self, segment, intent)
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None and not getattr(self, "_exam_manual_locked", False):
            record_btn.configure(text=RECORD_BTN_TEXT["recording"])
        self._sync_record_stop_enabled("recording")
        return segment

    def _show_new_dual_segment(self, segment, intent: SegmentIntent) -> None:
        """双摄新片段：记下本段是否自动比对，刷新比对状态栏。"""
        auto_compare = intent.auto_compare
        if auto_compare is None:
            auto_compare = App._current_auto_compare(self, prefer_ui=True)
        self._dual_auto_compare = bool(auto_compare)
        status_text = "自动比对：录制中" if auto_compare else "仅录制：录制中"
        for name, value in (
            ("compare_segment_var", f"片段：{segment.segment_id}"),
            ("compare_status_var", status_text),
            ("compare_score_var", "正面：-　侧面：-　综合：-"),
            ("compare_error_var", ""),
        ):
            var = getattr(self, name, None)
            if var is not None:
                var.set(value)

    def _end_recording_segment(self, *, discard: bool = False) -> None:
        """结束当前录制片段；discard=True 时只落盘、不入比对队列。

        discard 在关窗/中止路径也必须执行：不得因 ``_closing`` 提前 return，
        否则随后的普通 finalize 会把已 aborted 片段送去评分。
        """
        closing = bool(getattr(self, "_closing", False))
        stopped = not App._session_live(self)
        if not discard and (closing or stopped):
            return
        self._recording.finish(discard=discard)
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None and not getattr(self, "_exam_manual_locked", False):
            record_btn.configure(text=RECORD_BTN_TEXT["idle"])
        self._sync_record_stop_enabled("idle")

    def _on_record_stop(self) -> None:
        """「结束录制」点击回调：结束当前录制片段并落盘，但不结束识别会话。"""
        if bool(getattr(self, "_exam_manual_locked", False)):
            return
        if getattr(self, "_closing", False) or not App._session_live(self):
            return
        App._end_recording_segment(self, discard=False)

    def _open_exam_panel(self) -> None:
        if self.modes.is_active("practice"):
            messagebox.showinfo("学生练习", "请先退出学生练习模式，再打开考试模式。")
            return
        existing = self.exam_panel
        if existing is not None:
            try:
                existing.win.lift()
                return
            except Exception:
                pass
        from apps.exam_panel import ExamPanel

        ExamPanel(self)

    # ------------------------------------------------------------------
    # 学生练习模式（MediaPipe 关键点规则检查，无分数）；流程在 core.workflows.PracticeSession
    # ------------------------------------------------------------------

    def _enter_student_practice(self) -> None:
        """切换到学生练习极简界面：开始 / 结束 / 动作评判 / 预览。"""
        if self.modes.is_active("practice"):
            return
        if self.modes.is_active("exam") or bool(getattr(self, "_exam_manual_locked", False)):
            messagebox.showinfo("学生练习", "考试进行中，无法进入学生练习。")
            return
        if App._session_active(self):
            messagebox.showinfo(
                "学生练习", "请先停止当前识别会话，再进入学生练习。"
            )
            return

        # 骨架、手部、自动比对、在线识别由模式要求在会话配置里锁定为关，不改写用户勾选（V-5）
        self.modes.enter(PRACTICE)
        self._practice = PracticeSession(self.feedback_controls.identity)
        self._student_announcer = Announcer()
        ports = WorkflowPorts(
            recorder=_SegmentPort(self),
            occupancy=_OccupancyPort(self, "practice"),
            announcer=self._student_announcer,
            ui=_PracticeUi(self),
            analysis=_PracticeAnalysis(self),
        )
        self.workflows.activate(self._practice, CommandExecutor(ports))
        if hasattr(self, "feedback_result_panel"):
            self.feedback_result_panel.clear()
            self._show_student_preview_view()

        for block in (self._teacher_block, self.primary_secondary_separator, self._secondary_block,
                      self._info_teacher_extra):
            try:
                block.pack_forget()
            except Exception:
                pass
        try:
            self.settings_btn.configure(state="disabled")
        except Exception:
            pass

        self._student_block.pack(fill="both", expand=True, pady=(10, 0))
        self._set_student_result_text(
            "动作问题说明由MediaPipe关键点与几何规则计算，候选问题待教师确认，不打分。\n"
            "流程：填学号、选择动作与左/右式 → 开始 → 进入黄框站稳，播报开始并录制 → 离开2秒，播报结束 → 动作评判。\n"
            "黄框沿用考试模式保存的站位区域；可在考试模式中调整。也可手动点「结束」。\n"
            "也可导入仍保留的旧视频；个人历史支持文字导出和教师确认/撤销。"
        )
        self.student_status_var.set("请选择正/侧双摄后点「开始」")
        self._sync_student_buttons()
        self.root.title("学生练习（动作问题说明）")
        self.status_var.set("学生练习模式")
        self._sync_camera_warmup()

    def _exit_student_practice(self) -> None:
        """退出学生练习并恢复教师控件；用户的骨架/手部/自动比对/在线识别勾选从未被改写。"""
        practice = self._practice
        if practice is None:
            return
        if practice.busy:
            messagebox.showinfo("学生练习", "动作评判进行中，请稍候再退出。")
            return

        # 若仍在录制则先结束片段；再停会话
        self.workflows.dispatch("exit")
        if self._student_announcer is not None:
            self._student_announcer.close()
            self._student_announcer = None
        if App._session_active(self):
            self._stop()
        self.workflows.deactivate(practice)
        self._practice = None
        self.modes.leave("practice")
        if self.exam_panel is not None:
            # 考试面板仍开着：站位、tick 交还给考试流程
            self.workflows.activate(self.exam_panel.session, self.exam_panel.executor)
        if hasattr(self, "feedback_result_panel"):
            self.feedback_result_panel.clear()
            self._show_student_preview_view()
        App._restore_teacher_blocks(self)

        self.root.title("MediaPipe 动作识别（人体姿态 + 手部）")
        if App._session_active(self):
            # 会话尚在收尾：保持开始禁用，避免点击被静默丢弃；会话结束事件负责恢复。
            self.status_var.set("正在停止…")
            for name in ("start_btn", "stop_btn"):
                button = getattr(self, name, None)
                if button is not None:
                    button.configure(state="disabled")
            return
        self.status_var.set("就绪")
        self._set_running_controls(False)
        self._sync_camera_warmup()

    def _restore_teacher_blocks(self) -> None:
        """退出学生练习后恢复教师控件，pack 顺序：primary → separator → secondary → info。"""
        try:
            self._student_block.pack_forget()
        except Exception:
            pass
        try:
            self._teacher_block.pack(fill="x")
        except Exception:
            pass
        info_block = getattr(self, "_info_block", None)
        try:
            if info_block is not None:
                self.primary_secondary_separator.pack(
                    fill="x", pady=8, before=info_block
                )
            else:
                self.primary_secondary_separator.pack(fill="x", pady=8)
        except Exception:
            pass
        try:
            if info_block is not None:
                self._secondary_block.pack(
                    fill="x", pady=(0, 10), before=info_block
                )
            else:
                self._secondary_block.pack(fill="x", pady=(0, 10))
        except Exception:
            pass
        try:
            self._info_teacher_extra.pack(fill="x")
        except Exception:
            pass
        try:
            self.settings_btn.configure(state="normal")
        except Exception:
            pass

    def _student_dual_cameras_ready(self) -> tuple[bool, str]:
        label2 = (self.camera_choice_var_2.get() or "").strip()
        if not label2 or label2 == NO_SECOND_CAMERA:
            return False, "学生练习需要正/侧双摄像头，请选择第二摄像头（侧面）。"
        label1 = (self.camera_choice_var.get() or "").strip()
        if not label1:
            return False, "请选择第一摄像头（正面）。"
        return True, ""

    def _sync_student_buttons(self) -> None:
        practice = self._practice
        if practice is None:
            return
        judging = practice.busy
        recording = practice.recording
        waiting = practice.waiting
        if hasattr(self, "feedback_controls"):
            self.feedback_controls.set_recording(recording or waiting)

        try:
            self.student_start_btn.configure(
                state="disabled" if judging or recording or waiting else "normal"
            )
            self.student_end_btn.configure(
                state="normal" if ((recording or waiting) and not judging) else "disabled"
            )
            # 仅后处理终态就绪后才允许评判（避免与转码抢文件）
            self.student_judge_btn.configure(
                state="normal" if practice.can_judge else "disabled"
            )
            self.student_exit_btn.configure(
                state="disabled" if judging else "normal"
            )
        except Exception:
            pass

    def _set_student_result_text(self, text: str) -> None:
        widget = getattr(self, "student_result_text", None)
        if widget is None:
            return
        try:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")
        except Exception:
            pass

    def _practice_dispatch(self, event: str, **kw) -> None:
        """把界面事件交给学生练习流程（只在练习流程处于激活时）。"""
        practice = self._practice
        if practice is not None and self.workflows.active is practice:
            self.workflows.dispatch(event, **kw)

    def _on_practice_start(self) -> None:
        """「开始」：预检通过后开启双摄（未开时），进入站位区域站稳后由练习流程开录。"""
        practice = self._practice
        if practice is None or practice.busy or practice.waiting or practice.recording:
            return
        problems = PRACTICE.requirements.preflight(App._preflight_context(self, practice=True))
        if problems:
            messagebox.showwarning("学生练习", problems[0])
            return
        self._live_controls.set_roi(PreferenceStore().get(EXAM_ROI))
        if App._session_active(self):
            self._practice_dispatch("start", now=time.monotonic(), session_ready=bool(self._dual_recording_ready))
            return
        # 会话 state 冻结 1080p 请求；池发现预热格式不一致时会在旧进程释放后重开。
        self._practice_dispatch("start", now=time.monotonic(), session_ready=False, launching=True)
        try:
            self._start()
        except Exception as e:
            self._practice_dispatch("session_failed")
            messagebox.showerror("学生练习", f"启动失败：{e}")
            return
        if not App._session_active(self):
            self._practice_dispatch("session_failed", message="双摄未启动，请检查摄像头后重试")
        elif bool(getattr(self, "_dual_recording_ready", False)):
            self._practice_dispatch("session_ready", now=time.monotonic())

    def _on_practice_end(self) -> None:
        """「结束」：停录落盘（或取消等待到位），不自动评判。"""
        self._practice_dispatch("end")

    def _on_practice_judge(self) -> None:
        """「动作评判」：按开录时绑定的学生与动作后台分析。"""
        self._practice_dispatch("judge")

    def student_feedback_status(self, text: str) -> None:
        self.student_status_var.set(text)

    def _set_feedback_busy(self, busy: bool) -> None:
        self._practice_dispatch("analysis_busy", busy=bool(busy))

    def _on_student_identity_changed(self) -> None:
        """用户在学生练习界面修改了学号：作废当前任务代次，清空旧记录。"""
        self._practice_dispatch("identity_changed")

    def _on_student_cancelled(self) -> None:
        """用户点击了取消分析：作废任务代次。"""
        self._practice_dispatch("analysis_cancelled")

    def _show_student_result_view(self, record: dict) -> None:
        """切换至学生练习分析结果视图：平滑隐藏预览和左侧控制栏，展示结果面板。"""
        try:
            self.left_container.grid_remove()
        except Exception:
            pass
        try:
            self._preview_right.grid_remove()
        except Exception:
            pass
        self.feedback_result_panel.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.feedback_result_panel.set_record(record)
        try:
            self.status_var.set("学生练习分析结果")
        except Exception:
            pass

    def _show_student_preview_view(self) -> None:
        """切回学生练习录制预览视图：隐藏结果面板，恢复左侧控制栏与双摄预览。"""
        try:
            self.feedback_result_panel.grid_remove()
        except Exception:
            pass
        try:
            self.left_container.grid()
        except Exception:
            pass
        try:
            self._preview_right.grid()
        except Exception:
            pass
        try:
            self.status_var.set("学生练习模式")
        except Exception:
            pass

    def _on_student_practice_again(self) -> None:
        """从分析结果页切回录制预览，重置状态，但严格禁止自动启动录制。"""
        practice = self._practice
        if practice is None or practice.busy:
            return
        self._show_student_preview_view()
        self._practice_dispatch("practice_again")

    def _on_student_analysis_saved(self, token: int, student_id: str, record: dict) -> None:
        """分析完成且保存后的主线程回调：交给练习流程按任务代次与学号拦截晚到/作废结果。"""
        if bool(getattr(self, "_closing", False)):
            return
        curr_student_id = None
        if hasattr(self, "current_student_id") and hasattr(self.current_student_id, "get"):
            curr_student_id = self.current_student_id.get()
        elif hasattr(self, "feedback_controls") and hasattr(self.feedback_controls, "student_id"):
            curr_student_id = self.feedback_controls.student_id.get()
        self._practice_dispatch(
            "analysis_saved", token=token, student_id=student_id, record=record,
            current_student_id=curr_student_id,
        )

    def _on_student_export_report(self, record: dict) -> None:
        """从结果面板导出 HTML / TXT 图文报告。"""
        from core.reporting import render_html_report

        student_id = record.get("studentId", "student")
        action = record.get("action", "action")
        action_label = ACTIONS.get(action, action)
        default_name = f"{student_id}_{action_label}_练习分析报告"

        dest = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出动作问题说明报告",
            initialfile=default_name,
            defaultextension=".html",
            filetypes=[("图文报告网页 (*.html)", "*.html"), ("详细文字报告 (*.txt)", "*.txt")],
        )
        if not dest:
            return

        path = Path(dest)
        try:
            if path.suffix.lower() == ".txt":
                self.feedback_controls.history().export(record["id"], path)
            else:
                html_content = render_html_report(record, history_root=self.feedback_controls.history().root)
                path.write_text(html_content, encoding="utf-8")
            self.student_status_var.set(f"报告已成功导出至：{path.name}")
            messagebox.showinfo("导出成功", f"报告已导出至：\n{path}", parent=self.root)
        except Exception as exc:
            messagebox.showerror("导出失败", f"无法写入报告文件：{exc}", parent=self.root)

    def _on_student_view_history(self) -> None:
        """从结果面板打开个人历史对话框。"""
        self.feedback_controls.open_history()

    def _on_student_review_toggle(self, record_id: str, check_id: str, decision: str, teacher: str, reason: str) -> None:
        """教师在结果页切换复核状态并即时刷新结果面板。"""
        try:
            updated = self.feedback_controls.history().review(
                record_id, check_id, decision, teacher, reason
            )
            if self._practice is not None:
                self._practice.current_record = updated
            self.feedback_result_panel.set_record(updated)
        except Exception as exc:
            messagebox.showerror("复核失败", str(exc), parent=self.root)

    # ---- 考试面板使用的公共接口（面板不访问 App 私有成员） ----

    def exam_primary_rotation(self) -> int:
        """考试 ROI 按 primary 旋转后的画面判定；返回当前 primary 旋转角。"""
        try:
            return int(self._get_runtime_rotate("primary"))
        except Exception:
            try:
                return _parse_rotate(self.rotate_var.get())
            except Exception:
                return 0

    def exam_ports(self, *, ui, announcer) -> WorkflowPorts:
        """考试流程的端口：录制与站位检测接到本窗口的会话，界面与播报由面板提供。"""
        return WorkflowPorts(
            recorder=_SegmentPort(self), occupancy=_OccupancyPort(self, "exam"),
            announcer=announcer, ui=ui,
        )

    def exam_preflight(self) -> list[str]:
        """开考预检（考试模式声明的条件）；返回不满足的条件，空表示可以开考。"""
        return EXAM.requirements.preflight(App._preflight_context(self))

    def set_exam_mode(self, active: bool) -> None:
        """开考时进入考试模式；主会话停止、换名单或关闭面板时退出。"""
        if active:
            self.modes.enter(EXAM)
            self._dual_auto_compare = True
            return
        self.modes.leave("exam")
        self._live_controls.arm_occupancy("exam", False)

    def lock_manual_record(self, locked: bool) -> None:
        """考试进行中锁定手动录制按钮与骨架、自动比对开关（考试强制比对、骨架关）。"""
        self._exam_manual_locked = bool(locked)
        state = "disabled" if locked else "normal"
        for name in ("record_btn", "record_stop_btn", "record_skeleton_check", "auto_compare_check"):
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                if locked:
                    w.configure(state="disabled")
                else:
                    # 恢复由会话态控制
                    if name == "record_skeleton_check":
                        if not App._session_active(self):
                            w.configure(state="normal")
                    elif name == "auto_compare_check":
                        # 自动比对滑块：非考试时始终可调；考试锁定期间禁用。
                        w.configure(state="normal")
                    elif name == "record_btn":
                        if getattr(self, "_dual_recording_ready", False) or (
                            App._session_active(self)
                        ):
                            w.configure(state="normal")
            except Exception:
                pass
        if locked:
            try:
                self.record_skeleton_var.set(False)
                self.auto_compare_var.set(True)
                self.auto_compare_scale_var.set(1.0)
                self._dual_auto_compare = True
                self._sync_auto_compare_hint()
            except Exception:
                pass

    def _preflight_context(self, *, practice: bool = False) -> PreflightContext:
        """采集模式预检要用的事实；练习另查身份与双摄选择，考试另查模板池。"""
        skeleton_on = bool(getattr(self, "_dual_record_skeleton", False))
        try:
            skeleton_var = getattr(self, "record_skeleton_var", None)
            if skeleton_var is not None and bool(skeleton_var.get()):
                skeleton_on = True
        except Exception:
            pass
        identity_error = cameras_error = None
        front: list = []
        side: list = []
        if practice:
            try:
                self.feedback_controls.identity()
            except ValueError as exc:
                identity_error = str(exc)
            ok, err = self._student_dual_cameras_ready()
            cameras_error = None if ok else err
        elif bool(getattr(self, "_dual_active", False)):
            front, side = self._current_template_lists()
        return PreflightContext(
            dual_active=bool(getattr(self, "_dual_active", False)),
            recording_ready=bool(getattr(self, "_dual_recording_ready", False)),
            skeleton_on=skeleton_on,
            front_templates=tuple(front),
            side_templates=tuple(side),
            model_installed=_model_installed,
            identity_error=identity_error,
            cameras_error=cameras_error,
        )

    def _close_recording_session(self) -> None:
        """会话结束时关闭录制：仍在进行的片段照常提交；单摄写错误留给下一次 Tk tick 提示。"""
        outcome = self._recording.close_session()
        self._dual_active = False
        segment = outcome.segment
        if segment is not None and not segment.multi:
            for error in segment.errors.values():
                self._pending_record_errors.append(f"第一路：{error}")

    def _recording_plan(self, segment: FinishedSegment) -> ProcessingPlan | None:
        """按片段开录时登记的意图与结束时的用户设置构建处理计划（见 ``core.workflows.plans``）。

        在 RecordingService 的锁内、提交前调用：先把本片段登记为"最新"，提交后的首批更新不会被当成旧片段丢弃。
        """
        if not segment.multi:
            return build_segment_plan(segment, PlanSettings(record_skeleton=False, auto_compare=False))
        front_templates, side_templates = self._current_template_lists()
        plan = build_segment_plan(
            segment,
            PlanSettings(
                record_skeleton=bool(getattr(self, "_dual_record_skeleton", False)),
                auto_compare=bool(App._current_auto_compare(self, prefer_ui=False)),
                front_templates=tuple(front_templates),
                side_templates=tuple(side_templates),
            ),
        )
        self._latest_compare_segment_id = segment.segment_id
        return plan

    def _post_recording_compare_update(self, update: SegmentUpdate) -> None:
        """后处理线程只把不可变状态写入队列，不直接调用任何 Tk API。"""
        # 关窗时仍允许考试台账回填（面板或已移交的台账），避免裁剪晚到的终态被丢弃
        if getattr(self, "_closing", False):
            if self.exam_panel is None and not self.exam_scorebooks.has_detached:
                return
        self._compare_update_queue.put(update)

    def _drain_recording_compare_updates(self) -> None:
        """在 Tk 主线程消费后台状态。

        主界面比对条只显示最新片段；考试台账必须按 segment_id 回填每一位，
        不得因「非最新」而丢弃前面考生的 completed/failed。
        """
        update_queue = getattr(self, "_compare_update_queue", None)
        if update_queue is None:
            return
        while True:
            try:
                update = update_queue.get_nowait()
            except Empty:
                return
            closing = bool(getattr(self, "_closing", False))
            is_latest = update.segment_id == getattr(
                self, "_latest_compare_segment_id", None
            )
            if is_latest and not closing:
                status_labels = {
                    "queued": "排队中",
                    "transcoding": "正在转码",
                    "validating": "正在校验",
                    "comparing": "正在比对",
                    "completed": "已完成",
                    "failed": "失败",
                    "skipped": "已跳过",
                    "cancelled": "已取消",
                }
                self.compare_segment_var.set(f"片段：{update.segment_id}")
                self.compare_status_var.set(
                    f"自动比对：{status_labels.get(update.status, update.status)}"
                )
                if update.status == "completed":
                    def _pct(score: float | None) -> str:
                        if score is None:
                            return "-"
                        return f"{float(score) * 100.0:.1f}%"

                    combined = int(update.combined_percent or 0)
                    self.compare_score_var.set(
                        f"正面：{_pct(update.front_score)}　"
                        f"侧面：{_pct(update.side_score)}　"
                        f"综合：{combined}%"
                    )
                    self.compare_error_var.set("")
                elif update.status in {"failed", "skipped", "cancelled"}:
                    self.compare_score_var.set("正面：-　侧面：-　综合：-")
                    detail = update.message
                    if update.error_code:
                        detail = f"{update.error_code}：{detail}"
                    self.compare_error_var.set(detail)
                else:
                    self.compare_error_var.set("")
            # 考试台账按 run_id 回填（面板或关窗后移交的台账，所有片段终态/过程态都要回填）；
            # 学生练习等后处理终态 + 最终视频路径再允许评判。关窗收尾时只回填台账。
            self.workflows.segment_update(update, route=not closing)

    def _sync_record_stop_enabled(self, state: RecordingState) -> None:
        """根据录制状态联动「结束录制」按钮的可用性：recording/paused 启用，idle 禁用。"""
        record_stop_btn = getattr(self, "record_stop_btn", None)
        if record_stop_btn is not None:
            try:
                record_stop_btn.configure(
                    state="normal" if state in ("recording", "paused") else "disabled"
                )
            except Exception:
                pass

    def _refresh_recording_status(self) -> None:
        """周期性（由 _tick 每 tick 调用）刷新录制状态文本与错误提示。

        - recording/paused：在 recording_status_var 显示状态文本与 Result_Video
          的完整保存路径（需求 5.9）。
        - idle：清除录制状态文本与路径（需求 5.10）。
        - last_error 非空：弹出包含失败原因的错误提示、复位录制按钮文本到
          「开始录制」、清除录制文本（需求 5.11）。用 _record_error_shown 守卫，
          避免每 30ms 重复弹框。
        """
        if getattr(self, "_closing", False):
            return

        recorder = self._recording.snapshot()
        multi = len(recorder.views) > 1
        errors: list[str] = []
        while self._pending_record_errors:  # 逐条取出，worker 可能同时追加
            errors.append(self._pending_record_errors.pop(0))
        for view, error in recorder.errors.items():
            text = f"{'第二路' if view == 'side' else '第一路'}：{error}"
            if text not in errors:
                errors.append(text)

        if errors:
            if not self._record_error_shown:
                self._record_error_shown = True
                # 双摄录制必须保持成对状态。任一路 writer 失败时结束各路当前片段并提交，
                # 双摄的失败由结果 JSON 与比对状态栏承载，单摄弹框提示。
                self._recording.finish()
                record_btn = getattr(self, "record_btn", None)
                if record_btn is not None:
                    record_btn.configure(text=RECORD_BTN_TEXT["idle"])
                self._sync_record_stop_enabled("idle")
                self.recording_status_var.set("")
                if not multi:
                    messagebox.showerror("录制失败", "录制发生错误：\n" + "\n".join(errors))
            return

        # 无错误：复位守卫，下次错误可再次提示。
        self._record_error_shown = False

        if recorder.state in ("recording", "paused"):
            texts = [
                str(snap.result_path) if snap.result_path is not None else "（准备中）"
                for snap in recorder.views.values()
            ]
            path_text = f"正面 {texts[0]}；侧面 {texts[1]}" if multi else texts[0]
            label = "录制中" if recorder.state == "recording" else "已暂停"
            self.recording_status_var.set(f"{label}：{path_text}")
        else:  # idle
            self.recording_status_var.set("")
        # 「结束录制」按钮可用性跟随真实状态联动（覆盖 worker 端错误复位等情形）。
        self._sync_record_stop_enabled(recorder.state)

    def _set_running_controls(
        self, running: bool, *, recording_ready: bool | None = None
    ) -> None:
        """集中管理运行态控件的 enable/disable 与文本联动（需求 2.5、3.5、3.6、4.3、4.5、5.1、5.2、6.4、6.5）。

        运行中（running=True）：
          - 禁用 Camera_Selector（需求 2.5）与刷新控件（经 _set_refresh_enabled）。
          - 禁用 Model_Selector（需求 3.5）。
          - 启用 Record_Toggle 并置文本「开始录制」（需求 5.2）。
          - Start_Control 切换为「停止」语义：兼容现有分离的 start/stop 双按钮——禁用
            start_btn、启用 stop_btn，并将 start_btn 文本置为「停止」（需求 4.3）。
        未运行（running=False）：
          - 恢复 Camera_Selector 为 readonly（仅当存在可选摄像头列表时）与刷新控件。
          - 恢复 Model_Selector 为 readonly（需求 3.6）。
          - 禁用 Record_Toggle（需求 5.1）；录制状态由 worker 的 close_session 复位为 idle，
            此处不调用控制器方法以免误触发。
          - Start_Control 恢复为「开始」：启用 start_btn、禁用 stop_btn。
          - Status_Area 显示「就绪」（需求 4.5）。
        Compare_Control 始终保持 enabled（需求 6.4、6.5）。

        record_btn / model_combo / compare_btn 由任务 9.x 在 _build_ui 中创建，此处用
        getattr 守卫，保证在它们尚未创建时方法仍可安全调用。
        """
        # Camera_Selector：运行中禁用；未运行时仅当有可选摄像头列表才恢复 readonly。
        camera_combo = getattr(self, "camera_combo", None)
        if camera_combo is not None:
            try:
                if running:
                    camera_combo.configure(state="disabled")
                elif self._camera_entries:
                    camera_combo.configure(state="readonly")
            except Exception:
                pass

        # 第二摄像头下拉：与主摄像头下拉同步禁用/恢复（issue #57）。
        camera_combo_2 = getattr(self, "camera_combo_2", None)
        if camera_combo_2 is not None:
            try:
                if running:
                    camera_combo_2.configure(state="disabled")
                elif self._camera_entries:
                    camera_combo_2.configure(state="readonly")
            except Exception:
                pass

        # 旋转：已去掉下拉，仅点击预览 +90°，无控件需联动。

        # 骨架开关仍在会话启动时锁定；自动比对滑块运行中可快速切换。
        skeleton_check = getattr(self, "record_skeleton_check", None)
        if skeleton_check is not None:
            try:
                skeleton_check.configure(state="disabled" if running else "normal")
            except Exception:
                pass
        auto_compare_check = getattr(self, "auto_compare_check", None)
        if auto_compare_check is not None:
            try:
                exam_locked = bool(getattr(self, "_exam_manual_locked", False))
                auto_compare_check.configure(
                    state="disabled" if exam_locked else "normal"
                )
            except Exception:
                pass

        # 刷新控件：复用既有联动（枚举中 / 运行中禁用）。
        self._set_refresh_enabled()

        # Model_Selector：运行中禁用、未运行恢复 readonly。
        model_combo = getattr(self, "model_combo", None)
        if model_combo is not None:
            try:
                model_combo.configure(state="disabled" if running else "readonly")
            except Exception:
                pass

        # Record_Toggle：运行中启用并置「开始录制」；未运行禁用。
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None:
            try:
                if recording_ready is None:
                    recording_ready = bool(
                        getattr(self, "_dual_recording_ready", True)
                    )
                if running and recording_ready:
                    record_btn.configure(state="normal", text=RECORD_BTN_TEXT["idle"])
                else:
                    record_btn.configure(state="disabled")
            except Exception:
                pass

        # 「结束录制」：会话开始时尚无录制片段，故初始禁用；未运行同样禁用。
        # 录制开始后由 _on_record_toggle / _refresh_recording_status 联动启用。
        sync_record_stop = getattr(self, "_sync_record_stop_enabled", None)
        if sync_record_stop is not None:
            sync_record_stop("idle")

        # Compare_Control：始终 enabled。
        compare_btn = getattr(self, "compare_btn", None)
        if compare_btn is not None:
            try:
                compare_btn.configure(state="normal")
            except Exception:
                pass

        # Start_Control / stop_btn：兼容现有分离双按钮的同时切换 start_btn 文本。
        start_btn = getattr(self, "start_btn", None)
        if start_btn is not None:
            try:
                if running:
                    start_btn.configure(state="disabled", text="停止")
                else:
                    start_btn.configure(state="normal", text="开始")
            except Exception:
                pass
        stop_btn = getattr(self, "stop_btn", None)
        if stop_btn is not None:
            try:
                stop_btn.configure(state="normal" if running else "disabled")
            except Exception:
                pass

        # Status_Area：未运行显示「就绪」（需求 4.5）。
        if not running:
            self.status_var.set("就绪")

    # ---- 摄像头枚举与选择 ----

    def _set_refresh_enabled(self) -> None:
        """仅当非枚举中且采集未运行时启用刷新控件（需求 5.3、5.7）。"""
        running = App._session_active(self)
        enabled = (not self._enum_busy.is_set()) and (not running)
        try:
            self.refresh_btn.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass

    def _start_enumeration(self) -> None:
        """在后台线程枚举可用摄像头，结果经 root.after 回写 UI（需求 1.1、5.2）。"""
        if self._enum_busy.is_set():
            return
        self._enum_busy.set()
        self._set_refresh_enabled()
        self.camera_combo.configure(state="disabled")

        def _deliver(snapshot) -> None:
            # 后台线程只投递结果，由 _tick 在主线程应用；兜底探测跳过池占用的设备（V-6）。
            if getattr(self, "_closing", False):
                return
            result_queue = self._camera_enum_result_queue
            while True:
                try:
                    result_queue.get_nowait()
                except Empty:
                    break
            result_queue.put((list(snapshot.devices), snapshot.ok))

        self._camera_service.refresh(on_done=_deliver)

    def _drain_camera_enum_results(self) -> None:
        """在 Tk 主线程应用最新枚举结果，关窗后只丢弃不触碰控件。"""
        result_queue = getattr(self, "_camera_enum_result_queue", None)
        if result_queue is None:
            return
        latest: tuple[list[CameraEntry], bool] | None = None
        while True:
            try:
                latest = result_queue.get_nowait()
            except Empty:
                break
        if latest is not None and not getattr(self, "_closing", False):
            self._apply_camera_entries(*latest)

    def _apply_camera_entries(self, entries: list[CameraEntry], ok: bool) -> None:
        """枚举回调（主线程）：重建下拉项并更新输入源状态。

        需求 1.8、2.2、2.5、2.6、5.4、5.5、5.6。
        """
        try:
            if not ok:
                # 探测失败/超时：保留刷新前的列表与下拉项不变（需求 5.6）。
                self.status_var.set("刷新摄像头失败，已保留原有列表。")
                if self._camera_entries:
                    self.camera_combo.configure(state="readonly")
                self._sync_camera_warmup()
                return

            self._camera_entries = list(entries)
            if not entries:
                # 空列表：禁用下拉并提示，不记录输入源（需求 1.8、2.6、5.5）。
                self.camera_combo.configure(values=[], state="disabled")
                self.camera_choice_var.set("")
                self.source_hint_var.set("未检测到可用摄像头")
                if self._source_state.kind == "camera":
                    self._source_state.clear()
                    self.source_var.set("")
                self._release_camera_warmups()
                self.camera_combo_2.configure(values=[NO_SECOND_CAMERA], state="disabled")
                self.camera_choice_var_2.set(NO_SECOND_CAMERA)
                return

            labels = [e.label for e in entries]
            self.camera_combo.configure(values=labels, state="readonly")
            # 偏好设备暂时缺失时只做 UI 回退，不覆盖持久化值；设备恢复后可自动选回。
            current = self.camera_choice_var.get()
            preferred = self._camera_label_for_index(
                self._preferred_primary_camera_index
            )
            if preferred is not None:
                current = preferred
            elif current not in labels:
                current = labels[0]
            self.camera_choice_var.set(current)

            # 第二摄像头下拉：同步为「无」+ 真实摄像头列表；已选项不在新列表中则回退「无」。
            labels_2 = [NO_SECOND_CAMERA] + labels
            self.camera_combo_2.configure(values=labels_2, state="readonly")
            current_2 = self.camera_choice_var_2.get()
            preferred_2 = self._camera_label_for_index(
                self._preferred_secondary_camera_index
            )
            if preferred_2 is not None:
                current_2 = preferred_2
            elif current_2 not in labels_2:
                current_2 = NO_SECOND_CAMERA
            self.camera_choice_var_2.set(current_2)
            # 两个选择都稳定后再统一同步预热，避免枚举刷新过程中先按旧的第二路选择
            # 启动一次无效预热。
            self._select_camera_by_label(current)
        finally:
            self._enum_busy.clear()
            self._set_refresh_enabled()

    def _camera_index_for_label(self, label: str) -> int | None:
        """由显示文本反查摄像头编号，找不到返回 None。"""
        for e in self._camera_entries:
            if e.label == label:
                return e.index
        return None

    def _camera_label_for_index(self, index: int | None) -> str | None:
        if index is None:
            return None
        for entry in self._camera_entries:
            if entry.index == index:
                return entry.label
        return None

    def _persist_camera_selection(self) -> None:
        save_camera_selection(
            self._preferred_primary_camera_index,
            self._preferred_secondary_camera_index,
        )

    def _select_camera_by_label(self, label: str) -> None:
        """由显示文本反查编号并记录为摄像头输入源（需求 2.3、3.3）。

        唯一收敛点——`_on_camera_selected` 与枚举自动选中的 `_apply_camera_entries`
        都经此。单摄继续使用既有预打开；选中第二路时改由双路 pool 并发预热。
        """
        index = self._camera_index_for_label(label)
        if index is not None:
            self._source_state.select_camera(index)
            self.source_var.set(str(index))
            self.source_hint_var.set(self._source_state.hint_text())
            self._sync_camera_warmup()

    def _on_camera_selected(self, event=None) -> None:
        label = self.camera_choice_var.get()
        index = self._camera_index_for_label(label)
        if index is None:
            return
        self._preferred_primary_camera_index = index
        self._select_camera_by_label(label)
        self._persist_camera_selection()

    def _on_camera_2_selected(self, event=None) -> None:
        label = self.camera_choice_var_2.get()
        index = self._camera_index_for_label(label)
        if label != NO_SECOND_CAMERA and index is None:
            return
        self._preferred_secondary_camera_index = index
        self._persist_camera_selection()
        self._sync_camera_warmup()

    def _refresh_cameras(self) -> None:
        """刷新可用摄像头列表（需求 5.1、5.3、5.7）。"""
        if App._session_active(self):
            return
        if self._enum_busy.is_set():
            return
        # V-6：刷新不放弃预热；只读设备名，兜底探测跳过本程序占用的设备。
        self._start_enumeration()

    def _release_camera_warmups(self) -> None:
        """放弃所有预热 reader；释放在后台完成，重开同编号时由池等待真实释放。"""
        service = getattr(self, "_camera_service", None)
        if service is None:
            return
        try:
            service.prepare({})
        except Exception:
            pass

    def _camera_capture_size(self) -> tuple[int, int]:
        """当前模式的请求尺寸；只在主线程构造 spec 时读取一次，随后冻结。"""
        return self.modes.current.requirements.fmt.size

    def _camera_capture_policy(self) -> str:
        """学生练习严格要求 1080p，驱动给不出时失败并提示（D-6）；其余模式接受实际尺寸。"""
        return self.modes.current.requirements.fmt.policy

    def _camera_names(self) -> tuple[tuple[int, str], ...]:
        """枚举得到的设备名，供打开时复核（V-3）；没有名称的设备不复核。"""
        return tuple((e.index, e.name) for e in self._camera_entries if getattr(e, "name", None))

    def _camera_spec(self, index: int) -> CaptureSpec:
        width, height = App._camera_capture_size(self)
        return CaptureSpec(
            int(index),
            width,
            height,
            expected_name=dict(App._camera_names(self)).get(int(index)),
            policy=App._camera_capture_policy(self),
        )

    def _selected_camera_indices(self) -> tuple[int | None, int | None]:
        primary = None
        if self._source_state.kind == "camera" and self._source_state.value:
            try:
                primary = int(self._source_state.value)
            except (TypeError, ValueError):
                primary = None
        secondary_label = self.camera_choice_var_2.get().strip()
        secondary = (
            self._camera_index_for_label(secondary_label)
            if secondary_label and secondary_label != NO_SECOND_CAMERA
            else None
        )
        if primary is None or secondary == primary:
            secondary = None
        return primary, secondary

    def _sync_camera_warmup(self, *, allow_running: bool = False) -> None:
        """按两个下拉框和当前模式声明预热需求；配置一致则复用，不一致由池重新协商。"""
        if getattr(self, "_closing", False):
            return
        running = App._session_active(self)
        if running and not allow_running:
            return
        service = getattr(self, "_camera_service", None)
        if service is None:
            return
        primary, secondary = App._selected_camera_indices(self)
        specs = {
            role: App._camera_spec(self, index)
            for role, index in ((PRIMARY, primary), (SECONDARY, secondary))
            if index is not None
        }
        try:
            service.prepare(specs)
        except Exception as exc:
            App._release_camera_warmups(self)
            self.status_var.set(f"摄像头预热失败：{exc}")

    def _collect_state(self) -> UiState:
        # 无有效输入源：使用统一提示文案（需求 4.2）。
        if self._source_state.kind == "none":
            raise ValueError("请先选择摄像头或视频")
        source = self.source_var.get().strip()
        if not source:
            raise ValueError("请先选择摄像头或视频")

        workers = clamp_workers(self.workers_var.get() or 1)

        # 第二摄像头（可选，双摄双面视图，issue #57/#58）：未选、选不到真实条目、或主输入源
        # 不是摄像头（视频文件模式）→ source2=None，即今天的单摄/文件行为不变。双摄预览只在
        # 主输入源也是摄像头时才有意义，这里是唯一收敛点：调用方（_start/_session_config）
        # 只需检查 state.source2 是否非空，不必重复判断 source 是否为摄像头。
        source2: str | None = None
        camera_choice_var_2 = getattr(self, "camera_choice_var_2", None)
        label_2 = camera_choice_var_2.get().strip() if camera_choice_var_2 is not None else ""
        if source.isdigit() and label_2 and label_2 != NO_SECOND_CAMERA:
            index_2 = self._camera_index_for_label(label_2)
            if index_2 is not None:
                if int(source) == index_2:
                    raise ValueError("两个摄像头不能选同一个")
                source2 = str(index_2)

        # 保留旧的 save_var/out_var 测试桩与外部嵌入兼容；当前主窗口已由
        # RecordingController 接管录制，没有这两个控件时安全回退为 None。
        save_var = getattr(self, "save_var", None)
        out_var = getattr(self, "out_var", None)
        out_path = None
        if save_var is not None and out_var is not None and bool(save_var.get()):
            out_path = out_var.get().strip() or None

        return UiState(
            source=source,
            pose_variant=self.pose_var.get().strip() or "full",
            workers=workers,
            enable_hands=bool(self.enable_hands_var.get()),
            out_path=out_path,
            source2=source2,
            online_match_enabled=(
                bool(self.online_match_var.get())
                if hasattr(self, "online_match_var")
                else True
            ),
            rotate=_parse_rotate(self.rotate_var.get()) if hasattr(self, "rotate_var") else 0,
            rotate2=_parse_rotate(self.rotate_var_2.get()) if hasattr(self, "rotate_var_2") else 0,
            # 全局骨架开关：单摄/双摄共用；默认关 → 预览与录制均不叠骨架。
            record_skeleton=(
                bool(self.record_skeleton_var.get())
                if hasattr(self, "record_skeleton_var")
                else False
            ),
            auto_compare=(
                bool(self.auto_compare_var.get())
                if source2 is not None and hasattr(self, "auto_compare_var")
                else True
            ),
            capture_size=App._camera_capture_size(self),
            capture_policy=App._camera_capture_policy(self),
            camera_names=App._camera_names(self),
        )

    def _open_camera_process(
        self,
        index: int,
        *,
        width: int,
        height: int,
        stop_event: threading.Event,
        expected_name: str | None = None,
        policy: str = "prefer",
    ):
        """池的唯一打开入口：每路一个 spawn 采集进程，调用时解析模块级 ProcessCamera。"""
        return ProcessCamera(
            index, width=width, height=height, stop_event=stop_event,
            expected_name=expected_name, policy=policy,
        )

    @staticmethod
    def _session_spec(state: UiState, source: str) -> CaptureSpec:
        width, height = getattr(state, "capture_size", _PREVIEW_CAPTURE_SIZE)
        return CaptureSpec(
            int(source),
            int(width),
            int(height),
            expected_name=dict(getattr(state, "camera_names", ())).get(int(source)),
            policy=getattr(state, "capture_policy", "prefer"),
        )

    # ---- 实时会话（core.live）：配置、启动/停止、事件与预览 ----

    def _build_live_manager(self) -> LiveSessionManager:
        """组装会话管理器：摄像头经 CameraService，录制经 RecordingService，站位判定用髋中点 ROI。"""
        return LiveSessionManager(
            make_input=lambda config: App._make_session_input(self, config),
            on_event=lambda event: App._on_session_event(self, event),
            controls=self._live_controls,
            sink=_RecordingSink(self),
            probe=hip_midpoint_in_roi,
            preview_target=lambda role: self._preview_wh2 if role == SECONDARY else self._preview_wh,
        )

    def _make_session_input(self, config: SessionConfig) -> FrameInput:
        if config.is_file:
            return VideoFileInput(config.video_path)
        return CameraInput(self._camera_service.open_stream(config.cameras))

    @staticmethod
    def _session_config(state: UiState) -> SessionConfig:
        """把开始时冻结的界面状态换成会话配置；四种循环的差异由 §8.2 的表决定。"""
        if not state.source.isdigit():
            return video_session(
                state.source, pose_variant=state.pose_variant, enable_hands=state.enable_hands,
                workers=state.workers,
            )
        cameras = {PRIMARY: App._session_spec(state, state.source)}
        if state.source2:
            cameras[SECONDARY] = App._session_spec(state, state.source2)
        return camera_session(
            cameras, pose_variant=state.pose_variant, enable_hands=state.enable_hands,
            workers=state.workers, skeleton=bool(state.record_skeleton),
            online_match=bool(state.online_match_enabled),
        )

    def _session_active(self) -> bool:
        """有会话（含启动中与收尾中）。"""
        return self._live.active()

    def _session_live(self) -> bool:
        """会话已开始且没有在停止：录制、站位检测只在这时接受操作。"""
        return self._live.state in (STARTING, RUNNING)

    def _session_finished(self) -> bool:
        """会话已结束且结束事件已在 Tk 线程处理完。"""
        return not self._live.active() and self._live_events.empty()

    def _start(self) -> None:
        if App._session_active(self):
            self.status_var.set("上一会话仍在收尾，请稍候再点「开始」")
            return
        start_click = time.monotonic()

        try:
            # 会话配置 = 用户设置快照叠加当前模式的要求（§8.5），在开始时计算一次
            state = self.modes.session_settings(self._collect_state())
            config = App._session_config(state)
        except Exception as e:
            messagebox.showerror("配置错误", str(e))
            return
        if config.is_file:
            # 视频文件会话：先放弃选择阶段持有的摄像头预热。
            self._release_camera_warmups()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("启动中…（首次运行可能需要下载模型）")
        self.actions_var.set("-")
        self.match_var.set("识别：待机")
        self.progress_var.set(0.0)
        self.progress_text_var.set("")
        self.progress_bar.configure(mode="determinate", maximum=100.0, value=0.0)
        # 双摄双面视图（issue #58）：config.dual ⇔ 两路都是摄像头。
        self._set_dual_preview_visible(config.dual)
        # 启动时把界面旋转同步到运行时控制，供点击预览与会话线程共用。
        for which, degrees in (("primary", state.rotate), ("secondary", state.rotate2)):
            App._set_runtime_rotate(self, int(degrees or 0), which=which, sync_combo=True, announce=False)
        try:
            self._dual_auto_compare = App._current_auto_compare(self)
        except Exception:
            self._dual_auto_compare = bool(state.auto_compare)
        self._dual_recording_ready = not config.dual
        self._dual_record_skeleton = bool(state.record_skeleton)
        self._shown_preview_seq = 0

        token = self._live.start(config, start_click=start_click)
        self.workflows.session_started(token)
        foundation.diag.event(
            "session.start",
            session=token,
            primary=state.source,
            secondary=state.source2,
            requested="x".join(str(v) for v in state.capture_size),
            student=self.modes.is_active("practice"),
        )
        self._set_refresh_enabled()
        # 集中刷新运行态控件（禁用 Camera/Model_Selector、启用 Record_Toggle 等，需求 4.3 等）。
        self._set_running_controls(True)

    def _stop(self) -> None:
        # 先同步工作流：否则 occupancy 退出后考试仍停在 recording，指针/手动锁不解除；练习停止等待到位
        try:
            self.workflows.session_stopped(time.monotonic())
        except Exception:
            traceback.print_exc()
        App.set_exam_mode(self, False)
        foundation.diag.event("session.stop_requested", session=self._live.token, cancel="user_stop")
        self._live.stop("user_stop")
        self._dual_recording_ready = False
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None:
            try:
                record_btn.configure(state="disabled")
            except Exception:
                pass
        self.stop_btn.configure(state="disabled")
        self.status_var.set("正在停止…")

    def _on_session_event(self, event: SessionEvent) -> None:
        """会话线程回调：只入队，由 _tick 在 Tk 线程处理。会话终态同时写诊断行。"""
        if event.kind == "ended":
            outcome = event.payload
            foundation.diag.event(
                "session.end", session=event.token, status=outcome.status, rewarm=not outcome.failed,
            )
        self._live_events.put(event)

    def _drain_session_events(self) -> None:
        handlers = {
            "status": App._on_session_status,
            "progress": App._on_session_progress,
            "match": App._on_session_match,
            "presence": App._on_session_presence,
            "running": App._on_session_running,
            "ended": App._on_session_ended,
        }
        while True:
            try:
                event = self._live_events.get_nowait()
            except Empty:
                return
            if self._closing or not self._live.is_current(event.token):
                continue
            handlers[event.kind](self, event.payload)

    def _on_session_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_session_progress(self, progress: tuple[int, int]) -> None:
        done, total = progress
        if total > 0:
            self.progress_bar.configure(mode="determinate", maximum=float(total))
            self.progress_bar["value"] = float(done)
            self.progress_text_var.set(f"进度：{done}/{total}（{(done / total) * 100.0:.1f}%）")
        else:
            # 摄像头没有总帧数：不显示进度。
            self.progress_bar.configure(mode="determinate", maximum=100.0)
            self.progress_bar["value"] = 0.0
            self.progress_text_var.set("")

    def _on_session_match(self, text: str) -> None:
        if App._session_live(self):
            self.match_var.set(text)

    def _on_session_presence(self, sample: PresenceSample) -> None:
        """站位采样：会话未在停止时交给当前激活的工作流（考试或学生练习）。"""
        if not App._session_live(self):
            return
        self.workflows.presence(self._live.token, bool(sample.present), float(sample.sampled_at))

    def _on_session_running(self, _info: RunningInfo) -> None:
        """会话已接管设备并登记录制：双摄此时才允许录制，学生练习据此启用站位检测。"""
        if not self._live.config.dual or not App._session_live(self):
            return
        self._dual_active = True
        self._dual_auto_compare = self._current_auto_compare()
        self._dual_recording_ready = True
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None and not self.modes.is_active("practice"):
            record_btn.configure(state="normal", text=RECORD_BTN_TEXT["idle"])
        # 学生练习：双摄就绪只启用站位检测，到位事件才开录。
        self.workflows.session_ready(self._live.token, time.monotonic())

    def _on_session_ended(self, outcome: SessionOutcome) -> None:
        """会话终态（设备已归还或已隔离）：复位界面；未失败时重新声明预热。"""
        config = self._live.config
        self._dual_recording_ready = False
        self._dual_active = False
        # 开考后主会话结束：考试中止；学生练习回到可重新「开始」
        self.workflows.session_ended(self._live.token, time.monotonic(), status=outcome.status or "")
        App.set_exam_mode(self, False)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_refresh_enabled()
        # 会话结束：集中复位运行态控件并使 Status_Area 显示「就绪」（需求 4.5、5.1）。
        self._set_running_controls(False)
        if outcome.status:
            self.status_var.set(outcome.status)
        # 新 reader 若遇到仍在退出的旧进程，会在后台等待真实释放后再打开，Tk 线程不等待驱动。
        if not outcome.failed or (config is not None and config.is_file):
            try:
                self._sync_camera_warmup(allow_running=True)
            except Exception:
                pass

    def _render_preview(self) -> None:
        """每 tick 显示比已显示更新的一帧预览；双摄按两路画面方向选择布局。"""
        store = self._live.preview
        if store is None or not App._session_live(self):
            return
        frame = store.take_newer(self._shown_preview_seq)
        if frame is None:
            return
        self._shown_preview_seq = frame.seq
        self.actions_var.set(frame.actions_text)
        self._photo = ImageTk.PhotoImage(Image.fromarray(frame.images[PRIMARY]))
        self.preview.configure(image=self._photo)
        dual = SECONDARY in frame.images
        if dual:
            self._photo2 = ImageTk.PhotoImage(Image.fromarray(frame.images[SECONDARY]))
            self.preview2.configure(image=self._photo2)
            layout = _choose_dual_preview_layout(frame.sizes[PRIMARY], frame.sizes[SECONDARY])
            if layout != getattr(self, "_dual_preview_layout", None):
                self._set_dual_preview_layout(layout)
        first = store.mark_rendered(frame)
        metrics = store.metrics
        if dual and first and metrics is not None:
            # 双摄启动耗时提示：首个裸帧与首个标注帧各显示一次。
            elapsed_ms = metrics.elapsed_ms()
            if frame.stage == "raw":
                loading = "，模型加载中…" if metrics.inference else ""
                self.status_var.set(f"双摄画面已显示（{elapsed_ms:.0f} ms）{loading}")
            else:
                self.status_var.set(f"运行中…（双摄首个标注帧 {elapsed_ms:.0f} ms）")

    def _on_close(self) -> None:
        if self._closing:
            return
        # 1) 先中止考试、移交当前面板 scorebook 到 sinks（_closing 仍为 False，discard 可走）
        #    root.destroy 不会触发面板 WM_DELETE_WINDOW，必须 prepare_for_app_close。
        panel = self.exam_panel
        if panel is not None:
            try:
                panel.prepare_for_app_close()
            except Exception:
                pass
        elif self.modes.is_active("exam"):
            try:
                App._end_recording_segment(self, discard=True)
            except Exception:
                pass
        if hasattr(self, "feedback_controls"):
            self.feedback_controls.close()
        if self._practice is not None:
            # 作废分析代次并停止等待到位
            self._practice_dispatch("analysis_cancelled")
            self._practice_dispatch("session_stopped", now=time.monotonic())
        announcer = getattr(self, "_student_announcer", None)
        if announcer is not None:
            announcer.close()
            self._student_announcer = None
        # 2) 再进入关闭态
        self._closing = True
        self._live.stop(APP_CLOSE)
        self._dual_recording_ready = False
        App._release_camera_warmups(self)
        for name in (
            "record_btn",
            "record_stop_btn",
            "auto_compare_check",
            "record_skeleton_check",
            "record_dir_entry",
            "record_dir_btn",
        ):
            control = getattr(self, name, None)
            if control is not None:
                try:
                    control.configure(state="disabled")
                except Exception:
                    pass
        # 考试裁剪、已移交台账（含刚移交的面板台账）的回填与落盘需要更长预算
        scorebooks = self.exam_scorebooks
        extra = self._recording.finalizer.close_budget_s()
        if scorebooks.has_detached:
            extra = max(extra, 8.0)
        if scorebooks.has_processing:
            extra = max(extra, 12.0)
        self._close_deadline = time.monotonic() + _CLOSE_JOIN_TIMEOUT_S + extra

        def _prepare_close() -> None:
            try:
                # 考试 abort 已在 _closing 前 discard；此处只收尾仍在录的双摄片段，再取消后处理队列。
                if bool(getattr(self, "_dual_active", False)):
                    self._recording.finish()
                self._recording.cancel_all()
            finally:
                service = getattr(self, "_camera_service", None)
                if service is not None:
                    try:
                        # 关窗预算内等待采集进程真实退出；超时由诊断日志记录。
                        service.close(_CAMERA_RELEASE_WAIT_S)
                    except Exception:
                        pass

        self._close_prepare_worker = threading.Thread(
            target=_prepare_close,
            name="app-close-prepare",
            daemon=True,
        )
        self._close_prepare_worker.start()
        self._poll_close_workers()

    def _poll_close_workers(self) -> None:
        """分片等待关窗准备与会话线程、等待录后处理空闲；全部结束或到达期限后销毁 Tk 窗口。"""
        deadline = self._close_deadline or time.monotonic()
        remaining = deadline - time.monotonic()
        close_prepare_worker = getattr(self, "_close_prepare_worker", None)
        current = threading.current_thread()
        if (
            close_prepare_worker is not None
            and close_prepare_worker is not current
            and close_prepare_worker.is_alive()
            and remaining > 0.0
        ):
            close_prepare_worker.join(
                timeout=min(_CLOSE_JOIN_SLICE_S, remaining)
            )

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            self._live.wait_idle(min(_CLOSE_JOIN_SLICE_S, remaining))

        close_prepare_alive = bool(
            close_prepare_worker is not None and close_prepare_worker.is_alive()
        )
        # 会话线程收尾时会提交最后的片段；决定销毁前重新查看 finalizer 是否空闲。
        session_alive = not self._live.wait_idle(0)
        finalizer_busy = not self._recording.finalizer.idle

        if (
            close_prepare_alive
            or session_alive
            or finalizer_busy
        ) and time.monotonic() < deadline:
            try:
                self.root.after(_CLOSE_POLL_MS, self._poll_close_workers)
            except (TclError, RuntimeError):
                pass
            return

        # 最后一轮消费比对更新，并把所有考试台账落盘关闭
        try:
            App._drain_recording_compare_updates(self)
        except Exception:
            pass
        self.exam_scorebooks.close_all()

        try:
            self.root.destroy()
        except (TclError, RuntimeError):
            pass

    def _tick(self) -> None:
        self._drain_recording_compare_updates()
        self._drain_session_events()
        self.workflows.tick(time.monotonic())
        self._drain_camera_enum_results()
        # 录制状态刷新必须每 tick 执行，与是否有新帧无关（需求 5.9/5.10/5.11）。
        self._refresh_recording_status()
        self._render_preview()
        self.root.after(16, self._tick)


def _model_installed(key: str) -> bool:
    """按模型键（如 ``pose_lite``）查询 MediaPipe 模型是否已安装。"""
    spec = next((m for m in model_manager.MEDIAPIPE_MODELS if m.key == key), None)
    return spec is not None and model_manager.is_installed(spec)


class _SegmentPort:
    """工作流的录制端口（``SegmentControl``）：开录/结束接到 App 的录制服务。"""

    def __init__(self, app: App) -> None:
        self._app = app

    def start_segment(self, payload: dict):
        return App._begin_recording_segment(self._app, SegmentIntent(exam=payload.get("exam")))

    def finish(self, *, discard: bool) -> None:
        App._end_recording_segment(self._app, discard=discard)


class _OccupancyPort:
    """工作流的站位检测端口（``OccupancyControl``）：按所有者武装会话内的站位检测。"""

    def __init__(self, app: App, owner: str) -> None:
        self._app = app
        self._owner = owner

    def arm(self, armed: bool) -> None:
        self._app._live_controls.arm_occupancy(self._owner, bool(armed))

    def set_roi(self, roi) -> None:
        self._app._live_controls.set_roi(roi)


class _PracticeUi:
    """学生练习的界面端口（``WorkflowUi``）：状态栏、提示框、结果页与按钮联动。"""

    def __init__(self, app: App) -> None:
        self._app = app

    def status(self, text: str) -> None:
        self._app.student_status_var.set(text)

    def alert(self, title: str, text: str, *, error: bool = False) -> None:
        (messagebox.showerror if error else messagebox.showinfo)(title, text)

    def lock_manual_record(self, locked: bool) -> None:
        pass

    def show_result(self, record: dict) -> None:
        self._app._show_student_result_view(record)

    def refresh_rows(self) -> None:
        pass

    def refresh(self) -> None:
        self._app._sync_student_buttons()


class _PracticeAnalysis:
    """学生练习的分析端口（``AnalysisControl``）：后台分析并保存，保存后回到练习流程。"""

    def __init__(self, app: App) -> None:
        self._app = app

    def start_practice_analysis(self, request: dict) -> bool:
        app, token, student_id = self._app, request["token"], request["student_id"]

        def on_saved(record):
            app._practice_dispatch("record_saved", record_id=record["id"])
            App._on_student_analysis_saved(app, token=token, student_id=student_id, record=record)

        return app.feedback_controls.analyze_recording(
            request["identity"], request["front"], request["side"],
            record_id=request["record_id"], on_saved=on_saved, token=token,
        )


class _RecordingSink:
    """``core.live.FrameSink`` 的 Tk 端实现：转给 App 的 ``RecordingService``，会话结束走 App 的收尾。"""

    def __init__(self, app: App) -> None:
        self._app = app

    def begin_session(self, sizes, fps: float) -> None:
        self._app._recording.begin_session(sizes, fps)

    def update_sizes(self, sizes) -> None:
        self._app._recording.update_sizes(sizes)

    def write(self, frames) -> None:
        self._app._recording.write(frames)

    def close_session(self) -> None:
        App._close_recording_session(self._app)


def main() -> None:
    # 离线包首次启动释放内置模型到用户数据目录，源码运行保持原路径。
    models_dir()
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
