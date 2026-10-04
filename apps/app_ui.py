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

from core.action_compare import compare_video_to_templates, create_template_from_video
from analysis.tech_eval import to_jsonable
from core.vision_pipeline import MediaPipePipeline, PipelineConfig, draw_pose_frame
from core.paths import (
    load_camera_selection,
    load_record_dir,
    models_dir,
    outputs_dir,
    save_camera_selection,
    save_record_dir,
)
from core import model_manager, online_matcher, paths, video_writer
from core import pose_features as pf
from core.parallel_pose_engine import ParallelPoseEngine, default_pipeline_factory
from core.preview_smoother import PreviewLandmarkSmoother
from core.recording_controller import RecordingController, RecordingState
from apps.camera_enum import CameraEntry, InputSourceState, enumerate_cameras
from apps.camera_capture import open_camera
from apps.camera_warmup import (
    CameraWarmupPool,
    CameraWarmupStopped,
    CaptureSpec,
    PRIMARY,
    SECONDARY,
)
from apps.feedback_panel import FeedbackControls
from apps.recording_postprocess import (
    DualRecordingJob,
    DualRecordingPostProcessor,
    PostprocessUpdate,
    default_template_paths,
    load_configured_template_lists,
    load_configured_template_paths,
    save_configured_template_lists,
    save_configured_template_paths,
    template_action_name,
    try_load_heavy_template,
    validate_auto_compare_template_lists,
    validate_template_pair,
)


ACTION_LABELS_ZH = {
    "V_SIGN": "✌（V 手势）",
    "HANDS_UP": "双手举起",
    "LEFT_HAND_UP": "左手举起",
    "RIGHT_HAND_UP": "右手举起",
    "SQUAT": "下蹲",
}


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
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


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


def _apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """按度数旋转帧；0（或非 90 倍数）时原样返回（不复制）。"""
    code = _ROTATE_CODES.get(degrees)
    return cv2.rotate(frame, code) if code is not None else frame


def _rotated_size(width: int, height: int, degrees: int) -> tuple[int, int]:
    """返回旋转后的 ``(width, height)``，供 writer 和布局使用。"""
    if degrees in (90, 270):
        return height, width
    return width, height


def _fit_frame_to_size(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """将帧缩放到 writer 固定尺寸；尺寸一致时原样返回。"""
    tw, th = int(size[0]), int(size[1])
    if tw <= 0 or th <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == tw and h == th:
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)


def _choose_dual_preview_layout(
    size: tuple[int, int], size2: tuple[int, int]
) -> str:
    """两路都为竖画面时左右并排；横向、方形或混合方向时上下堆叠。"""
    w, h = size
    w2, h2 = size2
    return "side_by_side" if h > w and h2 > w2 else "stacked"


def _select_dual_recording_frames(
    frame: np.ndarray,
    frame2: np.ndarray,
    annotated: np.ndarray,
    annotated2: np.ndarray,
    *,
    record_skeleton: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """会话级全局骨架开关：开启时录像写标注帧；关闭时写原始帧（与预览一致）。"""
    if record_skeleton:
        return annotated, annotated2
    return frame, frame2

# 关窗时给采集线程收尾并等待 H.264 转码的最长时间。等待拆成短 join 片段，
# 避免 Tk 主线程在单次回调里长时间无响应。
_CLOSE_JOIN_TIMEOUT_S = 3.0
_CLOSE_JOIN_SLICE_S = 0.05
_CLOSE_POLL_MS = 50
_CAMERA_CANCEL_JOIN_TIMEOUT_S = 0.05
# 覆盖 ProcessCamera 12 秒首帧预算与旧设备释放等待；接管只需停掉预热 reader。
_CAMERA_START_TIMEOUT_S = 15.0
_CAMERA_CLAIM_TIMEOUT_S = 5.0
_CAMERA_RELEASE_WAIT_S = 3.0
_PREVIEW_CAPTURE_SIZE = (1280, 720)
_STUDENT_CAPTURE_SIZE = (1920, 1080)
_DUAL_RAW_PUMP_INTERVAL_S = 0.05
_DUAL_STAGE_RANK = {"raw": 0, "annotated": 1}


def _log_camera_event(payload: dict) -> None:
    """摄像头生命周期诊断行；窗口版 stdout 由 desktop_launcher 重定向到 desktop.log。"""
    try:
        print("[camera] " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception:
        pass


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
                with np.load(tpl_path, allow_pickle=True) as template_data:
                    template_meta = dict(template_data["meta"].item() or {})
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
                results = compare_video_to_templates(
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
    session_generation: int = 0
    start_click: float = 0.0


@dataclass(frozen=True)
class DualPreviewPacket:
    session_generation: int
    frame_rgb: np.ndarray
    actions: str
    frame_rgb2: np.ndarray
    actions2: str
    stage: str
    enqueued_at: float


@dataclass
class _DualStartupMetrics:
    session_generation: int
    primary_index: int
    secondary_index: int
    record_skeleton: bool
    start_click: float
    pair_ready: float | None = None
    first_pair_enqueued: float | None = None
    first_pair_rendered: float | None = None
    pipeline_ready: float | None = None
    first_annotated_enqueued: float | None = None
    first_annotated_rendered: float | None = None
    outcome: str | None = None
    error_type: str | None = None
    emitted: bool = False


@dataclass(frozen=True)
class _RecordingPairFinalization:
    dual_active: bool
    stamp: str | None
    segment_dir: Path | None
    record_skeleton: bool
    auto_compare: bool
    segment_started: bool
    front_path: Path | None
    side_path: Path | None
    front_frames: int
    side_frames: int
    front_error: str | None
    side_error: str | None

    @property
    def has_activity(self) -> bool:
        return bool(
            self.segment_started
            or self.front_path
            or self.side_path
            or self.front_frames > 0
            or self.side_frames > 0
            or self.front_error
            or self.side_error
        )


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
        # 运行时旋转：主线程写、worker 每帧读；点击预览可顺时针 +90°，两路隔离。
        self._runtime_rotate_lock = threading.Lock()
        self._runtime_rotate = 0
        self._runtime_rotate2 = 0
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

        # 物理摄像头唯一所有者：单摄/双摄预热、会话接管和释放隔离都经此池。
        # 采集配置（编号+请求尺寸）在请求时冻结为 CaptureSpec，不在打开途中读模式标志。
        self._camera_warmup_pool = CameraWarmupPool(
            capture_factory=self._open_camera_process,
            join_timeout=_CAMERA_CANCEL_JOIN_TIMEOUT_S,
            log=_log_camera_event,
        )

        # H.264 转码不能依赖 daemon 线程碰运气完成；登记所有 worker，关窗时有界等待。
        self._transcode_lock = threading.Lock()
        self._transcode_workers: set[threading.Thread] = set()
        self._closing = False
        self._close_deadline: float | None = None
        self._close_prepare_worker: threading.Thread | None = None
        self._latest_compare_segment_id: str | None = None
        self._submitted_record_segments: set[str] = set()
        self._compare_update_queue: Queue[PostprocessUpdate] = Queue()
        self._record_postprocessor = DualRecordingPostProcessor(
            on_update=self._post_recording_compare_update
        )
        # 考试模式（见 docs/exam_system_design.md）
        self._exam_lock = threading.Lock()
        self._exam_active = False
        self._exam_occupancy_armed = False
        self._exam_roi: tuple[float, float, float, float] = (0.2, 0.1, 0.8, 0.95)
        self._exam_run_id: str | None = None
        self._exam_run_dir: Path | None = None
        self._exam_pending_row = None
        self._exam_discard_next = False
        self._exam_manual_locked = False
        self._exam_occupancy_queue: Queue[tuple[bool, float]] = Queue(maxsize=8)
        self._exam_occupancy_stride = 3
        # 关考试面板后仍接收后台比对回填的台账 sinks（可多场并存，禁止单例覆盖）
        self._exam_scorebook_sinks: list = []
        self._exam_clip_lock = threading.Lock()
        self._exam_clip_threads: set[threading.Thread] = set()

        # 学生练习模式（MediaPipe 关键点规则检查，见 docs/error_analysis_plan.md §C）
        self._student_practice_active = False
        self._student_pending_record = False
        self._student_presence_gate = None
        self._student_presence_started_at = 0.0
        self._student_announcer = None
        self._student_judging = False
        self._student_last_segment_dir: Path | None = None
        self._student_last_segment_id: str | None = None
        self._student_front_video: Path | None = None
        self._student_side_video: Path | None = None
        self._student_segment_ready = False  # 后处理终态且视频可读后才可评判
        self._student_judge_worker: threading.Thread | None = None
        self._student_saved_auto_compare: bool | None = None
        self._student_saved_skeleton: bool | None = None
        self._student_saved_hands: bool | None = None
        self._student_saved_online_match: bool | None = None

        self._stop_evt = threading.Event()
        self._worker: threading.Thread | None = None
        self._queue: Queue[tuple[np.ndarray, str]] = Queue(maxsize=1)
        self._photo: ImageTk.PhotoImage | None = None
        # 预览区当前尺寸缓存（主线程 Configure 回调写、worker 线程读；GIL 下 tuple 读写
        # 原子，无需加锁）。worker 线程在 _post_frame/_post_frame2 里按此做 aspect-fit
        # resize，GUI 主线程 _tick 只剩 ImageTk.PhotoImage + configure（点1：预览渲染解绑）。
        self._preview_wh: tuple[int, int] = (0, 0)
        # 第二路预览队列（双摄像头双面视图，issue #58）：单摄模式恒空，preview2 恒隐藏，
        # 现有单摄路径零改动。
        self._queue2: Queue[tuple[np.ndarray, str]] = Queue(maxsize=1)
        self._dual_preview_queue: Queue[DualPreviewPacket] = Queue(maxsize=1)
        self._dual_preview_lock = threading.Lock()
        self._dual_render_lock = threading.Lock()
        self._session_generation = 0
        self._current_session_generation = 0
        self._active_dual_generation = 0
        self._dual_preview_stage = "raw"
        self._dual_recording_ready = False
        self._dual_done_generation = 0
        self._dual_metrics_lock = threading.Lock()
        self._dual_startup_metrics: dict[int, _DualStartupMetrics] = {}
        self._dual_first_render_events: dict[int, threading.Event] = {}
        self._dual_layout_queue: Queue[str] = Queue(maxsize=1)
        self._photo2: ImageTk.PhotoImage | None = None
        self._preview_wh2: tuple[int, int] = (0, 0)
        self._compare_win: CompareWindow | None = None
        self._settings_win: SettingsWindow | None = None
        self._settings_win: SettingsWindow | None = None

        # 录制/暂停运行时控制器（与 Tkinter 解耦的状态机）。使用模块默认的
        # writer_factory（绑定 core.video_writer.open_video_writer）。path_provider
        # 注入为读取 self.record_dir_var 的闭包：用户在「录制」分组选择的保存目录
        # 下，按含微秒的时间戳生成 record_<timestamp>.mp4，避免快速重录与后台转码争用同名文件。
        self._rec = RecordingController(path_provider=self._record_path)
        # 第二路录制（双摄双面视图，issue #58）：仅在双摄循环 begin_session，其余模式恒 idle
        # no-op，不影响单摄/文件路径。双摄时两路写入同一段目录下的 front.mp4 / side.mp4。
        self._rec2 = RecordingController(path_provider=self._record_path_cam2)
        # 双路状态切换、写帧、停止和关闭共享同一把锁，保证 front/side 的片段边界一致。
        self._record_pair_lock = threading.Lock()
        # 终结器在 pair lock 外提交任务；独立串行门覆盖“快照/释放 + submit”全程，
        # 避免关窗 cancel_all 插入两者之间导致当前片段既未提交也无 cancelled JSON。
        self._record_finalize_lock = threading.RLock()
        # worker 关闭会话前转存尚未被 Tk tick 消费的 writer 错误；close_session 可清理
        # controller 错误而不让终止边界上的失败静默丢失。
        self._pending_record_errors: list[str] = []
        self._dual_active = False
        self._dual_record_skeleton = False
        self._dual_auto_compare = True
        # 一段录制共用的时间戳（idle→recording 时刷新）：两路 front/side 一致 → 可配对，
        # 连续多段各段不同 → 相互隔离。None 表示尚未开始任何录制。
        self._record_stamp: str | None = None

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
        _front_list, _side_list = load_configured_template_lists()
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
            text="仅指出动作问题，不打分；MediaPipe Lite 本机CPU运行",
            foreground="#555555",
            wraplength=320,
        ).pack(anchor="w", pady=(6, 0))
        self.feedback_controls = FeedbackControls(
            self._student_block, root=self.root,
            set_busy=self._set_feedback_busy,
            show_result=self._set_student_result_text,
            set_status=self.student_feedback_status,
            can_analyze=lambda: self._student_practice_active
            and not self._student_judging
            and not self._student_pending_record
            and self._rec.state not in {"recording", "paused"},
        )
        self._student_feedback_identity = None
        self._student_feedback_record_id = None
        student_btn_row = ttk.Frame(self._student_block)
        student_btn_row.pack(fill="x", pady=(10, 0))
        self.student_start_btn = ttk.Button(
            student_btn_row, text="开始", command=self._student_start
        )
        self.student_start_btn.pack(side="left", fill="x", expand=True)
        self.student_end_btn = ttk.Button(
            student_btn_row,
            text="结束",
            command=self._student_end,
            state="disabled",
        )
        self.student_end_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.student_judge_btn = ttk.Button(
            self._student_block,
            text="动作评判",
            command=self._student_judge,
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

        # 任务代次并发防御与学生练习分析结果主视图 (Milestone 2)
        self._feedback_task_token: int = 0
        self._active_feedback_token: int | None = None
        self._active_feedback_student_id: str | None = None
        self._current_student_record: dict | None = None
        self.current_student_id = self.feedback_controls.student_id

        from apps.feedback_result_panel import FeedbackResultPanel

        self.feedback_result_panel = FeedbackResultPanel(
            outer,
            on_practice_again=self._on_student_practice_again,
            on_export_report=self._on_student_export_report,
            on_view_history=self._on_student_view_history,
            on_review_toggle=self._on_student_review_toggle,
            on_switch_to_preview=self._show_student_preview_view,
            history_store=self.feedback_controls.history(),
        )
        self.feedback_result_panel.grid_remove()

        self.feedback_controls.on_identity_changed = self._on_student_identity_changed
        self.feedback_controls.on_cancelled = self._on_student_cancelled

    def _on_preview_configure(self, event) -> None:
        """主预览 Label 尺寸变化（主线程）：缓存供 worker 线程 _post_frame 做 resize（点1）。"""
        self._preview_wh = (event.width, event.height)

    def _on_preview2_configure(self, event) -> None:
        """第二路预览 Label 尺寸变化（主线程），与 _on_preview_configure 同构。"""
        self._preview_wh2 = (event.width, event.height)

    def _get_runtime_rotate(self, which: str = "primary") -> int:
        lock = getattr(self, "_runtime_rotate_lock", None)
        if lock is None:
            if which == "secondary":
                return int(getattr(self, "_runtime_rotate2", 0))
            return int(getattr(self, "_runtime_rotate", 0))
        with lock:
            if which == "secondary":
                return int(getattr(self, "_runtime_rotate2", 0))
            return int(getattr(self, "_runtime_rotate", 0))

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
        lock = getattr(self, "_runtime_rotate_lock", None)
        if lock is None:
            if which == "secondary":
                self._runtime_rotate2 = d
            else:
                self._runtime_rotate = d
        else:
            with lock:
                if which == "secondary":
                    self._runtime_rotate2 = d
                else:
                    self._runtime_rotate = d
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
            if which != "secondary" and bool(getattr(self, "_exam_active", False)):
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
            save_configured_template_lists(front_list=front, side_list=side)
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

        - 考试中强制 True。
        - 默认读主线程维护的 ``_dual_auto_compare`` 缓存（worker 可安全读取）。
        - ``prefer_ui=True`` 时（仅主线程）从滑块/BooleanVar 刷新缓存。
        """
        if bool(getattr(self, "_exam_active", False)) or getattr(
            self, "_exam_pending_row", None
        ) is not None:
            return True
        if prefer_ui and hasattr(self, "auto_compare_var"):
            try:
                on = bool(self.auto_compare_var.get())
                self._dual_auto_compare = on
                return on
            except Exception:
                pass
        return bool(getattr(self, "_dual_auto_compare", True))

    def _write_rec_frame(self, rec, frame) -> None:
        """写录制帧；若运行中旋转导致尺寸与 writer 不一致则 resize 适配。"""
        try:
            size = getattr(rec, "session_size", None)
            if size is not None and hasattr(frame, "shape"):
                frame = _fit_frame_to_size(frame, size)
        except Exception:
            pass
        rec.write_frame(frame)

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

    def _post_dual_preview_layout(self, layout: str) -> None:
        """采集线程只投递布局数据，不直接调用 Tk API。"""
        if getattr(self, "_closing", False):
            return
        layout_queue = getattr(self, "_dual_layout_queue", None)
        if layout_queue is None:
            return
        while True:
            try:
                layout_queue.get_nowait()
            except Empty:
                break
        layout_queue.put(layout)

    def _drain_dual_preview_layout(self) -> None:
        """在 Tk 主线程应用最新双摄布局，关窗后只丢弃不触碰控件。"""
        layout_queue = getattr(self, "_dual_layout_queue", None)
        if layout_queue is None:
            return
        latest: str | None = None
        while True:
            try:
                latest = layout_queue.get_nowait()
            except Empty:
                break
        if latest is not None and not getattr(self, "_closing", False):
            self._set_dual_preview_layout(latest)

    def _build_online_matcher(self) -> online_matcher.OnlineActionMatcher | None:
        """加载实时模板库；模板缺失或不兼容时保持预览可用。"""
        try:
            templates = online_matcher.load_template_library(paths.templates_dir() / "online")
            if not templates:
                return None
            return online_matcher.OnlineActionMatcher(templates, self._post_match)
        except Exception:
            return None

    @staticmethod
    def _feed_online_matcher(matcher, pose_landmarks, timestamp_ms, frame_index) -> None:
        """把 raw pose 特征送入在线 matcher，不让预览平滑结果回流。"""
        if matcher is None or pose_landmarks is None:
            return
        feature = pf.normalize_pose_xy_v3(pose_landmarks)
        if feature is not None:
            matcher.push(feature, timestamp_ms, frame_index)

    def _post_match(self, result: online_matcher.MatchResult) -> None:
        """把后台 DTW 命中安全投递到 Tk 主线程。"""
        if self._stop_evt.is_set() or result.action is None:
            return
        name = result.action
        for suffix in ("_正面", "_侧面", "_左侧", "_右侧"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        text = f"识别到：{name} ({result.score:.2f})"

        def _apply() -> None:
            if self._stop_evt.is_set():
                return
            try:
                self.match_var.set(text)
            except (TclError, RuntimeError):
                pass

        try:
            self.root.after(0, _apply)
        except (TclError, RuntimeError):
            pass

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

    def _record_path(self) -> Path:
        """RecordingController 的 path_provider：在用户选择的录制目录下按时间戳生成文件名。

        目录取自 self.record_dir_var；为空时回退到 outputs_dir()。所有录制先按日期归到
        <YYYYMMDD>/ 父目录。单摄/文件模式平铺为 <日期>/record_<timestamp>.mp4；双摄模式下
        本段两路收进 <日期>/record_<timestamp>/ 子目录，第一路 front.mp4、第二路 side.mp4，
        一天一父集、一段一文件夹，便于按时间归档与配对。
        """
        if self._dual_active:
            return self._record_dir_for_stamp() / "front.mp4"
        return self._record_day_dir() / f"record_{self._current_stamp()}.mp4"

    def _record_path_cam2(self) -> Path:
        """第二路（侧面）录制的 path_provider：与第一路同段子目录，存 side.mp4。"""
        return self._record_dir_for_stamp() / "side.mp4"

    def _current_stamp(self) -> str:
        # 同一段录制共用 _record_stamp：两路 front/side 收进同一子目录，
        # 连续多段各自换新戳自然隔离。首帧懒创建 writer 时读取。
        return self._record_stamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _record_day_dir(self) -> Path:
        # 按日期的父集：同一天的录制归到 <YYYYMMDD>/ 下。日期取自本段时间戳前缀，
        # 跨零点续写也稳定落在开录当天。这里只读普通 Path 缓存，不从 worker 访问 Tk。
        directory = getattr(self, "_record_base_dir", outputs_dir())
        return Path(directory) / self._current_stamp()[:8]

    def _record_dir_for_stamp(self) -> Path:
        return self._record_day_dir() / f"record_{self._current_stamp()}"

    def _transcode_async(self, path: Path | None) -> None:
        """后台把 MJPG/AVI 回退产物转成 H.264，避免阻塞 GUI 或推理线程。"""
        if path is None:
            return

        def _run() -> None:
            try:
                video_writer.transcode_to_h264(path)
            finally:
                current = threading.current_thread()
                with self._transcode_lock:
                    self._transcode_workers.discard(current)

        worker = threading.Thread(target=_run, name="h264-transcode", daemon=False)
        with self._transcode_lock:
            self._transcode_workers.add(worker)
            try:
                worker.start()
            except Exception:
                self._transcode_workers.discard(worker)
                raise

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
        考试模式下手动录制锁定，直接返回。
        """
        if bool(getattr(self, "_exam_manual_locked", False)):
            return
        stop_evt = getattr(self, "_stop_evt", None)
        if getattr(self, "_closing", False) or (
            stop_evt is not None and stop_evt.is_set()
        ):
            return
        if not bool(getattr(self, "_dual_recording_ready", True)):
            return

        finalize_failed_pair = False
        with self._record_pair_lock:
            dual_active = bool(getattr(self, "_dual_active", False))
            prev_state = self._rec.state
            # 必须在「主路 idle → 开新录」之前检测分叉/错误：主路写失败回 idle、
            # 侧路仍 recording 时若直接 begin，会换 stamp 并把未 finalize 的旧片段弄丢。
            if dual_active and hasattr(self._rec, "snapshot") and hasattr(
                self._rec2, "snapshot"
            ):
                snap = self._rec.snapshot()
                snap2 = self._rec2.snapshot()
                finalize_failed_pair = bool(
                    snap.last_error
                    or snap2.last_error
                    or snap.state != snap2.state
                )

        if finalize_failed_pair:
            App._finalize_and_dispatch_recording_pair(self, close_session=False)
            new_state: RecordingState = "idle"
            record_btn = getattr(self, "record_btn", None)
            if record_btn is not None:
                record_btn.configure(text=RECORD_BTN_TEXT[new_state])
            sync_record_stop = getattr(self, "_sync_record_stop_enabled", None)
            if sync_record_stop is not None:
                sync_record_stop(new_state)
            return

        if prev_state == "idle":
            App._begin_recording_segment(self)
            return

        # 暂停 / 继续：仍用 toggle（考试路径禁止走这里）
        with self._record_pair_lock:
            dual_active = bool(getattr(self, "_dual_active", False))
            new_state = self._rec.request_toggle()
            new_state2 = self._rec2.request_toggle()
            if dual_active and new_state != new_state2:
                try:
                    self._rec.stop_recording()
                except Exception:
                    pass
                try:
                    self._rec2.stop_recording()
                except Exception:
                    pass
                new_state = "idle"
                finalize_failed_pair = True
        if finalize_failed_pair:
            App._finalize_and_dispatch_recording_pair(self, close_session=False)
            new_state = "idle"
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None:
            record_btn.configure(text=RECORD_BTN_TEXT[new_state])
        sync_record_stop = getattr(self, "_sync_record_stop_enabled", None)
        if sync_record_stop is not None:
            sync_record_stop(new_state)

    def _begin_recording_segment(self, exam_row=None) -> bool:
        """从 idle 开录一段（考试与手动共用）。成功返回 True。

        前置：双摄 claim 后已 begin_session；本方法只 publish stamp + toggle idle→recording。
        双路必须同时 idle→recording；任一路失败则两路都复位 idle，避免主/侧状态分叉。
        """
        stop_evt = getattr(self, "_stop_evt", None)
        if getattr(self, "_closing", False) or (
            stop_evt is not None and stop_evt.is_set()
        ):
            return False
        if not bool(getattr(self, "_dual_recording_ready", True)):
            return False

        record_dir_var = getattr(self, "record_dir_var", None)
        next_base_dir = getattr(self, "_record_base_dir", outputs_dir())
        if record_dir_var is not None:
            base = record_dir_var.get().strip()
            # 空串多为 Tk 读取时序异常，非"用户要默认目录"。保持上次已缓存的目录，
            # 不回退 outputs_dir()，避免下方 save_record_dir 把持久化路径覆盖成默认。
            if base:
                next_base_dir = Path(base)

        new_dual_segment_id: str | None = None
        with self._record_pair_lock:
            dual_active = bool(getattr(self, "_dual_active", False))
            main_state = self._rec.state
            side_state = self._rec2.state
            # 双摄时两侧都须 idle；分叉时先 stop 对齐
            if dual_active and (main_state != "idle" or side_state != "idle"):
                try:
                    self._rec.stop_recording()
                except Exception:
                    pass
                try:
                    self._rec2.stop_recording()
                except Exception:
                    pass
                main_state = self._rec.state
                side_state = self._rec2.state
                if main_state != "idle" or side_state != "idle":
                    return False
            elif (not dual_active) and main_state != "idle":
                try:
                    self._rec.stop_recording()
                except Exception:
                    pass
                if self._rec.state != "idle":
                    return False
            self._record_base_dir = Path(next_base_dir)
            save_record_dir(next_base_dir)
            self._record_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if dual_active:
                new_dual_segment_id = f"record_{self._record_stamp}"
            self._exam_pending_row = exam_row
            self._exam_discard_next = False
            # 始终双路 toggle（与历史行为一致；单摄时侧路 session 未激活则为 no-op idle）
            new_state = self._rec.request_toggle()
            new_state2 = self._rec2.request_toggle()
            if dual_active and (new_state != "recording" or new_state2 != "recording"):
                # 回滚：避免主 recording / 侧 paused 之类分叉
                if new_state in {"recording", "paused"}:
                    try:
                        self._rec.stop_recording()
                    except Exception:
                        pass
                if new_state2 in {"recording", "paused"}:
                    try:
                        self._rec2.stop_recording()
                    except Exception:
                        pass
                self._exam_pending_row = None
                return False
            if (not dual_active) and new_state != "recording":
                self._exam_pending_row = None
                return False

        if new_dual_segment_id is not None:
            auto_compare = (
                True
                if exam_row is not None
                else App._current_auto_compare(self, prefer_ui=True)
            )
            # 学生练习：永不自动模板比对
            if bool(getattr(self, "_student_practice_active", False)):
                auto_compare = False
            self._dual_auto_compare = bool(auto_compare)
            status_text = (
                "自动比对：录制中" if auto_compare else "仅录制：录制中"
            )
            for name, value in (
                ("compare_segment_var", f"片段：{new_dual_segment_id}"),
                ("compare_status_var", status_text),
                ("compare_score_var", "正面：-　侧面：-　综合：-"),
                ("compare_error_var", ""),
            ):
                var = getattr(self, name, None)
                if var is not None:
                    var.set(value)
            if exam_row is not None and hasattr(self, "_exam_panel"):
                panel = getattr(self, "_exam_panel", None)
                stamp = self._record_stamp
                if panel is not None and stamp:
                    base = Path(getattr(self, "_record_base_dir", outputs_dir()))
                    segment_dir = base / stamp[:8] / f"record_{stamp}"
                    try:
                        panel.bind_segment(
                            exam_row.row_id,
                            f"record_{stamp}",
                            str(segment_dir),
                        )
                    except Exception:
                        pass

        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None and not getattr(self, "_exam_manual_locked", False):
            record_btn.configure(text=RECORD_BTN_TEXT["recording"])
        self._sync_record_stop_enabled("recording")
        return True

    def _end_recording_segment(self, *, discard: bool = False, exam_row=None) -> None:
        """结束当前录制片段；discard=True 时不入比对队列。

        discard 在关窗/中止路径也必须执行：不得因 ``_closing`` 提前 return，
        否则随后的普通 finalize 会把已 aborted 片段送去评分。
        """
        stop_evt = getattr(self, "_stop_evt", None)
        closing = bool(getattr(self, "_closing", False))
        stopped = stop_evt is not None and stop_evt.is_set()
        if not discard and (closing or stopped):
            return
        if exam_row is not None:
            self._exam_pending_row = exam_row
        self._exam_discard_next = bool(discard)
        App._finalize_and_dispatch_recording_pair(self, close_session=False)
        self._exam_discard_next = False
        self._exam_pending_row = None
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None and not getattr(self, "_exam_manual_locked", False):
            record_btn.configure(text=RECORD_BTN_TEXT["idle"])
        self._sync_record_stop_enabled("idle")

    def _on_record_stop(self) -> None:
        """「结束录制」点击回调：结束当前录制片段并落盘，但不结束识别会话。"""
        if bool(getattr(self, "_exam_manual_locked", False)):
            return
        stop_evt = getattr(self, "_stop_evt", None)
        if getattr(self, "_closing", False) or (
            stop_evt is not None and stop_evt.is_set()
        ):
            return
        App._end_recording_segment(self, discard=False)

    def _open_exam_panel(self) -> None:
        if bool(getattr(self, "_student_practice_active", False)):
            messagebox.showinfo("学生练习", "请先退出学生练习模式，再打开考试模式。")
            return
        existing = getattr(self, "_exam_panel", None)
        if existing is not None:
            try:
                existing.win.lift()
                return
            except Exception:
                pass
        from apps.exam_panel import ExamPanel

        ExamPanel(self)

    # ------------------------------------------------------------------
    # 学生练习模式（MediaPipe 关键点规则检查，无分数）
    # ------------------------------------------------------------------

    def _enter_student_practice(self) -> None:
        """切换到学生练习极简界面：开始 / 结束 / 动作评判 / 预览。"""
        if bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_exam_active", False)) or bool(
            getattr(self, "_exam_manual_locked", False)
        ):
            messagebox.showinfo("学生练习", "考试进行中，无法进入学生练习。")
            return
        if self._worker and self._worker.is_alive():
            messagebox.showinfo(
                "学生练习", "请先停止当前识别会话，再进入学生练习。"
            )
            return

        self._student_practice_active = True
        self._student_pending_record = False
        self._student_judging = False
        self._student_last_segment_dir = None
        self._student_last_segment_id = None
        self._student_front_video = None
        self._student_side_video = None
        self._student_segment_ready = False
        self._active_feedback_token = None
        self._active_feedback_student_id = None
        self._current_student_record = None
        if hasattr(self, "feedback_result_panel"):
            self.feedback_result_panel.clear()
            self._show_student_preview_view()

        # 省 GPU：关骨架/手部/自动比对（不改正式评分算法，仅本模式会话策略）
        self._student_saved_auto_compare = bool(self.auto_compare_var.get())
        self._student_saved_skeleton = bool(self.record_skeleton_var.get())
        self._student_saved_hands = bool(self.enable_hands_var.get())
        self._student_saved_online_match = bool(self.online_match_var.get())
        self.auto_compare_var.set(False)
        self.auto_compare_scale_var.set(0.0)
        self._dual_auto_compare = False
        self.record_skeleton_var.set(False)
        self._dual_record_skeleton = False
        self.enable_hands_var.set(False)
        self.online_match_var.set(False)
        self._sync_auto_compare_hint()

        try:
            self._teacher_block.pack_forget()
        except Exception:
            pass
        try:
            self.primary_secondary_separator.pack_forget()
        except Exception:
            pass
        try:
            self._secondary_block.pack_forget()
        except Exception:
            pass
        try:
            self._info_teacher_extra.pack_forget()
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
        """退出学生练习并恢复教师控件。"""
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_student_judging", False)):
            messagebox.showinfo("学生练习", "动作评判进行中，请稍候再退出。")
            return

        # 若仍在录制则先结束片段；再停会话
        self._student_presence_gate = None
        self._student_pending_record = False
        if self._student_announcer is not None:
            self._student_announcer.close()
            self._student_announcer = None
        try:
            if self._rec.state in {"recording", "paused"}:
                App._end_recording_segment(self, discard=False)
        except Exception:
            pass
        if self._worker and self._worker.is_alive():
            self._stop()

        self._student_practice_active = False
        self._student_pending_record = False
        self._student_judging = False
        self._student_segment_ready = False
        self._student_front_video = None
        self._student_side_video = None
        self._active_feedback_token = None
        self._active_feedback_student_id = None
        self._current_student_record = None
        if hasattr(self, "feedback_result_panel"):
            self.feedback_result_panel.clear()
            self._show_student_preview_view()

        if self._student_saved_auto_compare is not None:
            self.auto_compare_var.set(bool(self._student_saved_auto_compare))
            self.auto_compare_scale_var.set(
                1.0 if self._student_saved_auto_compare else 0.0
            )
            self._dual_auto_compare = bool(self._student_saved_auto_compare)
        if self._student_saved_skeleton is not None:
            self.record_skeleton_var.set(bool(self._student_saved_skeleton))
        if self._student_saved_hands is not None:
            self.enable_hands_var.set(bool(self._student_saved_hands))
        if self._student_saved_online_match is not None:
            self.online_match_var.set(bool(self._student_saved_online_match))
        self._student_saved_auto_compare = None
        self._student_saved_skeleton = None
        self._student_saved_hands = None
        self._student_saved_online_match = None
        self._sync_auto_compare_hint()

        try:
            self._student_block.pack_forget()
        except Exception:
            pass
        try:
            self._teacher_block.pack(fill="x")
        except Exception:
            pass
        # 恢复 pack 顺序：primary → separator → secondary → info
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

        self.root.title("MediaPipe 动作识别（人体姿态 + 手部）")
        if self._worker and self._worker.is_alive():
            # 采集尚在收尾：保持开始禁用，避免点击被静默丢弃；_post_done 负责恢复。
            self.status_var.set("正在停止…")
            for name in ("start_btn", "stop_btn"):
                button = getattr(self, name, None)
                if button is not None:
                    button.configure(state="disabled")
            return
        self.status_var.set("就绪")
        self._set_running_controls(False)
        self._sync_camera_warmup()

    def _student_dual_cameras_ready(self) -> tuple[bool, str]:
        label2 = (self.camera_choice_var_2.get() or "").strip()
        if not label2 or label2 == NO_SECOND_CAMERA:
            return False, "学生练习需要正/侧双摄像头，请选择第二摄像头（侧面）。"
        label1 = (self.camera_choice_var.get() or "").strip()
        if not label1:
            return False, "请选择第一摄像头（正面）。"
        return True, ""

    def _sync_student_buttons(self) -> None:
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        judging = bool(getattr(self, "_student_judging", False))
        recording = False
        try:
            recording = self._rec.state in {"recording", "paused"}
        except Exception:
            recording = False
        waiting = bool(self._student_pending_record)
        # 仅后处理终态就绪后才允许评判（避免与转码抢文件）
        can_judge = bool(getattr(self, "_student_segment_ready", False))
        if hasattr(self, "feedback_controls"):
            self.feedback_controls.set_recording(recording or self._student_pending_record)

        try:
            self.student_start_btn.configure(
                state="disabled" if judging or recording or waiting else "normal"
            )
            self.student_end_btn.configure(
                state="normal" if ((recording or waiting) and not judging) else "disabled"
            )
            self.student_judge_btn.configure(
                state="normal"
                if (can_judge and not judging and not recording and not waiting)
                else "disabled"
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

    def _student_start(self) -> None:
        """开启双摄预览，等待进入站位区域后开录。"""
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_student_judging", False)):
            return
        if self._student_pending_record or self._rec.state in {"recording", "paused"}:
            return
        try:
            self.feedback_controls.identity()
        except ValueError as exc:
            messagebox.showwarning("学生练习", str(exc))
            return
        ok, err = self._student_dual_cameras_ready()
        if not ok:
            messagebox.showwarning("学生练习", err)
            return

        from apps.exam_panel import load_exam_prefs

        lite = next(m for m in model_manager.MEDIAPIPE_MODELS if m.key == "pose_lite")
        if not model_manager.is_installed(lite):
            messagebox.showwarning("学生练习", "缺少 Lite 模型，请先在设置→模型管理中安装。")
            return
        roi = load_exam_prefs().get("exam_roi_norm", (0.2, 0.1, 0.8, 0.95))
        try:
            x0, y0, x1, y1 = (float(v) for v in roi)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("invalid ROI")
        except (TypeError, ValueError):
            x0, y0, x1, y1 = (0.2, 0.1, 0.8, 0.95)
        self._exam_set_roi((x0, y0, x1, y1))

        # 强制仅录制、无骨架
        self.auto_compare_var.set(False)
        self.auto_compare_scale_var.set(0.0)
        self._dual_auto_compare = False
        self.record_skeleton_var.set(False)
        self._dual_record_skeleton = False
        self.enable_hands_var.set(False)

        session_alive = bool(self._worker and self._worker.is_alive())
        if not session_alive:
            # 会话 state 冻结 1080p 请求；池发现预热格式不一致时会在旧进程释放后重开。
            self._student_pending_record = True
            self.student_status_var.set("正在启动双摄预览…")
            try:
                self._start()
            except Exception as e:
                self._student_pending_record = False
                messagebox.showerror("学生练习", f"启动失败：{e}")
                self._sync_student_buttons()
                return
            if not (self._worker and self._worker.is_alive()):
                self._student_pending_record = False
                self.student_status_var.set("双摄未启动，请检查摄像头后重试")
            if bool(getattr(self, "_dual_recording_ready", False)):
                self._student_wait_for_position()
            self._sync_student_buttons()
            return

        self._student_pending_record = True
        self._student_wait_for_position()

    def _student_wait_for_position(self) -> None:
        if not self._student_practice_active or not self._student_pending_record:
            return
        if self._dual_recording_ready and self._student_presence_gate is None:
            from core.presence_gate import PresenceGate, PresenceGateConfig

            # 练习严格按离场2秒结束，不使用考试的最短录制时长限制。
            self._student_presence_started_at = time.monotonic()
            self._student_presence_gate = PresenceGate(PresenceGateConfig(min_record_s=0.0))
            self.student_status_var.set("请进入正面预览黄框，站稳0.8秒后播报开始并自动录制")
        self._sync_student_buttons()

    def _student_on_occupancy(self, present: bool, now: float) -> None:
        gate = self._student_presence_gate
        if (gate is None or not self._student_practice_active or self._student_judging
                or self._closing or self._stop_evt.is_set()
                or now < self._student_presence_started_at):
            return
        mode = "wait_enter" if self._student_pending_record else "recording"
        for event in gate.update(present, now, mode=mode):
            if event.kind == "enter_stable":
                self._student_try_begin_recording()
                if self._rec.state == "recording":
                    gate.begin_recording(now)
                else:
                    self._student_presence_gate = None
            elif event.kind == "empty_stable":
                self._student_end()

    def _student_try_begin_recording(self) -> None:
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_student_judging", False)):
            return
        if not bool(getattr(self, "_dual_recording_ready", False)):
            self._student_pending_record = False
            self.student_status_var.set("双摄尚未就绪，请重试「开始」")
            self._sync_student_buttons()
            return
        self._student_pending_record = False
        self._dual_auto_compare = False
        try:
            identity = self.feedback_controls.identity()
        except ValueError as exc:
            self.student_status_var.set(str(exc))
            self._sync_student_buttons()
            return
        ok = App._begin_recording_segment(self)
        if not ok:
            self.student_status_var.set("开录失败，请重试「开始」")
            messagebox.showerror("学生练习", "无法开始录制，请确认双摄已就绪。")
            self._sync_student_buttons()
            return
        self._student_feedback_identity = identity
        self._student_feedback_record_id = None
        if self._student_announcer is None:
            from core.exam_announcer import ExamAnnouncer

            self._student_announcer = ExamAnnouncer()
        self._student_announcer.announce("开始")
        self.student_status_var.set("开始，录制中…离开黄框连续2秒自动结束，也可点「结束」")
        self._sync_student_buttons()

    def _student_end(self) -> None:
        """结束 = 停录落盘，不自动评判；须等后处理终态才可评判。"""
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_student_judging", False)):
            return
        waiting = self._student_pending_record
        self._student_presence_gate = None
        self._student_pending_record = False
        if self._rec.state not in {"recording", "paused"}:
            self.student_status_var.set("已取消等待到位" if waiting else "当前没有进行中的录制")
            self._sync_student_buttons()
            return
        # 新一段：清空就绪态，避免误用上一段视频
        self._student_segment_ready = False
        self._student_front_video = None
        self._student_side_video = None
        App._end_recording_segment(self, discard=False)
        if self._student_announcer is not None:
            self._student_announcer.announce("结束")
        # segment_dir / segment_id 在 dispatch 里写入
        if self._student_last_segment_dir is None and self._record_stamp:
            base = Path(getattr(self, "_record_base_dir", outputs_dir()))
            stamp = self._record_stamp
            candidate = base / stamp[:8] / f"record_{stamp}"
            if candidate.is_dir():
                self._student_last_segment_dir = candidate
                self._student_last_segment_id = candidate.name
        if self._student_last_segment_dir is not None:
            self.student_status_var.set(
                f"已停录：{self._student_last_segment_dir.name}，"
                "正在转码落盘，完成后可「动作评判」…"
            )
        else:
            self.student_status_var.set("录制已结束（未解析到片段目录，请重试）")
        self._sync_student_buttons()

    def _on_student_postprocess_update(self, update: PostprocessUpdate) -> None:
        """后处理终态：记录最终视频路径，再允许动作评判。"""
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        seg_id = getattr(self, "_student_last_segment_id", None)
        if not seg_id or update.segment_id != seg_id:
            return
        terminal = {"completed", "failed", "skipped", "cancelled"}
        if update.status not in terminal:
            # 中间态：提示转码中
            if update.status in {"queued", "transcoding", "validating"}:
                try:
                    self.student_status_var.set(
                        f"处理中：{update.message or update.status}…"
                    )
                except Exception:
                    pass
            return

        front = getattr(update, "front_video", None)
        side = getattr(update, "side_video", None)
        front_ok = front is not None and Path(front).is_file()
        side_ok = side is not None and Path(side).is_file()

        # skipped(auto_compare_disabled) / completed：转码完成即可评
        # failed/cancelled：若仍有可读源文件也允许（降级），否则禁用
        usable = front_ok or side_ok
        if update.status in {"failed", "cancelled"} and not usable:
            self._student_segment_ready = False
            self._student_front_video = None
            self._student_side_video = None
            try:
                self.student_status_var.set(
                    f"片段处理失败，无法评判：{update.error_code or update.message}"
                )
            except Exception:
                pass
            self._sync_student_buttons()
            return

        self._student_front_video = Path(front) if front_ok else None
        self._student_side_video = Path(side) if side_ok else None
        # 目录内再兜底一次
        if (
            (self._student_front_video is None or self._student_side_video is None)
            and self._student_last_segment_dir is not None
        ):
            try:
                from core.feedback_history import resolve_segment_videos

                f2, s2 = resolve_segment_videos(self._student_last_segment_dir)
                if self._student_front_video is None:
                    self._student_front_video = f2
                if self._student_side_video is None:
                    self._student_side_video = s2
            except Exception:
                pass

        ready = (
            self._student_front_video is not None
            and self._student_side_video is not None
            and Path(self._student_front_video).is_file()
            and Path(self._student_side_video).is_file()
        )
        # 单路仍可检查可见项目；缺视角的规则明确标记无法判断
        if not ready:
            ready = (
                (self._student_front_video is not None and Path(self._student_front_video).is_file())
                or (self._student_side_video is not None and Path(self._student_side_video).is_file())
            )
        self._student_segment_ready = bool(ready)
        try:
            if self._student_segment_ready:
                self.student_status_var.set(
                    f"已就绪：{seg_id}，可点「动作评判」"
                )
            else:
                self.student_status_var.set(
                    f"处理结束但未找到可读视频：{update.error_code or update.status}"
                )
        except Exception:
            pass
        self._sync_student_buttons()

    def student_feedback_status(self, text: str) -> None:
        self.student_status_var.set(text)

    def _set_feedback_busy(self, busy: bool) -> None:
        self._student_judging = busy
        self._sync_student_buttons()

    def _student_judge(self) -> None:
        """按录制开始时绑定的学生与动作，后台分析并保存问题说明。"""
        if not self._student_practice_active or self._student_judging:
            return
        if self._student_pending_record:
            return
        if not self._student_segment_ready:
            messagebox.showinfo("动作评判", "请先结束练习，并等待录像转码落盘。")
            return
        if self._rec.state in {"recording", "paused"}:
            messagebox.showinfo("动作评判", "请先结束录制再评判。")
            return
        identity = self._student_feedback_identity
        if identity is None:
            messagebox.showinfo("动作评判", "这段录像未绑定学生，请使用「导入旧视频」分析。")
            return

        self._feedback_task_token = getattr(self, "_feedback_task_token", 0) + 1
        token = self._feedback_task_token
        self._active_feedback_token = token
        student_id = identity.get("studentId") if isinstance(identity, dict) else None
        self._active_feedback_student_id = student_id

        def on_saved(record):
            self._student_feedback_record_id = record["id"]
            if hasattr(self, "_on_student_analysis_saved"):
                self._on_student_analysis_saved(token=token, student_id=student_id, record=record)

        self.feedback_controls.analyze_recording(
            identity, self._student_front_video, self._student_side_video,
            record_id=self._student_feedback_record_id, on_saved=on_saved,
            token=token,
        )



    def _on_student_identity_changed(self) -> None:
        """用户在学生练习界面修改了学号：作废当前任务代次，清空旧记录。"""
        self._active_feedback_token = None
        self._active_feedback_student_id = None
        self._current_student_record = None

    def _on_student_cancelled(self) -> None:
        """用户点击了取消分析：作废任务代次。"""
        self._active_feedback_token = None

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
        if not bool(getattr(self, "_student_practice_active", False)):
            return
        if bool(getattr(self, "_student_judging", False)):
            return

        # 1. 视图平滑切回预览
        self._show_student_preview_view()

        # 2. 状态硬复位：解除闸门，清空就绪态，作废代次
        self._student_presence_gate = None
        self._student_pending_record = False
        self._student_segment_ready = False
        self._student_front_video = None
        self._student_side_video = None
        self._active_feedback_token = None

        # 3. 按钮与文案同步，不自动触发录制
        try:
            self.student_status_var.set("已返回录制预览。进入黄框就位后请点击「开始」")
            self._sync_student_buttons()
        except Exception:
            pass

    def _on_student_analysis_saved(self, token: int, student_id: str, record: dict) -> None:
        """分析完成且保存后的主线程回调：通过任务代次与学号双重拦截晚到/作废结果。"""
        if bool(getattr(self, "_closing", False)):
            return

        active_token = getattr(self, "_active_feedback_token", None)
        if active_token is None or token != active_token:
            return

        curr_student_id = None
        if hasattr(self, "current_student_id") and hasattr(self.current_student_id, "get"):
            curr_student_id = self.current_student_id.get()
        elif hasattr(self, "feedback_controls") and hasattr(self.feedback_controls, "student_id"):
            curr_student_id = self.feedback_controls.student_id.get()

        if curr_student_id is not None and curr_student_id != student_id:
            return

        rec = getattr(self, "_rec", None)
        if rec is not None and getattr(rec, "state", "idle") in ("recording", "paused"):
            return

        self._current_student_record = record
        self._student_feedback_record_id = record.get("id")
        self._show_student_result_view(record)

    def _on_student_export_report(self, record: dict) -> None:
        """从结果面板导出 HTML / TXT 图文报告。"""
        from core.feedback_report import render_html_report

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
            self._current_student_record = updated
            self.feedback_result_panel.set_record(updated)
        except Exception as exc:
            messagebox.showerror("复核失败", str(exc), parent=self.root)

    def _exam_primary_rotate(self) -> int:
        try:
            return int(self._get_runtime_rotate("primary"))
        except Exception:
            try:
                return _parse_rotate(self.rotate_var.get())
            except Exception:
                return 0

    def _exam_set_roi(self, roi: tuple[float, float, float, float]) -> None:
        with self._exam_lock:
            self._exam_roi = tuple(float(x) for x in roi)  # type: ignore[assignment]

    def _exam_set_active(
        self, active: bool, *, run_id: str | None = None, run_dir: Path | None = None
    ) -> None:
        with self._exam_lock:
            self._exam_active = bool(active)
            if active:
                self._exam_run_id = run_id
                self._exam_run_dir = run_dir
                self._dual_auto_compare = True
            else:
                self._exam_run_id = None
                self._exam_run_dir = None
                self._exam_occupancy_armed = False

    def _exam_arm_occupancy(self, armed: bool) -> None:
        with self._exam_lock:
            self._exam_occupancy_armed = bool(armed)

    def _exam_lock_manual_record(self, locked: bool) -> None:
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
                        if not getattr(self, "_worker", None) or (
                            self._worker is not None and not self._worker.is_alive()
                        ):
                            w.configure(state="normal")
                    elif name == "auto_compare_check":
                        # 自动比对滑块：非考试时始终可调；考试锁定期间禁用。
                        w.configure(state="normal")
                    elif name == "record_btn":
                        if getattr(self, "_dual_recording_ready", False) or (
                            self._worker is not None and self._worker.is_alive()
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

    def _exam_preflight(self) -> tuple[bool, str]:
        if not bool(getattr(self, "_dual_active", False)):
            return False, "请先选择双摄像头并点击「开始」进入双摄会话。"
        if not bool(getattr(self, "_dual_recording_ready", False)):
            return False, "双摄尚未就绪，请稍候再试。"
        # 骨架会话 worker 会跳过 occupancy，首位会永久卡在 wait_enter
        skeleton_on = bool(getattr(self, "_dual_record_skeleton", False))
        try:
            skeleton_var = getattr(self, "record_skeleton_var", None)
            if skeleton_var is not None and bool(skeleton_var.get()):
                skeleton_on = True
        except Exception:
            pass
        if skeleton_on:
            return (
                False,
                "考试模式不支持「开启骨架」会话（占用检测会被跳过）。"
                "请取消骨架后重新「开始」双摄，再开考。",
            )
        # 多模板池：正对正、侧对侧，至少一侧非空且模板可用（heavy+pose33_v3）即可开考。
        front_list, side_list = self._current_template_lists()
        try:
            validate_auto_compare_template_lists(front_list, side_list)
        except Exception as exc:
            return (
                False,
                f"正/侧模板池不可用（须至少一侧含可用 heavy+pose33_v3 模板，请在主界面录制区添加）：{exc}",
            )
        heavy = next(
            (m for m in model_manager.MEDIAPIPE_MODELS if m.key == "pose_heavy"),
            None,
        )
        lite = next(
            (m for m in model_manager.MEDIAPIPE_MODELS if m.key == "pose_lite"),
            None,
        )
        if heavy is None or not model_manager.is_installed(heavy):
            return False, "缺少 pose_landmarker_heavy.task，请在模型管理中安装。"
        if lite is None or not model_manager.is_installed(lite):
            return False, "缺少 pose_landmarker_lite.task（占用检测需要）。"
        return True, ""

    def _drain_exam_occupancy(self) -> None:
        q = getattr(self, "_exam_occupancy_queue", None)
        if q is None:
            return
        panel = getattr(self, "_exam_panel", None)
        while True:
            try:
                present, now = q.get_nowait()
            except Empty:
                return
            if bool(getattr(self, "_student_practice_active", False)):
                self._student_on_occupancy(bool(present), float(now))
            elif panel is not None:
                try:
                    panel.on_occupancy(bool(present), float(now))
                except Exception:
                    pass

    def _write_recording_pair(self, frame: np.ndarray, frame2: np.ndarray) -> None:
        """把双摄同一轮的两帧写入放在同一片段边界内。"""
        with self._record_pair_lock:
            for rec, fr in ((self._rec, frame), (self._rec2, frame2)):
                try:
                    size = getattr(rec, "session_size", None)
                    if size is not None and hasattr(fr, "shape"):
                        fr = _fit_frame_to_size(fr, size)
                except Exception:
                    pass
                rec.write_frame(fr)

    def _close_primary_recording_session(self) -> Path | None:
        """关闭主路会话，并保留 Tk 尚未消费的 writer 错误。"""
        with self._record_pair_lock:
            snap = self._rec.snapshot()
            path = self._rec.close_session()
            if snap.last_error is not None:
                self._pending_record_errors.append(f"第一路：{snap.last_error}")
            return path

    def _close_recording_pair(self) -> tuple[Path | None, Path | None]:
        """关闭双路会话并把有效双摄片段提交到统一后处理队列。"""
        finalization = App._finalize_and_dispatch_recording_pair(self, close_session=True)
        return finalization.front_path, finalization.side_path

    def _finalize_and_dispatch_recording_pair(
        self,
        *,
        close_session: bool,
    ) -> _RecordingPairFinalization:
        """串行执行片段终结和非阻塞提交，关闭两者之间的关窗竞态窗口。"""
        finalize_lock = getattr(self, "_record_finalize_lock", None)
        if finalize_lock is None:
            finalization = App._finalize_recording_pair(
                self, close_session=close_session
            )
            App._dispatch_recording_finalization(self, finalization)
            return finalization
        with finalize_lock:
            finalization = App._finalize_recording_pair(
                self, close_session=close_session
            )
            App._dispatch_recording_finalization(self, finalization)
            return finalization

    def _finalize_recording_pair(
        self,
        *,
        close_session: bool,
    ) -> _RecordingPairFinalization:
        """锁内快照并释放双 writer；任何转码/比对都留到锁外。"""
        with self._record_pair_lock:
            snap = self._rec.snapshot()
            snap2 = self._rec2.snapshot()
            dual_active = bool(getattr(self, "_dual_active", False))
            stamp = getattr(self, "_record_stamp", None)
            record_skeleton = bool(getattr(self, "_dual_record_skeleton", False))
            # worker 线程只读缓存（主线程滑块已写入 _dual_auto_compare）。
            # 用 unbound 调用，兼容 lifecycle 测试里的 SimpleNamespace stub。
            auto_compare = App._current_auto_compare(self, prefer_ui=False)
            segment_dir = None
            if dual_active and stamp:
                base = Path(getattr(self, "_record_base_dir", outputs_dir()))
                segment_dir = base / stamp[:8] / f"record_{stamp}"

            if close_session:
                front_path = self._rec.close_session()
                side_path = self._rec2.close_session()
                self._dual_active = False
            else:
                front_path = self._rec.stop_recording()
                side_path = self._rec2.stop_recording()

            # 旧测试桩/无后处理器嵌入仍沿用 pending error；正式双摄由 result.json/UI 承载。
            if close_session and not hasattr(self, "_record_postprocessor"):
                if snap.last_error is not None:
                    self._pending_record_errors.append(f"第一路：{snap.last_error}")
                if snap2.last_error is not None:
                    self._pending_record_errors.append(f"第二路：{snap2.last_error}")

            return _RecordingPairFinalization(
                dual_active=dual_active,
                stamp=stamp,
                segment_dir=segment_dir,
                record_skeleton=record_skeleton,
                auto_compare=auto_compare,
                segment_started=(
                    snap.state in {"recording", "paused"}
                    or snap2.state in {"recording", "paused"}
                ),
                front_path=front_path,
                side_path=side_path,
                front_frames=int(snap.frames_written),
                side_frames=int(snap2.frames_written),
                front_error=snap.last_error,
                side_error=snap2.last_error,
            )

    def _dispatch_recording_finalization(
        self,
        finalization: _RecordingPairFinalization,
    ) -> None:
        """锁外转码单摄，或把双摄片段提交到 FIFO 后处理器。"""
        submitted = getattr(self, "_submitted_record_segments", None)
        segment_id = (
            f"record_{finalization.stamp}" if finalization.stamp is not None else None
        )
        if not finalization.dual_active:
            # 显式结束、主停止、worker finally 和关窗可能先后到达这里。双摄片段
            # 一旦提交，后续 close_session 的快照不得再被当成单摄录像重复转码。
            if segment_id is not None and submitted is not None and segment_id in submitted:
                return
            self._transcode_async(finalization.front_path)
            self._transcode_async(finalization.side_path)
            return
        # 学生练习：记下片段目录/id；就绪态等后处理终态再开（见 _on_student_postprocess_update）
        if (
            bool(getattr(self, "_student_practice_active", False))
            and finalization.segment_dir is not None
            and finalization.has_activity
        ):
            self._student_last_segment_dir = Path(finalization.segment_dir)
            self._student_last_segment_id = (
                f"record_{finalization.stamp}"
                if finalization.stamp
                else Path(finalization.segment_dir).name
            )
            self._student_segment_ready = False
            self._student_front_video = None
            self._student_side_video = None

        processor = getattr(self, "_record_postprocessor", None)
        if processor is None:
            self._transcode_async(finalization.front_path)
            self._transcode_async(finalization.side_path)
            return
        if not finalization.has_activity or not finalization.stamp or finalization.segment_dir is None:
            return

        segment_id = f"record_{finalization.stamp}"
        if submitted is not None and segment_id in submitted:
            return

        # 考试放弃片段：落盘但不入比对队列
        if bool(getattr(self, "_exam_discard_next", False)):
            return

        front_templates, side_templates = self._current_template_lists()
        front_template = front_templates[0] if front_templates else Path("")
        side_template = side_templates[0] if side_templates else Path("")
        front_source = finalization.front_path or (finalization.segment_dir / "front.mp4")
        side_source = finalization.side_path or (finalization.segment_dir / "side.mp4")
        front_frames = int(finalization.front_frames)
        side_frames = int(finalization.side_frames)
        exam_row = getattr(self, "_exam_pending_row", None)
        exam_run_id = getattr(self, "_exam_run_id", None)
        auto_compare = bool(finalization.auto_compare)
        if exam_row is not None:
            auto_compare = True
        # 学生练习：强制仅录制（转码落盘，不走 DTW 比对）
        if bool(getattr(self, "_student_practice_active", False)):
            auto_compare = False

        previous_latest = getattr(self, "_latest_compare_segment_id", None)
        # submit() 会同步发布 queued，消费者也可能立即发布终态。先建立 guard，
        # 避免 worker finally 提交时 Tk tick 把该任务的首批更新当成旧片段丢弃。
        self._latest_compare_segment_id = segment_id
        # 防双提交：考试路径可能异步裁剪后再 submit，必须先占位
        if submitted is not None:
            submitted.add(segment_id)

        job_kwargs = dict(
            segment_id=segment_id,
            segment_dir=finalization.segment_dir,
            front_source=Path(front_source),
            side_source=Path(side_source),
            front_frames=front_frames,
            side_frames=side_frames,
            front_template=front_template,
            side_template=side_template,
            front_templates=tuple(front_templates),
            side_templates=tuple(side_templates),
            record_skeleton=finalization.record_skeleton,
            auto_compare=auto_compare,
            front_error=finalization.front_error,
            side_error=finalization.side_error,
            exam_run_id=str(exam_run_id) if exam_row is not None and exam_run_id else None,
            student_id=getattr(getattr(exam_row, "candidate", None), "student_id", None)
            if exam_row is not None
            else None,
            student_name=getattr(getattr(exam_row, "candidate", None), "name", None)
            if exam_row is not None
            else None,
            student_order=getattr(getattr(exam_row, "candidate", None), "order", None)
            if exam_row is not None
            else None,
            attempt_index=getattr(exam_row, "attempt_index", None) if exam_row is not None else None,
            exam_warnings=(),
        )

        if exam_row is not None:
            # 裁剪+重编码移出 Tk 主线程，避免冻结 UI / 阻塞下一位叫号
            def _exam_clip_then_submit(
                kw: dict = job_kwargs,
                src_f: Path = Path(front_source),
                src_s: Path = Path(side_source),
                ff: int = front_frames,
                sf: int = side_frames,
                out_dir: Path = Path(finalization.segment_dir),
                prev_latest: str | None = previous_latest,
                seg_id: str = segment_id,
            ) -> None:
                warnings: list[str] = []
                front_p, side_p = src_f, src_s
                f_frames, s_frames = ff, sf
                try:
                    from core.exam_clip import prepare_exam_pair

                    clip = prepare_exam_pair(
                        src_f,
                        src_s,
                        front_frames=ff,
                        side_frames=sf,
                        out_dir=out_dir,
                    )
                    front_p = clip.front_path
                    side_p = clip.side_path
                    f_frames = clip.front_frames
                    s_frames = clip.side_frames
                    warnings.extend(clip.warnings)
                except Exception as exc:
                    warnings.append(f"exam_clip_failed: {exc}")
                job = DualRecordingJob(
                    **{
                        **kw,
                        "front_source": Path(front_p),
                        "side_source": Path(side_p),
                        "front_frames": int(f_frames),
                        "side_frames": int(s_frames),
                        "exam_warnings": tuple(warnings),
                    }
                )
                # 关窗过程中仍尽量提交；失败则通知台账（含 sink）
                submitted_ok = False
                try:
                    submitted_ok = bool(processor.submit(job))
                except Exception:
                    submitted_ok = False
                if not submitted_ok:
                    if getattr(self, "_latest_compare_segment_id", None) == seg_id:
                        self._latest_compare_segment_id = prev_latest
                    try:
                        App._post_recording_compare_update(
                            self,
                            PostprocessUpdate(
                                segment_id=seg_id,
                                status="failed",
                                message="后处理队列提交失败",
                                error_code="submit_failed",
                            ),
                        )
                    except Exception:
                        pass
                with getattr(self, "_exam_clip_lock", threading.Lock()):
                    threads = getattr(self, "_exam_clip_threads", None)
                    if threads is not None:
                        threads.discard(threading.current_thread())

            t = threading.Thread(
                target=_exam_clip_then_submit,
                name=f"exam-clip-{segment_id}",
                daemon=True,
            )
            with getattr(self, "_exam_clip_lock", threading.Lock()):
                threads = getattr(self, "_exam_clip_threads", None)
                if threads is None:
                    self._exam_clip_threads = set()
                    threads = self._exam_clip_threads
                # 清理已结束线程
                dead = {x for x in threads if not x.is_alive()}
                threads.difference_update(dead)
                threads.add(t)
            t.start()
            return

        job = DualRecordingJob(**job_kwargs)
        if not processor.submit(job):
            if submitted is not None:
                submitted.discard(segment_id)
            if getattr(self, "_latest_compare_segment_id", None) == segment_id:
                self._latest_compare_segment_id = previous_latest

    def _join_exam_clip_threads(self, timeout: float = 30.0) -> None:
        """关窗/停止前等待考试裁剪线程，避免裁剪后 submit 必失败且无终态。"""
        with getattr(self, "_exam_clip_lock", threading.Lock()):
            threads = list(getattr(self, "_exam_clip_threads", set()) or set())
        deadline = time.monotonic() + max(0.0, float(timeout))
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                t.join(timeout=remaining)
            except Exception:
                pass
        with getattr(self, "_exam_clip_lock", threading.Lock()):
            threads_set = getattr(self, "_exam_clip_threads", None)
            if threads_set is not None:
                threads_set.difference_update({t for t in threads_set if not t.is_alive()})

    def _register_exam_scorebook_sink(self, scorebook) -> None:
        """登记关面板后的台账 sink；多场可并存，禁止覆盖丢失。"""
        sinks = getattr(self, "_exam_scorebook_sinks", None)
        if sinks is None:
            self._exam_scorebook_sinks = []
            sinks = self._exam_scorebook_sinks
        if scorebook is not None and scorebook not in sinks:
            sinks.append(scorebook)

    def _apply_exam_postprocess_update(self, update: PostprocessUpdate) -> None:
        """面板 + 所有 sink 尝试回填；新面板未匹配时不得吃掉旧场次更新。"""
        from apps.exam_panel import apply_postprocess_update_to_scorebook

        panel = getattr(self, "_exam_panel", None)
        panel_book = getattr(panel, "scorebook", None) if panel is not None else None
        if panel is not None and hasattr(panel, "on_postprocess_update"):
            try:
                panel.on_postprocess_update(update)
            except Exception:
                pass

        sinks = list(getattr(self, "_exam_scorebook_sinks", None) or [])
        still: list = []
        for sink in sinks:
            # 当前面板 scorebook 已由 on_postprocess_update 处理，避免重复
            if panel_book is not None and sink is panel_book:
                still.append(sink)
                continue
            try:
                apply_postprocess_update_to_scorebook(sink, update)
                if not any(r.status == "processing" for r in sink.rows):
                    try:
                        sink.flush()
                    except Exception:
                        pass
                still.append(sink)
            except Exception:
                still.append(sink)
        self._exam_scorebook_sinks = still

    def _post_recording_compare_update(self, update: PostprocessUpdate) -> None:
        """后处理线程只把不可变状态写入队列，不直接调用任何 Tk API。"""
        # 关窗时仍允许考试台账回填（panel 或 sinks），避免裁剪线程晚到的终态被丢弃
        if getattr(self, "_closing", False):
            has_exam = (
                getattr(self, "_exam_panel", None) is not None
                or bool(getattr(self, "_exam_scorebook_sinks", None))
            )
            if not has_exam:
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
            # 学生练习：等后处理终态 + 最终视频路径再允许评判
            if not closing and bool(getattr(self, "_student_practice_active", False)):
                try:
                    App._on_student_postprocess_update(self, update)
                except Exception:
                    pass
            # 考试台账：面板或关窗后 sink，所有片段终态/过程态都要回填
            App._apply_exam_postprocess_update(self, update)

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

        with self._record_pair_lock:
            snap = self._rec.snapshot()
            snap2 = self._rec2.snapshot()
            dual_active = self._dual_active
            errors = list(self._pending_record_errors)
            self._pending_record_errors.clear()
        if snap.last_error is not None:
            error = f"第一路：{snap.last_error}"
            if error not in errors:
                errors.append(error)
        if dual_active and snap2.last_error is not None:
            error = f"第二路：{snap2.last_error}"
            if error not in errors:
                errors.append(error)

        if errors:
            if not self._record_error_shown:
                self._record_error_shown = True
                # 双摄录制必须保持成对状态。任一路 writer 失败时结束两路当前片段，
                # 避免下一次暂停/继续后两个控制器进入相反状态。
                finalization = App._finalize_and_dispatch_recording_pair(
                    self, close_session=False
                )
                record_btn = getattr(self, "record_btn", None)
                if record_btn is not None:
                    record_btn.configure(text=RECORD_BTN_TEXT["idle"])
                self._sync_record_stop_enabled("idle")
                self.recording_status_var.set("")
                if not finalization.dual_active or not hasattr(self, "_record_postprocessor"):
                    messagebox.showerror("录制失败", "录制发生错误：\n" + "\n".join(errors))
            return

        # 无错误：复位守卫，下次错误可再次提示。
        self._record_error_shown = False

        if snap.state in ("recording", "paused"):
            path_text = str(snap.result_path) if snap.result_path is not None else "（准备中）"
            if dual_active:
                path_text2 = (
                    str(snap2.result_path) if snap2.result_path is not None else "（准备中）"
                )
                path_text = f"正面 {path_text}；侧面 {path_text2}"
            label = "录制中" if snap.state == "recording" else "已暂停"
            self.recording_status_var.set(f"{label}：{path_text}")
        else:  # idle
            self.recording_status_var.set("")
        # 「结束录制」按钮可用性跟随真实状态联动（覆盖 worker 端错误复位等情形）。
        self._sync_record_stop_enabled(snap.state)

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
        running = bool(self._worker and self._worker.is_alive())
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

        pool = getattr(self, "_camera_warmup_pool", None)

        def _run() -> None:
            ok = True
            entries: list[CameraEntry] = []
            try:
                if pool is not None:
                    pool.wait_released(_CAMERA_RELEASE_WAIT_S)
                entries = enumerate_cameras()
            except Exception:
                ok = False
            if getattr(self, "_closing", False):
                return
            result_queue = self._camera_enum_result_queue
            while True:
                try:
                    result_queue.get_nowait()
                except Empty:
                    break
            result_queue.put((entries, ok))

        threading.Thread(target=_run, daemon=True).start()

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
        if self._worker and self._worker.is_alive():
            return
        if self._enum_busy.is_set():
            return
        self._release_camera_warmups()
        self._start_enumeration()

    def _release_camera_warmups(self) -> None:
        """放弃所有预热 reader；释放在后台完成，重开同编号时由池等待真实释放。"""
        pool = getattr(self, "_camera_warmup_pool", None)
        if pool is None:
            return
        try:
            pool.release_all()
        except Exception:
            pass

    def _camera_capture_size(self) -> tuple[int, int]:
        """当前模式的请求尺寸；只在主线程构造 spec 时读取一次，随后冻结。"""
        if bool(getattr(self, "_student_practice_active", False)):
            return _STUDENT_CAPTURE_SIZE
        return _PREVIEW_CAPTURE_SIZE

    def _camera_spec(self, index: int) -> CaptureSpec:
        width, height = App._camera_capture_size(self)
        return CaptureSpec(int(index), width, height)

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
        running = bool(self._worker and self._worker.is_alive())
        if running and not allow_running:
            return
        pool = getattr(self, "_camera_warmup_pool", None)
        if pool is None:
            return
        primary, secondary = App._selected_camera_indices(self)
        try:
            pool.sync(
                None if primary is None else App._camera_spec(self, primary),
                None if secondary is None else App._camera_spec(self, secondary),
            )
        except Exception as exc:
            App._release_camera_warmups(self)
            self._post_status(f"摄像头预热失败：{exc}")

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
        # 主输入源也是摄像头时才有意义，这里是唯一收敛点：调用方（_start/_worker_loop）
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
        )

    @staticmethod
    def _drain_queue(queue: Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def _begin_preview_session(
        self, state: UiState, *, start_click: float | None = None
    ) -> UiState:
        start_click = time.monotonic() if start_click is None else float(start_click)
        if not hasattr(self, "_dual_preview_queue"):
            self._dual_preview_queue = Queue(maxsize=1)
        if not hasattr(self, "_dual_preview_lock"):
            self._dual_preview_lock = threading.Lock()
        if not hasattr(self, "_dual_render_lock"):
            self._dual_render_lock = threading.Lock()
        if not hasattr(self, "_dual_metrics_lock"):
            self._dual_metrics_lock = threading.Lock()
        if not hasattr(self, "_dual_startup_metrics"):
            self._dual_startup_metrics = {}
        if not hasattr(self, "_dual_first_render_events"):
            self._dual_first_render_events = {}
        if not hasattr(self, "_queue"):
            self._queue = Queue(maxsize=1)
        if not hasattr(self, "_queue2"):
            self._queue2 = Queue(maxsize=1)
        self._session_generation = getattr(self, "_session_generation", 0) + 1
        generation = self._session_generation
        self._current_session_generation = generation
        if isinstance(state, UiState):
            state = replace(
                state,
                session_generation=generation,
                start_click=start_click,
            )
        else:
            state.session_generation = generation
            state.start_click = start_click

        with self._dual_preview_lock:
            App._drain_queue(self._dual_preview_queue)
            self._dual_preview_stage = "raw"
            self._dual_recording_ready = not bool(state.source2)
            self._active_dual_generation = generation if state.source2 else 0
        if state.source2:
            App._drain_queue(self._queue)
            App._drain_queue(self._queue2)
            with self._dual_metrics_lock:
                self._dual_startup_metrics[generation] = _DualStartupMetrics(
                    session_generation=generation,
                    primary_index=int(state.source),
                    secondary_index=int(state.source2),
                    record_skeleton=bool(state.record_skeleton),
                    start_click=start_click,
                    outcome="starting",
                )
                self._dual_first_render_events[generation] = threading.Event()
        return state

    def _invalidate_dual_preview(self, generation: int | None = None) -> None:
        lock = getattr(self, "_dual_preview_lock", None)
        if lock is None:
            self._active_dual_generation = 0
            self._dual_recording_ready = False
            return
        render_lock = getattr(self, "_dual_render_lock", None)

        def invalidate_locked() -> None:
            with lock:
                if (
                    generation is not None
                    and self._active_dual_generation != generation
                ):
                    return
                self._active_dual_generation = 0
                self._dual_recording_ready = False
                App._drain_queue(self._dual_preview_queue)
                event = getattr(self, "_dual_first_render_events", {}).get(
                    generation or self._current_session_generation
                )
                if event is not None:
                    event.set()

        if render_lock is None:
            invalidate_locked()
        else:
            with render_lock:
                invalidate_locked()

    def _open_camera_process(
        self,
        index: int,
        *,
        width: int,
        height: int,
        stop_event: threading.Event,
    ):
        """池的唯一打开入口：每路一个 spawn 采集进程，调用时解析模块级 open_camera。"""
        return open_camera(index, width=width, height=height, stop_event=stop_event)

    @staticmethod
    def _session_spec(state: UiState, source: str) -> CaptureSpec:
        width, height = getattr(state, "capture_size", _PREVIEW_CAPTURE_SIZE)
        return CaptureSpec(int(source), int(width), int(height))

    def _cancel_session_warmups(self, generations: tuple[int, ...]) -> None:
        """失败会话只取消自己观察到的预热代次，不误伤之后重新声明的预热。"""
        pool = getattr(self, "_camera_warmup_pool", None)
        if pool is None:
            return
        for role, generation in zip((PRIMARY, SECONDARY), generations):
            try:
                pool.cancel(role, generation=generation)
            except Exception:
                pass

    def _open_session_camera(self, state: UiState):
        """单摄：等池内 reader 出有效首帧后接管为 lease；停止/超时由池回收。"""
        pool = self._camera_warmup_pool
        spec = App._session_spec(state, state.source)
        ready = pool.wait_one(
            spec, timeout=_CAMERA_START_TIMEOUT_S, stop_event=self._stop_evt
        )
        return pool.claim_one(
            spec,
            timeout=_CAMERA_CLAIM_TIMEOUT_S,
            expected_generation=ready.generation,
            stop_event=self._stop_evt,
        )

    def _release_session_camera(self, capture) -> str | None:
        """释放会话 lease；失败时设备保持隔离，返回可见原因。"""
        try:
            capture.release()
        except Exception as exc:
            traceback.print_exc()
            return f"摄像头释放失败：{exc}"
        return None

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            # _post_done 是 worker 的最后一步且已归还 lease；此时线程只剩返回，可短暂 join。
            if getattr(self, "_current_session_generation", 0) == 0:
                self._worker.join(timeout=0.2)
            if self._worker.is_alive():
                self.status_var.set("上一会话仍在收尾，请稍候再点「开始」")
                return
        start_click = time.monotonic()

        try:
            state = self._collect_state()
        except Exception as e:
            messagebox.showerror("配置错误", str(e))
            return

        state = self._begin_preview_session(state, start_click=start_click)
        _log_camera_event(
            {
                "event": "session_start",
                "session": getattr(state, "session_generation", None),
                "primary": getattr(state, "source", None),
                "secondary": getattr(state, "source2", None),
                "requested": "x".join(
                    str(v) for v in getattr(state, "capture_size", _PREVIEW_CAPTURE_SIZE)
                ),
                "student": bool(getattr(self, "_student_practice_active", False)),
            }
        )

        self._stop_evt.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_refresh_enabled()
        self.status_var.set("启动中…（首次运行可能需要下载模型）")
        self.actions_var.set("-")
        self.match_var.set("识别：待机")
        self.progress_var.set(0.0)
        self.progress_text_var.set("")
        self.progress_bar.configure(mode="determinate", maximum=100.0, value=0.0)

        # 双摄像头双面视图（issue #58）：与 _worker_loop_dual_camera 的分流条件同源
        # （state.source2 非空 ⇒ _collect_state 已保证 source 是摄像头），显示/隐藏/
        # 分流三处用同一个信号，不再各自重复判断。
        self._set_dual_preview_visible(bool(state.source2))

        # 启动时把 UI 旋转同步到 runtime，供点击预览/worker 共用。
        App._set_runtime_rotate(
            self,
            int(getattr(state, "rotate", 0) or 0),
            which="primary",
            sync_combo=True,
            announce=False,
        )
        App._set_runtime_rotate(
            self,
            int(getattr(state, "rotate2", 0) or 0),
            which="secondary",
            sync_combo=True,
            announce=False,
        )
        try:
            self._dual_auto_compare = App._current_auto_compare(self)
        except Exception:
            self._dual_auto_compare = bool(getattr(state, "auto_compare", True))

        self._worker = threading.Thread(target=self._worker_loop, args=(state,), daemon=True)
        self._worker.start()
        # 集中刷新运行态控件（禁用 Camera/Model_Selector、启用 Record_Toggle 等，需求 4.3 等）。
        self._set_running_controls(True)

    def _stop(self) -> None:
        # 先同步考试状态机：否则 occupancy 退出后仍停在 recording，指针/手动锁不解除
        self._student_presence_gate = None
        self._student_pending_record = False
        panel = getattr(self, "_exam_panel", None)
        if panel is not None:
            try:
                panel.on_session_stop()
            except Exception:
                pass
        dual_active = bool(getattr(self, "_active_dual_generation", 0))
        _log_camera_event(
            {
                "event": "stop_requested",
                "session": getattr(self, "_current_session_generation", 0),
            }
        )
        self._stop_evt.set()
        self._dual_recording_ready = False
        if dual_active:
            self._invalidate_dual_preview()
        record_btn = getattr(self, "record_btn", None)
        if record_btn is not None:
            try:
                record_btn.configure(state="disabled")
            except Exception:
                pass
        self.stop_btn.configure(state="disabled")
        self.status_var.set("正在停止…")

    def _on_close(self) -> None:
        if self._closing:
            return
        # 1) 先中止考试、移交当前面板 scorebook 到 sinks（_closing 仍为 False，discard 可走）
        #    root.destroy 不会触发面板 WM_DELETE_WINDOW，必须 prepare_for_app_close。
        panel = getattr(self, "_exam_panel", None)
        if panel is not None:
            try:
                if hasattr(panel, "prepare_for_app_close"):
                    panel.prepare_for_app_close()
                else:
                    panel.on_session_stop()
                    book = getattr(panel, "scorebook", None)
                    if book is not None:
                        App._register_exam_scorebook_sink(self, book)
                        try:
                            book.flush()
                        except Exception:
                            pass
                        panel.scorebook = None
            except Exception:
                pass
        elif bool(getattr(self, "_exam_active", False)):
            try:
                App._end_recording_segment(self, discard=True)
            except Exception:
                pass
        if hasattr(self, "feedback_controls"):
            self.feedback_controls.close()
        self._active_feedback_token = None
        self._student_presence_gate = None
        self._student_pending_record = False
        announcer = getattr(self, "_student_announcer", None)
        if announcer is not None:
            announcer.close()
            self._student_announcer = None
        # 2) 再进入关闭态
        self._closing = True
        self._stop_evt.set()
        self._dual_recording_ready = False
        App._invalidate_dual_preview(self)
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
        # 考试裁剪 / sink（含刚移交的面板台账）落盘需要更长预算
        clip_alive = False
        with getattr(self, "_exam_clip_lock", threading.Lock()):
            threads = getattr(self, "_exam_clip_threads", None) or set()
            clip_alive = any(t.is_alive() for t in threads)
        sinks = list(getattr(self, "_exam_scorebook_sinks", None) or [])
        sink_pending = bool(sinks)
        has_processing = any(
            any(getattr(r, "status", None) == "processing" for r in getattr(s, "rows", []))
            for s in sinks
        )
        extra = 0.0
        if clip_alive:
            extra = max(extra, 30.0)
        if sink_pending:
            extra = max(extra, 8.0)
        if has_processing:
            extra = max(extra, 12.0)
        self._close_deadline = time.monotonic() + _CLOSE_JOIN_TIMEOUT_S + extra
        postprocessor = getattr(self, "_record_postprocessor", None)
        finalize_lock = getattr(self, "_record_finalize_lock", None)

        def _prepare_close() -> None:
            try:
                def _finish_current_then_cancel() -> None:
                    # 考试 abort 已在 _closing 前 discard；此处只收尾仍在录的普通双摄片段。
                    # 若考试仍标记 discard（异常路径），保留 _exam_discard_next。
                    if bool(getattr(self, "_dual_active", False)):
                        App._finalize_and_dispatch_recording_pair(
                            self, close_session=False
                        )
                    # 先等考试裁剪线程尽量 submit，再 cancel 后处理队列
                    App._join_exam_clip_threads(self, timeout=28.0)
                    if postprocessor is not None:
                        postprocessor.cancel_all()

                if finalize_lock is None:
                    _finish_current_then_cancel()
                else:
                    with finalize_lock:
                        _finish_current_then_cancel()
            finally:
                pool = getattr(self, "_camera_warmup_pool", None)
                if pool is not None:
                    try:
                        pool.close()
                        # 关窗预算内等待采集进程真实退出；超时由诊断日志记录。
                        if not pool.wait_released(_CAMERA_RELEASE_WAIT_S):
                            _log_camera_event({"event": "close_release_timeout"})
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
        """分片 join 采集/转码/比对 worker；全部结束或到达期限后销毁 Tk 窗口。"""
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
        capture_worker = self._worker
        if (
            capture_worker is not None
            and capture_worker is not current
            and capture_worker.is_alive()
            and remaining > 0.0
        ):
            capture_worker.join(timeout=min(_CLOSE_JOIN_SLICE_S, remaining))

        with self._transcode_lock:
            finished = {worker for worker in self._transcode_workers if not worker.is_alive()}
            self._transcode_workers.difference_update(finished)
            transcode_workers = tuple(self._transcode_workers)

        remaining = deadline - time.monotonic()
        if transcode_workers and remaining > 0.0:
            join_budget = min(_CLOSE_JOIN_SLICE_S, remaining)
            per_worker = join_budget / len(transcode_workers)
            for worker in transcode_workers:
                if worker is not current:
                    worker.join(timeout=per_worker)

        postprocessor = getattr(self, "_record_postprocessor", None)
        remaining = deadline - time.monotonic()
        close_prepare_alive = bool(
            close_prepare_worker is not None and close_prepare_worker.is_alive()
        )
        if postprocessor is not None and not close_prepare_alive:
            # 即使采集/转码已耗尽关窗预算，也要以 0 秒 close 入队退出哨兵；
            # 否则 cancel_all 后消费者会永久阻塞在 queue.get()。
            postprocessor.close(min(_CLOSE_JOIN_SLICE_S, max(0.0, remaining)))

        # capture 的 finally 可能在上面 join 期间新建转码线程。决定销毁前
        # 必须重新快照两类 worker，不复用 join 前的旧集合。
        capture_alive = bool(self._worker and self._worker.is_alive())
        with self._transcode_lock:
            finished = {worker for worker in self._transcode_workers if not worker.is_alive()}
            self._transcode_workers.difference_update(finished)
            transcodes_alive = bool(self._transcode_workers)
        postprocess_worker = getattr(postprocessor, "_worker", None)
        postprocess_alive = bool(
            postprocess_worker is not None and postprocess_worker.is_alive()
        )

        if (
            close_prepare_alive
            or capture_alive
            or transcodes_alive
            or postprocess_alive
        ) and time.monotonic() < deadline:
            try:
                self.root.after(_CLOSE_POLL_MS, self._poll_close_workers)
            except (TclError, RuntimeError):
                pass
            return

        # 最后一轮消费比对更新，并把所有考试 sink 落盘关闭
        try:
            App._drain_recording_compare_updates(self)
        except Exception:
            pass
        for sink in list(getattr(self, "_exam_scorebook_sinks", None) or []):
            try:
                sink.flush()
                sink.close()
            except Exception:
                pass
        self._exam_scorebook_sinks = []

        try:
            self.root.destroy()
        except (TclError, RuntimeError):
            pass

    def _worker_loop(self, state: UiState) -> None:
        source = state.source
        is_file = not source.isdigit()

        if is_file:
            # 当前主界面虽以摄像头为主，仍保留旧视频输入兼容；进入文件会话前确保
            # 选择阶段持有的摄像头资源全部释放。
            self._release_camera_warmups()

        # 双摄 worker 自己负责 wait/snapshot/raw pump/claim，确保模型构造期间 pool
        # reader 继续提供裸帧；此处不得提前 claim 停掉 reader。
        if state.source2 and not is_file:
            self._worker_loop_dual_camera(state)
            return

        # 摄像头 + 多 worker + 开启骨架：让「打开摄像头」与「各 worker 加载模型」并行发生，
        # 而不是串行等摄像头开好再建模型（缩短点击→首帧）。骨架关闭时无需并行引擎，
        # 落入下方串行裸帧路径。open 下放到分支内部，故此处提前分流。
        if (
            (not is_file)
            and (not state.source2)
            and state.workers > 1
            and bool(state.record_skeleton)
        ):
            self._worker_loop_parallel_camera(state)
            return

        try:
            if source.isdigit():
                # 预热 reader 出首帧后接管；无预热时池在后台按本次 spec 冷启动。
                cap = App._open_session_camera(self, state)
            else:
                cap = cv2.VideoCapture(source)
        except CameraWarmupStopped:
            self._post_status("已停止")
            self._post_done(session_generation=getattr(state, "session_generation", None))
            return
        except Exception as exc:
            traceback.print_exc()
            final_status = f"打开输入源失败：{exc}"
            self._post_status(final_status)
            self._post_done(
                session_generation=getattr(state, "session_generation", None),
                final_status=final_status,
                rewarm=False,
            )
            return

        if not cap.isOpened():
            cap.release()
            final_status = f"无法打开输入源：{source}"
            self._post_status(final_status)
            self._post_done(
                session_generation=getattr(state, "session_generation", None),
                final_status=final_status,
            )
            return

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_for_ts = src_fps if (is_file and src_fps > 1e-3) else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_file else 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        if is_file and state.workers > 1:
            self._worker_loop_parallel_video(state, cap, fps_for_ts, total)
            return

        # 仅实时摄像头使用旋转选项；离线视频保持原尺寸和既有处理行为。
        # 旋转读 runtime（点击预览可运行中改），不再冻结 UiState.rotate。
        session_size = (
            (w, h)
            if is_file
            else _rotated_size(w, h, self._get_runtime_rotate("primary"))
        )
        self._rec.begin_session(fps=fps_for_ts, size=session_size)
        # 全局骨架开关：摄像头默认关 → 预览/录制裸帧、跳过推理；离线文件仍始终跑 annotate。
        draw_skeleton = bool(is_file or state.record_skeleton)
        matcher = (
            self._build_online_matcher()
            if (draw_skeleton and (not is_file) and state.online_match_enabled)
            else None
        )
        smoother = PreviewLandmarkSmoother() if (draw_skeleton and not is_file) else None

        pipe = None
        failed = False
        final_status: str | None = None
        try:
            if draw_skeleton:
                try:
                    models_dir_path = models_dir()
                    pipe = MediaPipePipeline(
                        models_dir=models_dir_path,
                        cfg=PipelineConfig(
                            pose_variant=state.pose_variant,
                            running_mode="video",
                            enable_hands=state.enable_hands,
                        ),
                    )
                except Exception as e:
                    failed = True
                    final_status = f"初始化失败：{e}"
                    self._post_status(final_status)
                    return

            t0 = time.monotonic()
            frame_count = 0
            actual_size_checked = is_file
            self._post_status("运行中…" if draw_skeleton else "运行中…（裸帧预览）")
            self._post_progress(0, total)

            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not is_file and not self._stop_evt.is_set():
                        raise RuntimeError("摄像头读帧失败，请检查连接后重新开始")
                    self._stop_evt.set()
                    break

                if not is_file:
                    # 在推理前转正，保证 landmarks、预览和录制使用同一坐标系。
                    frame = _apply_rotation(
                        frame, self._get_runtime_rotate("primary")
                    )
                    self._rec.update_session_size(
                        size=(int(frame.shape[1]), int(frame.shape[0]))
                    )
                    actual_size_checked = True

                if draw_skeleton and pipe is not None:
                    ts = pipe.next_timestamp_ms(is_file=is_file, fps_for_ts=fps_for_ts)
                    if is_file:
                        # 文件 VIDEO 模式保留 annotate()，包括其内部 frame index 自增语义。
                        annotated, actions = pipe.annotate(frame, timestamp_ms=ts)
                    else:
                        pose_landmarks, hands = pipe.infer(frame, timestamp_ms=ts)
                        self._feed_online_matcher(
                            matcher, pose_landmarks, ts, frame_count
                        )
                        smoothed = smoother.feed(pose_landmarks, timestamp_ms=ts)
                        annotated, actions = pipe.draw(
                            frame,
                            smoothed,
                            hands,
                            action_pose_landmarks=pose_landmarks,
                        )
                else:
                    annotated, actions = frame.copy(), []

                frame_count += 1
                if is_file and total > 0 and (frame_count % 5 == 0 or frame_count == total):
                    self._post_progress(frame_count, total)
                fps = frame_count / max(1e-6, (time.monotonic() - t0))
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "点击“停止”结束",
                    (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                self._write_rec_frame(self._rec, annotated)

                if actions:
                    actions_text = ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
                else:
                    actions_text = "-"
                self._post_frame(annotated, actions_text)

            cv2.destroyAllWindows()

            self._post_status("已停止")
            self._post_progress(frame_count, total)
        except Exception as exc:
            failed = not self._stop_evt.is_set()
            if failed:
                traceback.print_exc()
                final_status = f"运行失败：{exc}"
                self._post_status(final_status)
            else:
                self._post_status("已停止")
        finally:
            # 覆盖正常结束 / 停止 / 异常：释放 writer 并复位录制状态。
            self._stop_evt.set()
            release_error = App._release_session_camera(self, cap)
            if release_error is not None:
                failed = True
                final_status = final_status or release_error
                self._post_status(final_status)
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
            if matcher is not None:
                matcher.close()
            self._transcode_async(self._close_primary_recording_session())
            self._post_done(
                session_generation=getattr(state, "session_generation", None),
                final_status=final_status,
                rewarm=not failed,
            )

    def _worker_loop_dual_camera(self, state: UiState) -> None:
        """双摄 wait→裸帧→模型→claim→正式循环，所有资源在单一 finally 收敛。"""
        generation = int(
            getattr(state, "session_generation", 0)
            or getattr(self, "_current_session_generation", 0)
        )
        primary_spec = App._session_spec(state, state.source)
        secondary_spec = App._session_spec(state, state.source2)
        pool = self._camera_warmup_pool
        cap = None
        cap2 = None
        pipe = None
        pipe2 = None
        pool_claimed = False
        warm_generations: tuple[int, ...] = ()
        recording_pair_started = False
        raw_pump_stop = threading.Event()
        raw_pump: threading.Thread | None = None
        raw_pump_errors: list[BaseException] = []
        frame_count = 0
        phase = "warmup"
        outcome = "failed"
        error_type: str | None = None
        final_status: str | None = None
        occupancy_pipe = None

        def post_raw_pair(pair) -> bool:
            frame = _apply_rotation(
                pair[0].frame, self._get_runtime_rotate("primary")
            )
            frame2 = _apply_rotation(
                pair[1].frame, self._get_runtime_rotate("secondary")
            )
            return self._post_dual_frame_pair(
                frame,
                "-",
                frame2,
                "-",
                session_generation=generation,
                stage="raw",
            )

        def stop_raw_pump() -> None:
            raw_pump_stop.set()
            if (
                raw_pump is not None
                and raw_pump is not threading.current_thread()
                and raw_pump.is_alive()
            ):
                raw_pump.join(timeout=1.0)

        try:
            pair = pool.wait_pair(
                primary_spec,
                secondary_spec,
                timeout=_CAMERA_START_TIMEOUT_S,
                stop_event=self._stop_evt,
            )
            warm_generations = (pair[0].generation, pair[1].generation)
            self._mark_dual_startup_metric(generation, "pair_ready")
            if self._stop_evt.is_set():
                raise CameraWarmupStopped("camera warmup was stopped")

            size = (
                int(
                    _apply_rotation(
                        pair[0].frame, self._get_runtime_rotate("primary")
                    ).shape[1]
                ),
                int(
                    _apply_rotation(
                        pair[0].frame, self._get_runtime_rotate("primary")
                    ).shape[0]
                ),
            )
            size2 = (
                int(
                    _apply_rotation(
                        pair[1].frame, self._get_runtime_rotate("secondary")
                    ).shape[1]
                ),
                int(
                    _apply_rotation(
                        pair[1].frame, self._get_runtime_rotate("secondary")
                    ).shape[0]
                ),
            )
            self._post_dual_preview_layout(_choose_dual_preview_layout(size, size2))
            if not state.record_skeleton:
                self._mark_dual_startup_metric(generation, "pipeline_ready")
            post_raw_pair(pair)

            cursor = [pair[0].sequence, pair[1].sequence]
            if state.record_skeleton:
                def pump_raw_frames() -> None:
                    try:
                        while not raw_pump_stop.is_set() and not self._stop_evt.is_set():
                            latest = pool.snapshot_pair(
                                primary_spec,
                                secondary_spec,
                                after=(cursor[0], cursor[1]),
                            )
                            if latest is not None:
                                cursor[0] = latest[0].sequence
                                cursor[1] = latest[1].sequence
                                post_raw_pair(latest)
                            raw_pump_stop.wait(_DUAL_RAW_PUMP_INTERVAL_S)
                    except BaseException as exc:
                        raw_pump_errors.append(exc)
                        raw_pump_stop.set()

                raw_pump = threading.Thread(
                    target=pump_raw_frames,
                    name=f"dual-raw-preview-{generation}",
                    daemon=True,
                )
                raw_pump.start()
                self._post_status("模型加载中…（双摄裸帧预览）")
                phase = "pipeline"
                models_dir_path = models_dir()
                cfg = PipelineConfig(
                    pose_variant=state.pose_variant,
                    running_mode="video",
                    enable_hands=state.enable_hands,
                )
                pipe = MediaPipePipeline(models_dir=models_dir_path, cfg=cfg)
                if self._stop_evt.is_set():
                    raise CameraWarmupStopped("camera warmup was stopped")
                pipe2 = MediaPipePipeline(models_dir=models_dir_path, cfg=cfg)
                self._mark_dual_startup_metric(generation, "pipeline_ready")
                if raw_pump_errors:
                    raise RuntimeError(str(raw_pump_errors[0])) from raw_pump_errors[0]
                first_render = getattr(
                    self, "_dual_first_render_events", {}
                ).get(generation)
                if first_render is not None:
                    deadline = time.monotonic() + 0.25
                    while (
                        not self._stop_evt.is_set()
                        and not first_render.is_set()
                        and time.monotonic() < deadline
                    ):
                        first_render.wait(
                            min(0.02, max(0.0, deadline - time.monotonic()))
                        )
                self._advance_dual_preview_stage(generation, "annotated")
            stop_raw_pump()
            if self._stop_evt.is_set():
                raise CameraWarmupStopped("camera warmup was stopped")

            phase = "claim"
            cap, cap2 = pool.claim_pair(
                primary_spec,
                secondary_spec,
                timeout=_CAMERA_CLAIM_TIMEOUT_S,
                expected_generations=warm_generations,
                stop_event=self._stop_evt,
            )
            pool_claimed = True
            if self._stop_evt.is_set():
                raise CameraWarmupStopped("camera warmup was stopped")

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            with self._record_pair_lock:
                self._dual_active = True
                self._dual_record_skeleton = bool(state.record_skeleton)
                self._dual_auto_compare = self._current_auto_compare()
                recording_pair_started = True
                self._rec.begin_session(
                    fps=30.0,
                    size=_rotated_size(w, h, self._get_runtime_rotate("primary")),
                )
                self._rec2.begin_session(
                    fps=30.0,
                    size=_rotated_size(
                        w2, h2, self._get_runtime_rotate("secondary")
                    ),
                )
            self._set_dual_startup_outcome(generation, "running")
            self._emit_dual_startup_metrics(generation)
            self._post_dual_recording_ready(generation)

            phase = "runtime"
            t0 = time.monotonic()
            last_layout: str | None = None
            self._post_status(f"运行中…（双摄像头 {w}×{h} / {w2}×{h2}）")
            self._post_progress(0, 0)

            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                ok2, frame2 = cap2.read()
                if not ok or not ok2:
                    if self._stop_evt.is_set():
                        break
                    raise RuntimeError(f"第{1 if not ok else 2}路摄像头读帧失败，请检查连接后重新开始")

                frame = _apply_rotation(
                    frame, self._get_runtime_rotate("primary")
                )
                frame2 = _apply_rotation(
                    frame2, self._get_runtime_rotate("secondary")
                )
                actual_size = (int(frame.shape[1]), int(frame.shape[0]))
                actual_size2 = (int(frame2.shape[1]), int(frame2.shape[0]))
                with self._record_pair_lock:
                    # writer 未建时可随旋转更新；已建则写盘路径 resize。
                    self._rec.update_session_size(size=actual_size)
                    self._rec2.update_session_size(size=actual_size2)
                layout = _choose_dual_preview_layout(actual_size, actual_size2)
                if layout != last_layout:
                    self._post_dual_preview_layout(layout)
                    last_layout = layout

                if state.record_skeleton:
                    ts = pipe.next_timestamp_ms(is_file=False, fps_for_ts=30.0)
                    ts2 = pipe2.next_timestamp_ms(is_file=False, fps_for_ts=30.0)
                    annotated, actions = pipe.annotate(frame, timestamp_ms=ts)
                    annotated2, actions2 = pipe2.annotate(frame2, timestamp_ms=ts2)
                    stage = "annotated"
                else:
                    annotated, actions = frame.copy(), []
                    annotated2, actions2 = frame2.copy(), []
                    stage = "raw"

                # 考试 / 学生练习共用裸帧录制 + lite pose 的站位检测。
                with self._exam_lock:
                    occ_armed = bool(self._exam_occupancy_armed) or getattr(self, "_student_presence_gate", None) is not None
                    exam_roi = self._exam_roi
                    occ_stride = int(self._exam_occupancy_stride or 3)
                if occ_armed and not state.record_skeleton:
                    try:
                        if occupancy_pipe is None:
                            occupancy_pipe = MediaPipePipeline(
                                models_dir=models_dir(),
                                cfg=PipelineConfig(
                                    pose_variant="lite",
                                    running_mode="video",
                                    enable_hands=False,
                                ),
                            )
                        if frame_count % max(1, occ_stride) == 0:
                            from core.presence_gate import hip_midpoint_in_roi

                            ots = occupancy_pipe.next_timestamp_ms(
                                is_file=False, fps_for_ts=30.0
                            )
                            sampled_at = time.monotonic()
                            pose_lms, _ = occupancy_pipe.infer(
                                frame, timestamp_ms=ots
                            )
                            present = hip_midpoint_in_roi(pose_lms, exam_roi)
                            q = getattr(self, "_exam_occupancy_queue", None)
                            if q is not None:
                                try:
                                    while q.full():
                                        try:
                                            q.get_nowait()
                                        except Empty:
                                            break
                                    q.put_nowait((bool(present), sampled_at))
                                except Exception:
                                    pass
                        # 预览叠 ROI
                        h0, w0 = annotated.shape[:2]
                        x0 = int(exam_roi[0] * w0)
                        y0 = int(exam_roi[1] * h0)
                        x1 = int(exam_roi[2] * w0)
                        y1 = int(exam_roi[3] * h0)
                        cv2.rectangle(
                            annotated, (x0, y0), (x1, y1), (0, 255, 255), 2
                        )
                    except Exception:
                        pass
                elif occupancy_pipe is not None and not occ_armed:
                    try:
                        occupancy_pipe.close()
                    except Exception:
                        pass
                    occupancy_pipe = None

                frame_count += 1
                fps = frame_count / max(1e-6, (time.monotonic() - t0))
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "点击“停止”结束",
                    (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                record_frame, record_frame2 = _select_dual_recording_frames(
                    frame,
                    frame2,
                    annotated,
                    annotated2,
                    record_skeleton=state.record_skeleton,
                )
                self._write_recording_pair(record_frame, record_frame2)
                actions_text = (
                    ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
                    if actions
                    else "-"
                )
                actions_text2 = (
                    ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions2)
                    if actions2
                    else "-"
                )
                self._post_dual_frame_pair(
                    annotated,
                    actions_text,
                    annotated2,
                    actions_text2,
                    session_generation=generation,
                    stage=stage,
                )

            outcome = "stopped"
            final_status = "已停止"
            self._post_status(final_status)
            self._post_progress(frame_count, 0)
        except CameraWarmupStopped:
            outcome = "stopped"
            final_status = "已停止"
            if not getattr(self, "_closing", False):
                self._post_status(final_status)
        except Exception as exc:
            traceback.print_exc()
            if self._stop_evt.is_set():
                outcome = "stopped"
                final_status = "已停止"
            else:
                error_type = type(exc).__name__
                prefix = {
                    "warmup": "双摄像头预热失败",
                    "pipeline": "初始化失败",
                    "claim": "双摄像头接管失败",
                    "runtime": "双摄像头运行失败",
                }.get(phase, "双摄像头启动失败")
                final_status = f"{prefix}：{exc}"
            self._post_status(final_status)
        finally:
            stop_raw_pump()
            for pipeline in (pipe, pipe2, occupancy_pipe):
                if pipeline is not None:
                    try:
                        pipeline.close()
                    except Exception:
                        pass
            # 两路 lease 一起归还；释放失败的设备由池隔离，不影响已写片段收尾。
            release_errors = [
                error
                for error in (
                    App._release_session_camera(self, capture)
                    for capture in (cap, cap2)
                    if capture is not None
                )
                if error is not None
            ]
            if not pool_claimed:
                App._cancel_session_warmups(self, warm_generations)
            if release_errors:
                outcome = "failed"
                final_status = final_status if error_type else release_errors[0]
                error_type = error_type or "CameraReleaseError"
                self._post_status(final_status)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if recording_pair_started:
                self._close_recording_pair()
            self._invalidate_dual_preview(generation)
            self._finish_dual_startup_metrics(
                generation,
                outcome=outcome,
                error_type=error_type,
            )
            self._post_done(
                session_generation=generation,
                final_status=final_status,
                rewarm=outcome != "failed",
            )

    def _worker_loop_parallel_video(
        self,
        state: UiState,
        cap: cv2.VideoCapture,
        fps_for_ts: float,
        total: int,
    ) -> None:
        # Parallel processing for offline videos using IMAGE mode pipelines.
        models_dir_path = models_dir()
        workers = max(1, int(state.workers))
        frame_q: "Queue[tuple[int, object] | None]" = Queue(maxsize=workers * 2)
        result_q: "Queue[tuple[int, object, list[str]]]" = Queue(maxsize=workers * 2)

        # 录制由 RecordingController 管理：进入帧循环前登记本会话写入参数（fps/size）。
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self._rec.begin_session(fps=fps_for_ts, size=(w, h))

        def reader() -> None:
            idx = 0
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_q.put((idx, frame))
                idx += 1
            for _ in range(workers):
                frame_q.put(None)

        def worker() -> None:
            pipe = MediaPipePipeline(
                models_dir=models_dir_path,
                cfg=PipelineConfig(
                    pose_variant=state.pose_variant,
                    running_mode="image",
                    enable_hands=state.enable_hands,
                ),
            )
            while True:
                item = frame_q.get()
                if item is None:
                    break
                if self._stop_evt.is_set():
                    continue
                idx, frame = item
                annotated, actions = pipe.annotate(frame, timestamp_ms=None)
                result_q.put((idx, annotated, actions))

        self._post_status(f"运行中…（离线多线程：{workers}）")
        self._post_progress(0, total)

        t_reader = threading.Thread(target=reader, daemon=True)
        t_reader.start()
        worker_ts = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
        for t in worker_ts:
            t.start()

        next_idx = 0
        pending: dict[int, tuple[object, list[str]]] = {}
        written = 0
        t0 = time.monotonic()

        try:
            while not self._stop_evt.is_set():
                if next_idx in pending:
                    annotated, actions = pending.pop(next_idx)
                    written += 1

                    self._write_rec_frame(self._rec, annotated)

                    if total > 0 and (written % 5 == 0 or written == total):
                        self._post_progress(written, total)

                    fps = written / max(1e-6, (time.monotonic() - t0))
                    cv2.putText(
                        annotated,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if actions:
                        actions_text = ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
                    else:
                        actions_text = "-"
                    self._post_frame(annotated, actions_text)
                    next_idx += 1
                    continue

                try:
                    idx, annotated, actions = result_q.get(timeout=0.2)
                except Exception:
                    alive = t_reader.is_alive() or any(t.is_alive() for t in worker_ts)
                    if (not alive) and (not pending):
                        break
                    continue

                pending[int(idx)] = (annotated, actions)

            self._stop_evt.set()
            t_reader.join(timeout=2.0)
            for t in worker_ts:
                t.join(timeout=2.0)

            cap.release()
            cv2.destroyAllWindows()

            self._post_status("已停止")
            self._post_progress(written, total)
            self._post_done()
        finally:
            # 覆盖正常结束 / 停止 / 异常：释放 writer 并复位录制状态。
            self._transcode_async(self._close_primary_recording_session())

    def _worker_loop_parallel_camera(self, state: UiState) -> None:
        """实时摄像头多核并行推理（IMAGE 模式 + 满则丢帧）。

        单线程 VIDEO 模式在多核机上只用到少数核心，heavy 模型实时只能跑 ~15fps；并行多
        worker 能近线性提升吞吐。代价是失去 VIDEO 模式的时序平滑（骨架更抖），且高负载时
        丢弃新帧以约束端到端延迟。

        启动优化：先构造并 ``start()`` 引擎——各 worker 在后台线程各自加载一份模型——再
        打开摄像头，使「模型加载」与「摄像头冷启动」两段耗时重叠，缩短点击→首帧的等待。
        """
        workers = max(1, int(state.workers))
        matcher = (
            self._build_online_matcher()
            if state.online_match_enabled
            else None
        )
        smoother = PreviewLandmarkSmoother()
        draw_cfg = PipelineConfig(
            pose_variant=state.pose_variant,
            running_mode="image",
            enable_hands=state.enable_hands,
        )

        engine = ParallelPoseEngine(
            pipeline_factory=default_pipeline_factory(
                models_dir=models_dir(),
                pose_variant=state.pose_variant,
                enable_hands=state.enable_hands,
            ),
            workers=workers,
            drop_when_full=True,
            defer_draw=True,
        )

        cap = None
        t_reader: threading.Thread | None = None
        failed = False
        final_status: str | None = None
        capture_errors: list[Exception] = []
        try:
            try:
                engine.start()
            except Exception as e:
                failed = True
                final_status = f"初始化失败：{e}"
                self._post_status(final_status)
                return

            # 各 worker 后台建模型的同时由池接管摄像头，两段冷启动重叠（而非串行）。
            try:
                cap = App._open_session_camera(self, state)
            except CameraWarmupStopped:
                self._post_status("已停止")
                return
            if not cap.isOpened():
                failed = True
                final_status = f"无法打开输入源：{state.source}"
                self._post_status(final_status)
                return

            fps_for_ts = 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            session_size = _rotated_size(
                w, h, self._get_runtime_rotate("primary")
            )
            self._rec.begin_session(fps=fps_for_ts, size=session_size)

            capture_t0 = time.monotonic()
            timestamp_ready = threading.Condition()
            submitted_timestamps: dict[int, int] = {}

            def reader() -> None:
                try:
                    while not self._stop_evt.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            if not self._stop_evt.is_set():
                                raise RuntimeError("摄像头读帧失败，请检查连接后重新开始")
                            break
                        frame = _apply_rotation(
                            frame, self._get_runtime_rotate("primary")
                        )
                        self._rec.update_session_size(
                            size=(int(frame.shape[1]), int(frame.shape[0]))
                        )
                        timestamp_ms = int((time.monotonic() - capture_t0) * 1000.0)
                        submitted_index = engine.submit(frame)
                        if submitted_index is not None:
                            with timestamp_ready:
                                submitted_timestamps[submitted_index] = timestamp_ms
                                timestamp_ready.notify_all()
                except Exception as exc:
                    if not self._stop_evt.is_set():
                        traceback.print_exc()
                        capture_errors.append(exc)
                finally:
                    engine.signal_input_done()

            t_reader = threading.Thread(target=reader, daemon=True)
            t_reader.start()

            self._post_status(f"运行中…（实时多线程：{workers}，预览平滑开启）")
            self._post_progress(0, 0)

            t0 = time.monotonic()
            rendered = 0
            while not self._stop_evt.is_set():
                res = engine.get(timeout=0.2)
                if res is None:
                    if engine.is_drained():
                        break
                    continue
                if res.frame is None:
                    continue

                # submit() 仅给成功入队帧分配 index；另存采集时刻才能保留满队列丢帧造成的
                # 真实时间间隙。极端调度竞态下短暂等待 reader 发布对应 timestamp。
                deadline = time.monotonic() + 0.2
                with timestamp_ready:
                    while res.index not in submitted_timestamps:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        timestamp_ready.wait(timeout=remaining)
                    timestamp_ms = submitted_timestamps.pop(
                        res.index,
                        int(res.index * 1000.0 / fps_for_ts),
                    )

                self._feed_online_matcher(
                    matcher,
                    res.pose_landmarks,
                    timestamp_ms,
                    res.index,
                )
                smoothed = smoother.feed(res.pose_landmarks, timestamp_ms=timestamp_ms)
                annotated, actions = draw_pose_frame(
                    res.frame,
                    smoothed,
                    res.hands or [],
                    draw_face=draw_cfg.draw_pose_face,
                    draw_joint_angles=draw_cfg.draw_joint_angles,
                    action_pose_landmarks=res.pose_landmarks,
                )
                rendered += 1
                fps = rendered / max(1e-6, (time.monotonic() - t0))
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f} (x{workers})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                self._write_rec_frame(self._rec, annotated)

                if actions:
                    actions_text = ", ".join(ACTION_LABELS_ZH.get(a, a) for a in actions)
                else:
                    actions_text = "-"
                self._post_frame(annotated, actions_text)

            self._stop_evt.set()
            engine.close()
            t_reader.join(timeout=2.0)
            cv2.destroyAllWindows()

            err = capture_errors[0] if capture_errors else engine.take_error()
            if err is not None:
                failed = True
                final_status = f"运行失败：{err}"
                self._post_status(final_status)
            else:
                self._post_status("已停止")
            self._post_progress(0, 0)
        except Exception as exc:
            failed = not self._stop_evt.is_set()
            if failed:
                traceback.print_exc()
                final_status = f"运行失败：{exc}"
                self._post_status(final_status)
            else:
                self._post_status("已停止")
        finally:
            self._stop_evt.set()
            engine.close()
            if t_reader is not None and t_reader.is_alive():
                t_reader.join(timeout=2.0)
            if cap is not None:
                release_error = App._release_session_camera(self, cap)
                if release_error is not None:
                    failed = True
                    final_status = final_status or release_error
                    self._post_status(final_status)
            if matcher is not None:
                matcher.close()
            self._transcode_async(self._close_primary_recording_session())
            self._post_done(
                session_generation=getattr(state, "session_generation", None),
                final_status=final_status,
                rewarm=not failed,
            )

    def _prepare_preview_rgb(
        self, frame_bgr: np.ndarray, preview_wh: tuple[int, int]
    ) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pw, ph = preview_wh
        if pw > 1 and ph > 1:
            ih, iw = rgb.shape[:2]
            scale = min(pw / iw, ph / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return rgb

    def _mark_dual_startup_metric(
        self,
        generation: int,
        name: str,
        when: float | None = None,
    ) -> None:
        when = time.monotonic() if when is None else float(when)
        with self._dual_metrics_lock:
            metrics = self._dual_startup_metrics.get(generation)
            if metrics is None or not hasattr(metrics, name):
                return
            if getattr(metrics, name) is None:
                setattr(metrics, name, when)

    def _set_dual_startup_outcome(
        self, generation: int, outcome: str, error_type: str | None = None
    ) -> None:
        with self._dual_metrics_lock:
            metrics = self._dual_startup_metrics.get(generation)
            if metrics is None:
                return
            metrics.outcome = outcome
            metrics.error_type = error_type

    def _emit_dual_startup_metrics(
        self, generation: int, *, force: bool = False
    ) -> None:
        with self._dual_metrics_lock:
            metrics = self._dual_startup_metrics.get(generation)
            if metrics is None or metrics.emitted:
                return
            ready_to_emit = (
                metrics.first_annotated_rendered is not None
                if metrics.record_skeleton
                else metrics.first_pair_rendered is not None
            )
            if not force and (
                not ready_to_emit or metrics.outcome == "starting"
            ):
                return
            metrics.emitted = True
            fields = {
                "pair_ready": metrics.pair_ready,
                "first_pair_enqueued": metrics.first_pair_enqueued,
                "first_pair_rendered": metrics.first_pair_rendered,
                "pipeline_ready": metrics.pipeline_ready,
                "first_annotated_enqueued": metrics.first_annotated_enqueued,
                "first_annotated_rendered": metrics.first_annotated_rendered,
            }
            payload: dict[str, object] = {
                "session_generation": metrics.session_generation,
                "primary_index": metrics.primary_index,
                "secondary_index": metrics.secondary_index,
                "record_skeleton": metrics.record_skeleton,
                "outcome": metrics.outcome,
                "error_type": metrics.error_type,
            }
            for name, value in fields.items():
                payload[f"start_to_{name}_ms"] = (
                    round((value - metrics.start_click) * 1000.0, 3)
                    if value is not None
                    else None
                )
        try:
            print(
                "[dual-startup] "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )
        except Exception:
            # Packaged GUI processes may not have a writable stdout.
            pass

    def _finish_dual_startup_metrics(
        self,
        generation: int,
        *,
        outcome: str,
        error_type: str | None,
    ) -> None:
        self._set_dual_startup_outcome(generation, outcome, error_type)
        self._emit_dual_startup_metrics(generation, force=True)

    def _advance_dual_preview_stage(self, generation: int, stage: str) -> bool:
        if stage not in _DUAL_STAGE_RANK:
            raise ValueError(f"unsupported dual preview stage: {stage}")
        with self._dual_render_lock:
            with self._dual_preview_lock:
                if self._active_dual_generation != generation:
                    return False
                if _DUAL_STAGE_RANK[stage] < _DUAL_STAGE_RANK[self._dual_preview_stage]:
                    return False
                if stage != self._dual_preview_stage:
                    self._dual_preview_stage = stage
                    App._drain_queue(self._dual_preview_queue)
                return True

    def _post_dual_frame_pair(
        self,
        frame: np.ndarray,
        actions: str,
        frame2: np.ndarray,
        actions2: str,
        *,
        session_generation: int,
        stage: str,
    ) -> bool:
        if stage not in _DUAL_STAGE_RANK:
            raise ValueError(f"unsupported dual preview stage: {stage}")
        rgb = self._prepare_preview_rgb(frame, self._preview_wh)
        rgb2 = self._prepare_preview_rgb(frame2, self._preview_wh2)
        now = time.monotonic()
        packet = DualPreviewPacket(
            session_generation=session_generation,
            frame_rgb=rgb,
            actions=actions,
            frame_rgb2=rgb2,
            actions2=actions2,
            stage=stage,
            enqueued_at=now,
        )
        with self._dual_preview_lock:
            if self._active_dual_generation != session_generation:
                return False
            current_rank = _DUAL_STAGE_RANK[self._dual_preview_stage]
            stage_rank = _DUAL_STAGE_RANK[stage]
            if stage_rank != current_rank:
                return False
            App._drain_queue(self._dual_preview_queue)
            self._mark_dual_startup_metric(
                session_generation, "first_pair_enqueued", now
            )
            if stage == "annotated":
                self._mark_dual_startup_metric(
                    session_generation, "first_annotated_enqueued", now
                )
            self._dual_preview_queue.put_nowait(packet)
        return True

    def _post_dual_recording_ready(self, generation: int) -> None:
        def _ready() -> None:
            if (
                getattr(self, "_closing", False)
                or self._stop_evt.is_set()
                or self._current_session_generation != generation
                or self._active_dual_generation != generation
            ):
                return
            with self._dual_preview_lock:
                if self._active_dual_generation != generation:
                    return
                self._dual_recording_ready = True
            record_btn = getattr(self, "record_btn", None)
            if record_btn is not None and not bool(
                getattr(self, "_student_practice_active", False)
            ):
                record_btn.configure(
                    state="normal", text=RECORD_BTN_TEXT["idle"]
                )
            # 双摄就绪只启用站位检测，到位事件才开录。
            if bool(getattr(self, "_student_practice_active", False)) and bool(
                getattr(self, "_student_pending_record", False)
            ):
                try:
                    self._student_wait_for_position()
                except Exception:
                    self._student_pending_record = False
                    try:
                        self.student_status_var.set("站位检测启动失败，请再点「开始」")
                    except Exception:
                        pass
            elif bool(getattr(self, "_student_practice_active", False)):
                try:
                    self._sync_student_buttons()
                except Exception:
                    pass

        if getattr(self, "_closing", False):
            return
        try:
            self.root.after(0, _ready)
        except (TclError, RuntimeError):
            pass

    def _post_frame(self, frame_bgr: np.ndarray, actions: str) -> None:
        # 点1：cvtColor(BGR→RGB) + aspect-fit resize 移到 worker 线程，GUI 主线程 _tick
        # 只剩 ImageTk.PhotoImage+configure。cvtColor/resize 均返回新数组，不 mutate
        # 传入的 frame_bgr（调用方 post 前已用它 write_frame，需要保持原 BGR 不变）。
        rgb = self._prepare_preview_rgb(frame_bgr, self._preview_wh)
        # Keep only the latest frame.
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        self._queue.put((rgb, actions))

    def _post_frame2(self, frame_bgr: np.ndarray, actions: str) -> None:
        # 第二路预览队列（双摄像头双面视图，issue #58），与 _post_frame 同构。
        rgb = self._prepare_preview_rgb(frame_bgr, self._preview_wh2)
        while True:
            try:
                self._queue2.get_nowait()
            except Empty:
                break
        self._queue2.put((rgb, actions))

    def _post_status(self, text: str) -> None:
        # Tkinter updates must happen on the main thread.
        if getattr(self, "_closing", False):
            return

        def _set() -> None:
            if not getattr(self, "_closing", False):
                self.status_var.set(text)

        try:
            self.root.after(0, _set)
        except (TclError, RuntimeError):
            pass

    def _post_progress(self, done: int, total: int) -> None:
        def _set() -> None:
            if getattr(self, "_closing", False):
                return
            if total > 0:
                self.progress_bar.configure(mode="determinate", maximum=float(total))
                self.progress_bar["value"] = float(done)
                pct = (done / total) * 100.0
                self.progress_text_var.set(f"进度：{done}/{total}（{pct:.1f}%）")
            else:
                # Unknown total (camera): show nothing.
                self.progress_bar.configure(mode="determinate", maximum=100.0)
                self.progress_bar["value"] = 0.0
                self.progress_text_var.set("")

        if getattr(self, "_closing", False):
            return
        try:
            self.root.after(0, _set)
        except (TclError, RuntimeError):
            pass

    def _post_done(
        self,
        session_generation: int | None = None,
        final_status: str | None = None,
        rewarm: bool = True,
    ) -> None:
        # worker 最后一步：此时 lease 已归还（或已隔离），记录会话终态与原因。
        _log_camera_event(
            {
                "event": "session_done",
                "session": session_generation,
                "status": final_status,
                "rewarm": bool(rewarm),
            }
        )

        def _done() -> None:
            if getattr(self, "_closing", False):
                return
            if session_generation is not None:
                if self._current_session_generation != session_generation:
                    return
                if self._dual_done_generation == session_generation:
                    return
                self._dual_done_generation = session_generation
                self._current_session_generation = 0
                with self._dual_preview_lock:
                    self._dual_recording_ready = False
                    if self._active_dual_generation == session_generation:
                        self._active_dual_generation = 0
                    App._drain_queue(self._dual_preview_queue)
                metrics_lock = getattr(self, "_dual_metrics_lock", None)
                if metrics_lock is not None:
                    with metrics_lock:
                        getattr(self, "_dual_first_render_events", {}).pop(
                            session_generation, None
                        )
                        getattr(self, "_dual_startup_metrics", {}).pop(
                            session_generation, None
                        )
            if bool(getattr(self, "_exam_active", False)):
                panel = getattr(self, "_exam_panel", None)
                if panel is not None:
                    try:
                        panel.on_session_stop()
                    except Exception:
                        traceback.print_exc()
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._set_refresh_enabled()
            # 会话结束：集中复位运行态控件并使 Status_Area 显示「就绪」（需求 4.5、5.1）。
            self._set_running_controls(False)
            if bool(getattr(self, "_student_practice_active", False)):
                self._student_presence_gate = None
                self._student_pending_record = False
                self.student_status_var.set(final_status or "双摄会话已结束，可点「开始」重试")
                self._sync_student_buttons()
            if final_status:
                self.status_var.set(final_status)
            # worker 已在 _post_done 前归还 lease；新 reader 若遇到仍在退出的旧进程，
            # 会在后台等待真实释放后再打开，Tk 线程不等待驱动。
            try:
                if rewarm:
                    self._sync_camera_warmup(allow_running=True)
            except Exception:
                pass

        if getattr(self, "_closing", False):
            return
        try:
            self.root.after(0, _done)
        except (TclError, RuntimeError):
            pass

    def _tick(self) -> None:
        self._drain_recording_compare_updates()
        self._drain_exam_occupancy()
        self._drain_dual_preview_layout()
        self._drain_camera_enum_results()
        # 录制状态刷新必须每 tick 执行，与帧队列是否有新帧无关（需求 5.9/5.10/5.11）。
        self._refresh_recording_status()

        dual_generation = getattr(self, "_active_dual_generation", 0)
        if dual_generation:
            packet = None
            rendered = False
            with self._dual_render_lock:
                with self._dual_preview_lock:
                    try:
                        candidate = self._dual_preview_queue.get_nowait()
                    except Empty:
                        candidate = None
                    if (
                        candidate is not None
                        and candidate.session_generation == dual_generation
                        and _DUAL_STAGE_RANK[candidate.stage]
                        >= _DUAL_STAGE_RANK[self._dual_preview_stage]
                    ):
                        packet = candidate
                if packet is not None:
                    self.actions_var.set(packet.actions)
                    img = Image.fromarray(packet.frame_rgb)
                    img2 = Image.fromarray(packet.frame_rgb2)
                    self._photo = ImageTk.PhotoImage(img)
                    self._photo2 = ImageTk.PhotoImage(img2)
                    self.preview.configure(image=self._photo)
                    self.preview2.configure(image=self._photo2)
                    rendered = True
            if rendered and packet is not None:
                now = time.monotonic()
                self._mark_dual_startup_metric(
                    dual_generation, "first_pair_rendered", now
                )
                first_render = getattr(
                    self, "_dual_first_render_events", {}
                ).get(dual_generation)
                if first_render is not None:
                    first_render.set()
                if packet.stage == "annotated":
                    self._mark_dual_startup_metric(
                        dual_generation, "first_annotated_rendered", now
                    )
                with self._dual_metrics_lock:
                    metrics = self._dual_startup_metrics.get(dual_generation)
                    elapsed_ms = (
                        (now - metrics.start_click) * 1000.0
                        if metrics is not None
                        else None
                    )
                if elapsed_ms is not None:
                    if packet.stage == "raw":
                        self.status_var.set(
                            f"双摄画面已显示（{elapsed_ms:.0f} ms）"
                            + ("，模型加载中…" if metrics.record_skeleton else "")
                        )
                    else:
                        self.status_var.set(
                            f"运行中…（双摄首个标注帧 {elapsed_ms:.0f} ms）"
                        )
                self._emit_dual_startup_metrics(dual_generation)
            self.root.after(16, self._tick)
            return

        try:
            frame_rgb, actions = self._queue.get_nowait()
        except Empty:
            self.root.after(16, self._tick)
            return

        self.actions_var.set(actions)

        # 点1：cvtColor/resize 已在 worker 线程 _post_frame 完成，这里只剩 PhotoImage+configure。
        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._photo)

        # 第二路预览（双摄像头双面视图，issue #58）：单摄模式下 _queue2 恒空，本段恒跳过。
        try:
            frame_rgb2, _actions2 = self._queue2.get_nowait()
        except Empty:
            frame_rgb2 = None
        if frame_rgb2 is not None:
            img2 = Image.fromarray(frame_rgb2)
            self._photo2 = ImageTk.PhotoImage(img2)
            self.preview2.configure(image=self._photo2)

        self.root.after(16, self._tick)


def main() -> None:
    # 离线包首次启动释放内置模型到用户数据目录，源码运行保持原路径。
    models_dir()
    root = Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
