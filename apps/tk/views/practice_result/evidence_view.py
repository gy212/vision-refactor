# -*- coding: utf-8 -*-
"""结果页右栏的证据区：选中检查项的标题与元数据、正侧两路代表帧（叠加相关关节），以及标准与依据。

证据帧经 ``core.reporting.extract_evidence_frame`` 截取；录像只按片段目录内的文件名定位
（``SegmentLayout.child``，拒绝越出目录）；PhotoImage 有界缓存 30 张。
"""
from __future__ import annotations

import logging
from pathlib import Path
import tkinter as tk
from tkinter import StringVar

import cv2
from PIL import Image, ImageTk

from core.media import SegmentLayout
from core.reporting import RULE_JOINTS, NormalizedCheck, extract_evidence_frame

from apps.tk.views.practice_result.style import STYLE_A, card, label

logger = logging.getLogger(__name__)

_PHOTO_CACHE_LIMIT = 30
_VIEW_LABELS = {"front": "正面", "side": "侧面"}


class _FrameBox:
    """一路视角的证据帧：标题与画面。"""

    def __init__(self, parent: tk.Misc, view: str, column: int) -> None:
        self.view = view
        self.view_label = _VIEW_LABELS[view]
        self.box = card(parent, bg=STYLE_A["bg_page"], padx=4, pady=4)
        self.box.grid(row=0, column=column, sticky="nsew", padx=(0, 3) if column == 0 else (3, 0))
        self.title_var = StringVar(value=f"{self.view_label}视角")
        label(self.box, size=8, bold=True, bg=STYLE_A["bg_page"], textvariable=self.title_var).pack(anchor="w")
        self.image_label = label(self.box, bg=STYLE_A["bg_page"], fg=STYLE_A["text_secondary"],
                                 text=f"暂无{self.view_label}画面", width=28, height=9)
        self.image_label.pack(fill="both", expand=True, pady=2)
        self.photo: ImageTk.PhotoImage | None = None  # 正在显示的画面，保持引用以免被回收

    def reset(self) -> None:
        self.photo = None
        self.image_label.configure(image="", text=f"暂无{self.view_label}画面")
        self.title_var.set(f"{self.view_label}视角")

    def show_text(self, text: str) -> None:
        self.photo = None
        self.image_label.configure(image="", text=text)

    def show_photo(self, photo: ImageTk.PhotoImage) -> None:
        self.photo = photo
        self.image_label.configure(image=photo, text="")


class EvidenceView:
    """右栏卡片；复核操作区由调用方放进 ``frame`` 的下方。"""

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = card(parent)
        self.frame.rowconfigure(2, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.name_var = StringVar(value="请从左侧选择检查项目查看依据")
        label(self.frame, size=11, bold=True, textvariable=self.name_var, anchor="w", justify="left").pack(fill="x")
        self.meta_var = StringVar(value="")
        label(self.frame, size=8, fg=STYLE_A["text_secondary"], textvariable=self.meta_var,
              anchor="w", justify="left").pack(fill="x", pady=(2, 6))
        frames = tk.Frame(self.frame, bg=STYLE_A["bg_card"])
        frames.pack(fill="x", pady=(0, 6))
        frames.columnconfigure(0, weight=1)
        frames.columnconfigure(1, weight=1)
        self.boxes = {"front": _FrameBox(frames, "front", 0), "side": _FrameBox(frames, "side", 1)}
        basis = tk.Frame(self.frame, bg=STYLE_A["bg_card"])
        basis.pack(fill="x", pady=(0, 6))
        self.standard_var = StringVar(value="标准：-")
        label(basis, textvariable=self.standard_var, anchor="w", justify="left", wraplength=480).pack(fill="x")
        self.reason_var = StringVar(value="依据：-")
        label(basis, fg=STYLE_A["text_secondary"], textvariable=self.reason_var,
              anchor="w", justify="left", wraplength=480).pack(fill="x", pady=(2, 0))
        self._photo_cache: dict[str, ImageTk.PhotoImage] = {}

    @property
    def photo_count(self) -> int:
        return len(self._photo_cache)

    def show_check(self, check: NormalizedCheck) -> None:
        self.name_var.set(check.name)
        rev_tag = f"【{check.review_display}】" if check.review_display else ""
        self.meta_var.set(
            f"阶段：{check.segment_label} / {check.phase_label} · 部位：{check.body_part} · 状态：{check.status_display} {rev_tag}"
        )
        self.standard_var.set(f"动作标准：{check.standard}（出处：{check.source}）")
        self.reason_var.set(f"检查依据：{check.reason}")

    def show_empty(self, message: str) -> None:
        """没有可显示的检查项时的占位文字。"""
        self.reset_frames()
        self.name_var.set(message)
        self.meta_var.set("")
        self.standard_var.set("标准：-")
        self.reason_var.set("依据：-")

    def reset_frames(self) -> None:
        for box in self.boxes.values():
            box.reset()

    def clear(self) -> None:
        self._photo_cache.clear()
        self.show_empty("请从左侧选择检查项目查看依据")

    def load_frames(self, record: dict, check: NormalizedCheck, catalog: dict[str, dict], video_dir: Path | None) -> None:
        """加载当前检查项的正侧代表帧；本项没有某一视角的证据时，借用同阶段同视角的证据。"""
        self.reset_frames()
        refs = [catalog[ref] for ref in check.evidence_refs if ref in catalog]
        for view, box in self.boxes.items():
            ev = next((e for e in refs if e.get("view") == view), None)
            if not ev:
                ev = next((e for e in catalog.values() if e.get("view") == view and e.get("phase") == check.phase), None)
            if not ev:
                box.title_var.set(f"{box.view_label}视角（未关联证据帧）")
                box.show_text(f"该项无{box.view_label}有效证据帧")
                continue
            self._show_evidence(box, record, check, ev, video_dir)

    def _show_evidence(self, box: _FrameBox, record: dict, check: NormalizedCheck, ev: dict, video_dir: Path | None) -> None:
        frame_idx = ev.get("frame", 0)
        box.title_var.set(f"{box.view_label} {ev.get('timeSeconds', 0.0):.3f}s · 帧{frame_idx}")
        cache_key = f"{record.get('id')}_{check.id}_{box.view}_{frame_idx}"
        if cache_key in self._photo_cache:
            box.show_photo(self._photo_cache[cache_key])
            return
        video_path = SegmentLayout.child(video_dir, record.get("videos", {}).get(box.view)) if video_dir else None
        if video_path is None:  # 没有目录、没有文件名，或文件名越出目录
            box.show_text("原视频不可用")
            return
        frame_bgr = extract_evidence_frame(video_path, frame_idx, landmarks=ev.get("landmarks"),
                                           valid_mask=ev.get("validMask"), highlight_joints=RULE_JOINTS.get(check.code))
        if frame_bgr is None:
            box.show_text("原视频不可用或无法解码")
            return
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((320, 240))
            photo = ImageTk.PhotoImage(pil_img, master=self.frame)
            if len(self._photo_cache) >= _PHOTO_CACHE_LIMIT:  # 有界缓存：满了先丢最早的一张
                del self._photo_cache[next(iter(self._photo_cache))]
            self._photo_cache[cache_key] = photo
            box.show_photo(photo)
        except Exception as exc:
            logger.debug("Failed to render photo for %s: %s", box.view, exc)
            box.show_text("画面渲染错误")
