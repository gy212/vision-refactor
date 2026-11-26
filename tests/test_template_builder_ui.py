# -*- coding: utf-8 -*-
"""独立「制作模板」弹窗、控制器、主界面/动作分析联动与折叠标题宽度的测试。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from apps.tk.controllers.template_builder import (
    TemplateBuilderController,
    format_quality_error,
    frame_ranges_for_times,
    min_duration_seconds,
    nudge_time,
    sec_to_frame,
)
from core.evaluation import EXAM_POOL_TEMPLATES
from core.media import VideoMeta
from core.services import (
    TemplateCreateResult,
    TemplateQualityError,
    TemplateQualityReport,
)


class _Queue:
    def __init__(self) -> None:
        self.items: list = []
        self.lock = threading.Lock()

    def post(self, cb) -> None:
        with self.lock:
            self.items.append(cb)

    def drain(self) -> None:
        with self.lock:
            items, self.items = self.items, []
        for cb in items:
            cb()


class _FakeView:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))


class _FakeSource:
    def __init__(self, path: Path, *, fps: float = 30.0, frame_count: int = 90) -> None:
        self.meta = VideoMeta(
            path=Path(path),
            fps=fps,
            fps_reported=fps,
            frame_count=frame_count,
            width=64,
            height=48,
        )
        self.closed = False
        self.reads: list[int] = []

    def read_at(self, index: int):
        self.reads.append(index)
        return np.full((48, 64, 3), min(255, index), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _quality() -> TemplateQualityReport:
    return TemplateQualityReport(
        frame_count=45,
        detected_frames=44,
        detected_ratio=44 / 45,
        core_valid_frames=43,
        core_valid_ratio=43 / 45,
        limb_valid_ratio=0.92,
        warnings=(),
    )


def _write_heavy_template(path: Path) -> Path:
    meta = {
        "pose_variant": "heavy",
        "feature_layout": "pose33_v3",
        "normalizer_version": "v3",
        "running_mode": "video",
        "fps": 30.0,
    }
    np.savez_compressed(
        path,
        features=np.ones((12, 22, 2), dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )
    return path


def test_time_frame_conversion_and_nudge_helpers():
    metas = {
        "front": VideoMeta(
            Path("front.mp4"),
            fps=30.0,
            fps_reported=30.0,
            frame_count=90,
            width=64,
            height=48,
        ),
        "side": VideoMeta(
            Path("side.mp4"),
            fps=25.0,
            fps_reported=25.0,
            frame_count=50,
            width=64,
            height=48,
        ),
    }
    assert min_duration_seconds(metas) == pytest.approx(2.0)
    assert sec_to_frame(1.0, 30.0, 90) == 30
    assert sec_to_frame(99.0, 30.0, 90) == 89
    ranges = frame_ranges_for_times(metas, 1.0, 1.0)
    assert ranges["front"] == (30, 31)
    assert ranges["side"] == (25, 26)
    assert nudge_time(1.0, 5, 25.0, 0.0, 2.0) == pytest.approx(1.2)
    assert nudge_time(0.1, -10, 30.0, 0.0, 2.0) == pytest.approx(0.0)


def test_controller_directory_single_switch_and_cleanup(tmp_path):
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    (rec_dir / "front.mp4").write_bytes(b"f")
    (rec_dir / "side.mp4").write_bytes(b"s")
    opened: list[_FakeSource] = []

    def open_src(path: Path):
        src = _FakeSource(path)
        opened.append(src)
        return src

    view, queue = _FakeView(), _Queue()
    ctrl = TemplateBuilderController(
        view, queue.post, open_source=open_src, out_dir=tmp_path / "tpls"
    )
    metas = ctrl.load_directory(rec_dir)
    assert set(metas) == {"front", "side"}
    frames = ctrl.read_preview_frames(0.5)
    assert frames["front"][0] == 15 and frames["side"][0] == 15
    assert ctrl.plan_paths("上步直拳")["front"].name == "上步直拳_正面.npz"

    single = tmp_path / "clip.mp4"
    single.write_bytes(b"v")
    ctrl.load_single_file(single, "front")
    assert all(s.closed for s in opened[:2])
    assert len(opened) == 3
    ctrl.set_single_view("side")
    assert len(opened) == 3 and list(ctrl.metas) == ["side"]
    assert ctrl.plan_paths("高鞭腿")["side"].name == "高鞭腿_侧面.npz"
    ctrl.set_single_view("single")
    assert ctrl.plan_paths("高鞭腿")["single"].name == "高鞭腿.npz"

    ctrl.close()
    assert opened[2].closed


def test_controller_generate_cancel_and_quality_error(tmp_path):
    single = tmp_path / "clip.mp4"
    single.write_bytes(b"v")
    gate = threading.Event()
    mode = {"behavior": "ok"}

    def fake_create_pair(
        videos,
        *,
        action_name,
        frame_ranges,
        out_dir,
        pose_variant,
        preview,
        overwrite,
        progress_cb,
        stop_evt,
    ):
        assert pose_variant == "heavy" and overwrite is True
        progress_cb("提取骨架特征", 10, 90)
        if mode["behavior"] == "cancel":
            gate.wait(2.0)
            if stop_evt.is_set():
                raise InterruptedError("姿态特征提取已取消")
        if mode["behavior"] == "quality":
            raise TemplateQualityError("骨架质量不达标", report=_quality())
        return [
            TemplateCreateResult(
                "front",
                out_dir / f"{action_name}_正面.npz",
                5,
                49,
                45,
                30.0,
                _quality(),
            )
        ]

    view, queue = _FakeView(), _Queue()
    ctrl = TemplateBuilderController(
        view,
        queue.post,
        open_source=lambda p: _FakeSource(p),
        create_pair=fake_create_pair,
        out_dir=tmp_path,
    )
    ctrl.load_single_file(single, "front")

    assert ctrl.start_generate(action_name="直拳", start_s=0.1, end_s=1.5)
    while ctrl.busy:
        threading.Event().wait(0.01)
    queue.drain()
    assert any(c[0] == "on_generated" for c in view.calls)
    assert view.calls[-1] == ("set_running", False)

    view.calls.clear()
    mode["behavior"] = "cancel"
    ctrl.start_generate(action_name="直拳", start_s=0.1, end_s=1.5)
    ctrl.stop()
    gate.set()
    while ctrl.busy:
        threading.Event().wait(0.01)
    queue.drain()
    assert ("on_cancelled",) in view.calls
    assert not any(c[0] == "on_failed" for c in view.calls)

    view.calls.clear()
    mode["behavior"] = "quality"
    ctrl.start_generate(action_name="直拳", start_s=0.1, end_s=1.5)
    while ctrl.busy:
        threading.Event().wait(0.01)
    queue.drain()
    failed = [c for c in view.calls if c[0] == "on_failed"]
    assert len(failed) == 1 and "核心躯干有效率" in failed[0][1]
    assert "骨架质量不达标" in format_quality_error(
        TemplateQualityError("骨架质量不达标", report=_quality())
    )
    ctrl.close()


def test_collapsible_section_width_fits_chinese_titles():
    tk = pytest.importorskip("tkinter")
    from apps.tk.views import widgets as widgets_mod

    try:
        tk_root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk 不可用：{exc}")
    tk_root.withdraw()
    try:
        frame = tk.ttk.Frame(tk_root)
        sec = widgets_mod.CollapsibleSection(frame, "从视频生成模板")
        assert int(sec._toggle_btn.cget("width")) >= 18
        assert sec._toggle_btn.cget("text") == "▸ 从视频生成模板"
    finally:
        tk_root.destroy()


def test_dialog_preview_nudge_overwrite_and_app_integration(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from apps.tk import UiDispatcher
    from apps.tk.views.template_builder_dialog import TemplateBuilderDialog

    try:
        tk_root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk 不可用：{exc}")
    tk_root.withdraw()
    try:
        rec_dir = tmp_path / "rec"
        rec_dir.mkdir()
        (rec_dir / "front.mp4").write_bytes(b"f")
        (rec_dir / "side.mp4").write_bytes(b"s")
        tpl_dir = tmp_path / "tpls"
        tpl_dir.mkdir()
        existing_front = _write_heavy_template(tpl_dir / "实战式_正面.npz")

        created_results = [
            TemplateCreateResult("front", existing_front, 0, 44, 45, 30.0, _quality()),
            TemplateCreateResult(
                "side",
                _write_heavy_template(tpl_dir / "实战式_侧面.npz"),
                0,
                44,
                45,
                30.0,
                _quality(),
            ),
        ]
        calls: list[dict] = []

        def fake_create_pair(videos, **kwargs):
            calls.append({"videos": videos, **kwargs})
            return created_results

        answers = iter([False, True])
        monkeypatch.setattr(
            "apps.tk.views.template_builder_dialog.messagebox.askyesno",
            lambda *_a, **_k: next(answers),
        )

        dispatcher = UiDispatcher(tk_root)
        received: list[list[TemplateCreateResult]] = []
        dialog = TemplateBuilderDialog(
            tk_root,
            dispatcher.post,
            record_dir=rec_dir,
            on_created=received.append,
            controller_factory=lambda v, p: TemplateBuilderController(
                v,
                p,
                open_source=lambda path: _FakeSource(path),
                create_pair=fake_create_pair,
                out_dir=tpl_dir,
            ),
        )
        dialog.action_name_var.set("实战式")
        dialog.controller.load_directory(rec_dir)
        dialog._on_sources_loaded()
        assert dialog._slots["front"].photo is not None
        assert dialog._slots["side"].photo is not None
        assert "已存在同名模板" in dialog.output_hint_var.get()

        dialog._nudge("start", 5)
        assert dialog.anchor_var.get() == "start" and dialog.start_var.get() > 0.0
        dialog._nudge("end", -5)
        assert dialog.anchor_var.get() == "end" and dialog.end_var.get() < 3.0

        dialog.start_generate()
        assert not calls and not dialog.controller.busy

        dialog.start_generate()
        assert str(dialog.generate_btn.cget("state")) == "disabled"
        assert str(dialog.stop_btn.cget("state")) == "normal"
        while dialog.controller.busy:
            threading.Event().wait(0.01)
        dispatcher.pump()
        assert len(calls) == 1
        assert received == [created_results]
        assert str(dialog.generate_btn.cget("state")) == "normal"
        dialog.close()
    finally:
        tk_root.destroy()


def test_app_links_builder_to_pools_compare_and_close(tmp_path):
    tk = pytest.importorskip("tkinter")
    from apps.tk.views.compare_dialog import CompareDialog
    from tests.tk_app_doubles import build_app

    try:
        tk_root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk 不可用：{exc}")
    tk_root.withdraw()
    try:
        assert EXAM_POOL_TEMPLATES.name == "exam_pool"
        app = build_app(tk_root)
        front_npz = _write_heavy_template(tmp_path / "实战式_正面.npz")
        side_npz = _write_heavy_template(tmp_path / "实战式_侧面.npz")

        app.open_template_builder()
        builder = app._dialogs["template_builder"]
        assert builder.is_open()
        builder.on_generated(
            [
                TemplateCreateResult("front", front_npz, 0, 29, 30, 30.0, _quality()),
                TemplateCreateResult("side", side_npz, 0, 29, 30, 30.0, _quality()),
            ],
            "ok",
        )
        assert app.recording.pools.lists() == ([front_npz], [side_npz])

        app.open_compare()
        compare = app._dialogs["compare"]
        assert isinstance(compare, CompareDialog)
        compare._open_template_builder()
        assert app._dialogs["template_builder"] is builder
        builder.on_generated(
            [TemplateCreateResult("front", front_npz, 0, 29, 30, 30.0, _quality())],
            "ok",
        )
        assert compare.pool_paths() == [str(front_npz)]

        app._close_workflows()
        assert not compare.is_open()
        assert not builder.is_open()
    finally:
        tk_root.destroy()


def test_template_pools_add_paths_dedups_and_ignores_invalid(tmp_path):
    tk = pytest.importorskip("tkinter")
    from tests.tk_app_doubles import build_app

    try:
        tk_root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk 不可用：{exc}")
    tk_root.withdraw()
    try:
        app = build_app(tk_root)
        pools = app.recording.pools
        good = _write_heavy_template(tmp_path / "直拳_正面.npz")
        bad = tmp_path / "bad.npz"
        np.savez_compressed(
            bad,
            features=np.ones((5, 22, 2)),
            meta=np.array({"pose_variant": "full"}, dtype=object),
        )
        assert pools.add_paths("front", [good, good, bad]) == 1
        assert app.recording.pools.lists()[0] == [good]
    finally:
        tk_root.destroy()
