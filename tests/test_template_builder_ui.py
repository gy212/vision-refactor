# -*- coding: utf-8 -*-
"""独立「制作模板」弹窗、控制器、主界面/动作分析联动与折叠标题宽度的测试。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from apps.tk.controllers.template_builder import (
    COMPARE_BACKFILL_OPTIONS,
    TemplateBuilderController,
    filter_backfill_results,
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
        start_s,
        end_s,
        out_dir,
        pose_variant,
        preview,
        overwrite,
        progress_cb,
        stop_evt,
    ):
        assert pose_variant == "heavy" and overwrite is True
        assert start_s == pytest.approx(0.1) and end_s == pytest.approx(1.5)
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
            pool_mode="dual",
            controller_factory=lambda v, p: TemplateBuilderController(
                v,
                p,
                open_source=lambda path: _FakeSource(path),
                create_pair=fake_create_pair,
                out_dir=tpl_dir,
            ),
        )
        dialog.action_name_var.set("实战式")
        dialog.load_directory(rec_dir)
        assert dialog._slots["front"].photo is not None
        assert dialog._slots["side"].photo is not None
        assert "已存在同名模板" in dialog.output_hint_var.get()

        dialog.nudge_buttons[("start", 5)].invoke()
        assert dialog.anchor_var.get() == "start" and dialog.start_var.get() > 0.0
        dialog.nudge_buttons[("end", -5)].invoke()
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
        compare.build_tpl_btn.invoke()
        assert app._dialogs["template_builder"] is builder
        builder.on_generated(
            [TemplateCreateResult("front", front_npz, 0, 29, 30, 30.0, _quality())],
            "ok",
        )
        assert compare.pool_paths() == [str(front_npz)]

        app.shutdown_steps().prepare_workflows()
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


def test_controller_seconds_tolerates_frame_count_mismatch(tmp_path, monkeypatch):
    """容器元数据 frame_count=30 与真实解码 29 帧不一致时，按 start_s/end_s 建库不越界。"""
    clip = tmp_path / "front.mp4"
    clip.write_bytes(b"v")
    feats = np.ones((29, 22, 2), dtype=np.float32)
    monkeypatch.setattr(
        "core.services.templates.extract_features",
        lambda *_a, **_k: (feats, 30.0, None),
    )
    view, queue = _FakeView(), _Queue()
    ctrl = TemplateBuilderController(
        view,
        queue.post,
        open_source=lambda p: _FakeSource(p, fps=30.0, frame_count=30),
        out_dir=tmp_path / "tpls",
    )
    ctrl.load_single_file(clip, "front")
    assert ctrl.start_generate(action_name="直拳", start_s=0.0, end_s=1.0)
    while ctrl.busy:
        threading.Event().wait(0.01)
    queue.drain()
    ctrl.close()

    assert not any(c[0] == "on_failed" for c in view.calls), view.calls
    generated = [c for c in view.calls if c[0] == "on_generated"]
    assert len(generated) == 1
    results = generated[0][1]
    assert len(results) == 1
    assert results[0].frame_count == 29 and results[0].end_frame == 28
    assert results[0].path.exists()


def test_filter_backfill_modes(tmp_path):
    front = TemplateCreateResult(
        "front", tmp_path / "a_正面.npz", 0, 10, 11, 30.0, _quality()
    )
    side = TemplateCreateResult(
        "side", tmp_path / "a_侧面.npz", 0, 10, 11, 30.0, _quality()
    )
    single = TemplateCreateResult(
        "single", tmp_path / "a.npz", 0, 10, 11, 30.0, _quality()
    )
    pair = [front, side]

    assert filter_backfill_results(pair, pool_mode="none") == []
    assert filter_backfill_results(pair, pool_mode="dual") == pair
    assert filter_backfill_results([single], pool_mode="dual") == []
    assert filter_backfill_results([single], pool_mode="single") == [single]
    assert filter_backfill_results(
        pair, pool_mode="single", backfill_choice="仅加入正面"
    ) == [front]
    assert filter_backfill_results(
        pair, pool_mode="single", backfill_choice="仅加入侧面"
    ) == [side]
    assert (
        filter_backfill_results(
            pair, pool_mode="single", backfill_choice="仅保存不入池"
        )
        == []
    )
    assert COMPARE_BACKFILL_OPTIONS == ("仅加入正面", "仅加入侧面", "仅保存不入池")


def test_dialog_pool_modes_busy_isolation_and_video_filetypes(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from apps.tk import UiDispatcher
    from apps.tk.views.template_builder_dialog import TemplateBuilderDialog
    from apps.tk.views.widgets import video_filetypes
    from tests.tk_app_doubles import build_app

    patterns = video_filetypes()[0][1]
    assert "*.wmv" in patterns.split() and ";" not in patterns

    try:
        tk_root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover
        pytest.skip(f"Tk 不可用：{exc}")
    tk_root.withdraw()
    try:
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"v")
        rec_dir = tmp_path / "rec"
        rec_dir.mkdir()
        (rec_dir / "front.mp4").write_bytes(b"f")
        (rec_dir / "side.mp4").write_bytes(b"s")
        front_npz = _write_heavy_template(tmp_path / "直拳_正面.npz")
        side_npz = _write_heavy_template(tmp_path / "直拳_侧面.npz")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        pair_results = [
            TemplateCreateResult("front", front_npz, 0, 29, 30, 30.0, _quality()),
            TemplateCreateResult("side", side_npz, 0, 29, 30, 30.0, _quality()),
        ]

        monkeypatch.setattr(
            "apps.tk.views.template_builder_dialog.messagebox.askyesno",
            lambda *_a, **_k: True,
        )
        for kind in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                f"apps.tk.views.template_builder_dialog.messagebox.{kind}",
                lambda title, msg, **_k: pytest.fail(f"意外弹出模态框 {title}: {msg}"),
            )

        dispatcher = UiDispatcher(tk_root)
        gate = threading.Event()
        dual_calls: list[list[TemplateCreateResult]] = []
        single_calls: list[list[TemplateCreateResult]] = []

        def slow_pair(*_a, **_k):
            gate.wait(2.0)
            return pair_results

        dialog = TemplateBuilderDialog(
            tk_root,
            dispatcher.post,
            record_dir=rec_dir,
            on_created=dual_calls.append,
            pool_mode="dual",
            controller_factory=lambda v, p: TemplateBuilderController(
                v,
                p,
                open_source=lambda path: _FakeSource(path),
                create_pair=slow_pair,
                out_dir=out_dir,
            ),
        )
        dialog.load_single_file(clip, "single")
        assert dialog.auto_add_var.get() is False
        assert str(dialog.auto_add_check.cget("state")) == "disabled"
        assert "切换视角" in dialog.auto_add_hint_var.get()

        dialog.load_single_file(clip, "front")
        assert dialog.auto_add_var.get() is True
        assert str(dialog.auto_add_check.cget("state")) == "normal"

        # compare(single) 模式下加载双摄目录，支持“仅加入正面/侧面/仅保存不入池”
        dialog.set_on_created(single_calls.append, pool_mode="single")
        dialog.load_directory(rec_dir)
        dialog.compare_backfill_var.set("仅加入侧面")
        dialog.action_name_var.set("直拳")
        dialog.start_generate()
        assert dialog.is_busy()
        # 运行中修改回填回调，仍应使用启动时的快照 single_calls 与“仅加入侧面”
        other_calls: list[list[TemplateCreateResult]] = []
        dialog.set_on_created(other_calls.append, pool_mode="dual")
        dialog.compare_backfill_var.set("仅保存不入池")
        gate.set()
        while dialog.is_busy():
            threading.Event().wait(0.01)
        dispatcher.pump()
        assert (
            single_calls == [[pair_results[1]]] and not other_calls and not dual_calls
        )
        dialog.close()

        # app 级 busy 隔离 + 关闭未停完禁止并发重开
        app = build_app(tk_root)
        app.open_template_builder()
        builder = app._dialogs["template_builder"]
        infos: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "apps.tk.app.messagebox.showinfo", lambda t, m, **_k: infos.append((t, m))
        )
        monkeypatch.setattr(builder, "is_busy", lambda: True)
        app.open_compare()
        app._dialogs["compare"].build_tpl_btn.invoke()
        assert builder._pool_mode == "dual"
        assert builder._on_created == app.recording.pools.add_created
        builder.close()
        app.open_template_builder()
        assert infos and "正在停止收尾" in infos[0][1]
        assert app._dialogs["template_builder"] is builder
        app.shutdown_steps().prepare_workflows()
    finally:
        tk_root.destroy()
