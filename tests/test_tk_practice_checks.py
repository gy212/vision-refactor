# -*- coding: utf-8 -*-
"""学生练习检查项全局开关的 Tk 控制器、弹窗与面板接线测试。"""

from __future__ import annotations

import pytest

from core.evaluation import ACTIONS, DEFAULT_STANCE, catalog_checks
from core.foundation import load_practice_disabled_checks, save_practice_disabled_checks

from apps.tk.controllers import practice_analysis as analysis_mod
from apps.tk.controllers.practice_analysis import DEFAULT_MODEL, PracticeAnalysis
from apps.tk.controllers.practice_checks import (
    CheckGroup,
    build_check_groups,
    check_item_label,
    checks_button_text,
    checks_summary_text,
    validate_action_checks,
    validate_and_save_checks,
    zero_enabled_actions,
)


def test_build_check_groups_covers_all_151_checks_in_11_unified_groups():
    left_groups = build_check_groups("left")
    assert len(left_groups) == 11
    assert all(isinstance(g, CheckGroup) for g in left_groups)
    assert (left_groups[0].key, left_groups[0].title) == ("stance:start", "开始实战式")
    assert (left_groups[-1].key, left_groups[-1].title) == ("stance:end", "结束实战式")
    assert [g.key for g in left_groups[1:-1]] == [
        "front_straight",
        "rear_straight",
        "combo_transition",
        "step_footwork",
        "pad_step",
        "front_low_kick",
        "front_hook",
        "rear_hook",
        "rear_low_kick",
    ]
    assert [g.title for g in left_groups[1:-1]] == [
        "前手直拳",
        "后手直拳",
        "组合拳衔接",
        "上步步法",
        "后脚垫步",
        "前低鞭腿",
        "前手摆拳",
        "后手摆拳",
        "后低鞭腿",
    ]

    left_ids = [item["id"] for g in left_groups for item in g.items]
    assert len(left_ids) == 151
    assert len(set(left_ids)) == 151

    right_groups = build_check_groups("right")
    right_ids = [item["id"] for g in right_groups for item in g.items]
    assert right_ids == left_ids
    first_item = right_groups[0].items[0]
    assert check_item_label(first_item) == (
        f"[{first_item['phaseLabel']}] {first_item['bodyPart']}：{first_item['name']}"
    )


def test_checks_button_text_counts_from_pref_and_ignores_unknown_ids():
    all_ids = [item["id"] for item in catalog_checks(DEFAULT_STANCE)]
    assert checks_button_text() == "检查项设置（已开 151/151 项）"

    sample = [f" {all_ids[0]} ", all_ids[1], all_ids[0], "stale:unknown:id", ""]
    assert checks_button_text(sample) == "检查项设置（已开 149/151 项）"

    save_practice_disabled_checks([all_ids[0], all_ids[2], "stale:unknown:id"])
    assert checks_button_text() == "检查项设置（已开 149/151 项）"
    assert checks_summary_text(149, 151) == (
        "已启用 149 / 151 项（全局统一生效，自动作用于所有包含对应阶段的练习动作）"
    )


def test_validate_and_save_checks_rejects_empty_and_persists_selection():
    all_ids = [item["id"] for item in catalog_checks()]
    save_practice_disabled_checks([all_ids[0]])

    with pytest.raises(ValueError, match="至少需要保留 1 个开启的检查项"):
        validate_and_save_checks(all_ids, ())
    with pytest.raises(ValueError, match="至少需要保留 1 个开启的检查项"):
        validate_and_save_checks(all_ids, ["not-a-real-check-id"])
    assert load_practice_disabled_checks() == (all_ids[0],)

    disabled_expected = (all_ids[0], all_ids[5])
    enabled = [cid for cid in all_ids if cid not in disabled_expected]
    saved = validate_and_save_checks(all_ids, enabled)
    assert saved == disabled_expected
    assert load_practice_disabled_checks() == disabled_expected

    restored = validate_and_save_checks(all_ids, all_ids)
    assert restored == ()
    assert load_practice_disabled_checks() == ()
    assert checks_button_text() == "检查项设置（已开 151/151 项）"


def test_practice_analysis_analyzer_snapshots_disabled_checks(monkeypatch):
    all_ids = [item["id"] for item in catalog_checks()]
    snapshot = (all_ids[0], all_ids[10])
    save_practice_disabled_checks(snapshot)

    captured: list[dict] = []
    monkeypatch.setattr(
        analysis_mod,
        "analyze_feedback",
        lambda *args, **kwargs: captured.append({"args": args, "kwargs": kwargs}) or {},
    )

    controller = PracticeAnalysis(
        lambda cb: cb(),
        on_busy=lambda _b: None,
        on_result=lambda _t: None,
        on_status=lambda _t: None,
        on_finished=lambda _a: None,
    )
    run = controller.analyzer(DEFAULT_MODEL)
    # 分析任务创建后修改全局设置，不得污染已取快照的任务
    save_practice_disabled_checks(())
    run("front_straight", "left", None, None)

    assert len(captured) == 1
    assert captured[0]["kwargs"]["config"].pose_variant == "full"
    assert captured[0]["kwargs"]["disabled_checks"] == snapshot


def test_zero_enabled_actions_and_validate_action_checks():
    groups = build_check_groups("left")
    stance_ids = tuple(
        item["id"]
        for group in groups
        if group.key in {"stance:start", "stance:end"}
        for item in group.items
    )
    assert len(stance_ids) == 16
    assert zero_enabled_actions(stance_ids) == (ACTIONS["stance"],)

    front_straight_ids = tuple(
        item["id"] for group in groups if group.key == "front_straight" for item in group.items
    )
    assert zero_enabled_actions(stance_ids + front_straight_ids) == (
        ACTIONS["stance"],
        ACTIONS["front_straight"],
    )

    with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
        validate_action_checks("stance", "left", stance_ids)
    validate_action_checks("front_straight", "left", stance_ids)

    save_practice_disabled_checks(stance_ids)
    with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
        validate_action_checks("stance")
    validate_action_checks("front_straight", "right")


def test_practice_analysis_analyzer_validates_action_when_provided(monkeypatch):
    groups = build_check_groups("left")
    stance_ids = tuple(
        item["id"]
        for group in groups
        if group.key in {"stance:start", "stance:end"}
        for item in group.items
    )
    save_practice_disabled_checks(stance_ids)

    captured: list[dict] = []
    monkeypatch.setattr(
        analysis_mod,
        "analyze_feedback",
        lambda *args, **kwargs: captured.append(kwargs) or {},
    )

    controller = PracticeAnalysis(
        lambda cb: cb(),
        on_busy=lambda _b: None,
        on_result=lambda _t: None,
        on_status=lambda _t: None,
        on_finished=lambda _a: None,
    )
    with pytest.raises(ValueError, match="当前动作至少需要保留 1 个开启的检查项"):
        controller.analyzer(DEFAULT_MODEL, action="stance", stance="left")

    run_front = controller.analyzer(DEFAULT_MODEL, action="front_straight", stance="left")
    run_front("front_straight", "left", None, None)
    assert captured[-1]["disabled_checks"] == stance_ids

    run_compat = controller.analyzer(DEFAULT_MODEL)
    run_compat("front_straight", "left", None, None)
    assert captured[-1]["disabled_checks"] == stance_ids


def _make_tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - 无显示环境
        pytest.skip(f"Tk 不可用：{exc}")
    root.withdraw()
    return root


def test_checks_dialog_batch_toggle_validation_and_save(monkeypatch):
    root = _make_tk_root()
    import apps.tk.views.practice_checks_dialog as dialog_mod
    from apps.tk.views import PracticeChecksDialog

    warnings: list[tuple[str, str, object]] = []
    infos: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        dialog_mod.messagebox,
        "showwarning",
        lambda title, msg, **kw: warnings.append((title, msg, kw.get("parent"))),
    )
    monkeypatch.setattr(
        dialog_mod.messagebox,
        "showinfo",
        lambda title, msg, **kw: infos.append((title, msg, kw.get("parent"))),
    )
    saved_events: list[bool] = []
    dialog = PracticeChecksDialog(root, stance="right", on_saved=lambda: saved_events.append(True))
    try:
        assert dialog.is_open()
        assert dialog._win.title() == "检查项设置"
        assert len(dialog.group_frames) == 11
        assert len(dialog.vars) == 151
        assert dialog.summary_var.get().startswith("已启用 151 / 151 项")

        dialog.set_group("pad_step", False)
        assert dialog.summary_var.get().startswith("已启用 148 / 151 项")
        dialog.set_group("pad_step", True)
        assert dialog.summary_var.get().startswith("已启用 151 / 151 项")

        for group in dialog.groups:
            dialog.set_group(group.key, False)
        assert dialog.summary_var.get().startswith("已启用 0 / 151 项")
        assert dialog.save() is False
        assert warnings and "至少需要保留 1 个开启的检查项" in warnings[-1][1]
        assert warnings[-1][2] is dialog._win
        assert dialog.is_open()
        assert saved_events == []

        dialog.enable_all()
        assert dialog.summary_var.get().startswith("已启用 151 / 151 项")
        target_ids = [item["id"] for item in dialog.groups[0].items[:2]]
        for cid in target_ids:
            dialog.vars[cid].set(False)
        assert dialog.save() is True
        assert saved_events == [True]
        assert infos == []
        assert not dialog.is_open()
        assert load_practice_disabled_checks() == tuple(target_ids)

        stance_dialog = PracticeChecksDialog(root, stance="left")
        try:
            stance_dialog.set_group("stance:start", False)
            stance_dialog.set_group("stance:end", False)
            assert stance_dialog.save() is True
            assert len(infos) == 1
            assert "以下动作当前未开启任何检查项" in infos[0][1]
            assert ACTIONS["stance"] in infos[0][1]
            assert infos[0][2] is stance_dialog._win
            assert not stance_dialog.is_open()
        finally:
            stance_dialog.close()
    finally:
        dialog.close()
        root.destroy()


def test_practice_controls_checks_button_single_instance_lock_and_close():
    root = _make_tk_root()
    from apps.tk.views import PracticeControls

    controls = PracticeControls(
        root,
        root=root,
        post=lambda cb: cb(),
        set_busy=lambda _b: None,
        show_result=lambda _t: None,
        set_status=lambda _t: None,
        can_analyze=lambda: True,
    )
    try:
        assert str(controls.checks_button.cget("text")) == "检查项设置（已开 151/151 项）"
        controls.open_checks_dialog()
        first = controls.checks_dialog
        assert first is not None and first.is_open()
        controls.open_checks_dialog()
        assert controls.checks_dialog is first

        cid = first.groups[0].items[0]["id"]
        first.vars[cid].set(False)
        assert first.save() is True
        assert str(controls.checks_button.cget("text")) == "检查项设置（已开 150/151 项）"

        controls.open_checks_dialog()
        reopened = controls.checks_dialog
        assert reopened is not None and reopened.is_open()
        controls.set_recording(True)
        assert str(controls.checks_button.cget("state")) == "disabled"
        assert not reopened.is_open()
        assert controls.checks_dialog is None
        controls.open_checks_dialog()
        assert controls.checks_dialog is None

        controls.set_recording(False)
        assert str(controls.checks_button.cget("state")) == "normal"
        controls.open_checks_dialog()
        busy_dialog = controls.checks_dialog
        assert busy_dialog is not None and busy_dialog.is_open()
        controls.analysis.busy = True
        controls.set_recording(False)
        assert str(controls.checks_button.cget("state")) == "disabled"
        assert not busy_dialog.is_open()
        assert controls.checks_dialog is None
        controls.open_checks_dialog()
        assert controls.checks_dialog is None
        controls.analysis.busy = False
        controls.set_recording(False)

        controls.open_checks_dialog()
        final_dialog = controls.checks_dialog
        assert final_dialog is not None and final_dialog.is_open()
        controls.close()
        assert not final_dialog.is_open()
        assert controls.checks_dialog is None
    finally:
        controls.close()
        root.destroy()


def test_checks_dialog_save_oserror_keeps_dialog_open_and_preserves_state(monkeypatch):
    root = _make_tk_root()
    import apps.tk.views.practice_checks_dialog as dialog_mod
    from apps.tk.views import PracticeChecksDialog

    errors: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        dialog_mod.messagebox,
        "showerror",
        lambda title, msg, **kw: errors.append((title, msg, kw.get("parent"))),
    )
    monkeypatch.setattr(
        dialog_mod,
        "validate_and_save_checks",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("磁盘只读")),
    )
    saved_events: list[bool] = []
    dialog = PracticeChecksDialog(root, stance="left", on_saved=lambda: saved_events.append(True))
    try:
        cid = dialog.groups[0].items[0]["id"]
        dialog.vars[cid].set(False)
        assert dialog.save() is False
        assert len(errors) == 1
        assert errors[0][0] == "检查项设置"
        assert "保存设置失败，请检查偏好文件读写权限：磁盘只读" == errors[0][1]
        assert errors[0][2] is dialog._win
        assert dialog.is_open()
        assert dialog.vars[cid].get() is False
        assert saved_events == []
    finally:
        dialog.close()
        root.destroy()


def test_practice_controls_import_reanalyze_and_start_preflight_block_zero_checks(
    monkeypatch,
):
    root = _make_tk_root()
    from types import SimpleNamespace

    import apps.tk.controllers.practice as practice_mod
    import apps.tk.views.practice_controls as controls_mod
    from apps.tk.controllers.practice import PracticeController
    from apps.tk.views import PracticeControls
    from core.workflows import (
        ModeManager,
        PracticeSession,
        PreflightContext,
        ScorebookRegistry,
        WorkflowHost,
    )

    stance_ids = tuple(
        item["id"]
        for group in build_check_groups("left")
        if group.key in {"stance:start", "stance:end"}
        for item in group.items
    )
    save_practice_disabled_checks(stance_ids)

    controls = PracticeControls(
        root,
        root=root,
        post=lambda cb: cb(),
        set_busy=lambda _b: None,
        show_result=lambda _t: None,
        set_status=lambda _t: None,
        can_analyze=lambda: True,
    )
    try:
        controls.student_id.set("001")
        controls.action.set(ACTIONS["stance"])

        dialog_calls: list[str] = []
        errors: list[tuple[str, str, object]] = []
        monkeypatch.setattr(
            controls_mod.filedialog,
            "askopenfilename",
            lambda **_kw: dialog_calls.append("open") or "",
        )
        monkeypatch.setattr(
            controls_mod.messagebox,
            "showerror",
            lambda title, msg, **kw: errors.append((title, msg, kw.get("parent"))),
        )
        controls.import_video()
        assert dialog_calls == []
        assert errors == [("导入旧视频", "当前动作至少需要保留 1 个开启的检查项", root)]

        started: list[bool] = []
        controls.analysis.store = SimpleNamespace(
            get=lambda rid: {"id": rid, "action": "stance", "stance": "left"}
        )
        monkeypatch.setattr(
            controls.analysis,
            "start",
            lambda *_a, **_kw: started.append(True) or True,
        )
        controls.reanalyze("rec-1")
        assert started == []
        assert errors[-1] == ("补充分析", "当前动作至少需要保留 1 个开启的检查项", root)

        warnings: list[tuple[str, str]] = []
        session_starts: list[bool] = []
        monkeypatch.setattr(
            practice_mod.messagebox,
            "showwarning",
            lambda title, msg, **_kw: warnings.append((title, msg)),
        )
        window = SimpleNamespace(
            practice=SimpleNamespace(controls=controls),
            status=SimpleNamespace(status_var=SimpleNamespace(set=lambda _v: None)),
            result_panel=SimpleNamespace(clear=lambda: None),
            root=root,
        )
        controller = PracticeController(
            window,
            session=SimpleNamespace(
                active=lambda: False,
                start=lambda: session_starts.append(True),
            ),
            cameras=SimpleNamespace(dual_selection_problem=lambda: None),
            recording=SimpleNamespace(
                state="idle",
                preflight_context=lambda *,
                practice=False,
                identity_error=None,
                cameras_error=None: PreflightContext(
                    identity_error=identity_error,
                    cameras_error=cameras_error,
                ),
            ),
            modes=ModeManager(),
            workflows=WorkflowHost(ScorebookRegistry()),
            controls=SimpleNamespace(set_roi=lambda _r: None),
            flags=SimpleNamespace(exam_locked=False, recording_ready=False, closing=False),
            exam=lambda: None,
        )
        controller.practice = PracticeSession(controls.identity)
        controller.start()
        assert session_starts == []
        assert warnings == [("学生练习", "当前动作至少需要保留 1 个开启的检查项")]
    finally:
        controls.close()
        root.destroy()


def test_practice_controller_judge_blocks_zero_checks_and_exit_closes_dialogs(
    monkeypatch, tmp_path
):
    root = _make_tk_root()
    from tkinter import StringVar
    from types import SimpleNamespace

    import apps.tk.controllers.practice as practice_mod
    from apps.tk.controllers.practice import PracticeController
    from apps.tk.views.practice_block import PracticeBlock
    from core.live import LiveControls
    from core.reporting import practice_report_renderers
    from core.services import HistoryStore
    from core.workflows import (
        ModeManager,
        PracticeSegment,
        PreflightContext,
        ScorebookRegistry,
        WorkflowHost,
    )

    stance_ids = tuple(
        item["id"]
        for group in build_check_groups("left")
        if group.key in {"stance:start", "stance:end"}
        for item in group.items
    )

    block = PracticeBlock(root, root=root, post=lambda cb: cb())
    block.controls.analysis.store = HistoryStore(
        tmp_path / "history", renderers=practice_report_renderers()
    )
    warnings: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        practice_mod.messagebox,
        "showwarning",
        lambda title, msg, **kw: warnings.append((title, msg, kw.get("parent"))),
    )
    window = SimpleNamespace(
        practice=block,
        status=SimpleNamespace(status_var=StringVar(root, value="")),
        result_panel=SimpleNamespace(clear=lambda: None, set_record=lambda _r: None),
        show_practice=lambda _on: None,
        show_preview_page=lambda: None,
        capture=SimpleNamespace(
            start_btn=SimpleNamespace(configure=lambda **_k: None),
            stop_btn=SimpleNamespace(configure=lambda **_k: None),
        ),
        root=root,
    )
    controller = PracticeController(
        window,
        session=SimpleNamespace(
            active=lambda: False,
            start=lambda: None,
            stop=lambda: None,
            set_running_controls=lambda _b: None,
        ),
        cameras=SimpleNamespace(
            dual_selection_problem=lambda: None,
            sync_warmup=lambda: None,
        ),
        recording=SimpleNamespace(
            state="idle",
            finish=lambda **_k: None,
            preflight_context=lambda *,
            practice=False,
            identity_error=None,
            cameras_error=None: PreflightContext(
                identity_error=identity_error,
                cameras_error=cameras_error,
            ),
        ),
        modes=ModeManager(),
        workflows=WorkflowHost(ScorebookRegistry()),
        controls=LiveControls(),
        flags=SimpleNamespace(exam_locked=False, recording_ready=False, closing=False),
        exam=lambda: None,
        announcer_factory=lambda: SimpleNamespace(announce=lambda _t: None, close=lambda: None),
    )
    try:
        controller.enter()
        block.controls.student_id.set("001")
        block.controls.action.set(ACTIONS["stance"])
        front = tmp_path / "front.mp4"
        front.write_bytes(b"mock")
        assert controller.practice is not None
        controller.practice.identity = block.controls.identity()
        controller.practice.segment = PracticeSegment(
            segment_id="seg-1",
            directory=tmp_path,
            front=front,
            ready=True,
            terminal=True,
        )
        controller.practice.phase = "ready_to_judge"

        save_practice_disabled_checks(stance_ids)
        controller.judge()
        assert warnings == [("动作评判", "当前动作至少需要保留 1 个开启的检查项", root)]
        assert block.status_var.get() == "当前动作至少需要保留 1 个开启的检查项"
        assert controller.practice.phase == "ready_to_judge"
        assert controller.practice.active_token is None

        block.controls.open_checks_dialog()
        block.controls.open_history()
        checks_dlg = block.controls.checks_dialog
        history_win = block.controls.history_window
        assert checks_dlg is not None and checks_dlg.is_open()
        assert history_win is not None and history_win.exists()

        controller.exit()
        assert not checks_dlg.is_open()
        assert block.controls.checks_dialog is None
        assert not history_win.exists()
        assert block.controls.history_window is None
    finally:
        block.controls.close()
        root.destroy()
