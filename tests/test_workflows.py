"""P7 工作流层的纯状态机测试：PresenceRecordingCycle、PracticeSession、CommandExecutor、WorkflowHost、ModeManager。"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from core.workflows import (
    EXAM,
    NORMAL,
    PRACTICE,
    Command,
    CommandExecutor,
    ModeManager,
    PracticeSegment,
    PracticeSession,
    PreflightContext,
    WorkflowHost,
    WorkflowPorts,
)
from core.workflows.presence import PresenceGateConfig, PresenceRecordingCycle


# ---------------------------------------------------------------- PresenceRecordingCycle
def _kinds(events):
    return [ev.kind for ev in events]


def test_cycle_enter_stable_then_leave_ends_after_hold():
    cycle = PresenceRecordingCycle(PresenceGateConfig(enter_stable_s=0.8, empty_hold_s=2.0, min_record_s=0.0))
    assert cycle.update(True, 0.0) == []  # 未武装不出事件
    cycle.arm_enter()
    assert _kinds(cycle.update(True, 1.0)) == []
    assert _kinds(cycle.update(True, 1.8)) == ["enter_stable"]
    assert cycle.phase == "wait_enter"  # 等拥有者确认开录
    cycle.recording_started(1.8)
    cycle.update(False, 2.0)
    cycle.update(True, 3.0)  # 中途回来重置离场计时
    cycle.update(False, 3.5)
    assert _kinds(cycle.update(False, 5.49)) == []
    assert _kinds(cycle.update(False, 5.5)) == ["empty_stable"]
    assert cycle.phase == "idle"


def test_cycle_min_record_blocks_early_leave_but_not_disarm():
    cycle = PresenceRecordingCycle(PresenceGateConfig(enter_stable_s=0.8, empty_hold_s=2.0, min_record_s=3.0))
    cycle.arm_enter()
    cycle.update(True, 0.0)
    cycle.update(True, 0.8)
    cycle.recording_started(1.0)
    cycle.update(False, 1.1)
    assert _kinds(cycle.update(False, 3.5)) == []  # 离场已满 2 s，录制不足 3 s
    assert _kinds(cycle.update(False, 4.0)) == ["empty_stable"]
    # 强制结束不受最短时长约束
    cycle.arm_enter()
    cycle.recording_started(10.0)
    cycle.disarm()
    assert cycle.phase == "idle" and cycle.update(False, 20.0) == []


def test_cycle_ignores_samples_queued_before_arming():
    cycle = PresenceRecordingCycle(PresenceGateConfig(enter_stable_s=0.8))
    cycle.arm_enter(ignore_before=100.0)
    assert cycle.update(True, 90.0) == [] and cycle.update(True, 99.9) == []
    cycle.update(True, 100.0)
    assert _kinds(cycle.update(True, 101.0)) == ["enter_stable"]


# ---------------------------------------------------------------- PracticeSession
def _kinds_of(cmds):
    return [c.kind for c in cmds]


def _practice(identity=lambda: {"studentId": "001"}):
    return PracticeSession(identity)


def _record(session, now=0.0):
    session.handle("start", now=now, session_ready=True)
    session.handle("presence", present=True, now=now)
    cmds = session.handle("presence", present=True, now=now + 0.8)
    assert _kinds_of(cmds) == ["BEGIN_SEGMENT"]
    return session.handle("record_started", now=now + 0.8, segment=SimpleNamespace(segment_id="seg", directory=Path("seg")))


def test_practice_start_waits_for_session_then_arms():
    s = _practice()
    assert s.handle("start", now=0.0, session_ready=False, launching=True)[0].payload["text"] == "正在启动双摄预览…"
    assert s.phase == "waiting_session" and s.waiting
    cmds = s.handle("session_ready", now=1.0)
    assert _kinds_of(cmds) == ["ARM_OCCUPANCY", "STATUS"]
    assert s.phase == "waiting_enter" and s.presence.phase == "wait_enter"


def test_practice_session_failed_returns_to_rest():
    s = _practice()
    s.handle("start", now=0.0, session_ready=False)
    assert _kinds_of(s.handle("session_failed", message="双摄未启动")) == ["STATUS"]
    assert s.phase == "idle"


def test_practice_record_end_postprocess_judge_result(tmp_path):
    s = _practice()
    cmds = _record(s)
    assert _kinds_of(cmds) == ["ANNOUNCE", "STATUS"] and s.recording
    assert s.identity == {"studentId": "001"}
    assert not s.can_judge
    cmds = s.handle("end")
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "END_SEGMENT", "ANNOUNCE", "STATUS"]
    assert s.phase == "waiting_postprocess"
    # 其它片段的更新与处理中更新都不放行评判
    s.handle("segment_update", update=SimpleNamespace(segment_id="other", status="completed"))
    s.handle("segment_update", update=SimpleNamespace(segment_id="seg", status="transcoding", message="转码"))
    assert not s.can_judge
    front = tmp_path / "front.mp4"
    front.write_bytes(b"x")
    s.handle("segment_update", update=SimpleNamespace(segment_id="seg", status="completed", front_video=front, side_video=None))
    assert s.phase == "ready_to_judge" and s.can_judge
    cmds = s.handle("judge")
    assert _kinds_of(cmds) == ["START_ANALYSIS"]
    assert cmds[0].payload["token"] == 1 and cmds[0].payload["student_id"] == "001"
    assert s.phase == "analyzing"
    record = {"id": "rec1"}
    assert _kinds_of(s.handle("analysis_saved", token=1, student_id="001", record=record)) == ["SHOW_RESULT"]
    assert s.current_record is record and s.record_id == "rec1" and s.phase == "showing_result"


def test_practice_failed_postprocess_without_video_blocks_judge():
    s = _practice()
    _record(s)
    s.handle("end")
    s.handle("segment_update", update=SimpleNamespace(segment_id="seg", status="failed", error_code="E", message=""))
    assert s.phase == "idle" and not s.can_judge
    assert s.handle("judge")[0].payload["alert"] == "请先结束练习，并等待录像转码落盘。"


def test_practice_invalid_identity_or_begin_failure_disarm():
    def bad():
        raise ValueError("请先填写学号")

    s = _practice(bad)
    s.handle("start", now=0.0, session_ready=True)
    s.handle("presence", present=True, now=0.0)
    cmds = s.handle("presence", present=True, now=0.8)
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "STATUS"] and cmds[1].payload["text"] == "请先填写学号"
    assert s.phase == "idle" and s.presence.phase == "idle"
    s = _practice()
    s.handle("start", now=0.0, session_ready=True)
    cmds = s.handle("begin_failed")
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "STATUS"] and cmds[1].payload["error"] is True
    assert s.phase == "idle"


def test_practice_cancellation_and_stale_results_dropped():
    s = _practice()
    s.segment = PracticeSegment("seg", None, ready=True, terminal=True)
    s.identity = {"studentId": "001"}
    s.phase = "ready_to_judge"
    s.handle("judge")
    s.handle("analysis_cancelled")
    assert s.handle("analysis_saved", token=1, student_id="001", record={"id": "r"}) == []
    s.handle("judge")
    s.handle("identity_changed")
    assert s.active_token is None and s.current_record is None
    assert s.handle("analysis_saved", token=2, student_id="001", record={"id": "r"}) == []
    s.handle("judge")
    # 学号已变：代次相符也拦截
    assert s.handle("analysis_saved", token=3, student_id="001", record={"id": "r"}, current_student_id="002") == []
    # 记录落盘与上屏无关：重评时覆盖同一条记录
    s.handle("record_saved", record_id="r3")
    assert s.record_id == "r3" and s.current_record is None


def test_practice_busy_blocks_and_practice_again_disarms():
    s = _practice()
    s.handle("analysis_busy", busy=True)
    assert s.handle("start", now=0.0, session_ready=True) == []
    assert s.handle("practice_again") == []
    s.handle("analysis_busy", busy=False)
    s.segment = PracticeSegment("seg", None, ready=True, terminal=True)
    s.phase = "showing_result"
    s.active_token = 4
    s.presence.arm_enter()
    assert _kinds_of(s.handle("practice_again")) == ["DISARM_OCCUPANCY", "STATUS"]
    assert s.phase == "idle" and s.active_token is None and s.presence.phase == "idle"
    assert not s.segment.ready
    assert s.handle("presence", present=True, now=5.0) == []  # 不会自动开录


def test_practice_session_stop_end_and_exit():
    s = _practice()
    _record(s)
    s.handle("session_stopped", now=1.0)
    assert s.phase == "waiting_postprocess" and s.presence.phase == "idle"
    s = _practice()
    s.handle("start", now=0.0, session_ready=True)
    cmds = s.handle("session_ended", now=1.0, status="")
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "STATUS"] and not s.session_ready and s.phase == "idle"
    s = _practice()
    _record(s)
    assert _kinds_of(s.handle("exit")) == ["DISARM_OCCUPANCY", "END_SEGMENT"]
    assert s.phase == "idle" and s.segment is None and s.identity is None


# ---------------------------------------------------------------- CommandExecutor
class _Recorder:
    def __init__(self, segment):
        self.segment = segment
        self.calls = []

    def start_segment(self, payload, on_confirm):
        """开录确认同步回送（真实端口在两路写出首帧后经 Tk 线程回送）。"""
        self.calls.append(("start", payload))
        if self.segment:
            on_confirm(self.segment, None)
        return self.segment

    def pause_segment(self):
        self.calls.append(("pause",))

    def resume_segment(self):
        self.calls.append(("resume",))

    def finish(self, *, discard):
        self.calls.append(("finish", discard))


class _Ui:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


class _Flow:
    """记录收到的事件；BEGIN 后的回送事件再产出一条状态命令。"""

    def __init__(self, cmds):
        self.cmds = cmds
        self.events = []

    def handle(self, event, **kw):
        self.events.append((event, kw))
        if event == "go":
            return self.cmds
        if event == "record_started":
            return [Command("STATUS", {"text": "已开录"})]
        return []

    def row(self, row_id):
        return {"id": row_id}


def _ports(segment=None, **kw):
    occupancy = SimpleNamespace(calls=[])
    occupancy.arm = lambda armed: occupancy.calls.append(armed)
    occupancy.set_roi = lambda roi: None
    announcer = SimpleNamespace(texts=[])
    announcer.announce = announcer.texts.append
    return WorkflowPorts(_Recorder(segment), occupancy, announcer, _Ui(), **kw)


def test_executor_feeds_results_back_as_events():
    ports = _ports(segment=SimpleNamespace(segment_id="s1"))
    flow = _Flow([Command("ARM_OCCUPANCY"), Command("BEGIN_SEGMENT", {"x": 1}), Command("ANNOUNCE", {"text": "开始"})])
    CommandExecutor(ports, clock=lambda: 7.0).dispatch(flow, "go")
    assert ports.occupancy.calls == [True]
    assert ports.recorder.calls == [("start", {"x": 1})]
    assert flow.events[1] == ("record_started", {"now": 7.0, "segment": ports.recorder.segment})
    assert ports.announcer.texts == ["开始"]
    assert ("status", ("已开录",), {}) in ports.ui.calls
    assert ports.ui.calls[-1][0] == "refresh"


def test_executor_begin_failure_and_discard():
    ports = _ports(segment=None)
    flow = _Flow([Command("BEGIN_SEGMENT", {}), Command("DISCARD_SEGMENT")])
    CommandExecutor(ports, clock=lambda: 1.0).run(flow, flow.handle("go"))
    assert [e for e, _ in flow.events[1:]] == ["begin_failed", "record_stopped"]
    assert flow.events[2][1] == {"now": 1.0, "discarded": True}
    assert ports.recorder.calls[-1] == ("finish", True)


def test_executor_failed_command_reports_and_continues():
    ports = _ports()

    def boom(_armed):
        raise RuntimeError("坏了")

    ports.occupancy.arm = boom
    flow = _Flow([Command("ARM_OCCUPANCY"), Command("ANNOUNCE", {"text": "开始"})])
    CommandExecutor(ports).run(flow, flow.handle("go"))
    assert ("status", ("命令失败 ARM_OCCUPANCY: 坏了",), {}) in ports.ui.calls
    assert ports.announcer.texts == ["开始"]


def test_executor_rows_export_and_analysis():
    book = SimpleNamespace(path=Path("out.xlsx"), write_error=None, calls=[])
    book.update_row = lambda row, **f: book.calls.append(("update", row, f))
    book.bind_segment = lambda row, **f: book.calls.append(("bind", row, f))
    book.flush = lambda: book.calls.append(("flush",))
    analysis = SimpleNamespace(requests=[])
    analysis.start_practice_analysis = lambda request: analysis.requests.append(request) or True
    ports = _ports(scorebook=book, analysis=analysis)
    flow = _Flow([
        Command("UPDATE_ROW", {"row_id": "r1", "status": "recording"}),
        Command("BIND_SEGMENT_PENDING", {"row_id": "r1", "segment_id": "s1", "segment_dir": "d"}),
        Command("EXPORT"),
        Command("START_ANALYSIS", {"token": 1}),
        Command("SHOW_RESULT", {"record": {"id": "r"}}),
        Command("UI_LOCK_MANUAL_RECORD"),
    ])
    CommandExecutor(ports).run(flow, flow.handle("go"))
    assert book.calls == [
        ("update", {"id": "r1"}, {"status": "recording"}),
        ("bind", {"id": "r1"}, {"segment_id": "s1", "segment_dir": "d"}),
        ("flush",),
    ]
    assert analysis.requests == [{"token": 1}]
    names = [c[0] for c in ports.ui.calls]
    assert names.count("refresh_rows") == 2
    assert ("status", (f"成绩已写入：{Path('out.xlsx')}",), {}) in ports.ui.calls
    assert ("show_result", ({"id": "r"},), {}) in ports.ui.calls
    assert ("lock_manual_record", (True,), {}) in ports.ui.calls


def test_analysis_start_failure_restores_retryable_phase():
    """配置错误等同步启动失败：回到 ready_to_judge，修好后再点能真正发起分析。"""
    outcomes = [False, RuntimeError("坏配置"), True]
    analysis = SimpleNamespace(requests=[])

    def start(request):
        analysis.requests.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    analysis.start_practice_analysis = start
    executor = CommandExecutor(_ports(analysis=analysis))
    s = _practice()
    s.segment = PracticeSegment("seg", None, ready=True, terminal=True)
    s.identity = {"studentId": "001"}
    s.phase = "ready_to_judge"
    for expected_calls in (1, 2):
        executor.dispatch(s, "judge")
        assert len(analysis.requests) == expected_calls
        assert s.phase == "ready_to_judge" and s.active_token is None and not s.busy
    executor.dispatch(s, "judge")
    assert len(analysis.requests) == 3 and s.phase == "analyzing"
    assert s.active_token == analysis.requests[-1]["token"]
    # 迟到的旧代次失败回报不影响进行中的任务
    s.handle("analysis_start_failed", token=1)
    assert s.phase == "analyzing" and s.active_token == 3


# ---------------------------------------------------------------- WorkflowHost
class _Registry:
    def __init__(self, result=None):
        self.result = result
        self.updates = []

    def apply(self, update):
        self.updates.append(update)
        return self.result


def _host_with(flow, registry=None, scorebook=None):
    host = WorkflowHost(registry or _Registry())
    ports = _ports(scorebook=scorebook)
    host.activate(flow, CommandExecutor(ports))
    return host, ports


def test_host_routes_to_active_workflow_and_drops_stale_tokens():
    flow = _Flow([])
    host, _ = _host_with(flow)
    host.session_started(5)
    host.presence(4, True, 1.0)  # 旧会话的样本
    host.presence(5, True, 2.0)
    host.session_ready(4, 3.0)
    host.tick(4.0)
    assert flow.events == [("presence", {"present": True, "now": 2.0}), ("tick", {"now": 4.0})]
    host.session_ended(5, 5.0, "结束")
    host.presence(5, True, 6.0)  # 会话结束后同 token 也不再接受
    assert flow.events[-1] == ("session_ended", {"now": 5.0, "status": "结束"})


def test_host_deactivate_only_affects_current_workflow():
    old, new = _Flow([]), _Flow([])
    host, ports = _host_with(old)
    host.activate(new, CommandExecutor(ports))
    host.deactivate(old)
    assert host.active is new
    assert host.dispatch("tick", now=1.0) and new.events
    host.deactivate(new)
    assert host.active is None and not host.dispatch("tick", now=2.0)


def test_host_backfills_scorebook_before_routing():
    book = object()
    registry = _Registry(result=(book, "row"))
    flow = _Flow([])
    host, ports = _host_with(flow, registry, scorebook=book)
    update = SimpleNamespace(segment_id="s1", exam_run_id="run1")
    host.segment_update(update, route=False)  # 关窗收尾：只回填
    assert registry.updates == [update] and flow.events == []
    assert [c[0] for c in ports.ui.calls] == ["refresh_rows"]
    host.segment_update(update)
    assert flow.events == [("segment_update", {"update": update})]
    # 面板已换了台账：回填不刷新当前面板
    registry.result = (object(), "row")
    ports.ui.calls.clear()
    host.segment_update(update, route=False)
    assert ports.ui.calls == []


# ---------------------------------------------------------------- ModeManager
@dataclass(frozen=True)
class _Snapshot:
    record_skeleton: bool = True
    enable_hands: bool = True
    online_match_enabled: bool = True
    auto_compare: bool = True
    capture_size: tuple = (0, 0)
    capture_policy: str = "prefer"


def test_mode_manager_overrides_session_settings_without_touching_user_snapshot():
    modes = ModeManager()
    user = _Snapshot()
    assert modes.session_settings(user) == _Snapshot(capture_size=(1920, 1080), capture_policy="require")
    assert modes.segment_auto_compare is None
    modes.enter(PRACTICE)
    # V-5：只锁会话参数，用户勾选的快照本身不变
    assert modes.session_settings(user) == _Snapshot(False, False, False, False, (1920, 1080), "require")
    assert user == _Snapshot()
    assert modes.segment_auto_compare is False
    modes.leave("exam")  # 名字不符不退出
    assert modes.is_active("practice")
    modes.enter(EXAM)
    assert modes.session_settings(user).record_skeleton is False
    # D-17：考试与普通、学生练习同为 1080p require
    assert modes.session_settings(user).capture_size == (1920, 1080)
    assert modes.session_settings(user).capture_policy == "require"
    assert modes.session_settings(user).auto_compare is True
    assert modes.segment_auto_compare is True
    modes.leave()
    assert modes.current is NORMAL


def test_exam_and_practice_preflight_order():
    ctx = PreflightContext()
    assert ModeManager().preflight(EXAM, ctx) == ["请先选择双摄像头并点击「开始」进入双摄会话。"]
    ctx = PreflightContext(dual_active=True)
    assert ModeManager().preflight(EXAM, ctx) == ["双摄尚未就绪，请稍候再试。"]
    ctx = PreflightContext(identity_error="学号无效", cameras_error="未选双摄", model_installed=lambda key: False)
    problems = ModeManager().preflight(PRACTICE, ctx)
    assert problems[:2] == ["学号无效", "未选双摄"] and len(problems) == 3
    assert ModeManager().preflight(PRACTICE, PreflightContext()) == []


# ---------------------------------------------------------------- D-18 第 8 层：开录确认与录制失败（D-27）
def _starting(session, now=0.0):
    session.handle("start", now=now, session_ready=True)
    session.handle("presence", present=True, now=now)
    assert _kinds_of(session.handle("presence", present=True, now=now + 0.8)) == ["BEGIN_SEGMENT"]
    assert session.starting and session.phase == "waiting_enter"


def test_practice_begin_failure_shows_the_reason():
    s = _practice()
    _starting(s)
    cmds = s.handle("begin_failed", now=1.0, message="侧面：开录时还没有这一路的画面")
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "STATUS"]
    assert "侧面：开录时还没有这一路的画面" in cmds[1].payload["alert"]
    assert not s.starting and s.phase == "idle"


def test_practice_end_or_exit_before_confirmation_discards_the_segment():
    s = _practice()
    _starting(s)
    cmds = s.handle("end")
    assert _kinds_of(cmds) == ["DISARM_OCCUPANCY", "DISCARD_SEGMENT", "STATUS"]
    assert not s.starting and s.phase == "idle"
    assert s.handle("record_started", now=2.0, segment=SimpleNamespace(segment_id="seg", directory=None)) == []
    s = _practice()
    _starting(s)
    assert _kinds_of(s.handle("exit")) == ["DISARM_OCCUPANCY", "DISCARD_SEGMENT"]


def test_practice_recording_failure_is_not_judged_even_with_readable_files(tmp_path):
    """录制失败（队列溢出、断流、写盘失败）不走降级通路；录制中失败时离开录制阶段。"""
    front, side = tmp_path / "front.mp4", tmp_path / "side.mp4"
    front.write_bytes(b"x")
    side.write_bytes(b"x")
    s = _practice()
    _record(s)
    assert s.recording
    update = SimpleNamespace(segment_id="seg", status="failed", error_code="side_recording_failed",
                             message="侧面录像失败：录像队列已满", front_video=front, side_video=side)
    cmds = s.handle("segment_update", update=update)
    assert "DISARM_OCCUPANCY" in _kinds_of(cmds)
    assert "录制失败，请重录" in cmds[-1].payload["text"] and cmds[-1].payload["error"]
    assert not s.recording and s.phase == "idle" and not s.can_judge
    # 只是转码失败、源文件可读：仍按降级通路允许评判
    s = _practice()
    _record(s)
    s.handle("end")
    s.handle("segment_update", update=SimpleNamespace(segment_id="seg", status="failed", error_code="transcode_failed",
                                                     message="", front_video=front, side_video=side))
    assert s.can_judge


def test_executor_start_confirmation_is_asynchronous():
    confirm = []

    class Pending(_Recorder):
        def start_segment(self, payload, on_confirm):
            self.calls.append(("start", payload))
            confirm.append(on_confirm)
            return self.segment

    segment = SimpleNamespace(segment_id="s1")
    ports = _ports()
    ports.recorder = Pending(segment)
    flow = _Flow([Command("BEGIN_SEGMENT", {}), Command("PAUSE_SEGMENT"), Command("RESUME_SEGMENT")])
    executor = CommandExecutor(ports, clock=lambda: 3.0)
    executor.dispatch(flow, "go")
    assert [e for e, _ in flow.events] == ["go"]           # 还没确认：不回送
    assert ports.recorder.calls[1:] == [("pause",), ("resume",)]
    confirm[0](segment, None)
    assert flow.events[-1] == ("record_started", {"now": 3.0, "segment": segment})
    confirm[0](segment, "正面：写盘失败")
    assert flow.events[-1] == ("begin_failed", {"now": 3.0, "message": "正面：写盘失败"})


def test_executor_immediate_begin_failure_carries_the_reason():
    from core.recording import BeginError

    class Refused(_Recorder):
        def start_segment(self, payload, on_confirm):
            return BeginError("stopping")

    ports = _ports()
    ports.recorder = Refused(None)
    flow = _Flow([Command("BEGIN_SEGMENT", {})])
    CommandExecutor(ports, clock=lambda: 1.0).dispatch(flow, "go")
    assert flow.events[-1] == ("begin_failed", {"now": 1.0, "message": BeginError("stopping").message})
