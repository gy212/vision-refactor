# -*- coding: utf-8 -*-
"""考试系统无摄像头端到端模拟：状态机 + 占用闸门 + 名单/裁剪。

用法（仓库根）:
  .\\.venv\\Scripts\\python.exe scripts/sim_exam_flow.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.exam_clip import align_frame_counts, prepare_exam_pair
from core.exam_roster import (
    ExamCandidate,
    ExamResultRow,
    ExamScorebook,
    import_roster_xlsx,
    write_import_template,
    write_scorebook_xlsx,
)
from core.exam_session import ExamSession, ExamSessionConfig
from core.presence_gate import PresenceGate, PresenceGateConfig

PASS = 0
FAIL = 0


def log(msg: str) -> None:
    print(msg)


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        log(f"  [OK] {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        log(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


@dataclass
class SimApp:
    begins: list = field(default_factory=list)
    ends: list = field(default_factory=list)
    armed: list = field(default_factory=list)
    segments: dict = field(default_factory=dict)
    _stamp: int = 0

    def begin(self, row) -> bool:
        self._stamp += 1
        sid = f"record_sim_{self._stamp:03d}"
        self.begins.append((sid, row.row_id if row else None))
        if row is not None:
            self.segments[row.row_id] = sid
        return True

    def end(self, *, discard: bool, row) -> str:
        rid = row.row_id if row else ""
        sid = self.segments.get(rid, f"record_sim_{self._stamp:03d}")
        self.ends.append((sid, discard, rid))
        return sid


def drive_commands(
    session,
    cmds,
    app: SimApp,
    scorebook: ExamScorebook | None,
    gate: PresenceGate,
    now: float,
    *,
    quiet: bool = False,
):
    for cmd in cmds:
        k, p = cmd.kind, cmd.payload
        if k == "ANNOUNCE":
            if not quiet:
                log(f"    ANNOUNCE: {p.get('text')}")
        elif k == "ARM_OCCUPANCY":
            app.armed.append(True)
            if session.phase == "recording":
                gate.begin_recording(now)
            else:
                gate.begin_wait_enter()
        elif k == "DISARM_OCCUPANCY":
            app.armed.append(False)
        elif k == "BEGIN_SEGMENT":
            row_id = p.get("row_id")
            row = next((r for r in session.rows if r.row_id == row_id), None)
            ok = app.begin(row)
            if not quiet:
                check("BEGIN_SEGMENT ok", ok)
            elif not ok:
                check("BEGIN_SEGMENT ok", False)
            gate.begin_recording(now)
            if scorebook and row:
                scorebook.bind_segment(
                    row,
                    segment_id=app.segments[row.row_id],
                    segment_dir=f"/tmp/{app.segments[row.row_id]}",
                )
                scorebook.update_row(row, status="recording")
        elif k == "END_SEGMENT":
            row_id = p.get("row_id")
            row = next((r for r in session.rows if r.row_id == row_id), None)
            app.end(discard=False, row=row)
            if scorebook and row:
                scorebook.update_row(
                    row,
                    status="completed",
                    front_score=0.80,
                    side_score=0.85,
                    combined_score=0.83,
                    combined_percent=83,
                )
            more = session.handle("record_stopped", now=now + 0.01, discarded=False)
            drive_commands(
                session, more, app, scorebook, gate, now + 0.01, quiet=quiet
            )
            return
        elif k == "DISCARD_SEGMENT":
            row_id = p.get("row_id")
            row = next((r for r in session.rows if r.row_id == row_id), None)
            app.end(discard=True, row=row)
            if scorebook and row:
                scorebook.update_row(row, status="skipped")
            more = session.handle("record_stopped", now=now + 0.01, discarded=True)
            drive_commands(
                session, more, app, scorebook, gate, now + 0.01, quiet=quiet
            )
            return
        elif k == "UPDATE_ROW":
            if scorebook:
                row = next((r for r in session.rows if r.row_id == p.get("row_id")), None)
                if row:
                    fields = {
                        kk: vv
                        for kk, vv in p.items()
                        if kk != "row_id" and hasattr(row, kk)
                    }
                    scorebook.update_row(row, **fields)
        elif k == "EXPORT":
            if not quiet:
                log("    EXPORT")
        elif k == "UI_LOCK_MANUAL_RECORD":
            if not quiet:
                log("    LOCK manual record")
        elif k == "UI_UNLOCK_MANUAL_RECORD":
            if not quiet:
                log("    UNLOCK manual record")


def wait_phase(
    session,
    app,
    scorebook,
    gate,
    t: float,
    phases: set[str],
    limit: float = 10.0,
    *,
    quiet: bool = False,
) -> float:
    end = t + limit
    while t < end and session.phase not in phases:
        drive_commands(
            session, session.handle("tick", now=t), app, scorebook, gate, t, quiet=quiet
        )
        t += 0.1
    return t


def sim_student(
    session: ExamSession,
    app: SimApp,
    gate: PresenceGate,
    scorebook: ExamScorebook,
    *,
    t0: float,
    enter_at: float,
    leave_at: float,
    dt: float = 0.1,
    force_finish: bool = False,
    skip_while_recording: bool = False,
    quiet: bool = False,
) -> float:
    t = t0
    max_t = t0 + 90.0
    ends_before = len(app.ends)
    while t < max_t and session.phase not in ("completed", "aborted"):
        drive_commands(
            session, session.handle("tick", now=t), app, scorebook, gate, t, quiet=quiet
        )

        if session.phase == "wait_enter":
            present = t >= enter_at
            for ev in gate.update(present, t, mode="wait_enter"):
                if ev.kind == "enter_stable":
                    drive_commands(
                        session,
                        session.handle("enter_stable", now=t),
                        app,
                        scorebook,
                        gate,
                        t,
                        quiet=quiet,
                    )

        if session.phase == "recording":
            if skip_while_recording:
                drive_commands(
                    session,
                    session.handle("skip_current", now=t),
                    app,
                    scorebook,
                    gate,
                    t,
                    quiet=quiet,
                )
                return t
            if force_finish and t >= leave_at:
                drive_commands(
                    session,
                    session.handle("force_finish_current", now=t),
                    app,
                    scorebook,
                    gate,
                    t,
                    quiet=quiet,
                )
                return t
            present = t < leave_at
            for ev in gate.update(present, t, mode="recording"):
                if ev.kind == "empty_stable":
                    drive_commands(
                        session,
                        session.handle("empty_stable", now=t),
                        app,
                        scorebook,
                        gate,
                        t,
                        quiet=quiet,
                    )
                    return t

        if len(app.ends) > ends_before:
            # 片段已结束，推进 gap 直到下一位 calling 带 guard 或 completed
            if session.phase == "completed":
                return t
            if session.phase == "calling":
                if session.inter_gap_deadline is not None and t < session.inter_gap_deadline:
                    pass
                elif session.call_guard_deadline is not None:
                    return t
                elif session.inter_gap_deadline is None:
                    return t

        t += dt
    check("student loop finished in time", t < max_t, f"phase={session.phase} t={t}")
    return t


def scenario_1() -> None:
    log("\n=== SCENARIO 1: 3 人无人干预闭环 ===")
    cfg = ExamSessionConfig(
        post_call_guard_s=1.5,
        inter_student_gap_s=2.0,
        enter_stable_s=0.8,
        empty_hold_s=2.0,
        min_record_s=3.0,
    )
    session = ExamSession(cfg)
    cands = [
        ExamCandidate(1, "2026001", "张三", "一班"),
        ExamCandidate(2, "2026002", "李四", "一班"),
        ExamCandidate(3, "2026003", "王五", "一班"),
    ]
    session.load_roster(cands)
    with tempfile.TemporaryDirectory() as td:
        score_path = Path(td) / "scores.xlsx"
        scorebook = ExamScorebook(path=score_path, throttle_s=0.1)
        scorebook.load_candidates(cands)
        session.rows = scorebook.rows
        app = SimApp()
        gate = PresenceGate(
            PresenceGateConfig(
                enter_stable_s=cfg.enter_stable_s,
                empty_hold_s=cfg.empty_hold_s,
                min_record_s=cfg.min_record_s,
            )
        )
        t = 0.0
        t0_wall = time.perf_counter()
        drive_commands(
            session,
            session.handle("start_exam", now=t, run_id="sim_run"),
            app,
            scorebook,
            gate,
            t,
        )
        check("phase after start", session.phase == "calling", session.phase)

        for i in range(3):
            t = wait_phase(session, app, scorebook, gate, t, {"wait_enter", "completed"})
            if session.phase == "completed":
                break
            check(f"S{i+1} wait_enter", session.phase == "wait_enter", session.phase)
            enter_at = t + 0.5
            leave_at = enter_at + cfg.enter_stable_s + 5.0
            t = sim_student(
                session,
                app,
                gate,
                scorebook,
                t0=t,
                enter_at=enter_at,
                leave_at=leave_at,
                dt=0.1,
            )
            log(f"  student {i+1} done at t={t:.1f}s phase={session.phase}")
            # drain inter-gap into next call
            for _ in range(40):
                drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
                if session.phase in ("wait_enter", "completed"):
                    break
                if session.phase == "calling" and session.call_guard_deadline is not None:
                    # still in call guard for next — continue outer loop wait
                    break
                t += 0.1

        for _ in range(50):
            drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
            if session.phase == "completed":
                break
            t += 0.1

        wall = time.perf_counter() - t0_wall
        check("final phase completed", session.phase == "completed", session.phase)
        check("3 begins", len(app.begins) == 3, str(app.begins))
        check(
            "3 ends non-discard",
            sum(1 for _, d, _ in app.ends if not d) == 3,
            str(app.ends),
        )
        completed = [r for r in scorebook.rows if r.status == "completed"]
        check(
            "3 completed scores",
            len(completed) == 3,
            str([(r.candidate.name, r.combined_percent) for r in completed]),
        )
        scorebook.flush(timeout=2)
        err = scorebook.write_error
        check(
            "scorebook written",
            score_path.is_file() and score_path.stat().st_size > 0,
            f"path={score_path} exists={score_path.is_file()} err={err}",
        )
        scorebook.close()
        log(f"  sim clock t={t:.1f}s wall={wall*1000:.0f}ms")


def scenario_2() -> None:
    log("\n=== SCENARIO 2: skip / force_finish / 录制中 skip / 重考 ===")
    session = ExamSession(
        ExamSessionConfig(
            post_call_guard_s=0.5,
            inter_student_gap_s=0.5,
            enter_stable_s=0.5,
            empty_hold_s=1.0,
            min_record_s=3.0,
        )
    )
    cands = [
        ExamCandidate(1, "A1", "甲"),
        ExamCandidate(2, "A2", "乙"),
        ExamCandidate(3, "A3", "丙"),
        ExamCandidate(4, "A4", "丁"),
    ]
    session.load_roster(cands)
    scorebook = ExamScorebook(path=None)
    scorebook.load_candidates(cands)
    session.rows = scorebook.rows
    app = SimApp()
    gate = PresenceGate(
        PresenceGateConfig(enter_stable_s=0.5, empty_hold_s=1.0, min_record_s=3.0)
    )
    t = 0.0
    drive_commands(
        session, session.handle("start_exam", now=t, run_id="s2"), app, scorebook, gate, t
    )

    t = wait_phase(session, app, scorebook, gate, t, {"wait_enter"})
    check("S2 wait_enter for 甲", session.phase == "wait_enter", session.phase)
    drive_commands(session, session.handle("skip_current", now=t), app, scorebook, gate, t)
    check("甲 skipped", session.rows[0].status == "skipped", session.rows[0].status)

    t += 0.6
    drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
    t = wait_phase(session, app, scorebook, gate, t, {"wait_enter"})
    enter_at = t + 0.2
    leave_at = t + 0.5
    t = sim_student(
        session,
        app,
        gate,
        scorebook,
        t0=t,
        enter_at=enter_at,
        leave_at=leave_at,
        force_finish=True,
        dt=0.1,
    )
    check(
        "乙 force completed",
        session.rows[1].status == "completed",
        session.rows[1].status,
    )

    t += 0.6
    drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
    t = wait_phase(session, app, scorebook, gate, t, {"wait_enter"})
    enter_at = t + 0.2
    t = sim_student(
        session,
        app,
        gate,
        scorebook,
        t0=t,
        enter_at=enter_at,
        leave_at=enter_at + 10,
        skip_while_recording=True,
        dt=0.1,
    )
    check("丙 skipped discard", session.rows[2].status == "skipped", session.rows[2].status)
    check("has discard end", any(d for _, d, _ in app.ends), str(app.ends))

    t += 0.6
    drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
    t = wait_phase(session, app, scorebook, gate, t, {"wait_enter"})
    enter_at = t + 0.3
    leave_at = enter_at + 0.5 + 4.0
    t = sim_student(
        session, app, gate, scorebook, t0=t, enter_at=enter_at, leave_at=leave_at, dt=0.1
    )
    for _ in range(40):
        drive_commands(session, session.handle("tick", now=t), app, scorebook, gate, t)
        if session.phase == "completed":
            break
        t += 0.1
    check("S2 completed", session.phase == "completed", session.phase)

    row = scorebook.append_retest(cands[0])
    check("retest attempt 2", row.attempt_index == 2, str(row.attempt_index))


def scenario_3() -> None:
    log("\n=== SCENARIO 3: exam_clip 帧截齐 + 裁剪 ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def write_vid(path: Path, n: int, move_range: tuple[int, int]) -> None:
            w, h = 80, 60
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            for i in range(n):
                fr = np.zeros((h, w, 3), np.uint8)
                if move_range[0] <= i < move_range[1]:
                    fr[:] = (i % 50 + 20, 40, 40)
                    cv2.rectangle(
                        fr, (i % 40, 10), (i % 40 + 20, 50), (200, 200, 200), -1
                    )
                else:
                    fr[:] = (10, 10, 10)
                wr.write(fr)
            wr.release()

        front, side = td / "f.mp4", td / "s.mp4"
        write_vid(front, 90, (25, 65))
        write_vid(side, 85, (25, 65))
        n, _ = align_frame_counts(90, 85)
        check("align min", n == 85, str(n))
        t0 = time.perf_counter()
        clip = prepare_exam_pair(
            front, side, front_frames=90, side_frames=85, out_dir=td / "out"
        )
        elapsed = time.perf_counter() - t0
        check(
            "clip equal frames",
            clip.front_frames == clip.side_frames,
            f"{clip.front_frames}/{clip.side_frames}",
        )
        check("clip non-empty", clip.front_frames > 0)
        check("clip files exist", clip.front_path.is_file() and clip.side_path.is_file())
        check("clip under 5s", elapsed < 5.0, f"{elapsed:.2f}s")
        log(f"  clip frames={clip.front_frames} elapsed={elapsed:.2f}s warnings={clip.warnings[:3]}")


def scenario_4() -> None:
    log("\n=== SCENARIO 4: Excel 导入/导出 ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tpl = td / "tpl.xlsx"
        write_import_template(tpl)
        cands = import_roster_xlsx(tpl)
        check("import 2 rows", len(cands) == 2)
        out = td / "out.xlsx"
        rows = [
            ExamResultRow(
                candidate=cands[0],
                status="completed",
                front_score=0.8,
                side_score=0.9,
                combined_percent=85,
                combined_score=0.85,
            ),
            ExamResultRow(candidate=cands[1], status="skipped"),
        ]
        write_scorebook_xlsx(out, rows)
        check("export exists", out.is_file())


def scenario_5(n_students: int = 50) -> None:
    log(f"\n=== SCENARIO 5: {n_students} 人快速模拟 ===")
    session = ExamSession(
        ExamSessionConfig(
            post_call_guard_s=0.3,
            inter_student_gap_s=0.2,
            enter_stable_s=0.3,
            empty_hold_s=0.5,
            min_record_s=1.0,
        )
    )
    cands = [ExamCandidate(i + 1, f"S{i+1:03d}", f"学员{i+1}") for i in range(n_students)]
    session.load_roster(cands)
    scorebook = ExamScorebook(path=None)
    scorebook.load_candidates(cands)
    session.rows = scorebook.rows
    app = SimApp()
    gate = PresenceGate(
        PresenceGateConfig(enter_stable_s=0.3, empty_hold_s=0.5, min_record_s=1.0)
    )
    t = 0.0
    t0 = time.perf_counter()
    quiet = n_students > 10
    drive_commands(
        session,
        session.handle("start_exam", now=t, run_id="s5"),
        app,
        scorebook,
        gate,
        t,
        quiet=quiet,
    )
    for i in range(n_students):
        t = wait_phase(
            session,
            app,
            scorebook,
            gate,
            t,
            {"wait_enter", "completed"},
            limit=30,
            quiet=quiet,
        )
        if session.phase == "completed":
            break
        if session.phase != "wait_enter":
            check(f"S5-{i+1} wait_enter", False, session.phase)
        enter_at = t + 0.15
        leave_at = enter_at + 0.3 + 1.5
        t = sim_student(
            session,
            app,
            gate,
            scorebook,
            t0=t,
            enter_at=enter_at,
            leave_at=leave_at,
            dt=0.05,
            quiet=quiet,
        )
        for _ in range(30):
            drive_commands(
                session,
                session.handle("tick", now=t),
                app,
                scorebook,
                gate,
                t,
                quiet=quiet,
            )
            if session.phase == "wait_enter":
                break
            if session.phase == "calling" and session.call_guard_deadline is not None:
                break
            if session.phase == "completed":
                break
            t += 0.05
        if (i + 1) % 10 == 0 or (i + 1) == n_students:
            log(
                f"  progress {i+1}/{n_students} sim_t={t:.1f}s begins={len(app.begins)} phase={session.phase}"
            )
    for _ in range(80):
        drive_commands(
            session, session.handle("tick", now=t), app, scorebook, gate, t, quiet=quiet
        )
        if session.phase == "completed":
            break
        t += 0.05
    wall = time.perf_counter() - t0
    check(f"{n_students}-person completed", session.phase == "completed", session.phase)
    check(f"{n_students} begins", len(app.begins) == n_students, str(len(app.begins)))
    check(
        f"{n_students} completed rows",
        sum(1 for r in scorebook.rows if r.status == "completed") == n_students,
    )
    # 50 人状态机应仍远小于实时；放宽墙钟上限
    wall_limit = 5.0 if n_students <= 10 else 30.0
    check(
        f"{n_students}-person wall < {wall_limit}s",
        wall < wall_limit,
        f"{wall:.3f}s",
    )
    per = wall / max(1, n_students)
    log(
        f"  sim t={t:.1f}s wall={wall*1000:.0f}ms begins={len(app.begins)} "
        f"per_student_wall={per*1000:.2f}ms"
    )


def scenario_6() -> None:
    log("\n=== SCENARIO 6: 闸门防误触发 ===")
    gate = PresenceGate(
        PresenceGateConfig(enter_stable_s=0.8, empty_hold_s=2.0, min_record_s=3.0)
    )
    gate.begin_wait_enter()
    ev = []
    for tt in np.arange(0, 0.5, 0.1):
        ev.extend(gate.update(True, float(tt), mode="wait_enter"))
    ev.extend(gate.update(False, 0.5, mode="wait_enter"))
    check("no enter on flash", len(ev) == 0, str(ev))

    gate.begin_wait_enter()
    ev = []
    for tt in np.arange(0, 1.0, 0.1):
        ev.extend(gate.update(True, float(tt), mode="wait_enter"))
    check("enter after 0.8s", any(e.kind == "enter_stable" for e in ev), str(ev))

    gate.begin_recording(0.0)
    ev = []
    for tt in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ev.extend(gate.update(False, float(tt), mode="recording"))
    check("no empty before min_record", not any(e.kind == "empty_stable" for e in ev), str(ev))
    ev2 = gate.update(False, 3.0, mode="recording")
    check("empty at min_record+hold", any(e.kind == "empty_stable" for e in ev2), str(ev2))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="考试流程无摄像头模拟")
    parser.add_argument(
        "--students",
        type=int,
        default=50,
        help="场景5压力测试人数（默认 50）",
    )
    parser.add_argument(
        "--only-stress",
        action="store_true",
        help="只跑场景5压力测试",
    )
    args = parser.parse_args(argv)

    if not args.only_stress:
        scenario_1()
        scenario_2()
        scenario_3()
        scenario_4()
    scenario_5(n_students=max(1, int(args.students)))
    if not args.only_stress:
        scenario_6()
    log("\n" + "=" * 50)
    log(f"RESULT: {PASS} passed, {FAIL} failed")
    if FAIL:
        return 1
    log("SIMULATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
