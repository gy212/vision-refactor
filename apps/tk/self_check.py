"""离线自检：在打包后的 EXE 里逐项检查模型、录制转码、练习历史与报告导出、摄像头子进程和 Tk 桌面，不读写真实用户资料。

由 ``apps/desktop_launcher.py --offline-self-check <报告.json>`` 调用；每项检查是一个函数，
通过后把检查名追加到报告的 ``checks``。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path


class _CameraSmokeCapture:
    """打包检查用驱动；不接触物理摄像头。"""
    def __init__(self, index, **kwargs):
        self.index, self.reads = index, 0

    def read(self):
        import numpy as np
        self.reads += 1
        if self.index == 1 and self.reads > 1:
            raise RuntimeError("simulated cv::Mat stride failure")
        return True, np.zeros((24, 32, 3), dtype=np.uint8)[:, ::-1]

    def get(self, prop):
        return 30

    def release(self):
        pass


class _CameraSmokeRecovered(_CameraSmokeCapture):
    def __init__(self, index, **kwargs):
        super().__init__(0, **kwargs)


class _CameraSmokePreview(_CameraSmokeRecovered):
    def __init__(self, index, *, width, height):
        super().__init__(index)
        self.width, self.height = width, height

    def read(self):
        import numpy as np
        time.sleep(0.02)
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self):
        time.sleep(0.1)  # Realistic asynchronous driver shutdown across mode switches.


# 结果页状态图标：4 种状态 × 2 种命名 × 3 种尺寸 = 24 个，结果页按 parent.parent 定位。
FEEDBACK_ICON_STATES = ("candidate", "not_observed", "pending_rule", "unable")
FEEDBACK_ICON_SIZES = (24, 32, 48)

# 图文报告模板：随包分发到 ``core/reporting/html/templates``，缺一个渲染即失败。
REPORT_TEMPLATES = (
    "candidate_section", "count_pill", "diagnostics_section", "evidence_grid", "evidence_image",
    "evidence_placeholder", "incomplete_row", "incomplete_section", "issue_card", "issue_details",
    "no_candidates", "not_observed_row", "not_observed_section", "report", "revoked_row", "revoked_section",
)


def _package_root() -> Path:
    """源码模式为仓库根，冻结模式为 PyInstaller 解包目录；与模块的 ``__file__`` 定位一致。"""
    return Path(__file__).resolve().parents[2]


def load_build_info(root: Path) -> dict | None:
    """读取构建时打进包内的 ``build_info.json``；源码运行没有该文件时返回 None。"""
    path = root / "build_info.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_feedback_icons(root: Path) -> int:
    """断言结果页的全部状态图标都能解码且尺寸正确，返回图标个数。"""
    from PIL import Image
    icon_dir = root / "assets" / "feedback"
    count = 0
    for state in FEEDBACK_ICON_STATES:
        for prefix in ("", "status_"):
            for size in FEEDBACK_ICON_SIZES:
                path = icon_dir / f"{prefix}{state}_{size}.png"
                assert path.is_file(), f"包内缺少结果页图标：{path.name}"
                with Image.open(path) as image:
                    image.load()
                    assert image.size == (size, size), f"{path.name} 尺寸为 {image.size}"
                count += 1
    return count


def block_network() -> None:
    """离线检查期间任何联网尝试都直接失败。"""
    def blocked(*args, **kwargs):
        raise RuntimeError("离线检查禁止网络访问")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    urllib.request.urlopen = blocked
    urllib.request.urlretrieve = blocked


def check_build_info(report: dict) -> None:
    """冻结包必须带 ``build_info.json``，来自干净工作区且规则版本一致；同时检查结果页图标。"""
    from core.evaluation import RULE_VERSION
    build = load_build_info(_package_root())
    report["buildInfo"] = build
    if report["frozen"]:
        assert build is not None, "包内缺少 build_info.json"
        assert build["clean"] is True, "EXE 不是从干净的工作区构建的"
        assert build["ruleVersion"] == RULE_VERSION
    report["checks"].append("build_info")
    report["feedbackIcons"] = check_feedback_icons(_package_root())
    report["checks"].append("feedback_status_icons")


def check_models_and_inference(report: dict, temporary: Path) -> None:
    """内置的 4 个模型已种到临时数据目录，三档姿态加手部都能在 CPU 上推理。"""
    import numpy as np
    from core import foundation, pose
    from core.pose import EngineSpec, create_engine

    model_dir = foundation.models_dir()
    assert model_dir.is_relative_to(temporary)
    assert all(pose.is_installed(spec) for spec in pose.MEDIAPIPE_MODELS)
    report["checks"].append("four_bundled_models_seeded")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for variant in ("lite", "full", "heavy"):
        pipeline = create_engine(EngineSpec(
            pose_variant=variant, enable_hands=True, delegate="cpu", running_mode="image"),
            models_dir=model_dir)
        try:
            pipeline.infer(frame)
        finally:
            pipeline.close()
        report["checks"].append(variant + "_and_hand_cpu_inference")


def check_templates_and_preferences(report: dict) -> None:
    """在线模板库可加载；摄像头选择与录像目录偏好能写入并读回。"""
    from core import evaluation, foundation
    templates = evaluation.load_template_library(foundation.templates_dir() / "online")
    assert len(templates) == 2
    foundation.save_camera_selection(0, None)
    assert foundation.load_camera_selection() == (0, None)
    foundation.save_record_dir(foundation.outputs_dir())
    assert foundation.load_record_dir() == foundation.outputs_dir()
    report["checks"].append("templates_and_persistent_preferences")


def check_video_record(report: dict) -> Path:
    """写一段空白录像，经内置 FFmpeg 转成 MP4 并能解码；返回 MP4 路径。"""
    import numpy as np
    from core import foundation, media
    from core.media import first_frame_ok

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    writer, video, _codec = media.open_video_writer(
        foundation.outputs_dir() / "空白 测试.avi", fps=12, size=(320, 240))
    try:
        for _ in range(24):
            writer.write(frame)
    finally:
        writer.release()
    video = media.transcode_to_h264(video)
    assert video.suffix == ".mp4", "内置 FFmpeg 转码失败"
    assert first_frame_ok(video), "录制文件不能解码"
    report["checks"].append("video_record_transcode_decode")
    return video


def check_practice_history(report: dict, video: Path) -> tuple[object, dict]:
    """双视角录像走 MediaPipe 练习分析写入历史，结果字段齐全，文字报告能导出；返回历史库与记录。"""
    from core import foundation
    from core.evaluation import ACTIONS, RULE_VERSION, STANCES
    from core.reporting import practice_report_renderers
    from core.services import HistoryStore

    history = HistoryStore(renderers=practice_report_renderers())
    side_video = video.with_name("侧面 测试.mp4")
    shutil.copyfile(video, side_video)
    record = history.add({"studentId": "offline-check", "studentName": "打包检查",
                          "action": next(iter(ACTIONS)), "stance": next(iter(STANCES))}, video, side_video)
    result = record["result"]
    assert result["backend"] == "mediapipe"
    assert result["fusionMethod"] == "action_rule_quality_weighted_v1"
    assert result["inputViews"] == ["front", "side"]
    assert result["ruleVersion"] == RULE_VERSION
    assert result["poseVariant"] == "full"
    assert all(c["status"] in {"unable", "pending_rule"} for c in result["checks"])
    assert all(c["analyzedFrames"] == 24 and c["stride"] == 1 for c in result["capture"].values())
    report["feedbackVersion"] = result["ruleVersion"]
    report["feedbackCapture"] = result["capture"]
    assert history.get(record["id"])["id"] == record["id"]
    history.export(record["id"], foundation.outputs_dir() / "问题说明.txt")
    assert (foundation.outputs_dir() / "问题说明.txt").stat().st_size > 0
    assert not any("qwen" in name or name == "torch" for name in sys.modules)
    report["checks"].append("mediapipe_feedback_history_export")
    return history, record


def check_html_report(report: dict, history, record: dict) -> None:
    """图文报告的模板全部随包分发，导出的 HTML 已填充、不引用外部资源、不泄露本机数据目录。"""
    from core import foundation

    template_dir = _package_root() / "core" / "reporting" / "html" / "templates"
    missing = [name for name in REPORT_TEMPLATES if not (template_dir / f"{name}.html").is_file()]
    assert not missing, f"包内缺少图文报告模板：{missing}"
    target = foundation.outputs_dir() / "图文报告.html"
    history.export(record["id"], target)
    text = target.read_text(encoding="utf-8")
    assert text.lstrip().lower().startswith("<!doctype html"), "图文报告不是完整的 HTML 文档"
    assert "打包检查" in text and "${" not in text, "图文报告模板未填充"
    assert "http://" not in text and "https://" not in text, "图文报告引用了外部资源"
    assert str(foundation.repo_root()) not in text, "图文报告泄露了本机路径"
    report["htmlReportBytes"] = len(text.encode("utf-8"))
    report["checks"].append("html_report_export_with_bundled_templates")


def check_camera_spawn(report: dict) -> None:
    """采集子进程的异常传回主进程并清理；随后能重新打开，交出连续内存的帧。"""
    from core.media import ProcessCamera
    failed_camera = ProcessCamera(1, _factory=_CameraSmokeCapture)
    try:
        assert failed_camera.read()[0]
        try:
            failed_camera.read()
        except RuntimeError as exc:
            assert "simulated cv::Mat" in str(exc)
        else:
            raise AssertionError("摄像头异常没有传回主进程")
        assert not failed_camera.isOpened()
    finally:
        failed_camera.release()
    recovered_camera = ProcessCamera(1, _factory=_CameraSmokeRecovered)
    try:
        ok, recovered_frame = recovered_camera.read()
        assert ok and recovered_frame.flags.c_contiguous
        assert recovered_frame.flags.owndata
    finally:
        recovered_camera.release()
    report["checks"].append("camera_spawn_error_cleanup_and_reopen")


def check_camera_mode_switch(app, root):
    """Exercise real Tk callbacks, pool, spawn and claim with simulated devices."""
    from unittest.mock import patch
    from apps.tk.views.capture_panel import NO_SECOND_CAMERA
    from core.media import PRIMARY, SECONDARY, CaptureSpec, ProcessCamera
    from core.media import CameraDevice as CameraEntry
    from core.workflows import NORMAL

    FULL_HD = NORMAL.requirements.fmt.size  # D-17：三种模式同一采集格式

    def pump_until(predicate):
        deadline = time.monotonic() + 20
        errors = []

        def poll():
            try:
                if predicate():
                    root.quit()
                    return
                assert time.monotonic() < deadline, app.window.status.status_var.get()
                root.after(10, poll)
            except Exception as exc:
                errors.append(exc)
                root.quit()

        root.after(0, poll)
        root.mainloop()  # update() alone cannot dispatch worker-thread Tk calls.
        if errors:
            raise errors[0]

    def stopped():
        return app.session.finished()

    def ready():
        return app.flags.recording_ready and bool(app.recording.service.session_sizes)

    def capture(index, **kwargs):
        return ProcessCamera(index, _factory=_CameraSmokePreview, **kwargs)

    with patch("apps.tk.controllers.cameras.ProcessCamera", side_effect=capture):
        try:
            app.window.recording.record_skeleton_var.set(False)
            app.window.recording.auto_compare_var.set(False)
            app.cameras.preferred_primary = 0
            app.cameras.preferred_secondary = 1
            app.cameras.apply_entries([CameraEntry("摄像头 0", 0), CameraEntry("摄像头 1", 1)], True)
            app.window.practice.controls.student_id.set("offline-switch")
            service = app.cameras.service
            preview = {PRIMARY: CaptureSpec(0, *FULL_HD, policy="require"), SECONDARY: CaptureSpec(1, *FULL_HD, policy="require")}

            def preview_warm():
                # 会话结束后设备交回池（V-1）或重新预热，1080p 预览帧可用。
                return service.open_stream(preview).snapshot() is not None
            for _ in range(2):
                app.session.start()
                pump_until(ready)
                assert app.recording.service.session_sizes == {"front": FULL_HD, "side": FULL_HD}
                app.session.stop()
                pump_until(stopped)
                pump_until(preview_warm)
                # 已有预热时进入学生练习：三种模式同为 1080p（D-17），池复用或重新协商后格式不变。
                app.practice.enter()
                app.practice.start()
                pump_until(lambda: ready() and app.practice.practice.phase == "waiting_enter")
                assert app.recording.service.session_sizes == {"front": FULL_HD, "side": FULL_HD}
                assert app.recording.state == "idle"  # Blank frames cannot start a recording.
                app.practice.exit()
                pump_until(stopped)
                pump_until(preview_warm)
            # 开始后立即停止，再开始：无迟到打开，第二次会话正常就绪。
            app.session.start()
            app.session.stop()
            pump_until(stopped)
            app.session.start()
            pump_until(ready)
            app.session.stop()
            pump_until(stopped)
            # 双摄 → 单摄：第二路不再保留设备，单摄会话按 1080p 就绪。
            app.window.capture.camera_choice_2.set(NO_SECOND_CAMERA)
            app.cameras.secondary_selected()
            pump_until(lambda: not service.device_busy(1))
            app.session.start()
            pump_until(lambda: app.recording.service.session_sizes == {"front": FULL_HD})
            app.session.stop()
            pump_until(stopped)
        finally:
            if app.session.active():
                app.session.stop()
                pump_until(stopped)
            app.cameras.release_warmups()
            # 关窗前所有采集进程都必须真实退出，不遗留占用设备的子进程。
            assert app.cameras.service.close(5.0)


def check_tk_desktop(report: dict, root) -> None:
    """真实 Tk 桌面：预览与学生练习来回切换两次（真实 spawn），练习界面默认值正确，最后正常关窗。"""
    from unittest.mock import patch
    from apps.tk.app import DesktopApp

    with patch("apps.tk.controllers.cameras.CameraController.start_enumeration"):
        app = DesktopApp(root)
    root.update()
    check_camera_mode_switch(app, root)
    report["checks"].append("tk_preview_student_1080p_switch_twice_with_spawn")
    app.practice.enter()
    root.update()
    assert app.modes.is_active("practice")
    assert app.window.practice.controls.analysis_model.get() == "Full（精细分析）"
    report["checks"].append("tk_student_practice_ui")
    app.close()


def run_checks(report: dict, temporary: Path) -> None:
    """按顺序跑全部检查；任一失败即抛出，由 ``run`` 记入报告。"""
    from tkinter import Tk
    from core import foundation

    report["dataRoot"] = str(foundation.repo_root())
    check_build_info(report)
    check_models_and_inference(report, temporary)
    check_templates_and_preferences(report)
    video = check_video_record(report)
    history, record = check_practice_history(report, video)
    check_html_report(report, history, record)
    check_camera_spawn(report)
    root = Tk()
    root.withdraw()
    try:
        check_tk_desktop(report, root)   # 正常结束时由关窗协调器销毁窗口
    except BaseException:
        try:
            root.destroy()
        except Exception:
            pass
        raise


def run(report_path: Path) -> int:
    """在临时数据目录里离线跑全部检查，结果写入 ``report_path``（JSON）与同名 ``.log``；通过返回 0。"""
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "checks": []}
    block_network()
    try:
        with tempfile.TemporaryDirectory(prefix="散打 离线检查-") as temporary:
            os.environ["LOCALAPPDATA"] = temporary
            # 窗口版没有 stdout/stderr；保留可读诊断日志。
            with report_path.with_suffix(".log").open("w", encoding="utf-8", buffering=1) as log:
                sys.stdout = sys.stderr = log
                run_checks(report, Path(temporary))
                report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
