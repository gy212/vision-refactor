"""在实际打包 EXE 中运行的离线检查，不读写真实用户资料。"""
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


def _package_root() -> Path:
    """源码模式为仓库根，冻结模式为 PyInstaller 解包目录；与模块的 ``__file__`` 定位一致。"""
    return Path(__file__).resolve().parent.parent


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


def _check_camera_mode_switch(app, root):
    """Exercise real Tk callbacks, pool, spawn and claim with simulated devices."""
    from unittest.mock import patch
    from apps.app_ui import NO_SECOND_CAMERA
    from core.media import PRIMARY, SECONDARY, CaptureSpec, ProcessCamera
    from core.media import CameraDevice as CameraEntry

    def pump_until(predicate):
        deadline = time.monotonic() + 20
        errors = []

        def poll():
            try:
                if predicate():
                    root.quit()
                    return
                assert time.monotonic() < deadline, app.status_var.get()
                root.after(10, poll)
            except Exception as exc:
                errors.append(exc)
                root.quit()

        root.after(0, poll)
        root.mainloop()  # update() alone cannot dispatch worker-thread Tk calls.
        if errors:
            raise errors[0]

    def stopped():
        return app._session_finished()

    def ready():
        return app._dual_recording_ready and bool(app._recording.session_sizes)

    def capture(index, **kwargs):
        return ProcessCamera(index, _factory=_CameraSmokePreview, **kwargs)

    with patch("apps.app_ui.ProcessCamera", side_effect=capture):
        try:
            app.record_skeleton_var.set(False)
            app.auto_compare_var.set(False)
            app._preferred_primary_camera_index = 0
            app._preferred_secondary_camera_index = 1
            app._apply_camera_entries([CameraEntry("摄像头 0", 0), CameraEntry("摄像头 1", 1)], True)
            app.feedback_controls.student_id.set("offline-switch")
            service = app._camera_service
            preview = {PRIMARY: CaptureSpec(0, 1280, 720), SECONDARY: CaptureSpec(1, 1280, 720)}

            def preview_warm():
                # 会话结束后设备交回池（V-1）或重新预热，720p 预览帧可用。
                return service.open_stream(preview).snapshot() is not None
            for _ in range(2):
                app._start()
                pump_until(ready)
                assert app._recording.session_sizes == {"front": (1280, 720), "side": (1280, 720)}
                app._stop()
                pump_until(stopped)
                pump_until(preview_warm)
                # 已有 720p 预热时进入学生练习：池按 1080p 重新协商，旧格式不混入会话。
                app._enter_student_practice()
                app._on_practice_start()
                pump_until(lambda: ready() and app._practice.phase == "waiting_enter")
                assert app._recording.session_sizes == {"front": (1920, 1080), "side": (1920, 1080)}
                assert app._recording.state == "idle"  # Blank frames cannot start a recording.
                app._exit_student_practice()
                pump_until(stopped)
                pump_until(preview_warm)
            # 开始后立即停止，再开始：无迟到打开，第二次会话正常就绪。
            app._start()
            app._stop()
            pump_until(stopped)
            app._start()
            pump_until(ready)
            app._stop()
            pump_until(stopped)
            # 双摄 → 单摄：第二路不再保留设备，单摄会话按 720p 就绪。
            app.camera_choice_var_2.set(NO_SECOND_CAMERA)
            app._on_camera_2_selected()
            pump_until(lambda: not service.device_busy(1))
            app._start()
            pump_until(lambda: app._recording.session_sizes == {"front": (1280, 720)})
            app._stop()
            pump_until(stopped)
        finally:
            if app._session_active():
                app._stop()
                pump_until(stopped)
            app._release_camera_warmups()
            # 关窗前所有采集进程都必须真实退出，不遗留占用设备的子进程。
            assert app._camera_service.close(5.0)


def run(report_path: Path) -> int:
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "checks": []}

    def blocked(*args, **kwargs):
        raise RuntimeError("离线检查禁止网络访问")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    urllib.request.urlopen = blocked
    urllib.request.urlretrieve = blocked
    root = None
    try:
        with tempfile.TemporaryDirectory(prefix="散打 离线检查-") as temporary:
            os.environ["LOCALAPPDATA"] = temporary
            # 窗口版没有 stdout/stderr；保留可读诊断日志。
            with report_path.with_suffix(".log").open("w", encoding="utf-8", buffering=1) as log:
                sys.stdout = sys.stderr = log
                import numpy as np
                from core import paths, model_manager, online_matcher, video_writer
                from core.media import first_frame_ok
                from core.pose import EngineSpec, create_engine
                from core.services import HistoryStore
                from core.reporting import practice_report_renderers
                from core.evaluation import ACTIONS, RULE_VERSION, STANCES

                model_dir = paths.models_dir()
                report["dataRoot"] = str(paths.repo_root())
                build = load_build_info(_package_root())
                report["buildInfo"] = build
                if report["frozen"]:
                    assert build is not None, "包内缺少 build_info.json"
                    assert build["clean"] is True, "EXE 不是从干净的工作区构建的"
                    assert build["ruleVersion"] == RULE_VERSION
                report["checks"].append("build_info")
                report["feedbackIcons"] = check_feedback_icons(_package_root())
                report["checks"].append("feedback_status_icons")
                assert model_dir.is_relative_to(Path(temporary))
                assert all(model_manager.is_installed(spec) for spec in model_manager.MEDIAPIPE_MODELS)
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

                templates = online_matcher.load_template_library(paths.templates_dir() / "online")
                assert len(templates) == 2
                paths.save_camera_selection(0, None)
                assert paths.load_camera_selection() == (0, None)
                paths.save_record_dir(paths.outputs_dir())
                assert paths.load_record_dir() == paths.outputs_dir()
                report["checks"].append("templates_and_persistent_preferences")

                writer, video, codec = video_writer.open_video_writer(
                    paths.outputs_dir() / "空白 测试.avi", fps=12, size=(320, 240))
                try:
                    for _ in range(24):
                        writer.write(frame)
                finally:
                    writer.release()
                video = video_writer.transcode_to_h264(video)
                assert video.suffix == ".mp4", "内置 FFmpeg 转码失败"
                assert first_frame_ok(video), "录制文件不能解码"
                report["checks"].append("video_record_transcode_decode")

                history = HistoryStore(renderers=practice_report_renderers())
                side_video = video.with_name("侧面 测试.mp4")
                shutil.copyfile(video, side_video)
                record = history.add({"studentId": "offline-check", "studentName": "打包检查",
                                      "action": next(iter(ACTIONS)), "stance": next(iter(STANCES))}, video, side_video)
                assert record["result"]["backend"] == "mediapipe"
                assert record["result"]["fusionMethod"] == "action_rule_quality_weighted_v1"
                assert record["result"]["inputViews"] == ["front", "side"]
                assert record["result"]["ruleVersion"].endswith("-v4")
                assert record["result"]["poseVariant"] == "full"
                assert all(c["status"] in {"unable", "pending_rule"} for c in record["result"]["checks"])
                assert all(c["analyzedFrames"] == 24 and c["stride"] == 1 for c in record["result"]["capture"].values())
                report["feedbackVersion"] = record["result"]["ruleVersion"]
                report["feedbackCapture"] = record["result"]["capture"]
                assert history.get(record["id"])["id"] == record["id"]
                history.export(record["id"], paths.outputs_dir() / "问题说明.txt")
                assert (paths.outputs_dir() / "问题说明.txt").stat().st_size > 0
                assert not any("qwen" in name or name == "torch" for name in sys.modules)
                report["checks"].append("mediapipe_feedback_history_export")

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

                from tkinter import Tk
                from apps.app_ui import App
                from unittest.mock import patch
                root = Tk()
                root.withdraw()
                with patch.object(App, "_start_enumeration"):
                    app = App(root)
                root.update()
                _check_camera_mode_switch(app, root)
                report["checks"].append("tk_preview_student_1080p_switch_twice_with_spawn")
                app._enter_student_practice()
                root.update()
                assert app.modes.is_active("practice")
                assert app.feedback_controls.analysis_model.get() == "Full（精细分析）"
                report["checks"].append("tk_student_practice_ui")
                app._on_close()
                root = None
                report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1
