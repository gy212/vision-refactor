# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 单文件打包：把桌面 UI 打包成单个 vision_ui.exe（仓库唯一的 Tk spec）。

构建命令（仓库根目录，依赖见 requirements-dev.txt）：
    .\.venv\Scripts\python.exe -m PyInstaller app_ui_onefile.spec --noconfirm

产物：
  - dist/vision_ui.exe：单文件，含 mediapipe/opencv 运行时、模型、在线模板、
    结果页状态图标（assets/feedback）、ffmpeg 与 build_info.json。
  - dist/vision_ui_<日期>_<commit>/：交付目录，EXE + build_info.json +
    packaging/delivery/ 下的附带文件（使用说明、配置示例、第三方许可）。

只从干净的 commit 构建：packaging/pyinstaller/build_info.py 的 CLEAN_PATHS 下有
未提交改动或未跟踪文件时拒绝构建。自检：vision_ui.exe --offline-self-check <报告.json>。

体积控制（重要）
----------------
仓库装有 CUDA 版 torch（约 4GB）+ ultralytics 等 YOLO 依赖，但桌面 UI 实时识别只用
MediaPipe，完全不碰 YOLO。因此：
  - 不用 collect_submodules 粗放收集 analysis/batch（会顺着 core.yolo_adapter 的
    ``from ultralytics import YOLO`` 把 torch/CUDA 整条链打进来，使 exe 膨胀到 3GB+）。
  - 通过 excludes 显式切断 torch / torchvision / ultralytics / nvidia* / onnxruntime
    等重依赖，把 onefile 体积压到数百 MB。

模型策略
--------
内置 lite/full/heavy 三个人体模型及手部模型，首次启动释放到用户数据目录。
"""

import importlib.util
from pathlib import Path

import imageio_ffmpeg
import PyInstaller
from PyInstaller.utils.hooks import collect_all


def _load(name):
    # 顶层包名 packaging 与 PyPI 的 packaging 冲突，按文件路径加载构建辅助模块。
    path = Path("packaging") / "pyinstaller" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"vision_build_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_info = _load("build_info")
delivery = _load("delivery")
build_info.require_clean(Path("."))

model_names = ["pose_landmarker_lite.task", "pose_landmarker_full.task",
               "pose_landmarker_heavy.task", "hand_landmarker.task"]
for name in model_names:
    if not (Path("models") / name).is_file():
        raise SystemExit(f"离线打包缺少模型：models/{name}")
info = build_info.collect(Path("."), [Path("models") / name for name in model_names],
                          pyinstaller_version=PyInstaller.__version__)
info_path = build_info.write(info, Path("build") / build_info.BUILD_INFO_NAME)

datas = [(str(Path("models") / name), "models") for name in model_names]
datas += [("templates/online/直拳_左手.npz", "templates/online"),
          ("templates/online/直拳_右手.npz", "templates/online")]
datas += [("assets/feedback", "assets/feedback")]
datas += [(str(info_path), ".")]
datas += [(imageio_ffmpeg.get_ffmpeg_exe(), "ffmpeg")]
binaries = []
hiddenimports = []

# MediaPipe 携带大量数据文件（.binarypb / modules 等），必须整包收集。
for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 只显式声明 UI 真正用到的内部模块，避免 collect_submodules 把 YOLO 链拉进来。
hiddenimports += [
    "core.action_compare",
    "core.vision_pipeline",
    "core.pose_features",
    "core.feature_layout",
    "core.rule_scoring",
    "core.paths",
    "core.model_manager",
    "core.parallel_pose_engine",
    "core.recording_controller",
    "core.video_writer",
    "core.body_core_compare",
    "core.media",
    "core.media.camera",
    "core.media.camera.capture_process",
    "core.media.camera.devices",
    "core.media.camera.pool",
    "core.media.camera.service",
    "core.media.frames",
    "core.media.video_source",
    "core.media.writer",
    "core.pose",
    "core.pose.engine",
    "core.pose.extract",
    "core.pose.raw_series",
    "core.pose.render",
    "core.pose.backends.body_core",
    "analysis.tech_eval",
    "apps.camera_enum",
    "apps.input_source",
    "pygrabber",
    "pygrabber.dshow_graph",
    "PIL._tkinter_finder",
    "cv2",
]

# 切断 YOLO / 深度学习重依赖（UI 不使用）。这是把 exe 从 3GB 压到数百 MB 的关键。
heavy_excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "onnxslim",
    "tensorrt",
    "core.yolo_adapter",
    "core.pose.backends.yolo",
    "core.pose.backends.yolo_mapping",
    "analysis.spike_yolo_baseline",
    "hypothesis",
    "pytest",
]
# 排除所有 nvidia-* CUDA 运行时包。
nvidia_excludes = [
    "nvidia",
]


a = Analysis(
    ["apps/desktop_launcher.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["packaging/pyinstaller/pyi_rth_video_writer_alias.py"],
    excludes=heavy_excludes + nvidia_excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="vision_ui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# EXE() 返回时单文件已生成；按 P0-3 组织交付目录。
delivery_dir = delivery.assemble(Path(DISTPATH) / "vision_ui.exe", info_path, info, Path(DISTPATH))
print(f"交付目录：{delivery_dir}（commit {info['commit'][:7]}，规则 {info['ruleVersion']}）")
