# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 单文件打包：把桌面 UI (apps/app_ui.py) 打包成单个 vision_ui.exe。

构建命令（仓库根目录）：
    .\.venv\Scripts\python.exe -m PyInstaller app_ui_onefile.spec --noconfirm

产物：dist/vision_ui.exe（单文件，含 mediapipe/opencv 运行时）。

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
构建环境另需安装 pyinstaller、imageio-ffmpeg（仅构建依赖，视频转码工具随包分发）。
"""

from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import imageio_ffmpeg

model_names = ["pose_landmarker_lite.task", "pose_landmarker_full.task",
               "pose_landmarker_heavy.task", "hand_landmarker.task"]
for name in model_names:
    if not (Path("models") / name).is_file():
        raise SystemExit(f"离线打包缺少模型：models/{name}")
datas = [(str(Path("models") / name), "models") for name in model_names]
datas += [("templates/online/直拳_左手.npz", "templates/online"),
          ("templates/online/直拳_右手.npz", "templates/online")]
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
    "analysis.tech_eval",
    "apps.camera_enum",
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
