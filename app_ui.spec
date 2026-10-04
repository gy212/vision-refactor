# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：将 Tkinter 桌面 UI (apps/app_ui.py) 打包为 Windows 可执行程序。

构建命令（仓库根目录）：
    .\.venv\Scripts\python.exe -m PyInstaller app_ui.spec --noconfirm

产物位于 dist/vision_ui/，其中 vision_ui.exe 为入口。
模型文件 (models/*.task) 不嵌入二进制，由构建脚本复制到 exe 同级 models/ 目录，
便于后续替换 / 增量更新，且与 core/paths.py 的冻结模式路径解析（exe 所在目录）一致。
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# MediaPipe 携带大量数据文件（.binarypb / modules 等），必须整包收集。
for pkg in ("mediapipe",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 项目内部包 + 第三方运行时依赖的隐藏导入。
hiddenimports += collect_submodules("core")
hiddenimports += collect_submodules("apps")
hiddenimports += collect_submodules("analysis")
hiddenimports += collect_submodules("batch")
hiddenimports += [
    "pygrabber",
    "pygrabber.dshow_graph",
    "PIL._tkinter_finder",
    "cv2",
]


a = Analysis(
    ["apps/app_ui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["hypothesis", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vision_ui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="vision_ui",
)
