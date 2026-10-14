"""CLI 入口壳：摄像头或视频的实时骨架预览与导出；实现在 ``apps.cli.main``，命令行参数不变。"""
import sys
from pathlib import Path

# 允许以 ``python apps/xxx.py``、``python batch/xxx.py`` 方式运行（AGENTS.md 记载的命令）。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.cli.main import main

if __name__ == "__main__":
    main()
