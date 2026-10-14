"""离线 EXE 入口：`--offline-self-check` 跑 `apps.tk.self_check`，否则启动 Tk 桌面 `apps.tk.app`。"""
from __future__ import annotations

import multiprocessing
import sys
import traceback
from pathlib import Path


def main() -> None:
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--offline-self-check":
        from apps.tk.self_check import run
        sys.exit(run(Path(sys.argv[2])))

    log_path = None
    try:
        # core.paths 经 core.foundation 会导入 numpy，失败也要走下面的报错路径
        from core.paths import repo_root
        log_path = repo_root() / "desktop.log"
        if sys.stdout is None:
            sys.stdout = sys.stderr = log_path.open("a", encoding="utf-8", buffering=1)
        from apps.tk.app import main as run_ui
        run_ui()
    except Exception:
        traceback.print_exc()
        from tkinter import messagebox
        where = log_path if log_path is not None else "（日志尚未建立）"
        messagebox.showerror("启动失败", f"请把错误日志发给维护人员：\n{where}")
        raise


if __name__ == "__main__":
    main()
