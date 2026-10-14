"""生成 EXE 的构建信息 build_info，并在源码不干净时拒绝构建。

由 ``app_ui_onefile.spec`` 按文件路径加载（顶层包名 ``packaging`` 与 PyPI 的
``packaging`` 冲突，不能按包导入）。自检报告读取打进 EXE 的 ``build_info.json``。
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BUILD_INFO_NAME = "build_info.json"

# 参与打包的源码与资源；任何改动或未跟踪文件都算不干净（v4 曾从未跟踪源码出包）。
CLEAN_PATHS: tuple[str, ...] = (
    "core", "apps", "analysis", "batch", "assets", "templates", "packaging", "app_ui_onefile.spec",
)

RULE_VERSION_SOURCE = "core/evaluation/rules/sanda_v5.py"  # 规则版本（D-3 起考试、练习、离线共用）
_RULE_VERSION_RE = re.compile(r'^RULE_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


def dirty_entries(repo: Path, paths: tuple[str, ...] = CLEAN_PATHS) -> list[str]:
    """返回 ``git status --porcelain`` 在打包路径下的条目；空列表表示干净。"""
    out = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    return [line for line in out.splitlines() if line.strip()]


def require_clean(repo: Path, paths: tuple[str, ...] = CLEAN_PATHS) -> None:
    """工作区不干净时抛出 ``SystemExit``，列出前 20 个条目。"""
    entries = dirty_entries(repo, paths)
    if entries:
        listing = "\n".join(entries[:20])
        raise SystemExit(f"拒绝构建：打包路径下有未提交改动或未跟踪文件，先提交再构建：\n{listing}")


def read_rule_version(repo: Path) -> str:
    """从源码读取 ``RULE_VERSION_SOURCE`` 的 ``RULE_VERSION``，不导入业务模块。"""
    text = (repo / RULE_VERSION_SOURCE).read_text(encoding="utf-8")
    match = _RULE_VERSION_RE.search(text)
    if match is None:
        raise SystemExit(f"{RULE_VERSION_SOURCE} 中找不到 RULE_VERSION")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(repo: Path, model_files: list[Path], *, pyinstaller_version: str) -> dict:
    """汇总 commit、构建时间、规则版本、模型哈希、Python 与 PyInstaller 版本、是否干净。"""
    return {
        "commit": _git(repo, "rev-parse", "HEAD").strip(),
        "clean": not dirty_entries(repo),
        "cleanPaths": list(CLEAN_PATHS),
        "builtAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ruleVersion": read_rule_version(repo),
        "models": {path.name: sha256_file(path) for path in model_files},
        "python": platform.python_version(),
        "pyinstaller": pyinstaller_version,
    }


def write(info: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
