# -*- coding: utf-8 -*-
"""录制片段的路径生成与解析的唯一实现（纯路径逻辑，供 services、reporting、recording 共用）。

形状（设计文档 §8.4）：

- 单路：``<目录>/<YYYYMMDD>/record_<stamp>.mp4``
- 多路：``<目录>/<YYYYMMDD>/record_<stamp>/<view>.mp4``
- 采集记录（D-18）：多路为片段目录下的 ``capture.json``，单路为录像旁的 ``record_<stamp>.capture.json``

``stamp`` 形如 ``20261011_093015_123456``（含微秒，快速重录不撞名）；日期父目录取 stamp 前 8 位，
跨零点续写也落在开录当天。片段 id 是 ``record_<stamp>``。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

VIEWS = ("front", "side")
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"})
_ROLE_VIEWS = {"primary": "front", "secondary": "side"}
_ID_PREFIX = "record_"
CAPTURE_FILE = "capture.json"

DirectoryKey = tuple[str, int, int] | tuple[str, str]


def new_stamp(now: datetime | None = None) -> str:
    """新片段的时间戳。"""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def segment_id_for_stamp(stamp: str) -> str:
    return f"{_ID_PREFIX}{stamp}"


def view_for_role(role: str) -> str:
    """摄像头角色到视角名：primary→front，secondary→side。"""
    try:
        return _ROLE_VIEWS[role]
    except KeyError:
        raise ValueError(f"未知摄像头角色：{role!r}") from None


@dataclass(frozen=True)
class SegmentLayout:
    """以录制根目录为基准的片段路径。"""

    base_dir: Path

    def day_dir(self, stamp: str) -> Path:
        return Path(self.base_dir) / stamp[:8]

    def single_path(self, stamp: str) -> Path:
        """单路片段文件。"""
        return self.day_dir(stamp) / f"{segment_id_for_stamp(stamp)}.mp4"

    def segment_dir(self, stamp: str) -> Path:
        """多路片段目录。"""
        return self.day_dir(stamp) / segment_id_for_stamp(stamp)

    def view_path(self, stamp: str, view: str) -> Path:
        """多路片段中某一视角的文件。"""
        return self.segment_dir(stamp) / f"{view}.mp4"

    def capture_path(self, stamp: str, *, multi: bool) -> Path:
        """片段的采集记录 ``capture.json``：多路在片段目录里，单路是录像旁的同名旁文件。"""
        if multi:
            return self.segment_dir(stamp) / CAPTURE_FILE
        return self.day_dir(stamp) / f"{segment_id_for_stamp(stamp)}.{CAPTURE_FILE}"

    @staticmethod
    def segment_id(path: Path | str) -> str:
        """片段 id：多路取目录名，单路取文件名去掉后缀；多路片段里的视角文件取所在目录名。"""
        path = Path(path)
        if path.suffix.lower() in VIDEO_SUFFIXES:
            if path.stem in VIEWS and path.parent.name.startswith(_ID_PREFIX):
                return path.parent.name
            return path.stem
        return path.name

    @staticmethod
    def resolve_videos(directory: Path | str) -> tuple[Path | None, Path | None]:
        """多路片段目录里的正、侧视频；同一视角有多种后缀时优先 .mp4，再按文件名排序。"""
        directory = Path(directory)
        found: list[Path | None] = []
        for view in VIEWS:
            candidates = sorted(
                directory.glob(view + ".*"),
                key=lambda p: (p.suffix.lower() != ".mp4", p.name),
            )
            found.append(next(
                (p for p in candidates if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
                None,
            ))
        return found[0], found[1]

    @staticmethod
    def child(directory: Path | str, name: str) -> Path | None:
        """目录下的直接子项路径；名字含路径分隔或会越出目录时返回 None。不检查是否存在。"""
        if not isinstance(name, str) or not name or name in {".", ".."}:
            return None
        if Path(name).name != name or "/" in name or "\\" in name:
            return None
        return Path(directory) / name

    @staticmethod
    def resolve_contained(root: Path | str, *parts: str) -> Path | None:
        """逐级解析 ``root/parts...``：每一级都必须是上一级的直接子项、不是符号链接，末级是文件。"""
        current = Path(root).resolve()
        try:
            for part in parts:
                candidate = SegmentLayout.child(current, part)
                if candidate is None or candidate.is_symlink():
                    return None
                resolved = candidate.resolve()
                if resolved.parent != current:
                    return None
                current = resolved
            return current if current.is_file() else None
        except OSError:
            return None

    @staticmethod
    def directory_keys(path: Path | str) -> tuple[DirectoryKey, ...]:
        """片段目录的身份键：规范化路径文本，能取到 inode 时再加 ``(dev, inode)``。用于防止同一目录重复提交。"""
        resolved = Path(path).resolve(strict=False)
        text_key = _directory_text_key(resolved)
        try:
            stat = _directory_stat(resolved)
        except OSError:
            return (text_key,)
        inode = int(getattr(stat, "st_ino", 0) or 0)
        if inode:
            return (text_key, ("stat", int(stat.st_dev), inode))
        return (text_key,)


def _directory_text_key(path: Path) -> DirectoryKey:
    """规范化路径文本；Windows 下去掉扩展路径前缀并忽略大小写。"""
    text = str(path)
    if os.name == "nt":
        folded = text.casefold()
        if folded.startswith("\\\\?\\unc\\"):
            text = "\\\\" + text[8:]
        elif folded.startswith("\\\\?\\"):
            text = text[4:]
    return ("path", os.path.normcase(os.path.normpath(text)))


def _directory_stat(path: Path) -> os.stat_result:
    return path.stat()
