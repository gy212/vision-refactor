# -*- coding: utf-8 -*-
"""原子写文件的唯一实现：同目录临时文件写完、flush 并 fsync 后整体替换目标，失败不留半截文件。

Windows 上目标被其它进程短暂占用（杀毒、索引、Excel 预览）时 ``os.replace`` 会抛
``PermissionError``，这里有界重试；仍失败时沿用成绩表原来的 copy 回退，最后把异常抛给调用方。
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPLACE_RETRIES = 5
REPLACE_RETRY_DELAY_S = 0.05


def _fsync_file(path: Path) -> None:
    with open(path, "rb+") as fh:
        fh.flush()
        os.fsync(fh.fileno())


def replace_file(
    src: Path | str,
    dst: Path | str,
    *,
    durable: bool = True,
    retries: int = REPLACE_RETRIES,
    delay_s: float = REPLACE_RETRY_DELAY_S,
) -> None:
    """把写好的 ``src`` 替换为 ``dst``。

    ``durable`` 时先 fsync ``src``。``PermissionError`` 最多重试 ``retries`` 次；
    其它 ``OSError``（如跨卷）或重试用尽后改用"旧文件挪成 .bak → 复制 → 删 .bak"，
    复制失败时把 .bak 还原并抛出原异常。
    """
    src, dst = Path(src), Path(dst)
    if durable:
        _fsync_file(src)
    for attempt in range(retries + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= retries:
                break
            time.sleep(delay_s)
        except OSError:
            break
    _copy_replace(src, dst)


def _copy_replace(src: Path, dst: Path) -> None:
    bak: Path | None = dst.with_suffix(dst.suffix + ".bak")
    try:
        if dst.exists():
            try:
                os.replace(dst, bak)
            except OSError:
                bak = None
        shutil.copy2(src, dst)
        src.unlink(missing_ok=True)
        if bak is not None and bak.exists():
            bak.unlink(missing_ok=True)
    except OSError:
        if bak is not None and bak.exists() and not dst.exists():
            try:
                os.replace(bak, dst)
            except OSError:
                pass
        raise


def temporary_sibling(path: Path | str, *, tag: str = "tmp") -> Path:
    """与目标同目录、ASCII 文件名、保留原后缀的临时路径（部分库按后缀判断格式）。"""
    path = Path(path)
    return path.with_name(f"_{tag}_{uuid.uuid4().hex}{path.suffix}")


@contextmanager
def atomic_target(path: Path | str, *, durable: bool = True, temporary: Path | None = None) -> Iterator[Path]:
    """产出临时路径，由调用方写入；正常退出时替换为 ``path``，异常时删除临时文件并原样抛出。

    调用方没有写出临时文件就正常退出时，按 ``FileNotFoundError`` 处理。
    """
    path = Path(path)
    tmp = temporary if temporary is not None else temporary_sibling(path)
    try:
        yield tmp
        if not tmp.is_file():
            raise FileNotFoundError(f"临时文件未写出：{tmp}")
        replace_file(tmp, path, durable=durable)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    with atomic_target(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: Path | str,
    data: Any,
    *,
    indent: int | None = 2,
    trailing_newline: bool = False,
) -> None:
    """``json.dumps(ensure_ascii=False)`` 后原子写入，UTF-8。"""
    text = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text + ("\n" if trailing_newline else ""))
