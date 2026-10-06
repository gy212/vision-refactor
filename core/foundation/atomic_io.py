# -*- coding: utf-8 -*-
"""原子写文件的唯一实现：同目录临时文件写完、flush 并 fsync 后整体替换目标，失败不留半截文件。

Windows 上目标被其它进程短暂占用（杀毒、索引、Excel 预览）时 ``os.replace`` 会抛
``PermissionError``，这里有界重试；仍失败时先把旧文件挪成 ``.bak`` 再复制，复制失败就还原 ``.bak``。
旧文件挪不开时不就地覆盖，原样保留并把异常抛给调用方。
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


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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
    其它 ``OSError`` 或重试用尽后改走 ``.bak`` 复制回退（见模块说明）。回退也失败时抛出最初的异常，
    ``dst`` 保持原内容。
    """
    src, dst = Path(src), Path(dst)
    if dst.is_dir():
        raise IsADirectoryError(f"替换目标是目录：{dst}")
    if durable:
        _fsync_file(src)
    for attempt in range(retries + 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= retries:
                if not _copy_replace(src, dst):
                    raise
                return
            time.sleep(delay_s)
        except OSError:
            if not _copy_replace(src, dst):
                raise
            return


def _copy_replace(src: Path, dst: Path) -> bool:
    """``.bak`` 复制回退；返回是否已写入。失败时 ``dst`` 为原内容（原先不存在则仍不存在）。"""
    bak: Path | None = None
    if dst.exists():
        bak = dst.with_name(dst.name + ".bak")
        try:
            os.replace(dst, bak)
        except OSError:
            return False  # 旧文件挪不开：不就地覆盖
    try:
        shutil.copy2(src, dst)
    except OSError:
        if bak is None:
            _discard(dst)
        else:
            try:
                os.replace(bak, dst)
            except OSError:
                pass  # 还原失败时 .bak 留给人工恢复
        return False
    # 已写入；清理失败不影响结果
    _discard(src)
    if bak is not None:
        _discard(bak)
    return True


def temporary_sibling(path: Path | str) -> Path:
    """与目标同目录、ASCII 文件名的临时路径。

    后缀固定为 ``.tmp``：进程中途被杀留下的残片不会被按后缀扫描目录的代码（如 ``*.npz`` 模板库）当成正式文件。
    """
    path = Path(path)
    return path.with_name(f"_tmp_{uuid.uuid4().hex}.tmp")


@contextmanager
def atomic_target(path: Path | str, *, durable: bool = True) -> Iterator[Path]:
    """产出临时路径，由调用方写入；正常退出时替换为 ``path``，异常时删除临时文件并原样抛出。

    调用方没有写出临时文件就正常退出时，按 ``FileNotFoundError`` 处理。
    """
    path = Path(path)
    tmp = temporary_sibling(path)
    try:
        yield tmp
        if not tmp.is_file():
            raise FileNotFoundError(f"临时文件未写出：{tmp}")
        replace_file(tmp, path, durable=durable)
    finally:
        _discard(tmp)


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    with atomic_target(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    """按文本模式写入（换行随平台转换，与 ``Path.write_text`` 一致）后原子替换。"""
    with atomic_target(path) as tmp:
        tmp.write_text(text, encoding=encoding)


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
