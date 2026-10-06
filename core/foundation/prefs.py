# -*- coding: utf-8 -*-
"""用户偏好的唯一读写：``user_prefs.json`` 的键集中定义在这里，读改写在进程内锁下原子替换。

- 键名与 JSON 结构沿用现状（用户数据兼容）；未知键原样保留。
- 每个键的 ``parse`` 负责校验与旧格式迁移，非法值回退 ``default``，读取永不抛异常。
- 写入尽力而为：文件不可写时静默放弃，与原先三套实现一致。
- 偏好文件路径只由 ``_prefs_path()`` 提供，每次读写都现取；``tests/conftest.py`` 把它钉到临时文件。
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from core.foundation import paths
from core.foundation.atomic_io import atomic_write_json

T = TypeVar("T")


def _prefs_path() -> Path:
    return paths.repo_root() / "user_prefs.json"


def _identity(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class PrefKey(Generic[T]):
    """一个顶层键：``parse`` 把 JSON 值转成 T（非法值返回 ``default``），``dump`` 把 T 转回 JSON 值。"""

    name: str
    default: T
    parse: Callable[[Any], T]
    dump: Callable[[T], Any] = field(default=_identity)
    fallback: "PrefKey[T] | None" = None  # 本键缺失或为默认值时改读的旧格式键

    def read(self, raw: Mapping[str, Any]) -> T:
        value = self.default
        if self.name in raw:
            try:
                value = self.parse(raw[self.name])
            except (TypeError, ValueError, KeyError, IndexError):
                value = self.default
        if self.fallback is not None and value == self.default:
            return self.fallback.read(raw)
        return value


class PreferenceStore:
    """偏好文件的读改写。锁是类级的：同一进程内的所有实例共用，并发写不丢键。"""

    _lock = threading.RLock()

    def __init__(self, path_fn: Callable[[], Path] | None = None) -> None:
        self._path_fn = path_fn

    def path(self) -> Path:
        return self._path_fn() if self._path_fn is not None else _prefs_path()

    def raw(self) -> dict[str, Any]:
        """整份偏好；文件缺失、损坏或顶层不是对象时为空。"""
        try:
            data = json.loads(self.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return dict(data) if isinstance(data, dict) else {}

    def get(self, key: PrefKey[T]) -> T:
        return key.read(self.raw())

    def set(self, key: PrefKey[T], value: T) -> None:
        self.update({key: value})

    def update(self, values: Mapping[PrefKey[Any], Any]) -> None:
        """一次读改写写入多个键，其余键（含未知键）保留。"""
        with self._lock:
            data = self.raw()
            for key, value in values.items():
                data[key.name] = key.dump(value)
            try:
                atomic_write_json(self.path(), data)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# 键定义（JSON 顶层键名沿用现状）
# --------------------------------------------------------------------------- #


def _camera_index(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_camera_selection(raw: Any) -> tuple[int | None, int | None]:
    if not isinstance(raw, dict):
        return None, None
    return _camera_index(raw.get("primary_index")), _camera_index(raw.get("secondary_index"))


def _dump_camera_selection(value: tuple[int | None, int | None]) -> dict[str, int | None]:
    return {"primary_index": _camera_index(value[0]), "secondary_index": _camera_index(value[1])}


def _parse_text(raw: Any) -> str:
    return str(raw or "").strip()


def _parse_path_list(raw: Any) -> tuple[Path, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(Path(str(p).strip()) for p in raw if str(p).strip())


def _parse_single_path(raw: Any) -> tuple[Path, ...]:
    text = _parse_text(raw)
    return (Path(text),) if text else ()


def _dump_path_list(value: Any) -> list[str]:
    return [str(p) for p in value]


DEFAULT_EXAM_ROI = (0.2, 0.1, 0.8, 0.95)


def _parse_roi(raw: Any) -> tuple[float, float, float, float]:
    """归一化 ROI (x0, y0, x1, y1)，须满足 0≤x0<x1≤1 且 0≤y0<y1≤1；否则回退默认。"""
    x0, y0, x1, y1 = (float(v) for v in raw)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("invalid ROI")
    return x0, y0, x1, y1


def _float_key(name: str, default: float) -> PrefKey[float]:
    return PrefKey(name, default, float)


CAMERA_SELECTION: PrefKey[tuple[int | None, int | None]] = PrefKey(
    "camera_selection", (None, None), _parse_camera_selection, _dump_camera_selection
)
RECORD_DIR: PrefKey[str] = PrefKey("record_dir", "", lambda raw: str(raw or ""), str)

# 一条龙/考试的正、侧模板池；旧的单值键只读不写，作为列表键缺失时的回退。
FRONT_TEMPLATE_LEGACY: PrefKey[tuple[Path, ...]] = PrefKey("auto_compare_front_template", (), _parse_single_path)
SIDE_TEMPLATE_LEGACY: PrefKey[tuple[Path, ...]] = PrefKey("auto_compare_side_template", (), _parse_single_path)
FRONT_TEMPLATES: PrefKey[tuple[Path, ...]] = PrefKey(
    "auto_compare_front_templates", (), _parse_path_list, _dump_path_list, FRONT_TEMPLATE_LEGACY
)
SIDE_TEMPLATES: PrefKey[tuple[Path, ...]] = PrefKey(
    "auto_compare_side_templates", (), _parse_path_list, _dump_path_list, SIDE_TEMPLATE_LEGACY
)

EXAM_ROSTER_PATH: PrefKey[str] = PrefKey("exam_roster_path", "", _parse_text, str)
EXAM_ROI: PrefKey[tuple[float, float, float, float]] = PrefKey("exam_roi_norm", DEFAULT_EXAM_ROI, _parse_roi, list)
EXAM_ROI_ROTATE: PrefKey[int] = PrefKey("exam_roi_rotate", 0, int)
EXAM_POST_CALL_GUARD_S = _float_key("exam_post_call_guard_s", 1.5)
EXAM_INTER_STUDENT_GAP_S = _float_key("exam_inter_student_gap_s", 2.0)
EXAM_ENTER_STABLE_S = _float_key("exam_enter_stable_s", 0.8)
EXAM_EMPTY_HOLD_S = _float_key("exam_empty_hold_s", 2.0)
EXAM_MIN_RECORD_S = _float_key("exam_min_record_s", 3.0)


# --------------------------------------------------------------------------- #
# 常用读写（原 core/paths.py 与 apps/recording_postprocess.py 的公开函数）
# --------------------------------------------------------------------------- #


def load_camera_selection() -> tuple[int | None, int | None]:
    """读取主/次摄像头索引；损坏或旧格式配置按未设置处理。"""
    return PreferenceStore().get(CAMERA_SELECTION)


def save_camera_selection(primary_index: int | None, secondary_index: int | None) -> None:
    """持久化主/次摄像头索引并保留其他用户偏好。"""
    PreferenceStore().set(CAMERA_SELECTION, (primary_index, secondary_index))


def load_record_dir() -> Path:
    """读取用户上次选择的录制保存目录；无记录或目录已不存在时回退 ``outputs_dir()``。"""
    saved = PreferenceStore().get(RECORD_DIR)
    if saved and Path(saved).is_dir():
        return Path(saved)
    return paths.outputs_dir()


def save_record_dir(directory: Path | str) -> None:
    """持久化录制保存目录（尽力而为，失败静默）。"""
    PreferenceStore().set(RECORD_DIR, str(directory))


def load_template_pools() -> tuple[list[Path], list[Path]]:
    """正、侧模板池；列表键缺失时回落旧单值键，都缺为空列表（由调用方决定是否用默认模板）。"""
    store = PreferenceStore()
    raw = store.raw()
    return list(FRONT_TEMPLATES.read(raw)), list(SIDE_TEMPLATES.read(raw))


def save_template_pools(
    front: list[Path | str] | None = None,
    side: list[Path | str] | None = None,
) -> None:
    """持久化模板池，只写传入的一侧（尽力而为，失败静默）。"""
    values: dict[PrefKey[Any], Any] = {}
    if front is not None:
        values[FRONT_TEMPLATES] = tuple(front)
    if side is not None:
        values[SIDE_TEMPLATES] = tuple(side)
    if values:
        PreferenceStore().update(values)
