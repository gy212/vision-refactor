# -*- coding: utf-8 -*-
"""``PoolArgumentChecks``：摄像头池公共方法的参数校验与采集工厂关键字探测，由 ``pool.py`` 拆出（D-15）。"""
from __future__ import annotations

import inspect
from numbers import Integral
from typing import Any, Callable, Final

from core.media.camera.slot import WarmupSlot
from core.media.camera.spec import ROLES, CameraWarmupError, CaptureSpec

_FACTORY_KEYWORDS: Final = ("stop_event", "width", "height", "expected_name", "policy")

CaptureFactory = Callable[..., Any]


class PoolArgumentChecks:
    """角色、编号、规格、超时、游标与代号的校验；工厂支持的关键字；异常归一为 ``CameraWarmupError``。"""

    @staticmethod
    def _validate_role(role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES!r}")
        return role

    @staticmethod
    def _validate_index(index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("camera index must be a non-negative integer")
        normalized = int(index)
        if normalized < 0:
            raise ValueError("camera index must be a non-negative integer")
        return normalized

    @classmethod
    def _validate_spec(cls, spec: CaptureSpec | int) -> CaptureSpec:
        if isinstance(spec, CaptureSpec):
            return spec
        return CaptureSpec(cls._validate_index(spec))

    def _validate_pair(
        self, primary: CaptureSpec | int, secondary: CaptureSpec | int
    ) -> tuple[CaptureSpec, CaptureSpec]:
        first = self._validate_spec(primary)
        second = self._validate_spec(secondary)
        if first.index == second.index:
            raise ValueError("primary and secondary cameras must be different")
        return first, second

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        try:
            normalized = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be a non-negative number") from exc
        if normalized < 0:
            raise ValueError("timeout must be a non-negative number")
        return normalized

    @staticmethod
    def _validate_after(after: tuple[int, int] | None) -> tuple[int, int] | None:
        if after is None:
            return None
        if not isinstance(after, tuple) or len(after) != 2:
            raise ValueError("after must be a two-item sequence cursor")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in after):
            raise ValueError("after must contain integer sequences")
        return after

    @staticmethod
    def _validate_generations(
        generations: tuple[int, ...] | None, size: int
    ) -> tuple[int, ...] | None:
        if generations is None:
            return None
        if not isinstance(generations, tuple) or len(generations) != size:
            raise ValueError(f"expected_generations must be a {size}-item tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in generations
        ):
            raise ValueError("expected_generations must contain positive integers")
        return generations

    @staticmethod
    def _supported_keywords(factory: CaptureFactory) -> frozenset[str]:
        try:
            parameters = list(inspect.signature(factory).parameters.values())
        except (TypeError, ValueError):
            return frozenset()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
            return frozenset(_FACTORY_KEYWORDS)
        return frozenset(
            p.name
            for p in parameters
            if p.name in _FACTORY_KEYWORDS
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )

    @staticmethod
    def _normalize_error(slot: WarmupSlot, exc: BaseException) -> BaseException:
        if isinstance(exc, CameraWarmupError):
            return exc
        return CameraWarmupError(f"{slot.role} camera {slot.index} failed: {exc}")


__all__ = ["CaptureFactory", "PoolArgumentChecks"]
