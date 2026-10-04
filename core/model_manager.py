# -*- coding: utf-8 -*-
"""模型清单与下载管理（供 UI「设置 → 模型管理」使用）。

为什么需要它
------------
打包成单文件 exe 后，模型文件不再随程序分发。用户首次使用前需要把模型文件放到
可执行文件同级的 ``models/`` 目录下（见 ``core.paths.models_dir()``）。
本模块集中维护：

- 每个自动下载模型的官方下载地址（MediaPipe 使用 Google 官方资产）。
- YOLO 档位的能力展示与手动安装边界。
- 模型在本地的目标路径与「是否已安装」状态查询。
- 带进度回调、可中断的下载实现（下载到 ``.part`` 临时文件后原子改名，避免半成品被
  误判为已安装）。

约束
----
- 仅依赖标准库 + ``core.paths``，不导入 mediapipe / cv2 等重依赖，便于 UI 早期加载与单测。
- 下载源固定使用官方直链；开发侧默认走本机代理 ``http://127.0.0.1:7890``，可通过
  ``VISION_MODEL_PROXY`` 覆盖或置空。
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.paths import models_dir

# MediaPipe 官方模型资产基址。
_MP_BASE = "https://storage.googleapis.com/mediapipe-models"
DEFAULT_MODEL_PROXY = "http://127.0.0.1:7890"

# 进度回调签名：progress_cb(downloaded_bytes, total_bytes_or_None)。
ProgressCb = Callable[[int, int | None], None]


@dataclass(frozen=True)
class ModelSpec:
    """单个可下载模型的描述。

    - key:       稳定标识（用于 UI 选择 / 日志）。
    - filename:  落盘到 ``models/`` 下的文件名（与 ``vision_pipeline`` 加载时一致）。
    - url:       官方下载直链；不可自动下载档位可为空。
    - label:     UI 显示名。
    - approx_mb: 体积估计（仅用于 UI 提示，可为 None）。
    """

    key: str
    filename: str
    url: str
    label: str
    approx_mb: float | None = None
    category: str = "mediapipe"
    profile: str = ""
    downloadable: bool = True
    installed_supported: bool = True
    default_route_eligible: bool = True
    note: str = ""


MEDIAPIPE_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="pose_lite",
        filename="pose_landmarker_lite.task",
        url=f"{_MP_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        label="人体姿态 - lite（轻量，最快）",
        approx_mb=5.5,
        profile="pose_lite",
    ),
    ModelSpec(
        key="pose_full",
        filename="pose_landmarker_full.task",
        url=f"{_MP_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        label="人体姿态 - full（默认，均衡）",
        approx_mb=9.0,
        profile="pose_full",
    ),
    ModelSpec(
        key="pose_heavy",
        filename="pose_landmarker_heavy.task",
        url=f"{_MP_BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        label="人体姿态 - heavy（最准，最慢）",
        approx_mb=29.2,
        profile="pose_heavy",
    ),
    ModelSpec(
        key="hand",
        filename="hand_landmarker.task",
        url=f"{_MP_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        label="手部关键点 - hand_landmarker（V 手势 / 手部骨架）",
        approx_mb=7.5,
        profile="hand",
    ),
)

YOLO_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="yolo26n",
        filename="yolo26n-pose.pt",
        url="",
        label="YOLO26n pose（实时 body-only 预览档）",
        category="yolo",
        profile="yolo26n/s",
        downloadable=False,
        installed_supported=False,
        default_route_eligible=True,
        note="当前安装版 sidecar 不打包 YOLO runtime；仅显示能力状态，后续接入下载/依赖策略。",
    ),
    ModelSpec(
        key="yolo26s",
        filename="yolo26s-pose.pt",
        url="",
        label="YOLO26s pose（实时 body-only 预览档）",
        category="yolo",
        profile="yolo26n/s",
        downloadable=False,
        installed_supported=False,
        default_route_eligible=True,
        note="当前安装版 sidecar 不打包 YOLO runtime；仅显示能力状态，后续接入下载/依赖策略。",
    ),
    ModelSpec(
        key="yolo26l",
        filename="yolo26l-pose.pt",
        url="",
        label="YOLO26L pose（离线高质量 body-only 分析档）",
        category="yolo",
        profile="yolo26l",
        downloadable=False,
        installed_supported=False,
        default_route_eligible=True,
        note="当前安装版 sidecar 不打包 YOLO runtime；缺失时返回结构化下载/安装错误，不静默回退。",
    ),
    ModelSpec(
        key="yolo26x",
        filename="yolo26x-pose.pt",
        url="",
        label="YOLO26X pose（实验档，不进入默认路由）",
        category="yolo",
        profile="yolo26x",
        downloadable=False,
        installed_supported=False,
        default_route_eligible=False,
        note="实验档，不进入默认路由。",
    ),
)

MODEL_SPECS: tuple[ModelSpec, ...] = MEDIAPIPE_MODELS + YOLO_MODELS


def model_path(spec: ModelSpec) -> Path:
    """返回模型在本地 ``models/`` 目录下的目标路径。"""
    return models_dir() / spec.filename


def is_installed(spec: ModelSpec) -> bool:
    """模型是否已安装（文件存在且非空）。"""
    p = model_path(spec)
    return p.exists() and p.stat().st_size > 0


def installed_size_mb(spec: ModelSpec) -> float | None:
    """返回已安装模型的大小（MB）；未安装返回 None。"""
    p = model_path(spec)
    if p.exists() and p.stat().st_size > 0:
        return round(p.stat().st_size / (1024 * 1024), 1)
    return None


def download_model(
    spec: ModelSpec,
    *,
    progress_cb: ProgressCb | None = None,
    should_stop: Callable[[], bool] | None = None,
    chunk_size: int = 64 * 1024,
    timeout: float = 30.0,
    proxy: str | None = None,
) -> Path:
    """下载单个模型到 ``models/``。

    - 先写入同目录下的 ``<filename>.part`` 临时文件，完成后原子改名为正式文件名，
      避免中断产生的半成品被 ``is_installed`` 误判为已安装。
    - ``progress_cb(downloaded, total)``：total 可能为 None（服务器未返回 Content-Length）。
    - ``should_stop()`` 返回 True 时中断下载并删除 ``.part``，抛出 ``InterruptedError``。

    返回正式文件路径。任何网络异常向上抛出，由调用方（UI）展示并提示用户自行处理网络问题。
    """
    dest = model_path(spec)
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not bool(getattr(spec, "downloadable", True)):
        raise RuntimeError(
            f"{spec.key} 暂不支持自动下载；{getattr(spec, 'note', '') or '请按文档手动安装模型。'}"
        )
    req = urllib.request.Request(spec.url, headers={"User-Agent": "vision-ui/1.0"})
    opener = _url_opener(proxy=proxy)
    try:
        with opener.open(req, timeout=timeout) as resp:  # nosec - official model asset
            total: int | None
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl and cl.isdigit() else None
            downloaded = 0
            if progress_cb:
                progress_cb(0, total)
            with open(part, "wb") as fh:
                while True:
                    if should_stop is not None and should_stop():
                        raise InterruptedError("下载已取消")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)
            if total is not None and downloaded != total:
                raise OSError(f"下载不完整：期望 {total} 字节，实际 {downloaded} 字节")
    except BaseException:
        # 清理半成品。
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass
        raise

    # 原子替换为正式文件。
    os.replace(part, dest)
    return dest


def model_download_proxy() -> str:
    value = os.environ.get("VISION_MODEL_PROXY")
    if value is not None:
        return value.strip()
    return DEFAULT_MODEL_PROXY


def _url_opener(*, proxy: str | None) -> urllib.request.OpenerDirector:
    proxy_value = model_download_proxy() if proxy is None else proxy
    if proxy_value:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_value, "https": proxy_value})
        )
    return urllib.request.build_opener()
