# -*- coding: utf-8 -*-
"""Qwen3-VL 粗粒度动作点评客户端（学生练习，prompt-only）。

边界
----
- 只输出肉眼可见的问题与中文修正建议，**不输出分数**。
- 不改变现有 DTW 模板比对、规则评分、tech_eval 或考试台账。
- 第一阶段仅直拳；检查项走白名单，禁止开放式任意发现。
- 依赖本机 ``llama-server``（默认 ``http://127.0.0.1:8091``），模型不在本仓库。

调用方（``apps/app_ui`` 学生练习模式）负责：双摄录制落盘 → 取 front/side 视频 →
本模块抽帧 + HTTP 推理 → 展示 ``CoachResult``。
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 直拳检查项契约（v1 冻结）
# ---------------------------------------------------------------------------

JAB_ISSUE_CODES: dict[str, str] = {
    "arm_not_extended": "出拳手臂未充分伸直",
    "guard_hand_low": "护手偏低",
    "torso_lean": "躯干过度倾斜",
    "stance_unstable": "站姿/重心不稳",
    "shoulder_hip_no_rotate": "发力转体不足",
    "punch_path_off": "出拳轨迹明显偏离",
}

# 受控建议模板：展示层优先用模板，不用模型自由发挥。
JAB_ISSUE_SUGGESTIONS: dict[str, str] = {
    "arm_not_extended": "出拳至最远端时尽量伸直肘关节，避免半屈肘击打。",
    "guard_hand_low": "非击打侧手保持在下颌高度附近，回收时同步抬回护手。",
    "torso_lean": "出拳时躯干保持中立或微转，避免明显前倾/后仰。",
    "stance_unstable": "双脚站距稳定、支撑脚踩实，出拳时避免重心乱晃。",
    "shoulder_hip_no_rotate": "出拳时肩髋协调转动发力，避免只靠手臂甩打。",
    "punch_path_off": "拳沿直线打出并直线回收，避免明显画弧或偏离目标线。",
}

SEVERITIES = ("轻微", "中等", "严重")
VIEW_HINTS = ("front", "side", "both", "unknown")
CONFIDENCES = ("low", "medium", "high")
_JAB_ACTIONS = frozenset({"jab", "straight", "直拳", "zhiquan"})
_UNABLE_ACTIONS = frozenset(
    {"unable_to_judge", "unable", "unknown", "n/a", "na", "cannot_judge"}
)

DEFAULT_BASE_URL = "http://127.0.0.1:8091"
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_FRAMES_PER_VIEW = 6
DEFAULT_MAX_LONG_EDGE = 640
DEFAULT_JPEG_QUALITY = 85
DEFAULT_TEXT_MAX_LEN = 80

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)

# 禁止出现在展示文案中的内容：分数、排名、面部/着装/背景等无关评价。
_BANNED_TEXT_RE = re.compile(
    r"("
    r"\d+\s*分"
    r"|得分|评分|打分|分数|百分|满分|及格|不及格|排名|名次"
    r"|score|percent|percentage|ranking"
    r"|面部|表情|微笑|服装|衣服|发型|妆容|背景音乐|长相|颜值"
    r")",
    re.IGNORECASE,
)
_SCORE_SYMBOL_RE = re.compile(r"[％%]|\b\d{1,3}\s*/\s*100\b")


@dataclass(frozen=True)
class CoachIssue:
    code: str
    problem: str
    severity: str
    suggestion: str
    view_hint: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoachResult:
    ok: bool
    action: str = "jab"
    summary: str = ""
    confidence: str = "low"
    issues: list[CoachIssue] = field(default_factory=list)
    raw_text: str = ""
    error_code: str | None = None
    error_message: str | None = None
    latency_s: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化供 coach.json / 调试；不含模型原文 raw_text（防泄漏分数与无关内容）。"""
        return {
            "ok": self.ok,
            "action": self.action,
            "summary": self.summary,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_s": self.latency_s,
            "warnings": list(self.warnings),
        }

    def format_display(self) -> str:
        """面向学生端的纯文本展示（无分数；文案已受控/消毒）。"""
        if not self.ok:
            code = self.error_code or "error"
            msg = self.error_message or "未知错误"
            return f"评判失败（{code}）\n{msg}"

        if self.action == "unable_to_judge":
            lines = [
                "动作：直拳",
                f"结论：无法评判（置信度：{_confidence_zh(self.confidence)}）",
                self.summary
                or "证据不足，请重试：保证全身入画、光线充足、机位稳定。",
            ]
            return "\n".join(lines)

        conf_zh = _confidence_zh(self.confidence)
        if not self.issues:
            lines = [
                "动作：直拳",
                f"结论：未发现白名单内明显问题（置信度：{conf_zh}）",
            ]
            if self.summary:
                lines.append(self.summary)
            lines.append("（练习建议，非正式成绩）")
            return "\n".join(lines)

        lines = [
            "动作：直拳",
            f"结论：发现 {len(self.issues)} 个可改进点（置信度：{conf_zh}）",
        ]
        if self.summary:
            lines.append(self.summary)
        for idx, issue in enumerate(self.issues, start=1):
            view = _view_zh(issue.view_hint)
            label = JAB_ISSUE_CODES.get(issue.code, issue.code)
            lines.append(f"{idx}. [{issue.severity}] {label}（{view}）")
            if issue.problem and issue.problem != label:
                lines.append(f"   现象：{issue.problem}")
            if issue.suggestion:
                lines.append(f"   建议：{issue.suggestion}")
        lines.append("（练习建议，非正式成绩）")
        if self.warnings:
            lines.append("提示：" + "；".join(self.warnings))
        return "\n".join(lines)


def sanitize_coach_text(
    text: str | None,
    *,
    max_len: int = DEFAULT_TEXT_MAX_LEN,
) -> str | None:
    """校验并截断展示文案；含禁用词/分数符号则丢弃（返回 None）。"""
    if text is None:
        return None
    value = " ".join(str(text).split()).strip()
    if not value:
        return None
    if _BANNED_TEXT_RE.search(value) or _SCORE_SYMBOL_RE.search(value):
        return None
    # 拒绝明显英文评分口吻
    lowered = value.lower()
    if any(tok in lowered for tok in ("score", "points", "grade", "rank")):
        return None
    if len(value) > max_len:
        value = value[: max_len - 1].rstrip() + "…"
    return value


# unable_to_judge 固定原因码 → 固定中文（绝不回显模型原值）
_UNABLE_REASON_TEXT: dict[str, str] = {
    "not_jab": "未能确认为直拳。请正对镜头重做直拳后重试。",
    "missing_action": "未能识别动作类型。请正对镜头重做直拳后重试。",
    "issues_all_dropped": "有效检查项为空，无法可靠点评。请调整机位后重试。",
    "evidence": "证据不足，请重试：保证全身入画、光线充足、机位稳定。",
}


def controlled_summary(
    *,
    action: str,
    issues: Sequence[CoachIssue],
    unable_reason: str | None = None,
) -> str:
    """仅由 action / issues / 固定 reason 生成总结，**不使用模型自由文本**。"""
    if action == "unable_to_judge":
        key = unable_reason if unable_reason in _UNABLE_REASON_TEXT else "evidence"
        return _UNABLE_REASON_TEXT[key]
    if not issues:
        return "未发现白名单内明显问题，请继续保持。"
    labels = "、".join(JAB_ISSUE_CODES.get(i.code, i.code) for i in issues)
    return f"主要关注：{labels}。"


def controlled_issue_texts(
    code: str,
    *,
    severity: str,
    view_hint: str,
) -> tuple[str, str]:
    """仅由 code + severity + view_hint 生成现象/建议，**不使用模型自由文本**。"""
    label = JAB_ISSUE_CODES.get(code, code)
    view = _view_zh(view_hint)
    sev = severity if severity in SEVERITIES else "轻微"
    problem = f"{label}（{view}，{sev}）"
    suggestion = JAB_ISSUE_SUGGESTIONS.get(code) or (
        f"请针对「{label}」对照标准动作慢速练习。"
    )
    return problem, suggestion


def _confidence_zh(value: str) -> str:
    return {"low": "低", "medium": "中", "high": "高"}.get(value, value)


def _view_zh(value: str) -> str:
    return {
        "front": "正面",
        "side": "侧面",
        "both": "正侧",
        "unknown": "视角不明",
    }.get(value, value)


def resolve_base_url(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    env = os.environ.get("QWEN_COACH_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    return DEFAULT_BASE_URL


def build_jab_system_prompt() -> str:
    codes = "\n".join(f"- {k}: {v}" for k, v in JAB_ISSUE_CODES.items())
    return (
        "你是武术散打直拳动作的视觉教练助手。只根据图像中的人体动作给出粗粒度点评。\n"
        "硬性规则：\n"
        "1. 只评直拳；若不是直拳、看不清人、遮挡严重、机位错误或帧数不足，"
        "必须返回 action=unable_to_judge，issues 为空数组。\n"
        "2. 问题 code 只能来自白名单，禁止发明新 code，禁止评价面部表情、服装、背景、"
        "发型、力量、精确角度或分数。\n"
        "3. 绝不输出任何分数、百分比或排名。\n"
        "4. 只输出一个 JSON 对象，不要 Markdown 围栏，不要其它解释文字。\n"
        "白名单 code：\n"
        f"{codes}\n"
        "JSON 结构：\n"
        "{\n"
        '  "action": "jab" | "unable_to_judge",\n'
        '  "summary": "一句中文总结",\n'
        '  "confidence": "low" | "medium" | "high",\n'
        '  "issues": [\n'
        "    {\n"
        '      "code": "<白名单>",\n'
        '      "problem": "现象描述",\n'
        '      "severity": "轻微" | "中等" | "严重",\n'
        '      "suggestion": "可执行修正建议",\n'
        '      "view_hint": "front" | "side" | "both" | "unknown"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "合格动作可返回 action=jab、issues=[] 与简短肯定 summary。"
    )


def build_jab_user_prompt(*, front_label: str = "正面", side_label: str = "侧面") -> str:
    return (
        f"以下图像为同一段直拳练习的{front_label}与{side_label}抽帧拼图（contact sheet）。"
        "请按系统规则输出 JSON。"
    )


# ---------------------------------------------------------------------------
# 抽帧
# ---------------------------------------------------------------------------


def _resize_long_edge(frame: np.ndarray, max_long_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge or long_edge <= 0:
        return frame
    scale = max_long_edge / float(long_edge)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def sample_video_frames(
    video_path: Path | str,
    *,
    num_frames: int = DEFAULT_FRAMES_PER_VIEW,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> list[np.ndarray]:
    """均匀抽取 ``num_frames`` 帧（BGR）。文件不可读或空则返回 []。"""
    path = Path(video_path)
    if not path.is_file():
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # 部分容器帧数不可靠：顺序读到上限
            frames: list[np.ndarray] = []
            while len(frames) < max(num_frames * 4, 24):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
            if not frames:
                return []
            idxs = _even_indices(len(frames), num_frames)
            return [_resize_long_edge(frames[i], max_long_edge) for i in idxs]

        idxs = _even_indices(total, num_frames)
        out: list[np.ndarray] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out.append(_resize_long_edge(frame, max_long_edge))
        return out
    finally:
        cap.release()


def _even_indices(total: int, n: int) -> list[int]:
    if total <= 0 or n <= 0:
        return []
    if n == 1:
        return [total // 2]
    if total <= n:
        return list(range(total))
    # 端点内均匀，略避开首尾纯站立帧
    start = max(0, total // 20)
    end = max(start + 1, total - max(1, total // 20))
    span = end - start
    if span <= n:
        return list(range(start, end))
    step = (span - 1) / float(n - 1)
    return [int(round(start + i * step)) for i in range(n)]


def make_contact_sheet(
    frames: Sequence[np.ndarray],
    *,
    cols: int | None = None,
    pad: int = 4,
    bg: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray | None:
    """将多帧拼成 contact sheet（BGR）。"""
    if not frames:
        return None
    n = len(frames)
    if cols is None:
        cols = min(3, n)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols
    th = max(f.shape[0] for f in frames)
    tw = max(f.shape[1] for f in frames)
    sheet_h = rows * th + (rows + 1) * pad
    sheet_w = cols * tw + (cols + 1) * pad
    sheet = np.full((sheet_h, sheet_w, 3), bg, dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        y0 = pad + r * (th + pad)
        x0 = pad + c * (tw + pad)
        h, w = frame.shape[:2]
        # 居中贴入格子
        y1 = y0 + (th - h) // 2
        x1 = x0 + (tw - w) // 2
        sheet[y1 : y1 + h, x1 : x1 + w] = frame
    return sheet


def encode_jpeg_bgr(
    image: np.ndarray, *, quality: int = DEFAULT_JPEG_QUALITY
) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("jpeg_encode_failed")
    return bytes(buf)


def sample_dual_contact_sheets(
    front_path: Path | str,
    side_path: Path | str,
    *,
    frames_per_view: int = DEFAULT_FRAMES_PER_VIEW,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    stack_vertical: bool = True,
) -> list[bytes]:
    """从正/侧视频抽帧，返回 1～2 张 JPEG contact sheet 的 bytes 列表。"""
    front_frames = sample_video_frames(
        front_path, num_frames=frames_per_view, max_long_edge=max_long_edge
    )
    side_frames = sample_video_frames(
        side_path, num_frames=frames_per_view, max_long_edge=max_long_edge
    )
    if not front_frames and not side_frames:
        return []

    sheets: list[np.ndarray] = []
    front_sheet = make_contact_sheet(front_frames) if front_frames else None
    side_sheet = make_contact_sheet(side_frames) if side_frames else None

    if front_sheet is not None and side_sheet is not None and stack_vertical:
        # 统一宽度后上下拼接，减少多图请求复杂度
        w = max(front_sheet.shape[1], side_sheet.shape[1])
        front_sheet = _pad_to_width(front_sheet, w)
        side_sheet = _pad_to_width(side_sheet, w)
        gap = np.full((8, w, 3), (40, 40, 40), dtype=np.uint8)
        stacked = np.vstack([front_sheet, gap, side_sheet])
        sheets.append(stacked)
    else:
        if front_sheet is not None:
            sheets.append(front_sheet)
        if side_sheet is not None:
            sheets.append(side_sheet)

    return [encode_jpeg_bgr(s, quality=jpeg_quality) for s in sheets]


def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= width:
        return img
    pad = width - w
    left = pad // 2
    right = pad - left
    return cv2.copyMakeBorder(
        img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )


def resolve_segment_videos(
    segment_dir: Path | str,
) -> tuple[Path | None, Path | None]:
    """在片段目录中解析 front/side 视频（优先 mp4，回退常见扩展名）。"""
    d = Path(segment_dir)
    if not d.is_dir():
        return None, None
    front = _first_existing(
        d / "front.mp4",
        d / "front.avi",
        d / "front.mkv",
        *sorted(d.glob("front.*")),
    )
    side = _first_existing(
        d / "side.mp4",
        d / "side.avi",
        d / "side.mkv",
        *sorted(d.glob("side.*")),
    )
    return front, side


def _first_existing(*candidates: Path) -> Path | None:
    seen: set[Path] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# JSON 解析与白名单过滤
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty_model_output")

    # 纯 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # ```json ... ```
    m = _JSON_FENCE_RE.search(raw)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return obj

    # 截取首尾花括号
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("bad_json")


def normalize_coach_payload(
    payload: dict[str, Any],
    *,
    raw_text: str = "",
    latency_s: float | None = None,
) -> CoachResult:
    """将模型 JSON 归一为 CoachResult：白名单 + fail-closed + **纯固定文案**。

    模型自由文本（summary/problem/suggestion/action 原值）一律不进入展示与 coach.json。
    仅信任结构化字段：action 枚举、issue.code、severity、view_hint、confidence。
    """
    warnings: list[str] = []
    # 保留原始 action 仅用于分类，绝不写入 summary / 展示
    action_raw = str(payload.get("action") or "").strip().lower()
    unable_reason: str | None = None
    if action_raw in _UNABLE_ACTIONS:
        action = "unable_to_judge"
        unable_reason = "evidence"
    elif action_raw in _JAB_ACTIONS:
        action = "jab"
    elif not action_raw:
        action = "unable_to_judge"
        unable_reason = "missing_action"
        warnings.append("unexpected_action:missing")
    else:
        # 未知 action 一律 fail-closed；warning 可记短哈希/长度，不回显原文
        action = "unable_to_judge"
        unable_reason = "not_jab"
        warnings.append(f"unexpected_action:len={len(action_raw)}")

    confidence = str(payload.get("confidence") or "low").strip().lower()
    if confidence not in CONFIDENCES:
        confidence = "low"
        warnings.append("confidence_clamped")

    issues_raw = payload.get("issues") or []
    if not isinstance(issues_raw, list):
        issues_raw = []
        warnings.append("issues_not_list")

    issues: list[CoachIssue] = []
    dropped = 0
    for item in issues_raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        code = str(item.get("code") or "").strip()
        if code not in JAB_ISSUE_CODES:
            dropped += 1
            continue
        severity = str(item.get("severity") or "轻微").strip()
        if severity not in SEVERITIES:
            severity = "轻微"
        view_hint = str(item.get("view_hint") or "unknown").strip().lower()
        if view_hint not in VIEW_HINTS:
            view_hint = "unknown"
        problem, suggestion = controlled_issue_texts(
            code, severity=severity, view_hint=view_hint
        )
        issues.append(
            CoachIssue(
                code=code,
                problem=problem,
                severity=severity,
                suggestion=suggestion,
                view_hint=view_hint,
            )
        )

    if dropped:
        warnings.append(f"dropped_issues:{dropped}")

    # 宣称 jab 但有效 issue 全被滤掉 → 不可靠，fail-closed
    if action == "jab" and not issues and dropped > 0:
        action = "unable_to_judge"
        unable_reason = "issues_all_dropped"
        warnings.append("jab_with_only_invalid_issues")

    if action == "unable_to_judge":
        issues = []

    summary = controlled_summary(
        action=action,
        issues=issues,
        unable_reason=unable_reason,
    )
    # 模型原文永不进入结果对象（展示 / coach.json / 调试字段一律清空）
    _ = raw_text  # 显式丢弃

    return CoachResult(
        ok=True,
        action=action,
        summary=summary,
        confidence=confidence,
        issues=issues,
        raw_text="",
        latency_s=latency_s,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


def _http_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw:
                return None
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"http_{e.code}:{detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"unreachable:{e.reason}") from e
    except TimeoutError as e:
        raise TimeoutError("timeout") from e


def health_check(
    base_url: str | None = None,
    *,
    timeout_s: float = 5.0,
) -> tuple[bool, str]:
    """返回 (ok, message)。"""
    url = resolve_base_url(base_url) + "/health"
    try:
        payload = _http_json("GET", url, timeout_s=timeout_s)
        if isinstance(payload, dict):
            status = str(payload.get("status") or payload.get("ok") or "").lower()
            if status in {"ok", "healthy", "true", "1"} or payload.get("ok") is True:
                return True, "ok"
            # llama-server 有时返回 {"status":"ok"} 或其它
            if not status and payload:
                return True, "ok"
            return False, f"unexpected_health:{payload}"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _image_content_parts(images: Sequence[bytes]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for blob in images:
        b64 = base64.b64encode(blob).decode("ascii")
        # OpenAI-compatible multimodal content
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    return parts


def chat_completions_vision(
    images: Sequence[bytes],
    *,
    system_prompt: str,
    user_prompt: str,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    model: str = "qwen3-vl",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    http_json: Callable[..., Any] | None = None,
) -> str:
    """调用 ``/v1/chat/completions``，返回 assistant 文本。"""
    if not images:
        raise ValueError("empty_media")
    post = http_json or _http_json
    url = resolve_base_url(base_url) + "/v1/chat/completions"
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    user_content.extend(_image_content_parts(images))
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    payload = post("POST", url, body=body, timeout_s=timeout_s)
    if not isinstance(payload, dict):
        raise RuntimeError("empty_response")
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("no_choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # 部分实现返回多段 content
        texts = [
            str(p.get("text") or "")
            for p in content
            if isinstance(p, dict) and p.get("type") in {None, "text"}
        ]
        return "\n".join(t for t in texts if t).strip()
    if content is None:
        # 兼容 /completion 风格
        text = choices[0].get("text")
        if text:
            return str(text).strip()
        raise RuntimeError("empty_content")
    return str(content).strip()


def analyze_jab_images(
    images: Sequence[bytes],
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    http_json: Callable[..., Any] | None = None,
) -> CoachResult:
    """对已编码的 contact sheet 图片做直拳点评。"""
    t0 = time.perf_counter()
    if not images:
        return CoachResult(
            ok=False,
            error_code="empty_media",
            error_message="没有可用的抽帧图像，请确认正/侧视频已落盘且可读。",
            latency_s=0.0,
        )
    try:
        text = chat_completions_vision(
            images,
            system_prompt=build_jab_system_prompt(),
            user_prompt=build_jab_user_prompt(),
            base_url=base_url,
            timeout_s=timeout_s,
            http_json=http_json,
        )
        payload = extract_json_object(text)
        result = normalize_coach_payload(
            payload, raw_text=text, latency_s=time.perf_counter() - t0
        )
        return result
    except ValueError as e:
        code = str(e) if str(e) in {"bad_json", "empty_model_output", "empty_media"} else "bad_json"
        return CoachResult(
            ok=False,
            error_code=code,
            error_message=f"模型输出无法解析：{e}",
            raw_text="",
            latency_s=time.perf_counter() - t0,
        )
    except TimeoutError:
        return CoachResult(
            ok=False,
            error_code="timeout",
            error_message="视觉模型响应超时，请稍后重试或缩短录制时长。",
            latency_s=time.perf_counter() - t0,
        )
    except ConnectionError as e:
        return CoachResult(
            ok=False,
            error_code="service_unavailable",
            error_message=(
                "无法连接本机 Qwen 服务（默认 http://127.0.0.1:8091）。"
                "请先运行 scripts/start_qwen_server.ps1 或手动启动 llama-server。"
                f" 详情：{e}"
            ),
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        msg = str(e)
        if msg.startswith("http_"):
            return CoachResult(
                ok=False,
                error_code="service_unavailable",
                error_message=f"视觉服务返回错误：{msg}",
                latency_s=time.perf_counter() - t0,
            )
        return CoachResult(
            ok=False,
            error_code="infer_failed",
            error_message=f"评判失败：{e}",
            latency_s=time.perf_counter() - t0,
        )


def analyze_jab_segment(
    segment_dir: Path | str | None = None,
    *,
    front_path: Path | str | None = None,
    side_path: Path | str | None = None,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    frames_per_view: int = DEFAULT_FRAMES_PER_VIEW,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    http_json: Callable[..., Any] | None = None,
    save_coach_json: bool = True,
) -> CoachResult:
    """对学生练习片段做直拳点评。

    优先使用显式 ``front_path`` / ``side_path``（后处理终态路径）；
    否则在 ``segment_dir`` 内解析 front/side 视频。
    成功时可选写入 ``coach.json``（受控字段，无 raw 原文、无分数）。
    """
    front: Path | None = Path(front_path) if front_path else None
    side: Path | None = Path(side_path) if side_path else None
    if front is not None and not front.is_file():
        front = None
    if side is not None and not side.is_file():
        side = None

    if front is None and side is None and segment_dir is not None:
        front, side = resolve_segment_videos(segment_dir)

    if front is None and side is None:
        where = segment_dir if segment_dir is not None else "(no segment_dir)"
        return CoachResult(
            ok=False,
            error_code="invalid_video",
            error_message=f"缺少可读的 front/side 视频：{where}",
        )
    if front is None or side is None:
        # 单路也可评，但提示缺视角
        missing = "正面" if front is None else "侧面"
        only = front or side
        assert only is not None
        frames = sample_video_frames(
            only, num_frames=frames_per_view, max_long_edge=max_long_edge
        )
        sheet = make_contact_sheet(frames)
        images = [encode_jpeg_bgr(sheet)] if sheet is not None else []
        result = analyze_jab_images(
            images,
            base_url=base_url,
            timeout_s=timeout_s,
            http_json=http_json,
        )
        result.warnings.append(f"missing_view:{missing}")
    else:
        images = sample_dual_contact_sheets(
            front,
            side,
            frames_per_view=frames_per_view,
            max_long_edge=max_long_edge,
        )
        result = analyze_jab_images(
            images,
            base_url=base_url,
            timeout_s=timeout_s,
            http_json=http_json,
        )

    if save_coach_json and result.ok and segment_dir is not None:
        try:
            out = Path(segment_dir) / "coach.json"
            out.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            result.warnings.append("coach_json_write_failed")
    return result
