"""学生练习历史；只管理本模块导入的视频副本，永不删除外部原视频。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import threading
from uuid import uuid4

from core.action_feedback import ACTIONS, STANCES, analyze_feedback, format_report

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".m4v"}


def resolve_segment_videos(directory: Path) -> tuple[Path | None, Path | None]:
    directory = Path(directory)
    return tuple(next((p for p in sorted(directory.glob(view + ".*"), key=lambda p: (p.suffix.lower() != ".mp4", p.name))
                       if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), None)
                 for view in ("front", "side"))


def validate_identity(student_id: str, student_name: str, action: str, stance: str) -> dict:
    if not isinstance(student_id, str) or not re.fullmatch(r"[\w.-]{1,64}", student_id.strip()):
        raise ValueError("请填写学号（1～64位字母、数字、汉字、点、下划线或短横线）")
    if not isinstance(student_name, str) or len(student_name.strip()) > 80 or any(ord(c) < 32 for c in student_name):
        raise ValueError("姓名不超过80字，不能包含换行或控制字符")
    if not isinstance(action, str) or not isinstance(stance, str) or action not in ACTIONS or stance not in STANCES:
        raise ValueError("请选择动作和左式/右式")
    return {"studentId": student_id.strip(), "studentName": student_name.strip(), "action": action, "stance": stance}


class FeedbackHistory:
    def __init__(self, root: Path | None = None, *, clock=None):
        if root is None:
            from core.paths import outputs_dir
            root = outputs_dir() / "practice_feedback"
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        # ponytail: 一个桌面进程共享此锁；多进程同时使用时改为SQLite事务。
        self._lock = threading.RLock()

    def _directory(self, record_id: str) -> Path:
        if not isinstance(record_id, str) or not re.fullmatch(r"[0-9a-f]{32}", record_id):
            raise ValueError("无效历史记录编号")
        path = self.root / record_id
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ValueError("历史记录路径越界")
        return path

    def _read(self, record_id: str) -> dict:
        try:
            record = json.loads((self._directory(record_id) / "record.json").read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("id") != record_id or record.get("schemaVersion") != 1:
                raise ValueError("bad record header")
            validate_identity(record["studentId"], record["studentName"], record["action"], record["stance"])
            if datetime.fromisoformat(record["expiresAt"]).tzinfo is None:
                raise ValueError("missing timezone")
            if not isinstance(record["videos"], dict) or not isinstance(record["sourcePaths"], dict):
                raise ValueError("bad videos")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"历史记录格式损坏：{record_id}") from exc
        return record

    def _write(self, record: dict) -> None:
        directory = self._directory(record["id"])
        temp = directory / "record.tmp"
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "record.json")

    def cleanup(self) -> None:
        """到期时仅清理带合法本模块清单的受管目录；损坏记录显式报错。"""
        with self._lock:
            for path in self.root.iterdir():
                if not re.fullmatch(r"[0-9a-f]{32}", path.name) or not (path / "record.json").is_file():
                    continue
                record = self._read(path.name)
                if datetime.fromisoformat(record["expiresAt"]) <= self.clock():
                    shutil.rmtree(self._directory(record["id"]))

    def list(self, student_id: str) -> list[dict]:
        with self._lock:
            self.cleanup()
            rows = [self._read(path.parent.name) for path in self.root.glob("*/record.json")
                    if re.fullmatch(r"[0-9a-f]{32}", path.parent.name)]
            return sorted((r for r in rows if r["studentId"] == student_id and r.get("status") == "ready"), key=lambda r: r["createdAt"], reverse=True)

    def get(self, record_id: str) -> dict:
        with self._lock:
            record = self._read(record_id)
            if datetime.fromisoformat(record["expiresAt"]) <= self.clock():
                self.cleanup()
                raise ValueError("记录已超过两周保留期")
            if record.get("status") != "ready":
                raise ValueError("该记录的分析尚未完成")
            return record

    def add(self, identity: dict, front: Path | None, side: Path | None, *,
            stopped=lambda: False, analyzer=analyze_feedback) -> dict:
        identity = validate_identity(identity["studentId"], identity["studentName"], identity["action"], identity["stance"])
        sources = {k: Path(v).resolve() for k, v in (("front", front), ("side", side)) if v is not None}
        if not sources or any(not p.is_file() or p.suffix.lower() not in VIDEO_SUFFIXES for p in sources.values()):
            raise ValueError("请选择仍然存在的正面或侧面视频")
        if len(sources) == 2 and sources["front"].samefile(sources["side"]):
            raise ValueError("正侧面不能使用同一个视频，请选择同次录制的两路视频")
        self.cleanup()
        created = self.clock()
        record_id = uuid4().hex
        directory = self._directory(record_id)
        directory.mkdir()
        try:
            record = {"schemaVersion": 1, "id": record_id, **identity, "createdAt": created.isoformat(),
                      "expiresAt": (created + timedelta(days=14)).isoformat(), "revision": 1,
                      "sourcePaths": {k: str(v) for k, v in sources.items()}, "videos": {},
                      "result": None, "previousResults": [], "status": "analyzing"}
            # 先记保留期限，进程意外退出留下的副本也能在到期后被清理。
            with self._lock:
                self._write(record)
            videos = {}
            for view, source in sources.items():
                if stopped():
                    raise InterruptedError("分析已取消")
                target = directory / (view + source.suffix.lower())
                shutil.copyfile(source, target)
                videos[view] = target.name
            result = analyzer(identity["action"], identity["stance"],
                              directory / videos["front"] if "front" in videos else None,
                              directory / videos["side"] if "side" in videos else None, stopped=stopped)
            if stopped():
                raise InterruptedError("分析已取消")
            record.update(videos=videos, result=result, status="ready")
            with self._lock:
                self._write(record)
            return record
        except BaseException:
            shutil.rmtree(self._directory(record_id))
            raise

    def reanalyze(self, record_id: str, *, stopped=lambda: False, analyzer=analyze_feedback) -> dict:
        record = self.get(record_id)
        directory = self._directory(record_id)
        videos = {}
        for view, name in record["videos"].items():
            path = directory / name
            if path.resolve().parent != directory or path.is_symlink():
                raise ValueError("历史视频路径越界")
            if path.is_file():
                videos[view] = path
        if not videos:
            raise ValueError("历史原视频已不存在，无法补充分析；请重新录制")
        result = analyzer(record["action"], record["stance"], videos.get("front"), videos.get("side"), stopped=stopped)
        with self._lock:
            current = self.get(record_id)
            if stopped():
                raise InterruptedError("分析已取消")
            if current["revision"] != record["revision"]:
                raise ValueError("分析期间记录已被复核，请重新打开记录后再试")
            current["previousResults"].append({"replacedAt": self.clock().isoformat(), "result": current["result"]})
            current["result"] = result
            current["revision"] += 1
            self._write(current)
            return current

    def review(self, record_id: str, check_id: str, decision: str, teacher: str, reason: str) -> dict:
        teacher, reason = teacher.strip(), reason.strip()
        if decision not in {"confirmed", "revoked"} or not teacher or len(teacher) > 80 or len(reason) > 300:
            raise ValueError("请选择确认/撤销，填写教师姓名（最多80字），原因最多300字")
        if any(ord(c) < 32 for c in teacher + reason):
            raise ValueError("复核姓名和原因不能包含控制字符")
        with self._lock:
            record = self.get(record_id)
            check = next((c for c in record["result"]["checks"] if c["id"] == check_id), None)
            if check is None or check["status"] != "candidate":
                raise ValueError("只能确认或撤销已识别的候选问题")
            record["result"]["reviewHistory"].append({"checkId": check_id, "before": check["review"],
                "after": decision, "teacher": teacher, "reason": reason, "at": self.clock().isoformat()})
            check["review"] = decision
            record["revision"] += 1
            self._write(record)
            return record

    def export(self, record_id: str, destination: Path) -> Path:
        with self._lock:
            record = self.get(record_id)
            destination = Path(destination).resolve()
            ext = destination.suffix.lower()
            if ext not in {".txt", ".html", ".htm"}:
                raise ValueError("请选择 .html 图文报告或 .txt 文字报告文件")
            if destination == self.root or self.root in destination.parents:
                raise ValueError("请将导出报告保存在受管历史目录之外")
            if str(destination) in record["sourcePaths"].values():
                raise ValueError("不能覆盖原视频")
            temp = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
            try:
                if ext == ".txt":
                    temp.write_text(format_report(record), encoding="utf-8-sig")
                else:
                    from core.feedback_report import render_html_report
                    temp.write_text(render_html_report(record, history_root=self.root), encoding="utf-8")
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
            return destination
