"""HTML 图文报告差分基准的用例：P8 模板化前用旧 ``render_html_report`` 录下输出，之后逐字节比对。

时间字段只用空串或无法解析的字符串，避免输出随本机时区变化；内嵌证据帧换成内容摘要，
基准文件不随图片体积膨胀，但证据帧字节有任何变化都会被发现。
"""
from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from tests.test_feedback_html_export import make_html_test_record
from tests.test_feedback_report import make_dummy_video, make_synthetic_check, make_synthetic_record

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "report_html"
RECORD_ID = "0123456789abcdef0123456789abcdef"
_DATA_URI = re.compile(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+")


def _untimed(record: dict, created: str = "") -> dict:
    record = copy.deepcopy(record)
    record["createdAt"] = created
    for entry in record.get("result", {}).get("reviewHistory", []) or []:
        entry["at"] = ""
    return record


def _with_video(record: dict, root: Path, frames: int = 12) -> dict:
    record = copy.deepcopy(record)
    record["id"] = RECORD_ID
    folder = root / RECORD_ID
    folder.mkdir(parents=True, exist_ok=True)
    make_dummy_video(folder / "front.avi", width=320, height=240, frames=frames)
    make_dummy_video(folder / "side.avi", width=240, height=320, frames=frames)
    record["videos"] = {"front": "front.avi", "side": "side.avi"}
    return record


def _evidence(view: str, frame: int) -> dict:
    lms = [[0.4 + 0.01 * (i % 7), 0.2 + 0.02 * (i % 11), 0.0, 1.0] for i in range(33)]
    return {
        "id": f"{view}:{frame}",
        "view": view,
        "frame": frame,
        "timeSeconds": frame / 10.0,
        "landmarks": lms,
        "validMask": [i % 5 != 0 for i in range(33)],
    }


def build_cases(root: Path) -> dict[str, tuple[dict, dict]]:
    """用例名 → (记录, ``render_html_report`` 的关键字参数)。``root`` 是可写的临时历史目录。"""
    cases: dict[str, tuple[dict, dict]] = {}
    cases["synthetic_default"] = (_untimed(make_synthetic_record()), {})
    cases["synthetic_bogus_time"] = (_untimed(make_synthetic_record(), "not-a-date"), {})
    cases["html_record"] = (_untimed(make_html_test_record()), {})
    cases["html_record_rerecord"] = (_untimed(make_html_test_record(needs_rerecord=True, student_name="")), {})

    reviewed = make_synthetic_record(checks=[
        make_synthetic_check(code="stance_elbow", phase="start", review="confirmed"),
        make_synthetic_check(code="guard_low", segment="front_straight", phase="motion", review="revoked"),
        make_synthetic_check(code="guard_back", segment="rear_straight", phase="motion", status="unable"),
        make_synthetic_check(code="hip_turn", segment="rear_straight", phase="finish"),
    ])
    reviewed["result"]["reviewHistory"] = [
        {"checkId": "stance:start:front:stance_elbow", "teacher": "王老师", "reason": "<b>确认</b>", "after": "confirmed"},
        {"checkId": "front_straight:motion:front:guard_low", "teacher": "李老师", "reason": "误报", "after": "revoked"},
    ]
    cases["reviewed"] = (_untimed(reviewed), {})

    legacy = make_synthetic_record(action="hook_combo", stance="right")
    legacy["result"]["checks"] = []
    legacy["result"].pop("diagnostics")
    legacy["result"].pop("capture")
    legacy["result"].pop("alignment")
    cases["legacy_no_checks"] = (_untimed(legacy), {})
    stance_legacy = make_synthetic_record(action="stance")
    stance_legacy["result"] = {"checks": "broken"}
    cases["stance_broken_result"] = (_untimed(stance_legacy), {})

    clean = make_synthetic_record(checks=[
        make_synthetic_check(code="stance_elbow", phase="start", status="not_observed"),
        make_synthetic_check(code="stance_knee", phase="end", status="not_observed"),
    ], action="stance")
    cases["all_clear"] = (_untimed(clean), {})

    xss = make_synthetic_record(student_id="<script>alert(1)</script>", student_name='"><svg onload=1>')
    xss["result"]["checks"][0]["name"] = r"C:\Users\x\secret.mp4 </title><script>"
    xss["result"]["checks"][0]["reason"] = "/home/user/clip.mp4 & 'quote'"
    xss["result"]["diagnostics"]["front"]["reason"] = r"\\server\share\a.mp4"
    cases["xss_paths"] = (_untimed(xss), {})

    many = [
        make_synthetic_check(code=code, segment="front_straight", phase="motion", role=view,
                             evidence=[f"{view}:{frame}", f"side:{frame + 1}"])
        for frame, (code, view) in enumerate([
            ("guard_low", "front"), ("guard_back", "front"), ("guard_elbow", "front"),
            ("shoulder_level", "front"), ("hip_turn", "front"), ("arm_swing", "front"),
            ("unknown_rule", "front"),
        ])
    ]
    evidence = [_evidence(v, f) for f in range(9) for v in ("front", "side")]
    images = _with_video(make_synthetic_record(checks=many, evidence=evidence), root)
    cases["evidence_images"] = (_untimed(images), {"history_root": root})
    cases["evidence_images_capped"] = (_untimed(images), {"history_root": root, "max_images": 3, "max_image_edge": 160})
    cases["evidence_no_root"] = (_untimed(images), {})
    missing = copy.deepcopy(images)
    missing["videos"] = {"front": "absent.avi"}
    cases["evidence_missing_video"] = (_untimed(missing), {"history_root": root})
    past_end = copy.deepcopy(images)
    past_end["result"]["evidence"] = [_evidence("front", 500), _evidence("side", 1)]
    past_end["result"]["checks"] = [make_synthetic_check(evidence=["front:500", "side:1", "front:404"])]
    cases["evidence_decode_failure"] = (_untimed(past_end), {"history_root": root})
    return cases


def normalize(html_text: str) -> str:
    """把内嵌 JPEG 换成摘要，便于基准文件入库与阅读。"""
    return _DATA_URI.sub(lambda m: "data:image/jpeg;sha256," + hashlib.sha256(m.group(0).encode("ascii")).hexdigest(), html_text)


def record_fixtures(render, root: Path) -> list[Path]:
    """用给定的渲染函数写出全部基准文件（只在模板化前运行一次）。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (record, kwargs) in build_cases(root).items():
        path = FIXTURE_DIR / f"{name}.html"
        path.write_bytes(normalize(render(record, **kwargs)).encode("utf-8"))
        written.append(path)
    return written
