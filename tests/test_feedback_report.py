"""Unit tests for core.feedback_report and HTML export integration."""
from datetime import datetime, timezone
import html
from pathlib import Path
import re
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
import pytest

from core.action_feedback import ACTIONS, STANCES, build_checks
from core.feedback_history import FeedbackHistory
from core.feedback_report import (
    StatusCounts,
    HeadlineSummary,
    NormalizedCheck,
    PhaseTimelineItem,
    GroupedReason,
    NormalizedReportModel,
    build_report_model,
    get_headline_summary,
    group_checks_by_reason,
    generate_phase_timeline_svg,
    extract_evidence_frame,
    frame_to_data_uri,
    render_html_report,
    format_local_datetime,
    resolve_record_video_path,
)


def make_synthetic_check(
    code: str = "stance_elbow",
    segment: str = "stance",
    phase: str = "start",
    role: str = "front",
    status: str = "candidate",
    review: str = "pending",
    reason: str = "前手大小臂夹角不足90°",
    evidence: list[str] | None = None
) -> dict:
    return {
        "id": f"{segment}:{phase}:{role}:{code}",
        "code": code,
        "name": f"测试问题-{code}",
        "bodyPart": "前侧臂肘（左）",
        "segment": segment,
        "segmentLabel": ACTIONS.get(segment, segment),
        "phase": phase,
        "phaseLabel": "开始实战式" if phase == "start" else "动作过程",
        "standard": "前手大小臂夹角90°～135°",
        "source": "武术散打得分点.docx",
        "status": status,
        "review": review,
        "reason": reason,
        "blockedReason": "",
        "evidence": evidence or ["front:0"],
        "measurements": [{
            "view": "front",
            "validFrames": 20,
            "totalFrames": 24,
            "violationRatio": 0.8,
            "values": [{"name": "肘角", "median": 82.5, "min": 78.0, "max": 88.0, "unit": "°"}]
        }],
        "fusion": {
            "method": "action_rule_quality_weighted_v1",
            "baseWeights": {"front": 0.5, "side": 0.5},
            "effectiveWeights": {"front": 0.5, "side": 0.5},
            "support": 0.8,
            "threshold": 0.5,
            "calibrationStatus": "unvalidated"
        }
    }


def make_synthetic_record(
    student_id: str = "001",
    student_name: str = "张同学",
    action: str = "straight_combo",
    stance: str = "left",
    checks: list[dict] | None = None,
    needs_rerecord: bool = False,
    evidence: list[dict] | None = None,
    videos: dict[str, str] | None = None
) -> dict:
    if checks is None:
        checks = [
            make_synthetic_check(code="stance_elbow", phase="start", status="candidate", review="pending"),
            make_synthetic_check(code="guard_low", segment="front_straight", phase="motion", status="not_observed"),
            make_synthetic_check(code="guard_back", segment="rear_straight", phase="motion", status="unable", reason="侧面视角被遮挡"),
            make_synthetic_check(code="vertical_fist", segment="front_straight", phase="finish", status="pending_rule", reason="需教师人工查看立拳"),
        ]

    if evidence is None:
        # 33 MediaPipe landmarks
        lms = [[0.5, 0.5, 0.0, 1.0] for _ in range(33)]
        # Add distinct position for wrist/elbow
        lms[11] = [0.4, 0.3, 0.0, 1.0]
        lms[13] = [0.35, 0.45, 0.0, 1.0]
        lms[15] = [0.3, 0.4, 0.0, 1.0]
        evidence = [{
            "id": "front:0",
            "view": "front",
            "frame": 0,
            "timeSeconds": 0.125,
            "landmarks": lms,
            "validMask": [True] * 33
        }]

    return {
        "schemaVersion": 1,
        "id": "a" * 32,
        "studentId": student_id,
        "studentName": student_name,
        "action": action,
        "stance": stance,
        "createdAt": "2026-09-26T12:00:00Z",
        "expiresAt": "2026-10-10T12:00:00Z",
        "revision": 1,
        "status": "ready",
        "sourcePaths": {
            "front": "D:/CodeProject/vision/outputs/front.mp4",
            "side": "D:/CodeProject/vision/outputs/side.mp4"
        },
        "videos": videos or {"front": "front.mp4", "side": "side.mp4"},
        "result": {
            "schemaVersion": 1,
            "ruleVersion": "sanda-feedback-mediapipe-2026-09-20-v4",
            "action": action,
            "stance": stance,
            "backend": "mediapipe",
            "summary": "测试动作问题说明总结",
            "needsRerecord": needs_rerecord,
            "checks": checks,
            "evidence": evidence,
            "phaseWindows": {
                "front": [
                    {"segment": "stance", "phase": "start", "ranges": [[0.0, 0.35]]},
                    {"segment": "front_straight", "phase": "motion", "ranges": [[0.35, 0.75]]},
                    {"segment": "front_straight", "phase": "finish", "ranges": [[0.70, 0.80]]},
                    {"segment": "rear_straight", "phase": "motion", "ranges": [[0.80, 1.20]]},
                    {"segment": "rear_straight", "phase": "finish", "ranges": [[1.15, 1.25]]},
                    {"segment": "stance", "phase": "end", "ranges": [[1.25, 1.60]]},
                ]
            },
            "diagnostics": {
                "front": {
                    "phaseSource": "own_view",
                    "bodyValidFrames": 45,
                    "totalFrames": 48,
                    "jointValidRatios": [1.0] * 33,
                    "reason": "正常"
                }
            },
            "capture": {
                "front": {
                    "width": 640,
                    "height": 480,
                    "sourceFps": 30.0,
                    "sourceFrames": 48,
                    "analyzedFrames": 48,
                    "inferenceWidth": 256,
                    "inferenceHeight": 256,
                    "analysisSeconds": 0.5
                }
            },
            "alignment": {
                "method": "action_peaks",
                "sideOffsetSeconds": 0.02,
                "reliable": True
            },
            "reviewHistory": []
        }
    }


def make_dummy_video(path: Path, width: int = 120, height: int = 120, frames: int = 5) -> Path:
    """Create a minimal real OpenCV readable AVI/MP4 video file."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (i * 40 % 255, 100, 150)
        out.write(img)
    out.release()
    return path


# ---------------------------------------------------------------------------
# Test 1: Item Conservation Law and Review States
# ---------------------------------------------------------------------------

def test_status_counts_conservation_law():
    """Verify strict conservation invariant across all categories and teacher review transitions."""
    # 1. Initial synthetic record with 1 candidate, 1 not_observed, 1 unable, 1 pending_rule
    rec = make_synthetic_record()
    model = build_report_model(rec)
    c = model.counts

    assert c.total_checks == 4
    assert c.candidate_count == 1
    assert c.unconfirmed_count == 1
    assert c.confirmed_count == 0
    assert c.revoked_count == 0
    assert c.not_observed_count == 1
    assert c.unable_count == 1
    assert c.pending_rule_count == 1
    assert c.verify_conservation()

    # 2. Teacher confirms the candidate
    rec["result"]["checks"][0]["review"] = "confirmed"
    model_conf = build_report_model(rec)
    c_conf = model_conf.counts
    assert c_conf.total_checks == 4
    assert c_conf.candidate_count == 1
    assert c_conf.confirmed_count == 1
    assert c_conf.unconfirmed_count == 0
    assert c_conf.revoked_count == 0
    assert c_conf.not_observed_count == 1
    assert c_conf.verify_conservation()

    # 3. Teacher revokes the candidate
    rec["result"]["checks"][0]["review"] = "revoked"
    model_rev = build_report_model(rec)
    c_rev = model_rev.counts
    assert c_rev.total_checks == 4
    assert c_rev.candidate_count == 0
    assert c_rev.unconfirmed_count == 0
    assert c_rev.confirmed_count == 0
    assert c_rev.revoked_count == 1
    # Revoked NEVER increases not_observed!
    assert c_rev.not_observed_count == 1
    assert c_rev.verify_conservation()
    assert len(model_rev.candidate_checks) == 0
    assert len(model_rev.revoked_checks) == 1


# ---------------------------------------------------------------------------
# Test 2: 7-Level Priority Headline Hierarchy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "checks,needs_rerecord,expected_title,expected_level",
    [
        # Level 1: All unable
        (
            [make_synthetic_check(code="c1", status="unable"), make_synthetic_check(code="c2", status="unable")],
            True,
            "本次未能完成有效判断",
            "error"
        ),
        # Level 2: Zero candidates, zero not_observed, only pending_rule
        (
            [make_synthetic_check(code="c1", status="pending_rule"), make_synthetic_check(code="c2", status="pending_rule")],
            False,
            "本次需教师判断",
            "info"
        ),
        # Level 3: Zero candidates, pending_rule > 0, not_observed > 0
        (
            [make_synthetic_check(code="c1", status="not_observed"), make_synthetic_check(code="c2", status="pending_rule")],
            False,
            "已检查项目未发现问题，仍有待判断项目",
            "info"
        ),
        # Level 4: Zero candidates, unable > 0, not_observed > 0
        (
            [make_synthetic_check(code="c1", status="not_observed"), make_synthetic_check(code="c2", status="unable")],
            False,
            "分析完成，部分项目暂无法判断",
            "warning"
        ),
        # Level 5: Candidate > 0, unable > 0
        (
            [make_synthetic_check(code="c1", status="candidate"), make_synthetic_check(code="c2", status="unable")],
            False,
            "分析完成，部分项目暂无法判断",
            "warning"
        ),
        # Level 6: Candidate > 0, unable == 0
        (
            [make_synthetic_check(code="c1", status="candidate"), make_synthetic_check(code="c2", status="not_observed")],
            False,
            "发现疑似动作问题，请核对依据",
            "warning"
        ),
        # Level 7: Candidate == 0, unable == 0, pending == 0
        (
            [make_synthetic_check(code="c1", status="not_observed"), make_synthetic_check(code="c2", status="not_observed")],
            False,
            "已检查项目未发现问题",
            "success"
        ),
    ]
)
def test_headline_priority_hierarchy(checks, needs_rerecord, expected_title, expected_level):
    rec = make_synthetic_record(checks=checks, needs_rerecord=needs_rerecord)
    model = build_report_model(rec)
    assert model.headline.title == expected_title
    assert model.headline.level == expected_level
    if needs_rerecord and expected_title != "本次未能完成有效判断":
        assert "请重新录制" in model.headline.subtitle


# ---------------------------------------------------------------------------
# Test 3: Prohibit Grading Artifacts
# ---------------------------------------------------------------------------

def test_prohibit_grading_artifacts():
    """Ensure no scores, penalty points, or pass rates exist in models or HTML report."""
    rec = make_synthetic_record()
    model = build_report_model(rec)
    html_text = render_html_report(rec)

    # Prohibit score patterns and penalty point displays
    assert not re.search(r"总分[：:\s]*\d+", html_text)
    assert not re.search(r"得分[：:\s]*\d+", html_text)
    assert not re.search(r"扣分[：:\s]*\d+", html_text)
    assert not re.search(r"扣\s*\d+\s*分", html_text)
    assert not re.search(r"合格率[：:\s]*\d+%", html_text)
    assert not re.search(r"准确率[：:\s]*\d+%", html_text)
    assert "<progress" not in html_text
    assert "评分进度条" not in html_text
    assert "动作得分" not in html_text
    for kw in ["得分", "扣分", "扣 2 分", "合格率", "准确率", "评分进度条", "动作得分"]:
        assert kw not in model.headline.title
        assert kw not in model.headline.subtitle


# ---------------------------------------------------------------------------
# Test 4: Same-Reason Grouping
# ---------------------------------------------------------------------------

def test_same_reason_grouping():
    """Group checks with identical reason within same phase/view into collapsible clusters."""
    checks = [
        make_synthetic_check(code="c1", phase="start", reason="缺少侧面录像"),
        make_synthetic_check(code="c2", phase="start", reason="缺少侧面录像"),
        make_synthetic_check(code="c3", phase="start", reason="缺少侧面录像"),
        make_synthetic_check(code="c4", phase="motion", reason="出拳未与肩平齐"),
    ]
    rec = make_synthetic_record(checks=checks)
    model = build_report_model(rec)

    groups = model.candidate_groups
    assert len(groups) == 2
    assert groups[0].count == 3
    assert groups[0].reason == "缺少侧面录像"
    assert len(groups[0].checks) == 3
    assert groups[1].count == 1
    assert groups[1].reason == "出拳未与肩平齐"


# ---------------------------------------------------------------------------
# Test 5: Legacy Record Compatibility
# ---------------------------------------------------------------------------

def test_legacy_record_compatibility():
    """Ensure legacy or degraded records missing fields normalize gracefully without error."""
    legacy_rec = {
        "id": "b" * 32,
        "studentId": "002",
        "studentName": "",
        "action": "stance",
        "stance": "right",
        "createdAt": "2026-09-20T10:00:00",  # Naive UTC string
        "result": {
            "backend": "unknown_legacy",
            "needsRerecord": False
        }
    }
    model = build_report_model(legacy_rec)
    assert model.header.student_id == "002"
    assert model.header.student_name == ""
    assert "旧版规则检查记录" in model.header.backend_description
    assert model.counts.total_checks > 0
    assert model.counts.verify_conservation()
    assert all(c.status == "unable" for c in model.all_checks)
    assert len(model.phases) == 2
    assert model.phases[0].title == "开始实战式"
    assert model.phases[1].title == "结束实战式"

    html_output = render_html_report(legacy_rec)
    assert "散打学生练习分析图文报告" in html_output
    assert "002" in html_output


# ---------------------------------------------------------------------------
# Test 6: Diagram-Design Compliant SVG Timeline
# ---------------------------------------------------------------------------

def test_phase_timeline_svg_conformance():
    """Verify inline SVG timeline complies strictly with diagram-design Process/Timeline rules."""
    rec = make_synthetic_record()
    model = build_report_model(rec)
    svg = generate_phase_timeline_svg(model.phases, model.counts)

    # 1. Structural requirements
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")
    assert 'role="img"' in svg
    assert 'aria-labelledby="flow-title flow-desc"' in svg
    assert '<title id="flow-title">' in svg
    assert '<desc id="flow-desc">' in svg

    # 2. 2px stroke, round caps/joins
    assert 'stroke-width="2"' in svg
    assert 'stroke-linecap="round"' in svg
    assert 'stroke-linejoin="round"' in svg

    # 3. No external fonts or URLs
    assert "http://" not in svg
    assert "https://" not in svg
    assert "fonts.googleapis.com" not in svg

    # 4. Correct stage labels rendered
    for p in model.phases:
        assert html.escape(p.title) in svg


# ---------------------------------------------------------------------------
# Test 7: Safe Evidence Frame Extraction
# ---------------------------------------------------------------------------

def test_safe_evidence_frame_extraction(tmp_path):
    """Verify OpenCV frame reading, height-normalized landmark coordinates, and boundary safety."""
    video_file = make_dummy_video(tmp_path / "test_video.avi", width=160, height=120, frames=10)

    landmarks = [[0.5, 0.4, 0.0, 1.0] for _ in range(33)]
    landmarks[13] = [0.3, 0.6, 0.0, 1.0]  # Elbow
    vmask = [True] * 33
    vmask[0] = False  # Nose invalid

    frame = extract_evidence_frame(
        video_file,
        frame_idx=2,
        landmarks=landmarks,
        valid_mask=vmask,
        highlight_joints={11, 13, 15}
    )
    assert frame is not None
    assert frame.shape == (120, 160, 3)

    # Verify height-normalized scaling calculation:
    # h = 120, point[13] = [0.3, 0.6] -> px = round(0.3 * 120) = 36, py = round(0.6 * 120) = 72
    assert frame[72, 36].tolist() != [0, 0, 0]

    missing = extract_evidence_frame(tmp_path / "non_existent.mp4", 0)
    assert missing is None

    symlink_path = tmp_path / "link.avi"
    try:
        symlink_path.symlink_to(video_file)
        assert extract_evidence_frame(symlink_path, 0) is None
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Test 8: Self-Contained Offline HTML Report & Image Embedding
# ---------------------------------------------------------------------------

def test_render_html_report_self_contained(tmp_path):
    """Verify HTML report is 100% offline, contains Base64 images, and embeds SVG."""
    record_id = "c" * 32
    history_root = tmp_path / "history"
    rec_dir = history_root / record_id
    rec_dir.mkdir(parents=True)
    video_path = rec_dir / "front.mp4"
    make_dummy_video(video_path, width=160, height=120, frames=5)

    rec = make_synthetic_record(videos={"front": "front.mp4", "side": "side.mp4"})
    rec["id"] = record_id

    html_text = render_html_report(rec, history_root=history_root, max_image_edge=160)

    # 1. Pure offline verification: zero external http/https requests
    assert "http://" not in html_text, "Found external http:// link in HTML report!"
    assert "https://" not in html_text, "Found external https:// link in HTML report!"
    assert "<script" not in html_text, "Found javascript tag in HTML report!"

    # 2. Base64 JPEG data URI embedded
    assert "data:image/jpeg;base64," in html_text

    # 3. Print style rules present
    assert "@media print" in html_text
    assert "A4 portrait" in html_text

    # 4. SVG icons and timeline present
    assert "<svg" in html_text
    assert "动作阶段时序流图" in html_text


# ---------------------------------------------------------------------------
# Test 9: HTML Escaping and Path Desensitization
# ---------------------------------------------------------------------------

def test_html_escaping_and_path_desensitization():
    """Verify strict HTML escaping against XSS and stripping of local machine paths."""
    malicious_name = '<script>alert("xss")</script>'
    malicious_reason = '"><img src=x onerror=alert(1)>'
    secret_path = "D:\\CodeProject\\vision\\internal\\secret.mp4"

    rec = make_synthetic_record(
        student_id="<img/onerror=1>",
        student_name=malicious_name,
        checks=[
            make_synthetic_check(code="stance_elbow", reason=malicious_reason)
        ]
    )
    rec["result"]["diagnostics"]["front"]["reason"] = f"文件位于 {secret_path}"

    html_report = render_html_report(rec)

    # 1. No raw unescaped script tag
    assert "<script>" not in html_report
    assert "&lt;script&gt;" in html_report
    assert "&lt;img" in html_report

    # 2. Absolute machine path desensitized / not in rendered document
    assert "D:\\CodeProject\\vision\\internal" not in html_report
    assert "D:/CodeProject/vision/internal" not in html_report
    assert "[已脱敏路径]" in html_report


# ---------------------------------------------------------------------------
# Test 10: FeedbackHistory Export Integration
# ---------------------------------------------------------------------------

def test_feedback_history_export_html(tmp_path):
    """Test FeedbackHistory.export() supporting .html alongside .txt with atomic safety."""
    history_root = tmp_path / "history"
    store = FeedbackHistory(history_root)

    source_path = tmp_path / "source.mp4"
    make_dummy_video(source_path, width=80, height=80, frames=4)

    def dummy_analyzer(*args, **kwargs):
        res = make_synthetic_record()["result"]
        return res

    identity = {"studentId": "001", "studentName": "张同学", "action": "straight_combo", "stance": "left"}
    saved = store.add(identity, source_path, None, analyzer=dummy_analyzer)
    record_id = saved["id"]

    # 1. Export as .html
    out_html = tmp_path / "export_report.html"
    res_path = store.export(record_id, out_html)
    assert res_path == out_html
    assert out_html.is_file()
    html_content = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "散打学生练习分析图文报告" in html_content
    assert "张同学" in html_content

    # 2. Export as .txt still works
    out_txt = tmp_path / "export_report.txt"
    res_txt = store.export(record_id, out_txt)
    assert res_txt == out_txt
    assert "散打动作问题说明" in out_txt.read_text(encoding="utf-8-sig")

    # 3. Invalid extension raises ValueError
    with pytest.raises(ValueError, match="请选择 .html 图文报告或 .txt 文字报告文件"):
        store.export(record_id, tmp_path / "report.pdf")

    # 4. Confinement: cannot export inside history root
    with pytest.raises(ValueError, match="受管历史目录之外"):
        store.export(record_id, history_root / "leak.html")
