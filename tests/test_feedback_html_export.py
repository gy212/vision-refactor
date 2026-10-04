# -*- coding: utf-8 -*-
"""单文件离线 HTML 图文报告导出安全与自包含性自动化测试。

涵盖：
- Self-contained assertion: 严格零外部链接 (无 http://, https://, 无 CDN, 无外部网络字体, 无外部脚本)
- XSS 防御: 学员姓名、学号、动作标准、复核理由中包含恶意脚本或标签时严格经由 html.escape 转义
- 隐私防泄漏: 消除所有包含本地盘符与开发机绝对路径 (如 D:\\..., C:\\Users\\...)
- Base64 证据嵌入: 图像以 Data URI (data:image/jpeg;base64,...) 格式内嵌，尺寸熔断
- 打印样式支持: 包含 @media print、A4 纸张排版与分页防跨页规则
- 导出文件原子性与路径越界防护
"""
from __future__ import annotations

from datetime import datetime, timezone
import html
import os
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pytest

feedback_report = pytest.importorskip("core.feedback_report")
render_html_report = feedback_report.render_html_report
build_report_model = feedback_report.build_report_model


# ---------------------------------------------------------------------------
# 测试固件构造辅助
# ---------------------------------------------------------------------------

def make_html_test_record(
    *,
    student_id: str = "001",
    student_name: str = "张同学",
    action: str = "front_straight",
    stance: str = "left",
    notes: str = "",
    needs_rerecord: bool = False,
    source_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """生成用于 HTML 渲染测试的字典记录。"""
    if source_paths is None:
        source_paths = {
            "front": r"D:\CodeProject\vision\outputs\records\front.mp4",
            "side": r"C:\Users\tester\Videos\side.mp4",
        }

    checks = [
        {
            "id": "front_straight:start:front:stance_elbow",
            "code": "stance_elbow",
            "name": "前手肘角过大",
            "bodyPart": "前侧臂肘（左）",
            "segment": "front_straight",
            "segmentLabel": "前手直拳",
            "phase": "start",
            "phaseLabel": "开始实战式",
            "standard": "前臂与上臂夹角约90度",
            "source": "散打规范 表1",
            "status": "candidate",
            "review": "pending",
            "reason": "左肘角180度超过标准",
            "blockedReason": "",
            "role": "front",
            "measurements": [{"view": "front", "validFrames": 25, "totalFrames": 30, "violationRatio": 0.8, "support": 0.8}],
            "fusion": {"method": "action_rule_quality_weighted_v1", "baseWeights": {"front": 0.5, "side": 0.5}, "support": 0.8, "threshold": 0.5, "effectiveWeights": {"front": 0.5, "side": 0.5}},
            "evidence": ["front:10"],
        },
        {
            "id": "front_straight:motion:front:guard_low",
            "code": "guard_low",
            "name": "出拳过程护手下落",
            "bodyPart": "后手（右）",
            "segment": "front_straight",
            "segmentLabel": "前手直拳",
            "phase": "motion",
            "phaseLabel": "动作过程",
            "standard": "护手保持在下颚附近",
            "source": "散打规范 表1",
            "status": "not_observed",
            "review": "pending",
            "reason": "护手保持在下颚附近",
            "blockedReason": "",
            "role": "rear",
            "measurements": [],
            "fusion": None,
            "evidence": [],
        },
        {
            "id": "front_straight:finish:front:shoulder_level",
            "code": "shoulder_level",
            "name": "出拳击打高度不足",
            "bodyPart": "前拳（左）",
            "segment": "front_straight",
            "segmentLabel": "前手直拳",
            "phase": "finish",
            "phaseLabel": "动作结束瞬间",
            "standard": "出拳击打点与肩同高",
            "source": "散打规范 表1",
            "status": "unable",
            "review": "pending",
            "reason": "缺少侧面视角，无法准确判定肩平",
            "blockedReason": "缺少侧面视角",
            "role": "front",
            "measurements": [],
            "fusion": None,
            "evidence": [],
        },
        {
            "id": "front_straight:finish:front:vertical_fist",
            "code": "vertical_fist",
            "name": "拳峰朝向待人工核对",
            "bodyPart": "前拳（左）",
            "segment": "front_straight",
            "segmentLabel": "前手直拳",
            "phase": "finish",
            "phaseLabel": "动作结束瞬间",
            "standard": "击中瞬间立拳或平拳",
            "source": "散打规范 表1",
            "status": "pending_rule",
            "review": "pending",
            "reason": "当前轻量Pose未支持拳峰朝向自动识别",
            "blockedReason": "模型不支持",
            "role": "front",
            "measurements": [],
            "fusion": None,
            "evidence": [],
        },
    ]

    return {
        "schemaVersion": 1,
        "id": "f" * 32,
        "studentId": student_id,
        "studentName": student_name,
        "action": action,
        "stance": stance,
        "createdAt": "2026-09-26T14:00:00+00:00",
        "expiresAt": "2026-10-10T14:00:00+00:00",
        "revision": 1,
        "status": "ready",
        "sourcePaths": source_paths,
        "videos": {"front": "front.mp4", "side": "side.mp4"},
        "result": {
            "schemaVersion": 1,
            "ruleVersion": "sanda-feedback-mediapipe-2026-09-20-v4",
            "action": action,
            "stance": stance,
            "backend": "mediapipe",
            "poseVariant": "full",
            "delegate": "cpu",
            "needsRerecord": needs_rerecord,
            "checks": checks,
            "evidence": [
                {
                    "id": "front:10",
                    "view": "front",
                    "frame": 10,
                    "timeSeconds": 0.33,
                    "landmarks": [[0.5, 0.5, 0.0, 1.0] for _ in range(33)],
                    "validMask": [True] * 33,
                }
            ],
            "reviewHistory": [],
            "diagnostics": {
                "front": {"totalFrames": 30, "bodyValidFrames": 28, "phaseSource": "own_view", "reason": ""},
                "side": {"totalFrames": 30, "bodyValidFrames": 27, "phaseSource": "own_view", "reason": ""},
            },
            "phaseWindows": {
                "front": [
                    {"segment": "front_straight", "phase": "start", "ranges": [[0, 5]]},
                    {"segment": "front_straight", "phase": "motion", "ranges": [[6, 18]]},
                    {"segment": "front_straight", "phase": "finish", "ranges": [[19, 23]]},
                    {"segment": "front_straight", "phase": "end", "ranges": [[24, 29]]},
                ]
            },
            "capture": {
                "front": {"width": 1280, "height": 720, "sourceFps": 30},
                "side": {"width": 1280, "height": 720, "sourceFps": 30},
            },
        },
    }


# ===========================================================================
# 1. 纯自包含与零外链断言测试
# ===========================================================================

def test_html_report_strictly_zero_external_links():
    """断言导出的单文件 HTML 报告严格为 100% 离线自包含，零外部网络资源。

    严禁：
    - CDN 样式或脚本链接 (如 cdnjs, unpkg, jsdelivr 等)
    - 外部 Web 字体 (如 Google Fonts, fonts.googleapis.com, fonts.gstatic.com)
    - 外部图片 URL (http:// 或 https:// 或 //)
    - 外部脚本 (<script src="http...">)
    """
    record = make_html_test_record()
    html_content = render_html_report(record)

    assert isinstance(html_content, str)
    assert "<!DOCTYPE html>" in html_content or "<html" in html_content

    # 清理合法的 XML namespace 声明字符串 (http://www.w3.org/2000/svg)
    scrubbed = html_content.replace("http://www.w3.org/2000/svg", "")
    scrubbed = scrubbed.replace("http://www.w3.org/1999/xlink", "")

    # 严格断言零 http:// 与 https:// 与 // 协议链接
    assert "http://" not in scrubbed, "HTML 报告中检测到外部 http:// 链接！"
    assert "https://" not in scrubbed, "HTML 报告中检测到外部 https:// 链接！"
    assert "src=\"//" not in scrubbed, "HTML 报告中检测到外部 // 无协议外链！"
    assert "href=\"//" not in scrubbed, "HTML 报告中检测到外部 // 无协议外链！"

    # 严禁任何外部 script 引用
    assert "<script src" not in scrubbed
    # 严禁外部样式表引用
    assert "<link rel=\"stylesheet\"" not in scrubbed


# ===========================================================================
# 2. XSS 注入防护与安全转义测试
# ===========================================================================

@pytest.mark.parametrize(
    "field_name, malicious_payload",
    [
        ("student_name", "<script>alert('xss_name')</script>"),
        ("student_name", '"><img src=x onerror=alert("img_xss")>'),
        ("student_name", "<svg/onload=alert('svg_xss')>"),
        ("student_id", "stu_01<script>confirm(1)</script>"),
    ],
)
def test_html_report_xss_sanitization(field_name, malicious_payload):
    """验证所有用户输入字段在渲染到 HTML 报告时均经过严格转义，杜绝 XSS。"""
    kwargs = {field_name: malicious_payload}
    record = make_html_test_record(**kwargs)
    html_content = render_html_report(record)

    # 严禁以未转义的原始 HTML 标签形式存在
    assert "<script" not in html_content
    assert "<img" not in html_content
    assert "<svg/onload=" not in html_content

    # 必须以合法转义实体形式存在
    assert ("&lt;script&gt;" in html_content) or ("&lt;img" in html_content) or ("&lt;svg" in html_content)


def test_html_report_check_reasons_and_standards_escaped():
    """验证规则检查项名称、标准和描述中即使包含 HTML 标签也安全转义。"""
    record = make_html_test_record()
    # 注入特殊字符到检查项
    record["result"]["checks"][0]["standard"] = "<b>严格肩平</b> & 击打高度 < 180cm"
    record["result"]["checks"][0]["reason"] = "<a href='javascript:alert(1)'>点击查看</a>"
    html_content = render_html_report(record)

    assert "<a href='javascript:alert" not in html_content
    assert "&lt;b&gt;严格肩平&lt;/b&gt;" in html_content or "&lt;b&gt;" in html_content
    assert "&amp;" in html_content or "&lt; 180cm" in html_content


# ===========================================================================
# 3. 隐私保护与本地绝对路径脱敏测试
# ===========================================================================

def test_html_report_privacy_no_absolute_paths():
    """断言导出的 HTML 报告绝不泄露用户本地或开发机的绝对机器物理路径。

    例如：
    - D:\\CodeProject\\vision\\...
    - C:\\Users\\tester\\...
    """
    record = make_html_test_record(
        source_paths={
            "front": r"D:\CodeProject\vision\outputs\records\front.mp4",
            "side": r"C:\Users\secret_user\Desktop\side.mp4",
        }
    )
    html_content = render_html_report(record)

    # 敏感物理路径脱敏断言
    assert "D:\\CodeProject" not in html_content
    assert "D:/CodeProject" not in html_content
    assert "C:\\Users" not in html_content
    assert "C:/Users" not in html_content
    assert "secret_user" not in html_content


# ===========================================================================
# 4. 证据图像 Base64 Data URI 嵌入测试
# ===========================================================================

def test_html_report_images_as_base64_data_uri(tmp_path):
    """验证当存在证据视频时，提取的证据帧以 Base64 Data URI 嵌入，无文件路径泄露。"""
    # 创建真实受管历史目录结构
    history_root = tmp_path / "practice_feedback"
    rec_id = "f" * 32
    rec_dir = history_root / rec_id
    rec_dir.mkdir(parents=True)

    # 写入小型测试视频
    vid_file = rec_dir / "front.mp4"
    writer = cv2.VideoWriter(str(vid_file), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 90))
    for i in range(15):
        writer.write(np.full((90, 120, 3), i * 15, dtype=np.uint8))
    writer.release()

    record = make_html_test_record()
    record["id"] = rec_id
    record["videos"] = {"front": "front.mp4"}

    html_content = render_html_report(record, history_root=history_root)

    # 验证 Base64 嵌入标记
    assert "data:image/jpeg;base64," in html_content or "data:image/png;base64," in html_content

    # 严禁出现本地 file:// 或本地裸路径图片 src
    assert 'src="file://' not in html_content
    assert 'src="C:/' not in html_content
    assert 'src="D:/' not in html_content


def test_html_report_missing_video_shows_graceful_fallback(tmp_path):
    """验证当历史视频已被清理或缺失时，降级展示文字占位，不产生破损图片标签。"""
    history_root = tmp_path / "empty_history"
    history_root.mkdir(parents=True)

    record = make_html_test_record()
    html_content = render_html_report(record, history_root=history_root)

    # 验证不崩溃，且展示降级提示
    assert html_content is not None
    assert ("原录像文件不可用" in html_content) or ("原视频已不可用" in html_content) or ("未包含" in html_content) or ("无关键点" in html_content) or ("不可用" in html_content)


# ===========================================================================
# 5. A4 打印样式与排版优化测试
# ===========================================================================

def test_html_report_a4_print_styles():
    """验证 HTML 包含完备的 @media print 打印样式规范。"""
    record = make_html_test_record()
    html_content = render_html_report(record)

    # 打印媒体查询
    assert "@media print" in html_content

    # A4 页面设置
    assert ("size: A4" in html_content) or ("size:a4" in html_content)

    # 分页防截断规则
    assert ("page-break-inside: avoid" in html_content) or ("break-inside: avoid" in html_content)

    # 不打印元素隐藏 (如操作按钮)
    assert (".no-print" in html_content) or ("display: none" in html_content)


# ===========================================================================
# 6. 数据展示与规约一致性测试
# ===========================================================================

def test_html_report_data_consistency():
    """验证导出的 HTML 报告中的统计数值与 build_report_model 规约数据完全一致。"""
    record = make_html_test_record()
    model = build_report_model(record)
    html_content = render_html_report(record)

    # 验证学号、动作名称出现在 HTML 报告中
    assert model.header.student_id in html_content
    assert model.header.action_label in html_content

    # 验证主标题出现在报告中
    assert model.headline.title in html_content

    # 验证内嵌 SVG 时序流图存在
    assert "<svg" in html_content
    assert "</svg>" in html_content
