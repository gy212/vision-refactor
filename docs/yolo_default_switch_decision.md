# YOLO 迁移 S6 默认切换决策（Issue #28）

来源：GitHub Issue #28 `S6：默认切换决策（纯决策，无实现）`
日期：2026-05-31
阶段：S6（默认切换决策）　依赖：#1 #10 #23 #24 #25 #26 #27

> **结论先行：正式评分与 full tech_eval 全部不切默认。**
> 当前只保留受控的 body-only 入口：batch 显式 `body_core_v1` 调试 / 标定输出继续保留
> `score_authorized=False`；Vue/Tauri 新桌面 bridge 可在模型与 runtime 明确可用时进入
> YOLO26n/s 实时 body-only 预览或 YOLO26L 离线高质量 body-only 内部分析，但结果必须标
> `displayScope=limited|internal` 且不得进入正式评分或 full tech_eval。
> `apps/main.py` 与 `apps/app_ui.py` 均保持 MediaPipe 默认路径。

---

## 一、决策输入汇总

| 输入 | 当前结论 | 对默认切换的影响 |
|---|---|---|
| #1 许可 | Ultralytics AGPL-3.0 仅适合内部研发 / 评估；闭源 / 商业部署需 Enterprise 或替代方案 | 不满足对外默认切换前置条件 |
| #10 标定 | `body_core_v1` 结论为“仅预览 / 内部标定参考”；J1 corr=0.280 < 0.70，J4 一致率=0.50 < 0.75 | 模板匹配默认不切 YOLO / body_core |
| #23 GPU 复测 | no-go；CUDA 已修复，6/6 样本有效，但 Hands 关 FPS 比仅 1/6 达到 1.30，Hands 开 0/6 达到 1.20，YOLO raw FPS 0/6 达到 62.85，抖动 3/6 超阈值 | 实时预览和 UI 默认不切 |
| #24 batch | 只授权离线显式 `body_core_v1` 调试 / 标定入口；`calibration_status=unvalidated`，`score_authorized=False` | 可保留内部入口，不作为默认评分 |
| #25 CLI | `apps/main.py --backend yolo` 实时预览入口关闭 / 不实现 | CLI 默认不切 |
| #26 UI | `apps/app_ui.py`（Tkinter 旧入口）YOLO 后端选择关闭 / 不实现；2026-06-11 Vue/Tauri 新桌面 bridge 另行接入受控 body-only 预览 / 内部分析路由 | 旧 UI 默认不切；新桌面入口仍不得授权评分 |
| #27 Hybrid | `yolo_body_mp_pose_supplement` 不实现，不保留半成品 runtime | Hybrid 不进入默认 / 离线路径 |

---

## 二、三条链路默认决策

| 链路 | 默认决策 | 数字 / 事实依据 | 允许保留的入口 |
|---|---|---|---|
| 实时预览（CLI / Tkinter UI） | **不切默认，继续 MediaPipe** | #23 CUDA 实测 no-go：6/6 有效 GPU 行，但 Hands 关 / Hands 开实时阈值未通过；#25 / #26 仍关闭 | 无 CLI / Tkinter YOLO 入口 |
| Vue/Tauri 实时预览 | **受控 body-only，不是评分默认** | 2026-06-11 桌面迁移：`enableHands=false` 且 YOLO26n/s 模型与 runtime 可用时可选 YOLO body-only；模型不可用或安装版不支持时回退 MediaPipe 并展示 fallback | `displayScope=limited`、`scoreAuthorized=false` |
| 模板匹配 | **不切默认，继续 `pose33_v3` / MediaPipe** | #10 J1 corr=0.280、J4 一致率=0.50，`score_authorized=False` | 显式离线 `body_core_v1` 调试 / 标定入口 |
| Vue/Tauri 离线高质量 body-only 分析 | **内部分析，不是正式评分** | 2026-06-11 桌面迁移：YOLO26L 可用时走 `body_core_v1` 内部 payload；缺模型 / 安装版不支持时返回 `yolo26l_unavailable` 结构化错误，不回退 MediaPipe | `displayScope=internal`、`scoreAuthorized=false`、`calibrationStatus=unvalidated` |
| 规则 / 技术评估 | **不切默认，继续 MediaPipe full** | COCO17 缺嘴角、脚跟脚尖、手指；#10 仅预览；#11 只做 MediaPipe 侧结构化状态 | YOLO 不进入 full tech_eval / 对外评分 |
| Hybrid | **不实现，不参与默认切换** | #27 未满足 CUDA / Hybrid benchmark / 补点价值触发条件 | 无 |

---

## 三、默认切换前置条件

若未来要重新评估任何默认切换，必须先另开实现 / 决策子任务，并同时满足：

1. 许可路径可用于目标部署形态：闭源 / 商业部署需 Enterprise license 或替代模型方案。
2. CUDA-enabled 环境下 #23 GPU 复测 6/6 样本有效，Hands 关 / Hands 开端到端 FPS 比分别达到 1.30 / 1.20，且 YOLO raw FPS 与抖动阈值达标。
3. #10 或后续标定在目标样本集上重新通过，模板匹配分数相关性、pass/fail 一致率和误判样例均达标。
4. 规则 / 技术评估未评估比例、缺失能力和误判样例可接受，且仍不得把缺失点伪造成有效点。
5. CLI / UI / batch 均能显示结果来源、`feature_layout`、`calibration_status`、`score_authorized`。
6. 有一键回滚配置，可在 YOLO 入口异常时恢复 MediaPipe 默认路径。

---

## 四、当前收尾结论

- `apps/main.py`：默认保持 MediaPipe；不新增 `--backend yolo` 实时预览入口。
- `apps/app_ui.py`：默认保持 MediaPipe；不新增 YOLO 后端选择或未标定评分提示入口。
- `frontend/` Vue/Tauri 新桌面：允许通过 `core/backend_router.py` 进入受控 body-only 路由；
  YOLO26n/s 仅用于实时预览，YOLO26L 仅用于离线高质量 body-only 内部分析。
  安装版 sidecar 当前仍声明不打包 YOLO runtime；若 runtime/model 不可用，实时预览回退
  MediaPipe，离线 high-quality 返回结构化下载 / 安装错误。
- `apps/make_template.py` / `apps/match_template.py` / batch：保留显式实验 / 标定入口，但输出必须标明
  `calibration_status=unvalidated` 与 `score_authorized=False`，不得进入用户对外评分。
- `analysis/tech_eval.py` / `core/rule_scoring.py`：默认继续 MediaPipe full；YOLO-only 不进入 full tech_eval。
- `yolo_body_mp_pose_supplement`：不实现。

因此 Issue #28 的 S6 决策更新为：**正式评分 / full tech_eval / CLI / Tkinter 默认路径全部不切 YOLO；
Vue/Tauri 仅保留受控 body-only 预览与内部分析入口，且不授权对外评分**。
