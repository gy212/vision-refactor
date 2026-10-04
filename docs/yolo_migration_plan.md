# MediaPipe 到 YOLO 视觉方案迁移计划

## 结论先行

YOLO-pose 作为主要人体姿态后端是可行的，但不建议把 YOLO COCO 17 点简单补齐成 BlazePose 33 点后直接复用全部下游逻辑。当前系统的模板匹配、规则评分、技术评估都深度依赖 BlazePose 11..32 的完整语义，其中手指、嘴角、脚跟、脚尖在 YOLO COCO pose 中不存在。更稳的迁移路线是：

1. 保留 `VisionPipeline` 兼容门面，让旧调用仍能拿到 Pose33-like landmarks。
2. 新增后端能力声明和关键点布局声明，区分 `pose33`、`coco17`、`body_core`。
3. 模板匹配默认迁到双方都可靠的 `body_core` 特征布局，不让伪造点进入 DTW。
4. 规则评分和技术评估按所需关键点显式降级；缺嘴角/脚尖/脚跟时返回“未评估”或启用 MediaPipe Pose 补充。
5. 在性能、精度、许可验收通过前，默认后端不要直接切到 YOLO。

## 已验证事实

- Ultralytics YOLO pose 默认 COCO keypoints 为 17 点：nose、eyes、ears、shoulders、elbows、wrists、hips、knees、ankles，不包含嘴角、手指、脚跟、脚尖。
- Ultralytics Python 结果提供 `result.keypoints.xy`、`result.keypoints.xyn`、`result.keypoints.data`，可映射为归一化关键点容器。
- Ultralytics tracking 支持 pose 模型，连续帧应使用 `model.track(frame, persist=True, tracker=...)`；这提供实例 ID 跟踪，不等价于 MediaPipe VIDEO 模式的 landmark smoothing。
- MediaPipe Pose Landmarker 输出 33 个 normalized landmarks 和 world landmarks；Hand Landmarker 输出每手 21 点。
- Ultralytics Python 包许可证为 AGPLv3+；商业闭源部署必须在迁移前确认 Enterprise license 或替代推理方案。
- 当前仓库没有 `.venv`，本次只完成方案验证和文档优化，未运行真实推理基准。

## 当前代码影响面

| 模块                                               | 当前职责                             | 迁移影响                                                                                                                 |
| ------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `core/vision_pipeline.py`                        | MediaPipe Pose/Hands 封装、绘制、动作标签  | 必须拆出后端实现，保留 `infer()`、`annotate()`、`next_timestamp_ms()` API                                                         |
| `core/pose_features.py`                          | 模板特征 `(T,22,2)`，BlazePose 11..32 | 高风险；必须新增 `body_core` normalizer，旧 Pose33 模板保持 MediaPipe-only                                                         |
| `core/action_compare.py`                         | 模板生成、DTW、双模板比对                   | 需要透传 backend/layout，并在 metadata 记录后端、模型、布局、补点策略                                                                      |
| `core/rule_scoring.py`                           | Pose33 原始数组和规则扣分                 | 每条规则声明 required landmarks；缺失时未评估或启用补充后端                                                                              |
| `analysis/tech_eval.py`                          | 直拳技术指标，依赖脸部/脚部/可见度               | 必须纳入迁移范围，不能只改 `core`                                                                                                 |
| `apps/main.py`、`apps/app_ui.py`                  | CLI/UI 实时和离线入口                   | `apps/main.py` 按 Issue #25 no-go 当前保持 MediaPipe CLI；`apps/app_ui.py` 按 Issue #26 no-go 当前不增加 YOLO 后端选择；默认切换另按 #28 决策 |
| `apps/make_template.py`、`apps/match_template.py` | 独立模板脚本                           | 增加 backend/layout 参数，拒绝不兼容模板静默混用                                                                                     |
| `batch/*.py`                                     | 批量比对、骨架导出、技术评估                   | 增加 backend/layout 参数，导出 metadata                                                                                     |

## 关键设计

### 1. 统一结果容器

新增 `core/landmarks.py`：

```python
@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0

@dataclass(frozen=True)
class PoseResult:
    landmarks33: tuple[Landmark, ...] | None
    layout: str                  # "pose33" | "coco17_mapped33"
    source: str                  # "mediapipe" | "yolo"
    model_name: str
    track_id: int | None = None
    capabilities: frozenset[str] = frozenset()
```

兼容要求：

- `VisionPipeline.infer()` 继续返回 `(pose_landmarks, hands)`，其中 `pose_landmarks` 默认仍可按 `lm[i].x` 访问。
- 需要 metadata 的新调用使用 `infer_result()` 或内部 backend result，不破坏旧 API。
- 缺失点坐标可以填 fallback，但 visibility 必须低于阈值，并且不能进入 `body_core` 特征。

### 2. 后端协议

新增 `core/pose_backend.py`：

```python
class PoseBackend(Protocol):
    name: str
    capabilities: frozenset[str]

    def detect_pose(self, frame_bgr, *, timestamp_ms: int | None = None) -> PoseResult | None:
        ...
```

后端能力建议：

- `pose33_full`
- `coco17_body`
- `video_tracking`
- `hands21`
- `face_mouth`
- `foot_heel_toe`
- `world_landmarks`

### 3. 后端实现

MediaPipe：

- 从现有 `MediaPipePipeline` 中拆出 `MediaPipePoseBackend` 和 `MediaPipeHandBackend`。
- 保留 VIDEO/IMAGE 语义和 timestamp 校验。
- 输出 `layout="pose33"`，capabilities 包含 `pose33_full`、`face_mouth`、`foot_heel_toe`。

YOLO：

- 新增 `core/yolo_backend.py`，懒加载 `from ultralytics import YOLO`。
- 模型映射建议先固定到一个明确版本，迁移实施时二选一：
  - 保守路线：`ultralytics==8.x.y` + `yolov8n/s/m-pose.pt`
  - 新代际路线：`ultralytics==8.x.y` + `yolo11/yolo26*-pose.pt`
- `running_mode="image"` 使用 `model.predict(frame, verbose=False)`。
- `running_mode="video"` 使用 `model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)`。
- 多人选择策略必须固定：优先延续当前 `track_id`，其次选最大 person box，若 box 接近则选画面中心最近者。
- COCO17 映射到 Pose33 时只把真实对应点设为高 visibility；嘴角、手指、脚跟、脚尖不得伪装成有效点。

COCO17 到 BlazePose33 真实可用映射：

| BlazePose | COCO | 名称             |
| --------- | ---- | -------------- |
| 0         | 0    | nose           |
| 2         | 1    | left_eye 近似    |
| 5         | 2    | right_eye 近似   |
| 7         | 3    | left_ear       |
| 8         | 4    | right_ear      |
| 11        | 5    | left_shoulder  |
| 12        | 6    | right_shoulder |
| 13        | 7    | left_elbow     |
| 14        | 8    | right_elbow    |
| 15        | 9    | left_wrist     |
| 16        | 10   | right_wrist    |
| 23        | 11   | left_hip       |
| 24        | 12   | right_hip      |
| 25        | 13   | left_knee      |
| 26        | 14   | right_knee     |
| 27        | 15   | left_ankle     |
| 28        | 16   | right_ankle    |

不可用点：

- BlazePose 1、3、4、6 是眼睛细分点，不能从 COCO 无损得到。
- BlazePose 9、10 嘴角不可用。
- BlazePose 17..22 手指不可用。
- BlazePose 29..32 脚跟/脚尖不可用。

### 4. 特征布局策略

新增 `body_core_v1`，作为 YOLO 与 MediaPipe 共享模板布局：

```text
11 L_SHOULDER
12 R_SHOULDER
13 L_ELBOW
14 R_ELBOW
15 L_WRIST
16 R_WRIST
23 L_HIP
24 R_HIP
25 L_KNEE
26 R_KNEE
27 L_ANKLE
28 R_ANKLE
```

要求：

- `normalize_pose_body_core_v1()` 只使用上述 12 点，输出 `(12,2)`。
- 原 `normalize_pose_xy_v3()` 和旧 `(22,2)` 模板继续保留，标记为 `pose33_v3`。
- YOLO 默认只能创建/匹配 `body_core_v1` 模板。
- MediaPipe 模板默认也逐步迁到 `body_core_v1`，便于跨后端比较。
- 比对时若模板 layout/backend 不兼容，默认报错或显式警告，不静默套用旧 baseline。
- 评分 baseline 需要按 layout 单独标定，`pose33_v3` 的 `baseline=2.0` 不应直接沿用给 `body_core_v1`。

### 5. 规则和技术评估策略

每条规则声明 required landmarks：

| 规则/指标                            | 必需关键点                               | YOLO-only 策略              |
| -------------------------------- | ----------------------------------- | ------------------------- |
| 肘角、膝角、站距                         | shoulder/elbow/wrist/hip/knee/ankle | 可评估                       |
| 拳峰与鼻尖同高                          | nose/wrist                          | 可评估，但需重新标定阈值              |
| 后手贴近下颌、护手位置                      | mouth 或 face proxy                  | 未评估，或启用 MediaPipe Pose 补充 |
| 脚尖方向、踝角、脚掌支撑边界                   | heel/foot_index                     | 未评估，或启用 MediaPipe Pose 补充 |
| V_SIGN                           | hands21                             | 继续使用 MediaPipe Hand       |
| `analysis/tech_eval.py` 中发力/重心部分 | 多处依赖 heel/foot_index/mouth/ear      | 分指标声明可用性                  |

Hybrid 模式命名：

- `mediapipe_full`：Pose33 + Hands，旧行为。
- `yolo_body`：YOLO body only，最快，但规则不完整。
- `yolo_body_mp_hands`：YOLO body + MediaPipe Hands，实时主链路候选。
- `yolo_body_mp_pose_supplement`：已由 Issue #27 / `docs/yolo_gpu_recheck_report.md`
  supersede；当前不实现，只保留为未来满足触发条件后另开实现子任务的历史命名。

注意：MediaPipe Pose 补脚/脸不是轻量补点，实际仍要跑完整 PoseLandmarker。Issue #27
已基于 GPU 复测 no-go、S0 CPU speedup=0.41 和 COCO17 结构性缺点作出决议：当前不实现
Hybrid，也不在代码库保留半成品 runtime 路径。

## 分阶段实施计划

### P0. 决策门和基准预研

负责人：算法/工程共同。

任务：

- 确认 Ultralytics 许可证路径：AGPLv3+ 是否可接受，或采购 Enterprise license，或改用 ONNXRuntime/自训权重/其他许可更合适的 pose 模型。
- 固定 Python、torch、ultralytics、模型文件名、设备策略（CPU/GPU）。
- 选 3 到 5 段现有标准/学员视频，建立 MediaPipe 当前结果基线：FPS、检测失败帧率、模板分数、规则/技术评估结果。
- 用 YOLO 单独跑同样样本，导出关键点 NPZ/JSON、FPS、多人/丢帧情况。

注意：

- 没有许可结论前，不做默认后端切换。
- 不要用单帧截图代表视频稳定性，必须看连续帧 track id 和关键点抖动。

验收标准：

- 有明确许可结论。
- 有 `docs/yolo_baseline_report.md` 或同等记录，包含样本、命令、版本、FPS、失败帧率、主要差异。
- YOLO 输出 shape、归一化坐标、confidence、多人选择策略可被代码稳定解析。

### P1. 抽象层和 MediaPipe 等价重构

负责人：核心 pipeline 工程。

任务：

- 新建 `core/landmarks.py`、`core/pose_backend.py`。
- 将现有 MediaPipe 逻辑拆成 backend，但保留 `VisionPipeline` 门面。
- `MediaPipePipeline` 保留为兼容别名或薄包装，旧 import 不崩。
- 统一模型目录，建议使用仓库顶层 `models/`，并修正当前 `apps/models`、`core/models` 不一致的问题。
- 抽出 Pose33 转 `(33,4)` 的工具函数，减少 `rule_scoring.py`、`tech_eval.py`、batch 重复循环。

注意：

- 本阶段不改变默认行为，不引入 YOLO。
- `running_mode="video"` 仍要求 timestamp，`image` 仍可并发。

验收标准：

- `py_compile` 覆盖 `core/vision_pipeline.py`、`core/pose_features.py`、`core/action_compare.py`、`core/rule_scoring.py`、`analysis/tech_eval.py`、`apps/main.py`、`apps/app_ui.py`。
- `--backend mediapipe` 或默认 MediaPipe 路径与重构前输出一致或差异可解释。
- 旧入口 `from core.vision_pipeline import MediaPipePipeline, PipelineConfig` 仍可用。

### P2. YOLO body-only 后端

负责人：模型后端工程。

任务：

- 实现 `YoloPoseBackend`：模型加载、predict/track、COCO17 解析、Pose33-like 映射、capabilities。
- 实现有状态 subject 选择：track id 延续、最大框/中心距离回退、丢失重置。
- 实现 YOLO 绘制连接，不依赖 MediaPipe `PoseLandmarksConnections`。
- 增加最小单元测试或脚本验证空帧、无人帧、多人帧、单人帧。

注意：

- 缺失点 visibility 必须为 0 或低于规则阈值。
- 不要在 YOLO backend 内默认启动 MediaPipe Pose 补点。
- `model.track()` 的 persist 状态不要跨不同视频复用。

验收标准：

- 单帧 YOLO 输出长度 33，坐标有限且在合理范围，真实 COCO 对应点 confidence 正确传递。
- 连续视频 track id 在单人场景稳定；断轨后能恢复且有日志或 metadata。
- 无人帧返回 `None` 或上层明确 fallback，不抛异常。
- `annotate()` 可显示 YOLO 骨架并保持 `(frame, actions)` 返回格式。

### P3. 特征布局和模板迁移

负责人：算法/模板匹配工程。

任务：

- 在 `core/pose_features.py` 新增 `normalize_pose_body_core_v1()`、layout 常量、左右镜像映射。
- 修改 `core/action_compare.py`，让 `_extract_pose_features()`、`create_template_from_video()`、`compare_video_to_template()`、`compare_video_to_dual_templates()` 接受 `backend` 和 `feature_layout`。
- 模板 metadata 增加：`backend`、`model_name`、`feature_layout`、`normalizer_version`、`running_mode`、`capabilities`、`supplement_mode`。
- 对旧 `pose33_v3` 模板保持兼容，但 YOLO 默认拒绝使用。
- 为 `body_core_v1` 重新标定 DTW baseline。

注意：

- `body_core_v1` shape 是 `(T,12,2)`，不能让旧代码假设 `(T,22,2)`。
- 跨 layout 比对必须报错。
- 跨 backend 同 layout 可允许，但要在结果中记录并提示基线可能不同。

验收标准：

- MediaPipe 可生成 `pose33_v3` 和 `body_core_v1` 模板。
- YOLO 可生成和匹配 `body_core_v1` 模板。
- 旧模板仍能用 MediaPipe 路径比对。
- 双模板比对、重复动作匹配、错误分析在 layout 不兼容时给出清晰错误。

### P4. 规则评分和技术评估分级

负责人：规则/技术评估工程。

任务：

- 为 `Rule` 增加 `required_landmarks` 和 `required_capabilities`。
- `score_rules()` 输出每条规则的状态：`evaluated`、`skipped_missing_landmarks`、`skipped_low_confidence`。
- `extract_pose_raw()` 和 `analysis.extract_pose_and_view_scores()` 接受 backend/supplement 配置。
- 将 `analysis/tech_eval.py` 的指标按 YOLO-only、MediaPipe-full，以及“未来可能另开任务评估的
  Hybrid”三档声明可用性；当前实现不得依赖 Hybrid。
- 不实现 `yolo_body_mp_pose_supplement`。该项已由 Issue #27 / GPU 复测报告 supersede；
  若未来要恢复，必须另开实现子任务并先满足 #27 的 CUDA 有效复测、Hybrid 专用 benchmark、
  补点指标恢复证据等触发条件。

注意：

- 有效帧不足和关键点缺失要区分，避免把“无法评估”当成合格或不合格。
- 补点模式会带来第二个 pose detector；当前不实现。未来另开实现子任务时必须先记录并验收额外耗时。

验收标准：

- YOLO-only 下可评估规则正常输出，不可评估规则有明确原因。
- 当前不实现 Hybrid 补充模式；脚尖/脚跟相关规则缺失时必须输出未评估原因，不能伪造成有效点。
- 技术评估 UI/CSV/JSONL 包含规则完整度和未评估原因。
- 同一视频 MediaPipe 旧路径回归不退化。

### P5. CLI、UI、batch 集成

负责人：应用工程。

任务：

- `apps/main.py` 的 YOLO 实时预览入口已由 Issue #25 / `docs/yolo_gpu_recheck_report.md`
  supersede：#23 CUDA 修复后真实 GPU 复测仍 no-go，当前不实现 `--backend yolo` /
  `--feature-layout body_core_v1`，保持既有 MediaPipe CLI 默认路径；未来若要恢复，
  必须先让 Hands 关 / Hands 开实时阈值和 YOLO raw / 抖动阈值全部达标，并另开实现子任务。
- `apps/app_ui.py` 的 YOLO 后端选择与规则完整度提示已由 Issue #26 /
  `docs/yolo_gpu_recheck_report.md` supersede：#23 CUDA 实测 no-go 且 #25 仍不实现时当前不实现，
  保持既有 MediaPipe UI；未来若要恢复，必须先重新打开并落地 #25 后续实现子任务，再另开 UI 实现子任务。
- `apps/make_template.py`、`apps/match_template.py`、`batch/batch_dual_compare.py`、`batch/batch_export_skeleton.py`、`batch/batch_tech_eval.py` 透传 backend/layout/supplement。
- 更新 README 或使用说明。

注意：

- 默认值在 P6 前建议仍为 MediaPipe 或“自动但偏保守”，避免未标定 YOLO 影响用户结果。
- UI 文案要避免承诺 YOLO-only 可完整评估所有规则；当前 #26 no-go 分支不新增 YOLO UI 文案或控件。

验收标准：

- CLI 当前可运行 / 可保留：
  - `apps/main.py --source 0`
  - `apps/main.py --source input.mp4 --no-show --out out.mp4`
  - `apps/make_template.py --backend yolo --feature-layout body_core_v1 ...`
  - `apps/match_template.py --backend yolo --feature-layout body_core_v1 ...`
- `apps/main.py` 的 YOLO 实时预览命令不属于当前验收范围；除非未来另开 #25 后续实现子任务，否则不加入当前 CLI 可运行清单。
- UI 主预览、动作比对、直拳检测当前保持 MediaPipe 旧路径；YOLO 后端选择不属于当前验收范围。
- batch 输出包含 backend/layout/supplement metadata。

### P6. 标定、性能和默认切换

负责人：算法/产品/工程共同。

任务：

- 在标准动作和学员样本上跑 MediaPipe vs YOLO 回归；Hybrid 已由 Issue #27 决议为当前不实现，
  仅能在未来另开实现子任务后再纳入回归。
- 标定 `body_core_v1` 的 DTW baseline 和规则阈值。
- 统计 CPU/GPU FPS、初始化时间、显存/内存、失败帧率。
- 决定默认后端：
  - 实时预览可默认 `yolo_body_mp_hands`，前提是 FPS 和许可达标。
  - 规则/技术评估默认仍使用 `mediapipe_full`；Hybrid 当前不实现，未来若恢复必须先另开实现子任务并重新决策。

注意：

- 默认切换必须有回滚开关。
- 不能只看平均分，要看误判样例和未评估比例。

验收标准：

- 有回归报告，列出每个样本的分数差异、规则差异、未评估规则、FPS。
- 默认切换后的主要用户路径有端到端验证。
- 文档记录推荐配置、硬件要求和已知限制。

## 风险与缓解

| 风险                  | 影响                       | 缓解                                        |
| ------------------- | ------------------------ | ----------------------------------------- |
| AGPLv3+ 许可不适合部署     | 无法合法闭源商用                 | P0 前置确认 Enterprise license 或替代模型          |
| YOLO 缺失 Pose33 点    | 模板/规则误判                  | 使用 `body_core_v1`，规则按能力降级                 |
| 伪造点污染 DTW           | 跨后端分数漂移                  | 缺失点不进入 shared layout，旧 layout 限 MediaPipe |
| YOLO track ID 丢失/换人 | 视频分数不稳定                  | subject 选择策略和断轨恢复                         |
| Hybrid 性能低于预期       | 实时卡顿                     | 实时默认不跑 MediaPipe Pose 补点                  |
| 模型版本漂移              | 结果不可复现                   | 固定 ultralytics/torch/model 版本和 metadata   |
| 模型目录混乱              | 重复下载/部署困难                | 统一顶层 `models/` 或明确用户缓存                    |
| 技术评估漏改              | UI/批量结果仍走 MediaPipe 或误评估 | 将 `analysis/tech_eval.py` 纳入 P4 验收        |

## 最小完成定义

迁移只有在以下条件全部满足时才算完成：

1. MediaPipe 旧路径可用且回归通过。
2. YOLO body-only 路径可端到端预览、导出、模板匹配。
3. `body_core_v1` 模板布局完成标定，metadata 完整。
4. 规则和技术评估对缺失关键点有明确“未评估”输出。
5. 已授权入口能记录结果来源；CLI / UI 后端选择按 #25 / #26 / #28 决策，no-go 分支不作为完成硬条件。
6. 性能、精度、许可三项都有书面结论。
