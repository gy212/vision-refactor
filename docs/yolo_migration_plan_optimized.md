# MediaPipe 到 YOLO 视觉方案迁移计划（审批优化版）

来源：基于 `docs/yolo_migration_plan.md` 审批后另存副本  
日期：2026-05-29  
复审日期：2026-05-30（对照当前代码二次核验后修订）  
三审日期：2026-05-30（吸收第三轮代码核对意见后修订）  
状态：建议作为后续实施版本，原文件保留不动

> 二次复审修订摘要（2026-05-30）：
> 1. 将“脚部/嘴部缺失对核心评分（重心、发力顺序）的影响”提升为 S0 的显式 go/no-go 停止条件，不再拖到 S3/S4 才暴露。
> 2. S1 补入实打实的漏项：`core/action_compare.py` 中 `_select_representative_cycle()` 写死 `(22,2)` 的布局守卫（约 337-343 行），`body_core_v1` 进来会静默跳过周期裁切。
> 3. 收敛措辞：`FrameResult`/`Landmark` 是兼容边界上的结果容器，约束改为“不得进入序列提取热路径”，而非否定该抽象本身。
> 4. 收敛措辞：段内 `track_id` 策略不是逻辑冲突，而是“段边界语义需写清”。
> 5. 收敛措辞：`rule_scoring.py` 已有 `RuleScore`/`RuleViolation` 结构化结果，属半结构化，改为“补 `state`/`skip_reason` 字段”，不推倒重来。

> 三审修订摘要（2026-05-30，对照代码核对后定稿）：
> 1. 首页新增**「默认前提（可证伪）」**：YOLO-only 默认不进 full tech_eval，定位为实时预览 / 模板匹配 / skip-aware partial eval；该前提是默认假设而非定论，S0 数据可推翻。修正二审里“tech_eval 绝对不能碰 YOLO”的过头措辞。
> 2. `valid_mask` 迁移**提级为 S1 专项任务并配逐位等价回归**：当前 `analysis/tech_eval.py:104` 的 `_valid()` 与 `core/rule_scoring.py:93` 的 `_valid_frame()` 仍是 `lm[idx,3] >= thr` 风格，调用面铺满算分函数体，是 S1 真正的大头，不是一条小项。
> 3. S0/S3 的“业务可接受/明显劣于/可用于评分”等措辞**全部预注册成数字阈值**（FPS 提升比例、失败帧率上限、模板分数相关性、pass/fail 一致率、score MAE、未评估比例上限），避免事后主观争论。
> 4. S1 新增 **`pose33_v3` golden 回归**：S1 改动 `_extract_pose_features()`/`mirror_pose_features()`/`_select_representative_cycle()`/误差统计等热路径，`py_compile` 对行为零保证，必须用固定 fixture 断言旧分数不漂移。
> 5. S1 **拆成两层**：① 无条件有价值的防御性修复（layout shape 参数化 + `_select_representative_cycle()` 静默退化修复 + 旧路径 golden 回归）；② `body_core_v1` normalizer 与 baseline **下移到 S2**（真接 YOLO 离线闭环时再引入），避免 S0 可能毙掉时的提前建模。
> 6. `confidence_kind` **落地为可配置阈值策略** `validity_policy`/`valid_conf_thr`，并明确 S3 标定前 YOLO 侧阈值是「待标定参数」，不允许拿未校准默认值进入正式规则/tech_eval 判定。
> 7. S2 新增**多人闸门**：`batch_tech_eval` 跑的是学员视频（教练/路人/镜面易入镜），检测到 `num_persons > 1` 必须标 `multi_person_detected` 并拒绝或降级人工复核，不得静默选最大框。
> 8. 实时链路前置改为**条件式**：仅当 YOLO 的正当理由主要是实时 FPS 时，S0 才直接 spike 真实实时路径；若首要目标是离线模板闭环，则“离线先行”仍成立。

## 默认前提（可证伪）

下面这条是整份计划的**默认工作假设**，写在最前面是为了从第一天就把工作量收敛到正确范围；它不是定论，S0 的实测数据可以推翻它。

> **YOLO-only（COCO17）默认不是 full tech_eval 的候选后端。** 静态读代码即可确认原因：`analysis/tech_eval.py` 的重心支撑面（`_foot_edges_x` 依赖 `HEEL`/`FOOT_INDEX`）、发力顺序（`_push_off_ok`/`_calc_heel_lift`/`_foot_angle_deg`/`_rotation_fail_front` 依赖脚跟脚尖）、以及 `rule_scoring.py` 的后手贴近/护手（依赖 `MOUTH`）、脚尖平行（依赖 `HEEL`/`FOOT_INDEX`），其所需关键点在 COCO17 中**结构性缺失**，不是“精度下降”而是“能力消失”。

因此 YOLO 在本项目中的合理定位分三档（**注意不是“绝对不能碰 tech_eval”，而是分级**）：

| 用途 | YOLO-only 是否候选 | 说明 |
|---|---|---|
| 实时预览 / `annotate()` 绘制 | ✅ 最适合 | 预览不算发力顺序，对缺脚趾嘴角最不敏感；速度优势在此兑现 |
| 模板匹配（`body_core_v1`） | ⚠️ 待 S3 标定 | 躯干四肢核心点足够，但 baseline 与精度需标定后才能下结论 |
| skip-aware partial tech_eval | ⚠️ 仅限 COCO17 够的指标 | 肘角/膝角/站距等可评，缺点指标必须诚实输出 `skipped_missing_landmarks` |
| full tech_eval | ❌ 默认排除 | 重心支撑面、蹬地/脚旋转、护手等核心指标结构性失效 |

> 推翻条件：若 S0 实测表明 Hybrid 补点的性能代价可接受、或缺失指标的业务权重低到可忽略，则可重新评估“YOLO 进 partial/ full eval”的边界。在此之前，所有阶段按上表定位推进。

## 审批结论

YOLO-pose 可以作为人体躯干和四肢核心点的补充或替代后端，但不适合作为 BlazePose 33 点能力的无缝替换。最优路线不是一上来重构出完整多后端框架，而是先把当前代码里最脆弱的三件事隔离出来：

1. 关键点序列提取。
2. 特征布局和模板 metadata。
3. 规则/技术评估的“可评估/不可评估”契约。

原计划的核心判断是正确的：不能把 YOLO COCO 17 点伪装成完整 Pose33，不能让伪造点进入 DTW，不能在许可、性能、精度没有书面结论前切默认后端。但原计划有一些过重部分：P1 过早拆成完整 `PoseBackend` 协议，P4 过早做 Hybrid 补点，P5 过早要求 CLI/UI/batch 全入口同时接入。

优化后的 MVP 应改成：先做决策 spike，再做 layout 隔离，然后做最薄 YOLO 离线闭环。UI、Hybrid、默认切换全部后置。

## 当前代码事实

本次审批按当前仓库代码重新校验，关键事实如下：

- 真实兼容入口是 `MediaPipePipeline` 和 `PipelineConfig`，不是文档里泛称的 `VisionPipeline`。`apps/main.py`、`apps/app_ui.py`、`core/action_compare.py`、`core/rule_scoring.py`、`analysis/tech_eval.py` 都直接或间接依赖它。
- `MediaPipePipeline.infer()` 返回 `(pose_landmarks, hands)`；`annotate()` 同时负责 pose、hands、动作标签、V 手势和绘制。只抽象 `detect_pose()` 会覆盖不到手部、动作标签和实时预览。
- 模板链路强依赖 `(T,22,2)`：`_extract_pose_features()`、缺帧补零、`mirror_pose_features()`、双模板关节误差、关节名表都默认 BlazePose 11..32。
- 除上述函数外，`core/action_compare.py` 的 `_select_representative_cycle()`（约 337-343 行）也写死了 `if features.shape[1:] != (22, 2): return features` 守卫。`body_core_v1`(12,2) 进来会**静默跳过周期裁切**，导致 DTW query 退化成整段模板。此项不报错、不抛异常，是隐性退化，最难排查，必须在 S1 一并处理。
- `normalize_pose_xy_v3()` 不检查 visibility。如果 YOLO 缺失点只填低 visibility 但坐标仍参与 normalizer，DTW 仍会被污染。
- `rule_scoring.py` 已有 `RuleScore`/`RuleViolation` 结构化结果（约 351-405 行），但“未评估/合格/扣分”三态仍靠往中文 `detail` 字符串里拼（如 `（有效帧不足，未评估）`、`（合格）`）来区分。结论：是**半结构化**，下游无法用稳定字段判断状态。改法是补 `state`/`skip_reason` 字段，不必推倒重来。
- `analysis/tech_eval.py` 对 Pose33 依赖很深：mouth、ear、index/pinky、heel、foot_index 都参与方向、拳面角度、支撑边界、上步、重心和发力顺序。具体地：`eval_cog_side` 经 `_foot_edges_x` 用 `HEEL`/`FOOT_INDEX` 算支撑面边界（约 340-353 行附近）；`eval_force_sequence` 的 `_push_off_ok`/`_calc_heel_lift`/`_foot_angle_deg`/`_rotation_fail_front` 直接依赖脚跟脚尖判蹬地与脚旋转（约 1418-1436、1555-1595 行）；`后手贴近`/`护手位置` 规则依赖 `MOUTH`。COCO17 既无脚跟脚尖也无嘴角，这些指标在 YOLO-only 下会直接失效或退化，属核心评分能力，不是边角功能。
- 模型和模板目录不统一。当前代码可能使用 `apps/models`、`core/models`、`analysis/models`、`batch/models`，模板也可能进入 `apps/templates` 或 `core/templates`。
- `requirements.txt` 目前没有 `ultralytics`、`torch`、`onnxruntime` 等 YOLO 相关依赖。

## 外部事实边界

迁移实施前应按固定版本再核验一次。当前审批采用的外部事实：

- Ultralytics Pose 官方文档显示其 pose 模型使用 COCO 17 keypoints；当前文档示例已包含 YOLO26 pose 模型。参考：[Ultralytics Pose](https://docs.ultralytics.com/tasks/pose/)。
- Ultralytics 结果对象提供 keypoints 访问接口，如 `xy`、`xyn`、`conf`/`data`，可用于构造归一化关键点容器。参考：[Ultralytics Results](https://docs.ultralytics.com/reference/results/)。
- Ultralytics tracking 支持 `model.track(...)`，跟踪器包括 BoT-SORT 和 ByteTrack；这提供实例 ID 跟踪，不等价于 MediaPipe VIDEO 模式的 landmark smoothing。参考：[Ultralytics Track](https://docs.ultralytics.com/modes/track/)。
- Ultralytics 官方说明是 AGPL-3.0 和 Enterprise 双许可。闭源或商业部署必须在 P0 给出明确许可结论。参考：[Ultralytics Licensing](https://docs.ultralytics.com/) 和 [Ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。
- 如果 YOLO 许可或效果不满足要求，替代路线应进入 P0 决策，不应硬推 YOLO。可调研 RTMPose/MMPose、ONNXRuntime/TensorRT 推理链路、自训或继续优化 MediaPipe。

## 优化原则

1. 保持旧行为默认不变。`MediaPipePipeline`、`PipelineConfig`、`infer()`、`annotate()`、`pose33_v3` 模板默认路径都必须继续工作。
2. 不把 capability 系统做成第一阶段大工程。第一版 metadata 只需要 `backend`、`model_name`、`feature_layout`、`keypoint_source`、`running_mode`、`confidence_kind`。
3. 不让缺失点进入共享模板布局。YOLO 输出 Pose33-like 只是兼容容器，不代表完整 Pose33。
4. `body_core_v1` 先作为显式 opt-in，不马上替代 MediaPipe 默认模板布局。
5. Hybrid 补点不进 MVP。MediaPipe Pose 补脸/脚本质仍是跑完整 PoseLandmarker，应在离线评估明确开启。
6. 每个阶段都要有可运行命令、输出字段检查和回滚判断。

## 推荐目标架构

### 1. 兼容门面

保留并优先兼容：

```python
MediaPipePipeline(models_dir=..., cfg=PipelineConfig(...))
pipe.infer(frame, timestamp_ms=...)
pipe.annotate(frame, timestamp_ms=...)
```

新增抽象不要抢占旧入口。推荐先新增内部结果容器：

```python
@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    confidence: float | None = None
    synthetic: bool = False

@dataclass(frozen=True)
class FrameResult:
    pose33: tuple[Landmark, ...] | None
    hands: tuple[tuple[Landmark, ...], ...] = ()
    track_id: int | None = None
    meta: dict[str, object] | None = None
```

定位与边界（复审修订）：

- `FrameResult`/`Landmark` 是**兼容边界上的结果容器**，服务实时 `annotate()`、单帧调试和后端适配输出，本身是合理抽象，不是要否定它。
- **硬约束：不得进入序列提取热路径。** 序列层（见下一节 `extract_landmark_series`）必须直接返回 `landmarks[T,33,4]` 和 `valid_mask[T,33]` 的 numpy 数组，不得逐帧构造 33 个冻结对象——否则在批量/长视频下会产生大量小对象分配，明显慢于 numpy，且与现有 `(T,33,4)` 契约割裂。边界容器用完应立刻摊平成 numpy。
- `source`/`model_name`/`keypoint_source`/`confidence_kind` 等在一次提取内是常量，只进 `meta`，**不逐帧存**，避免数据模型膨胀。
- `pose33` 是兼容容器，不是能力承诺。
- YOLO 映射出来的缺失点必须 `synthetic=True` 且 `visibility=0.0`。
- `confidence` 和 `visibility` 分开记录，避免把 YOLO keypoint confidence 当成 MediaPipe visibility 直接复用。
- 手部继续由 MediaPipe Hand 提供，第一版无需强行纳入 YOLO 后端。

### 2. 序列提取层

新增统一序列提取函数，优先服务模板、规则、技术评估和 batch：

```python
def extract_landmark_series(
    video_path: Path,
    *,
    backend: str,
    pose_variant: str = "full",
    yolo_model: str | None = None,
    running_mode: str = "video",
    start_frame: int | None = None,
    end_frame: int | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return landmarks[T,33,4], valid_mask[T,33], meta."""
```

要求：

- MediaPipe 旧路径输出与当前 `extract_pose_raw()` 等价。
- YOLO body-only 输出 `(T,33,4)`，只真实填充 COCO 对应点，缺失点 validity 为 false。
- **第 4 通道与 valid_mask 契约必须写死（复审强调）**：
  - `landmarks[...,3]` 的语义按 `confidence_kind` 区分——MediaPipe 存 visibility，YOLO 存 keypoint confidence，二者分布不同，不可直接套用同一阈值。
  - `valid_mask[T,33]` 是**唯一**的“此点可用”判据，由序列提取层负责灌入（缺失点、低置信点、合成点一律置 false）。
  - 下游（normalizer、规则、tech_eval）的有效性判断必须迁移到读 `valid_mask`，不能继续写死 `lm[idx,3] >= 0.5`。这是防止 YOLO 缺失点混进 DTW 和规则判定的关键闸门，S1/S4 落地时必须验证。
  - **阈值策略必须落地为可配置项（三审新增）**：`confidence_kind` 只命名不够，必须配套一组真正消费它的字段，写入 meta/config：
    - `validity_policy ∈ {visibility_thr, confidence_thr, synthetic_false}`——声明这条 mask 是按 visibility 阈值、按 confidence 阈值、还是因合成点强制 false 得来的。
    - `valid_conf_thr: float`——该 policy 对应的数值阈值。
    - **时序约束**：YOLO 侧的 `valid_conf_thr` 在 S3 标定完成前是「待标定参数」。在 S3 之前，**禁止**用未校准的默认值让 YOLO 数据进入正式规则/tech_eval 判定（可进预览/调试，但不得产出对外评分）。MediaPipe 侧沿用现有 `0.5` 视为已标定。
    - 这条同时消解了 S1↔S3 的时序裂缝：S1 就要求下游改读 `valid_mask`，但 YOLO mask 的含义要到 S3 才可信——用 `validity_policy` 把“可信/待标定”显式标出来，避免下游误把未标定 mask 当真。
- 多线程处理时，每个 segment 独立 backend 实例，tracker state 不跨视频或 segment 复用。
- metadata 至少包含 `backend`、`model_name`、`running_mode`、`confidence_kind`、`frame_count`、`fps`、`layout`。

### 3. 特征布局注册表

不要只加一个 normalizer。应把 feature layout 做成显式注册对象：

```python
@dataclass(frozen=True)
class FeatureLayoutSpec:
    name: str
    source_indices: tuple[int, ...]
    shape: tuple[int, int]
    mirror_pairs: tuple[tuple[int, int], ...]
    joint_names: tuple[str, ...]
    default_baseline: float | None
    # required_landmarks: frozenset[int]  # 延后到 S4 规则分级时再加，S1 不引入未使用字段
```

> 字段范围说明（复审）：`required_landmarks` 只有在 S4“规则/技术评估分级”才会被消费，S1 引入属于提前建模。S1 先不加，等 S4 再补，避免布局 spec 带着无人使用的字段。

首批布局：

| layout | shape | 用途 | 默认状态 |
|---|---:|---|---|
| `pose33_v3` | `(22,2)` | 旧 MediaPipe 模板，BlazePose 11..32 | 默认保留 |
| `body_core_v1` | `(12,2)` | YOLO/MediaPipe 共享躯干四肢核心点 | 显式 opt-in |

`body_core_v1` 索引：

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

强制规则：

- `mirror_pose_features()` 必须按 layout 的 `mirror_pairs` 工作，不能继续假设 22 点。
- 缺帧补零必须按 layout shape 生成，不能写死 `(22,2)`。
- 关节误差统计必须从 layout 的 `joint_names` 取名。
- 不同 layout 默认报错，不允许静默比对。
- `body_core_v1` baseline 必须重新标定，不能沿用 `2.0`。

### 4. YOLO 映射边界

COCO17 到 Pose33-like 的真实映射：

| BlazePose | COCO | 名称 |
|---:|---:|---|
| 0 | 0 | nose |
| 2 | 1 | left_eye 近似 |
| 5 | 2 | right_eye 近似 |
| 7 | 3 | left_ear |
| 8 | 4 | right_ear |
| 11 | 5 | left_shoulder |
| 12 | 6 | right_shoulder |
| 13 | 7 | left_elbow |
| 14 | 8 | right_elbow |
| 15 | 9 | left_wrist |
| 16 | 10 | right_wrist |
| 23 | 11 | left_hip |
| 24 | 12 | right_hip |
| 25 | 13 | left_knee |
| 26 | 14 | right_knee |
| 27 | 15 | left_ankle |
| 28 | 16 | right_ankle |

不可用点：

- 1、3、4、6：眼睛细分点。
- 9、10：嘴角。
- 17..22：手指。
- 29..32：脚跟、脚尖。

要求：

- 不可用点不得伪装成有效点。
- 眼睛近似点只能作为低风险显示或方向辅助，不能用于严肃规则阈值。
- YOLO-only 下的规则和技术评估必须能输出 `skipped_missing_landmarks`。

## 分阶段实施计划

### S0. 决策 Spike：先决定值不值得做

目标：不改主架构，先确认 YOLO 是否值得进入工程迁移。

任务：

- 明确许可路径：AGPL-3.0 是否可接受，是否需要 Enterprise license，是否改用替代模型。
- 固定实验环境：Python、torch、ultralytics、模型文件、CPU/GPU、CUDA。
- 选 3 到 5 段视频作为 `docs/yolo_eval_samples.json`，包含正面、侧面、长视频、遮挡或多人边界样本。
- 写一个临时 spike 脚本导出 YOLO keypoints、track id、FPS、失败帧率，不接入主代码。
- 同样样本导出当前 MediaPipe baseline。
- **核心指标降级评估（复审新增，必做）**：对照 `analysis/tech_eval.py` 列出 YOLO-only（COCO17）下会失效或退化的指标清单，并给业务结论：
  - 重心（`eval_cog_side`/`eval_cog_com`）：缺脚跟脚尖后，支撑面边界退化为单踝，量化精度损失。
  - 发力顺序（`eval_force_sequence`）：缺脚跟脚尖后，蹬地检测与脚旋转判定基本失效，确认是否可接受。
  - 后手贴近/护手：缺嘴角后无法判定。
  - 给出明确判断：**“YOLO-only 降级后的评分能力是否满足业务最低要求”**——这是 go/no-go，不是后续阶段的优化项。
- **明确本次迁移的首要动机（三审新增，决定 spike 重点）**：在 S0 一开始就写死“为什么要上 YOLO”，因为它决定 spike 测什么：
  - 若动机主要是**实时 FPS / 端侧吞吐**：S0 必须直接 spike **真实实时路径**——`apps/main.py --source 0`、`annotate()` 全流程（pose + 绘制 + 动作标签 + V 手势）、Hands 开/关两种配置下的端到端 FPS，而**不是**只用临时脚本导 keypoint 的裸推理 FPS（裸推理快不代表 `annotate()` 全链路快）。
  - 若动机主要是**离线模板闭环 / 批处理吞吐**：维持本计划“离线先行”的阶段顺序，实时入口按 S5/S6 处理即可。
  - 两种动机都不成立（既不提速也不解决精度）时，应直接进入“替代方案决策”。

- **预注册量化阈值（三审新增，必做）**：S0 的 go/no-go 不允许用“明显劣于 / 业务可接受 / 可用于评分”这类主观词。必须在动手前把下列阈值填成数字写入 `docs/yolo_baseline_report.md` 的头部，事后只对数字，不对感觉：

  | 指标 | 预注册阈值（示例占位，实施前由业务确认填数） | 判定方向 |
  |---|---|---|
  | 实时端到端 FPS 提升比例（仅实时动机时） | ≥ `__%`（相对 MediaPipe full） | 低于则实时动机不成立 |
  | 关键点失败/漏检帧率 | ≤ `__%` | 高于则 no-go |
  | 单人标准动作关键点抖动（与 MediaPipe 的逐帧位移差） | ≤ `__`（归一化坐标） | 高于则 no-go |
  | 模板分数与 MediaPipe 路径相关性（同样本） | ≥ `__`（如 Pearson r） | 低于则模板匹配不可替代 |
  | 初始化 / 模型加载耗时 | ≤ `__s` | 仅记录，超标记风险 |

  注：上表是 S0 自身的 go/no-go 门槛；模板匹配与评分一致性的更细阈值在 S3 标定时进一步收紧（见 S3）。

验收：

- 产出 `docs/yolo_baseline_report.md`，记录样本、命令、版本、硬件、FPS、失败帧率、主要差异。
- **报告头部已填入上表的全部预注册数字阈值，且每条 go/no-go 结论都引用具体数字而非主观判断。**
- 许可结论写清楚：可用、不可用、需采购、或进入替代方案调研。
- YOLO 输出 shape、confidence、多人场景主目标选择可被稳定解析。
- **核心指标降级清单与业务可接受性结论已写入报告。**
- **若动机为实时 FPS：报告含 `annotate()` 全链路（含 Hands 开/关）的端到端 FPS，而非仅裸推理 FPS。**

停止条件（全部对照预注册数字，不对感觉）：

- 许可不允许目标部署。
- CPU/GPU 端到端性能未达预注册 FPS 阈值（实时动机下尤其看 `annotate()` 全链路，不看裸推理）。
- 单人标准动作关键点抖动或漏检超过预注册阈值，且无可接受修正路径。
- **（复审新增）YOLO-only 降级后，重心与发力顺序等核心评分指标的精度损失业务不可接受，且 Hybrid 补点的性能代价又抵消了 YOLO 收益——此时应停在 S0，直接进入“替代方案决策”，不进 S1。**

### S1. 布局隔离、旧路径回归与 valid_mask 迁移

目标：在仍然只跑 MediaPipe 的情况下，把 layout 硬编码拆开，并把有效性判据迁到 `valid_mask`。**S1 只做无论 YOLO 成不成都有价值的工作**；任何只为 YOLO 服务的东西都下移到 S2。

> 三审拆层说明：S1 只有在 S0 通过后才启动，但若把 `body_core_v1` normalizer 和 baseline 放进 S1，会在“S0 仍可能毙掉整件事”时提前为一个还没人用的布局建模。故 S1 拆成两层——**S1a 防御性修复（无条件有价值）** 与 **S1b 契约迁移（无条件有价值）**；`body_core_v1` 的 normalizer / baseline **下移到 S2**，等真接 YOLO 离线闭环时再引入。

#### S1a. 防御性修复（无条件有价值）

- 新增 `FeatureLayoutSpec` 注册表骨架，**先只注册 `pose33_v3`**（`body_core_v1` 留到 S2 注册）。
- 保留 `pose33_v3` 默认行为完全不变。
- 改造 `_extract_pose_features()`、`mirror_pose_features()`、双模板误差分析和缺帧 fallback，使其**按 layout shape 取值**，不再写死 `(22,2)`——但此步只是参数化，默认仍走 22 点路径。
- **（复审新增，必做）修正 `core/action_compare.py` 的 `_select_representative_cycle()`（约 337-343 行）布局守卫**：当前 `if features.shape[1:] != (22, 2): return features` 会让非 22 点布局**静默跳过周期裁切**，DTW query 退化成整段模板（不报错、最难查）。改为按 layout shape 判定，使任意已注册布局都能正常裁切代表周期。
- 统一 artifact root：建议顶层 `models/`、`templates/`、`outputs/`，同步修 `.gitignore`。

#### S1b. valid_mask 契约迁移（三审提级为专项，S1 真正的大头）

> 为什么单列：`analysis/tech_eval.py:104` 的 `_valid()` 与 `core/rule_scoring.py:93` 的 `_valid_frame()` 都是 `_lm_vis(lm,i) >= thr` 风格，且被 `eval_cog_side`/`eval_cog_front`/`eval_cog_com`/`eval_force_sequence` 及全部 `_rule_*` 函数体反复调用，调用面是**几十处**。这是一次又宽又机械、极易引入回归的重构，体量很可能超过 YOLO adapter 本身，必须当成独立任务排期，不能塞进一条 bullet。

- 在序列提取层产出 `valid_mask[T,33]`，MediaPipe 路径下由现有 `visibility >= 0.5` 规则灌入（`validity_policy=visibility_thr`、`valid_conf_thr=0.5`）。
- 把 `tech_eval._valid` / `rule_scoring._valid_frame` 及其全部调用点改为读 `valid_mask`，不再各自 `lm[idx,3] >= thr`。
- **逐位等价回归（验收硬指标）**：同一 fixture 下，迁移后 MediaPipe 旧 `>=0.5` 路径与新 `valid_mask` 路径必须产出**逐位一致**的规则扣分、tech_eval 状态与关节误差统计。不一致即视为迁移失败，不允许“差不多”。

#### 通用要求

- 模板 metadata 增加 `backend`、`model_name`、`feature_layout`、`normalizer_version`、`confidence_kind`、`validity_policy`、`valid_conf_thr`。
- 不同 layout 比对时报清晰错误，不允许静默比对。

验收：

```powershell
.\.venv\Scripts\python.exe -m py_compile .\core\vision_pipeline.py .\core\pose_features.py .\core\action_compare.py .\core\rule_scoring.py .\analysis\tech_eval.py .\apps\main.py .\apps\app_ui.py .\apps\make_template.py .\apps\match_template.py .\batch\batch_dual_compare.py .\batch\batch_export_skeleton.py .\batch\batch_tech_eval.py
```

> 注意：`py_compile` 只证明“能编译”，对“行为不变”零保证。S1 改的是最热的几个函数，**必须**叠加下面的 golden 回归才算验收通过。

新增测试建议：

- `tests/test_pose33_v3_golden.py`（三审新增，**S1 最重要的一条**）：用固定 fixture（保存好的 landmark/feature 序列或一段短 fixture 视频），断言改造前后 **① 单模板 `compare_video_to_template` 分数、② 双模板 `compare_video_to_dual_templates` 的 `combined_percent` 与各视角分数、③ tech_eval 各指标的 `status`/关键 `detail`** 均不漂移（数值用容差极小的 `allclose`，状态用精确相等）。这是“默认行为不变”原则的唯一硬保证。
- `tests/test_valid_mask_migration.py`（三审新增）：对照 fixture 验证“旧 `>=0.5` 路径 == 新 `valid_mask` 路径”逐位一致（规则扣分、tech_eval 状态、关节误差）。
- `tests/test_layout_shape_param.py`：验证 `_extract_pose_features`/`mirror_pose_features`/缺帧 fallback 在 `pose33_v3` 下行为不变，且 shape 取值已参数化（不再出现字面量 `22`/`(22,2)`）。
- `tests/test_representative_cycle_layout.py`（复审项保留）：构造非 22 点（如 `(T,12,2)`）输入，验证 `_select_representative_cycle()` 不再静默整段返回，而是正常按周期裁切。
- `tests/test_artifact_roots.py`：验证入口使用统一模型目录。

### S2. 最薄 YOLO Body-only 离线闭环

目标：只接离线模板链路，不碰 UI 默认、不做 Hybrid。

任务：

- 新增轻量 YOLO adapter，懒加载 `from ultralytics import YOLO`。
- **引入 `body_core_v1`（从 S1 下移）**：在 `FeatureLayoutSpec` 注册 `body_core_v1`(12,2) 布局，新增其 normalizer，并让 MediaPipe 也能生成该布局模板（用于 S3 三方对比）。这一步放在 S2 是因为它**只在真接 YOLO 时才有消费者**，放进 S1 属提前建模。`body_core_v1` 的 baseline 此时先用占位值，正式标定在 S3。
- 实现 COCO17 解析、Pose33-like 映射、valid mask、`confidence_kind`/`validity_policy`/`valid_conf_thr`。YOLO 侧 `valid_conf_thr` 此阶段为**待标定占位值**，按首页前提仅供预览/调试与 S3 标定使用，不得产出对外评分。
- **多人闸门（三审新增，必做）**：`batch_tech_eval` / `batch_dual_compare` 跑的是**学员视频**，教练、镜面反射、旁边路人入镜很常见，YOLO“最大框/最高分取单人”很可能稳定选错实例。这是**正确性风险，不是鲁棒性锦上添花**。MVP 至少做到：
  - 检测到 `num_persons > 1` 时，在 `meta` 写入 `multi_person_detected=True` 及检出人数；
  - tech_eval / batch 链路对该视频**拒绝出分或降级为「需人工复核」**，不得静默选最大框当唯一目标；
  - 单人场景（`num_persons <= 1`）才走“最大框/最高分取单人”的简化策略。
- **track 延续策略推后**：中心最近、tie-break 等完整多人鲁棒策略，等真有多人样本时再做；标准训练视频基本单人，S2 不必先上完整策略。
- **track 边界语义写清（复审强调，非逻辑冲突）**：`model.track(..., persist=True)` 状态只在单个视频或 segment 内有效，不跨任务复用。由于序列提取按帧段并行 + warm-up overlap，**段边界处 track_id 必然重置**；因此“延续 track_id”一旦启用，必须明确：warm-up 区如何接续、跨段如何对齐同一单人目标。S2 单人 MVP 下可暂不依赖 track 延续，但该语义须在 metadata 与代码注释中写明，避免后续多人扩展时踩坑。
- 先只接 `apps/make_template.py`、`apps/match_template.py` 或 `batch/batch_dual_compare.py` 中的一条离线闭环。
- YOLO 只允许 `body_core_v1`。

验收：

- `tests/test_body_core_layout.py`（从 S1 下移）：验证 `body_core_v1` 的 shape、mirror pairs、joint names、layout mismatch。
- `tests/test_template_metadata.py`：验证旧模板兼容，新模板 metadata 完整（含 `validity_policy`/`valid_conf_thr`）。
- `tests/test_yolo_landmark_mapping.py`：验证 33 长度、COCO 对应点 confidence 传递、嘴角/手指/脚跟/脚尖无效。
- `tests/test_yolo_backend_contract.py`：验证无人帧、空结果、track id 重置，以及**多人帧触发 `multi_person_detected` 并被拒绝/降级**。
- 可生成 YOLO `body_core_v1` 模板并匹配同一视频。
- MediaPipe 旧模板仍能按 `pose33_v3` 路径比对。

不做：

- 不做 UI 后端选择。
- 不做 MediaPipe Pose 补点。
- 不切默认后端。

### S3. 标定和基准报告

目标：确定 `body_core_v1` 是否可用于评分，以及 baseline 应该是多少。

任务：

- 同一批样本分别跑：
  - MediaPipe `pose33_v3`
  - MediaPipe `body_core_v1`
  - YOLO `body_core_v1`
- 统计模板分数、avg_cost、有效帧率、失败帧率、FPS、初始化耗时。
- 重新标定 `body_core_v1` 的 baseline，并标定 YOLO 侧 `valid_conf_thr`（把 S2 的占位值替换为标定值，此后 YOLO 才允许产出对外评分）。
- 输出差异样例：误判高、误判低、未评估比例高的片段。
- 预注册 pass/fail 判定口径：明确使用业务人工标签、MediaPipe full 阈值，还是模板分数阈值；阈值、标签来源和样本范围必须在跑数据前写死。
- **预注册「可用于评分」的数字判据（三审新增，必做）**：在 `docs/yolo_body_core_calibration.md` 头部先填死阈值，再用数据对照，禁止用“可用/不建议”这类主观结论收尾：

  | 指标 | 预注册阈值（实施前由业务确认填数） | 含义 |
  |---|---|---|
  | YOLO 与 MediaPipe `body_core_v1` 模板分数相关性 | ≥ `__`（Pearson/Spearman r） | 低于则 YOLO 不可替代模板匹配 |
  | pass/fail 判定一致率（口径/标签来源预注册后，同样本对照 MediaPipe full） | ≥ `__%` | 低于则不可用于对外评分 |
  | 模板分数 MAE / 偏差 | ≤ `__` | 量化分数漂移上限 |
  | 未评估（skip）比例 | ≤ `__%` | 高于则可评估覆盖不足 |
  | 失败/漏检帧率 | ≤ `__%` | 与 S0 阈值衔接 |

验收：

- 产出 `docs/yolo_body_core_calibration.md`，头部含上述全部预注册数字，结论逐条引用数字。
- 报告明确记录 pass/fail 判定口径、阈值/标签来源和样本范围，不允许事后改口径解释结果。
- `body_core_v1` baseline 写入 layout spec 或配置，不再写死全局 `2.0`。
- YOLO 侧 `valid_conf_thr` 已标定并写入 config，S2 的占位值被替换。
- **明确结论（必须对照数字阈值，不能是感觉）**：仅用于预览、可用于模板匹配、可用于 skip-aware partial eval、或不建议使用——四选一并附数字依据。

### S4. 规则和技术评估分级

目标：YOLO-only 能诚实输出能评估什么，不能评估什么。

任务：

- `Rule` 增加 `required_landmarks`、`required_capabilities`。
- **规则结果补结构化状态字段，不推倒重来（复审收敛）**：`rule_scoring.py` 已有 `RuleScore`/`RuleViolation` dataclass，只是三态靠拼中文 `detail` 字符串区分。改法是在 `RuleViolation` 上加 `state` + `skip_reason` 两个字段：
  - `state ∈ {evaluated, skipped}`；
  - `skip_reason ∈ {missing_landmarks, low_confidence, insufficient_valid_frames}`（仅 skipped 时有值）。
  - `detail` 仍可保留中文供 UI 展示，但下游判断一律读 `state`/`skip_reason`，不再解析字符串。
- `analysis/tech_eval.py` 每个指标声明 `required_landmarks` 和 `missing_landmarks`。
- YOLO-only 只启用 COCO17 足够的指标，如肘角、膝角、站距、鼻尖与手腕高度等；阈值必须重新标定。
- mouth、index/pinky、heel、foot_index 相关指标默认 `无法判定`（含重心支撑面、发力顺序蹬地/脚旋转、后手贴近、护手位置）。

验收：

- `tests/test_rule_availability.py` 覆盖规则 skipped reason。
- `tests/test_tech_eval_contract.py` 验证每个指标都有 `status`、`reason`、`required_landmarks`、`missing_landmarks`、`backend`。
- YOLO-only CSV/JSONL 中未评估原因清晰，不把不可评估当合格或不合格。
- MediaPipe 旧技术评估回归不退化。

### S5. 扩展入口和 Hybrid 决议

目标：在离线闭环、标定、分级评估都稳定后，才扩到完整入口。

顺序：

1. `batch_dual_compare.py`、`batch_export_skeleton.py`、`batch_tech_eval.py` 增加 backend/layout 参数和输出 metadata。
2. `apps/main.py` 的 YOLO 实时预览入口已由 Issue #25 / `docs/yolo_gpu_recheck_report.md`
   supersede：#23 CUDA 修复后真实 GPU 复测仍 no-go，当前不实现 `--backend yolo` /
   `--feature-layout body_core_v1`，保持既有 MediaPipe CLI 默认路径；未来若重新打开，
   必须先让 Hands 关 / Hands 开实时阈值和 YOLO raw / 抖动阈值全部达标，并另开实现子任务。
3. `apps/app_ui.py` 的 YOLO 后端选择与规则完整度提示已由 Issue #26 /
   `docs/yolo_gpu_recheck_report.md` supersede：#23 CUDA 实测 no-go 且 #25 仍不实现时当前不实现，
   保持既有 MediaPipe UI；未来若重新打开，必须先满足 GPU 复测前置条件并等待 #25 后续实现子任务落地。
4. Hybrid 已由 Issue #27 / `docs/yolo_gpu_recheck_report.md` supersede：当前不实现
   `yolo_body_mp_pose_supplement`，不保留半成品 runtime 路径；若未来要恢复，必须另开实现子任务并先满足
   #27 的 CUDA 有效复测、Hybrid 专用 benchmark、补点指标恢复证据等触发条件。

> Hybrid 性质提醒（复审）：靠 MediaPipe 补脚/补脸，本质是把整个 PoseLandmarker 又跑一遍，YOLO 的性能收益基本被抵消。在“重心/发力顺序必须要脚部”的业务前提下，应在 S0 就初判 **“Hybrid 是否任何时候都不如直接用 MediaPipe full”**；若 S0 已判定 Hybrid 不划算，则本步直接删除，不实现。

验收命令示例：

```powershell
.\.venv\Scripts\python.exe apps/main.py --source 0
.\.venv\Scripts\python.exe apps/make_template.py --backend yolo --feature-layout body_core_v1 --video input.mp4
.\.venv\Scripts\python.exe apps/match_template.py --backend yolo --feature-layout body_core_v1 --template template.npz --video input.mp4
.\.venv\Scripts\python.exe -m batch.batch_dual_compare --backend yolo --feature-layout body_core_v1 --standard_dir 标准样本 --student_dir 学员样本
.\.venv\Scripts\python.exe -m batch.batch_export_skeleton --backend yolo --source_dir 标准动作视频--分解版
.\.venv\Scripts\python.exe -m batch.batch_tech_eval --backend yolo --video_dir 学员样本
```

输出字段检查：

- CSV/JSONL/NPZ meta 必须包含 `backend`、`model_name`、`feature_layout`、`confidence_kind`。
- 规则和技术评估输出必须包含未评估原因和完整度统计。
- CLI `apps/main.py` 在 #25 no-go 分支下不得出现 `--backend yolo` 实时入口或 YOLO realtime runtime。
- UI `apps/app_ui.py` 在 #26 no-go 分支下不得出现 YOLO 后端选择、`body_core_v1` 布局选择、
  YOLO-only 完整度提示或未标定评分入口。

### S6. 默认切换决策

目标：决定是否、在哪里默认使用 YOLO。

当前 S6 决策见 `docs/yolo_default_switch_decision.md`（Issue #28）：**全部不切默认，仅保留离线 / 实验入口**。

决策摘要：

- 实时预览（CLI / UI）：#23 CUDA 修复后真实 GPU 复测仍 no-go，#25 / #26 仍关闭，默认继续 MediaPipe。
- 模板匹配：#10 结论仅预览，J1 corr=0.280、J4 一致率=0.50，默认继续 `pose33_v3` / MediaPipe。
- 规则 / 技术评估：COCO17 缺嘴角、脚跟脚尖、手指，默认继续 MediaPipe full。
- Hybrid：#27 已决议不实现 `yolo_body_mp_pose_supplement`。

未来重新评估默认切换前必须满足：

- 许可结论适配目标部署形态，闭源 / 商业部署需 Enterprise 或替代模型方案。
- #23 GPU 复测在 CUDA-enabled 环境下 6/6 样本有效，Hands 关 / Hands 开 FPS 比达到 1.30 / 1.20，且 YOLO raw FPS 与抖动阈值达标。
- 标定 / 回归报告通过，误判样例与未评估比例可接受。
- CLI / UI / batch 的已授权入口都能显示结果来源与 `score_authorized`，并具备一键回滚配置。

## 替代方案决策

如果 P0 或 S3 不通过，不建议继续硬推 Ultralytics YOLO。替代路线：

| 路线 | 适用条件 | 风险 |
|---|---|---|
| 继续 MediaPipe 优化 | 当前精度可接受，主要瓶颈是工程结构和稳定性 | 性能提升有限 |
| Ultralytics Enterprise | YOLO 效果好，预算和授权可解决 | 商业条款需确认 |
| RTMPose/MMPose | 需要更宽松开源许可或自部署控制 | 接入和模型转换成本更高 |
| ONNXRuntime/TensorRT | 已有可合法使用的 pose 权重，需要部署优化 | 仍需确认模型和训练代码许可 |
| 自训 Pose33-like 模型 | 业务长期依赖脚跟、脚尖、手部细节 | 数据和训练成本高 |

## 最小完成定义

迁移最小完成不是“所有入口都切 YOLO”，而是：

1. MediaPipe 旧路径默认可用，且**有 `pose33_v3` golden 回归**证明分数/状态不漂移（不只是 `py_compile` 通过）。
2. `FeatureLayoutSpec` 消除旧 `(22,2)` 硬编码风险。
3. 有效性判据已从 `lm[idx,3] >= thr` 迁到 `valid_mask`，且通过**旧路径与新路径逐位等价回归**。
4. YOLO body-only 可以完成一个离线模板闭环，且**多人场景被 `multi_person_detected` 拒绝/降级**，不静默选最大框。
5. `body_core_v1` baseline 与 YOLO `valid_conf_thr` 有标定报告，结论对照预注册数字阈值。
6. 规则和技术评估能结构化输出未评估原因（`state`/`skip_reason`）。
7. 输出 metadata 可追踪 `backend`、`model`、`layout`、`confidence_kind`、`validity_policy`、`valid_conf_thr`。
8. 许可、性能、精度都有书面结论，且 S0/S3 的 go/no-go 均引用预注册数字而非主观判断。

## 审批后删减/后置项

| 原计划项 | 审批处理 |
|---|---|
| 第一阶段完整拆 `PoseBackend` | 后置，先做序列提取和 layout 注册 |
| 完整 capability lattice | 简化为首批 metadata 字段 |
| MediaPipe 默认逐步迁到 `body_core_v1` | 改为显式 opt-in，旧默认不变 |
| P4 实现 Hybrid 补点 | 已由 Issue #27 / GPU 复测报告 supersede，当前不实现；未来需另开实现子任务并满足触发条件 |
| CLI/UI/batch 同时接入 | 改为先离线模板闭环，再 batch，再 CLI/UI |
| 默认实时链路候选 `yolo_body_mp_hands` | 保留为未来可能，不进 MVP |

## 审批保留项

- 许可前置。
- 不伪造完整 Pose33。
- 缺失点不进入共享模板布局。
- 模板 metadata 完整记录后端和布局。
- 技术评估必须纳入迁移范围。
- 默认后端最后再决策。
