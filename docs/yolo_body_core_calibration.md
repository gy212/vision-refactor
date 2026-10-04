# YOLO 迁移 S3 标定与基准报告：body_core_v1（Issue #10）

来源：`docs/yolo_migration_plan_optimized.md`（三审定稿）/ `docs/yolo_migration_issues.md` Issue #10
日期：2026-05-31
阶段：S3（标定与基准报告）　依赖：#8（body_core_v1 离线闭环）#9（多人闸门）　阻塞：#11（评分阈值部分）
关联脚本：`analysis/calibrate_body_core.py`（组合既有生产函数做三路对照，不改主代码默认行为）
关联原始数据：`outputs/calib_body_core/calibration_summary.json` / `*.csv` / `raw_cache/*.npz`

> **方法论铁律（与 S0 报告一致）**：本报告严格「**先预注册阈值与判定口径、后对照数据**」。
> 第二、三节的预注册数字与 pass/fail 口径在跑标定数据**之前**写死，下文每条结论逐条引用具体数字，
> 不使用「可用/不建议」这类主观措辞，也不允许事后改口径解释结果。

---

## 一、任务目标（Issue #10）

确定 `body_core_v1` 是否可用于评分、baseline 取值，以及标定 YOLO 侧 `valid_conf_thr`
（把 #7 的占位值 0.5 替换为标定值；是否允许出分由本报告 go/no-go 判据决定）。三路对照：

1. MediaPipe `pose33_v3`（旧默认布局，22 点，已标定 baseline=2.0，作参照基准）；
2. MediaPipe `body_core_v1`（12 点共享布局，同后端换布局）；
3. YOLO `body_core_v1`（12 点共享布局，COCO17 映射，换后端同布局）。

---

## 二、预注册 pass/fail 判定口径（跑数据前写死）

> 以下口径在跑标定数据前确定，不允许事后修改以迎合结果。

- **判定口径**：采用**模板分数阈值口径**（template-score threshold），不使用业务人工标签
  （本批样本无逐对人工「相似/不相似」标注），也不使用 MediaPipe full tech_eval 阈值
  （full eval 是另一条评估链路，本报告只标定 body_core_v1 模板匹配）。
- **pass/fail 定义**：对每个 (模板, 视频) 对，以 **`pose33_v3` 分数 ≥ 0.55** 记为 `pass(相似)`，
  否则 `fail`。`pose33_v3` 是已标定的生产参照基准，故用它定义 ground-truth 口径；
  `body_core_v1`（MP / YOLO）的 pass/fail 用**相同阈值 0.55** 施加在各自分数上，
  与 pose33_v3 比对得到「pass/fail 一致率」。
- **阈值来源**：pose33_v3 baseline=2.0 来自生产既有标定；pass 阈值 0.55 为本报告预注册值
  （对应 avg_cost ≈ baseline，即「相似度过半」线）。
- **标签来源**：无人工标签；ground-truth 口径 = pose33_v3 分数阈值（已声明，固定）。
- **样本范围**：标定统计集 = **单人样本**（YOLO 多人闸门 `review_required=False`）：
  `std_front`、`std_side_long`、`punch_front`、`punch_side`（2 正面 + 2 侧面）。
  学员样本 `student_1`、`student_4_long` 被多人闸门排除出标定集（仅用于第七节闸门展示），
  因为多人 = 正确性风险，闸门契约要求拒绝出分，故也不得进入标定。
- **匹配口径**：分数由**跨视频成对匹配**产生（视角组内：每段活跃区间为模板 query，
  与组内其它样本整段做 subsequence DTW）。self-match（query 是 seq 子段）恒为满分、无方差，
  仅作健全性检查，不计入相关性 / baseline 标定的「跨样本对」统计。

---

## 三、预注册数字判据（跑数据前写死）

> 数值方向在跑数据前固定。判据用于「body_core_v1 是否可用于模板匹配」的 go/no-go。
> 数值依据本批 CPU 单人样本规模（4 段 / 8 对 / 4 跨样本对）设定，标注 `预注册（待业务确认）`。

| # | 判据 | 预注册阈值 | 判定方向 |
|---|---|---|---|
| J1 | 分数相关性下限 corr(MP body_core, pose33) | ≥ 0.70 | 低于则 body_core 布局不跟随 pose33 → no-go |
| J2 | 分数相关性下限 corr(YOLO body_core, MP body_core) | ≥ 0.80 | 低于则 YOLO 后端不跟随 MediaPipe → 仅预览 |
| J3 | 分数 MAE 上限 MAE(YOLO body_core, MP body_core) | ≤ 0.05 | 高于则 YOLO 分数偏差过大 → 仅预览 |
| J4 | pass/fail 一致率下限（YOLO body_core vs pose33） | ≥ 0.75 | 低于则不可作 pass/fail 判定 |
| J5 | body_core 有效帧率下限（标定阈值下，单人样本最小值） | ≥ 0.70 | 低于则 skip 过多、不可模板匹配 |
| J6 | skip 比例上限（= 1 − 有效帧率，单人样本最大值） | ≤ 0.30 | 高于则不可模板匹配 |
| J7 | 失败帧率上限（漏检/无人帧，单人样本） | ≤ 0.02 | 高于则后端不稳定 → no-go |

baseline 标定方法（预注册，跑数据前确定）：
> `body_core_v1` 的 DTW avg_cost 尺度与 pose33_v3 不同，故按**尺度对齐**标定，使两布局分数同尺度、
> 可共享 pass/fail 阈值：
> `baseline_bodycore = baseline_pose33(2.0) × median(bodycore_avg_cost) / median(pose33_avg_cost)`
> （在单人跨样本对上统计，self 对不计入）。

YOLO `valid_conf_thr` 标定方法（预注册）：
> 在网格 `{0.2,0.3,0.4,0.5,0.6,0.7}` 上扫描，选**同时满足 J2/J3（相关性最高、MAE 最低）且 J5/J6
> （有效帧率 ≥ 0.70）** 的最小可接受阈值；并列时取使 corr 最大、MAE 最小者。

---

## 四、实验环境（可复现）

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.9（`.venv`） |
| numpy | 2.4.1 |
| opencv-python | 4.13.0 |
| mediapipe | 0.10.31 |
| ultralytics | 8.4.57 |
| torch | 2.12.0+cpu（`cuda_available=False`，CPU 推理） |
| MediaPipe 模型 | `models/pose_landmarker_full.task`（变体 `full`） |
| YOLO 模型 | `models/yolo11n-pose.pt`（COCO17） |
| 样本清单 | `docs/yolo_eval_samples.json`（6 段；标定统计仅用 4 段单人） |

复现命令（仓库根，`.venv`）：

```powershell
.\.venv\Scripts\python.exe -m analysis.calibrate_body_core `
  --samples docs/yolo_eval_samples.json `
  --yolo-model models/yolo11n-pose.pt `
  --pose-variant full `
  --calib-conf-thr 0.6 `
  --out outputs/calib_body_core
```

> 每段每后端只推理一次并缓存到 `outputs/calib_body_core/raw_cache/*.npz`；
> `valid_conf_thr` 扫描只对缓存 conf 重新阈值化，不重复推理。

---

## 五、三路对照实测结果

### 5.1 baseline 标定（尺度对齐）

| 量 | 值 |
|---|---|
| median(pose33 avg_cost)（单人跨样本对） | 0.9117 |
| median(body_core avg_cost)（单人跨样本对） | 0.5847 |
| 尺度比 scale_ratio | 0.6413 |
| **标定 baseline_bodycore = 2.0 × 0.6413** | **1.2826** |
| 跨样本对数 n_cross_pairs | 4 |

→ `body_core_v1` baseline 标定为 **1.2826**，已写入 `core/feature_layout.py`
（`BODY_CORE_V1.default_baseline`），替换全局占位 `2.0`。

### 5.2 分数相关性 / MAE / 一致率（跨视频主指标，thr=0.6、baseline=1.2826）

> 主指标严格使用**跨视频 pair**（4 对），不含 self-match。self-match 只作健全性检查，
> 因为同视频子段匹配会产生 4 个平凡满分 1.0，混入相关性 / MAE / pass-fail 会虚高。

| 判据 | 预注册阈值 | 实测值 | 是否达标 |
|---|---|---|---|
| J1 corr(MP body_core, pose33) | ≥ 0.70 | **0.280** | ❌ 未达标 |
| J2 corr(YOLO body_core, MP body_core) | ≥ 0.80 | **0.980** | ✅ 达标 |
| J3 MAE(YOLO body_core, MP body_core) | ≤ 0.05 | **0.0468** | ✅ 达标（临界） |
| J4 pass/fail 一致率（YOLO body_core vs pose33） | ≥ 0.75 | **0.50** | ❌ 未达标 |
| J5 body_core 有效帧率最小值 | ≥ 0.70 | **0.875** | ✅ 达标 |
| J6 skip 比例最大值 | ≤ 0.30 | **0.125** | ✅ 达标 |
| J7 失败帧率（单人样本，漏检/无人帧） | ≤ 0.02 | MP 0.0 / YOLO 0.0 | ✅ 达标 |

补充：跨视频 pass/fail 一致率（MP body_core vs pose33）= 0.75；
（YOLO body_core vs MP body_core）= 0.75。

健全性检查（含 self-match 的 8 对，不作为验收判据）：corr(MP body_core, pose33)=0.838、
corr(YOLO body_core, MP body_core)=0.990、MAE(YOLO, MP body_core)=0.0234、
pass/fail 一致率（YOLO vs pose33）=0.75。该数字解释了为什么 self-match 不可混入主判据。

### 5.3 YOLO `valid_conf_thr` 扫描（标定核心证据）

| thr | corr(YOLO, MP body_core) | MAE | YOLO vs pose33 一致率 | YOLO vs MP 一致率 | 有效帧率均值 | 有效帧率最小 | skip 比例最大 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 0.105 | 0.1466 | 0.75 | 0.50 | 1.000 | 1.000 | 0.000 |
| 0.3 | 0.105 | 0.1466 | 0.75 | 0.50 | 1.000 | 1.000 | 0.000 |
| 0.4 | 0.105 | 0.1466 | 0.75 | 0.50 | 1.000 | 1.000 | 0.000 |
| 0.5 | 0.105 | 0.1466 | 0.75 | 0.50 | 0.992 | 0.970 | 0.030 |
| **0.6** | **0.980** | **0.0468** | **0.50** | **0.75** | **0.961** | **0.875** | **0.125** |
| 0.7 | 0.977 | 0.0501 | 0.50 | 0.75 | 0.905 | 0.784 | 0.216 |

**参数结论：YOLO `valid_conf_thr = 0.6`。** 依据预注册阈值扫描方法：
- thr 从 0.5→0.6 时 corr(YOLO, MP body_core) 由 0.105 跃升到 **0.980**、MAE 由 0.1466 降到 **0.0468**，
  J2/J3 在 0.6 才达标；
- thr=0.6 有效帧率最小值 **0.875 ≥ 0.70（J5）**、skip 最大 0.125 ≤ 0.30（J6）；
- thr=0.7 相关性略降（0.977）、MAE 略超阈值（0.0501），且有效帧率最小值掉到 0.784、skip 升到 0.216。

`valid_conf_thr=0.6` 已写入 `core/yolo_adapter.py`（`DEFAULT_YOLO_VALID_CONF_THR`），替换 #7 占位 0.5。
但该参数只允许用于预览 / 内部标定参考，**不等于评分授权**。

### 5.4 逐样本有效帧率 / 失败帧率 / 提取 FPS

| 样本 | 视角 | MP 失败率 | YOLO 失败率 | MP body_core 有效率 | YOLO body_core 有效率(thr=0.6) | MP 提取FPS | YOLO 提取FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| std_front | front | 0.0 | 0.0 | 1.000 | 1.000 | 50.5 | 20.7 |
| std_side_long | side | 0.0 | 0.0 | 0.599 | ~0.92 | 51.4 | 22.6 |
| punch_front | front | 0.0 | 0.0 | 1.000 | 1.000 | 49.2 | 17.8 |
| punch_side | side | 0.0 | 0.0 | 0.147 | ~1.00 | 32.6 | 12.2 |

> 关键观察：**侧面样本上 MediaPipe `body_core` 有效帧率显著偏低**（punch_side 0.147、
> std_side_long 0.599），因为 MediaPipe `visibility>=0.5` 把远侧被遮挡的肩/肘/腕/髋/膝/踝
> 判为不可见；而 YOLO `conf>=0.6` 对这些点仍给出较高置信（侧面有效率 ~0.92–1.0）。
> 这说明两后端 body_core 有效性策略差异主要体现在侧面遮挡，YOLO 在侧面**保留更多帧**。

---

## 六、差异样例（误判高 / skip 比例高的片段）

- **侧面 pass/fail 分歧（误判样例）**：side 视角跨样本对 `std_side_long ↔ punch_side`：
  pose33 分数 0.665 / 0.608（pass），而 MP body_core 0.243 / 0.659、YOLO body_core 0.249 / 0.547，
  在 0.55 线附近来回。MP 与 YOLO body_core **彼此较一致（跨视频 corr 0.980）**，但都与 pose33 在侧面分歧——
  这是 **body_core_v1 布局本身在侧面信息少（12 点、无脚跟脚尖、远侧遮挡）** 导致，非 YOLO 后端缺陷。
  这正是 J1 未达标、J4 一致率（YOLO vs pose33）只有 0.50 的来源。
- **MediaPipe skip 比例高片段**：`punch_side` 的 MP body_core 有效率仅 0.147（skip 0.853），
  远侧关节大面积 `visibility<0.5`；同片段 YOLO body_core 有效率接近 1.0。侧面动作若用 MediaPipe
  body_core 模板匹配，需注意大量帧被 skip。
- **多人 skip / 拒绝片段**：`student_1`（最多 4 人、23 帧多人）、`student_4_long`（最多 8 人、274 帧多人）
  被多人闸门 `review_required=True` 拒绝，未进入标定（见第七节）。

---

## 七、多人闸门复核（沿用 #9 判定，不改逻辑）

| 样本 | 声明多人 | YOLO 实测最大人数 | 多人帧数 | review_required | 进入标定集 |
|---|---|---:|---:|---|---|
| std_front | false | 1 | 0 | False | ✅ |
| std_side_long | false | 1 | 0 | False | ✅ |
| punch_front | false | 1 | 0 | False | ✅ |
| punch_side | false | 1 | 0 | False | ✅ |
| student_1 | unknown | 4 | 23 | **True** | ❌（闸门拒绝） |
| student_4_long | unknown | 8 | 274 | **True** | ❌（闸门拒绝） |

→ 多人闸门正确识别两段学员视频并拒绝出分；标定统计集严格只含 4 段单人样本，
与第二节预注册「样本范围」一致。

---

## 八、结论（四选一，附数字依据）

**结论：仅预览 / 内部标定参考，不得用于 body_core_v1 模板匹配对外出分，也不进 full tech_eval。**

逐条数字依据：

1. **J1 corr(MP body_core, pose33)=0.280 < 0.70** → body_core_v1 布局没有在跨视频模板分数上跟随生产参照基准，布局级 go/no-go 失败。
2. **J4 pass/fail 一致率（YOLO vs pose33）=0.50 < 0.75** → 不可作 pass/fail 判定，更不可进入用户报告 / 正式评分。
3. **J2 corr(YOLO body_core, MP body_core)=0.980 ≥ 0.80**、**J3 MAE=0.0468 ≤ 0.05**（thr=0.6）
   → YOLO 后端在 body_core_v1 上能较好跟随 MediaPipe，可用于预览、调试和后续扩大样本复核。
4. **J5 有效帧率最小 0.875 ≥ 0.70、J6 skip 最大 0.125 ≤ 0.30、J7 失败率 0.0 ≤ 0.02**（thr=0.6）
   → 检测覆盖不是主要阻塞；主要问题是 body_core_v1 与 pose33 参照分数不一致。
5. **不进 full tech_eval 的硬依据（S0 降级清单，结构性、与精度无关）**：COCO17 缺嘴角（9/10）、
   脚跟脚尖（29–32）、手指（17–22），重心（支撑面/分段质心）与发力顺序（蹬地/脚旋转）结构性失效。
   body_core_v1 只有 12 点（无脚部细分），无法承担 full tech_eval。

因此：

- ✅ **可保留参数落库**：baseline=1.2826、YOLO `valid_conf_thr`=0.6 替换旧占位，供预览 / 内部复核使用。
- ❌ **不得用于 `body_core_v1` 模板匹配对外出分**：`calibration_status=unvalidated`，模板 meta 额外标 `score_authorized=False`。
- ❌ **不可作 full tech_eval / 高精度评分**：侧面 pass/fail 分歧 + COCO17 结构性缺点。
- ➡️ **对 #11 的分流**：按 Issue #11 前置条件，本结论 = 「仅用于预览」分支 →
  #11 **只做 MediaPipe 侧结构化状态改造**，不启用 YOLO partial tech_eval 指标。

> 排除项说明：未选「可模板匹配」（J1/J4 未达标）；未选「skip-aware partial eval」
> （pass/fail 一致率不足，且 COCO17 结构性缺点使 partial eval 收益有限）；未选「不建议」
> （YOLO 与 MediaPipe body_core 的 J2/J3、有效帧率和失败率达标，仍有预览 / 调试价值）。

---

## 九、落库清单（验收对照）

| 项 | 落库位置 | 值 |
|---|---|---|
| body_core_v1 baseline | `core/feature_layout.py::BODY_CORE_V1.default_baseline` | 1.2826（替换 None/占位 2.0） |
| 闭环 baseline 取值 | `core/body_core_compare.py::BODY_CORE_V1_CALIBRATED_BASELINE` | 取自 layout default_baseline |
| YOLO valid_conf_thr | `core/yolo_adapter.py::DEFAULT_YOLO_VALID_CONF_THR` | 0.6（替换 #7 占位 0.5） |
| 标定状态 | `core/yolo_adapter.py::YOLO_CALIBRATION_STATUS` / `body_core_compare` | `unvalidated`（参数已落库，但不授权评分） |
| 评分授权 | `core/body_core_compare.py` 模板 meta | `score_authorized=False` |
| 回归测试 | `tests/test_s3_calibration.py` | baseline/阈值/状态/报告头部断言 |

> 标定范围声明：上述 baseline / 阈值只用于预览与内部复核，**不授权模板匹配对外出分**。
> 若目标硬件换为 GPU 或更换样本集，需在该硬件 / 样本上复跑本脚本复核数字（结构性降级结论无需复测）。

---

## 十、复现与原始数据

```powershell
# 三路对照 + 标定（缓存命中则不重复推理）
.\.venv\Scripts\python.exe -m analysis.calibrate_body_core --calib-conf-thr 0.6 --out outputs/calib_body_core

# 回归测试（标定值入库守卫）
.\.venv\Scripts\python.exe -m pytest tests\test_s3_calibration.py -q
```

原始输出：
- `outputs/calib_body_core/calibration_summary.json`（全量数字 + 逐对矩阵）
- `outputs/calib_body_core/calibration_pairwise.csv`（逐 (模板,视频) 对三路分数）
- `outputs/calib_body_core/calibration_conf_sweep.csv`（valid_conf_thr 扫描）
- `outputs/calib_body_core/calibration_per_sample.csv`（逐样本有效/失败帧率、FPS）
- `outputs/calib_body_core/calibration_multi_person_gate.csv`（多人闸门复核）
