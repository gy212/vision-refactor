# YOLO 迁移 S5 GPU 复测决策报告（Issue #23）

来源：GitHub Issue #23 `S5：GPU 复测决策门 + YOLO 价值再评估（前置闸门）`
首次报告日期：2026-05-31
修复复测日期：2026-06-01
关联脚本：`analysis/bench_annotate_fps.py`（独立 benchmark harness，不接主链路）
关联样本：`docs/yolo_eval_samples.json`（6 段 S0 同批样本）

> **结论先行：no-go，但 no-go 原因已纠正。**
> 2026-05-31 的初版报告把 `.venv` 中 `torch=2.12.0+cpu` 导致的
> `torch.cuda.is_available() = False` 当成最终 GPU 复测 no-go 依据，这是环境前提错误。
> 2026-06-01 已改装 CUDA 版 torch，并在 RTX 4060 Laptop GPU 上跑满 6/6 样本。
> 新结论不是“GPU 不可用”，而是：**CUDA 已可用，但真实端到端 `annotate()` FPS 比、裸推理 FPS
> 与 body_core 抖动没有达到 #23 预注册阈值，因此仍不启动 #25 / #26，也不改变 #28 默认切换决策。**
>
> CUDA 修正版关键口径：旧 CPU torch 前提已纠正；`torch=2.11.0+cu128`；
> 6/6 样本真实 GPU 复测完成；真实 GPU 数据 no-go。

---

## 一、实验环境固定

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.9（复用仓库 `.venv`） |
| OpenCV | 4.13.0 |
| MediaPipe | 0.10.31 |
| Ultralytics | 8.4.57 |
| Torch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| `torch.cuda.is_available()` | **True** |
| `torch_cuda_device_count` | **1** |
| `torch.version.cuda` | `12.8` |
| `nvidia-smi` | 可用 |
| GPU 硬件 | NVIDIA GeForce RTX 4060 Laptop GPU |
| NVIDIA Driver | 581.08 |
| Driver CUDA | 13.0 |
| YOLO 模型文件 | `models/yolo11n-pose.pt`（本地存在，6,255,593 bytes） |
| MediaPipe Pose 模型 | `models/pose_landmarker_full.task`（本地存在，9,398,198 bytes） |
| MediaPipe Hand 模型 | `models/hand_landmarker.task`（本地存在，7,819,105 bytes） |

环境修复命令（经本地代理 7890）：

```powershell
$env:HTTP_PROXY  = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
E:\CodeProject\vision\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall `
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128 `
  --proxy http://127.0.0.1:7890
```

环境探针：

```powershell
E:\CodeProject\vision\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
# torch 2.11.0+cu128
# torchvision 0.26.0+cu128
# True
# 12.8
# NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 二、预注册阈值（先写数字，再判定）

本 issue 是 S5 前置闸门，目标是判断是否值得扩实时入口 / UI / 默认切换。与 S0 不同，本轮必须看
`annotate()` 端到端链路，而不是只看裸 keypoint FPS。

| 指标 | 预注册阈值 | 判定方向 | 2026-06-01 CUDA 实测 | 是否达标 |
|---|---:|---|---:|---|
| GPU runtime 可用性 | `torch.cuda.is_available() == True` 且 device_count >= 1 | 不满足则不跑实时扩展 | `True`，device_count=1 | ✅ |
| S0 同批样本 GPU 有效复测覆盖 | 6/6 个样本都有有效 GPU benchmark 行 | 少于 6 段则不得 go | 6/6 | ✅ |
| `annotate()` Hands 关：YOLO body 预览 / MediaPipe pose-only FPS 比 | >= 1.30 | 低于则实时提速动机不成立 | min=0.586，mean=0.994 | ❌ |
| `annotate()` Hands 开：YOLO body + MediaPipe Hands / MediaPipe Pose+Hands FPS 比 | >= 1.20 | 低于则 UI/实时扩展不成立 | min=0.735，mean=0.904 | ❌ |
| YOLO 裸推理 GPU FPS | >= MediaPipe full CPU S0 均值 62.85 FPS，且 YOLO/MP >= 1.0 | 低于则不作为提速依据 | min=22.363，mean=41.343 | ❌ |
| YOLO 失败 / 漏检帧率 | <= 2% | 高于则 no-go | max=0.0% | ✅ |
| body_core 抖动中位数（归一化坐标） | <= 0.006 | 高于则 no-go | max=0.0104 | ❌ |

本轮 6/6 样本均产生有效 GPU benchmark 行，已经消除了“CPU torch 导致无 GPU 数据”的错误前提。
但端到端 FPS 比和抖动仍未达到预注册阈值，因此 #23 仍判 no-go。

---

## 三、benchmark harness 边界

`analysis/bench_annotate_fps.py` 只做以下事情：

- 读取 `docs/yolo_eval_samples.json` 的 6 段样本。
- MediaPipe 路径调用 `MediaPipePipeline.annotate()`，分 Hands 开 / 关两档。
- YOLO 路径调用 `YoloPoseAdapter.infer_frame()`，只绘制 `body_core_v1` 有效点；Hands 开时只额外调用
  MediaPipe HandLandmarker，不调用 MediaPipe Pose 补点。
- CUDA 可用时，YOLO 记录 `yolo_raw_infer_fps`、`yolo_miss_rate`、
  `yolo_body_core_missing_rate`、`yolo_body_core_jitter_median`，用于对照第二节裸推理 FPS、
  失败 / 漏检帧率与抖动阈值。
- 写出 `outputs/gpu_recheck/gpu_recheck_env.json`、
  `outputs/gpu_recheck/annotate_fps_gpu_recheck.json`、CSV；`outputs/` 受 `.gitignore` 排除。
- 不修改 `apps/main.py`、不改 `MediaPipePipeline` 默认行为、不接入主链路。

本次执行命令：

```powershell
E:\CodeProject\vision\.venv\Scripts\python.exe -m analysis.bench_annotate_fps `
  --samples docs/yolo_eval_samples.json `
  --asset-root E:\CodeProject\vision `
  --models-dir E:\CodeProject\vision\models `
  --yolo-model E:\CodeProject\vision\models\yolo11n-pose.pt `
  --device cuda `
  --out outputs/gpu_recheck
```

输出关键事实：

- `status = ok`
- `torch.cuda.is_available() = True`
- `torch_cuda_device_count = 1`
- `torch_cuda_device_name = NVIDIA GeForce RTX 4060 Laptop GPU`
- `requested_device = cuda`
- `samples_total = 6`
- 有效 GPU benchmark 样本覆盖 = 6/6

### Issue #38 benchmark 口径修正

Issue #38 对 benchmark harness 做的是**测量口径修正**，不是改变 #23 的历史 no-go 判定。
修正后必须用新的 CSV/JSON 重新形成二次决策基线，旧的逐样本表继续作为 2026-06-01 CUDA
修正版历史记录保留。

新口径：

- `--warmup-frames` 默认 10；warmup 帧不计入 `annotate_fps`、`timed_latency_ms_p50/p90/p99`
  和 `yolo_raw_infer_latency_ms_p50/p90/p99`。
- YOLO warmup 只调用同一个 `YoloPoseAdapter` 的原始推理，再重置 benchmark 指标与 tracker，
  使 CUDA/model cold-start 留在 `cold_first_infer_sec`，不污染正式计时；Hands 的 VIDEO
  时间戳仍在正式段从 0 开始。
- MediaPipe warmup 使用独立 VIDEO runner 与独立 capture，正式段重新从 0 开始，避免
  `detect_for_video()` 时间戳倒退。
- `--device cuda` 时不再因 CUDA 不可用跳过整个任务；只跳过需要 CUDA 的 YOLO case，同时保留
  `mediapipe_cpu` 和 `yolo_cpu` 基线行。
- CSV/JSON 新增或固定字段：`backend`、`device`、`delegate`、`warmup_frames`、`timed_frames`、
  `init_sec`、`cold_first_infer_sec`、`cold_first_annotate_sec`、
  `timed_latency_ms_p50/p90/p99`、`yolo_raw_infer_latency_ms_p50/p90/p99`。

### Issue #39 PyTorch FP16 / imgsz / adapter warmup 实验口径

Issue #39 在 #38 口径上继续增加**显式 opt-in** 的 YOLO PyTorch 实验维度：

- `YoloPoseAdapter` 默认仍为 `imgsz=640`、`device="cpu"`、`half=False`、`warmup=False`，
  构造与 import 不触发 ultralytics 加载或推理。
- GPU case 可显式开启 `half=True` 与 adapter 内部 warmup；CPU / MPS 请求 half 时自动降级为
  `half=False`，避免误用 FP16。
- `extract_yolo_landmark_series` 透传并记录 `imgsz`、`device`、`half`、`warmup`、
  `warmup_shape`，避免只优化 benchmark。
- benchmark 新增 `imgsz`、`half`、`adapter_warmup`、`max_persons`、
  `warmup_shape`、`multi_person_frames`、`review_required`、`gate_status` 字段，用同一
  CSV/JSON 同时观察性能、实验 warmup 尺寸与多人质量守卫。
- `--device cuda` 时除 #38 的 FP32 baseline 外，额外跑 FP16 640 与 FP16 512 的 body-only /
  body+hands case；这些 case 只用于二次决策，不打开 CLI/UI YOLO 实时入口，也不授权评分。

---

## 四、逐样本结果

| sample | MP pose FPS | YOLO body FPS | Hands 关比值 | MP hands FPS | YOLO+Hands FPS | Hands 开比值 | YOLO raw FPS | miss | body_core missing | jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| std_front | 34.185 | 20.037 | 0.586 | 18.423 | 13.542 | 0.735 | 22.363 | 0.0000 | 0.0000 | 0.0027 |
| std_side_long | 29.539 | 33.454 | 1.133 | 18.095 | 16.110 | 0.890 | 38.905 | 0.0000 | 0.1254 | 0.0040 |
| student_1 | 30.777 | 41.767 | 1.357 | 18.300 | 19.094 | 1.043 | 49.033 | 0.0000 | 0.2533 | 0.0104 |
| student_4_long | 40.374 | 43.400 | 1.075 | 23.234 | 19.193 | 0.826 | 51.077 | 0.0000 | 0.2832 | 0.0068 |
| punch_front | 37.685 | 41.975 | 1.114 | 18.431 | 17.560 | 0.953 | 49.087 | 0.0000 | 0.0000 | 0.0034 |
| punch_side | 46.844 | 32.795 | 0.700 | 21.871 | 21.316 | 0.975 | 37.593 | 0.0000 | 0.0310 | 0.0068 |

汇总：

- Hands 关端到端 FPS 比：min=0.586，mean=0.994，未达 >=1.30。
- Hands 开端到端 FPS 比：min=0.735，mean=0.904，未达 >=1.20。
- YOLO raw FPS：min=22.363，mean=41.343，未达 >=62.85。
- YOLO miss rate：max=0.0%，达标。
- body_core missing rate：max=28.32%，虽不是单独预注册硬阈值，但说明边界样本缺点风险仍高。
- body_core jitter median：max=0.0104，未达 <=0.006。

达标计数：

- Hands 关 FPS 比仅 1/6 达标。
- Hands 开 FPS 比 0/6 达标。
- YOLO raw FPS 0/6 达标。
- body_core 抖动 3/6 超阈值。

---

## 五、go / no-go 判定

**判定：no-go。**

逐条对照：

1. GPU runtime 可用性：`True`，device_count=1，RTX 4060 可被 torch 调用 → 通过。
2. S0 同批样本 GPU 覆盖：6/6 → 通过。
3. Hands 关 `annotate()` FPS 比：min=0.586 < 1.30，mean=0.994 < 1.30 → no-go。
4. Hands 开 `annotate()` FPS 比：min=0.735 < 1.20，mean=0.904 < 1.20 → no-go。
5. YOLO 裸推理 GPU FPS：min=22.363 < 62.85 → no-go。
6. 漏检率：max=0.0% <= 2% → 通过。
7. body_core 抖动：max=0.0104 > 0.006 → no-go。

因此，本次修复推翻的是“CUDA 无效 / 0 行 GPU 数据”的错误前提；没有推翻 #23 的最终 no-go 结果。
后续 S5/S6 仍不能把实时入口或默认切换建立在当前 YOLO GPU 数字上。

---

## 六、对 S5 / S6 的分流结论

- #25（CLI `apps/main.py --backend yolo` 实时预览）：**关闭 / 不实现**。当前已具备 CUDA，
  但 Hands 关 / 开端到端 FPS 比和 raw FPS 未达阈值。
- #26（UI 后端选择 + 完整度提示）：**关闭 / 不实现**，因为它依赖 #25 与实时 go 结论。
- #24（batch backend/layout 参数 + metadata 透传）：**可独立保留**。它服务离线调试 / 标定分数接线，
  不依赖 #23 的实时 go 结论；仍必须保持 `calibration_status=unvalidated` 与
  `score_authorized=False`。
- #27（Hybrid）：**默认不实现**。当前 GPU 复测没有证明 YOLO body 实时收益；Hybrid 还会叠加
  MediaPipe Pose 补点代价，必须另开专门 benchmark 才能重评。
- #28（S6 默认切换决策）：**已执行**，见 `docs/yolo_default_switch_decision.md`。当前实测 no-go 分支下结论仍为
  **全部不切默认，仅保留离线 / 实验入口**。

---

## 七、CLI 实时预览决议（Issue #25）

**结论：当前仍不实现 `apps/main.py --backend yolo` / `--feature-layout body_core_v1` 实时预览入口。**

Issue #25 的启动条件是 #23 GPU 复测 = go；2026-06-01 已修复 CUDA torch 并跑满 6/6 样本，
但 #23 仍未达 go 阈值。因此本期不把 YOLO 接入 `apps/main.py` 的实时预览主链路，也不新增会被用户
误认为可用的 `--backend yolo` 参数。`apps/main.py` 继续保持 MediaPipe 旧默认行为：`--source` /
`--pose` / `--workers` 等既有参数不变，实时预览仍由 `MediaPipePipeline.annotate()` 负责。

本决议按 Issue #25 的验收标准逐条对照：

| 判据 | Issue #25 要求 | 当前数字 / 事实 | 结论 |
|---|---|---|---|
| 前置条件 | #23 GPU 复测 = go 才启动 | #23 CUDA 实测 no-go | 不启动 |
| GPU runtime | CUDA runtime 可用 | `torch.cuda.is_available() = True`，device_count=1 | 满足 |
| 样本覆盖 | 6/6 个样本有有效 GPU benchmark 行 | 6/6 | 满足 |
| Hands 关实时提速 | YOLO body 预览 / MP pose-only FPS 比 >= 1.30 | min=0.586，mean=0.994 | 不满足 |
| Hands 开实时提速 | YOLO body + MP Hands / MP Pose+Hands FPS 比 >= 1.20 | min=0.735，mean=0.904 | 不满足 |
| raw FPS | YOLO raw FPS >= 62.85 | min=22.363，mean=41.343 | 不满足 |
| 抖动 | body_core 抖动中位数 <= 0.006 | max=0.0104 | 不满足 |
| 评分边界 | YOLO-only 不产出规则 / tech_eval 对外评分 | #24 仅保留离线 batch 调试 / 标定入口，`score_authorized=False` | 满足边界 |

因此 #25 当前只能关闭，不做实现。若未来要重新打开 CLI 实时预览入口，必须先让 #23 的真实性能和稳定性阈值通过，
并另开新的实现子任务；在此之前，不得预先写入 YOLO realtime runtime 或参数入口。

---

## 八、UI 后端选择决议（Issue #26）

**结论：当前仍不实现 `apps/app_ui.py` 的 YOLO 后端选择控件 / 规则完整度提示。**

Issue #26 的启动条件是 #25 已落地；当前 #25 已按 #23 CUDA 实测 no-go 关闭，不存在可供 UI 选择的
`apps/main.py --backend yolo` 实时预览入口。因此本期不在 Tkinter UI 中增加 YOLO 后端选择、
`body_core_v1` 布局选择、YOLO-only 规则完整度提示或未标定评分展示。`apps/app_ui.py` 继续保持
MediaPipe 旧默认界面与行为。

换言之：CUDA 可用，但 #23 实时阈值未达标；#25 仍关闭 / 不实现，#26 没有可依赖的实时入口。

本决议按 Issue #26 的验收标准逐条对照：

| 判据 | Issue #26 要求 | 当前数字 / 事实 | 结论 |
|---|---|---|---|
| 前置条件 | #25 落地后启动 | #25 已关闭 / 不实现 | 不启动 |
| GPU runtime | 实时 UI 扩展需 #23 go | CUDA 可用，但 #23 性能 / 抖动未达标 | 不满足 go |
| 样本覆盖 | 6/6 个样本有有效 GPU benchmark 行 | 6/6 | 满足 |
| CLI 依赖 | UI 后端选择依赖 `apps/main.py --backend yolo` | #25 不实现该入口 | 不满足 |
| 评分边界 | YOLO-only 不显示对外评分 / 合格判定 | #24 仅保留离线 batch 调试 / 标定入口，`score_authorized=False` | 满足边界 |

因此 #26 当前只能关闭，不做实现。若未来要重新打开 UI 后端选择，必须先满足 #23 性能 / 稳定性阈值，
并在 #25 后续实现子任务真正落地后另开 UI 实现子任务；在此之前，不得预先写入 YOLO UI control、
YOLO-only 完整度展示或未标定评分入口。

---

## 九、Hybrid 决议（Issue #27）

**结论：不实现 `yolo_body_mp_pose_supplement`，代码库不保留半成品 Hybrid 路径。**

本决议按 Issue #27 的触发条件逐条对照：

| 判据 | 触发实现所需条件 | 当前数字 / 事实 | 结论 |
|---|---|---|---|
| GPU runtime | `torch.cuda.is_available() == True` 且 device_count >= 1 | `True`，device_count=1 | 满足 |
| S0 同批样本 GPU 覆盖 | 6/6 个样本跑完有效 GPU benchmark | 6/6 | 满足 |
| 实时 / 默认入口提速 | Hands 关 FPS 比 >= 1.30；Hands 开 FPS 比 >= 1.20 | min=0.586 / 0.735 | 不满足 |
| YOLO raw FPS | >= 62.85 | min=22.363 | 不满足 |
| 稳定性 | body_core 抖动 <= 0.006 | max=0.0104 | 不满足 |
| CPU 基线收益 | 至少不慢于 MediaPipe full | S0 CPU 纯推理 speedup = 0.41（YOLO 更慢） | 不满足 |
| 补点价值 | 补回重心 / 发力顺序所需脚跟脚尖，且性能代价可接受 | COCO17 缺脚跟脚尖；靠 MediaPipe Pose 补脚/脸本质仍需跑完整 PoseLandmarker | 未证明 |

达标计数：Hands 关 1/6 达标，Hands 开 0/6 达标。

因此，Hybrid 的核心触发条件没有满足：虽然 CUDA runtime 已修复，但当前 YOLO body 预览没有达到实时收益阈值，
更没有 Hybrid 专用 benchmark 证明“YOLO body + MediaPipe Pose supplement”比直接使用 MediaPipe full 更有工程收益。
当前更稳妥的边界是：

- 实时 / UI：按 #23 CUDA 实测 no-go，#25 / #26 不实现。
- 离线 batch：按 #24，只保留显式 `body_core_v1` 调试 / 标定入口，且 `score_authorized=False`。
- full tech_eval / 规则评分：默认仍 MediaPipe full；YOLO-only / Hybrid 不进入对外评分。

若未来要重新评估 Hybrid，必须先单独满足以下全部条件，再另开实现子任务：

1. #23 端到端 Hands 关 / Hands 开 FPS 比和稳定性阈值先达标。
2. 新增 Hybrid 专用 benchmark，明确记录第二个 PoseLandmarker 的额外耗时、内存 / 显存占用。
3. 证明补回的脚跟脚尖 / 嘴角能让重心、发力顺序等核心指标恢复评估，且误判样例可接受。
4. Hybrid 仍只能作为离线显式模式，不能进入默认 / 实时路径，直到 S6 重新决策。

---

## 十、后续若要重新打开实时 / UI 的前置条件

若之后要重新打开 #25 / #26，先不要改主链路，必须先满足以下条件并复跑本报告命令：

1. CUDA-enabled torch 环境保持可用，`torch.cuda.is_available() == True`。
2. `analysis/bench_annotate_fps.py --device cuda` 对 `docs/yolo_eval_samples.json` 6/6 样本跑完。
3. Hands 关 / Hands 开两档 `annotate()` FPS 比分别达到 >=1.30 / >=1.20。
4. YOLO raw FPS 达到 >=62.85，漏检率 <=2%，body_core 抖动中位数 <=0.006。
5. 仍不得绕过 #10 标定结论：YOLO 对外评分只有在标定 / 回归重新达标后才能另行授权。
