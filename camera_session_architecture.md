# Tk 桌面摄像头会话架构（2026-09-30）

本文记录本次重构后的所有权、状态与验收边界，对应交接说明 `摄像头架构设计交接说明.md` 的问题清单。范围仅限 Tk 桌面入口（`apps/app_ui.py`，打包入口 `apps/desktop_launcher.py`）；CLI、`camera_enum.open_camera` 进程内打开、Vue/Tauri bridge 不在本次范围。

## 1. 现状问题（重构前，代码依据）

| 问题 | 依据 | 性质 |
| --- | --- | --- |
| 单摄预打开与双摄预热池两套所有者并存，靠 `_camera_open_locks` + `_ExclusiveCameraCapture` 互斥 | `_kick_preopen/_preopen_camera/_take_preopen_cap`、`CameraWarmupPool(capture_factory=_open_camera_exclusive)` | 重复管理 |
| 单摄预打开/串行打开的锁只覆盖 open，不覆盖句柄生命周期 | `_open_camera_serialized` 在 open 返回后即释放锁 | 同设备可重叠打开 |
| 采集尺寸在 reader 打开时读取可变的 `_student_practice_active`；预热复用只比较编号 | `_open_camera_exclusive`、`warm()` 的复用条件 | 旧格式混入新会话 |
| `cancel()` 返回被当作"已释放"；学生冷启动先盲目 release 再 start | `_student_start` → `_release_camera_warmups` → `_start` | 已复现的竞态（前次补丁在 reader 内等待） |
| 失败会话按角色取消预热，不区分代次；`_stop` 在 Tk 线程取消池 | `_cancel_dual_warmup_roles` | 旧会话可取消新预热 |
| 退出学生练习在 worker 未退出时就恢复「开始」，点击被静默丢弃 | `_exit_student_practice` 末尾 `_set_running_controls(False)` | 状态与 UI 不一致 |
| 会话自行终止（断流/失败）时考试面板未收到通知 | `on_session_stop` 仅在 `_stop/_on_close` 调用 | 考试状态机悬挂 |
| 释放失败：`_ExclusiveCameraCapture` 永久持锁，池内无重试 | `ProcessCamera.release()` 超时抛错 | 需重启 App |

真实设备约束（保留）：Windows 驱动 open/read/release 可能阻塞；MSMF/DSHOW 编号映射只在友好名唯一时可信；格式必须在首帧前协商；PyInstaller onefile 需 `freeze_support()` + spawn。

## 2. 方案

### 所有权

- `CameraWarmupPool`（`apps/camera_warmup.py`）是 Tk 桌面**唯一**的物理设备所有者。所有打开都经池的工厂 `App._open_camera_process` → `ProcessCamera`（每路一个 spawn 子进程）。
- 设备状态：`reader`（池内预热线程持有）→ `lease`（会话独占读帧）→ 释放完成；释放失败 → 隔离（仍记为占用）。
- `claim_*` 把句柄转成 `CameraLease`，lease 期间设备仍在池的占用表中；只有 `lease.release()` 成功才解除占用。失败时所有权回到池并隔离，下一次打开同编号前由 reader 线程按间隔重试释放。

### 配置与模式

- `CaptureSpec(index, width, height)` 在请求时冻结：选择回调由主线程按当前模式构造；会话在 `_collect_state` 时把 `capture_size` 写入 `UiState`。打开过程不再读取模式标志。
- 预热复用条件为 spec 完全相同；尺寸不同则替换 reader，新 reader 在后台等待旧进程**真实释放**后才按新尺寸打开。
- 普通/考试 1280×720，学生练习 1920×1080。进入学生练习即按 1080p 声明预热；退出且 worker 结束后按 720p 重新声明。

### 状态与完成条件

| 状态 | 完成判定 |
| --- | --- |
| 请求停止 | `_stop_evt` 置位（Tk 线程只做这一步与 UI 禁用） |
| 读帧停止 | worker 主循环退出，录像按原片段边界 `_close_recording_pair()` 收尾 |
| 设备释放 | `lease.release()` 返回（子进程已退出）；失败则 `final_status` 显示原因并隔离 |
| 可以重新开始 | worker 调用 `_post_done`（最后一步）；`_start` 若线程仍在返回途中会短暂 join，否则提示仍在收尾 |

### 并发与取消

- 会话代次：`_session_generation`/`_current_session_generation` 保留；单摄 `_post_done` 现在也带代次，旧会话的完成回调不会恢复新会话的按钮。
- 预热代次：失败会话只取消 `wait_pair` 观察到的代次（`cancel(role, generation=...)`）。
- 池的 `cancel/sync/replace` 只发信号，释放由 reader/reaper 线程完成，Tk 线程不碰驱动。
- 交换正/侧：`sync()` 一次声明两路，旧 reader 退役后按新角色重开，不再因"同编号不能两角色"报错。
- 刷新枚举：先后台等待池内进程释放，再走可能在进程内探测设备的回退路径。
- 关窗：池 `close()` 后在关窗准备线程内有界等待 `wait_released(3s)`，超时写日志。

### 失败与恢复

- 双摄为整体事务：任一路打开/首帧失败则两路都不交给会话；运行中任一路断流则整对 lease 归还、录像收尾。
- 失败不自动重预热（避免覆盖错误），用户再点「开始」时按冻结 spec 重新协商。
- 不支持 1080p 的设备：`ProcessCamera` 按实际首帧尺寸工作并记录 `actual`，不伪装成功；是否提示降级属于产品决定（见 §5）。

### 线程/进程职责

| 工作 | 所在 |
| --- | --- |
| open、格式协商、首帧、read、release | spawn 子进程（`ProcessCamera`），父进程有界等待 |
| 等待旧设备释放、预热读帧 | 池 reader 线程 |
| 取消后的收尾 | 池 reaper 线程 |
| 模型加载、推理、录像写入、lease 释放 | 会话 worker 线程 |
| 按钮、状态、预览渲染 | Tk 线程（仅经 `root.after`/队列） |

预热保留：USB 摄像头冷启动 0.5–2.5 s，1080p MJPG 更慢，预热让「开始」与模型加载重叠。进程隔离保留：驱动阻塞只能靠终止子进程有界回收，线程无法中断。

## 3. 删除与迁移

| 旧代码 | 处理 |
| --- | --- |
| `_preopen_lock/_preopen_cap/_preopen_index/_preopen_pending_index/_preopen_generation`、`_kick_preopen/_preopen_camera/_take_preopen_cap/_release_preopen_cap` | 删除，由 `pool.sync(primary, None)` + `wait_one/claim_one` 替代 |
| `_camera_open_locks(_guard)`、`_camera_open_lock/_acquire_camera_open_lock`、`_open_camera_serialized/_open_camera_exclusive`、`_ExclusiveCameraCapture` | 删除，由池的占用表替代 |
| `_warm_dual_cameras`、`_cancel_dual_warmup_roles` | 删除，由 `pool.sync` 与 `_cancel_session_warmups(generations)` 替代 |
| `_student_start` 中的盲目 `_release_camera_warmups()` | 删除，由 spec 不一致触发的有序重建替代 |
| `claim_pair` 返回裸句柄 | 改为返回 `CameraLease` |

测试迁移：删除针对预打开/索引锁的单元测试，改为 `sync` 声明、spec 冻结、代次取消、lease 占用、释放失败隔离与重试、角色交换等测试；真实 Tk + spawn 模式切换检查扩展为"720p→学生 1080p→退出"两轮、开始即停止再开始、双摄转单摄与关窗后无残留子进程。

## 4. 验收映射与日志

| 交接 §7 场景 | 覆盖 |
| --- | --- |
| 单/双摄冷启动，首帧后就绪 | `test_camera_warmup` wait/claim；`desktop_smoke._check_camera_mode_switch` |
| 普通→停止→学生→退出，重复 | `_check_camera_mode_switch` 两轮（真实 Tk mainloop + spawn） |
| 已有预热/预热中进入学生 | `test_spec_change_renegotiates_after_old_format_releases`、`test_restart_waits_for_cancelled_capture_release` |
| 开始后立即停止、停止中再开始 | smoke 开始即停止段；`test_single_camera_stop_during_open_finishes_session_quietly` |
| 双摄单路失败/首帧失败/断流 | `test_claim_pair_failure_releases_other_ready_capture`、`_exercise_dual_runtime_failure` 系列 |
| open/read/release 阻塞、慢释放 | `test_camera_capture` 卡死用例、`test_failed_lease_release_is_quarantined_then_retried_before_reopen`、`test_leased_device_blocks_reopen_until_session_releases` |
| 单↔双、交换、重新枚举 | `test_sync_swaps_roles_without_rejecting_same_index`、smoke 双转单、枚举测试 |
| 模型加载期间停止、预热中关窗 | `test_pipeline_loading_keeps_raw_pair_flowing_without_writes`、关窗测试、smoke `wait_released` |
| 学生/考试开始、结束、录制交接 | `test_student_presence`、考试回归；会话自行终止通知考试面板 |
| 冻结 EXE 与甲方目标双摄 | **未完成**：需重新打包运行 `--offline-self-check`，并在甲方双摄上实机验收 |

日志（`desktop.log`，前缀 `[camera]`，JSON 一行）：`event`（warm/replace/cancel/ready/error/lease/released/release_failed/session_start/stop_requested/session_done/close_release_timeout）、`role`、`index`、`requested`、`actual`、`generation`、`session`、`phase`（wait_release/open/first_frame/read）、`open_ms`、`error`、`status`。子进程另有 `[camera-capture] index= requested= actual= fps=`。

## 5. 基线与待决事项

- 本分支基于 `origin/main`（958dd2f）+ 前次未提交的摄像头补丁（`outputs/camera-mode-repair-20260929.patch`，不含 change.md）重建，**不包含**主工作区中尚未提交的结果页/反馈工作，也不含已交付 EXE 的 v4 分析规则。
- 已交付 v4 EXE 对应的源码只存在于本机 Codex checkpoint 树，Git 分支中没有。发布前需先把 v4 源码整理为正式提交，再与本分支合并，统一从源码重新打包；不再使用替换 EXE 内模块的方式交付。
- 待用户决定：设备实际不支持 1080p 时是降级继续（当前行为，日志记录 actual）还是阻止学生练习并提示。
