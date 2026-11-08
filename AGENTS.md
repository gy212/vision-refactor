# Repository Guidelines

本文件是 vision 重构仓库的统一规范，适用于所有 coding agent。`CLAUDE.md` 只导入本文件；规则有变化时只改这里。

## 仓库现状

武术散打动作识别与评分系统，包括 Tk 桌面（实时预览、双摄录制、考试、学生练习）、CLI、batch 与 analysis 离线工具，以 Windows 单文件 EXE 交付。正式评分与姿态识别专注于 MediaPipe 链路。

- **本仓库**：工作区的 `new/` 目录，远端 `https://github.com/gy212/vision-refactor`（私有）。本仓库是当前所有代码、文档、测试与提交的唯一活跃开发区。包含核心视觉与评分库 `core/`、Tkinter 桌面应用 `apps/tk/`、命令行入口 `apps/cli/`、离线批处理与分析工具（`batch/`、`analysis/`）以及自动化测试 `tests/`。
- **旧项目只读参照**：工作区的 `old/` 目录属于外层仓库（`../old/`，工作区根目录的 `.git`，远端 `YGuo-2/vision`），仅用于只读对比与历史溯源。严禁在 `old/` 中修改、新增或删除任何文件；必须运行旧代码对照时，必须使用 `python -B` 且工作目录与输出显式指向 `old/` 之外。

## 专项任务规范（按需自动读取 `spec/`）

本仓库在 `spec/` 目录下维护三份专项工程契约。**当你的任务涉及以下任一场景时，在动笔或执行前必须先主动读取对应规范文档，并严格按其中的工作流与强制输出模板执行（严禁凭默认习惯盲写或删减小节）**：

| 触发任务场景 | 必读规范路径 | 核心约束重点 |
|---|---|---|
| **编写或更新 Issue**<br>（问题建单、缺陷分析、重构任务定义） | `spec/issue_spec.md` | 强制从触发样本向上抽象一层、锁定源头模块与同级兄弟场景全集、语言简洁易懂且小节完整不丢项 |
| **发起或更新 PR**<br>（创建拉取请求、撰写 PR 标题与正文描述） | `spec/pr_spec.md` | 强制填写“架构破立交底表”（如实招供未覆盖项与保留差异）与“业务结果前后对比表”，拒绝测试数字注水与免责废话 |
| **执行代码与方案审查**<br>（PR Review、方案评审、复审验收） | `spec/review_spec.md` | 强制“先脱钩独立建模、后对比审判方向”的反锚定流程，按五大工程维度拷问做法是否正确、有无更优解，采用倒金字塔输出 |

## 功能完善工作方式

系统架构重构（P0–P9）已全量闭环，当前处于**功能完善、算法精细化与产品化交付阶段**。所有新增功能与修改直接在现有架构结构上推进，不再保留任何过渡期双轨或旧路径门面。

- **分层依赖底线**（设计文档 §2.1）：严格遵守单向分层与依赖倒置原则：`foundation (L0) → media (L1) → pose (L2) → evaluation (L3) → {services, recording, live, reporting} (L4) → workflows (L5) → apps (入口)`。
  - 只能自上而下单向依赖更低层，严禁同层交叉依赖或逆向跨层依赖；
  - L4 域内四个包（`services/`、`recording/`、`live/`、`reporting/`）彼此平级自治、互不导入，共用纯逻辑统一下沉至 L0/L1，跨域协作由调用方在上层组合注入；
  - L4 不向上依赖 L5（如录制 finalizer 的具体执行步骤由工作流计划构建方注入）；
  - 入口层只消费各包 `__init__` 导出的公共 API，严禁跨模块导入下划线开头的私有实现（`tests/` 除外）；
  - `core/` 领域核心严格与 GUI 解耦，严禁导入 `tkinter`。由 `tests/test_architecture.py` 机器门禁强制。

## 软件工程规范

**核心职责与严谨性准绳**：你的职责是在既定架构边界内高效完成用户交付的目标，**而不是证明整个仓库、运行环境或历史基线“绝对无风险”**。工程严谨性唯一取决于**“核心逻辑是否切中本质、目标场景是否真正跑通、关键证据是否准确对位”**，绝不以检查次数、对比样本数量、跑测规模、全仓扫描范围或防御代码厚度来衡量。准备工作、前置盘点、抽空架子或跑通检查都不算完成，唯以用户要求的实际业务结果落地为准。能由机器检验的规则由 `tests/test_architecture.py` 强制，其余在协作与代码审查中把关。

### 1. 领域抽象与本体归属（从“触发样本”向上抽象一层）

- **区分“观测样本”与“概念归属”**：用户提供任何具体案例（某个功能入口、某个具体动作、某份报告、某张表格）时，那只是**暴露问题的观测窗口与验证载体**，绝不等同于修改的作用域边界。
- **动手前“概念归属三问”**：在写下任何代码或条件分支前，必须先回答：
  1. 这个逻辑在现实物理世界或业务领域中，**本质上属于哪个层级的概念**？（是全局基础能力、某类子域/阶段共性，还是唯独该个体才具备的特殊属性？）
  2. 系统中还有哪些**同级兄弟场景、其他动作或上下游入口**共享同一本质？
  3. 如果把代码写在这里，当用户明天切换到另一个同类动作或入口时，该逻辑是**自动生效**，还是会因为特判而失效？
- **严禁按触发场景写死分支**：永远将逻辑绑定在它**最本质的领域概念粒度**上（如基础几何测量、公共阶段、基本动作段、通用服务），严禁把用户在对话中碰巧用来举例或测试的对象硬编码为代码的判断条件（如 `if action == "..."`）。

### 2. 价值闭环与主次分明（拒绝“煮熟饭不吃”的形式主义）

- **建用一体，演进即替换**：重构、抽象、提取公共模块只是**手段**，让全系统所有适用业务场景真正跑通、产出正确结果才是**目的**。提取公共能力或升级算法判法时，**“建立新抽象 + 接入所有适用消费方 + 淘汰旧残缺路径”必须在同一次交付中完整闭环**。严禁“把公共层搭好了，却搞出 `legacy` 默认分支让其他同类业务继续跑在已淘汰的旧烂逻辑上”。
- **单一事实来源（Single Source of Truth）**：全库每项业务能力只保留唯一的标准实现路径。同源事实（如同步采集的多路视频动作识别与计时、统一评估产出的证据与测量部位）在源头只计算一次、产出一份共享记录，下游多视角或多出口（练习、考试、batch、Tk、HTML）统一消费，严禁各出口或各视角重复计算后再拿结果互相对账。
- **主次分明，严禁配角一票否决主角**：任何处理链路必须严格区分**核心主判定**与**辅助附加项**（如：核心规则结论 vs 辅助对照图；局部阶段缺失 vs 其他已看清的独立阶段）。辅助信息、非必需参照帧或无关阶段的缺失，只允许降级该局部或省略附加项并如实说明，**严禁一刀切连坐或一票否决核心主链路**。
- **失败可解释底线**：当系统不得不返回失败、拒绝或“无法判断”时，只要原始输入存在，就必须保留并展示**现场证据（所选代表画面、已测得数值）与具体死因（卡在哪一步、缺哪个关键点或哪个时段）**，严禁黑盒吞没上下文或只甩一句笼统结论。

### 3. 第一性原理与受阻熔断（物理常识优先，拒绝教条炫技）

- **直觉与物理常识检验**：设计算法或架构前，先退回第一性原理问：**“如果由专业人类肉眼判断，最直接、最稳定的依据是什么？”** 能用单一观测维度、直接 2D 几何关系或简单确定性逻辑一眼看清的问题（哪面看得清就用哪面），**严禁教条式地强行捆绑多视角融合、高维拟合或复杂状态机**。
- **死胡同熔断机制（Stop-Loss）**：当某条技术路径在真实数据上验证受阻、逻辑越来越迂回拧巴、或尝试 1 轮后仍算不出/误报超标时，**必须立即触发熔断停手**。严禁在错误路径上闷头死磕，必须立刻向用户汇报具体反例与瓶颈，并退回探讨更简单直接的替代解法。

### 4. 最小化核查与“回答即停”铁律（严禁防御性仪式与无休止自检）

- **无决策价值不检查（四要素准入）**：任何前置核查、只读诊断、环境盘点或事后自检，都必须服务于**能够改变当前具体决策**的真实问题。启动检查前必须明确：**① 单一待回答问题、② 最小检查对象、③ 最贴近结论的单个可运行检查、④ 明确停止条件**。严禁为了“更放心”、“保证没有遗漏”、“全面确认状态”或纯理论上的假想风险自创前置检查门槛。
- **单点验证与“回答即停”**：默认只执行**一个**最贴近当前改动或决策的最小可运行检查（优先使用最小合成输入、定向单测或已落盘缓存）。**一旦该检查回答了当前问题且无相反证据，必须立即停止检查并转入下一步执行或直接交付**——严禁换工具重复收集等价证据，严禁把剩余时间用于寻找更多检查项，严禁在收尾时自行追加“最终全面复核”。
- **无反证不扩圈（受控逐级定位）**：只有当最近一次最小检查失败，或出现明确反证表明影响面更大时，才允许沿**实际失败路径**扩大一个最小层级；没有直接关联证据时，**严禁将局部问题扩展为扫描全仓文件、比对无关哈希、重算未改动输入或重跑全量历史基线**。
- **默认信任并复用既有状态**：切换会话、切换分支/worktree、或工作区存在无关未提交改动，本身绝不意味着既有结果失效。只要相关输入、代码或配置未发生实质变化且无反证，**必须直接复用**已有缓存、中间产物、基线与前序验证结果，严禁动辄推翻重算或全盘复查。
- **区分“真实阻塞”与“尚未排除的理论可能”**：只有必要输入缺失、硬约束冲突或出现确定性反例时才构成真实阻塞；**“可能还有风险”、“尚未全面审查”、“可以再跑全量确认一次”绝不是阻塞主任务推进或拖延交付的理由**。

### 5. 单一职责与高内聚低耦合

- **专一职责与显式声明**：`core/`、`apps/`、`batch/`、`analysis/`、`scripts/` 下的每个 Python 模块必须在头部 docstring 第一句清晰写明其承担的核心职责。如果需要使用“和”、“以及”连接两个互不相关的业务域才能概括其功能，说明职责发生发散，必须立即进行正规拆分。
- **策略维度正交分离**：不同维度的业务策略（如动作编排、计时切分、测量标准、验证资格、评分模式、报告选图）必须各自独立声明、各司其职。严禁把某一个维度的集合或开关当成控制其他维度的“万能开关”，更严禁在状态/资格登记函数里夹带删减业务项、篡改标准或暗换测量公式。
- **杜绝杂物箱模块**：禁止引入 `utils`、`helpers`、`common`、`misc` 这类语义模糊的模块（包括 `xxx_utils.py` 等后缀形式）。共用逻辑必须按照具体的业务领域归属下沉至专用模块（如几何计算归入 `foundation/geometry.py`、信号去抖归入 `foundation/signal.py`）。各包的 `__init__.py` 仅用于导出公共 API，只包含 docstring、导入语句与 `__all__`，严禁放置具体实现。
- **按领域职责切分，拒绝机械切片**：模块与函数超限时的拆分必须按领域处理流程或业务关注点进行（例如将直拳综合评估按“事件检测 → 相位截取 → 姿态测量 → 规则判定”拆分），并赋予每个拆出部分清晰、自解释的专业名称。严禁按物理行数机械切割（如 `xxx_part2.py`、`xxx_impl.py`、`_step1()` 等无领域语义的文件或函数）。

### 6. 显式假设、全局对齐与持续推进（沟通与授权准则）

- **动手前先对齐“抽象层级与生效范围”**：面对用户的局部指令或单场景反馈时，拒绝做只会按字面机械执行的木偶。动手编码前，必须先站在全系统架构高度审视受影响的同类场景，主动向用户确认：**“该问题本质属于 [XX 公共层]，我将直接在公共层修改，并让 [A、B、C] 场景全部接上生效，是否确认？”**
- **有效授权下持续推进，不反复索权**：以用户真实意图判断授权（“继续”、“直接改”、“按这个办”均属有效授权）。获得授权后必须直奔核心目标持续推进直至真实结果落地；同目标内的常规修复、重构接通与定向验证无需每走一步就停下索要授权。仅当面临显著改变结果的方案抉择、实质扩大任务范围、触碰高风险/不可逆操作或遇到无法自行排除的真实阻塞时，才停下向用户请示。
- **多义解读全量陈述与建设性反驳**：面对存在多种合理路径的需求时，完整罗列备选方案及利弊供用户拍板，禁止擅自默选；当发现拟定方向存在架构隐患或可能导致工程倒退时，基于事实直言不讳提出反驳或推介极简替代方案，而非盲目顺从。

### 7. 极简实现与正交边界约束（YAGNI 与禁止防御性分叉）

- **最小化有效代码（YAGNI）**：严格遵循奥卡姆剃刀准则，仅编写解决当前问题所必需的最少代码。未被明确要求的功能特性、配置项或所谓的“通用扩展性”一律不加。
- **横向无关隔离，纵向彻底打通（严禁为保旧测试而写死分叉）**：改动应聚焦于当前任务目标，不顺手修改无关模块的逻辑、注释与格式；但在当前任务所属的业务域内，**必须沿公共领域模型纵向彻底打通所有适用场景，并同步更新受影响的测试基准**。**严禁为了少改几个既有测试快照或规避影响面核对，就用条件分支或伪策略开关（如 `legacy_xxx`）把新修复圈禁在局部场景内！**
- **历史既存代码维持现状**：定位或阅读代码过程中若发现与当前任务无关的历史死代码、废弃逻辑或冗余引用，仅在沟通说明或提交说明中明确提示用户，严禁擅自修改或删除。

### 8. 适度防御与拒绝自创防御工程

- **边界严格校验，内部信任契约**：仅在系统真正的外部边界（用户输入、外部文件与目录、视频解码流、摄像头设备接入、CLI 参数及网络通讯）执行前置校验。系统内部各层基于类型注解与契约交互，坚持快速失败（fail fast）：严禁滥用 `getattr(obj, ..., None)` 掩盖契约不清、严禁对不可达分支层层堆砌容错 fallback、严禁静默吞没异常。
- **修实际缺陷，不顺带建防御体系**：定位具体失配点并在既有入口补上针对性验证即可。没有已证实的缺陷或明确要求，**严禁顺带发明通用校验框架、多层包装器、重复 checker、额外快照、备用执行链或为了“证明已遵守规范”而另建审查流程**。

### 9. 代码可读性、结构体量与改动自洁

- **体量上限与预警线**：非测试源码文件（`tests/` 除外）遵循**模块不超过 800 行、函数不超过 100 行**的硬性上限（按物理行计算，含注释与空行）。Tk 桌面层执行更为严格的预算：`apps/tk/app.py` 组合根不超过 300 行，每个控制器不超过 400 行，每个视图不超过 600 行。600 行设为模块预警线，超过 600 行的模块不再堆叠新职责。
- **保证代码可读性，严禁压缩式编码**：坚决反对为了规避行数上限而牺牲代码可读性。严禁将多条独立语句拼接至单行、严禁将清晰的多分支判断压缩成难以理解的超长三元表达式或复杂嵌套推导式、严禁删除合理的垂直空行。
- **统一格式、风格依从与改动自洁**：全库由 `ruff==0.14.3` 与 `ruff.toml` 统一格式化标准（100 列、双引号、4 空格）；体量 allowlist（`tests/architecture_allowlist.py`）遵循只减不增的棘轮机制。修改既有模块必须遵循现有命名与结构风格；开发引入的临时调试语句、未引用导入、无用变量及过渡函数必须在提交前彻底清理干净。

### 10. 真实业务闭环验证（拒绝假性通过与毒药测试）

- **面向真实业务闭环，拒绝保护分叉的毒药测试**：测试必须验证跨场景的业务语义一致性、接口契约与真实端到端路径（从输入经核心链路到最终产出），预期结果须由业务定义独立推导而非照抄被测实现输出。**严禁编写“断言某项公共修复不影响其他同类动作/入口”这类把局部特判合理化的毒药测试**；严禁让生成端与检查端共享同一错误假设而制造假性 PASS。
- **触发场景与兄弟场景必验**：凡因某个具体场景暴露出的共性缺陷，收尾验收时只需用少量有判别力的样例，同时贯通**触发该问题的原场景**与**代表性兄弟场景**，确认实际结果与呈现真正闭环即可，不为凑覆盖率扩成全组合重跑。
- **守牢核心门禁网**：日常提交以最小测试开销保证核心门禁与直接受影响套件通过（具体命令见下文 [测试门禁](#测试门禁)）。

### 11. 编码与资源约定

- **函数与类型约定**：函数保持小而清晰，公共接口与领域边界必须携带完整、明确的类型标注。
- **视频产物与转码**：视频导出的最终标准产物为 H.264 编码的 MP4。本机 OpenCV 编码器回退生成的 MJPG/XVID AVI 必须在出片后由系统 `ffmpeg` 自动转码（`-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p`），转码成功后清理临时 AVI 文件，严禁将回退的 AVI 直接作为最终产物交付。
- **禁止提交无关大文件**：禁止将模型权重文件（`.task` 等）、原始视频文件、以及运行输出目录（`outputs/`、`dist/`、`build/`）提交到 Git 仓库中。

### 12. 提交与 PR 协作规范

- **分支与独立 PR**：提交只进入本仓库，默认主分支 `main`。每个独立特性或任务单独建分支，工作区保持干净后提交，验收通过后由 `gh pr create --base main` 发起 PR（简单的纯文档/配置维护可由管理员直接 commit 推送 `main`）。
- **跨层协作与 PR 粒度**：跨多层的大型改动使用 stacked PR（`gh stack`）逐层递进，每一层独立通过门禁且具备独立审阅价值，避免一次性提交大而无当的超大 PR。
- **规范化提交与专项契约必读**：提交说明遵循 Conventional Commits（`feat:`、`fix:`、`refactor:`、`docs:`、`test:`、`chore:`）。**撰写 Issue 前必读 `spec/issue_spec.md`，撰写 PR 标题与正文前必读 `spec/pr_spec.md`，执行审查前必读 `spec/review_spec.md`**——统一使用简洁直白的中文，严格遵循对应模板的所有必填小节与表格，严禁偷工减料丢小节或隐瞒未闭环分支。
- **确认提交身份**：提交或推送前确认 Git / GitHub 身份（`gh auth status`、`git config user.name`、`git config user.email`）。

## 硬约束

- **MediaPipe 默认路径与基准零漂移**：系统专注于 MediaPipe 姿态识别与动作评分，`pose33_v3` 模板兼容性及 `infer()`/`annotate()` 行为语义严格保持一致。`tests/test_pose33_v3_golden.py`、`test_tech_eval_contract.py`、`test_valid_mask_migration.py` 必须保持零漂移；只有经批准的决策可以改变数值基准。
- **点位有效性统一判定准绳**：关键点有效性统一由 `core/pose/validity.py` 中的 `derive_valid_mask` 判定，严禁在业务代码中散落 `lm[idx, 3] >= thr` 之类的私有判据；结构性缺失点不得伪造成有效点。
- **用户数据格式向下兼容**：`%LOCALAPPDATA%\VisionSanda` 目录结构、`user_prefs.json` 键名、片段目录与文件名规则、练习历史 `schemaVersion=1`、成绩表列格式以及 `feedback_config.json` 的文件名与字段严格兼容。分数口径：内部统一为 0..1，导出时乘以 100，综合得分列统一使用 `combined_percent`。
- **测试环境数据隔离**：偏好文件路径只由 `core/foundation/prefs.py` 的 `_prefs_path()` 提供，`tests/conftest.py` 的 autouse fixture 将其钉在临时文件，防止测试污染真实用户数据；移动该函数时必须在同一提交内更新打桩目标。
- **打包与运行时独立**：严禁随意引入新增的外部运行时依赖。构建的单文件 EXE 必须能独立完整运行；子进程入口必须为模块顶层函数，参数保持可 pickle 序列化，保留 `freeze_support()`。
- **动作规则上线判定红线**：规则项与阶段实例启用必须具备真实样本标定证据：**正确可判样本误报率 ≤ 10%、目标缺陷检出率 ≥ 70% 且至少来自 2 名执行者**；无法判断或点位缺失不得伪造成通过或计入检出；`full` 与 `heavy` 姿态模型资格独立验证登记，不跨模型或跨阶段隐式继承；未达标项保持停用，不得擅自放宽姿势或角度阈值上线。
- **历史命令入口保持可用**：`apps/app_ui.py`、`apps/main.py` 等经典命令保持行为可用，入口壳保持极简（≤ 30 行），逻辑收敛于组合根与核心库中。

## 环境与命令

命令在本仓库根运行，按 PowerShell 写。用 Git Bash 时，路径写成 `./.venv/Scripts/python.exe`，环境变量用 `export`。

- 联网下载（pip 包、模型 `.task`、GitHub 等）一律走本地代理 `http://127.0.0.1:7879`（端口以 Windows 系统代理设置为准）：

  ```powershell
  $env:HTTP_PROXY  = "http://127.0.0.1:7879"
  $env:HTTPS_PROXY = "http://127.0.0.1:7879"
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt --proxy http://127.0.0.1:7879
  ```

- Python 用本仓库的 `.\.venv\Scripts\python.exe`；pip 一律用 `python -m pip`。不使用也不修改 `old/.venv`。
- MediaPipe 模型首次运行时自动下载到 `models/`。

```powershell
.\.venv\Scripts\python.exe apps/app_ui.py                  # Tk 桌面
.\.venv\Scripts\python.exe apps/main.py --source 0         # CLI 摄像头
.\.venv\Scripts\python.exe apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
.\.venv\Scripts\python.exe apps/make_template.py --video action.mp4 --pose heavy --preview
.\.venv\Scripts\python.exe apps/match_template.py --template template.npz --video test.mp4 --preview
.\.venv\Scripts\python.exe batch/batch_dual_compare.py --standard_dir "标准样本" --student_dir "学员样本" --pose full --out_dir "输出目录" --rules --rule-action stance
.\.venv\Scripts\python.exe scripts/sim_exam_flow.py         # 无摄像头模拟考试流程
.\.venv\Scripts\python.exe scripts/check_format.py origin/main  # 增量格式检查
.\.venv\Scripts\python.exe -m ruff format <文件路径>       # 代码格式化修复
```

- `batch_dual_compare.py --paired`：按学员文件名里的正/侧关键词配对并调用 `compare_dual_streams`，CSV 的 `video` 列写学员 id，`--export_raw` 在 paired 模式下跳过。
- 打包构建：单文件 spec 为 `app_ui_onefile.spec`，构建命令：`.\.venv\Scripts\python.exe -m PyInstaller app_ui_onefile.spec --noconfirm`；产物为 `dist/vision_ui.exe`。EXE 自检命令：`vision_ui.exe --offline-self-check <报告.json>`。

## 测试门禁

每个 PR 都要过：

- 增量格式检查：`.\.venv\Scripts\python.exe scripts/check_format.py origin/main`（包含已跟踪的工作区改动；新文件先 `git add`）。使用 `ruff==0.14.3` 和 `ruff.toml`，检查新增或修改的整个 `.py` / `.pyi` 文件；修复命令：`.\.venv\Scripts\python.exe -m ruff format <文件路径>`。
- golden 与契约：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- 架构测试：`.\.venv\Scripts\python.exe -m pytest tests/test_architecture.py -q`
- 直接受影响模块的最小套件（见下表；严格遵循 §4“单点验证与回答即停”，无关联反证严禁擅自扩跑全库套件或全量历史差分）
- `git diff --check`：行尾提示可以接受，空白错误不行

CI（`.github/workflows/gates.yml`，检查名 `gates`）在每个 PR 和 `main` 的推送上自动跑增量格式检查、golden / 契约、架构测试和空白检查，任何一步退出码非 0 即检查失败。格式检查以 PR 的 base SHA 或 push 的 before SHA 为基线，首次推送使用空树；排除已删除文件，没有 Python 改动时跳过。环境：`windows-latest`、Python 3.12，安装 `requirements.txt` 和 `requirements-dev.txt`；不用摄像头，也不预置 `models/` 下的模型，门禁测试只读 `tests/fixtures/`。受影响模块的套件和全量测试不在 CI 里，仍在本地跑。本地复现 CI 的空白检查：`git -c core.whitespace=cr-at-eol diff --check origin/main HEAD`。合入约束由仓库规则集 `main-gates`（作用于默认分支）强制：`gates` 是必需检查，未通过不能合并；`main` 只能经 PR 合入，禁止强推和删除。

| 改动范围 | 测试（`tests/` 下） |
|---|---|
| 考试系统 | `test_exam_session.py test_exam_roster.py test_exam_clip.py test_presence_gate.py test_recording_postprocess.py test_exam_panel.py`，再跑 `scripts/sim_exam_flow.py` |
| batch paired / 双流 | `test_batch_backend_args.py test_compare_dual_streams.py` |
| 布局与路由契约 | `test_body_core_layout.py test_backend_routing_contract.py` |
| 录制与性能基准 | `test_recording_controller.py test_recording_postprocess.py test_recording_benchmark.py test_benchmark_probes.py test_tk_benchmark.py` |
| 学生练习与报告 | `test_action_feedback.py test_student_presence.py test_feedback_report.py test_feedback_html_export.py test_report_html_golden.py test_feedback_result_panel.py test_evidence_contract.py test_evidence_budget.py test_timing_diagnostics.py` |
| 规则引擎 / 复用与动作计时 | `test_rule_availability.py test_rule_engine_entry.py test_rule_reuse.py test_rule_evidence_entries.py test_shared_motion_timing.py test_shared_phase_coverage.py test_shared_action_records.py test_step_straight_combo.py test_straight_refinements.py`（全量差分 `test_p3_eval_differential.py` 仅在明确更新差分基准时按需单跑） |
| 摄像头与运行时 | `test_camera_warmup.py test_camera_service.py test_camera_enum.py test_tk_cameras.py`；源码模式跑 `test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），再构建 EXE 跑 `--offline-self-check`；整套自检只能在冻结包里跑 |

全量基线和已知失败见 `docs/archive/重构阶段任务/README.md` 的"通用任务前检查"，不得新增失败。替换掉的测试要在 PR 里列出对应的新测试（D-13）。

## 变更记录

- 每次完成任务后，把修改总结写进 `change.md`，最新记录放在最上面。记录内容：修改日期、问题描述、修改内容、验证方法。只改文档时也要写同步依据，并至少做一次轻量验证。阶段 PR 要包含对应的记录。
- `change.md` 过长时，用 `git mv` 把它归档为 `docs/archive/change（<起>~<止月>）.md`，再新建一个空的 `change.md`，把"归档与重建"作为新文件的第一条记录，并更新下面的归档清单。
- 历史归档：本仓库 `docs/archive/change（2026.7~2026.11）.md`；更早归档在旧项目里：`old/change（start~2026.5）.md`、`old/change（2026.5~2026.6）.md`、`old/change（2026.6~2026.7）.md`。

## 代码地图

仓库严格遵循分层架构设计（设计文档 §2），从底层基础能力到顶层应用入口如下：

### 核心库 `core/`

分层架构体系（L0–L5）的核心模块与职责划分如下：

| 层级 | 模块包 | 核心职责与关键模块 |
|---|---|---|
| **L0** | `core/foundation/` | **系统基础能力**<br>• `paths.py`：应用数据与临时目录统一解析<br>• `prefs.py`：用户偏好设置持久化与隔离接口<br>• `atomic_io.py` / `npz.py` / `jsonable.py`：原子写盘与格式序列化<br>• `geometry.py` / `signal.py`：通用几何基元与信号去抖<br>• `host.py` / `power.py` / `memory.py` / `volume.py` / `winapi.py`：宿主机软硬件环境探针 |
| **L1** | `core/media/` | **媒体流采集与处理**<br>• `camera/`：摄像头硬件管理（`devices.py` 枚举、`service.py` 统一服务、`pool.py` 租约池、`reader.py` 独立读帧线程、`capture_process.py` 隔离子进程）<br>• `frames.py` / `frame_clock.py` / `pacing.py`：帧契约、采集时钟与录像节奏控制<br>• `writer.py` / `mjpeg_writer.py` / `ffmpeg.py`：视频写入与随包 ffmpeg 转码管道 |
| **L2** | `core/pose/` | **姿态识别与特征提取（专注 MediaPipe）**<br>• `engine.py` / `mediapipe_runtime.py`：MediaPipe Tasks 推理引擎运行时<br>• `extract.py` / `sequence.py`：姿态序列提取与时间序列封装<br>• `validity.py`：`derive_valid_mask` 关键点有效性判定准绳<br>• `layouts.py` / `features.py` / `normalize.py`：`pose33_v3` 关键点布局、归一化与特征对齐<br>• `render.py` / `annotate.py`：姿态骨架平滑与骨架视频导出 |
| **L3** | `core/evaluation/` | **动作编排、统一计时、对比与公共规则引擎**<br>• `action_catalog.py`：10 类动作编排、动作段组成与命名策略声明<br>• `action_timing.py` / `action_record.py` / `phase_alignment.py` / `phase_instances.py`：同步录像唯一计时来源选定、四入口共享动作记录与逐次机位覆盖核对<br>• `motion_events.py` / `punch_events.py` / `punch_phases.py` / `step_reps.py` / `stance_windows.py`：通用往返位移与保持识别、直拳事件/相位、上步组合逐次切分与稳定实战式取窗<br>• `measures/` / `paired_torso.py`：臂腿脚躯干、姿态与运动公共几何测量及双摄同步躯干朝向<br>• `rules/`：公共规则引擎（`engine.py` 调度、`rule_definitions.py` 统一规则定义、`catalog_measurements.py` 公共测量、`sanda_v5.py` 目录编排、`qualification.py` / `straight_policy.py` 独立模型与阶段资格登记、`evidence.py` 统一证据构建）<br>• `templates.py` / `matching/` / `scoring/` / `techeval/`：标准模板、`dtw.py` / `dual.py` / `repetition.py` 匹配、考试动作池评分与力序分析 |
| **L4** | `core/services/`<br>`core/recording/`<br>`core/live/`<br>`core/reporting/` | **领域服务层（平级自治，互不导入）**<br>• `services/`：离线分析服务（`analyze.py`）、模板生成（`templates.py`）、练习历史（`practice_history.py`）、录制基准评测（`recording_benchmark.py`）<br>• `recording/`：双摄录制控制（`recorder.py`）、录像工作线程（`worker.py`）、后处理链与骨架成片（`finalizer.py`、`steps.py`、`skeleton.py`）<br>• `live/`：实时交互会话（`runner.py`）、站位监测与 ROI 判定（`presence.py`）、实时预览流（`preview.py`）<br>• `reporting/`：结构化评测模型（`practice_model.py`）、公共优先级选图与证据预算（`evidence_selection.py`、`evidence.py`）、计时诊断与问题分组（`timing_diagnostics.py`、`issue_groups.py`）、文本与 HTML 报告生成（`text.py`、`html/`） |
| **L5** | `core/workflows/` | **业务工作流编排**<br>• `exam/`：考试考务完整状态机编排（`session.py` 考场状态、`roster.py` 名册台账、`clip.py` 动作剪裁、`scorebook.py` 成绩台账）<br>• `practice/`：学生练习会话编排（`session.py`）<br>• `presence.py` / `executor.py` / `plans.py`：考生进退场闸门与异步任务执行流 |

### 应用入口 `apps/`

消费 `core/` 公共 API 组织的交互界面与命令行入口：
- **Tkinter 桌面应用**（`apps/tk/`，入口壳 `apps/app_ui.py`）：
  - `app.py`：Tk 组合根与全局事件分发
  - `controllers/`：业务控制器（`cameras.py`、`recording.py`、`practice.py`、`exam.py`、`benchmark.py`、`session.py` 等）
  - `views/`：界面组件与视图布局（主窗、录制面板、练习控制、考试面板、练习分析结果页等）
  - `desktop_launcher.py`：EXE 打包自检与引导入口（`--offline-self-check`）
- **CLI 命令行工具**（`apps/cli/`，入口壳 `apps/main.py` 等）：
  - `main.py`：实时摄像头与离线视频骨架推理 CLI
  - `make_template.py` / `match_template.py`：模板录制生成与单/双模板匹配验证工具

### 离线工具、支持脚本与资源

- **离线分析与批处理**（`batch/`、`analysis/`）：
  - `batch/batch_dual_compare.py`：双摄大批量离线双模比对（含 paired 模式）
  - `batch/batch_export_skeleton.py` / `batch_tech_eval.py`：批量骨架导出与技术指标评估
  - `analysis/`：离线性能 Profile、标定测试与 Benchmark Harness
- **支持脚本**（`scripts/`）：
  - `check_format.py`：基于 Git变动的增量代码格式检查
  - `sim_exam_flow.py`：无硬件摄像头环境下的全流程考试状态机模拟器
  - `generate_feedback_icons.py`：练习结果页状态图标生成脚本
- **工程配置、规范与文档交付**：
  - `spec/`：Agent 专项任务强制规范（`issue_spec.md`、`pr_spec.md`、`review_spec.md`）
  - `docs/`：专项评估报告与历史重构设计归档（`docs/archive/`）
  - `ruff.toml`：全库代码格式化配置（100 列、双引号、4 空格缩进）
  - `app_ui_onefile.spec`：PyInstaller Windows 单文件 EXE 打包规范
  - `tests/`：pytest 契约与回归测试套件（含架构测试 `test_architecture.py` 与基准快照）
  - `assets/feedback/`：结果页与报告嵌入的状态图标资产
