# Python / CLI / AST 技术栈边界样例包

本文件是 Python / CLI / AST 静态分析技术栈具象边界与踩坑样例的唯一事实来源（SSOT），供 `specs/engineering_contract.md`（测试契约 `R-TEST`）与 `specs/review_contract.md`（代码级证伪）单向引用。其他技术栈可按同类结构新增独立样例包。

## 1. 外部协议与 Shell / Hook / OS 边界

- **宿主 Hook 协议字段与退出码真值**：
  - 严格核对宿主真实 JSON 载荷与返回契约（例如 Antigravity 的 `toolCall` / 放行 `{"decision": "ask"}` 与拦截 `{"decision": "deny", "reason": "..."}`；Claude Code 的 `tool_input` / 放行 `{}` 与拦截 `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "..."}}`；Codex 的放行 `{"allow": true, "action": "proceed"}` 与拦截 `action: "block"|"intercept"` + `decision: "block"` + `hookSpecificOutput`）。
  - 严禁自造 `approve` 等虚构字段，严禁在 Antigravity 放行路径返回越权跳过宿主用户确认弹窗的 `{"decision": "allow"}`，且必须同步推翻保护假协议字段的同谋测试。
- **Shell 只读快路径绕过**：
  - 除主命令名外，必须覆盖 `>` / `>>` 输出重定向、管道写入与未闭合引号，防止写操作借只读快路径绕过拦截。
- **Bash 退出码捕获时机**：
  - `if ...; then ...; fi` 语句返回已执行分支中最后一条命令的退出状态，无分支命中且无 `else` 时返回 `0`；因此失败条件的退出状态可能被整个 `if` 的退出码掩盖，须在失败点立刻捕获原始失败退出码，不得在块外事后读取 `$?`。
- **解释器优先级与虚拟环境软链接**：
  - 显式指定解释器时须防全局 `$PATH` 旧命令劫持。
  - 保留虚拟环境入口软链接路径必须使用 `os.path.abspath`，严禁使用解引用破坏 venv 隔离的 `Path.resolve()`。
- **多用户共享目录安全**：
  - 共享临时目录或跨进程状态文件落盘时，必须校验目录属主（`uid`）、收紧权限位（`0700` / `0600`）并防止预置软链接借原子替换覆盖任意文件。

## 2. 语法词法与导入等价变体

- **Python 导入语法全集**：
  - AST 导入规则除检查 `import a.b` 外，必须同时穿透 `from a import b`、相对导入 `from . import b` / `from .. import b` 的子项列表（`imp.names`）以及嵌套属性访问链，避免只查包基名而放行子模块跨层违规或私有符号访问。
- **CLI 参数等价形式**：
  - 同时验证空格分隔（`--opt val`）、等号连接（`--opt=val`）与等价短参数（`-o val`），避免仅按空格拆分而漏判。
- **词法切分替代子串误伤**：
  - 命令与参数识别须用 `shlex` 等词法切分替代裸子串匹配，覆盖单双引号字面量（如 `echo 'pytest'`、`grep "md5sum"` 不触发误拦截）与未闭合引号回退路径。

## 3. 路径坐标系与过滤对偶

- **相对扫描根路径匹配**：
  - 忽略目录与排除模式必须基于 `relative_to(root)` 且严格区分目录段与文件名，严禁拿绝对路径 `parts` 匹配忽略词——否则项目位于 `/work/test/...` 等含同名祖先目录时会导致全仓静默空跑漏检。
- **包布局与映射双向对偶**：
  - **防漏扫**：同时覆盖常规包布局、仓库根目录单文件模块，以及 `src/` 布局（显式区分含 `src/__init__.py` 与不含 `src/__init__.py`，避免残留 `src.` 前缀导致分层规则失配）。
  - **防误判**：禁止将短层名无差别拼接至所有顶层包，防止把合法同名业务子模块（如 `api/db.py`）误判为底层逆向依赖。
- **共享配置可移植性与求值时机**：
  - 入库共享配置（如 `.agents/hooks.json`、`.claude/settings.json`）默认写入可移植命令（如 `square hook ...`），严禁硬编码宿主机私有绝对路径。
  - 核查依赖当前工作目录或环境变量的配置是否在模块导入期被错误冻结为全局常量。

## 4. 有状态生命周期与 Python 类型陷阱

- **缓存与会话快照四态闭环**：
  - **创建与更新**：验证首次建基线落盘，以及目标文件改动后缓存正确失效并更新。
  - **删除（`unlink`）**：验证外部删除文件且其余存留文件 `mtime` 均未变化时，缓存指纹仍能准确失效而不误命中过期结果。
  - **连续二次重入**：验证连续执行不同命令或中途并发改动时，持久化逻辑不会错误覆写初始时间戳或初始文件快照基线。
- **Python 布尔与整型混淆陷阱**：
  - Python 中 `isinstance(True, int)` 为 `True`；校验整型配置项、行数预算或计数字段时必须显式排除 `bool`（`not isinstance(v, bool)`），防止布尔值混入整型语义。
