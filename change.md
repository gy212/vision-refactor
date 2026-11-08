## 2026-11-08: [chore] change.md 归档与重建

- 问题描述：`change.md` 累积至 2936 行（覆盖 2026-07-09 ~ 2026-11-08），按变更记录规范需归档并重建新文件。
- 修改内容：
  1. `git mv change.md "change（2026.7~2026.11）.md"`，新建 `change.md` 并以本条作为首条记录。
  2. 同步更新 `AGENTS.md` 归档清单：新增本仓库 `change（2026.7~2026.11）.md`。
- 验证方法：`git status` 与 `git -c core.whitespace=cr-at-eol diff --cached --check` 检查通过，`pytest tests/test_architecture.py -q` 通过。
