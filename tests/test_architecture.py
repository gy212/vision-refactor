# -*- coding: utf-8 -*-
"""架构守护测试（设计文档 §13.2）：分层、无环、私有符号、原语单一入口、克隆、体量、职责、Tk 依赖、防御式访问。

只用标准库 ``ast``。现有违规登记在 ``tests/architecture_allowlist.py``，每条写明计划消除的阶段。
所有登记项都按棘轮校验：实际值高于登记值判失败（新增违规）；低于登记值也判失败，
提示在同一提交里把登记值改成实际值，防止降下来又涨回去。职责检查与 Tk 依赖不设 allowlist。
"""
from __future__ import annotations

import ast
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tests import architecture_allowlist as AL

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("core", "apps", "batch", "analysis", "scripts")
ENTRY_TOPS = ("apps", "batch", "analysis", "scripts")
MODULE_MAX_LINES = 800
MODULE_WARN_LINES = 600
FUNCTION_MAX_LINES = 100
CLONE_MIN_STATEMENTS = 6
PHASES = {f"P{i}" for i in range(1, 10)}

# §2.1 的分层；L4 的四个包互不导入。
LAYER_OF_PACKAGE = {
    "core.foundation": 0,
    "core.media": 1,
    "core.pose": 2,
    "core.evaluation": 3,
    "core.services": 4,
    "core.recording": 4,
    "core.live": 4,
    "core.reporting": 4,
    "core.workflows": 5,
}
_JUNK_NAME = re.compile(r"(^|_)(utils?|helpers?|common|misc)(_|$)")


@dataclass(frozen=True)
class Module:
    rel: str
    name: str
    is_pkg: bool
    tree: ast.Module
    lines: int


@dataclass(frozen=True)
class Import:
    target: str
    names: tuple[str, ...]
    lineno: int
    type_only: bool


def _physical_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


@lru_cache(maxsize=1)
def modules() -> dict[str, Module]:
    """扫描范围内的全部模块，键为点分模块名。"""
    found: dict[str, Module] = {}
    for top in SCAN_DIRS:
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            parts = rel[:-3].split("/")
            is_pkg = parts[-1] == "__init__"
            name = ".".join(parts[:-1] if is_pkg else parts)
            text = path.read_text(encoding="utf-8-sig")
            found[name] = Module(rel, name, is_pkg, ast.parse(text, filename=rel), _physical_lines(text))
    return found


def _type_checking_nodes(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                for child in node.body:
                    ids.update(id(n) for n in ast.walk(child))
    return ids


def _project_prefix(dotted: str) -> str | None:
    """点分名对应的最长项目模块名；不属于项目返回 None。"""
    mods = modules()
    parts = dotted.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in mods:
            return candidate
        parts.pop()
    return None


@lru_cache(maxsize=None)
def imports_of(name: str) -> tuple[Import, ...]:
    """模块内全部导入（含函数内的延迟导入），解析到项目模块；外部依赖忽略。"""
    mod = modules()[name]
    type_only = _type_checking_nodes(mod.tree)
    package = name if mod.is_pkg else name.rpartition(".")[0]
    out: list[Import] = []
    for node in ast.walk(mod.tree):
        flag = id(node) in type_only
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _project_prefix(alias.name)
                if target:
                    out.append(Import(target, (), node.lineno, flag))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base == "__future__":
                continue
            base_target = _project_prefix(base) if base else None
            plain: list[str] = []
            for alias in node.names:
                sub = f"{base}.{alias.name}" if base else alias.name
                if sub in modules():
                    out.append(Import(sub, (), node.lineno, flag))
                else:
                    plain.append(alias.name)
            if base_target and plain:
                out.append(Import(base_target, tuple(plain), node.lineno, flag))
    return tuple(out)


def layer_of(name: str) -> tuple[int, str]:
    """返回 (层号, 包标识)。旧的平铺模块按附录 B 的去向归层，见 allowlist 的 LEGACY_TARGETS。"""
    top = name.split(".")[0]
    if top in ENTRY_TOPS:
        return 6, top
    for pkg, layer in LAYER_OF_PACKAGE.items():
        if name == pkg or name.startswith(pkg + "."):
            return layer, pkg
    if name in AL.LEGACY_TARGETS:
        pkg = AL.LEGACY_TARGETS[name]
        return LAYER_OF_PACKAGE[pkg], pkg
    raise AssertionError(
        f"{name} 不属于 §2.2 的任何包：新代码放进 core/<层>/ 下，或在 LEGACY_TARGETS 登记旧模块的去向"
    )


def _is_l4(layer: int) -> bool:
    return layer == 4


def layering_violations() -> Counter:
    """只依赖更低层；L4 同层包互不导入；入口层只用包 ``__init__`` 导出的公共 API。"""
    found: Counter = Counter()
    for name, mod in modules().items():
        src_layer, src_pkg = layer_of(name)
        for imp in imports_of(name):
            dst_layer, dst_pkg = layer_of(imp.target)
            if imp.target == name:
                continue
            bad = dst_layer > src_layer or (
                _is_l4(src_layer) and _is_l4(dst_layer) and src_pkg != dst_pkg
            )
            if src_layer == 6 and dst_layer < 6 and imp.target.count(".") >= 2:
                if any(imp.target == pkg or imp.target.startswith(pkg + ".") for pkg in LAYER_OF_PACKAGE):
                    bad = True  # 入口层越过包 __init__ 直接导入子模块
            if bad:
                found[f"{mod.rel} -> {imp.target}"] += 1
    return found


def import_cycles() -> Counter:
    """项目模块导入图（不含 TYPE_CHECKING）的强连通分量，Tarjan 算法。"""
    graph = {
        name: sorted({i.target for i in imports_of(name) if not i.type_only and i.target != name})
        for name in modules()
    }
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for v in graph:
        if v not in index:
            visit(v)
    return Counter(" <-> ".join(c) for c in sccs)


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def private_imports() -> Counter:
    """跨模块导入或访问下划线开头的符号（``tests/`` 不在扫描范围内）。"""
    found: Counter = Counter()
    for name, mod in modules().items():
        aliases: dict[str, str] = {}
        for imp in imports_of(name):
            for sym in imp.names:
                if _private(sym) and imp.target != name:
                    found[f"{mod.rel}: {imp.target}.{sym}"] += 1
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _project_prefix(alias.name) == alias.name:
                        aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    for imp in imports_of(name):
                        if imp.lineno == node.lineno and not imp.names and imp.target.endswith("." + alias.name):
                            aliases[alias.asname or alias.name] = imp.target
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Attribute) and _private(node.attr):
                dotted = _dotted(node.value)
                if dotted in aliases and aliases[dotted] != name:
                    found[f"{mod.rel}: {aliases[dotted]}.{node.attr}"] += 1
    return found


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_name(node: ast.Call) -> str:
    return _dotted(node.func) or (node.func.attr if isinstance(node.func, ast.Attribute) else "")


def _is_file_replace(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = _call_name(node)
    if name in {"os.replace", "os.rename", "shutil.move"}:
        return True
    # Path.replace(dst) / Path.rename(dst)：单个位置参数、无关键字；str.replace 要两个参数
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"replace", "rename"}
        and len(node.args) == 1
        and not node.keywords
    )


def _call_ends(*suffixes: str):
    def match(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        name = _call_name(node)
        return any(name == s or name.endswith("." + s) for s in suffixes)
    return match


def _attr_named(*attrs: str):
    return lambda node: isinstance(node, ast.Attribute) and node.attr in attrs


def _string(value: str):
    return lambda node: isinstance(node, ast.Constant) and node.value == value


# 原语 → (匹配器, 允许出现的位置)。位置是相对路径前缀或精确文件。
PRIMITIVES = {
    "video_capture": (_call_ends("cv2.VideoCapture"), ("core/media/",)),
    "engine_construct": (_call_ends("MediaPipePipeline"), ("core/pose/engine.py",)),
    "np_load": (_call_ends("np.load", "numpy.load"), ("core/evaluation/templates.py",)),
    "subsequence_dtw": (_call_ends("subsequence_dtw"), ("core/evaluation/matching/",)),
    "file_replace": (_is_file_replace, ("core/foundation/atomic_io.py",)),
    "user_prefs_literal": (_string("user_prefs.json"), ("core/foundation/",)),
    "thread": (
        _call_ends("threading.Thread"),
        (
            "core/foundation/tasks.py",
            "core/pose/parallel.py",
            "core/media/camera/pool.py",
            "core/media/camera/capture_process.py",
        ),
    ),
    "tk_after": (_call_ends("after"), ("apps/tk/",)),
}
RULE_MATH = _attr_named("arccos", "norm")
RULE_PREFIX = "core/evaluation/rules/"


def _allowed(rel: str, places: tuple[str, ...]) -> bool:
    return any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in places)


def primitive_violations() -> Counter:
    found: Counter = Counter()
    for mod in modules().values():
        is_rule = mod.rel.startswith(RULE_PREFIX) or mod.name in AL.LEGACY_RULE_MODULES
        for node in ast.walk(mod.tree):
            for key, (match, places) in PRIMITIVES.items():
                if not _allowed(mod.rel, places) and match(node):
                    found[f"{key}: {mod.rel}"] += 1
            if is_rule and RULE_MATH(node):
                found[f"rule_math: {mod.rel}"] += 1
    return found


def _functions(tree: ast.Module):
    """(限定名, 节点)，限定名形如 ``Class.method`` 或 ``outer.<locals>.inner``。"""
    def walk(node: ast.AST, prefix: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}{child.name}"
                yield qual, child
                yield from walk(child, qual + ".<locals>.")
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, f"{prefix}{child.name}.")
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def clone_groups() -> Counter:
    """函数体 AST 完全相同（不少于 6 条语句）且分布在两个以上模块中的函数组。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for mod in modules().values():
        for qual, fn in _functions(mod.tree):
            body = fn.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                body = body[1:]
            statements = sum(1 for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.stmt))
            if statements < CLONE_MIN_STATEMENTS:
                continue
            key = ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
            groups[key].append(f"{mod.rel}::{qual}")
    found: Counter = Counter()
    for members in groups.values():
        if len({m.split("::")[0] for m in members}) > 1:
            found[" == ".join(sorted(members))] += 1
    return found


def module_sizes() -> dict[str, int]:
    return {m.rel: m.lines for m in modules().values()}


def function_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for mod in modules().values():
        for qual, fn in _functions(mod.tree):
            key = f"{mod.rel}::{qual}"
            sizes[key] = max(sizes.get(key, 0), fn.end_lineno - fn.lineno + 1)
    return sizes


def defensive_getattr() -> Counter:
    """``apps/`` 中 ``getattr(self, "_…")`` 的个数（Tk 层目前在 apps/ 下，P8 起在 apps/tk/）。"""
    found: Counter = Counter()
    for mod in modules().values():
        if not mod.rel.startswith("apps/"):
            continue
        for node in ast.walk(mod.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value.startswith("_")
            ):
                found[mod.rel] += 1
    return found


# --------------------------------------------------------------------------- #
# 棘轮
# --------------------------------------------------------------------------- #


def ratchet(category: str, actual: Mapping[str, int], allowed: Mapping[str, tuple[int, str]]) -> list[str]:
    """逐条比对实际值与登记值，返回错误信息；两者必须完全相等。"""
    errors: list[str] = []
    for key, value in sorted(actual.items()):
        if key not in allowed:
            errors.append(f"[{category}] 新增违规：{key}（{value}）")
            continue
        limit, phase = allowed[key]
        if phase not in PHASES:
            errors.append(f"[{category}] {key} 的计划阶段 {phase!r} 无效")
        if value > limit:
            errors.append(f"[{category}] {key} 为 {value}，超过登记值 {limit}")
        elif value < limit:
            errors.append(f"[{category}] {key} 已降到 {value}，请在同一提交里把登记值 {limit} 改成 {value}")
    for key in sorted(set(allowed) - set(actual)):
        errors.append(f"[{category}] {key} 已消除，请从 allowlist 删除该条")
    return errors


def _over(sizes: Mapping[str, int], limit: int) -> dict[str, int]:
    return {k: v for k, v in sizes.items() if v > limit}


def _check(category: str, actual: Mapping[str, int], allowed: Mapping[str, tuple[int, str]]) -> None:
    errors = ratchet(category, actual, allowed)
    assert not errors, "\n".join(errors)


def test_layering():
    _check("layering", layering_violations(), AL.LAYERING)


def test_no_import_cycles():
    _check("cycles", import_cycles(), AL.CYCLES)


def test_no_cross_module_private_symbols():
    _check("private", private_imports(), AL.PRIVATE)


def test_primitives_have_single_entry():
    _check("primitives", primitive_violations(), AL.PRIMITIVES)


def test_no_cross_module_clones():
    _check("clones", clone_groups(), AL.CLONES)


def test_module_size_budget():
    sizes = module_sizes()
    _check("module_size", _over(sizes, MODULE_MAX_LINES), AL.MODULE_SIZE)
    for rel, lines in sorted(sizes.items()):
        if MODULE_WARN_LINES < lines <= MODULE_MAX_LINES:
            warnings.warn(f"{rel} 有 {lines} 行，超过 600 行预警线，不再接收新职责", stacklevel=1)


def test_function_size_budget():
    _check("function_size", _over(function_sizes(), FUNCTION_MAX_LINES), AL.FUNCTION_SIZE)


def test_defensive_getattr_only_shrinks():
    _check("defensive_getattr", defensive_getattr(), AL.DEFENSIVE_GETATTR)


def test_allowlist_entry_count_only_shrinks():
    tables = (AL.LAYERING, AL.CYCLES, AL.PRIVATE, AL.PRIMITIVES, AL.CLONES,
              AL.MODULE_SIZE, AL.FUNCTION_SIZE, AL.DEFENSIVE_GETATTR)
    total = sum(len(t) for t in tables)
    assert total <= AL.ENTRY_CEILING, f"allowlist 共 {total} 条，超过上限 {AL.ENTRY_CEILING}：条数只减不增"
    assert total == AL.ENTRY_CEILING, f"allowlist 已降到 {total} 条，请把 ENTRY_CEILING 改成 {total}"


def test_every_module_declares_responsibility():
    missing = []
    for mod in modules().values():
        doc = ast.get_docstring(mod.tree)
        if not doc or not doc.strip():
            missing.append(mod.rel)
    assert not missing, "缺少写明职责的模块 docstring：\n" + "\n".join(missing)


def test_no_junk_drawer_modules():
    bad = [m.rel for m in modules().values()
           if any(_JUNK_NAME.search(part) for part in m.rel[:-3].split("/"))]
    assert not bad, f"不建 utils/helpers/common/misc 类模块：{bad}"


def test_init_modules_only_export():
    bad = []
    for mod in modules().values():
        if not mod.is_pkg:
            continue
        for i, stmt in enumerate(mod.tree.body):
            if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(stmt, ast.Assign) and [getattr(t, "id", None) for t in stmt.targets] == ["__all__"]:
                continue
            bad.append(f"{mod.rel}:{stmt.lineno}")
    assert not bad, "__init__.py 只含 docstring、导入和 __all__：" + ", ".join(bad)


def test_core_does_not_import_tk():
    bad = []
    for mod in modules().values():
        if not mod.rel.startswith("core/"):
            continue
        for node in ast.walk(mod.tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                if n == "tkinter" or n.startswith("tkinter.") or n == "PIL.ImageTk":
                    bad.append(f"{mod.rel}:{node.lineno} {n}")
    assert not bad, "core/ 不得导入 tkinter 或 PIL.ImageTk：\n" + "\n".join(bad)


def test_every_scanned_module_has_a_layer():
    for name in modules():
        layer_of(name)


# --------------------------------------------------------------------------- #
# 棘轮本身的自检：登记值多一或少一都必须失败
# --------------------------------------------------------------------------- #


def test_ratchet_rejects_growth_and_stale_limits():
    allowed = {"a.py": (900, "P3")}
    assert ratchet("t", {"a.py": 900}, allowed) == []
    assert ratchet("t", {"a.py": 901}, allowed)
    assert ratchet("t", {"a.py": 899}, allowed)
    assert ratchet("t", {}, allowed)
    assert ratchet("t", {"a.py": 900, "b.py": 801}, allowed)
    assert ratchet("t", {"a.py": 900}, {"a.py": (900, "P10")})


def test_size_entries_track_real_line_counts():
    """每个体量条目都对应真实存在的模块或函数，行数与登记值逐条一致（多一行、少一行都失败）。"""
    sizes = module_sizes()
    for rel, (limit, _phase) in AL.MODULE_SIZE.items():
        assert rel in sizes, f"{rel} 不存在"
        assert ratchet("module_size", {rel: sizes[rel]}, {rel: (limit, _phase)}) == []
        assert ratchet("module_size", {rel: limit + 1}, {rel: (limit, _phase)})
        assert ratchet("module_size", {rel: limit - 1}, {rel: (limit, _phase)})
