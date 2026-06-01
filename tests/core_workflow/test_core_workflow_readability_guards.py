#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "scripts" / "core_workflow"
NUMBERED_STEPS = sorted(CORE.glob("[0-9][0-9]_*_*.py"))
MAX_FN_LINES = 250
LENGTH_WHITELIST = {
    "05_02_run_group_atlas.py::main",
    "06_03_build_cluster_cortex_panels.py::main",
    "06_01_build_group_overview.py::render_combo",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_module_functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node)
    return out


def test_no_local_save_json_now_utc_or_file_meta_helpers_in_numbered_steps() -> None:
    banned_prefixes = ("save_json", "now_utc", "file_meta")
    for path in NUMBERED_STEPS:
        tree = ast.parse(_source(path))
        for fn in _iter_module_functions(tree):
            if fn.name.startswith(banned_prefixes):
                raise AssertionError(f"banned local helper in {path.name}: {fn.name}")


def test_numbered_step_function_length_guard() -> None:
    for path in NUMBERED_STEPS:
        src = _source(path)
        tree = ast.parse(src)
        for fn in _iter_module_functions(tree):
            if fn.end_lineno is None:
                continue
            fn_len = int(fn.end_lineno) - int(fn.lineno) + 1
            key = f"{path.name}::{fn.name}"
            if key in LENGTH_WHITELIST:
                continue
            assert fn_len <= MAX_FN_LINES, f"{key} too long: {fn_len} > {MAX_FN_LINES}"


def main() -> int:
    test_no_local_save_json_now_utc_or_file_meta_helpers_in_numbered_steps()
    test_numbered_step_function_length_guard()
    print("ok: core workflow readability guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
