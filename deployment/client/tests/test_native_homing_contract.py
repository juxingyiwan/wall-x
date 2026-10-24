"""Regression guard for the native B601 entry point; no hardware access."""

import ast
from pathlib import Path


def test_native_follower_has_no_motion_mixin():
    source = Path(__file__).resolve().parents[1] / "src/wallx_client/hardware/b601.py"
    tree = ast.parse(source.read_text())
    follower = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GuardedFollower"
    )
    assert [ast.unparse(n) for n in follower.bases] == ["SeeedB601RSFollower"]
    assert not any(
        isinstance(n, ast.ImportFrom) and (n.module or "").startswith("b601_motion")
        for n in ast.walk(tree)
    )
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "_home" and n.decorator_list
        for n in ast.walk(tree)
    )
