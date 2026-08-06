"""Regression checks for application docstrings and CLI documentation."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from stress_engine.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "stress_engine"
README = ROOT / "README.md"


class DocumentationCoverageTest(unittest.TestCase):
    """Keep code and user-facing command documentation from drifting."""

    def test_every_application_module_class_and_function_has_a_docstring(self):
        """Require docstrings on every definition shipped by the package."""
        missing = []
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative = path.relative_to(ROOT)
            if ast.get_docstring(tree, clean=False) is None:
                missing.append(f"{relative}: module")
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    continue
                if ast.get_docstring(node, clean=False) is None:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    missing.append(
                        f"{relative}:{node.lineno}: {kind} {node.name}"
                    )

        self.assertEqual(
            missing,
            [],
            "Application documentation gaps:\n" + "\n".join(missing),
        )

    def test_readme_lists_every_long_cli_option(self):
        """Require the README to mention every public long-form CLI flag."""
        documented = README.read_text(encoding="utf-8")
        options = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
            if option.startswith("--")
        }
        missing = sorted(
            option
            for option in options
            if not re.search(
                rf"`{re.escape(option)}(?:`|[ =])",
                documented,
            )
        )
        self.assertEqual(
            missing,
            [],
            "README is missing CLI options: " + ", ".join(missing),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
