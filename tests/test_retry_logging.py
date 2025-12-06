"""
Test that all @retry decorators have proper exception logging.

This test enforces the project convention that all retry logic must include
exception details in the before_sleep callback. This prevents silent retries
where we don't know why they're happening.

IMPORTANT: If this test fails, use `retry_with_logging()` from
`pdf2epub.utils.retry_utils` instead of raw `tenacity.retry`.
"""

import ast
from pathlib import Path
from typing import List, Dict
import pytest


class RetryUsageChecker(ast.NodeVisitor):
    """AST visitor that checks @retry decorator usage."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._check_decorators(node)
        self.generic_visit(node)

    def _check_decorators(self, node):
        for decorator in node.decorator_list:
            if self._is_raw_retry_decorator(decorator):
                if not self._has_before_sleep_with_exception(decorator):
                    self.violations.append({
                        'file': self.filename,
                        'line': decorator.lineno,
                        'function': node.name,
                        'message': '@retry without before_sleep that logs exception()'
                    })

    def _is_raw_retry_decorator(self, node) -> bool:
        """Check if this is a raw @retry decorator (not our wrapper)."""
        if isinstance(node, ast.Call):
            func = node.func
            # Direct @retry(...) call
            if isinstance(func, ast.Name) and func.id == 'retry':
                return True
            # @tenacity.retry(...) call
            if isinstance(func, ast.Attribute) and func.attr == 'retry':
                return True
        return False

    def _has_before_sleep_with_exception(self, node: ast.Call) -> bool:
        """Check if before_sleep callback includes .exception() call."""
        for keyword in node.keywords:
            if keyword.arg == 'before_sleep':
                return self._is_valid_before_sleep(keyword.value)
        return False

    def _is_valid_before_sleep(self, node) -> bool:
        """
        Check if before_sleep value is valid:
        1. Contains .exception() call directly (lambda case)
        2. Calls a known safe function (_create_method_before_sleep)
        """
        # Check for direct .exception() call in lambda
        if self._contains_exception_call(node):
            return True

        # Check for known safe factory functions
        if isinstance(node, ast.Call):
            func = node.func
            # _create_method_before_sleep("operation_name")
            if isinstance(func, ast.Name):
                if func.id in ('_create_method_before_sleep', 'create_before_sleep_callback'):
                    return True
            # module._create_method_before_sleep(...)
            if isinstance(func, ast.Attribute):
                if func.attr in ('_create_method_before_sleep', 'create_before_sleep_callback'):
                    return True

        return False

    def _contains_exception_call(self, node) -> bool:
        """Recursively check if AST node contains .exception() method call."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr == 'exception':
                        return True
        return False


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files, excluding venv and test files."""
    files = []
    for path in root_dir.rglob('*.py'):
        path_str = str(path)
        # Exclude virtual environments
        if '.venv' in path_str or 'venv' in path_str:
            continue
        # Exclude this test file
        if 'test_retry_logging' in path_str:
            continue
        # Exclude retry_utils.py itself - it's the wrapper implementation
        if path.name == 'retry_utils.py':
            continue
        files.append(path)
    return files


def test_all_retry_decorators_log_exceptions():
    """
    Ensure all @retry decorators include exception logging in before_sleep.

    This is a static analysis test that scans all Python files in the project
    and verifies that any use of @retry includes a before_sleep callback
    that calls .exception() on the retry state.

    If you need to add retry logic, use:
        from pdf2epub.utils.retry_utils import retry_with_logging

        @retry_with_logging(
            operation_name="My operation",
            retry_condition=is_transient_error,
        )
        def my_function():
            ...
    """
    project_root = Path(__file__).parent.parent / 'pdf2epub'

    all_violations = []

    for py_file in find_python_files(project_root):
        try:
            source = py_file.read_text(encoding='utf-8')
            tree = ast.parse(source)

            checker = RetryUsageChecker(str(py_file.relative_to(project_root.parent)))
            checker.visit(tree)

            all_violations.extend(checker.violations)
        except SyntaxError as e:
            # Skip files with syntax errors
            print(f"Warning: Could not parse {py_file}: {e}")

    if all_violations:
        violation_msgs = [
            f"  {v['file']}:{v['line']} in {v['function']}(): {v['message']}"
            for v in all_violations
        ]

        pytest.fail(
            f"\nFound {len(all_violations)} @retry usage(s) without proper exception logging:\n\n"
            + "\n".join(violation_msgs)
            + "\n\n"
            + "Fix: Use retry_with_logging() from pdf2epub.utils.retry_utils instead of raw @retry.\n"
            + "See: pdf2epub/utils/retry_utils.py for usage examples."
        )
