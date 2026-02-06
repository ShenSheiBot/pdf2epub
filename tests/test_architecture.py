"""
Architecture tests: ensure no one violates design principles.

These tests use AST analysis to verify code structure at import time.
Violations will cause CI to fail.

What we check:
1. Processors don't define validation/saving methods
2. Processors don't import forbidden classes
3. Hooks don't define executor/state methods
4. Hooks don't import executor/state classes
5. Executor doesn't define validation logic (delegates to hooks)
6. Core classes are properly frozen
7. All processors implement ProcessorProtocol
"""

import ast
import pytest
from pathlib import Path
from typing import Set


# ========== CONFIGURATION ==========

# Methods that processors are NOT allowed to define
FORBIDDEN_PROCESSOR_METHODS: Set[str] = {
    # Validation methods
    'validate',
    'validate_output',
    '_validate',
    '_validate_batch',
    '_batch_validate',
    '_batch_validate_and_save',

    # Saving methods
    'save',
    'save_raw',
    '_save',
    '_save_result',
    '_save_raw',

    # State methods
    'is_complete',
    'mark_complete',
    'get_pending',
}

# Classes that processors are NOT allowed to import
FORBIDDEN_PROCESSOR_IMPORTS: Set[str] = {
    # State management
    'ProcessingTracker',
    'BatchTranslateState',
    'BatchPolishState',
    'ResultPersistence',
    'StateManager',
    # Validation runners
    'IndividualValidationRunner',
    'BatchValidationRunner',
    # Executor (new architecture)
    'Executor',
    'UnitState',
    'CompositeHooks',
    'ErrorEffect',
}

# Methods that hooks are NOT allowed to define
FORBIDDEN_HOOK_METHODS: Set[str] = {
    # Executor methods
    'execute',
    '_process_single',
    '_process_batch',
    '_process_online',
    '_handle_failure',
    '_get_ready_ids',
    # Processor methods
    'build_prompt',
    'get_model_configs',
    # Persistence methods
    'save_raw',
    'save_validated',
    'promote_batch',
}

# Classes that hooks are NOT allowed to import
FORBIDDEN_HOOK_IMPORTS: Set[str] = {
    # Executor internals
    'Executor',
    'UnitState',
    'ThreadPoolExecutor',
    'Future',
    # Persistence
    'ResultPersistence',
}

# Methods that should NOT be in executor (belongs to hooks)
FORBIDDEN_EXECUTOR_METHODS: Set[str] = {
    # Validation methods (belongs to hooks)
    'validate',
    'validate_output',
    '_validate_result',
    'check_truncation',
    # Transform methods (belongs to hooks)
    'restore_images',
    'remove_artifacts',
}

# Directories to check
PROCESSORS_DIR = Path('pdf2epub/processors_v2')
CORE_DIR = Path('pdf2epub/core')
VALIDATORS_DIR = Path('pdf2epub/validators')
HOOKS_DIR = Path('pdf2epub/core/hooks')
EXECUTOR_DIR = Path('pdf2epub/core/executor')


# ========== AST ANALYSIS HELPERS ==========

def get_python_files(directory: Path, recursive: bool = False) -> list:
    """Get all Python files in a directory."""
    if not directory.exists():
        return []
    if recursive:
        return [f for f in directory.rglob('*.py') if not f.name.startswith('_')]
    return [f for f in directory.glob('*.py') if not f.name.startswith('_')]


def get_defined_methods(tree: ast.AST) -> list:
    """Get all method definitions from an AST."""
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            methods.append((node.name, node.lineno))
        elif isinstance(node, ast.AsyncFunctionDef):
            methods.append((node.name, node.lineno))
    return methods


def get_imports(tree: ast.AST) -> list:
    """Get all imported names from an AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
    return imports


def get_class_names(tree: ast.AST) -> list:
    """Get all class names defined in an AST."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return classes


# ========== TESTS ==========

class TestProcessorArchitecture:
    """Tests for processor code structure."""

    def test_no_forbidden_methods_in_processors(self):
        """Ensure processors don't define forbidden methods."""
        violations = []

        for py_file in get_python_files(PROCESSORS_DIR):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            methods = get_defined_methods(tree)

            for method_name, lineno in methods:
                if method_name in FORBIDDEN_PROCESSOR_METHODS:
                    violations.append(
                        f"{py_file.name}:{lineno} - forbidden method '{method_name}'"
                    )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Forbidden methods in processors\n"
                f"{'='*60}\n"
                f"Processors must NOT define validation/saving/state methods.\n"
                f"These are handled by ProcessingPipeline.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )

    def test_no_forbidden_imports_in_processors(self):
        """Ensure processors don't import forbidden classes."""
        violations = []

        for py_file in get_python_files(PROCESSORS_DIR):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            imports = get_imports(tree)

            for import_name, lineno in imports:
                if import_name in FORBIDDEN_PROCESSOR_IMPORTS:
                    violations.append(
                        f"{py_file.name}:{lineno} - forbidden import '{import_name}'"
                    )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Forbidden imports in processors\n"
                f"{'='*60}\n"
                f"Processors must NOT import validation/persistence/state classes.\n"
                f"These are injected by the command layer.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )

    def test_no_file_io_in_processors(self):
        """Ensure processors don't do direct file I/O."""
        violations = []
        forbidden_attrs = {'write_text', 'read_text', 'mkdir', 'open'}

        for py_file in get_python_files(PROCESSORS_DIR):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in forbidden_attrs:
                            violations.append(
                                f"{py_file.name}:{node.lineno} - direct file I/O '{node.func.attr}'"
                            )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Direct file I/O in processors\n"
                f"{'='*60}\n"
                f"Processors must NOT do file I/O directly.\n"
                f"Use ResultPersistence instead.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )


class TestHooksArchitecture:
    """Tests for hooks code structure (new architecture)."""

    def test_no_forbidden_methods_in_hooks(self):
        """Ensure hooks don't define executor/state methods."""
        violations = []

        for py_file in get_python_files(HOOKS_DIR):
            # Skip protocol file (it's allowed to define method signatures)
            if py_file.name == '_protocol.py':
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            methods = get_defined_methods(tree)

            for method_name, lineno in methods:
                if method_name in FORBIDDEN_HOOK_METHODS:
                    violations.append(
                        f"{py_file.name}:{lineno} - forbidden method '{method_name}'"
                    )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Forbidden methods in hooks\n"
                f"{'='*60}\n"
                f"Hooks must NOT define executor/state/processor methods.\n"
                f"Hooks only handle: pre-processing, transform, validate, classify errors.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )

    def test_no_forbidden_imports_in_hooks(self):
        """Ensure hooks don't import executor/state classes."""
        violations = []

        for py_file in get_python_files(HOOKS_DIR):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            imports = get_imports(tree)

            for import_name, lineno in imports:
                if import_name in FORBIDDEN_HOOK_IMPORTS:
                    violations.append(
                        f"{py_file.name}:{lineno} - forbidden import '{import_name}'"
                    )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Forbidden imports in hooks\n"
                f"{'='*60}\n"
                f"Hooks must NOT import executor/state/persistence classes.\n"
                f"Hooks are pure functions operating on content.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )


class TestExecutorArchitecture:
    """Tests for executor code structure (new architecture)."""

    def test_no_validation_logic_in_executor(self):
        """Ensure executor delegates validation to hooks."""
        violations = []

        for py_file in get_python_files(EXECUTOR_DIR):
            # Skip protocol file
            if py_file.name == '_protocol.py':
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            methods = get_defined_methods(tree)

            for method_name, lineno in methods:
                if method_name in FORBIDDEN_EXECUTOR_METHODS:
                    violations.append(
                        f"{py_file.name}:{lineno} - forbidden method '{method_name}'"
                    )

        if violations:
            pytest.fail(
                f"\n{'='*60}\n"
                f"ARCHITECTURE VIOLATION: Validation logic in executor\n"
                f"{'='*60}\n"
                f"Executor must NOT contain validation/transform logic.\n"
                f"These are handled by CompositeHooks.\n"
                f"\n"
                f"Violations found:\n" +
                "\n".join(f"  - {v}" for v in violations) +
                f"\n{'='*60}"
            )


class TestCoreArchitecture:
    """Tests for core module structure."""

    def test_core_classes_have_final_methods(self):
        """Verify core classes mark critical methods as @final."""
        # This is a structural check - the actual @final behavior
        # is enforced at runtime by check_final_methods

        for py_file in get_python_files(CORE_DIR):
            if py_file.name.startswith('_'):
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            # Check for @final decorator usage
            has_final = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id == 'final':
                            has_final = True
                            break

            # Not a hard requirement, but good to have
            if not has_final and 'pipeline' in py_file.name:
                pytest.skip(f"{py_file.name} should have @final methods (warning)")


class TestValidatorArchitecture:
    """Tests for validator code structure (pdf2epub/validators/)."""

    def test_validators_have_name_property(self):
        """Verify validators have name property."""
        for py_file in get_python_files(VALIDATORS_DIR):
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Skip base classes and non-validator classes
                    skip_patterns = ['Base', 'Protocol', 'File', 'Tools', 'Result', 'State']
                    if any(p in node.name for p in skip_patterns):
                        continue
                    # Only check classes ending with Validator
                    if not node.name.endswith('Validator'):
                        continue
                    classes.append(node)

            for cls_node in classes:
                has_name = False

                for item in cls_node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == 'name':
                            # Check if it's a property
                            for decorator in item.decorator_list:
                                if isinstance(decorator, ast.Name) and decorator.id == 'property':
                                    has_name = True

                if not has_name:
                    pytest.fail(
                        f"Validator {cls_node.name} in {py_file.name} "
                        f"missing required 'name' property"
                    )


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_processors_implement_protocol(self):
        """Verify processors implement ProcessorProtocol."""
        try:
            from pdf2epub.core._protocol import ProcessorProtocol
            from pdf2epub.core.registry import ComponentRegistry
        except ImportError:
            pytest.skip("Core modules not importable")

        for name in ComponentRegistry.list_processors():
            processor = ComponentRegistry.get_processor(name, config={}, book_title="test")
            assert isinstance(processor, ProcessorProtocol), \
                f"Processor '{name}' doesn't implement ProcessorProtocol"

    def test_validators_implement_protocol(self):
        """Verify validators implement ValidatorProtocol."""
        try:
            from pdf2epub.core._protocol import ValidatorProtocol
            from pdf2epub.core.registry import ComponentRegistry
        except ImportError:
            pytest.skip("Core modules not importable")

        for name in ComponentRegistry.list_validators():
            validator = ComponentRegistry.get_validator(name)
            assert isinstance(validator, ValidatorProtocol), \
                f"Validator '{name}' doesn't implement ValidatorProtocol"

    def test_hooks_implement_protocols(self):
        """Verify hooks implement their respective protocols."""
        try:
            from pdf2epub.core.hooks import (
                PreProcessor, Transformer, Validator, SkipValidator, ErrorClassifier,
                EmptyContentFilter, ImageOnlyFilter,
                RestoreImagesTransformer, RemoveArtifactsTransformer,
                LengthRatioValidator, TruncationValidator,
                ChapterTypeSkipper, ShortContentSkipper,
                DefaultErrorClassifier,
            )
        except ImportError:
            pytest.skip("Hooks modules not importable")

        # Pre-processors
        assert isinstance(EmptyContentFilter(), PreProcessor)

        # Transformers
        assert isinstance(RestoreImagesTransformer(), Transformer)
        assert isinstance(RemoveArtifactsTransformer(), Transformer)

        # Validators
        assert isinstance(LengthRatioValidator(), Validator)
        assert isinstance(TruncationValidator(), Validator)

        # Skip validators
        assert isinstance(ChapterTypeSkipper(), SkipValidator)
        assert isinstance(ShortContentSkipper(), SkipValidator)

        # Error classifiers
        assert isinstance(DefaultErrorClassifier(), ErrorClassifier)


class TestFrozenEnforcement:
    """Tests for frozen class enforcement."""

    def test_cannot_inherit_frozen_class(self):
        """Verify frozen classes cannot be inherited."""
        try:
            from pdf2epub.core._frozen import Frozen, FrozenMeta
        except ImportError:
            pytest.skip("Frozen module not importable")

        # Create a frozen class
        class TestFrozen(Frozen, frozen=True):
            pass

        # Try to inherit from it - should raise TypeError
        with pytest.raises(TypeError) as exc_info:
            class BadChild(TestFrozen):
                pass

        assert "frozen class" in str(exc_info.value).lower()

    def test_forbidden_methods_raise_error(self):
        """Verify forbidden methods in subclass raise TypeError."""
        try:
            from pdf2epub.core._frozen import Frozen
        except ImportError:
            pytest.skip("Frozen module not importable")

        class TestBase(Frozen):
            _FORBIDDEN_METHODS = {'forbidden_method'}

        with pytest.raises(TypeError) as exc_info:
            class BadSubclass(TestBase):
                def forbidden_method(self):
                    pass

        assert "forbidden" in str(exc_info.value).lower()


# ========== INTEGRATION TESTS ==========

class TestIntegration:
    """Integration tests for the new architecture."""

    def test_can_create_executor(self):
        """Verify Executor can be instantiated."""
        try:
            from pdf2epub.core.executor import Executor, ChainEntry, QuotaConfig
            from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
        except ImportError:
            pytest.skip("Required modules not importable")

        # Mock objects
        class MockProcessor:
            name = "test"
            def build_prompt(self, content, context):
                return f"Process: {content}"
            def clean_response(self, response):
                return response
            def post_process(self, content, context):
                return content
            def get_model_configs(self):
                return [{"provider": "test", "model": "test"}]

        class MockLLMClient:
            def generate(self, prompt, model_configs, operation_name):
                return "Processed"

        chain = [ChainEntry(provider="test", model="test", mode="online")]
        hooks = CompositeHooks(error_classifier=DefaultErrorClassifier())

        # Should not raise
        executor = Executor(
            llm_client=MockLLMClient(),
            model_chain=chain,
            processor=MockProcessor(),
            hooks=hooks,
        )

        assert executor is not None

    def test_can_create_pipeline(self):
        """Verify pipeline can be created with all components."""
        pytest.skip("Integration test - requires full setup")

    def test_validation_flow(self):
        """Verify validation flow works correctly."""
        pytest.skip("Integration test - requires full setup")


# ========== PHASE 2: STRUCTURAL CONSTRAINTS ==========

import re


class TestSingleSourceOfTruth:
    """
    防止 AI 复制类型定义 - Phase 2 结构性约束.

    核心原则：WorkUnit/ErrorType/SplitType 只能在 core/types.py 定义。
    """

    def test_workunit_only_defined_once(self):
        """WorkUnit 类只能在 core/work_unit.py 定义."""
        core_dir = Path("pdf2epub/core")

        workunit_definitions = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text()
            # 匹配 class WorkUnit 定义（不是 import）
            if re.search(r'^class WorkUnit[:\(]', content, re.MULTILINE):
                workunit_definitions.append(py_file)

        assert workunit_definitions == [core_dir / "work_unit.py"], \
            f"WorkUnit defined in multiple places: {workunit_definitions}"

    def test_errortype_only_defined_once(self):
        """ErrorType 类只能在 core/types.py 定义."""
        core_dir = Path("pdf2epub/core")

        errortype_definitions = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text()
            if re.search(r'^class ErrorType\(Enum\)', content, re.MULTILINE):
                errortype_definitions.append(py_file)

        assert errortype_definitions == [core_dir / "types.py"], \
            f"ErrorType defined in multiple places: {errortype_definitions}"

    def test_splittype_only_defined_once(self):
        """SplitType 类只能在 core/work_unit.py 定义 (与 WorkUnit 在一起避免循环导入)."""
        core_dir = Path("pdf2epub/core")

        splittype_definitions = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text()
            if re.search(r'^class SplitType\(Enum\)', content, re.MULTILINE):
                splittype_definitions.append(py_file)

        # SplitType is defined in work_unit.py (with WorkUnit) to avoid circular imports
        # types.py re-exports it
        assert splittype_definitions == [core_dir / "work_unit.py"], \
            f"SplitType defined in multiple places: {splittype_definitions}"


    def test_workunit_has_split_type_validation(self):
        """WorkUnit must validate split_type naming convention."""
        from pdf2epub.core.types import WorkUnit, SplitType

        # PROACTIVE split must use .part naming
        with pytest.raises(ValueError) as exc_info:
            WorkUnit(
                id="test.sub1",  # Wrong: using .sub for PROACTIVE
                file_key="test",
                content="test",
                split_type=SplitType.PROACTIVE
            )
        assert ".part" in str(exc_info.value)

        # DYNAMIC split must use .sub naming
        with pytest.raises(ValueError) as exc_info:
            WorkUnit(
                id="test.part1",  # Wrong: using .part for DYNAMIC
                file_key="test",
                content="test",
                split_type=SplitType.DYNAMIC
            )
        assert ".sub" in str(exc_info.value)

        # Valid PROACTIVE split
        unit = WorkUnit(
            id="test.part1",
            file_key="test",
            content="test",
            split_type=SplitType.PROACTIVE
        )
        assert unit.split_type == SplitType.PROACTIVE

        # Valid DYNAMIC split
        unit = WorkUnit(
            id="test.sub0",
            file_key="test",
            content="test",
            split_type=SplitType.DYNAMIC
        )
        assert unit.split_type == SplitType.DYNAMIC


class TestLegacyIsolation:
    """
    隔离 legacy 代码，禁止新代码使用 - Phase 2 结构性约束.

    扫描整个 codebase，确保 WorkUnit/SplitType/ErrorType 都从正确位置导入。
    """

    # Files allowed to import directly from work_unit.py (re-exporters)
    REEXPORTER_FILES = {'types.py', '__init__.py', '_protocol.py'}

    # Directories with their own work_unit.py (not the core one)
    EXTERNAL_WORKUNIT_DIRS = {'processors'}

    def test_workunit_imported_from_correct_location(self):
        """
        WorkUnit 必须从 core.types 或 core 导入，不能直接从 work_unit.py.

        例外:
        - types.py 负责重新导出，可以从 work_unit 导入
        - __init__.py 可以重新导出
        - _protocol.py 可以导入用于类型定义
        - processors/ 有自己的 work_unit.py (不同模块)
        """
        core_dir = Path("pdf2epub/core")
        violations = []

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # Skip re-exporter files
            if py_file.name in self.REEXPORTER_FILES:
                continue
            # work_unit.py defines WorkUnit, skip it
            if py_file.name == "work_unit.py":
                continue

            content = py_file.read_text()

            # Check for direct imports from work_unit
            for line_num, line in enumerate(content.split('\n'), 1):
                if 'WorkUnit' in line and 'import' in line:
                    # Invalid: importing directly from work_unit module
                    if '.work_unit import' in line or 'from .work_unit' in line:
                        violations.append(f"{py_file}:{line_num} - Should import WorkUnit from types: {line.strip()}")

        assert not violations, f"WorkUnit imported from work_unit.py instead of types:\n" + "\n".join(violations)

    def test_splittype_imported_from_correct_location(self):
        """SplitType 必须从 core.types 或 core 导入."""
        core_dir = Path("pdf2epub/core")
        violations = []

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # Skip re-exporter files
            if py_file.name in self.REEXPORTER_FILES:
                continue
            if py_file.name == "work_unit.py":
                continue

            content = py_file.read_text()

            for line_num, line in enumerate(content.split('\n'), 1):
                if 'SplitType' in line and 'import' in line:
                    # Should import from types or core, not work_unit
                    if '.work_unit import' in line or 'from .work_unit' in line:
                        violations.append(f"{py_file}:{line_num} - SplitType should be imported from types: {line.strip()}")

        assert not violations, f"SplitType imported from wrong location:\n" + "\n".join(violations)


class TestBoundaryEnforcement:
    """
    职责边界不可突破 - Phase 2 结构性约束.

    扫描整个 codebase 检查职责边界，不依赖硬编码文件路径。
    """

    def test_no_llm_retry_loops_outside_executor(self):
        """LLM 重试循环只能在 executor/ 目录内."""
        core_dir = Path("pdf2epub/core")
        violations = []

        # Patterns that indicate LLM retry loops specifically
        # Not general while True loops (which are used for other things)
        llm_retry_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*max_retries', "for in range(max_retries)"),
            (r'for\s+attempt\s+in\s+range', "for attempt in range()"),
            (r'while.*retries_left', "while retries_left"),
            (r'retry_count\s*<\s*max', "retry_count < max"),
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # executor/ is allowed to have retry loops
            if "executor" in str(py_file):
                continue
            # llm/ client may have retry logic
            if "llm" in str(py_file):
                continue

            content = py_file.read_text()

            for pattern, desc in llm_retry_patterns:
                if re.search(pattern, content):
                    line_num = content[:re.search(pattern, content).start()].count('\n') + 1
                    violations.append(f"{py_file}:{line_num} - LLM retry pattern: {desc}")

        assert not violations, f"LLM retry loops outside executor:\n" + "\n".join(violations)

    def test_no_longest_fallback_outside_executor(self):
        """longest fallback 逻辑只能在 executor/ 目录内."""
        core_dir = Path("pdf2epub/core")
        violations = []

        forbidden_patterns = [
            r'_apply_longest_fallback',
            r'get_longest',
            r'max\s*\(\s*attempts',
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # executor/ is allowed
            if "executor" in str(py_file):
                continue

            content = py_file.read_text()

            for pattern in forbidden_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append(f"{py_file}: has longest fallback pattern '{pattern}'")

        assert not violations, f"Longest fallback outside executor:\n" + "\n".join(violations)

    def test_no_direct_llm_calls_in_pipeline_or_phase(self):
        """
        Pipeline/Phase 不能直接调用 LLM (必须通过 Executor).

        例外:
        - executor/ 是 LLM 调用的主要入口
        - validators/ 可能需要 LLM 进行截断检测等
        - llm/ 是 LLM 客户端本身
        """
        core_dir = Path("pdf2epub/core")
        violations = []

        # Directories specifically checked for NO direct LLM usage
        forbidden_dirs = ['pipeline', 'phase']

        # Patterns that indicate direct LLM usage
        llm_patterns = [
            (r'\.generate\s*\(', "generate() call"),
            (r'llm_client\.', "llm_client usage"),
            (r'_llm_client\.', "_llm_client usage"),
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            # Only check specific directories (pipeline, phase)
            rel_path = str(py_file.relative_to(core_dir))
            if not any(rel_path.startswith(d) or d in py_file.name for d in forbidden_dirs):
                continue

            content = py_file.read_text()

            for pattern, desc in llm_patterns:
                matches = list(re.finditer(pattern, content))
                for m in matches:
                    # Check if it's just assignment in __init__
                    line_start = content.rfind('\n', 0, m.start()) + 1
                    line = content[line_start:].split('\n')[0]
                    if 'self._llm_client = ' in line or 'self.llm_client = ' in line:
                        continue  # Assignment is ok
                    line_num = content[:m.start()].count('\n') + 1
                    violations.append(f"{py_file}:{line_num} - {desc}")

        assert not violations, f"Direct LLM calls in pipeline/phase:\n" + "\n".join(violations)

    def test_no_file_write_in_executor(self):
        """Executor 不能直接写文件 (必须通过 Persistence)."""
        core_dir = Path("pdf2epub/core")
        violations = []

        forbidden_patterns = [
            (r'\.write_text\s*\(', "write_text()"),
            (r'open\s*\([^)]*["\']w["\']', "open() with write mode"),
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # Only check executor/ for this rule
            if "executor" not in str(py_file):
                continue
            # batch_state.py is exempt - it's low-level state persistence for batch resume
            if "batch_state.py" in str(py_file):
                continue

            content = py_file.read_text()

            for pattern, desc in forbidden_patterns:
                if re.search(pattern, content):
                    violations.append(f"{py_file}: has {desc}")

        assert not violations, f"File write in executor:\n" + "\n".join(violations)

    def test_no_direct_file_write_in_phase(self):
        """Phase 不能直接写文件 (必须通过 Pipeline/Persistence)."""
        core_dir = Path("pdf2epub/core")
        violations = []

        forbidden_patterns = [
            (r'\.write_text\s*\(', "write_text()"),
            (r'open\s*\([^)]*["\']w["\']', "open() with write mode"),
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            # Only check phase/ for this rule
            if "phase" not in str(py_file):
                continue

            content = py_file.read_text()

            for pattern, desc in forbidden_patterns:
                if re.search(pattern, content):
                    line_num = content[:re.search(pattern, content).start()].count('\n') + 1
                    violations.append(f"{py_file}:{line_num} - has {desc}")

        assert not violations, f"Direct file write in phase:\n" + "\n".join(violations)


# ========== PHASE 2: MALICIOUS DEVIATIONS (should fail until fixed) ==========
# These tests scan the ENTIRE codebase, not specific files.
# They will catch violations in new modules too.


class TestWorkUnitContractCodebase:
    """
    WorkUnit 契约完整性测试 - 扫描整个 codebase.

    问题：新架构代码可能按"简化版 WorkUnit"写，但实际 WorkUnit 需要 file_key 等字段。
    """

    def test_workunit_instantiation_uses_required_fields(self):
        """所有 WorkUnit 实例化必须包含 file_key 字段."""
        core_dir = Path("pdf2epub/core")
        violations = []

        # WorkUnit required fields (from the canonical definition)
        required_fields = {'id', 'file_key', 'content'}

        for py_file in core_dir.rglob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            content = py_file.read_text()

            # Find WorkUnit instantiations: WorkUnit( with named args
            # Pattern: WorkUnit(\n followed by keyword args
            pattern = r'WorkUnit\s*\(\s*\n?\s*(\w+)\s*='
            matches = list(re.finditer(pattern, content))

            for match in matches:
                # Get the full instantiation (find matching paren)
                start = match.start()
                paren_count = 0
                end = start
                for i, char in enumerate(content[start:], start):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            end = i + 1
                            break

                instantiation = content[start:end]

                # Check for required fields
                for field in required_fields:
                    if f'{field}=' not in instantiation and f'{field} =' not in instantiation:
                        # Get line number
                        line_num = content[:start].count('\n') + 1
                        violations.append(
                            f"{py_file}:{line_num} - WorkUnit missing '{field}' field"
                        )

        assert not violations, \
            f"WorkUnit instantiations missing required fields:\n" + "\n".join(violations)


class TestPartNumberingCodebase:
    """
    .part 编号体系一致性测试 - 扫描整个 codebase.

    问题：一部分代码用 .part0 (0-based)，但既有基础设施 ChapterIdentity 是 1-based。
    所有代码必须统一使用 1-based (.part1, .part2, ...)。
    """

    def test_no_zero_based_part_numbering_in_codebase(self):
        """整个 codebase 不允许出现 0-based .part 编号."""
        pdf2epub_dir = Path("pdf2epub")
        violations = []

        # Patterns that indicate 0-based numbering
        zero_based_patterns = [
            (r'\.part0\b', ".part0 literal"),
            (r'\.part0["\']', ".part0 in string"),
            (r'part_index\s*=\s*0\b', "part_index = 0"),
            (r'part\s*=\s*0\b', "part = 0"),
            (r'f["\'].*\.part\{.*\}.*["\']', None),  # f-string, check separately
        ]

        for py_file in pdf2epub_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue

                for pattern, description in zero_based_patterns:
                    if description is None:
                        # Special handling for f-strings with .part{index}
                        # Check if it could produce .part0
                        if re.search(r'\.part\{[^}]*\b0\b', line):
                            violations.append(f"{py_file}:{i} - f-string may produce .part0")
                    elif re.search(pattern, line):
                        violations.append(f"{py_file}:{i} - {description}")

        assert not violations, \
            f"0-based part numbering found (should be 1-based):\n" + "\n".join(violations)

    def test_chapter_identity_contract(self):
        """验证 ChapterIdentity 是 1-based 的权威实现."""
        from pdf2epub.chapter_identity import ChapterIdentity

        # ChapterIdentity is the canonical source - verify it's 1-based
        identity = ChapterIdentity.parse("chapter_1.part1")
        assert identity is not None
        assert identity.part == 1, "ChapterIdentity.part should be 1 for .part1"

        # make_part_name should produce 1-based names
        assert ChapterIdentity.make_part_name("ch", 1) == "ch.part1"
        assert ChapterIdentity.make_part_name("ch", 2) == "ch.part2"


class TestNoAggregationBetweenPhases:
    """
    阶段间不聚合测试 - 扫描整个 codebase.

    问题：Pipeline/Phase 不应该在 process_all 中聚合 parts。
    聚合只应该在 build-epub 时发生。
    """

    def test_no_aggregation_in_processing_methods(self):
        """process_all 类型的处理方法不应该调用聚合."""
        core_dir = Path("pdf2epub/core")
        violations = []

        # Methods that should NOT contain aggregation
        processing_method_names = ['process_all', 'process_batch', 'run']

        # Aggregation patterns
        aggregation_patterns = [
            'aggregate_parts',
            '_aggregate_all_parts',
            'aggregate_results',
        ]

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()

            for method_name in processing_method_names:
                # Find method definition
                method_pattern = rf'def {method_name}\s*\('
                match = re.search(method_pattern, content)
                if not match:
                    continue

                # Find method body (until next def at same indentation or end)
                method_start = match.start()
                # Get indentation
                line_start = content.rfind('\n', 0, method_start) + 1
                indent = len(content[line_start:method_start]) - len(content[line_start:method_start].lstrip())

                # Find next method at same indentation
                next_method = re.search(rf'\n {{{indent}}}def \w+\s*\(', content[match.end():])
                if next_method:
                    method_end = match.end() + next_method.start()
                else:
                    method_end = len(content)

                method_body = content[method_start:method_end]

                # Check for aggregation calls
                for agg_pattern in aggregation_patterns:
                    if agg_pattern in method_body:
                        line_num = content[:method_start].count('\n') + 1
                        violations.append(
                            f"{py_file}:{line_num} - {method_name}() calls {agg_pattern}. "
                            f"Aggregation breaks phase composability."
                        )

        assert not violations, \
            f"Aggregation in processing methods (should only be in build-epub):\n" + "\n".join(violations)


class TestContextReadyEnforcement:
    """
    context_ready 信号强制检查 - 扫描整个 codebase.

    问题：缓存结果用于 context injection 前必须检查 context_ready。
    """

    def test_cache_completed_checks_context_ready(self):
        """cache_completed 调用前必须检查 context_ready."""
        core_dir = Path("pdf2epub/core")
        violations = []

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines):
                # Find cache_completed calls (not definitions)
                if 'cache_completed' in line and '(' in line:
                    # Skip function definitions
                    if 'def cache_completed' in line:
                        continue

                    # Check surrounding context for context_ready check
                    context_start = max(0, i - 15)
                    context_end = min(len(lines), i + 3)
                    context = '\n'.join(lines[context_start:context_end])

                    if 'context_ready' not in context:
                        violations.append(
                            f"{py_file}:{i+1} - cache_completed() called without checking context_ready"
                        )

        assert not violations, \
            f"cache_completed called without context_ready check:\n" + "\n".join(violations)


class TestFallbackCompletion:
    """
    Fallback 完成标记测试 - 扫描整个 codebase.

    问题：使用 fallback 结果时必须同时标记 unit 为 completed。
    """

    def test_fallback_results_mark_completed(self):
        """写入 fallback 结果到 results 时必须同时标记 completed."""
        core_dir = Path("pdf2epub/core")
        violations = []

        for py_file in core_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            content = py_file.read_text()
            lines = content.split('\n')

            for i, line in enumerate(lines):
                # Find fallback result assignments
                # Pattern: results[xxx] = something with fallback/longest
                if 'results[' in line and ('fallback' in line.lower() or 'longest' in line.lower()):
                    # Check surrounding context for completed.add
                    context_start = max(0, i - 5)
                    context_end = min(len(lines), i + 10)
                    context = '\n'.join(lines[context_start:context_end])

                    has_completion = any(x in context for x in [
                        'completed.add',
                        'completed_ids.add',
                        'successful.add',
                    ])

                    if not has_completion:
                        violations.append(
                            f"{py_file}:{i+1} - Fallback written to results without marking completed"
                        )

        assert not violations, \
            f"Fallback results not marked as completed:\n" + "\n".join(violations)


class TestProcessContextConstruction:
    """
    ProcessContext 构建测试.

    确保 Executor 正确从 WorkUnit 构建 ProcessContext，
    使 skip validators 和 processor prompts 可靠。
    """

    def test_executor_builds_context_from_workunit(self):
        """Executor 必须使用 ProcessContext.from_work_unit 或等效方式构建 context."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Should NOT have minimal context construction anymore
        # Old pattern: ProcessContext(file_key=unit.id, book_title="")
        minimal_pattern = r'ProcessContext\(\s*file_key=unit\.id,\s*book_title=""'

        if re.search(minimal_pattern, content):
            pytest.fail("Should use ProcessContext.from_work_unit, not minimal construction")

    def test_processcontext_from_work_unit_exists(self):
        """ProcessContext 必须有 from_work_unit 类方法."""
        from pdf2epub.core._protocol import ProcessContext
        assert hasattr(ProcessContext, 'from_work_unit'), \
            "ProcessContext should have from_work_unit classmethod"

    def test_processcontext_from_work_unit_propagates_fields(self):
        """from_work_unit 必须正确传递 WorkUnit 字段."""
        from pdf2epub.core._protocol import ProcessContext
        from pdf2epub.core.work_unit import WorkUnit

        unit = WorkUnit(
            id="chapter_1.part2",
            file_key="chapter_1",
            content="test content",
            part_index=2,
            total_parts=3,
            chapter_type="notes",
            chapter_title="Chapter Notes",
        )

        ctx = ProcessContext.from_work_unit(unit, book_title="Test Book")

        assert ctx.file_key == "chapter_1"
        assert ctx.book_title == "Test Book"
        assert ctx.part_index == 2
        assert ctx.total_parts == 3
        assert ctx.chapter_type == "notes"
        assert ctx.chapter_title == "Chapter Notes"
        assert ctx.is_notes_chapter == True
        assert ctx.is_front_back_matter == True  # notes is front/back matter


class TestNestedSplitContextInjection:
    """
    嵌套分割的上下文注入测试.

    确保 ContextInjector 正确处理嵌套 .part 的情况，
    如 chapter_1.part1.part2 应该注入 chapter_1.part1.part1 的上下文。
    """

    def test_context_injector_derives_prev_id_from_unit_id(self):
        """ContextInjector 必须从 unit.id 推导 prev_id，而不是 file_key + part_index."""
        context_path = Path("pdf2epub/core/context.py")
        content = context_path.read_text()

        # Should NOT use file_key + part_index pattern
        bad_pattern = r'prev_id\s*=\s*f["\'].*file_key.*part.*part_index'
        if re.search(bad_pattern, content):
            pytest.fail("Should derive prev_id from unit.id, not file_key + part_index")

        # Should have helper function that uses unit_id.rsplit
        # (the helper is _get_prev_part_id which uses unit_id.rsplit('.part', 1))
        good_pattern = r"unit_id\.rsplit\(['\"]\.part['\"]"
        assert re.search(good_pattern, content), \
            "Should have _get_prev_part_id helper using unit_id.rsplit('.part', 1)"

        # Should use the helper function for prev_id
        helper_pattern = r"_get_prev_part_id\("
        assert re.search(helper_pattern, content), \
            "Should use _get_prev_part_id helper function"

    def test_nested_split_prev_id_calculation(self):
        """验证嵌套分割的 prev_id 计算正确."""
        from pdf2epub.core.work_unit import WorkUnit
        from pdf2epub.core.context import ContextInjector

        # Create a mock nested split unit
        unit = WorkUnit(
            id="chapter_1.part1.part2",
            file_key="chapter_1",  # Ultimate base
            content="test",
            part_index=2,
        )

        # The prev_id should be chapter_1.part1.part1, not chapter_1.part1
        injector = ContextInjector(mode="sequential")

        # Simulate completed results
        completed_results = {
            "chapter_1.part1.part1": "previous content",
        }
        originals = {
            "chapter_1.part1.part1": "original previous",
        }

        result = injector.get_context_for_unit(unit, completed_results, originals)

        # Should find the correct previous part
        assert result is not None, "Should find previous part chapter_1.part1.part1"
        assert result[0] == "original previous"
        assert result[1] == "previous content"


class TestQuotaEffectConsistency:
    """
    Quota 和 Effect 一致性测试.

    确保每个 ErrorType 的 quota 和 effect 配置一致。
    """

    def test_error_effects_use_own_quota_type(self):
        """ErrorEffect 的 quota_type 应该与 ErrorType 对应，不能借用其他类型."""
        from pdf2epub.core.hooks.error_classifier import DefaultErrorClassifier
        from pdf2epub.core.hooks import ErrorType

        classifier = DefaultErrorClassifier()

        # These error types should use their own quota
        self_quota_types = [
            ErrorType.TRUNCATION,  # Has its own quota in QuotaConfig
            ErrorType.VALIDATION,
            ErrorType.SAFETY,
        ]

        for error_type in self_quota_types:
            effect = classifier.get_effect(error_type)
            assert effect.quota_type == error_type, \
                f"{error_type.name} should use its own quota, not {effect.quota_type.name}"

    def test_quota_config_covers_all_error_types(self):
        """QuotaConfig 应该为所有 ErrorType 定义 quota."""
        from pdf2epub.core.executor.state import QuotaConfig
        from pdf2epub.core.hooks import ErrorType

        config = QuotaConfig()
        quotas = config.create_quotas()

        for error_type in ErrorType:
            assert error_type in quotas, \
                f"QuotaConfig missing quota for {error_type.name}"


class TestBatchInterfaceCompatibility:
    """
    Batch 接口兼容性测试.

    确保 Executor 的 batch 路径使用正确的 GeminiBatchClient 接口。
    """

    def test_executor_uses_correct_batch_methods(self):
        """Executor 必须使用 GeminiBatchClient 的实际方法名."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Correct methods (GeminiBatchClient interface)
        correct_patterns = [
            r'batch_client\.submit\(',        # submit(requests: List[BatchRequest])
            r'batch_client\.get_status\(',    # get_status(job_name) -> BatchJobInfo
            r'batch_client\.get_results\(',   # get_results(job_name) -> List[BatchResponse]
        ]

        # Wrong methods (non-existent)
        wrong_patterns = [
            (r'batch_client\.submit_batch\(', "submit_batch doesn't exist, use submit()"),
        ]

        # Check for wrong patterns
        violations = []
        for pattern, msg in wrong_patterns:
            if re.search(pattern, content):
                violations.append(msg)

        assert not violations, f"Executor uses wrong batch methods:\n" + "\n".join(violations)

    def test_executor_handles_batch_response_objects(self):
        """Executor 必须正确处理 BatchResponse 对象，不是 dict."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Should import BatchRequest, BatchJobState from batch_utils
        if '_process_batch' in content:
            # Check for correct imports in _process_batch
            if 'from ...utils.batch_utils import' not in content:
                pytest.fail("_process_batch should import from utils.batch_utils")

            # Should not treat results as dict[unit_id] directly
            # (old code did: raw_results[unit_id])
            if re.search(r'raw_results\[unit_id\]', content):
                pytest.fail("Should not access raw_results as dict - use BatchResponse.key")

    def test_executor_uses_workunit_chapter_type(self):
        """Executor 必须使用 WorkUnit.chapter_type，不是 metadata.get()."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Wrong: unit.metadata.get('chapter_type')
        if re.search(r'unit\.metadata\.get\([\'"]chapter_type[\'"]\)', content):
            pytest.fail("Should use unit.chapter_type, not unit.metadata.get('chapter_type')")

        # Wrong: unit.metadata (WorkUnit doesn't have metadata field)
        if re.search(r'\bunit\.metadata\b', content):
            pytest.fail("WorkUnit doesn't have metadata field")


class TestScreenerToBatchSkip:
    """
    验证 screener 放行的结果不会被 batch validator 重复验证（省钱）。

    流程：
    1. Executor 跟踪 context_ready=True 的结果 → screener_passed
    2. ExecutionResult 包含 screener_passed 字段
    3. Pipeline 把 screener_passed 传给 _run_batch_validation
    4. _run_batch_validation 跳过这些 keys（不创建 VerificationFile）
    """

    def test_execution_result_has_screener_passed_field(self):
        """ExecutionResult 必须有 screener_passed 字段."""
        from pdf2epub.core.executor._protocol import ExecutionResult
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ExecutionResult)}
        assert 'screener_passed' in fields, (
            "ExecutionResult must have screener_passed field to track screener-validated keys"
        )

    def test_executor_tracks_screener_passed(self):
        """Executor 必须在 result.context_ready=True 时记录到 screener_passed."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Must track screener_passed when context_ready
        assert 'screener_passed' in content, "Executor must track screener_passed"
        assert re.search(r'if.*context_ready.*screener_passed\.add', content, re.DOTALL), (
            "Executor must add to screener_passed when result.context_ready=True"
        )

    def test_execute_returns_screener_passed_from_online_processing(self):
        """execute() 返回的 screener_passed 必须包含 context_ready=True 的结果.

        这是真正的数据流测试：实际运行 executor，验证返回值。
        """
        from unittest.mock import MagicMock, patch
        from pdf2epub.core.executor import Executor as OnlineExecutor
        from pdf2epub.core.executor._protocol import ChainEntry, ProcessResult
        from pdf2epub.core.work_unit import WorkUnit
        from pdf2epub.core.hooks import CompositeHooks

        # Create minimal executor
        mock_llm = MagicMock()
        mock_processor = MagicMock()
        mock_processor.name = "test"

        # Processor returns success with context_ready=True
        mock_processor.process.return_value = "processed content"

        hooks = CompositeHooks()
        # Make validator return context_ready=True
        mock_validator = MagicMock()
        mock_validator.name = "test_validator"
        mock_validator.validate.return_value = MagicMock(accepted=True, context_ready=True)
        hooks._validators = [mock_validator]

        chain = [ChainEntry(provider="test", model="test-model", mode="online")]

        executor = OnlineExecutor(
            llm_client=mock_llm,
            model_chain=chain,
            processor=mock_processor,
            hooks=hooks,
            max_workers=1,
        )

        # Mock LLM to return success
        mock_llm.generate.return_value = "llm output"

        # Create test unit
        unit = WorkUnit(id="test_unit", file_key="test", content="test content")

        # Execute
        result = executor.execute([unit])

        # THE REAL TEST: screener_passed must contain the unit
        assert "test_unit" in result.screener_passed, (
            f"execute() must return screener_passed with context_ready units. "
            f"Got: {result.screener_passed}"
        )

    def test_pipeline_passes_screener_passed_to_batch_validation(self):
        """Pipeline 必须把 screener_passed 传给 batch validation."""
        pipeline_path = Path("pdf2epub/core/pipeline_v2.py")
        content = pipeline_path.read_text()

        # Must pass screener_passed to _run_batch_validation
        assert re.search(r'_run_batch_validation\([^)]*screener_passed', content), (
            "Pipeline must pass screener_passed to _run_batch_validation"
        )

    def test_batch_validation_skips_screener_passed_before_creating_files(self):
        """Batch validation 必须在创建 VerificationFile 之前跳过 screener_passed keys."""
        pipeline_path = Path("pdf2epub/core/pipeline_v2.py")
        content = pipeline_path.read_text()

        # Find _run_batch_validation method
        method_match = re.search(
            r'def _run_batch_validation\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            content, re.DOTALL
        )
        if not method_match:
            pytest.fail("Cannot find _run_batch_validation method")

        method_body = method_match.group(1)

        # Must include screener_passed in skip_keys
        assert 'screener_passed' in method_body, (
            "_run_batch_validation must use screener_passed"
        )

        # Must skip BEFORE creating VerificationFile (check order)
        skip_pos = method_body.find('skip_keys')
        file_create_pos = method_body.find('VerificationFile(')

        assert skip_pos < file_create_pos, (
            "Must check skip_keys BEFORE creating VerificationFile to save cost"
        )


class TestNetworkErrorPolicy:
    """
    验证网络错误处理策略：

    1. 底层 retry 窗口短（30s 而不是 300s）
    2. NETWORK/TIMEOUT/RATE_LIMIT 的 effect 是 remove_current_model=True（推进 chain）
    3. 网络错误不触发 split
    4. 有全局熔断机制
    """

    def test_llm_client_attempt_based_retry(self):
        """底层 retry 使用次数限制，不是时间限制."""
        llm_client_path = Path("pdf2epub/utils/llm_client.py")
        content = llm_client_path.read_text()

        # Should use stop_after_attempt, not stop_after_delay
        assert "stop_after_attempt" in content, (
            "LLMClient should use attempt-based retry (stop_after_attempt)"
        )

        # Should NOT use time-based retry
        assert "stop_after_delay" not in content, (
            "LLMClient should NOT use time-based retry (stop_after_delay)"
        )

        # Should use max_retries config, not max_retry_duration_seconds
        assert "max_retries" in content, (
            "LLMClient should use max_retries config"
        )

    def test_network_errors_remove_current_model(self):
        """NETWORK/TIMEOUT/RATE_LIMIT 的 effect 必须 remove_current_model=True."""
        from pdf2epub.core.hooks.error_classifier import DefaultErrorClassifier
        from pdf2epub.core.hooks._protocol import ErrorType

        classifier = DefaultErrorClassifier()
        network_errors = [ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT]

        for error_type in network_errors:
            effect = classifier.get_effect(error_type)
            assert effect.remove_current_model is True, (
                f"{error_type.value} must have remove_current_model=True "
                f"to push chain forward (fail fast)"
            )

    def test_network_errors_never_split(self):
        """网络错误永远不触发 split."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Find _handle_failure method
        method_match = re.search(
            r'def _handle_failure\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            content, re.DOTALL
        )
        if not method_match:
            pytest.fail("Cannot find _handle_failure method")

        method_body = method_match.group(1)

        # Must check for network errors before split
        assert 'network_errors' in method_body or 'NETWORK' in method_body, (
            "_handle_failure must handle network errors specially"
        )

        # Must skip split for network errors
        assert re.search(r'if error_type in network_errors.*skipping split', method_body, re.DOTALL), (
            "Network errors must skip split (splitting won't fix network issues)"
        )

    def test_circuit_breaker_exists(self):
        """必须有全局网络熔断机制."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Must have circuit breaker threshold parameter
        assert 'network_circuit_breaker_threshold' in content, (
            "Executor must have network_circuit_breaker_threshold parameter"
        )

        # Must track consecutive failures
        assert 'consecutive_network_failures' in content, (
            "Executor must track consecutive network failures"
        )

        # Must have circuit broken flag
        assert 'circuit_broken' in content or 'circuit_breaker' in content, (
            "Executor must have circuit breaker mechanism"
        )


class TestBatchQuotaPolicy:
    """
    验证 Batch/Online 统一 Quota 政策（Section 20）。

    核心功能：
    1. 字符串错误分类 (classify_from_string)
    2. Poison unit 归因 (attribute_job_failure)
    3. 三态熔断器 (BatchCircuitBreaker)
    4. Batch 失败动作 (get_batch_failure_action)
    """

    def test_classify_from_string_exists(self):
        """DefaultErrorClassifier 必须有 classify_from_string 方法."""
        from pdf2epub.core.hooks.error_classifier import DefaultErrorClassifier

        classifier = DefaultErrorClassifier()
        assert hasattr(classifier, 'classify_from_string'), (
            "DefaultErrorClassifier must have classify_from_string method for batch errors"
        )

    def test_classify_from_string_classifies_correctly(self):
        """classify_from_string 能正确分类各种错误类型."""
        from pdf2epub.core.hooks.error_classifier import DefaultErrorClassifier
        from pdf2epub.core.types import ErrorType

        classifier = DefaultErrorClassifier()

        # Test various error messages
        # Note: Order matters - SAFETY keywords are checked before CONTENT_FILTER
        # "blocked" is a SAFETY keyword, so messages with "blocked" -> SAFETY
        test_cases = [
            ("Request too large for model", ErrorType.VALIDATION),
            ("Token limit exceeded", ErrorType.VALIDATION),
            ("Safety policy violation", ErrorType.SAFETY),
            ("content_filter triggered", ErrorType.CONTENT_FILTER),  # content_filter is specific
            ("finish_reason: recitation", ErrorType.CONTENT_FILTER),  # recitation is specific
            ("Connection timeout", ErrorType.TIMEOUT),
            ("Rate limit reached", ErrorType.RATE_LIMIT),
            ("Network error occurred", ErrorType.NETWORK),
            ("Response truncated", ErrorType.TRUNCATION),
        ]

        for msg, expected in test_cases:
            result = classifier.classify_from_string(msg)
            assert result == expected, (
                f"classify_from_string('{msg}') should return {expected.value}, "
                f"got {result.value}"
            )

    def test_attribution_dataclass_exists(self):
        """Attribution dataclass 必须存在."""
        from pdf2epub.core.hooks.error_classifier import Attribution

        # Check it's a dataclass with required fields
        import dataclasses
        assert dataclasses.is_dataclass(Attribution), "Attribution must be a dataclass"

        fields = {f.name for f in dataclasses.fields(Attribution)}
        assert 'type' in fields, "Attribution must have 'type' field"
        assert 'unit_id' in fields, "Attribution must have 'unit_id' field"
        assert 'error_type' in fields, "Attribution must have 'error_type' field"

    def test_attribute_job_failure_exists(self):
        """attribute_job_failure 函数必须存在."""
        from pdf2epub.core.hooks.error_classifier import attribute_job_failure

        assert callable(attribute_job_failure), "attribute_job_failure must be callable"

    def test_attribute_job_failure_unit_key_in_error(self):
        """attribute_job_failure 能识别错误消息中的 unit key."""
        from pdf2epub.core.hooks.error_classifier import attribute_job_failure

        class MockUnit:
            def __init__(self, id: str, content: str):
                self.id = id
                self.content = content

        units = [
            MockUnit("page_001", "short content"),
            MockUnit("page_002", "much longer content" * 100),
        ]

        # Error that mentions specific unit
        attr = attribute_job_failure("Error processing page_001: invalid format", units)
        assert attr.type == "unit"
        assert attr.unit_id == "page_001"

    def test_attribute_job_failure_size_issue_blames_largest(self):
        """attribute_job_failure 对 size 问题归咎于最大的 unit."""
        from pdf2epub.core.hooks.error_classifier import attribute_job_failure
        from pdf2epub.core.types import ErrorType

        class MockUnit:
            def __init__(self, id: str, content: str):
                self.id = id
                self.content = content

        units = [
            MockUnit("small", "x" * 100),
            MockUnit("large", "x" * 10000),
            MockUnit("medium", "x" * 1000),
        ]

        # Error about size
        attr = attribute_job_failure("Request payload too large", units)
        assert attr.type == "unit"
        assert attr.unit_id == "large"
        assert attr.error_type == ErrorType.VALIDATION

    def test_attribute_job_failure_systemic_goes_to_job(self):
        """attribute_job_failure 对系统性问题归咎于 job quota."""
        from pdf2epub.core.hooks.error_classifier import attribute_job_failure

        class MockUnit:
            def __init__(self, id: str, content: str):
                self.id = id
                self.content = content

        units = [MockUnit("page_001", "content")]

        # Systemic error (no unit key, not size issue)
        attr = attribute_job_failure("Service temporarily unavailable", units)
        assert attr.type == "job"

    def test_batch_circuit_breaker_exists(self):
        """BatchCircuitBreaker 必须存在且有三态."""
        from pdf2epub.core.hooks.error_classifier import BatchCircuitBreaker

        breaker = BatchCircuitBreaker()

        # Check initial state
        assert breaker.state == "closed"
        assert breaker.failure_count == 0

        # Check methods exist
        assert hasattr(breaker, 'record_failure')
        assert hasattr(breaker, 'should_try_batch')
        assert hasattr(breaker, 'record_success')
        assert hasattr(breaker, 'reset')

    def test_batch_circuit_breaker_transitions(self):
        """BatchCircuitBreaker 状态转换正确."""
        from pdf2epub.core.hooks.error_classifier import BatchCircuitBreaker

        # Use a long cooldown to test open state blocking
        breaker = BatchCircuitBreaker(threshold=2, cooldown_seconds=3600)

        # Initial: closed
        assert breaker.state == "closed"
        assert breaker.should_try_batch() is True

        # First failure: still closed
        breaker.record_failure()
        assert breaker.state == "closed"
        assert breaker.should_try_batch() is True

        # Second failure: opens
        breaker.record_failure()
        assert breaker.state == "open"
        # With long cooldown, should_try_batch returns False
        assert breaker.should_try_batch() is False

        # Now test quick recovery with short cooldown
        breaker2 = BatchCircuitBreaker(threshold=1, cooldown_seconds=0)
        breaker2.record_failure()
        assert breaker2.state == "open"

        # With 0 cooldown, immediately transitions to half_open on should_try_batch
        import time
        time.sleep(0.01)  # Small delay to ensure time passes
        assert breaker2.should_try_batch() is True
        assert breaker2.state == "half_open"

        # Success in half_open: closes
        breaker2.record_success()
        assert breaker2.state == "closed"
        assert breaker2.failure_count == 0

    def test_batch_circuit_breaker_half_open_failure(self):
        """BatchCircuitBreaker half_open 失败后回到 open."""
        from pdf2epub.core.hooks.error_classifier import BatchCircuitBreaker
        import time

        breaker = BatchCircuitBreaker(threshold=1, cooldown_seconds=0)

        # Trigger open
        breaker.record_failure()
        assert breaker.state == "open"

        # Wait and enter half_open
        time.sleep(0.01)
        breaker.should_try_batch()
        assert breaker.state == "half_open"

        # Failure in half_open: back to open
        breaker.record_failure()
        assert breaker.state == "open"

    def test_get_batch_failure_action_exists(self):
        """get_batch_failure_action 函数必须存在."""
        from pdf2epub.core.hooks.error_classifier import get_batch_failure_action

        assert callable(get_batch_failure_action)

    def test_get_batch_failure_action_safety_removes_provider(self):
        """SAFETY/CONTENT_FILTER 移除整个 provider."""
        from pdf2epub.core.hooks.error_classifier import get_batch_failure_action, BatchFailureAction
        from pdf2epub.core.types import ErrorType

        assert get_batch_failure_action(ErrorType.SAFETY) == BatchFailureAction.REMOVE_PROVIDER
        assert get_batch_failure_action(ErrorType.CONTENT_FILTER) == BatchFailureAction.REMOVE_PROVIDER

    def test_get_batch_failure_action_validation_retries_online(self):
        """VALIDATION 给 online 一次机会."""
        from pdf2epub.core.hooks.error_classifier import get_batch_failure_action, BatchFailureAction
        from pdf2epub.core.types import ErrorType

        assert get_batch_failure_action(ErrorType.VALIDATION) == BatchFailureAction.RETRY_ONLINE_SAME_MODEL

    def test_get_batch_failure_action_truncation_removes_model(self):
        """TRUNCATION 移除当前 model."""
        from pdf2epub.core.hooks.error_classifier import get_batch_failure_action, BatchFailureAction
        from pdf2epub.core.types import ErrorType

        assert get_batch_failure_action(ErrorType.TRUNCATION) == BatchFailureAction.REMOVE_MODEL

    def test_get_batch_failure_action_network_removes_model(self):
        """NETWORK/TIMEOUT/RATE_LIMIT 移除当前 model (fail fast)."""
        from pdf2epub.core.hooks.error_classifier import get_batch_failure_action, BatchFailureAction
        from pdf2epub.core.types import ErrorType

        assert get_batch_failure_action(ErrorType.NETWORK) == BatchFailureAction.REMOVE_MODEL
        assert get_batch_failure_action(ErrorType.TIMEOUT) == BatchFailureAction.REMOVE_MODEL
        assert get_batch_failure_action(ErrorType.RATE_LIMIT) == BatchFailureAction.REMOVE_MODEL

    def test_batch_quota_exports_from_hooks_module(self):
        """所有 batch quota 相关类和函数必须从 hooks 模块导出."""
        from pdf2epub.core.hooks import (
            Attribution,
            BatchFailureAction,
            BatchCircuitBreaker,
            attribute_job_failure,
            extract_unit_key_from_error,
            get_batch_failure_action,
        )

        # Just verify they're importable
        assert Attribution is not None
        assert BatchFailureAction is not None
        assert BatchCircuitBreaker is not None
        assert callable(attribute_job_failure)
        assert callable(extract_unit_key_from_error)
        assert callable(get_batch_failure_action)


class TestFallbackObservability:
    """
    验证 fallback 可观测性（P1 问题修复）。

    问题：longest fallback 直接当正常完成返回，无法审计/回溯质量问题。

    修复：
    1. ExecutionResult 有 fallback_used 字段
    2. Executor 记录使用 fallback 的 key
    3. Pipeline 用 save_with_warning() 保存 fallback 结果
    4. Tracker 用 completed_fallback 状态区分
    """

    def test_execution_result_has_fallback_used_field(self):
        """ExecutionResult 必须有 fallback_used 字段."""
        from pdf2epub.core.executor._protocol import ExecutionResult
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ExecutionResult)}
        assert 'fallback_used' in fields, (
            "ExecutionResult must have fallback_used field to track longest fallback usage"
        )

    def test_executor_tracks_fallback_used(self):
        """Executor 在使用 longest fallback 时必须记录到 fallback_used."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # Must have fallback_used initialization
        assert 'fallback_used: Set[str] = set()' in content, (
            "Executor must initialize fallback_used set"
        )

        # Must add to fallback_used when using longest fallback
        assert re.search(r'fallback_used\.add\(unit_id\)', content), (
            "Executor must add unit_id to fallback_used when using longest fallback"
        )

    def test_execute_returns_fallback_used_when_longest_fallback_triggered(self):
        """execute() 返回的 fallback_used 必须包含使用了 longest fallback 的 unit.

        这是真正的数据流测试：实际运行 executor，触发 fallback，验证返回值。
        """
        from unittest.mock import MagicMock
        from pdf2epub.core.executor import Executor as OnlineExecutor
        from pdf2epub.core.executor._protocol import ChainEntry
        from pdf2epub.core.executor.state import QuotaConfig
        from pdf2epub.core.work_unit import WorkUnit
        from pdf2epub.core.hooks import CompositeHooks, DefaultErrorClassifier
        from pdf2epub.core.types import ErrorType

        # Create executor with minimal quota (forces fallback quickly)
        mock_llm = MagicMock()
        mock_processor = MagicMock()
        mock_processor.name = "test"

        hooks = CompositeHooks()
        hooks._error_classifier = DefaultErrorClassifier()

        # Validator always rejects (to trigger retry -> fallback)
        mock_validator = MagicMock()
        mock_validator.name = "test_validator"
        mock_validator.validate.return_value = MagicMock(accepted=False, context_ready=False)
        hooks._validators = [mock_validator]

        chain = [ChainEntry(provider="test", model="test-model", mode="online")]

        # Quota of 1 means first failure exhausts retries -> triggers fallback
        quota_config = QuotaConfig(total=1, per_type={ErrorType.VALIDATION: 1})

        executor = OnlineExecutor(
            llm_client=mock_llm,
            model_chain=chain,
            processor=mock_processor,
            hooks=hooks,
            quota_config=quota_config,
            max_workers=1,
        )

        # LLM returns content (so there's something for fallback)
        mock_llm.generate.return_value = "llm output for fallback"

        # Create test unit
        unit = WorkUnit(id="fallback_unit", file_key="test", content="test content")

        # Execute - should trigger validation failure -> longest fallback
        result = executor.execute([unit])

        # THE REAL TEST: fallback_used must contain the unit
        assert "fallback_unit" in result.fallback_used, (
            f"execute() must return fallback_used when longest fallback is triggered. "
            f"Got: {result.fallback_used}, completed: {result.completed}, failed: {result.failed}"
        )

    def test_executor_passes_fallback_used_to_handle_failure(self):
        """Executor 必须把 fallback_used 传给 _handle_failure."""
        executor_path = Path("pdf2epub/core/executor/executor.py")
        content = executor_path.read_text()

        # _handle_failure must have fallback_used parameter
        assert re.search(r'def _handle_failure\([^)]*fallback_used', content, re.DOTALL), (
            "_handle_failure must accept fallback_used parameter"
        )

    def test_pipeline_uses_save_with_warning_for_fallback(self):
        """Pipeline 必须对 fallback 结果使用 save_with_warning."""
        pipeline_path = Path("pdf2epub/core/pipeline_v2.py")
        content = pipeline_path.read_text()

        # Must use save_with_warning for fallback keys
        assert 'save_with_warning' in content, (
            "Pipeline must use save_with_warning for fallback results"
        )

        # Must reference fallback_used from exec_result
        assert 'exec_result.fallback_used' in content, (
            "Pipeline must access fallback_used from exec_result"
        )

    def test_pipeline_mark_complete_distinguishes_fallback(self):
        """Pipeline._mark_complete 必须区分 fallback 和正常完成."""
        pipeline_path = Path("pdf2epub/core/pipeline_v2.py")
        content = pipeline_path.read_text()

        # _mark_complete must accept fallback parameter
        assert re.search(r'def _mark_complete\([^)]*fallback', content), (
            "_mark_complete must accept fallback parameter"
        )

        # Must use completed_fallback status
        assert 'completed_fallback' in content, (
            "_mark_complete must use completed_fallback status for fallback results"
        )

    def test_persistence_save_with_warning_exists(self):
        """ResultPersistence 必须有 save_with_warning 方法."""
        from pdf2epub.core.persistence import ResultPersistence

        assert hasattr(ResultPersistence, 'save_with_warning'), (
            "ResultPersistence must have save_with_warning method"
        )
