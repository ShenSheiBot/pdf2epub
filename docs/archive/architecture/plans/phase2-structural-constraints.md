# Phase 2: 结构性约束重构 (v2)

## 核心原则
**对 AI 的约束必须是"结构性的"而非"约定性的"**：
- 用 Enum/Literal 而非 string
- 用 显式异常 而非 assert（`python -O` 会优化掉 assert）
- 用 测试扫描 而非 运行时检查
- 用 唯一定义 + 强制 import 路径 防止平行体系再生

---

## 1. 统一类型到中立位置 (`core/types.py`)

**问题**: WorkUnit/ErrorType 分裂，且放在 executor 子包会导致依赖倒置

**方案**: 创建 `core/types.py` 作为唯一真相源

```python
# core/types.py - 所有核心类型的唯一定义点
"""
Core types - SINGLE SOURCE OF TRUTH.

All other modules MUST import from here, not re-define.
"""
from enum import Enum
from typing import Literal, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# Split Type (替代 .part/.sub 命名约定)
# ============================================================

class SplitType(Enum):
    """拆分类型 - 结构性约束，不可扩展"""
    NONE = "none"           # 未拆分
    PROACTIVE = "proactive" # Pipeline 的 .part 拆分（持久化）
    DYNAMIC = "dynamic"     # Executor 的 .sub 拆分（虚拟）


# ============================================================
# Error Type (统一所有错误分类)
# ============================================================

class ErrorType(Enum):
    """错误类型 - 唯一定义"""
    # 安全类
    SAFETY = "safety"
    CONTENT_FILTER = "content_filter"

    # 网络类
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"

    # 验证类
    VALIDATION = "validation"
    TRUNCATION = "truncation"

    # 解析类
    PARSE_ERROR = "parse_error"

    # 兜底
    UNKNOWN = "unknown"

    # Legacy 兼容（来自 model_chain，标记为 deprecated）
    # 如果确认不需要，后续删除
    # WRONG_LANGUAGE = "wrong_language"
    # EMPTY_RESPONSE = "empty_response"


# ============================================================
# Work Unit (统一所有工作单元)
# ============================================================

@dataclass
class WorkUnit:
    """
    工作单元 - 唯一定义.

    所有 Pipeline/Executor/Phase/ContextInjector 必须使用此类。
    """
    id: str
    content: str
    source_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 结构性约束
    split_type: SplitType = SplitType.NONE
    parent_id: Optional[str] = None

    # 依赖（用于 context injection）
    depends_on: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # 显式异常，不用 assert（python -O 会优化掉 assert）
        self._validate_naming()

    def _validate_naming(self):
        """验证命名与 split_type 一致 - fail fast"""
        if self.split_type == SplitType.PROACTIVE:
            if ".part" not in self.id or ".sub" in self.id:
                raise ValueError(
                    f"Proactive split must use .part naming (not .sub): {self.id}"
                )
        elif self.split_type == SplitType.DYNAMIC:
            if ".sub" not in self.id:
                raise ValueError(
                    f"Dynamic split must use .sub naming: {self.id}"
                )

    @property
    def is_virtual(self) -> bool:
        """虚拟单元不落盘"""
        return self.split_type == SplitType.DYNAMIC
```

---

## 2. 禁止平行体系再生（测试强制）

**核心测试**: 全仓扫描，确保类型只定义一次

```python
# tests/test_architecture.py

class TestSingleSourceOfTruth:
    """防止 AI 复制类型定义"""

    def test_workunit_only_defined_once(self):
        """WorkUnit 类只能在 core/types.py 定义"""
        core_dir = Path("pdf2epub/core")

        workunit_definitions = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text()
            # 匹配 class WorkUnit 定义（不是 import）
            if re.search(r'^class WorkUnit[:\(]', content, re.MULTILINE):
                workunit_definitions.append(py_file)

        assert workunit_definitions == [core_dir / "types.py"], \
            f"WorkUnit defined in multiple places: {workunit_definitions}"

    def test_errortype_only_defined_once(self):
        """ErrorType 类只能在 core/types.py 定义"""
        core_dir = Path("pdf2epub/core")

        errortype_definitions = []
        for py_file in core_dir.rglob("*.py"):
            content = py_file.read_text()
            if re.search(r'^class ErrorType\(Enum\)', content, re.MULTILINE):
                errortype_definitions.append(py_file)

        assert errortype_definitions == [core_dir / "types.py"], \
            f"ErrorType defined in multiple places: {errortype_definitions}"

    def test_splittype_only_defined_once(self):
        """SplitType 类只能在 core/types.py 定义"""
        # 同上模式
```

---

## 3. 禁止 Legacy 导入（测试强制）

```python
class TestLegacyIsolation:
    """隔离 legacy 代码，禁止新代码使用"""

    LEGACY_MODULES = {
        "core.work_unit",      # 旧 WorkUnit
        "core.model_chain",    # 旧 ErrorType
    }

    NEW_CODE_DIRS = [
        "pdf2epub/core/executor",
        "pdf2epub/core/hooks",
        "pdf2epub/core/phase",
        "pdf2epub/core/pipeline_v2.py",
        "pdf2epub/core/factory_v2.py",
    ]

    def test_new_code_no_legacy_imports(self):
        """新架构代码不能 import legacy 模块"""
        violations = []

        for path in self.NEW_CODE_DIRS:
            p = Path(path)
            files = [p] if p.is_file() else p.rglob("*.py")

            for py_file in files:
                content = py_file.read_text()
                for legacy in self.LEGACY_MODULES:
                    # 检查 import 语句
                    if f"from {legacy}" in content or f"import {legacy}" in content:
                        violations.append(f"{py_file}: imports {legacy}")

        assert not violations, f"Legacy imports in new code:\n" + "\n".join(violations)
```

---

## 4. Pipeline/Executor 职责边界（测试强制）

**比 _FORBIDDEN_METHODS 更硬的约束**：扫描实际代码模式

```python
class TestBoundaryEnforcement:
    """职责边界不可突破 - 扫描代码模式"""

    def test_pipeline_no_retry_loop(self):
        """Pipeline 不能有重试循环"""
        pipeline_code = Path("pdf2epub/core/pipeline_v2.py").read_text()

        # 检测 while True + retry/attempt 组合
        has_retry_loop = bool(re.search(
            r'while\s+(True|.*retry|.*attempt).*?:', pipeline_code, re.DOTALL
        ))
        # 检测 for + retry/attempt 变量
        has_retry_for = bool(re.search(
            r'for\s+.*\bretry\b', pipeline_code
        ))

        assert not has_retry_loop, "Pipeline has retry loop (should be in Executor)"
        assert not has_retry_for, "Pipeline has retry for-loop"

    def test_pipeline_no_longest_fallback(self):
        """Pipeline 不能有 longest fallback 逻辑"""
        pipeline_code = Path("pdf2epub/core/pipeline_v2.py").read_text()

        forbidden_patterns = [
            r'longest',
            r'fallback',
            r'get_longest',
            r'max\s*\(\s*attempts',
        ]

        for pattern in forbidden_patterns:
            assert not re.search(pattern, pipeline_code, re.IGNORECASE), \
                f"Pipeline contains forbidden pattern: {pattern}"

    def test_pipeline_no_direct_llm_call(self):
        """Pipeline 不能直接调用 LLM"""
        pipeline_code = Path("pdf2epub/core/pipeline_v2.py").read_text()

        forbidden = [
            r'\.generate\s*\(',
            r'llm_client\.',
            r'_llm_client\.',
        ]

        for pattern in forbidden:
            # 排除在 __init__ 中存储 llm_client
            matches = list(re.finditer(pattern, pipeline_code))
            for m in matches:
                # 检查是否在 __init__ 赋值中
                line_start = pipeline_code.rfind('\n', 0, m.start()) + 1
                line = pipeline_code[line_start:m.end() + 50].split('\n')[0]
                if 'self._llm_client = llm_client' not in line:
                    raise AssertionError(f"Pipeline directly uses LLM: {line.strip()}")

    def test_executor_no_file_write(self):
        """Executor 不能直接写文件"""
        executor_dir = Path("pdf2epub/core/executor")

        forbidden_patterns = [
            r'\.write_text\s*\(',
            r'open\s*\([^)]*["\']w["\']',
            r'Path\s*\([^)]*\)\.write',
        ]

        for py_file in executor_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                if re.search(pattern, content):
                    raise AssertionError(f"{py_file} has file write: {pattern}")

    def test_phase_uses_only_persistence(self):
        """Phase 只能通过 Persistence 保存"""
        phase_dir = Path("pdf2epub/core/phase")

        forbidden_patterns = [
            r'\.write_text\s*\(',
            r'open\s*\([^)]*["\']w["\']',
        ]

        for py_file in phase_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                if re.search(pattern, content):
                    raise AssertionError(
                        f"{py_file} writes directly instead of using Persistence"
                    )
```

---

## 5. ErrorType 统一决策

**问题**: model_chain.ErrorType 有 WRONG_LANGUAGE/EMPTY_RESPONSE 等

**决策**:
- `model_chain.py` 标记为 **legacy**
- 新架构只用 `core/types.py` 的 ErrorType
- 如果需要细粒度语义，在 hooks 层用 metadata/reason 字段承载

**迁移**:
```python
# core/model_chain.py - 添加 deprecation
import warnings
warnings.warn(
    "model_chain.ErrorType is deprecated. Use core.types.ErrorType instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

## 6. 导入守卫（import-time fail fast）

```python
# core/work_unit.py - 删除 WorkUnit 类后添加
"""
Legacy module - WorkUnit has moved.

This file is kept for WorkUnitDiscovery only.
"""
def __getattr__(name):
    if name == "WorkUnit":
        raise ImportError(
            "WorkUnit has moved to core.types. "
            "Use: from pdf2epub.core.types import WorkUnit"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## 实施顺序

| 步骤 | 内容 | Fail Fast 机制 |
|------|------|----------------|
| 1 | 创建 `core/types.py` | 无（新文件） |
| 2 | 添加测试 `TestSingleSourceOfTruth` | 测试时 fail |
| 3 | 迁移 WorkUnit 到 types.py | `__post_init__` 显式异常 |
| 4 | 迁移 ErrorType 到 types.py | 测试扫描 |
| 5 | 添加测试 `TestLegacyIsolation` | 测试时 fail |
| 6 | 添加测试 `TestBoundaryEnforcement` | 测试时 fail |
| 7 | 添加导入守卫到旧模块 | import 时 fail |
| 8 | 更新所有 import 到 types.py | - |
| 9 | 更新 ContextInjector 使用新 WorkUnit | - |

---

## 成功标准

运行以下测试全部通过：
```bash
pytest tests/test_architecture.py -v
```

测试覆盖：
- [ ] WorkUnit 只定义一次
- [ ] ErrorType 只定义一次
- [ ] SplitType 只定义一次
- [ ] 新代码不 import legacy
- [ ] Pipeline 无重试循环
- [ ] Pipeline 无 longest fallback
- [ ] Pipeline 不直接调用 LLM
- [ ] Executor 不直接写文件
- [ ] Phase 只通过 Persistence 写入
