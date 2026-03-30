> **OUTDATED**: 本文档已被 `executor-design-v2.md` 取代。仅供历史参考。

# PDF2EPUB 架构重设计：防止重造轮子

## Claude 违规模式总结

通过分析当前代码库，识别出以下 Claude 违规模式：

| 违规模式 | 具体表现 | 出现次数 |
|---------|---------|---------|
| **覆盖基类方法** | `_batch_validate_and_save` 被完全重写 | 2处 |
| **不继承基类** | batch_translator/polisher 完全独立 | 2处 |
| **复制粘贴代码** | entity loading, prompt building, validation | 10+处 |
| **分散状态管理** | ProcessingTracker + batch_state 两套系统 | 4处 |
| **硬编码配置** | validation_mode, auto_save 各处硬编码 | 4处 |
| **重复初始化** | truncation detector, split manager | 6处 |

## 设计原则：让违规不可能

1. **组合优于继承** - 注入组件，无法覆盖
2. **Protocol 定义接口** - 只能实现规定的方法
3. **@final 禁止覆盖** - 核心方法不能重写
4. **__init_subclass__ 检查** - 违规时 import 就崩溃
5. **单例注册表** - 组件只能通过注册表获取
6. **私有化状态** - 双下划线防止访问
7. **AST 测试守护** - CI 检查代码结构

---

## 新架构设计

### 目录结构

```
pdf2epub/
├── core/                        # 核心组件 - 禁止修改
│   ├── __init__.py
│   ├── _frozen.py               # 冻结基类，禁止继承
│   ├── pipeline.py              # ProcessingPipeline - 唯一处理流程
│   ├── validation.py            # ValidationPipeline - 唯一验证逻辑
│   ├── persistence.py           # ResultPersistence - 唯一保存逻辑
│   ├── state.py                 # StateManager - 单一状态真相
│   └── registry.py              # ComponentRegistry - 组件注册表
│
├── processors/                   # 处理器 - 只能实现 Protocol
│   ├── __init__.py
│   ├── _protocol.py             # ProcessorProtocol 定义
│   ├── polish.py                # 只实现 build_prompt, clean_response
│   └── translate.py             # 只实现 build_prompt, clean_response
│
├── validators/                   # 验证器 - 通过注册表使用
│   ├── __init__.py
│   ├── _protocol.py             # ValidatorProtocol 定义
│   ├── ngram.py                 # N-gram 验证
│   ├── agent.py                 # Agent 验证
│   └── chinese.py               # 中文验证
│
├── prompts/                      # Prompt 构建 - 纯函数
│   ├── __init__.py
│   ├── polish.py                # create_polish_prompt()
│   ├── translate.py             # create_translate_prompt()
│   └── entities.py              # create_entity_reference() - 唯一实现
│
└── commands/                     # CLI - 组装组件
    ├── polish.py
    └── translate.py
```

---

### 核心组件设计

#### 1. 冻结基类 (core/_frozen.py)

```python
"""
冻结基类：禁止继承、禁止覆盖、禁止修改。
任何违规在 import 时立即崩溃。
"""
from typing import final, ClassVar, Set
import inspect


class FrozenMeta(type):
    """元类：禁止继承被标记为 frozen 的类"""

    _frozen_classes: ClassVar[Set[str]] = set()

    def __new__(mcs, name, bases, namespace, frozen=False, **kwargs):
        # 检查是否试图继承 frozen 类
        for base in bases:
            if base.__name__ in mcs._frozen_classes:
                raise TypeError(
                    f"类 {name} 试图继承被冻结的类 {base.__name__}。"
                    f"这是禁止的。请使用组合而非继承。"
                )

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if frozen:
            mcs._frozen_classes.add(name)

        return cls


class Frozen(metaclass=FrozenMeta):
    """
    被冻结的基类。子类：
    1. 不能被继承
    2. @final 方法不能被覆盖
    3. 禁止定义特定方法名
    """

    # 子类禁止定义的方法名
    _FORBIDDEN_METHODS: ClassVar[Set[str]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 检查禁止的方法
        for method_name in cls._FORBIDDEN_METHODS:
            if method_name in cls.__dict__:
                raise TypeError(
                    f"类 {cls.__name__} 定义了被禁止的方法 '{method_name}'。\n"
                    f"该功能由核心组件提供，不允许自定义实现。"
                )
```

#### 2. 处理器协议 (processors/_protocol.py)

```python
"""
处理器协议：定义 Processor 必须且只能实现的接口。
"""
from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class ProcessorProtocol(Protocol):
    """
    Processor 必须实现的接口。

    注意：Processor 只负责：
    1. 构建 prompt
    2. 清理 response
    3. 后处理结果

    Processor 不负责（由 Pipeline 处理）：
    - 验证
    - 保存
    - 状态管理
    - 重试逻辑
    """

    @property
    def name(self) -> str:
        """处理器名称，用于日志和状态跟踪"""
        ...

    def build_prompt(self, content: str, context: "ProcessContext") -> str:
        """
        构建发送给 LLM 的 prompt。

        Args:
            content: 要处理的内容
            context: 处理上下文（文件名、part 信息等）

        Returns:
            完整的 prompt 字符串
        """
        ...

    def clean_response(self, response: str) -> str:
        """
        清理 LLM 返回的 response。

        Args:
            response: 原始 LLM response

        Returns:
            清理后的 response
        """
        ...

    def post_process(self, result: str, context: "ProcessContext") -> str:
        """
        后处理结果。

        Args:
            result: 清理后的 response
            context: 处理上下文

        Returns:
            最终结果
        """
        ...


class ProcessContext:
    """处理上下文 - 不可变"""

    __slots__ = ('file_key', 'part_index', 'total_parts', 'book_title',
                 'source_language', 'target_language', 'extra')

    def __init__(
        self,
        file_key: str,
        book_title: str,
        part_index: int = 1,
        total_parts: int = 1,
        source_language: str = "",
        target_language: str = "",
        extra: Dict[str, Any] = None
    ):
        object.__setattr__(self, 'file_key', file_key)
        object.__setattr__(self, 'book_title', book_title)
        object.__setattr__(self, 'part_index', part_index)
        object.__setattr__(self, 'total_parts', total_parts)
        object.__setattr__(self, 'source_language', source_language)
        object.__setattr__(self, 'target_language', target_language)
        object.__setattr__(self, 'extra', extra or {})

    def __setattr__(self, name, value):
        raise AttributeError("ProcessContext is immutable")
```

#### 3. 验证管道 (core/validation.py)

```python
"""
验证管道：唯一的验证实现。
Processor 无法覆盖或绑过这个逻辑。
"""
from typing import Dict, List, final
from dataclasses import dataclass
from .._frozen import Frozen


@dataclass
class ValidationResult:
    """验证结果"""
    key: str
    is_valid: bool
    reason: str
    confidence: str = "high"


class ValidationPipeline(Frozen, frozen=True):
    """
    验证管道 - 被冻结，不能继承。

    实现两阶段验证：
    1. Phase 1: 快速筛选（N-gram）
    2. Phase 2: Agent 精确验证
    """

    def __init__(self, validators: List["ValidatorProtocol"]):
        self._validators = validators

    @final
    def validate_batch(
        self,
        results: Dict[str, str],
        originals: Dict[str, str]
    ) -> Dict[str, ValidationResult]:
        """
        批量验证 - 不能被覆盖。

        Args:
            results: {key: processed_content}
            originals: {key: original_content}

        Returns:
            {key: ValidationResult}
        """
        validation_results = {}
        suspicious = {}

        # Phase 1: 快速筛选
        for key, processed in results.items():
            original = originals.get(key, "")

            # 所有 validator 依次检查
            is_valid = True
            reason = "passed"

            for validator in self._validators:
                if validator.phase != 1:
                    continue
                result = validator.validate(original, processed, key)
                if not result.is_valid:
                    is_valid = False
                    reason = result.reason
                    suspicious[key] = (original, processed)
                    break

            if is_valid:
                validation_results[key] = ValidationResult(
                    key=key, is_valid=True, reason=reason
                )

        # Phase 2: Agent 验证 suspicious
        if suspicious:
            phase2_validators = [v for v in self._validators if v.phase == 2]
            for validator in phase2_validators:
                for key, (original, processed) in suspicious.items():
                    if key in validation_results:
                        continue
                    result = validator.validate(original, processed, key)
                    validation_results[key] = result

        return validation_results
```

#### 4. 持久化管理 (core/persistence.py)

```python
"""
持久化管理：唯一的保存逻辑。
实现 raw -> validated 的两阶段保存。
"""
from pathlib import Path
from typing import Dict, final
from .._frozen import Frozen


class ResultPersistence(Frozen, frozen=True):
    """
    结果持久化 - 被冻结，不能继承。

    保存策略：
    1. LLM response 立即保存到 raw/
    2. 验证通过后移动到 validated/
    3. 聚合后保存到根目录
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._raw_dir = output_dir / "raw"
        self._validated_dir = output_dir / "validated"

        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._validated_dir.mkdir(parents=True, exist_ok=True)

    @final
    def save_raw(self, key: str, content: str) -> Path:
        """
        保存原始 LLM response - 不能被覆盖。
        在验证前调用，确保数据不丢失。
        """
        path = self._raw_dir / f"{key}.md"
        path.write_text(content, encoding='utf-8')
        return path

    @final
    def save_raw_batch(self, results: Dict[str, str]) -> None:
        """批量保存原始响应"""
        for key, content in results.items():
            self.save_raw(key, content)

    @final
    def promote_to_validated(self, key: str) -> Path:
        """
        将 raw 文件提升为 validated - 不能被覆盖。
        验证通过后调用。
        """
        raw_path = self._raw_dir / f"{key}.md"
        validated_path = self._validated_dir / f"{key}.md"

        if raw_path.exists():
            content = raw_path.read_text(encoding='utf-8')
            validated_path.write_text(content, encoding='utf-8')

        return validated_path

    @final
    def aggregate_parts(self, base_key: str, part_keys: List[str]) -> Path:
        """聚合多个 part 文件"""
        parts = []
        for key in sorted(part_keys):
            path = self._validated_dir / f"{key}.md"
            if path.exists():
                parts.append(path.read_text(encoding='utf-8'))

        output_path = self._output_dir / f"{base_key}.md"
        output_path.write_text('\n\n'.join(parts), encoding='utf-8')
        return output_path

    @final
    def get_raw_content(self, key: str) -> str:
        """读取 raw 内容（用于重新验证）"""
        path = self._raw_dir / f"{key}.md"
        if path.exists():
            return path.read_text(encoding='utf-8')
        return ""

    @final
    def has_validated(self, key: str) -> bool:
        """检查是否已有 validated 文件"""
        return (self._validated_dir / f"{key}.md").exists()
```

#### 5. 状态管理 (core/state.py)

```python
"""
状态管理：单一真相来源。
用文件系统作为状态，消除 tracker 和文件不一致的问题。
"""
from pathlib import Path
from typing import Set, final
from .._frozen import Frozen


class StateManager(Frozen, frozen=True):
    """
    状态管理 - 被冻结，不能继承。

    设计原则：文件存在 = 完成
    不再维护独立的 tracker.json，以文件系统为准。
    """

    def __init__(self, persistence: "ResultPersistence"):
        self._persistence = persistence

    @final
    def is_complete(self, key: str) -> bool:
        """
        检查是否完成 - 不能被覆盖。

        完成的定义：validated 目录中存在该文件。
        """
        return self._persistence.has_validated(key)

    @final
    def get_pending_keys(self, all_keys: Set[str]) -> Set[str]:
        """获取待处理的 keys"""
        return {k for k in all_keys if not self.is_complete(k)}

    @final
    def get_raw_keys_for_revalidation(self) -> Set[str]:
        """
        获取需要重新验证的 keys。
        （raw 中存在但 validated 中不存在）
        """
        raw_dir = self._persistence._raw_dir
        validated_dir = self._persistence._validated_dir

        raw_keys = {f.stem for f in raw_dir.glob("*.md")}
        validated_keys = {f.stem for f in validated_dir.glob("*.md")}

        return raw_keys - validated_keys
```

#### 6. 处理管道 (core/pipeline.py)

```python
"""
处理管道：唯一的处理流程。
Processor 只是这个管道中的一个组件。
"""
from typing import Dict, List, final, TYPE_CHECKING
from dataclasses import dataclass
from .._frozen import Frozen
from ..processors._protocol import ProcessorProtocol, ProcessContext

if TYPE_CHECKING:
    from .validation import ValidationPipeline
    from .persistence import ResultPersistence
    from .state import StateManager


@dataclass
class ProcessingResult:
    """处理结果"""
    total: int
    completed: int
    failed: int
    failed_keys: List[str]


class ProcessingPipeline(Frozen, frozen=True):
    """
    处理管道 - 被冻结，不能继承。

    这是唯一的处理流程：
    1. 发现待处理单元
    2. 调用 Processor 生成内容
    3. 立即保存到 raw
    4. 批量验证
    5. 验证通过的移动到 validated
    6. 聚合 parts
    """

    def __init__(
        self,
        processor: ProcessorProtocol,
        validation: "ValidationPipeline",
        persistence: "ResultPersistence",
        state: "StateManager",
        llm_client: "LLMClient"
    ):
        # 检查 processor 是否符合协议
        if not isinstance(processor, ProcessorProtocol):
            raise TypeError(
                f"{processor} 不符合 ProcessorProtocol。"
                f"请确保实现了 build_prompt, clean_response, post_process 方法。"
            )

        self._processor = processor
        self._validation = validation
        self._persistence = persistence
        self._state = state
        self._llm_client = llm_client

    @final
    def process_all(
        self,
        units: List["WorkUnit"],
        max_workers: int = 4
    ) -> ProcessingResult:
        """
        处理所有单元 - 不能被覆盖。

        这是唯一的处理入口。
        """
        # 1. 检查是否有 raw 文件需要重新验证
        raw_keys = self._state.get_raw_keys_for_revalidation()
        if raw_keys:
            self._revalidate_raw(raw_keys, units)

        # 2. 获取待处理单元
        all_keys = {u.id for u in units}
        pending_keys = self._state.get_pending_keys(all_keys)
        pending_units = [u for u in units if u.id in pending_keys]

        if not pending_units:
            return ProcessingResult(
                total=len(units),
                completed=len(units),
                failed=0,
                failed_keys=[]
            )

        # 3. 处理并立即保存 raw
        results = {}
        originals = {}

        for unit in pending_units:
            context = self._build_context(unit)

            # 调用 Processor 构建 prompt
            prompt = self._processor.build_prompt(unit.content, context)

            # 调用 LLM
            response = self._llm_client.generate(prompt)

            # 清理 response
            cleaned = self._processor.clean_response(response)

            # 后处理
            final_result = self._processor.post_process(cleaned, context)

            # 立即保存到 raw（防止崩溃丢失数据）
            self._persistence.save_raw(unit.id, final_result)

            results[unit.id] = final_result
            originals[unit.id] = unit.content

        # 4. 批量验证
        validation_results = self._validation.validate_batch(results, originals)

        # 5. 验证通过的移动到 validated
        failed_keys = []
        for key, vr in validation_results.items():
            if vr.is_valid:
                self._persistence.promote_to_validated(key)
            else:
                failed_keys.append(key)

        # 6. 聚合 parts
        self._aggregate_all(units)

        completed = len(units) - len(failed_keys)
        return ProcessingResult(
            total=len(units),
            completed=completed,
            failed=len(failed_keys),
            failed_keys=failed_keys
        )

    def _revalidate_raw(self, raw_keys: Set[str], units: List["WorkUnit"]):
        """重新验证 raw 文件"""
        results = {}
        originals = {}

        unit_map = {u.id: u for u in units}

        for key in raw_keys:
            content = self._persistence.get_raw_content(key)
            if content and key in unit_map:
                results[key] = content
                originals[key] = unit_map[key].content

        if results:
            validation_results = self._validation.validate_batch(results, originals)
            for key, vr in validation_results.items():
                if vr.is_valid:
                    self._persistence.promote_to_validated(key)

    def _build_context(self, unit: "WorkUnit") -> ProcessContext:
        """构建处理上下文"""
        return ProcessContext(
            file_key=unit.file_key,
            book_title=self._processor.name,
            part_index=unit.part_index or 1,
            total_parts=unit.total_parts or 1
        )

    def _aggregate_all(self, units: List["WorkUnit"]):
        """聚合所有 parts"""
        from collections import defaultdict

        groups = defaultdict(list)
        for unit in units:
            if unit.part_index:
                groups[unit.file_key].append(unit.id)

        for base_key, part_keys in groups.items():
            if len(part_keys) > 1:
                # 检查所有 parts 都已 validated
                all_validated = all(
                    self._state.is_complete(k) for k in part_keys
                )
                if all_validated:
                    self._persistence.aggregate_parts(base_key, part_keys)
```

#### 7. 组件注册表 (core/registry.py)

```python
"""
组件注册表：集中管理所有组件。
禁止直接实例化，必须通过注册表获取。
"""
from typing import Dict, Type, TypeVar, final
from .._frozen import Frozen


T = TypeVar('T')


class ComponentRegistry(Frozen, frozen=True):
    """
    组件注册表 - 被冻结，不能继承。

    所有组件必须通过注册表获取，禁止直接实例化。
    """

    _validators: Dict[str, Type] = {}
    _processors: Dict[str, Type] = {}

    @classmethod
    @final
    def register_validator(cls, name: str, validator_cls: Type) -> None:
        """注册验证器"""
        if name in cls._validators:
            raise ValueError(
                f"验证器 '{name}' 已注册。"
                f"不允许重复注册，请使用已有的验证器。"
            )
        cls._validators[name] = validator_cls

    @classmethod
    @final
    def register_processor(cls, name: str, processor_cls: Type) -> None:
        """注册处理器"""
        if name in cls._processors:
            raise ValueError(
                f"处理器 '{name}' 已注册。"
                f"不允许重复注册。"
            )
        cls._processors[name] = processor_cls

    @classmethod
    @final
    def get_validator(cls, name: str):
        """获取验证器"""
        if name not in cls._validators:
            raise KeyError(
                f"未知验证器: '{name}'。"
                f"可用的验证器: {list(cls._validators.keys())}"
            )
        return cls._validators[name]()

    @classmethod
    @final
    def get_processor(cls, name: str):
        """获取处理器"""
        if name not in cls._processors:
            raise KeyError(
                f"未知处理器: '{name}'。"
                f"可用的处理器: {list(cls._processors.keys())}"
            )
        return cls._processors[name]()
```

---

### Processor 实现示例

#### polish.py

```python
"""
Polish 处理器：只负责构建 prompt 和清理 response。
验证、保存、状态管理全部由 Pipeline 处理。
"""
from ..core.registry import ComponentRegistry
from ._protocol import ProcessorProtocol, ProcessContext
from ..prompts.polish import create_polish_prompt


class PolishProcessor:
    """
    Polish 处理器。

    注意：这个类只实现 ProcessorProtocol 定义的接口。
    不允许添加 validate, save, _batch_validate_and_save 等方法。
    """

    # 禁止定义的方法 - 由 AST 测试守护
    # _FORBIDDEN = {'validate', 'save', '_batch_validate', '_save_result'}

    @property
    def name(self) -> str:
        return "polish"

    def build_prompt(self, content: str, context: ProcessContext) -> str:
        """构建 polish prompt"""
        return create_polish_prompt(
            content=content,
            chapter_name=context.file_key,
            book_title=context.book_title,
            part_idx=context.part_index,
            total_parts=context.total_parts
        )

    def clean_response(self, response: str) -> str:
        """清理 markdown response"""
        # 移除 code block
        lines = response.strip().split('\n')
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines)

    def post_process(self, result: str, context: ProcessContext) -> str:
        """后处理（polish 不需要额外处理）"""
        return result


# 注册到注册表
ComponentRegistry.register_processor("polish", PolishProcessor)
```

---

### 测试守护

#### test_architecture.py

```python
"""
架构测试：确保没有人违反设计原则。
"""
import ast
from pathlib import Path
import pytest


FORBIDDEN_METHODS = {
    'validate', 'save', '_batch_validate', '_save_result',
    '_batch_validate_and_save', '_validate_batch', 'save_raw'
}

FORBIDDEN_IMPORTS_IN_PROCESSORS = {
    'ProcessingTracker', 'BatchTranslateState', 'BatchPolishState'
}


def test_no_validation_methods_in_processors():
    """确保 processors/ 中没有验证方法"""
    processors_dir = Path('pdf2epub/processors')

    for py_file in processors_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue

        tree = ast.parse(py_file.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in FORBIDDEN_METHODS:
                    pytest.fail(
                        f"{py_file}:{node.lineno} 定义了被禁止的方法 '{node.name}'。\n"
                        f"验证和保存逻辑应该由 core/pipeline.py 处理。"
                    )


def test_no_state_management_in_processors():
    """确保 processors/ 中没有状态管理"""
    processors_dir = Path('pdf2epub/processors')

    for py_file in processors_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue

        tree = ast.parse(py_file.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in FORBIDDEN_IMPORTS_IN_PROCESSORS:
                        pytest.fail(
                            f"{py_file}:{node.lineno} 导入了被禁止的类 '{alias.name}'。\n"
                            f"状态管理应该由 core/state.py 处理。"
                        )


def test_processors_implement_protocol():
    """确保所有 processors 实现 ProcessorProtocol"""
    from pdf2epub.processors._protocol import ProcessorProtocol
    from pdf2epub.core.registry import ComponentRegistry

    for name, cls in ComponentRegistry._processors.items():
        instance = cls()
        assert isinstance(instance, ProcessorProtocol), \
            f"处理器 '{name}' 不符合 ProcessorProtocol"


def test_no_direct_file_io_in_processors():
    """确保 processors 不直接进行文件 IO"""
    processors_dir = Path('pdf2epub/processors')

    forbidden_calls = {'open', 'write_text', 'read_text', 'mkdir'}

    for py_file in processors_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue

        tree = ast.parse(py_file.read_text())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in forbidden_calls:
                        pytest.fail(
                            f"{py_file}:{node.lineno} 调用了被禁止的方法 '{node.func.attr}'。\n"
                            f"文件 IO 应该由 core/persistence.py 处理。"
                        )


def test_core_classes_are_frozen():
    """确保核心类不能被继承"""
    from pdf2epub.core._frozen import FrozenMeta

    frozen_classes = {'ValidationPipeline', 'ResultPersistence',
                      'StateManager', 'ProcessingPipeline', 'ComponentRegistry'}

    assert frozen_classes.issubset(FrozenMeta._frozen_classes), \
        "某些核心类没有被正确冻结"
```

---

## 迁移计划

### Phase 1: 基础设施
1. 创建 core/ 目录和 _frozen.py
2. 实现 ProcessorProtocol
3. 实现 ValidationPipeline（提取现有逻辑）
4. 实现 ResultPersistence（raw -> validated 保存）
5. 实现 StateManager（文件系统状态）

### Phase 2: 迁移 Processors
1. 重写 PolishProcessor（只保留 prompt 构建）
2. 重写 TranslateProcessor（只保留 prompt 构建）
3. 删除旧的验证/保存代码

### Phase 3: 迁移 Batch Processors
1. BatchPolishProcessor 复用 ProcessingPipeline
2. BatchTranslateProcessor 复用 ProcessingPipeline
3. 删除重复的 state 类

### Phase 4: 测试和清理
1. 添加架构测试
2. 删除旧代码
3. 更新文档

---

## 预期效果

| 违规尝试 | 结果 |
|---------|------|
| 继承 ValidationPipeline | import 时 TypeError |
| 在 Processor 中定义 validate | import 时 TypeError（__init_subclass__） |
| 在 Processor 中导入 ProcessingTracker | CI 测试失败 |
| 覆盖 @final 方法 | mypy 报错 |
| 直接实例化未注册的组件 | KeyError |
| 重复注册组件 | ValueError |

**重造轮子的难度：几乎不可能。**
