# Pipeline + Executor + Hooks 架构设计 V2

## 设计理念

**核心目标**：整章进去，整章出来。

**现实**：翻译充满扑朔迷离和 special cases。

**核心原则**：所有脱离主目标的逻辑都应该是 Hooks，不应该污染主流程。

| 脱离主目标的情况 | 处理方式 | 不应该做的 |
|-----------------|---------|-----------|
| 跳过处理（image-only, empty） | PreProcessor hook | 在 Pipeline/Executor 里 if-else |
| 修改结果（restore images） | Transformer hook | 在 Executor 里硬编码 |
| 验证失败 → 重试 | Validator hook → ErrorEffect → 状态更新 + 重新入池 | retry for loop |
| API 错误 → 重试 | ErrorClassifier hook → ErrorEffect → 状态更新 + 重新入池 | retry for loop |
| 跳过验证（front_matter） | SkipValidator hook | 在 Pipeline 里 if-else |

**解决方案**：
- 主流程（Pipeline + Executor）保持纯粹：加载 → 调用 LLM → 保存
- 所有 edge cases 通过 Hooks 处理
- 失败重试通过状态更新 + 重新入池，无 retry for loop
- Batch 和 Online 同时进行，统一的 model chain
- 动态分割变成依赖树的一部分（虚拟 units）
- 阶段之间通用接口，任意组合

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│ Pipeline                                                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ SplitManager│    │   Tracker   │    │ ContextInjector     │  │
│  │ (proactive) │    │  (resume)   │    │ (dependency order)  │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Executor（统一处理 batch + online）                          ││
│  │  ┌──────────────────────────────────────────────────────┐   ││
│  │  │ ThreadPoolExecutor (并发) + Batch Job (异步)          │   ││
│  │  │                                                       │   ││
│  │  │  Unit States:                                         │   ││
│  │  │  ┌─────────────────────────────────────────────────┐ │   ││
│  │  │  │ unit_id → { chain, total_quota, quotas }        │ │   ││
│  │  │  └─────────────────────────────────────────────────┘ │   ││
│  │  │                                                       │   ││
│  │  │  pending ←──┐                                         │   ││
│  │  │      │      │ 失败后状态更新 + 重新入池               │   ││
│  │  │      ▼      │ （无 retry loop）                       │   ││
│  │  │  process ───┴─→ success / fail → longest fallback     │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Hooks (可配置)                                               ││
│  │  PreProcessors | Transformers | Validators | ErrorClassifier ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │ BatchValidator  │    │ Persistence (save results)          │ │
│  │ (batch 验证)    │    │                                     │ │
│  └─────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 一、Hooks 体系

### 1.1 Hooks 分类

```
ExecutionHooks
├── PreProcessors (处理前)
│   ├── ImageOnlyFilter      → 纯图片跳过
│   └── EmptyContentFilter   → 空内容跳过
│
├── Transformers (修改结果)
│   ├── RestoreImagesTransformer
│   └── RemoveArtifactsTransformer
│
├── Validators (判断结果)
│   ├── LengthValidator      → 长度检查
│   └── NGramValidator       → 内容完整性
│
├── SkipValidators (跳过验证)
│   └── ChapterTypeSkipper   → front_matter 等跳过
│
├── ErrorClassifier (错误分类)
│   └── DefaultErrorClassifier → 返回 ErrorEffect
│
└── BatchValidator (批量验证，介于 hook 和 pipeline 之间)
    ├── TranslationBatchValidator
    └── PolishBatchValidator
```

### 1.2 Protocol 定义

```python
# ============================================================
# Pre-processing
# ============================================================

@dataclass
class PreProcessResult:
    should_process: bool
    skip_reason: str = ""
    fallback_result: Optional[str] = None


class PreProcessor(Protocol):
    @property
    def name(self) -> str: ...

    def check(
        self,
        key: str,
        content: str,
        context: ProcessContext
    ) -> PreProcessResult: ...


# ============================================================
# Post-processing: Transform + Validate
# ============================================================

@dataclass
class HookResult:
    accepted: bool
    context_ready: bool


class Transformer(Protocol):
    @property
    def name(self) -> str: ...

    def transform(
        self,
        key: str,
        original: str,
        result: str
    ) -> str: ...


class Validator(Protocol):
    @property
    def name(self) -> str: ...

    def validate(
        self,
        key: str,
        original: str,
        result: str
    ) -> HookResult: ...


# ============================================================
# Skip Validation
# ============================================================

class SkipValidator(Protocol):
    @property
    def name(self) -> str: ...

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: ProcessContext
    ) -> bool: ...


# ============================================================
# Error Classification
# ============================================================

class ErrorType(Enum):
    SAFETY = "safety"
    NETWORK = "network"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


@dataclass
class ErrorEffect:
    """错误如何影响 unit 状态"""
    remove_current_model: bool  # 是否从 chain 移除当前模型
    remove_provider: bool       # 是否移除整个 provider（safety block）
    remove_all_batch: bool      # 是否移除所有 batch entries
    quota_type: ErrorType       # 消耗哪种 quota


class ErrorClassifier(Protocol):
    def classify(self, error: Exception) -> ErrorType: ...

    def get_effect(self, error_type: ErrorType) -> ErrorEffect: ...


# ============================================================
# Batch Validation (介于 hook 和 pipeline)
# ============================================================

@dataclass
class BatchValidationResult:
    passed: Set[str]
    failed: Set[str]
    records: List[ValidationRecord]


class BatchValidator(Protocol):
    @property
    def name(self) -> str: ...

    def validate_batch(
        self,
        files: Dict[str, VerificationFile],
        skip_keys: Set[str] = None
    ) -> BatchValidationResult: ...
```

### 1.3 CompositeHooks 实现

```python
class CompositeHooks:
    def __init__(
        self,
        pre_processors: List[PreProcessor] = None,
        transformers: List[Transformer] = None,
        validators: List[Validator] = None,
        skip_validators: List[SkipValidator] = None,
        error_classifier: ErrorClassifier = None,
        tracker: ProcessingTracker = None
    ):
        self._pre_processors = pre_processors or []
        self._transformers = transformers or []
        self._validators = validators or []
        self._skip_validators = skip_validators or []
        self._error_classifier = error_classifier or DefaultErrorClassifier()
        self._tracker = tracker

    def pre_process(
        self,
        key: str,
        content: str,
        context: ProcessContext
    ) -> PreProcessResult:
        """任一 pre_processor 说跳过就跳过"""
        for pp in self._pre_processors:
            result = pp.check(key, content, context)
            if not result.should_process:
                return result
        return PreProcessResult(should_process=True)

    def post_process(
        self,
        key: str,
        original: str,
        result: str,
        chapter_type: str = "",
        context: ProcessContext = None
    ) -> Tuple[str, HookResult]:
        """Transform + Validate"""
        # Step 1: Transform（链式）
        transformed = result
        for t in self._transformers:
            transformed = t.transform(key, original, transformed)

        # Step 2: Skip validation check
        for sv in self._skip_validators:
            if sv.should_skip(key, chapter_type, context):
                return (transformed, HookResult(accepted=True, context_ready=True))

        # Step 3: Validate（screener/final 逻辑）
        accepted = True
        context_ready = False

        for v in self._validators:
            hook_result = v.validate(key, original, transformed)

            if self._tracker:
                self._tracker.record_validation(key, {
                    "validator": v.name,
                    "accepted": hook_result.accepted,
                    "context_ready": hook_result.context_ready
                })

            if hook_result.context_ready:
                context_ready = True

            if not hook_result.accepted:
                accepted = False
                break

        return (transformed, HookResult(accepted=accepted, context_ready=context_ready))

    def classify_error(self, error: Exception) -> Tuple[ErrorType, ErrorEffect]:
        """分类错误并返回影响"""
        error_type = self._error_classifier.classify(error)
        effect = self._error_classifier.get_effect(error_type)
        return (error_type, effect)

    def get_error_effect(self, error_type: ErrorType) -> ErrorEffect:
        """根据已知的 error_type 获取 effect（不需要重新分类）"""
        return self._error_classifier.get_effect(error_type)
```

### 1.4 内置 Hooks 实现

#### PreProcessors

```python
class ImageOnlyFilter(PreProcessor):
    def __init__(self, book_structure: BookStructure):
        self._book_structure = book_structure

    @property
    def name(self) -> str:
        return "ImageOnlyFilter"

    def check(self, key, content, context):
        if self._book_structure.is_image_only_content(content):
            return PreProcessResult(
                should_process=False,
                skip_reason="Image-only content",
                fallback_result=content
            )
        return PreProcessResult(should_process=True)


class EmptyContentFilter(PreProcessor):
    @property
    def name(self) -> str:
        return "EmptyContentFilter"

    def check(self, key, content, context):
        if not content or not content.strip():
            return PreProcessResult(
                should_process=False,
                skip_reason="Empty content",
                fallback_result=""
            )
        return PreProcessResult(should_process=True)
```

#### Transformers

```python
class RestoreImagesTransformer(Transformer):
    @property
    def name(self) -> str:
        return "RestoreImages"

    def transform(self, key, original, result):
        from .utils import restore_lost_images_fast
        return restore_lost_images_fast(original, result)


class RemoveArtifactsTransformer(Transformer):
    """移除 LLM 常见 artifact"""

    PATTERNS = [
        r'^```\w*\n',           # 开头的 code block
        r'\n```$',              # 结尾的 code block
        r'^Here is the .*:\n',  # "Here is the translation:"
    ]

    @property
    def name(self) -> str:
        return "RemoveArtifacts"

    def transform(self, key, original, result):
        import re
        cleaned = result
        for pattern in self.PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)
        return cleaned
```

#### SkipValidators

```python
class ChapterTypeSkipper(SkipValidator):
    SKIP_TYPES = {"front_matter", "back_matter", "notes", "appendix", "toc"}

    @property
    def name(self) -> str:
        return "ChapterTypeSkipper"

    def should_skip(self, key, chapter_type, context):
        return chapter_type in self.SKIP_TYPES
```

#### ErrorClassifier

```python
class DefaultErrorClassifier(ErrorClassifier):
    EFFECTS = {
        ErrorType.SAFETY: ErrorEffect(
            remove_current_model=True,
            remove_provider=True,        # Safety block 移除整个 provider
            remove_all_batch=False,
            quota_type=ErrorType.SAFETY
        ),
        ErrorType.NETWORK: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
        ErrorType.VALIDATION: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.VALIDATION
        ),
        ErrorType.RATE_LIMIT: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
        ErrorType.UNKNOWN: ErrorEffect(
            remove_current_model=False,
            remove_provider=False,
            remove_all_batch=False,
            quota_type=ErrorType.NETWORK
        ),
    }

    def classify(self, error: Exception) -> ErrorType:
        msg = str(error).lower()

        if any(kw in msg for kw in ["safety", "blocked", "harmful", "content filter"]):
            return ErrorType.SAFETY
        if any(kw in msg for kw in ["rate limit", "quota", "429"]):
            return ErrorType.RATE_LIMIT
        if any(kw in msg for kw in ["timeout", "connection", "network", "503", "502"]):
            return ErrorType.NETWORK
        if any(kw in msg for kw in ["truncat", "incomplete", "validation"]):
            return ErrorType.VALIDATION

        return ErrorType.UNKNOWN

    def get_effect(self, error_type: ErrorType) -> ErrorEffect:
        return self.EFFECTS.get(error_type, self.EFFECTS[ErrorType.UNKNOWN])
```

---

## 二、统一的 Model Chain

### 2.1 Chain Entry

Model chain 包含 batch 和 online 两种模式的 entries：

```python
@dataclass
class ChainEntry:
    provider: str                      # "gemini", "deepseek", "anthropic"
    model: str                         # "gemini-2.0-flash", "deepseek-chat"
    mode: Literal["batch", "online"]   # 执行模式

# 典型的 chain
DEFAULT_CHAIN = [
    ChainEntry("gemini", "gemini-2.0-flash", mode="batch"),
    ChainEntry("gemini", "gemini-2.0-flash", mode="online"),
    ChainEntry("deepseek", "deepseek-chat", mode="online"),
    ChainEntry("anthropic", "claude-3-haiku", mode="online"),
]
```

### 2.2 Chain 操作

```python
def remove_batch_entries(chain: List[ChainEntry]) -> List[ChainEntry]:
    """移除所有 batch entries（少于 threshold 失败时）"""
    return [e for e in chain if e.mode != "batch"]

def remove_provider(chain: List[ChainEntry], provider: str) -> List[ChainEntry]:
    """移除指定 provider 的所有 entries（safety block 时）"""
    return [e for e in chain if e.provider != provider]
```

### 2.3 统一的错误处理

ErrorEffect 和 EFFECTS 定义见 Section 1.2 和 1.4（DefaultErrorClassifier）。

**Chain 如何响应错误**：
```python
# Safety block: 移除同 provider 的 batch + online
if effect.remove_provider:
    state.chain = remove_provider(state.chain, current_entry.provider)

# Batch 少量失败: 移除所有 batch entries
if effect.remove_all_batch:
    state.chain = remove_batch_entries(state.chain)

# 普通失败: 只消耗 quota
state.quotas[effect.quota_type] -= 1
state.total_quota -= 1
```

---

## 三、Per-Unit 状态管理

### 3.1 UnitState

```python
@dataclass
class UnitState:
    """每个 unit 的可变状态"""
    chain: List[ChainEntry]       # 可用模型列表（会被修改）
    total_quota: int              # 总重试次数
    quotas: Dict[ErrorType, int]  # 分类型 quota

    # 依赖关系（统一的依赖树模型）
    depends_on: Set[str] = field(default_factory=set)  # context injection 依赖
    children: List[str] = None    # 分割产生的子 units
    is_virtual: bool = False      # 虚拟 = 不持久化（分割产生的）
    is_aggregation: bool = False  # 聚合 unit = 等待 children 完成后聚合
    aggregates_to: Optional[str] = None  # 聚合到哪个 parent
    content: str = ""             # unit 内容（虚拟 unit 需要）

    # 处理结果（用于 longest fallback）
    attempts: List[Tuple[str, int]] = field(default_factory=list)

    def can_retry(self, error_type: ErrorType) -> bool:
        """判断是否可以重试"""
        return (
            self.total_quota > 0 and
            self.quotas.get(error_type, 0) > 0 and
            len(self.chain) > 0
        )

    def apply_effect(self, effect: ErrorEffect, current_entry: ChainEntry):
        """应用错误效果"""
        if effect.remove_provider:
            # Safety block: 移除同 provider 的所有 entries
            self.chain = [e for e in self.chain if e.provider != current_entry.provider]
        elif effect.remove_current_model:
            # 只移除当前模型
            if self.chain and self.chain[0] == current_entry:
                self.chain.pop(0)

        if effect.remove_all_batch:
            # 移除所有 batch entries
            self.chain = [e for e in self.chain if e.mode != "batch"]

        self.quotas[effect.quota_type] = self.quotas.get(effect.quota_type, 0) - 1
        self.total_quota -= 1

    def record_attempt(self, result: str):
        """记录尝试"""
        self.attempts.append((result, len(result)))

    def get_longest(self) -> Optional[str]:
        """获取最长结果"""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda x: x[1])[0]

    def has_batch_available(self) -> bool:
        """是否还有 batch 模式可用"""
        return any(e.mode == "batch" for e in self.chain)

    def get_current_mode(self) -> Optional[str]:
        """获取当前模式"""
        return self.chain[0].mode if self.chain else None

    def get_current_entry(self) -> Optional[ChainEntry]:
        """获取当前 chain entry"""
        return self.chain[0] if self.chain else None
```

### 3.2 Quota 配置

```python
@dataclass
class QuotaConfig:
    """Quota 配置"""
    total: int = 5
    per_type: Dict[ErrorType, int] = field(default_factory=lambda: {
        ErrorType.SAFETY: 999,      # 不限制，只要 chain 没空
        ErrorType.NETWORK: 3,
        ErrorType.VALIDATION: 1,
        ErrorType.RATE_LIMIT: 3,
        ErrorType.UNKNOWN: 2,
    })

    def create_quotas(self) -> Dict[ErrorType, int]:
        return dict(self.per_type)
```

---

## 四、统一的依赖树模型

### 4.1 两种依赖

NestedPartProcessor 消失，动态分割变成依赖树的一部分：

| 依赖类型 | 来源 | 关系 |
|---------|------|------|
| **Context injection** | Sequential 模式 | part2 等待 part1 完成后获取 context |
| **聚合依赖** | 处理失败时分割 | parent 等待所有 children 完成后聚合 |

```
chapter_1.part1 (持久化)
  depends_on: {chapter_1.part0}  # context injection

chapter_1.part2 (持久化，处理失败，分割)
  depends_on: {chapter_1.part1}  # context injection
  children: [chapter_1.part2.sub0, chapter_1.part2.sub1]

  chapter_1.part2.sub0 (虚拟)
    aggregates_to: chapter_1.part2

  chapter_1.part2.sub1 (虚拟)
    aggregates_to: chapter_1.part2
```

### 4.2 Ready 检查

```python
def is_ready(unit_id: str, unit_states: Dict, completed: Set[str]) -> bool:
    """统一的 ready 检查：context 依赖 + children 依赖"""
    state = unit_states[unit_id]

    # 等待 context injection 依赖
    if state.depends_on and not state.depends_on.issubset(completed):
        return False

    # 等待 children 完成（聚合）
    if state.children and not all(c in completed for c in state.children):
        return False

    return True
```

### 4.3 分割时创建虚拟 Units

```python
def handle_split(unit_id: str, unit_states: Dict, pending: Set[str], splitter):
    """
    处理失败时分割，创建虚拟 children。

    关键：
    - parent 也入池，作为 aggregation unit 等待 children
    - children 继承一半 quota（向下取整），防止 quota 膨胀
    - 内容没有换行符时无法分割
    """
    state = unit_states[unit_id]

    # 无法分割的判断
    if '\n' not in state.content:
        return False  # 没有换行符，无法分割

    # 分割内容
    child_contents = splitter.split(state.content)
    if len(child_contents) <= 1:
        return False  # 分割器无法分割

    # 计算 children 的 quota（一半，向下取整）
    child_total_quota = state.total_quota // 2
    child_quotas_template = {k: v // 2 for k, v in state.quotas.items()}

    # 创建虚拟 children
    child_ids = []
    for i, content in enumerate(child_contents):
        child_id = f"{unit_id}.sub{i}"
        child_ids.append(child_id)

        unit_states[child_id] = UnitState(
            chain=list(state.chain),  # 继承 chain
            total_quota=child_total_quota,  # 一半 quota
            quotas=dict(child_quotas_template),  # 每个 child 独立的 dict！
            is_virtual=True,
            aggregates_to=unit_id,
            content=content
        )
        pending.add(child_id)

    # parent 变成 aggregation unit
    state.children = child_ids
    state.is_aggregation = True
    pending.add(unit_id)  # parent 也入池（is_ready 会等 children）

    return True
```

**Quota 继承效果**：
```
unit_A (quota=5) 失败 → 分割
  ├── unit_A.sub0 (quota=2)
  └── unit_A.sub1 (quota=2)

unit_A.sub0 (quota=2) 失败 → 分割
  ├── unit_A.sub0.sub0 (quota=1)
  └── unit_A.sub0.sub1 (quota=1)

unit_A.sub0.sub0 (quota=1) 失败 → 分割
  ├── unit_A.sub0.sub0.sub0 (quota=0) → 无法重试，直接失败
  └── unit_A.sub0.sub0.sub1 (quota=0) → 无法重试，直接失败
```

quota 自然衰减，无需共享引用。

### 4.4 Aggregation 在 Executor 主循环中处理

Aggregation 不作为 hook，因为它需要 `unit_states` 和 `results` 参数，而 PreProcessor Protocol 不支持。直接在 Executor 主循环中处理：

```python
# 在 Executor._get_ready_ids 返回 ready units 后，提交前检查
for unit_id in ready_ids:
    state = unit_states[unit_id]

    if state.is_aggregation:
        # 直接聚合，不提交给线程池
        child_results = [results[c] for c in sorted(state.children)]
        aggregated = "\n\n".join(child_results)
        results[unit_id] = aggregated
        completed.add(unit_id)
        pending.discard(unit_id)
        continue

    # 正常提交给线程池
    ...
```

**流程**：

```
1. unit_A 处理失败，触发 handle_split
2. 创建 unit_A.sub0, unit_A.sub1 入池（quota=2）
3. unit_A 标记为 is_aggregation=True，也入池
4. 主循环检查 is_ready：
   - unit_A.sub0: ready
   - unit_A.sub1: ready
   - unit_A: NOT ready（等待 children）
5. unit_A.sub0, unit_A.sub1 处理完成
6. 主循环再次检查 is_ready：
   - unit_A: ready（children 都完成了）
7. unit_A 检测到 is_aggregation，直接聚合，不调用 LLM
```

### 4.5 递归分割

**虚拟 unit 也可以失败并再次分割**：

```
chapter_1.part0 失败
  ├── 分割成 chapter_1.part0.sub0, chapter_1.part0.sub1
  └── chapter_1.part0 变成 aggregation unit

chapter_1.part0.sub0 也失败（内容还是太大）
  ├── 分割成 chapter_1.part0.sub0.sub0, chapter_1.part0.sub0.sub1
  └── chapter_1.part0.sub0 变成 aggregation unit（虚拟的 aggregation unit）

最终依赖树：
chapter_1.part0 (aggregation)
  ├── chapter_1.part0.sub0 (aggregation, virtual)
  │     ├── chapter_1.part0.sub0.sub0 (virtual)
  │     └── chapter_1.part0.sub0.sub1 (virtual)
  └── chapter_1.part0.sub1 (virtual)

聚合顺序（自底向上）：
1. chapter_1.part0.sub0.sub0, chapter_1.part0.sub0.sub1 完成
2. chapter_1.part0.sub0 聚合 → 完成
3. chapter_1.part0.sub1 完成
4. chapter_1.part0 聚合 → 完成
```

**代码天然支持递归**：
- `child_id = f"{unit_id}.sub{i}"` 生成任意深度的 ID
- 任何 unit（包括虚拟 unit）都可以设置 `is_aggregation = True`
- `is_ready` 检查 children 完成，自底向上自然触发
- Executor 主循环处理 aggregation，不区分虚拟/非虚拟

**命名规则**：
| 后缀 | 含义 | 来源 | 持久化 |
|------|------|------|--------|
| `.part{N}` | 持久化分割 | proactive split（Pipeline 层） | 是 |
| `.sub{N}` | 虚拟分割 | 动态 split（Executor 层，处理失败时） | 否 |

示例：`chapter_1.part0.sub0.sub1` = 持久化的 part0，运行时分割两次

---

## 五、Executor 并发模型

### 5.1 核心设计

**关键思想**：
- 没有 retry for loop，失败后更新状态重新入池
- Batch 和 Online **同时进行**，不是 if-else
- 统一的 Executor，根据 chain entry 的 mode 决定行为

**终止条件**：
```
pending 空 && futures 空 && batch_job 空 → 终止
```

终止时，剩余 pending 全部标记失败：
- quota 耗尽无法重试的
- 子女全部失败导致无法聚合的
- 依赖死锁的

### 5.2 Batch + Online 同时执行

**不是 if-else，而是并行**：

```
                     ┌──────────────────┐
                     │   所有 Units     │
                     └────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Batch Mode      │             │ Online Mode     │
    │ (chain[0].mode  │             │ (chain[i].mode  │
    │  == "batch")    │             │  == "online")   │
    └────────┬────────┘             └────────┬────────┘
             │                               │
    ┌────────▼────────┐             ┌────────▼────────┐
    │ 提交 Batch Job  │             │ 立即进入线程池  │
    │ (异步等待结果)  │             │ (并发执行)      │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ 合并结果        │
                    │ 统一失败处理    │
                    └─────────────────┘
```

**执行流程**：

```python
def execute(self, units: List[WorkUnit]) -> ExecutionResult:
    # 根据 chain 分流
    batch_entries = [e for e in self._model_chain if e.mode == "batch"]
    online_entries = [e for e in self._model_chain if e.mode == "online"]

    # 同时启动
    batch_future = None
    if batch_entries:
        # 提交 batch job（异步，返回 future）
        batch_future = self._submit_batch_async(units, batch_entries[0])

    # Online 立即并发处理
    # 注意：每个 unit 的 state.chain 只保留 online entries
    for unit in units:
        state = unit_states[unit.id]
        if state.has_batch_available():
            # 有 batch，先等 batch 结果
            batch_pending.add(unit.id)
        else:
            # 没有 batch，直接入 online 池
            online_pending.add(unit.id)

    # 并发执行 online
    # 同时轮询 batch 结果
    while batch_pending or online_pending or futures:
        # 检查 batch 是否完成
        if batch_future and batch_future.done():
            batch_results = batch_future.result()
            for unit_id, result in batch_results.items():
                # 处理结果...
                batch_pending.remove(unit_id)

        # 处理 online 队列
        # ...
```

**失败后的状态更新 + 重新入池**：

```python
# Safety 失败 → 移除同 provider 的所有 entries（batch + online）
if error_type == ErrorType.SAFETY:
    state.chain = remove_provider(state.chain, current_entry.provider)

# Batch 少量失败 → 移除所有 batch entries
if len(batch_failed) <= threshold:
    for unit_id in batch_failed:
        unit_states[unit_id].chain = remove_batch_entries(unit_states[unit_id].chain)

# 重新入池（主循环自然处理，chain[0] 已经变成 online entry）
for unit_id in batch_failed:
    pending.add(unit_id)
```

### 5.3 Executor 实现

```python
class Executor:
    def __init__(
        self,
        llm_client: Any,
        batch_client: Optional[Any],  # None = online only
        model_chain: List[ChainEntry],
        processor: ProcessorProtocol,
        hooks: CompositeHooks,
        quota_config: QuotaConfig = None,
        max_workers: int = 4,
        context_injector: Optional[ContextInjector] = None,
        tracker: Optional[ProcessingTracker] = None,
        splitter: Optional[ContentSplitter] = None,  # 用于动态分割
    ):
        self._llm_client = llm_client
        self._batch_client = batch_client
        self._model_chain = model_chain
        self._processor = processor
        self._hooks = hooks
        self._quota_config = quota_config or QuotaConfig()
        self._max_workers = max_workers
        self._context_injector = context_injector
        self._tracker = tracker
        self._splitter = splitter

    def execute(
        self,
        units: List[WorkUnit],
        context_base: Optional[ProcessContext] = None
    ) -> ExecutionResult:
        """
        并发执行所有 units。

        - 依赖满足后入池
        - 失败后更新状态重新入池
        - 没有 retry for loop
        """
        # 初始化状态
        unit_states: Dict[str, UnitState] = {
            u.id: UnitState(
                chain=list(self._model_chain),
                total_quota=self._quota_config.total,
                quotas=self._quota_config.create_quotas(),
                content=u.content  # handle_split 需要
            )
            for u in units
        }
        originals: Dict[str, str] = {u.id: u.content for u in units}
        unit_map: Dict[str, WorkUnit] = {u.id: u for u in units}

        # 结果收集
        results: Dict[str, str] = {}
        completed: Set[str] = set()
        failed: Set[str] = set()
        skipped: Set[str] = set()

        # 并发状态
        pending: Set[str] = {u.id for u in units}
        in_progress: Set[str] = set()
        futures: Dict[Future, str] = {}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending or futures:
                # 获取 ready 的 units（依赖满足）
                ready_ids = self._get_ready_ids(pending, completed, in_progress, unit_states)

                # 处理 ready units
                for unit_id in ready_ids:
                    pending.discard(unit_id)
                    state = unit_states[unit_id]

                    # Aggregation unit: 直接聚合，不调用 LLM
                    if state.is_aggregation:
                        child_results = [results[c] for c in sorted(state.children)]
                        results[unit_id] = "\n\n".join(child_results)
                        completed.add(unit_id)
                        continue

                    # 正常 unit: 提交到线程池
                    in_progress.add(unit_id)
                    unit = unit_map.get(unit_id) or WorkUnit(id=unit_id, content=state.content)
                    context = self._build_context(unit, context_base, completed, results, originals)

                    future = pool.submit(
                        self._process_single,
                        unit, state, context, originals.get(unit_id, state.content)
                    )
                    futures[future] = unit_id

                if not futures:
                    # 终止条件：没有 in-progress，没有 ready
                    # 剩余 pending 全部标记失败（quota 耗尽 / 子女失败 / 依赖死锁）
                    if pending:
                        failed.update(pending)
                        pending.clear()
                    break

                # 等待任意一个完成
                done, _ = wait(futures, return_when=FIRST_COMPLETED)

                for future in done:
                    unit_id = futures.pop(future)
                    in_progress.remove(unit_id)

                    try:
                        result = future.result()
                        self._handle_result(
                            unit_id, result, unit_states,
                            pending, completed, failed, skipped,
                            results, originals
                        )
                    except Exception as e:
                        logger.error(f"{unit_id}: Unexpected error: {e}")
                        failed.add(unit_id)

        return ExecutionResult(
            results=results,
            completed=completed,
            failed=failed,
            skipped=skipped,
        )
```

### 5.4 单 Unit 处理

```python
@dataclass
class ProcessResult:
    """单次处理结果"""
    success: bool
    content: Optional[str] = None
    error_type: Optional[ErrorType] = None
    skipped: bool = False
    skip_reason: str = ""


def _process_single(
    self,
    unit: WorkUnit,
    state: UnitState,
    context: ProcessContext,
    original: str
) -> ProcessResult:
    """处理单个 unit（在线程池中执行）"""

    # Pre-processing
    pre_result = self._hooks.pre_process(unit.id, unit.content, context)
    if not pre_result.should_process:
        return ProcessResult(
            success=True,
            content=pre_result.fallback_result,
            skipped=True,
            skip_reason=pre_result.skip_reason
        )

    # 获取当前模型
    if not state.chain:
        return ProcessResult(success=False, error_type=ErrorType.SAFETY)

    model = state.chain[0]

    try:
        # 调用 LLM
        prompt = self._processor.build_prompt(unit.content, context)
        response = self._llm_client.generate(
            prompt=prompt,
            model_configs=[model.to_dict()],
            operation_name=f"{self._processor.name}:{unit.id}"
        )
        cleaned = self._processor.clean_response(response)
        final = self._processor.post_process(cleaned, context)

        # Post-processing (transform + validate)
        chapter_type = context.chapter_type or ""
        transformed, hook_result = self._hooks.post_process(
            unit.id, original, final, chapter_type, context
        )

        # 记录尝试
        state.record_attempt(transformed)

        if hook_result.accepted:
            return ProcessResult(success=True, content=transformed)
        else:
            return ProcessResult(success=False, error_type=ErrorType.VALIDATION)

    except Exception as e:
        error_type, _ = self._hooks.classify_error(e)
        return ProcessResult(success=False, error_type=error_type)
```

### 5.5 结果处理（状态更新 + 重新入池）

```python
def _handle_result(
    self,
    unit_id: str,
    result: ProcessResult,
    unit_states: Dict[str, UnitState],
    pending: Set[str],
    completed: Set[str],
    failed: Set[str],
    skipped: Set[str],
    results: Dict[str, str],
    originals: Dict[str, str]
):
    """处理结果：成功则完成，失败则更新状态并判断是否重新入池"""
    state = unit_states[unit_id]

    if result.skipped:
        skipped.add(unit_id)
        results[unit_id] = result.content
        completed.add(unit_id)
        logger.info(f"{unit_id}: Skipped - {result.skip_reason}")
        return

    if result.success:
        results[unit_id] = result.content
        completed.add(unit_id)

        # 缓存用于 context injection
        if self._context_injector:
            self._context_injector.cache_completed(
                unit_id, originals[unit_id], result.content
            )

        logger.info(f"{unit_id}: Completed successfully")
        return

    # 失败：获取错误效果并应用
    error_type = result.error_type
    effect = self._hooks.get_error_effect(error_type)
    current_entry = state.get_current_entry()
    state.apply_effect(effect, current_entry)

    # 判断是否可以重试
    if state.can_retry(error_type):
        pending.add(unit_id)  # 重新入池！
        logger.info(f"{unit_id}: {error_type.value}, re-queued (quota: {state.total_quota})")
    else:
        # 无法重试，尝试分割（需要 splitter 存在）
        if self._splitter and handle_split(unit_id, unit_states, pending, self._splitter):
            logger.info(f"{unit_id}: Split into {len(state.children)} children")
            return  # 分割成功，等待 children 完成后聚合

        # 无法分割，尝试 longest fallback
        longest = state.get_longest()
        if longest:
            results[unit_id] = longest
            completed.add(unit_id)
            logger.warning(f"{unit_id}: Using longest fallback ({len(longest)} chars)")
        else:
            failed.add(unit_id)
            logger.error(f"{unit_id}: Failed, no fallback available")
```

### 5.6 依赖顺序处理

```python
def _get_ready_ids(
    self,
    pending: Set[str],
    completed: Set[str],
    in_progress: Set[str],
    unit_states: Dict[str, UnitState]
) -> List[str]:
    """获取依赖满足的 unit IDs"""
    ready = []
    is_sequential = self._context_injector and self._context_injector.is_sequential

    for unit_id in pending:
        if unit_id in in_progress:
            continue

        state = unit_states.get(unit_id)
        if state:
            # 检查 children（aggregation unit 等待 children）
            # 注意：这个检查是无条件的，parallel/sequential 都需要！
            if state.children and not all(c in completed for c in state.children):
                continue

            # Sequential 模式：检查 depends_on（context injection 依赖）
            if is_sequential:
                if state.depends_on and not state.depends_on.issubset(completed):
                    continue

        # Sequential 模式：检查 part 顺序依赖（chapter_5.part1 依赖 chapter_5.part0）
        if is_sequential and '.part' in unit_id:
            parts = unit_id.rsplit('.part', 1)
            if len(parts) == 2:
                base, part_num_str = parts
                try:
                    part_num = int(part_num_str)
                    if part_num > 0:  # part1 依赖 part0
                        prev_id = f"{base}.part{part_num - 1}"
                        if prev_id not in completed and prev_id in pending | in_progress:
                            continue
                except ValueError:
                    pass

        ready.append(unit_id)

    return ready


def _build_context(
    self,
    unit: WorkUnit,
    context_base: Optional[ProcessContext],
    completed: Set[str],
    results: Dict[str, str],
    originals: Dict[str, str]
) -> ProcessContext:
    """构建上下文，包含前置 part 的结果"""
    context = context_base or ProcessContext()

    if not self._context_injector or not self._context_injector.is_sequential:
        return context

    # 获取前置 part 的结果
    prev_context = self._context_injector.get_context_for_unit(
        unit, results, originals
    )

    if prev_context:
        prev_original, prev_processed = prev_context
        context = self._context_injector.inject_context(
            context, prev_original, prev_processed
        )

    return context
```

---

## 六、Pipeline

### 6.1 职责

```
Pipeline 职责：
├── Proactive Split (SplitManager)
├── Resume Tracking (ProcessingTracker)
├── 调用 Executor（所有重试由 Executor 内部处理）
├── Batch Validation（所有文件完成后）
└── Save Results (Persistence)

注意：
- 没有 retry 逻辑（Executor 通过状态更新 + 重新入池处理）
- 没有 longest fallback（Executor 内部处理）
- 没有 aggregate parts（只在 build-epub 时聚合）
```

### 6.2 实现

```python
class ProcessingPipeline:
    def __init__(
        self,
        processor: ProcessorProtocol,
        executor: Executor,
        persistence: ResultPersistence,
        tracker: ProcessingTracker,
        hooks: CompositeHooks,
        # Batch validation
        batch_validators: List[BatchValidator] = None,
        # 组件
        split_manager: Optional[SplitManager] = None,
        context_injector: Optional[ContextInjector] = None,
        book_structure: Optional[BookStructure] = None,
    ):
        self._processor = processor
        self._executor = executor
        self._persistence = persistence
        self._tracker = tracker
        self._hooks = hooks
        self._batch_validators = batch_validators or []
        self._split_manager = split_manager
        self._context_injector = context_injector
        self._book_structure = book_structure

    def process_all(
        self,
        units: List[WorkUnit],
        context_base: Optional[ProcessContext] = None
    ) -> ProcessingResult:
        """处理所有 units"""
        if not units:
            return ProcessingResult(total=0, completed=0, failed=0)

        start_time = time.time()

        # Step 1: Proactive split
        units = self._proactive_split(units)

        all_keys = {u.id for u in units}
        originals = {u.id: u.content for u in units}

        # Step 2: Get pending keys (resume)
        pending_keys = self._get_pending_keys(all_keys)
        pending_units = [u for u in units if u.id in pending_keys]

        if not pending_units:
            logger.info(f"All {len(units)} units already completed")
            return ProcessingResult(total=len(units), completed=len(units), failed=0)

        # Step 3: Execute via Executor
        # Executor 内部处理所有重试（状态更新 + 重新入池），Pipeline 不做任何 retry
        exec_result = self._executor.execute(pending_units, context_base)

        # Step 4: Save results
        for key, content in exec_result.results.items():
            self._persistence.save_raw(key, content)

        # Step 5: Batch validation (if configured)
        # Batch validation 失败 = 最终失败（Pipeline 不做 retry）
        batch_failed: Set[str] = set()
        if self._batch_validators:
            batch_failed = self._run_batch_validation(exec_result.results, originals)

        # Step 6: Determine final failures
        all_failed = exec_result.failed | batch_failed

        # Step 7: Promote successful to validated
        successful = set(exec_result.results.keys()) - all_failed
        self._persistence.promote_batch(list(successful))
        for key in successful:
            self._mark_complete(key)

        # Result
        completed = len(units) - len(all_failed)
        duration = time.time() - start_time

        return ProcessingResult(
            total=len(units),
            completed=completed,
            failed=len(all_failed),
            failed_keys=list(all_failed),
            duration=duration
        )

    def _run_batch_validation(
        self,
        results: Dict[str, str],
        originals: Dict[str, str]
    ) -> Set[str]:
        """运行 batch validation，返回失败的 keys"""
        failed: Set[str] = set()

        # 获取 skip keys
        skip_keys: Set[str] = set()
        if self._book_structure:
            for key in results:
                info = self._book_structure.get_chapter_info(key)
                if not info:
                    continue
                chapter_type = info.chapter_type or ""
                for sv in self._hooks._skip_validators:
                    if sv.should_skip(key, chapter_type, None):
                        skip_keys.add(key)
                        break

        # 创建 verification files
        files = create_verification_files(results, originals)

        # 运行每个 batch validator
        for validator in self._batch_validators:
            batch_result = validator.validate_batch(files, skip_keys)

            # 记录
            for record in batch_result.records:
                self._tracker.record_validation(record.file_key, record.to_dict())

            failed.update(batch_result.failed)

        return failed
```

---

## 七、Hooks 执行流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ Pre-processing (hooks.pre_process)                               │
│                                                                  │
│   ImageOnlyFilter → 纯图片？→ skip, return original              │
│   EmptyContentFilter → 空内容？→ skip                            │
│                                                                  │
│   如果 should_process=False → 直接返回 fallback_result           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Processing (Executor 核心)                                       │
│                                                                  │
│   model = state.chain[0]                                        │
│   processor.build_prompt(content, context)                      │
│     → llm_client.generate(model)                                │
│   processor.clean_response(response)                            │
│   processor.post_process(cleaned, context)                      │
│                                                                  │
│   如果出错 → hooks.classify_error(error)                        │
│     → ErrorEffect(remove_current_model, quota_type)             │
│     → state.apply_effect(effect)                                │
│     → state.can_retry(error_type) ? pending.add(unit) : fail    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Post-processing (hooks.post_process)                             │
│                                                                  │
│   1. Transform (链式):                                          │
│      → RestoreImagesTransformer                                 │
│      → RemoveArtifactsTransformer                               │
│                                                                  │
│   2. Skip validation check:                                     │
│      → ChapterTypeSkipper: front_matter 等跳过                  │
│                                                                  │
│   3. Validate (screener/final):                                 │
│      → LengthValidator                                          │
│      → NGramValidator                                           │
│                                                                  │
│   如果 accepted=False → ErrorType.VALIDATION                    │
│     → state.apply_effect(...)                                   │
│     → state.can_retry(...) ? pending.add(unit) : fail           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、配置示例

```python
# Quota 配置
quota_config = QuotaConfig(
    total=5,
    per_type={
        ErrorType.SAFETY: 999,      # 只要 chain 没空就继续
        ErrorType.NETWORK: 3,
        ErrorType.VALIDATION: 1,
        ErrorType.RATE_LIMIT: 3,
    }
)

# Model chain
model_chain = [
    ChainEntry(provider="gemini", model="gemini-2.0-flash", mode="batch"),
    ChainEntry(provider="gemini", model="gemini-2.0-flash", mode="online"),
    ChainEntry(provider="deepseek", model="deepseek-chat", mode="online"),
    ChainEntry(provider="anthropic", model="claude-3-haiku", mode="online"),
]

# Hooks
hooks = CompositeHooks(
    pre_processors=[
        EmptyContentFilter(),
        ImageOnlyFilter(book_structure),
    ],
    transformers=[
        RestoreImagesTransformer(),
        RemoveArtifactsTransformer(),
    ],
    validators=[
        IndividualValidatorAdapter(LengthValidator(), role="screener", context_ready=True),
        IndividualValidatorAdapter(NGramValidator(), role="screener", context_ready=True),
    ],
    skip_validators=[
        ChapterTypeSkipper(),
    ],
    error_classifier=DefaultErrorClassifier(),
    tracker=tracker,
)

# Executor（统一处理 batch + online）
executor = Executor(
    llm_client=llm_client,
    batch_client=batch_client,  # 可选，None 则只用 online
    model_chain=model_chain,
    processor=processor,
    hooks=hooks,
    quota_config=quota_config,
    max_workers=4,
    context_injector=context_injector,
    tracker=tracker,
    splitter=MarkdownStructureSplitter(),  # 可选，用于动态分割
)

# Pipeline
pipeline = ProcessingPipeline(
    processor=processor,
    executor=executor,
    persistence=persistence,
    tracker=tracker,
    hooks=hooks,
    batch_validators=[TranslationBatchValidator()],
    split_manager=split_manager,
    context_injector=context_injector,
    book_structure=book_structure,
)

# 执行
result = pipeline.process_all(units, context_base)
```

---

## 九、文件结构

```
pdf2epub/core/
├── executor/
│   ├── __init__.py
│   ├── _protocol.py          # WorkUnit, ChainEntry, ExecutionResult, ProcessResult
│   ├── state.py              # UnitState, QuotaConfig
│   └── executor.py           # Executor（统一处理 batch + online）
│
├── hooks/
│   ├── __init__.py
│   ├── _protocol.py          # 所有 Hook Protocols
│   ├── composite.py          # CompositeHooks
│   ├── pre_processors.py     # ImageOnlyFilter, EmptyContentFilter
│   ├── transformers.py       # RestoreImages, RemoveArtifacts
│   ├── validators.py         # Adapters for IndividualValidator
│   ├── skip_validators.py    # ChapterTypeSkipper
│   └── error_classifier.py   # DefaultErrorClassifier
│
├── phase/
│   ├── __init__.py
│   ├── _protocol.py          # WorkUnit, WorkUnitLoader, Phase Protocol
│   ├── loader.py             # PartBasedLoader
│   ├── phase.py              # Phase 实现
│   └── registry.py           # 阶段注册（polish, translate 等）
│
├── pipeline.py               # ProcessingPipeline (重写)
├── factory.py                # 工厂函数 (更新)
└── ...
```

---

## 十、Phase 通用接口

### 10.1 设计原则

**任意组合**：阶段之间可以任意衔接。

```
polish → translate    ✓ 常见流程
translate → polish    ✓ 翻译后再润色
polish → polish       ✓ 多轮润色
translate → translate ✓ 双重翻译（中间语言）
```

**不做中间聚合**：

```
旧设计（错误）:
  polish:
    chapter_1.part0.md → polished
    chapter_1.part1.md → polished
    ↓
    聚合为 chapter_1.md  ← 这是错的！
    ↓
  translate:
    chapter_1.md → translated

新设计（正确）:
  polish:
    chapter_1.part0.md → polished
    chapter_1.part1.md → polished
    （不聚合）
    ↓
  translate:
    chapter_1.part0.md → translated
    chapter_1.part1.md → translated
    （不聚合）
    ↓
  build-epub:
    聚合所有 parts → chapter_1.xhtml
```

**聚合只发生在 build-epub**：每个阶段只处理最小单位（parts），最终 build-epub 负责聚合。

### 10.2 通用 WorkUnit 接口

每个阶段都可以从任何目录读取 work units：

```python
@dataclass
class WorkUnit:
    id: str           # e.g., "chapter_1.part0"
    content: str      # 文件内容
    source_path: Path # 原始文件路径
    metadata: Dict    # 额外信息（chapter_type 等）


class WorkUnitLoader(Protocol):
    """通用的 work unit 加载器"""

    def load_units(
        self,
        input_dir: Path,
        pattern: str = "*.md"
    ) -> List[WorkUnit]: ...


class PartBasedLoader:
    """
    基于 part 文件的加载器。

    核心逻辑：
    1. 发现所有文件
    2. 跳过已有 parts 的父文件（如果有 chapter_1.part0.md，跳过 chapter_1.md）
    3. 返回最小粒度的 work units
    """

    def load_units(self, input_dir: Path, pattern: str = "*.md") -> List[WorkUnit]:
        all_files = sorted(input_dir.glob(pattern))

        # 收集所有已有 parts 的 base keys
        # chapter_1.part0.md → base = "chapter_1"
        bases_with_parts: Set[str] = set()
        for path in all_files:
            stem = path.stem
            if '.part' in stem:
                base = stem.rsplit('.part', 1)[0]
                bases_with_parts.add(base)

        units = []
        for path in all_files:
            stem = path.stem

            # 跳过已有 parts 的父文件
            # 如果存在 chapter_1.part0.md，跳过 chapter_1.md
            if stem in bases_with_parts:
                continue

            content = path.read_text()
            units.append(WorkUnit(
                id=stem,
                content=content,
                source_path=path,
                metadata={}
            ))

        return units
```

**Split 在 Pipeline 层处理**：

```python
# Pipeline.process_all 的第一步
def _proactive_split(self, units: List[WorkUnit]) -> List[WorkUnit]:
    """对太大的 units 进行分割"""
    if not self._split_manager:
        return units

    result = []
    for unit in units:
        if self._split_manager.needs_split(unit.content):
            # 分割成多个 sub units
            parts = self._split_manager.split(unit.id, unit.content)
            for i, part_content in enumerate(parts):
                result.append(WorkUnit(
                    id=f"{unit.id}.part{i}",
                    content=part_content,
                    source_path=unit.source_path,
                    metadata={**unit.metadata, 'parent_id': unit.id}
                ))
        else:
            result.append(unit)
    return result
```

### 10.3 阶段衔接

```python
class Phase:
    """通用阶段"""

    def __init__(
        self,
        name: str,
        input_dir: Path,
        output_dir: Path,
        processor: ProcessorProtocol,
        pipeline: ProcessingPipeline,
        loader: WorkUnitLoader = None,
    ):
        self._name = name
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._processor = processor
        self._pipeline = pipeline
        self._loader = loader or PartBasedLoader()

    def run(self, resume: bool = False) -> PhaseResult:
        """执行阶段"""
        # 1. 加载 work units（自动跳过已有 parts 的父文件）
        units = self._loader.load_units(self._input_dir)

        # 2. 执行（Pipeline 内部处理 proactive split）
        result = self._pipeline.process_all(units)

        # 3. 保存结果（不聚合，保持最小粒度）
        for unit_id, content in result.results.items():
            out_path = self._output_dir / f"{unit_id}.md"
            out_path.write_text(content)

        return PhaseResult(
            phase=self._name,
            total=result.total,
            completed=result.completed,
            failed=result.failed
        )
```

### 10.4 阶段串联

```python
# 使用示例：polish → translate

# Phase 1: Polish
polish_phase = Phase(
    name="polish",
    input_dir=output_dir / "pages_merged",    # OCR 结果
    output_dir=output_dir / "polished",
    processor=PolishProcessor(),
    pipeline=polish_pipeline,
)
polish_phase.run(resume=True)

# Phase 2: Translate（接收 polish 的输出）
translate_phase = Phase(
    name="translate",
    input_dir=output_dir / "polished",        # 上一阶段的输出
    output_dir=output_dir / "translated",
    processor=TranslateProcessor(),
    pipeline=translate_pipeline,
)
translate_phase.run(resume=True)

# Phase 3: 可选再 polish
polish_again = Phase(
    name="polish_final",
    input_dir=output_dir / "translated",      # 翻译结果
    output_dir=output_dir / "polished_final",
    processor=PolishProcessor(),
    pipeline=polish_pipeline,
)
polish_again.run(resume=True)

# Final: Build EPUB（唯一的聚合点）
build_epub(
    input_dir=output_dir / "polished_final",  # 或 translated
    output_path=output_dir / "book.epub",
    aggregate=True  # 在这里聚合 parts
)
```

### 10.5 WorkUnit 发现与分割流程

**完整流程**：

```
1. PartBasedLoader.load_units(input_dir)
   ├── 扫描所有 *.md 文件
   ├── 跳过已有 parts 的父文件（如果有 chapter_1.part0.md，跳过 chapter_1.md）
   └── 返回最小粒度的 work units

2. Pipeline._proactive_split(units)
   ├── 检查每个 unit 是否太大
   ├── 太大的进行 split（chapter_1 → chapter_1.part0, chapter_1.part1）
   └── 返回分割后的 units

3. Executor.execute(units)
   ├── 处理所有 units
   └── 失败的通过状态更新 + 重新入池

4. 保存结果（保持原有粒度，不聚合）
```

**跳过逻辑示例**：

```
输入目录包含:
  chapter_1.md           ← 已有 parts，跳过！
  chapter_1.part0.md     ← 读取
  chapter_1.part1.md     ← 读取（太大，需要 split）
  chapter_2.md           ← 没有 parts，读取

加载后:
  chapter_1.part0 (3000 tokens)
  chapter_1.part1 (8000 tokens) ← 需要 proactive split
  chapter_2 (2000 tokens)

Proactive Split 后:
  chapter_1.part0
  chapter_1.part1.part0 (4000 tokens)
  chapter_1.part1.part1 (4000 tokens)
  chapter_2

处理后输出（不聚合）:
  chapter_1.part0.md
  chapter_1.part1.part0.md
  chapter_1.part1.part1.md
  chapter_2.md
```

| 操作 | 时机 | 说明 |
|------|------|------|
| **跳过已有 parts** | 加载时 | PartBasedLoader 跳过已分割的父文件 |
| **Proactive Split** | 加载后 | Pipeline 对太大的 unit 分割 |
| **No Aggregation** | 阶段之间 | 保持最小单位，下一阶段同样加载 parts |
| **Final Aggregation** | build-epub | 唯一的聚合点 |

下一阶段同样处理这些 parts。

build-epub:
  聚合：chapter_1.part0 + chapter_1.part1.part0 + chapter_1.part1.part1
        → chapter_1.xhtml
```

注意：如果运行时有动态分割（`.sub`），虚拟 units 会在 Executor 内部聚合：
- `.sub` 会保存到 `raw/`（用于调试）
- `.sub` 不会 promote 到 `validated/`（不是下一阶段的输入）
- `.sub` 不计入 completed/failed 统计

---

## 十一、Validator 适配器

### 11.1 IndividualValidatorAdapter

将现有的 IndividualValidator 适配为 hooks.Validator：

```python
class IndividualValidatorAdapter:
    """将 IndividualValidator 适配为 hooks.Validator"""

    def __init__(
        self,
        validator: IndividualValidator,
        role: Literal["screener", "final"],
        context_ready: bool = False
    ):
        self._validator = validator
        self._role = role
        self._context_ready = context_ready

    @property
    def name(self) -> str:
        return self._validator.name

    def validate(self, key, original, result):
        validation_result = self._validator.validate(original, result, key)

        # Screener: 通过 = accepted + context_ready，不通过 = 继续
        # Final: 通过 = accepted，不通过 = rejected
        if self._role == "screener":
            if validation_result.is_valid:
                return HookResult(accepted=True, context_ready=self._context_ready)
            else:
                # Screener 不通过 = 不确定，继续后续 validator
                return HookResult(accepted=True, context_ready=False)
        else:  # final
            return HookResult(
                accepted=validation_result.is_valid,
                context_ready=self._context_ready if validation_result.is_valid else False
            )
```

### 11.2 ExecutionResult 详细结构

```python
@dataclass
class ExecutionResult:
    """执行结果"""
    results: Dict[str, str]           # key -> processed content
    completed: Set[str]               # 成功完成的 keys
    failed: Set[str]                  # 处理失败的 keys
    skipped: Set[str]                 # 跳过的 keys（image-only, empty 等）
    safety_blocked: Set[str]          # 被 safety filter 阻止的 keys
    validation_failed: Set[str]       # 验证失败的 keys
    stats: Dict[str, Any]             # 执行统计
        # duration_seconds: float
        # batch_jobs_submitted: int
        # total_requeued: int  # 重新入池次数（不是 retry）
```

---

## 十二、已确认的设计决策

### 12.1 Context Injection 在 Batch 模式下

**决策**：忽略。Batch 服务端并行执行，无法注入 context。不需要警告。

### 12.2 Individual Validation 在 Batch 模式下

**决策**：支持。Batch 完成后批量调用 individual validators。

区别在于：
- Online: 每文件处理后立即验证，失败则状态更新 + 重新入池
- Batch: batch 完成后批量验证，失败的 units 移除 batch entries 后重新入池

这样 ngram（individual screener）+ agent（batch final）组合可以正常工作。

### 12.3 NestedPartProcessor 与 Batch

**决策**：
- NestedPartProcessor 被虚拟 unit 依赖树替代
- Batch 模式使用 proactive splitting
- 处理失败时创建虚拟 children，聚合后继续

### 12.4 状态持久化

**决策**：
- 每个 unit 维护自己的 UnitState（chain, quotas）
- Pipeline 层使用 ProcessingTracker 记录完成状态
- Batch job 状态是 Executor 内部细节

### 12.5 Batch + Online 同时执行

**决策**：不是 if-else，而是同时执行：
- Batch job 提交后异步等待
- Online 任务立即入池并发执行
- 结果合并后统一处理失败

---

## 十三、职责划分

### 13.1 Pipeline 职责

| 类型 | 处理方式 |
|------|----------|
| **Proactive Split** | 处理前分割太大的文件 |
| **Resume Tracking** | ProcessingTracker 记录进度 |
| **调用 Executor** | 传入 units（Executor 内部处理所有重试） |
| **Save Results** | 保存到 raw/ 目录 |
| **Batch Validation** | 所有文件完成后批量验证 |

**Pipeline 不做**：
- 任何 retry 逻辑（Executor 通过重新入池处理）
- Longest fallback（Executor 内部处理）
- Aggregate parts（只在 build-epub 时聚合）

### 13.2 Executor 职责

| 职责 | 说明 |
|------|------|
| **调用 LLM** | processor.build_prompt → llm_client → processor.clean_response |
| **Per-Unit 状态管理** | chain, quotas, 依赖 |
| **Hooks 调用** | pre_process, post_process, classify_error |
| **失败重新入池** | 无 retry loop，状态更新后重新入池 |
| **Batch + Online 同时** | 异步 batch job + 并发 online 线程池 |
| **Longest Fallback** | 无法继续时使用最长结果 |

### 13.3 Hooks 职责

| 职责 | 说明 |
|------|------|
| **PreProcess** | 判断是否需要处理（image-only, empty 跳过） |
| **Transform** | restore_images, remove_artifacts 等后处理 |
| **Validate** | length check, ngram check 等验证 |
| **SkipValidation** | front_matter 等跳过验证 |
| **ErrorClassify** | 错误分类 + 返回 ErrorEffect |

---

## 十四、迁移策略

### Phase 1: 创建 Hooks 模块

1. 创建 `hooks/` 目录结构
2. 定义所有 Protocol
3. 实现内置 hooks

### Phase 2: 创建 Executor 模块

1. 创建 `executor/` 目录结构
2. 实现 UnitState, ChainEntry, QuotaConfig
3. 实现统一的 Executor（batch + online 同时处理，状态更新 + 重新入池）

### Phase 3: 重写 Pipeline

1. 修改 ProcessingPipeline 接收 Executor
2. 移除直接 LLM 调用代码
3. 移除 NestedPartProcessor（用虚拟 unit 替代）

### Phase 4: 创建 Phase 模块

1. 定义 WorkUnit, WorkUnitLoader
2. 实现通用 Phase 类
3. 删除旧的中间聚合逻辑

### Phase 5: 更新 CLI

1. 更新命令使用新 Phase 接口
2. 删除 batch_pipeline.py

### Phase 6: 测试

1. 验证 online 模式功能不变
2. 验证 batch 模式功能不变
3. 验证 batch + online 同时执行
4. 验证 phase 串联

---

## 十五、已解决的设计挑战

本设计解决了以下架构问题：

| 挑战 | 旧方案的问题 | 新设计的解决 |
|------|-------------|-------------|
| **Batch vs Online 两套代码** | ~400 行重复，逻辑分散 | 统一 Executor + ChainEntry.mode |
| **Retry for loop 复杂** | 嵌套循环，状态混乱 | 无 loop，状态更新 + 重新入池 |
| **NestedPartProcessor 耦合** | 与 Pipeline 紧密耦合 | 虚拟 unit + 依赖树，完全解耦 |
| **Edge cases 污染主流程** | restore_images 等散落各处 | 全部抽象为 Hooks |
| **Phase 间中间聚合** | polish 生成 chapter_X.md | 不聚合，只在 build-epub 聚合 |
| **Safety block 处理复杂** | 需要维护 blocked_models 列表 | 直接从 chain 移除 provider |
| **Batch/Online 模式切换** | if-else fallback 逻辑 | 同时执行，失败后移除 batch entries + 重新入池 |

---

## 十六、实现清单

### Phase 1: Hooks 体系
1. [x] 创建 `hooks/` 目录
2. [x] 定义 `_protocol.py`
3. [x] 实现 `pre_processors.py`
4. [x] 实现 `transformers.py`
5. [x] 实现 `validators.py` (adapters)
6. [x] 实现 `skip_validators.py`
7. [x] 实现 `error_classifier.py`
8. [x] 实现 `composite.py`

### Phase 2: Executor
9. [x] 创建 `executor/` 目录
10. [x] 定义 `_protocol.py`
11. [x] 实现 `state.py` (UnitState, QuotaConfig, ChainEntry)
12. [ ] 实现 `executor.py` (统一 Executor：batch + online 同时，状态更新 + 重新入池)

### Phase 3: Pipeline
13. [x] 创建 `pipeline_v2.py`（新架构 Pipeline，无 retry 逻辑）
14. [x] 创建 `factory_v2.py`（新架构工厂函数）

### Phase 4: Phase 通用接口
15. [x] 定义 `WorkUnit`, `WorkUnitLoader` Protocol
16. [x] 实现 `PartBasedLoader`
17. [x] 实现通用 `Phase` 类
18. [x] 添加 CLI 命令 `polish-v2`, `translate-v2`
19. [ ] 删除旧的中间聚合逻辑

### Phase 5: 清理
20. [ ] 删除 `batch_pipeline.py`
21. [x] 迁移现有 validators（通过 IndividualValidatorAdapter）
22. [ ] 删除 `NestedPartProcessor`（虚拟 unit 替代）
23. [ ] 删除 `online.py`, `unified.py`, `batch.py`（合并为 `executor.py`）

### Phase 6: 测试
24. [x] 测试 Hooks
25. [x] 测试 UnitState（can_retry, apply_effect）
26. [ ] 测试统一 Executor（batch + online 同时执行）
27. [ ] 测试 Phase 串联

---

## 十七、实现偏离与设计决策 (2025-02)

### 17.1 善意偏离（实现优于设计）

以下偏离是实现过程中发现的更好方案，应反向更新本设计文档：

#### 1. Batch 重试策略

**位置**：`executor/executor.py`

**设计文档说**：失败后移除 batch entries，退化到 online。

**实际实现**：失败集较大时先重跑 batch，避免大量单位退化成逐个 online。

**为什么更好**：90 个里 30 个 truncate，重跑 batch 比逐个 online 更高效。

**政策明确**：
- batch_retry_threshold: 失败率超过多少触发 batch 重试（默认 10%）
- max_batch_retries: 最多重试几次 batch（默认 2）
- 哪些错误类型触发 batch 重试：TRUNCATION, VALIDATION（不包括 SAFETY）

#### 2. ErrorType 细分

**位置**：`core/types.py:26`

**设计文档说**：5 种错误类型（SAFETY, NETWORK, VALIDATION, RATE_LIMIT, UNKNOWN）

**实际实现**：增加了 TIMEOUT, TRUNCATION, PARSE_ERROR, CONTENT_FILTER

**为什么更好**：允许更精确的 quota/effect 配置。例如 TRUNCATION 可以触发 split，而 VALIDATION 不一定。

#### 3. 先 validate 再 post_process

**位置**：`executor/executor.py:621`

**设计文档说**：processor.post_process → hooks.post_process（先处理再验证）

**实际实现**：hooks.post_process（transform + validate）→ processor.post_process（验证通过后）

**为什么更好**：验证失败的结果不应该进入 post_process，避免副作用。协议 `_protocol.py:206` 已更新。

#### 4. raw→validated 两阶段持久化

**位置**：`core/persistence.py`

**设计文档说**：Pipeline 保存到 output_dir

**实际实现**：
- raw/: 未验证的结果（LLM 返回后立即保存）
- validated/: 验证通过后 promote

**为什么更好**：
- 永不丢数据（raw 始终保留）
- 写盘有唯一入口（Persistence.save_raw, promote_batch）
- 支持 resume 时重新验证 raw 文件

### 17.2 结构性约束（测试强制）

以下约束由 `tests/test_architecture.py` 强制执行：

#### 单一定义约束
| 类型 | 定义位置 | 测试 |
|------|----------|------|
| WorkUnit | `core/work_unit.py` | `TestSingleSourceOfTruth::test_workunit_only_defined_once` |
| SplitType | `core/work_unit.py` | `TestSingleSourceOfTruth::test_splittype_only_defined_once` |
| ErrorType | `core/types.py` | `TestSingleSourceOfTruth::test_errortype_only_defined_once` |

#### 导入约束
| 规则 | 测试 |
|------|------|
| WorkUnit 必须从 types 导入 | `TestLegacyIsolation::test_workunit_imported_from_correct_location` |
| SplitType 必须从 types 导入 | `TestLegacyIsolation::test_splittype_imported_from_correct_location` |

#### 职责边界约束
| 规则 | 测试 |
|------|------|
| LLM 重试循环只在 executor/ | `TestBoundaryEnforcement::test_no_llm_retry_loops_outside_executor` |
| longest_fallback 只在 executor/ | `TestBoundaryEnforcement::test_no_longest_fallback_outside_executor` |
| Pipeline/Phase 不直接调用 LLM | `TestBoundaryEnforcement::test_no_direct_llm_calls_in_pipeline_or_phase` |
| Executor 不直接写文件 | `TestBoundaryEnforcement::test_no_file_write_in_executor` |
| Phase 不直接写文件 | `TestBoundaryEnforcement::test_no_direct_file_write_in_phase` |

#### WorkUnit 契约约束
| 规则 | 测试 |
|------|------|
| 必须包含 file_key | `TestWorkUnitContractCodebase::test_workunit_instantiation_uses_required_fields` |
| part 编号必须 1-based | `TestPartNumberingCodebase::test_no_zero_based_part_numbering_in_codebase` |

#### Phase 可组合性约束
| 规则 | 测试 |
|------|------|
| process_all 不能聚合 | `TestNoAggregationBetweenPhases::test_no_aggregation_in_processing_methods` |
| cache_completed 需检查 context_ready | `TestContextReadyEnforcement::test_cache_completed_checks_context_ready` |
| fallback 必须标记 completed | `TestFallbackCompletion::test_fallback_results_mark_completed` |

### 17.3 命名规范

| 后缀 | 含义 | 索引 | 持久化 |
|------|------|------|--------|
| `.part{N}` | 主动分割（Pipeline proactive split） | **1-based** | 是 |
| `.sub{N}` | 动态分割（Executor 失败时分割） | 0-based | 否（虚拟） |

**示例**：
- `chapter_1.part1` = 第 1 个主动分割部分
- `chapter_1.part2.sub0` = 第 2 个部分的第 0 个动态子分割

---

## 十八、过时文档

以下文档已被本文档取代，仅供历史参考：

| 文档 | 状态 | 说明 |
|------|------|------|
| `.claude/architecture-v3.md` | **OUTDATED** | V3 设计被本文档 (V2) 取代 |
| `.claude/architecture-redesign.md` | **OUTDATED** | 早期架构探索 |
| `.claude/validation-redesign.md` | **OUTDATED** | 验证体系已合并到 hooks |
| `.claude/pipeline-executor-redesign.md` | **OUTDATED** | 已合并到本文档 |

---

## 十九、修复记录

### 19.1 Batch 接口修复 (2025-02)

**问题**：Executor._process_batch 假设的接口与实际 GeminiBatchClient 不匹配。

| Executor 假设 | 实际 GeminiBatchClient |
|--------------|----------------------|
| `submit_batch(dict)` | `submit(List[BatchRequest])` |
| `get_status()` → dict["state"] | `get_status()` → BatchJobInfo.state |
| `get_results()` → dict[unit_id] | `get_results()` → List[BatchResponse] |
| `unit.metadata.get('chapter_type')` | `unit.chapter_type` |

**修复**：
- `executor.py:_process_batch` 重写使用正确的接口
- 添加 `TestBatchInterfaceCompatibility` 测试确保接口正确

**测试强制**：
- `test_executor_uses_correct_batch_methods` - 禁止 submit_batch
- `test_executor_handles_batch_response_objects` - 必须从 batch_utils 导入
- `test_executor_uses_workunit_chapter_type` - 禁止 unit.metadata

### 19.2 ProcessContext 构建修复 (2025-02)

**问题**：ProcessContext 没有从 WorkUnit 正确传递字段。

| 缺失字段 | 影响 |
|---------|------|
| part_index, total_parts | Processor prompt 缺失 part 信息 |
| chapter_type | Skip validator 无法识别 front/back matter |
| chapter_title, chapter_number | Prompt 缺失章节上下文 |
| toc_path, page_range | 缺失定位信息 |

**修复**：
- 添加 `ProcessContext.from_work_unit()` 类方法
- Executor._build_context 使用 from_work_unit 或合并 WorkUnit 字段
- Executor._process_batch 同样处理

**测试强制**：
- `test_executor_builds_context_from_workunit` - 禁止最小化构建
- `test_processcontext_from_work_unit_exists` - 必须有 from_work_unit
- `test_processcontext_from_work_unit_propagates_fields` - 字段必须正确传递

### 19.3 嵌套分割上下文注入修复 (2025-02)

**问题**：ContextInjector 使用 `file_key + part_index` 查找前一个 part，对嵌套分割无效。

```
chapter_1.part1.part2 (file_key="chapter_1", part_index=2)
  prev_id = f"{file_key}.part{part_index - 1}"
          = "chapter_1.part1"  ← 错误！应该是 "chapter_1.part1.part1"
```

**修复**：
- 添加 `_get_prev_part_id(unit_id)` 辅助函数
- 使用 `unit_id.rsplit('.part', 1)` 推导前一个兄弟
- 更新所有使用 file_key + part_index 的位置

**影响范围**：
- `build_dependency_graph()` - 构建依赖图
- `get_ready_units()` - 判断就绪状态
- `get_context_for_unit()` - 获取前一 part 上下文
- `sort_by_dependencies()` - 拓扑排序

**测试强制**：
- `test_context_injector_derives_prev_id_from_unit_id` - 禁止 file_key + part_index
- `test_nested_split_prev_id_calculation` - 验证嵌套分割正确

### 19.4 TRUNCATION quota 修复 (2025-02)

**问题**：QuotaConfig 定义了 TRUNCATION 独立 quota (2次)，但 ErrorEffect 把 TRUNCATION 记在 VALIDATION quota 上，导致 TRUNCATION quota 形同虚设。

**修复**：
```python
# Before (bug):
ErrorType.TRUNCATION: ErrorEffect(quota_type=ErrorType.VALIDATION)

# After (fix):
ErrorType.TRUNCATION: ErrorEffect(quota_type=ErrorType.TRUNCATION)
```

**语义差异**：
| 错误类型 | 含义 | 重试策略 |
|---------|------|---------|
| VALIDATION | 输出质量不达标 | 同模型可能成功 |
| TRUNCATION | 模型系统性截断 | 需切换模型或分割 |

**测试强制**：
- `test_error_effects_use_own_quota_type` - 关键 ErrorType 必须用自己的 quota
- `test_quota_config_covers_all_error_types` - QuotaConfig 必须覆盖所有 ErrorType

### 19.5 遗留/重复模块清理 (2025-02)

**问题**：多个未提交的重复模块导致循环导入和 ErrorType.API_ERROR 引用错误。

**删除的未跟踪文件**：
| 路径 | 原因 |
|------|------|
| `core/splitting/` | 与 `processors/utils/` 重复，引用不存在的 `ErrorType.API_ERROR` |
| `core/validation.py` | 从未被导入，未完成的 ValidationRunner |
| `core/utils/` | 与 `processors/utils/` 重复，导致循环导入 |

**更新的导入**：
| 文件 | 修改 |
|------|------|
| `core/__init__.py` | 从 `processors.utils` 导入工具函数 |
| `core/executor/executor.py` | TYPE_CHECKING 导入更新 |
| `core/pipeline_v2.py` | TYPE_CHECKING 导入更新 |
| `core/factory_v2.py` | 实际导入更新 |
| `processors/utils/__init__.py` | 恢复为仅导入本地文件（消除循环依赖） |

**ErrorType.API_ERROR 说明**：
- V1 系统 (`processors/tracker.py`) 定义了 `API_ERROR`
- V2 系统 (`core/types.py`) 没有定义 `API_ERROR`
- V1 系统内部自洽，V2 系统也内部自洽
- 被删除的 `core/splitting/nested_processor.py` 错误地混用了两者

**测试验证**：已通过

### 19.6 Screener 放行 → Batch 跳过 (2025-02)

**问题**：Screener 验证通过的文件仍然被发送给 batch validator（如 agent verifier），导致：
- 浪费 API 费用（agent 调用）
- 重复验证已经验证过的文件

**根因**：
- `ExecutionResult` 没有返回 screener 放行信息
- `Pipeline._run_batch_validation()` 不知道哪些文件已经通过 screener

**修复**：
```
Executor                        Pipeline
   │                               │
   │ context_ready=True            │
   │ ────────────────────►         │
   │ screener_passed.add(key)      │
   │                               │
   │ ExecutionResult               │
   │   .screener_passed            │
   │ ────────────────────►         │
   │                               │ _run_batch_validation(
   │                               │   screener_passed=exec_result.screener_passed
   │                               │ )
   │                               │
   │                               │ skip_keys = screener_passed | chapter_type_skips
   │                               │ files = {k: v for k, v in results
   │                               │          if k NOT in skip_keys}
   │                               │
   │                               │ validator.validate_batch(files)  # 只验证需要的
```

**代码修改**：
1. `ExecutionResult` 新增 `screener_passed: Set[str]` 字段
2. `Executor.execute()` 在 `result.context_ready=True` 时记录到 `screener_passed`
3. `Pipeline._run_batch_validation()` 接受 `screener_passed` 参数，合并到 `skip_keys`
4. 在创建 `VerificationFile` **之前**跳过，避免无谓的 API 调用

**测试强制**：
- `test_execution_result_has_screener_passed_field` - ExecutionResult 必须有该字段
- `test_executor_tracks_screener_passed` - Executor 必须跟踪 context_ready → screener_passed
- `test_pipeline_passes_screener_passed_to_batch_validation` - Pipeline 必须传递
- `test_batch_validation_skips_screener_passed_before_creating_files` - 必须在创建文件前跳过

**测试验证**：已通过

### 19.7 网络错误单一重试权威 (2025-02)

**问题**：网络错误在多层叠加重试，导致：
- 底层 LLMClient tenacity retry（5分钟窗口）
- 上层 Executor quota requeue（同模型再来）
- 叠加后单个网络错误可能卡住 15+ 分钟

**新策略**：单一重试权威 + Fail Fast

| 层级 | 职责 | 策略 |
|------|------|------|
| 底层 (LLMClient) | 瞬时抖动重试 | 最多 2 次（次数限制，非时间） |
| 上层 (Executor) | 切换模型/provider | 立即 |

**代码修改**：

1. **改为次数限制重试**（时间限制对长内容不合理）：
```python
# llm_client.py - 从 stop_after_delay 改为 stop_after_attempt
from tenacity import stop_after_attempt, wait_random_exponential

max_retries = config.get('retry', {}).get('max_retries', 2)
max_wait_between = config.get('retry', {}).get('max_wait_seconds', 10)

@retry_with_logging(
    stop_strategy=stop_after_attempt(max_retries),  # 次数限制
    wait_strategy=wait_random_exponential(max=max_wait_between),
)
```

2. **NETWORK/TIMEOUT/RATE_LIMIT 推进 chain**：
```python
# error_classifier.py
ErrorType.NETWORK: ErrorEffect(
    remove_current_model=True,  # 推进 chain，换下一个模型
    remove_provider=False,      # provider 可能恢复
)
# 同理 TIMEOUT, RATE_LIMIT
```

3. **网络错误不触发 split**：
```python
# executor.py - _handle_failure
network_errors = (ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT)
if error_type in network_errors:
    # 跳过 split，直接 longest fallback 或 fail
    pass
else:
    # 内容相关错误可以尝试 split
    if self._splitter:
        handle_split(...)
```

4. **全局网络熔断**：
```python
# executor.py
if self._consecutive_network_failures >= self._network_circuit_breaker_threshold:
    self._network_circuit_broken = True
    logger.error("CIRCUIT BREAKER: systemic network issue detected")
    # 中止所有 pending，退出
```

**行为矩阵**：

| 错误类型 | 底层重试 | 上层行为 | Split? |
|---------|---------|---------|--------|
| NETWORK | 30秒 | 换模型 | ❌ |
| TIMEOUT | 30秒 | 换模型 | ❌ |
| RATE_LIMIT | 30秒 | 换模型 | ❌ |
| TRUNCATION | 无 | 优先 split | ✅ |
| VALIDATION | 无 | 同模型重试 | ✅ (quota耗尽后) |
| SAFETY | 无 | 换 provider | ❌ |

**测试强制**：
- `test_llm_client_attempt_based_retry` - 底层使用次数限制（非时间）
- `test_network_errors_remove_current_model` - 网络错误必须推进 chain
- `test_network_errors_never_split` - 网络错误永不触发 split
- `test_circuit_breaker_exists` - 必须有熔断机制

**测试验证**：47 tests passed, 4 skipped

---

## 二十、Batch/Online 统一 Quota 政策

本节定义 batch 和 online 路径的统一 quota 规则。

### 20.1 核心原则

**双层 Quota 模型**：
```
┌─────────────────────────────────────────────────────────┐
│ Job-level quota (batch only)                            │
│   - 控制 batch job 整体失败次数                          │
│   - 同时限制次数 + 墙钟时间                              │
│   - 超过后 → 熔断，走 online                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Unit-level quota (shared by batch & online)             │
│   - 单一真相：两条路径共享                               │
│   - 错误分类统一                                        │
│   - 模型链推进规则统一                                   │
└─────────────────────────────────────────────────────────┘
```

### 20.2 Job 失败的成本归属

| 场景 | Unit Quota | Job Quota | 说明 |
|------|-----------|-----------|------|
| Job 成功，Unit 成功 | 不扣 | 不扣 | 正常完成 |
| Job 成功，Unit 失败 | **扣** | 不扣 | 和 online 一致 |
| Job 失败（整体） | 不扣 | **扣** | 避免冤枉 unit |

**关键认知**：Job 失败可能意味着"部分 unit 已执行/计费但结果不可得"。
不扣 unit quota 是合理的（避免冤枉），但会导致"真实开销"与 quota 脱钩。

**补充约束**：Job quota 同时限制 **次数 + 墙钟时间**，防止无限重试错觉：
```python
@dataclass
class BatchQuotaConfig:
    max_job_failures: int = 3             # 最多 3 次 job 失败
    max_job_wallclock_seconds: int = 600  # 累计等待不超过 10 分钟
```

### 20.3 Poison Unit 快速归因

**问题**：有些 job 失败根因是某个 unit 的请求非法（prompt 太大、字段不对）。
如果一律算 job failure，同一个 poison unit 会连续打爆 batch。

**归因策略**（按优先级）：

```python
def attribute_job_failure(job_error: str, units: List[WorkUnit]) -> Attribution:
    """
    1. 明确指向某个 unit → 扣该 unit quota，不扣 job quota
    2. 请求形态问题（prompt 太大）→ 扣该 unit quota
    3. 系统性问题（网络、服务挂了）→ 扣 job quota
    """
    # 检查 error 是否包含 unit key
    if unit_key := extract_unit_key_from_error(job_error):
        return Attribution(type="unit", unit_id=unit_key,
                          error_type=classify_from_string(job_error))

    # 检查是否有 size/shape 问题
    if "too large" in job_error or "exceeded" in job_error:
        # 找最大的 unit（最可能的 poison）
        largest = max(units, key=lambda u: len(u.content))
        return Attribution(type="unit", unit_id=largest.id,
                          error_type=ErrorType.VALIDATION)

    # 默认：系统性问题
    return Attribution(type="job", error_type=ErrorType.NETWORK)
```

**效果**：
- Poison unit 被精确扣 quota → 快速用完 → 走 split 或 fail
- 其他 unit 不受影响
- Job quota 只用于真正的系统性问题

### 20.4 字符串错误分类规则

**挑战**：Online 有异常对象，batch 多半只有 `error: str`（甚至 missing key）。

**分类规则**：

| 关键词 | ErrorType | 说明 |
|--------|-----------|------|
| `"too large"`, `"exceeded"`, `"limit"` | VALIDATION | 请求形态问题 |
| `"timeout"`, `"deadline"` | TIMEOUT | 超时 |
| `"rate"`, `"quota"`, `"429"` | RATE_LIMIT | 限流 |
| `"safety"`, `"blocked"`, `"prohibited"` | SAFETY | 安全拦截 |
| `"truncat"`, `"incomplete"` | TRUNCATION | 截断 |
| `"network"`, `"connection"`, `"unavailable"` | NETWORK | 网络问题 |
| 其他 | UNKNOWN | 兜底，按 NETWORK 处理 |

**精度承诺**：字符串分类精度 < 异常分类，但足够做 quota 决策。
接受误分类风险，优先保证系统稳定。

### 20.5 Batch 失败后的链路推进

**问题**：Online 是 unit 失败立刻切模型；batch 是 job 结束才知道谁失败。
Unit quota 统一了，但**模型链推进时机不统一**。

**政策**：

| Batch Unit 失败类型 | 后续路径 | 理由 |
|--------------------|---------|------|
| SAFETY/CONTENT_FILTER | 移除 provider，**online 同 provider 也跳过** | 内容问题，provider 级别 |
| TRUNCATION | 移除当前 model，可尝试同 provider 其他 model | 模型能力问题 |
| VALIDATION | **同 model online 重试一次**，失败再推进 | 可能是 batch 特有问题 |
| NETWORK/TIMEOUT | 移除当前 model | 快速切换，batch 可能不稳定 |

```python
def get_batch_failure_action(error_type: ErrorType) -> Action:
    """Batch unit 失败后的处理动作."""
    if error_type in (ErrorType.SAFETY, ErrorType.CONTENT_FILTER):
        return Action.REMOVE_PROVIDER  # 整个 provider 不可用
    elif error_type == ErrorType.TRUNCATION:
        return Action.REMOVE_MODEL     # 该模型能力不足
    elif error_type == ErrorType.VALIDATION:
        return Action.RETRY_ONLINE_SAME_MODEL  # 给 online 一次机会
    else:
        return Action.REMOVE_MODEL     # 默认：快速切换
```

**VALIDATION 特殊处理的理由**：
- Batch API 和 online API 可能有细微差异
- 同一个请求在 batch 失败，online 可能成功
- 只给一次机会，失败后正常推进 chain

### 20.6 Circuit Breaker 三态模型

**问题**：当前 `_batch_disabled=True` 是一刀切永久禁用。
一次短暂抖动会让后续长时间都失去 batch 的成本优势。

**三态模型**：

```
     ┌─────────────────────────────────────────────┐
     │                                             │
     ▼                                             │
 ┌────────┐  failure_count >= threshold   ┌──────┐│
 │ CLOSED │ ──────────────────────────► │ OPEN ││
 │ (正常) │                              │(熔断)││
 └────────┘                              └──────┘│
     ▲                                      │    │
     │ success                   cooldown   │    │
     │                           expires    ▼    │
     │                              ┌──────────┐ │
     └────────────────────────────  │HALF_OPEN │─┘
              探测成功               │ (探测)   │ 探测失败
                                    └──────────┘
```

**实现**：

```python
@dataclass
class BatchCircuitBreaker:
    state: Literal["closed", "open", "half_open"] = "closed"
    failure_count: int = 0
    last_failure_time: float = 0

    # 配置
    threshold: int = 3           # 3 次失败触发熔断
    cooldown_seconds: int = 300  # 5 分钟后尝试恢复

    def record_failure(self):
        """记录一次失败."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.state = "open"
            logger.warning(f"Batch circuit breaker OPEN after {self.failure_count} failures")

    def should_try_batch(self) -> bool:
        """是否应该尝试 batch."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.state = "half_open"
                logger.info("Batch circuit breaker HALF_OPEN, probing...")
                return True  # 探测一次
            return False
        if self.state == "half_open":
            return True  # 正在探测
        return False

    def record_success(self):
        """记录一次成功."""
        if self.state == "half_open":
            logger.info("Batch circuit breaker CLOSED, batch recovered")
        self.state = "closed"
        self.failure_count = 0
```

**效果**：
- 短暂抖动 → 熔断 5 分钟 → 自动恢复
- 持续故障 → 反复探测失败 → 保持熔断
- 不会永久失去 batch 的成本优势

### 20.7 政策总结

| 维度 | 政策 |
|------|------|
| Quota 结构 | Unit-level 共享 + Job-level 独立 |
| Job 失败归属 | 先归因，poison unit 扣 unit quota，系统性扣 job quota |
| 错误分类 | 字符串关键词匹配，接受精度损失 |
| 链路推进 | VALIDATION 给 online 一次机会，其他立即推进 |
| 熔断机制 | 三态 circuit breaker + cooldown 自动恢复 |

**实现优先级**：
1. 字符串错误分类（基础设施）
2. Poison unit 归因（避免打爆 batch）
3. 三态 circuit breaker（保护成本优势）
4. VALIDATION 特殊处理（可选优化）
