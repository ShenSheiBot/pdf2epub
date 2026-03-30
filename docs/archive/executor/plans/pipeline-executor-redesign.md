> **OUTDATED**: 本文档已被 `executor-design-v2.md` 取代。已合并到主设计文档。

# Pipeline + Executor 架构重设计

## 问题分析

### 当前架构

两个独立的 Pipeline：
- **ProcessingPipeline** (`pipeline.py`, 809行): Online 处理
- **BatchPipeline** (`batch_pipeline.py`, 910行): Gemini Batch API 处理

### 代码重复

| 功能 | ProcessingPipeline | BatchPipeline | 重复程度 |
|------|-------------------|---------------|----------|
| `_build_context` | 行 674-720 | 行 433-467 | ~80% 相同 |
| `_proactive_split` / `_split_large_units` | 行 460-494 | 行 368-397 | ~90% 相同 |
| `_apply_longest_fallback` | 行 722-762 | 行 835-883 | ~70% 相同 |
| `_mark_complete` | 行 186-193 | 行 207-215 | 几乎相同 |
| `_get_pending_keys` | 行 496-502 | 行 199-205 | 完全相同 |
| Safety block 处理 | 行 624-672 | 行 746-788 | ~60% 相同 |
| Batch validation 调用 | 行 389-434 | 行 663-688 | ~70% 相同 |

估计重复代码：**~400行**

---

## 核心差异分析

### 根本性差异（不可调和）

| 特性 | Online | Batch | 原因 |
|------|--------|-------|------|
| **递归分割** | NestedPartProcessor 失败时递归分割 | 只有 proactive 分割 | Batch 作业提交后无法中途分割 |
| **Context Injection** | 支持（sequential 模式） | 不支持 | Batch 服务端并行执行，无法注入 |
| **实时重试** | 支持（validation_retry_quota） | 不支持 | Batch 作业无法中途重试 |

### 可调和的差异

| 特性 | Online | Batch | 统一方案 |
|------|--------|-------|----------|
| **Individual Validators** | 处理后立即验证 | batch 完成后批量验证 | 执行时机不同，但都支持 |

### 可统一的差异

| 特性 | Online | Batch | 统一方案 |
|------|--------|-------|----------|
| 状态持久化 | ProcessingTracker | BatchState + ProcessingTracker | 统一用 ProcessingTracker |
| LLM 调用 | 直接调用 + model chain | Batch API + online fallback | Executor 封装 |
| 失败处理 | 立即重试 | 收集失败后批量重试 | Executor 内部处理 |

---

## 设计原则

### 核心理念

**理想情况**：整章进去，整章出来。

**现实**：翻译充满 special cases —— 需要跳过某些内容、需要修复 LLM 输出、需要验证结果、需要处理各种错误...

**解决方案**：所有偏离"整章进去整章出来"的逻辑都作为 Hooks 存在，不污染主流程。

```
主流程（纯粹）：content → LLM → result

Hooks（处理所有 edge cases）：
├── Pre-processing: 要不要处理？
├── Post-transform: 修改结果
├── Post-validate: 结果可接受？
├── Skip-validation: 要不要验证？
└── Error-classify: 错误类型？怎么重试？
```

### 为什么用 Hooks？

1. **主流程纯粹**：Pipeline/Executor 只管"调用 LLM"
2. **可扩展**：遇到新的 edge case，加个 hook，不改主代码
3. **可配置**：不同任务（translate vs polish）用不同的 hooks 组合
4. **可测试**：每个 hook 独立测试

---

## Hooks 体系

### 完整 Hooks 分类

```
ExecutionHooks
├── PreProcessingHooks (处理前)
│   └── should_process(key, content) → (bool, reason)
│
├── PostProcessingHooks (处理后)
│   ├── Transformers (修改结果)
│   │   ├── RestoreImagesTransformer
│   │   ├── RemoveArtifactsTransformer
│   │   └── ...
│   │
│   └── Validators (判断结果)
│       ├── LengthValidator
│       ├── NGramValidator
│       └── ...
│
├── SkipValidationHooks (跳过验证)
│   └── should_skip_validation(key, chapter_type) → bool
│
└── ErrorHooks (错误处理)
    └── classify_error(error) → ErrorType
```

---

## 设计方案

### Hooks Protocol

```python
# ============================================================
# Pre-processing Hooks
# ============================================================

@dataclass
class PreProcessResult:
    """Pre-processing 结果"""
    should_process: bool    # 是否需要处理
    skip_reason: str = ""   # 跳过原因（用于日志）
    fallback_result: Optional[str] = None  # 跳过时的替代结果


class PreProcessor(Protocol):
    """预处理器 - 判断是否需要处理"""

    @property
    def name(self) -> str: ...

    def check(
        self,
        key: str,
        content: str,
        context: ProcessContext
    ) -> PreProcessResult:
        """
        判断是否需要处理。

        示例：
        - ImageOnlyFilter: 纯图片页面 → skip, return original
        - EmptyContentFilter: 空内容 → skip
        - AlreadyProcessedFilter: 已完成 → skip

        Returns:
            PreProcessResult(should_process, skip_reason, fallback_result)
        """
        ...


# ============================================================
# Post-processing Hooks: Transform + Validate
# ============================================================

@dataclass
class HookResult:
    """Validate 钩子返回结果"""
    accepted: bool          # 是否接受结果（False 触发重试）
    context_ready: bool     # 结果是否可用于 context injection


class Transformer(Protocol):
    """转换器 - 修改处理结果"""

    @property
    def name(self) -> str: ...

    def transform(
        self,
        key: str,
        original: str,
        result: str
    ) -> str:
        """
        转换处理结果。

        示例：
        - RestoreImagesTransformer: 恢复被 LLM 删除的图片
        - RemoveArtifactsTransformer: 移除 LLM 生成的 artifact

        Returns:
            转换后的结果
        """
        ...


class Validator(Protocol):
    """验证器 - 判断是否接受结果"""

    @property
    def name(self) -> str: ...

    def validate(
        self,
        key: str,
        original: str,
        result: str
    ) -> HookResult:
        """
        验证处理结果。

        示例：
        - LengthValidator: 长度比例检查
        - NGramValidator: 内容完整性检查

        Returns:
            HookResult(accepted, context_ready)
        """
        ...


# ============================================================
# Skip-validation Hooks
# ============================================================

class SkipValidator(Protocol):
    """跳过验证判断器"""

    @property
    def name(self) -> str: ...

    def should_skip(
        self,
        key: str,
        chapter_type: str,
        context: ProcessContext
    ) -> bool:
        """
        判断是否跳过验证。

        示例：
        - ChapterTypeSkipper: front_matter, back_matter 等跳过验证
        - ShortContentSkipper: 内容太短跳过验证

        Returns:
            True = 跳过验证
        """
        ...


# ============================================================
# Error Hooks
# ============================================================

class ErrorClassifier(Protocol):
    """错误分类器"""

    def classify(self, error: Exception) -> ErrorType:
        """
        分类错误类型。

        Returns:
            ErrorType: SAFETY, NETWORK, VALIDATION, RATE_LIMIT, ...
        """
        ...

    def get_retry_strategy(self, error_type: ErrorType) -> RetryStrategy:
        """
        获取重试策略。

        Returns:
            RetryStrategy: 同模型重试 / 切换模型 / 不重试
        """
        ...


# ============================================================
# Composite Hooks
# ============================================================

class ExecutionHooks(Protocol):
    """执行钩子 - 组合所有 hooks"""

    def pre_process(
        self,
        key: str,
        content: str,
        context: ProcessContext
    ) -> PreProcessResult:
        """处理前检查"""
        ...

    def post_process(
        self,
        key: str,
        original: str,
        result: str
    ) -> Tuple[str, HookResult]:
        """处理后：transform + validate"""
        ...

    def should_skip_validation(
        self,
        key: str,
        chapter_type: str,
        context: ProcessContext
    ) -> bool:
        """判断是否跳过验证"""
        ...

    def classify_error(self, error: Exception) -> Tuple[ErrorType, RetryStrategy]:
        """分类错误并获取重试策略"""
        ...
```

### 内置 Hooks 实现

#### PreProcessors

```python
class ImageOnlyFilter(PreProcessor):
    """跳过纯图片页面"""

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
                fallback_result=content  # 原样返回
            )
        return PreProcessResult(should_process=True)


class EmptyContentFilter(PreProcessor):
    """跳过空内容"""

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
    """恢复被 LLM 删除的图片标签"""

    @property
    def name(self) -> str:
        return "RestoreImages"

    def transform(self, key, original, result):
        return restore_lost_images_fast(original, result)


class RemoveArtifactsTransformer(Transformer):
    """移除 LLM 生成的 artifact"""

    @property
    def name(self) -> str:
        return "RemoveArtifacts"

    def transform(self, key, original, result):
        # 移除常见的 LLM artifact
        # - 多余的 markdown code blocks
        # - 解释性文字 "Here is the translation:"
        # - ...
        ...
```

#### SkipValidators

```python
class ChapterTypeSkipper(SkipValidator):
    """基于章节类型跳过验证"""

    SKIP_TYPES = {"front_matter", "back_matter", "notes", "appendix", "toc"}

    @property
    def name(self) -> str:
        return "ChapterTypeSkipper"

    def should_skip(self, key, chapter_type, context):
        return chapter_type in self.SKIP_TYPES
```

#### ErrorClassifiers

```python
class DefaultErrorClassifier(ErrorClassifier):
    """默认错误分类器"""

    def classify(self, error):
        error_msg = str(error).lower()

        if any(kw in error_msg for kw in ["safety", "blocked", "harmful"]):
            return ErrorType.SAFETY

        if any(kw in error_msg for kw in ["rate limit", "quota", "429"]):
            return ErrorType.RATE_LIMIT

        if any(kw in error_msg for kw in ["timeout", "connection", "network"]):
            return ErrorType.NETWORK

        if any(kw in error_msg for kw in ["truncat", "incomplete"]):
            return ErrorType.VALIDATION

        return ErrorType.UNKNOWN

    def get_retry_strategy(self, error_type):
        if error_type == ErrorType.SAFETY:
            return RetryStrategy.SWITCH_MODEL  # 切换到 Anthropic
        elif error_type == ErrorType.NETWORK:
            return RetryStrategy.SAME_MODEL    # 同模型重试
        elif error_type == ErrorType.VALIDATION:
            return RetryStrategy.SAME_MODEL    # 同模型重试
        elif error_type == ErrorType.RATE_LIMIT:
            return RetryStrategy.WAIT_AND_RETRY  # 等待后重试
        else:
            return RetryStrategy.SWITCH_MODEL  # 切换模型
```

### CompositeHooks 实现

```python
class CompositeHooks(ExecutionHooks):
    """
    组合所有类型的 Hooks。

    包含：
    - PreProcessors: 处理前检查
    - Transformers: 修改结果
    - Validators: 验证结果
    - SkipValidators: 跳过验证判断
    - ErrorClassifier: 错误分类
    """

    def __init__(
        self,
        pre_processors: List[PreProcessor] = None,
        transformers: List[Transformer] = None,
        validators: List[Validator] = None,
        skip_validators: List[SkipValidator] = None,
        error_classifier: Optional[ErrorClassifier] = None,
        tracker: Optional[ProcessingTracker] = None
    ):
        self._pre_processors = pre_processors or []
        self._transformers = transformers or []
        self._validators = validators or []
        self._skip_validators = skip_validators or []
        self._error_classifier = error_classifier or DefaultErrorClassifier()
        self._tracker = tracker

    def pre_process(self, key, content, context):
        """处理前检查：任一 pre_processor 说跳过就跳过"""
        for pp in self._pre_processors:
            result = pp.check(key, content, context)
            if not result.should_process:
                return result
        return PreProcessResult(should_process=True)

    def post_process(self, key, original, result):
        """处理后：transform + validate"""
        # Step 1: Transform（链式调用）
        transformed = result
        for transformer in self._transformers:
            transformed = transformer.transform(key, original, transformed)

        # Step 2: Validate（遵循 screener/final 逻辑）
        accepted = True
        context_ready = False

        for validator in self._validators:
            hook_result = validator.validate(key, original, transformed)

            # 记录到 tracker
            if self._tracker:
                self._tracker.record_validation(key, {
                    "validator": validator.name,
                    "accepted": hook_result.accepted,
                    "context_ready": hook_result.context_ready
                })

            if hook_result.context_ready:
                context_ready = True

            if not hook_result.accepted:
                accepted = False
                break  # 有一个不通过就停止

        return (transformed, HookResult(accepted=accepted, context_ready=context_ready))

    def should_skip_validation(self, key, chapter_type, context):
        """判断是否跳过验证：任一 skip_validator 说跳过就跳过"""
        for sv in self._skip_validators:
            if sv.should_skip(key, chapter_type, context):
                return True
        return False

    def classify_error(self, error):
        """分类错误并获取重试策略"""
        error_type = self._error_classifier.classify(error)
        retry_strategy = self._error_classifier.get_retry_strategy(error_type)
        return (error_type, retry_strategy)
```

### Validator 适配器

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

### Executor Protocol

```python
@runtime_checkable
class Executor(Protocol):
    """执行器协议 - 负责实际的内容处理"""

    @property
    def name(self) -> str: ...

    @property
    def supports_hooks(self) -> bool:
        """
        是否支持执行时钩子。

        True: 在处理完每个 unit 后调用 hooks，支持 immediate validation + retry
        False: 忽略 hooks，批量返回后由 Pipeline 手动处理
        """
        ...

    def execute(
        self,
        units: List[WorkUnit],
        processor: ProcessorProtocol,
        context_base: Optional[ProcessContext] = None,
        # 以下参数只在 supports_hooks=True 时有效
        context_injector: Optional[ContextInjector] = None,
        hooks: Optional[ExecutionHooks] = None,
        retry_quota: int = 1,
    ) -> ExecutionResult:
        """
        执行处理。

        Args:
            units: 要处理的单元
            processor: 处理器（build_prompt, clean_response, post_process）
            context_base: 基础上下文
            context_injector: 用于依赖顺序和 context 注入（supports_hooks=True 时）
            hooks: 执行钩子（supports_hooks=True 时）
            retry_quota: 每个 unit 的重试次数（supports_hooks=True 时）

        Returns:
            ExecutionResult
        """
        ...

    def execute_single(
        self,
        unit: WorkUnit,
        processor: ProcessorProtocol,
        context: ProcessContext
    ) -> str:
        """
        执行单个单元。

        用于：
        - Batch 模式的 online fallback
        - Pipeline 手动重试
        """
        ...
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    results: Dict[str, str]     # key -> processed content
    failed: Set[str]            # 处理失败的 keys
    safety_blocked: Set[str]    # 被 safety filter 阻止的 keys
    validation_failed: Set[str] # 验证失败的 keys（hooks 模式）
    stats: Dict[str, Any]       # 执行统计
```

### OnlineExecutor

```python
class OnlineExecutor:
    """
    Online 执行器 - 使用 NestedPartProcessor。

    支持：
    - Hooks 模式（immediate validation + retry）
    - Context injection（sequential 模式）
    - 递归分割（处理失败时）
    """

    def __init__(
        self,
        llm_client: Any,
        model_chain: ModelChain,
        tracker: ProcessingTracker,
        # NestedPartProcessor 配置
        total_retries: int = 5,
        min_tokens_to_split: int = 500,
        max_workers: int = 4,
    ):
        self._llm_client = llm_client
        self._model_chain = model_chain
        self._tracker = tracker
        ...

    @property
    def supports_hooks(self) -> bool:
        return True

    def execute(self, units, processor, context_base,
                context_injector=None, hooks=None, retry_quota=1):
        """
        执行流程：
        1. 根据 context_injector.mode 决定顺序/并行
        2. 对每个 unit：
           a. 获取 context（如果 sequential 且有依赖）
           b. 调用 NestedPartProcessor 处理
           c. 调用 hooks.on_unit_completed
           d. 如果 accepted=False 且 retry_quota > 0，重试
           e. 如果 context_ready=True，缓存结果供后续使用
        """
        if context_injector and context_injector.is_sequential:
            return self._execute_sequential(units, processor, context_base,
                                           context_injector, hooks, retry_quota)
        else:
            return self._execute_parallel(units, processor, context_base,
                                         hooks, retry_quota)
```

### BatchExecutor

```python
class BatchExecutor:
    """
    Batch 执行器 - 使用 Gemini Batch API。

    不支持：
    - Hooks 模式（批量执行，无法逐个验证）
    - Context injection（服务端并行执行）

    失败处理：
    - 小规模失败：使用 OnlineExecutor 做 fallback
    - 大规模失败：提交新 batch job
    """

    def __init__(
        self,
        batch_provider: str = "gemini",
        batch_model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        poll_interval: int = 60,
        max_batch_retries: int = 3,
        online_fallback_threshold: int = 5,
        # 用于 online fallback
        online_executor: Optional[OnlineExecutor] = None,
    ):
        self._online_executor = online_executor
        ...

    @property
    def supports_hooks(self) -> bool:
        return False

    def execute(self, units, processor, context_base,
                context_injector=None, hooks=None, retry_quota=1):
        """
        执行流程：
        1. 构建 batch requests
        2. 提交 batch job
        3. 轮询等待完成
        4. 解析结果
        5. 失败处理：
           - 失败数 <= threshold: 用 online_executor fallback
           - 失败数 > threshold: 提交新 batch job（不超过 max_batch_retries）

        注意：hooks 参数被忽略（不支持）
        """
        ...
```

---

## 统一的 ProcessingPipeline

```python
class ProcessingPipeline:
    def __init__(
        self,
        processor: ProcessorProtocol,
        persistence: ResultPersistence,
        tracker: ProcessingTracker,
        executor: Executor,  # 注入执行器
        # Validators
        individual_validators: Optional[List[ValidatorConfig]] = None,
        batch_validators: Optional[List[ValidatorConfig]] = None,
        # 共享组件
        split_manager: Optional[SplitManager] = None,
        validation_strategy: Optional[ValidationStrategy] = None,
        context_injector: Optional[ContextInjector] = None,
        book_structure: Optional[BookStructure] = None,
        # 配置
        use_longest_fallback: bool = True,
        restore_images: bool = True,
        validation_retry_quota: int = 1,
    ):
        # Context injection 只在支持 hooks 的执行器上有效
        if context_injector and context_injector.is_sequential:
            if not executor.supports_hooks:
                # 静默降级为 parallel 模式
                context_injector = ContextInjector(mode="parallel")

        # 创建 validation runner
        self._individual_runner = IndividualValidationRunner(
            individual_validators,
            has_batch_validators=bool(batch_validators)
        ) if individual_validators else None

        self._batch_runner = BatchValidationRunner(
            batch_validators
        ) if batch_validators else None

    def process_all(self, units, context_base):
        # Step 1: 过滤不需要处理的 units
        units = self._filter_units(units)  # 过滤 image-only, special units 单独处理

        # Step 2: Proactive splitting
        units = self._proactive_split(units)

        # Step 3: 构建 hooks
        hooks = self._build_hooks()

        # Step 4: 执行
        result = self._executor.execute(
            units=units,
            processor=self._processor,
            context_base=context_base,
            context_injector=self._context_injector,
            hooks=hooks,
            retry_quota=self._validation_retry_quota
        )

        # Step 5: 保存 raw（已经过 transform）
        for key, content in result.results.items():
            self._persistence.save_raw(key, content)

        # Step 6: 手动 hooks.process（如果 executor 不支持 hooks）
        if not self._executor.supports_hooks and hooks:
            result = self._run_hooks_manually(result, hooks)

        # Step 7: Batch validation
        if self._batch_runner:
            self._run_batch_validation(result)

        # Step 8: Longest fallback
        # Step 9: Aggregate parts
        ...

    def _build_hooks(self) -> CompositeHooks:
        """构建 CompositeHooks - 组合所有 hooks"""

        # Pre-processors
        pre_processors = [
            EmptyContentFilter(),
        ]
        if self._book_structure:
            pre_processors.append(ImageOnlyFilter(self._book_structure))

        # Transformers
        transformers = []
        if self._restore_images:
            transformers.append(RestoreImagesTransformer())
        # transformers.append(RemoveArtifactsTransformer())

        # Validators（从 individual_validators 适配）
        validators = []
        if self._individual_validators:
            for config in self._individual_validators:
                validators.append(IndividualValidatorAdapter(
                    validator=config.validator,
                    role=config.role,
                    context_ready=config.context_ready
                ))

        # Skip validators
        skip_validators = [
            ChapterTypeSkipper(),
        ]

        # Error classifier
        error_classifier = DefaultErrorClassifier()

        return CompositeHooks(
            pre_processors=pre_processors,
            transformers=transformers,
            validators=validators,
            skip_validators=skip_validators,
            error_classifier=error_classifier,
            tracker=self._tracker
        )
```

---

## 执行流程

### 统一流程

```
Pipeline.process_all():
1. 过滤 units（image-only 跳过，special units 单独处理）
2. Proactive splitting (SplitManager)
3. 构建 CompositeHooks（transformers + validators）
4. Executor.execute(units, processor, context_base, hooks, ...)
   - OnlineExecutor (supports_hooks=True):
     * 逐个/并行处理（根据 context_injector.mode）
     * 处理完每个 unit 后调用 hooks.process(key, original, result)
       → transform: restore_images, remove_artifacts, ...
       → validate: length check, ngram check, ...
     * 如果 accepted=False，重试（不超过 retry_quota）
     * 如果 context_ready=True，缓存供后续 context injection
   - BatchExecutor (supports_hooks=False):
     * 批量提交，等待完成
     * 忽略 hooks 参数
5. Save raw（已经过 transform）
6. 手动 hooks.process（仅当 !executor.supports_hooks）
7. Batch validation (screeners → finals)
8. 失败处理
   - Online: 已在步骤 4 通过 hooks 处理
   - Batch: batch retry 或 online fallback
9. Longest fallback
10. Aggregate parts
```

### Hooks 执行流程

```
┌─────────────────────────────────────────────────────────────────┐
│ Pre-processing                                                   │
│                                                                  │
│   hooks.pre_process(key, content, context)                      │
│     → ImageOnlyFilter: 纯图片？→ skip, return original          │
│     → EmptyContentFilter: 空内容？→ skip                        │
│     → ... 任一说 skip 就 skip                                   │
│                                                                  │
│   如果 should_process=False → 直接返回 fallback_result          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Processing (Executor 核心)                                       │
│                                                                  │
│   processor.build_prompt(content, context)                      │
│     → LLM call                                                  │
│   processor.clean_response(response)                            │
│                                                                  │
│   如果出错 → hooks.classify_error(error)                        │
│     → (ErrorType, RetryStrategy)                                │
│     → 根据策略重试或切换模型                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Post-processing                                                  │
│                                                                  │
│   hooks.post_process(key, original, result)                     │
│                                                                  │
│   1. Transform (链式):                                          │
│      → RestoreImagesTransformer                                 │
│      → RemoveArtifactsTransformer                               │
│      → ...                                                      │
│                                                                  │
│   2. Skip validation check:                                     │
│      hooks.should_skip_validation(key, chapter_type, context)   │
│      → ChapterTypeSkipper: front_matter 等跳过                  │
│      → 如果 skip → 直接返回 transformed, accepted=True          │
│                                                                  │
│   3. Validate (screener/final):                                 │
│      → LengthValidator (screener)                               │
│      → NGramValidator (screener)                                │
│      → ...                                                      │
│                                                                  │
│   返回 (transformed_result, HookResult)                         │
│                                                                  │
│   如果 accepted=False → 重试（回到 Processing）                  │
│   如果 context_ready=True → 缓存供 context injection            │
└─────────────────────────────────────────────────────────────────┘
```

### Hooks 模式 vs 批量模式

| 特性 | Hooks 模式 (Online) | 批量模式 (Batch) |
|------|---------------------|------------------|
| Transform (restore_images 等) | execute 过程中 | execute 后（Pipeline 手动） |
| Validate (length, ngram 等) | execute 过程中 | execute 后（Pipeline 手动） |
| Validation 失败重试 | Executor 内部立即重试 | Pipeline 调用 online fallback |
| Context injection | 支持（sequential 模式） | 不支持 |
| 依赖顺序 | Executor 内部处理 | 无依赖（并行） |

### OnlineExecutor 内部流程（Sequential 模式）

```python
def _execute_sequential(self, units, processor, context_base,
                        context_injector, hooks, retry_quota):
    results = {}
    skipped = set()
    failed = set()
    validation_failed = set()

    for unit in self._get_ready_units(units, context_injector):
        context = self._build_context(unit, context_base, context_injector)

        # ========== Pre-processing ==========
        if hooks:
            pre_result = hooks.pre_process(unit.id, unit.content, context)
            if not pre_result.should_process:
                results[unit.id] = pre_result.fallback_result
                skipped.add(unit.id)
                logger.debug(f"{unit.id}: Skipped - {pre_result.skip_reason}")
                continue

        # ========== Processing + Post-processing ==========
        final_result = None
        accepted = False

        for attempt in range(retry_quota + 1):
            try:
                # 调用 LLM（含 NestedPartProcessor）
                raw_result = self._process_single(unit, processor, context)

                if hooks:
                    # Skip validation check
                    chapter_type = context.chapter_type or ""
                    if hooks.should_skip_validation(unit.id, chapter_type, context):
                        # 只做 transform，不 validate
                        transformed, _ = hooks.post_process(unit.id, unit.content, raw_result)
                        final_result = transformed
                        accepted = True
                        break

                    # Transform + Validate
                    transformed, hook_result = hooks.post_process(
                        unit.id, unit.content, raw_result
                    )

                    if hook_result.accepted:
                        final_result = transformed
                        accepted = True
                        if hook_result.context_ready:
                            context_injector.cache_completed(
                                unit.id, unit.content, transformed
                            )
                        break
                    # 不接受，继续重试
                else:
                    final_result = raw_result
                    accepted = True
                    break

            except Exception as e:
                # ========== Error Classification ==========
                if hooks:
                    error_type, retry_strategy = hooks.classify_error(e)
                    if retry_strategy == RetryStrategy.SWITCH_MODEL:
                        # 切换模型重试（通过 model_chain）
                        ...
                    elif retry_strategy == RetryStrategy.SAME_MODEL:
                        # 同模型重试
                        continue
                    elif retry_strategy == RetryStrategy.NO_RETRY:
                        failed.add(unit.id)
                        break
                else:
                    raise

        if final_result is not None:
            results[unit.id] = final_result
        if not accepted and unit.id not in failed:
            validation_failed.add(unit.id)

    return ExecutionResult(
        results=results,
        skipped=skipped,
        failed=failed,
        validation_failed=validation_failed,
    )
```

### BatchExecutor 内部流程

```
1. Build all batch requests
2. Submit batch job
3. Poll until completion
4. Parse results
5. 失败处理：
   - 失败数 <= online_fallback_threshold: 用 OnlineExecutor.execute_single
   - 失败数 > threshold: 提交新 batch job（不超过 max_batch_retries）
```

---

## Pipeline 职责划分

### Pipeline 处理（不传给 Executor）

| 类型 | 处理方式 |
|------|----------|
| **image-only content** | 直接返回原内容，不调用 LLM |
| **special units** (TOC, metadata) | 单独处理，不持久化到文件，不验证 |

### Pipeline 后处理（Executor 返回后）

| 步骤 | 说明 |
|------|------|
| **save_raw** | 保存到 raw/ 目录（已经过 transform） |
| **手动 hooks.process** | 仅当 !executor.supports_hooks |
| **batch validation** | 所有文件完成后 |

### Executor 职责

| 职责 | 说明 |
|------|------|
| 调用 LLM | processor.build_prompt → llm_client → processor.clean_response |
| Hooks 调用 | hooks.process() = transform + validate |
| 处理失败重试 | NestedPartProcessor（Online）/ batch retry（Batch） |
| Context injection | 仅 supports_hooks=True 且 sequential 模式 |

### Hooks 职责

| 职责 | 说明 |
|------|------|
| **Transform** | restore_images, remove_artifacts 等后处理 |
| **Validate** | length check, ngram check 等验证 |

---

## 文件变更清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `pdf2epub/core/executor/__init__.py` | 导出 |
| `pdf2epub/core/executor/_protocol.py` | 所有 Protocol 和 dataclass |
| `pdf2epub/core/executor/online.py` | OnlineExecutor |
| `pdf2epub/core/executor/batch.py` | BatchExecutor |
| `pdf2epub/core/executor/hooks.py` | CompositeHooks |
| `pdf2epub/core/executor/pre_processors.py` | ImageOnlyFilter, EmptyContentFilter |
| `pdf2epub/core/executor/transformers.py` | RestoreImagesTransformer, RemoveArtifactsTransformer |
| `pdf2epub/core/executor/validators.py` | IndividualValidatorAdapter, LengthValidator, NGramValidator |
| `pdf2epub/core/executor/skip_validators.py` | ChapterTypeSkipper |
| `pdf2epub/core/executor/error_classifier.py` | DefaultErrorClassifier |

### 修改文件

| 文件 | 变更 |
|------|------|
| `pdf2epub/core/pipeline.py` | 重写，接收 Executor，使用 hooks 模式 |
| `pdf2epub/core/factory.py` | 添加 `create_online_executor`, `create_batch_executor`, `create_pipeline` |
| `pdf2epub/cli.py` | 更新命令行参数（--executor-type 等） |

### 删除文件

| 文件 | 原因 |
|------|------|
| `pdf2epub/core/batch_pipeline.py` | 功能移入 BatchExecutor + 统一 Pipeline |

---

## 已确认的设计决策

### 1. Context Injection 在 Batch 模式下

**决策**：静默降级为 parallel 模式。Batch 服务端并行执行，无法注入 context。

### 2. Individual Validation 在 Batch 模式下

**决策**：支持。Batch 完成后批量调用 individual validators（for loop）。

区别在于：
- Online: 每文件处理后立即验证，失败可用 validation_retry_quota 重试
- Batch: batch 完成后批量验证，失败进入 failed_keys，走 batch retry 或 online fallback

这样 ngram（individual screener）+ agent（batch final）组合可以正常工作。

### 3. NestedPartProcessor 与 Batch

**决策**：
- Batch 模式只使用 SplitManager 做 proactive splitting
- Batch 作业失败后，online fallback 时使用 NestedPartProcessor

### 4. 状态持久化

**决策**：
- BatchState 是 BatchExecutor 的内部实现细节（记录 active_job_name 用于 resume）
- Pipeline 层统一使用 ProcessingTracker 记录完成状态
- 不将 BatchState 合并到 ProcessingTracker

---

## 迁移策略

### Phase 1: 创建 Executor 模块

1. 创建 `executor/` 目录结构
2. 实现 Executor Protocol
3. 实现 OnlineExecutor（从 pipeline.py 提取）
4. 实现 BatchExecutor（从 batch_pipeline.py 提取）

### Phase 2: 重写 Pipeline

1. 修改 ProcessingPipeline 接收 Executor
2. 移除直接 LLM 调用代码
3. 添加执行器能力检查

### Phase 3: 更新工厂和 CLI

1. 更新 factory.py
2. 更新 cli.py
3. 删除 batch_pipeline.py

### Phase 4: 测试

1. 验证 online 模式功能不变
2. 验证 batch 模式功能不变
3. 验证 online fallback 工作正常

---

## 风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| 功能回归 | 高 | 分阶段迁移，每阶段单独测试 |
| 性能影响 | 低 | Executor 只是封装，无额外开销 |
| API 变更 | 中 | factory.py 保持向后兼容 |

---

## 实现清单

### Phase 1: 创建 Hooks 体系

1. [ ] 创建 `executor/` 目录结构
2. [ ] 定义 `_protocol.py`:
   - [ ] PreProcessResult, HookResult dataclass
   - [ ] PreProcessor Protocol
   - [ ] Transformer Protocol
   - [ ] Validator Protocol
   - [ ] SkipValidator Protocol
   - [ ] ErrorClassifier Protocol
   - [ ] ExecutionHooks Protocol
   - [ ] ExecutionResult dataclass
   - [ ] Executor Protocol
3. [ ] 实现 `pre_processors.py`:
   - [ ] ImageOnlyFilter
   - [ ] EmptyContentFilter
4. [ ] 实现 `transformers.py`:
   - [ ] RestoreImagesTransformer
   - [ ] RemoveArtifactsTransformer（可选）
5. [ ] 实现 `validators.py`:
   - [ ] IndividualValidatorAdapter（适配现有 IndividualValidator）
6. [ ] 实现 `skip_validators.py`:
   - [ ] ChapterTypeSkipper
7. [ ] 实现 `error_classifier.py`:
   - [ ] DefaultErrorClassifier
8. [ ] 实现 `hooks.py`:
   - [ ] CompositeHooks（组合所有 hooks）

### Phase 2: 创建 Executor

9. [ ] 实现 `online.py`:
   - [ ] OnlineExecutor
   - [ ] 集成 NestedPartProcessor
   - [ ] 集成 hooks（pre_process, post_process, classify_error）
   - [ ] 实现 _execute_sequential（context injection）
   - [ ] 实现 _execute_parallel
10. [ ] 实现 `batch.py`:
    - [ ] BatchExecutor
    - [ ] 集成 BatchState
    - [ ] 集成 OnlineExecutor 做 fallback

### Phase 3: 重写 Pipeline

11. [ ] 重写 `pipeline.py`:
    - [ ] 接收 Executor 参数
    - [ ] 实现 _build_hooks()（根据配置构建 CompositeHooks）
    - [ ] 移除所有硬编码的 special case 处理
    - [ ] 移除直接 LLM 调用代码

### Phase 4: 更新工厂和 CLI

12. [ ] 更新 `factory.py`:
    - [ ] create_hooks()（构建 CompositeHooks）
    - [ ] create_online_executor()
    - [ ] create_batch_executor()
    - [ ] 更新 create_processing_pipeline()
13. [ ] 更新 `cli.py`:
    - [ ] --executor-type 参数

### Phase 5: 清理和测试

14. [ ] 删除 `batch_pipeline.py`
15. [ ] 迁移现有 validators 到 hooks 体系
16. [ ] 测试 Pre-processing hooks
17. [ ] 测试 Transform + Validate hooks
18. [ ] 测试 Skip validation hooks
19. [ ] 测试 Error classification + retry
20. [ ] 测试 Online 模式（sequential + parallel）
21. [ ] 测试 Batch 模式 + fallback
