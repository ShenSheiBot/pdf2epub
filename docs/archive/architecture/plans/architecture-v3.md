> **OUTDATED**: 本文档已被 `executor-design-v2.md` 取代。仅供历史参考。

# PDF2EPUB 架构 V3 - 完整设计

## 设计目标

1. **功能完整性**: 包含旧架构的所有功能，无一遗漏
2. **架构安全性**: 违反设计原则的代码在导入时崩溃
3. **模块化**: 每个组件单一职责，通过组合而非继承扩展
4. **可测试性**: 每个组件可独立测试

---

## 模块清单（共 12 个核心模块）

```
pdf2epub/core/
├── _frozen.py          # 冻结基类、@final、__init_subclass__
├── _protocol.py        # ProcessorProtocol, ValidatorProtocol, 所有接口
├── pipeline.py         # ProcessingPipeline (实时处理)
├── batch_pipeline.py   # BatchPipeline (Gemini Batch API)
├── validation.py       # ValidationPipeline (两阶段验证)
├── persistence.py      # ResultPersistence (raw→validated)
├── state.py            # StateManager + ProcessingTracker
├── registry.py         # ComponentRegistry
├── discovery.py        # UnitDiscovery
├── splitting.py        # SplitManager (版本追踪、动态分割)
├── context.py          # ContextInjector (上下文注入、依赖调度)
├── model_chain.py      # ModelChain (多模型回退)
├── diagnostics.py      # DiagnosticsCollector (token追踪、错误分类)
└── book_structure.py   # BookStructure (章节解析、内容类型检测)
```

---

## 一、WorkUnit 完整定义

```python
@dataclass(frozen=True)
class WorkUnit:
    """不可变工作单元 - 包含旧架构所有字段"""
    # 基础标识
    id: str                          # "chapter_5" 或 "chapter_5.part2"
    file_key: str                    # "chapter_5"
    content: str                     # 待处理内容
    input_path: Path                 # 源文件路径

    # 分割信息
    part_index: Optional[int]        # 1-based 部分索引，None 表示单文件
    total_parts: int                 # 总部分数，默认 1
    split_version: int               # 分割版本号

    # 依赖与调度
    dependencies: Tuple[str, ...]    # 依赖的 unit IDs（用于上下文注入）
    priority: int                    # 调度优先级（越小越高）

    # 内容元数据
    token_count: int                 # 缓存的 token 数
    chapter_type: Optional[str]      # "notes", "appendix", "front_matter", etc.
    chapter_title: Optional[str]     # 章节标题

    # 结构信息
    toc_path: Optional[str]          # 目录中的路径
    page_range: Optional[Tuple[int, int]]  # (起始页, 结束页)

    # 脚注
    footnote_refs: Tuple[int, ...]   # 引用的脚注编号
    footnotes: Optional[Dict[int, str]]  # 脚注定义 {编号: 内容}

    @property
    def is_part(self) -> bool:
        return self.part_index is not None

    @property
    def is_first_part(self) -> bool:
        return self.part_index == 1

    @property
    def is_front_back_matter(self) -> bool:
        return self.chapter_type in ("front_matter", "back_matter", "notes", "appendix")
```

---

## 二、ProcessContext 完整定义

```python
@dataclass(frozen=True)
class ProcessContext:
    """不可变处理上下文"""
    # 基础信息
    file_key: str
    book_title: str

    # 分割信息
    part_index: Optional[int] = None
    total_parts: int = 1

    # 语言信息
    source_language: str = "Japanese"
    target_language: str = "Chinese"

    # 内容类型（自动检测或指定）
    content_type: str = "general"  # "general", "japanese", "academic"

    # 章节信息
    chapter_type: Optional[str] = None
    chapter_title: Optional[str] = None
    is_notes_chapter: bool = False

    # 上下文注入（前一部分的信息）
    previous_original: Optional[str] = None
    previous_processed: Optional[str] = None

    # book_structure 信息
    is_vertical_text: bool = False
    has_global_footnotes: bool = False
    book_language: Optional[str] = None

    # 扩展字段
    extra: Dict[str, Any] = field(default_factory=dict)
```

---

## 三、SplitManager（分割管理）

```python
@check_final_methods
class SplitManager(Frozen, frozen=True):
    """
    分割管理器 - 处理大文件的动态分割

    功能：
    1. 版本追踪（永不覆盖旧版本）
    2. 根据 token 限制自动分割
    3. 失败时自动重分割
    4. 支持多种分割策略
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save'}

    def __init__(
        self,
        output_dir: Path,
        max_tokens_per_part: int = 4000,
        max_resplits: int = 3,
        consecutive_failures_threshold: int = 2
    ): ...

    @final
    def get_or_create_split(
        self,
        file_key: str,
        content: str,
        strategy: str = "auto"
    ) -> SplitResult:
        """获取或创建分割，返回 SplitResult"""
        ...

    @final
    def trigger_resplit(
        self,
        file_key: str,
        failed_part_index: int,
        reason: str
    ) -> Optional[SplitResult]:
        """触发重分割（失败时调用）"""
        ...

    @final
    def get_current_version(self, file_key: str) -> int:
        """获取当前分割版本"""
        ...

    @final
    def get_split_history(self, file_key: str) -> List[SplitRecord]:
        """获取分割历史"""
        ...


@dataclass(frozen=True)
class SplitResult:
    """分割结果"""
    parts: Tuple[str, ...]           # 分割后的内容
    part_infos: Tuple[PartInfo, ...] # 每部分的元信息
    version: int                      # 版本号
    method: str                       # "content_splitter", "llm_resplit", "no_split"
    reason: str                       # 分割原因
    old_to_new_mapping: Optional[Dict[int, List[int]]]  # 重分割时的映射


@dataclass(frozen=True)
class SplitRecord:
    """分割记录（持久化）"""
    timestamp: float
    split_points: Tuple[int, ...]    # 字符位置
    total_tokens: int
    part_count: int
    method: str
    reason: str
    triggered_by: Optional[str]      # 触发重分割的 unit
    version: int
    content_hash: str                # 内容哈希（用于检测变更）
```

---

## 四、ContextInjector（上下文注入）

```python
@check_final_methods
class ContextInjector(Frozen, frozen=True):
    """
    上下文注入器 - 管理部分之间的上下文传递

    功能：
    1. 依赖感知调度
    2. 上下文传递（前一部分的原文+译文）
    3. 支持顺序/并行模式
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save'}

    def __init__(
        self,
        mode: str = "parallel",  # "parallel" 或 "sequential"
        persistence: Optional[ResultPersistence] = None
    ): ...

    @final
    def build_dependency_graph(
        self,
        units: List[WorkUnit]
    ) -> Dict[str, List[str]]:
        """构建依赖图"""
        ...

    @final
    def get_ready_units(
        self,
        all_units: List[WorkUnit],
        completed: Set[str]
    ) -> List[WorkUnit]:
        """获取可以开始处理的 units（依赖已满足）"""
        ...

    @final
    def get_context_for_unit(
        self,
        unit: WorkUnit,
        completed_results: Dict[str, str]
    ) -> Optional[Tuple[str, str]]:
        """获取 unit 的上下文（前一部分的 original, processed）"""
        ...

    @final
    def inject_context(
        self,
        context: ProcessContext,
        previous_original: str,
        previous_processed: str
    ) -> ProcessContext:
        """注入上下文到 ProcessContext"""
        ...
```

---

## 五、ModelChain（多模型回退）

```python
@check_final_methods
class ModelChain(Frozen, frozen=True):
    """
    模型链 - 管理多模型回退策略

    功能：
    1. 按优先级尝试多个模型
    2. 根据错误类型决定是否切换模型
    3. 安全屏蔽自动切换到 Anthropic
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save'}

    def __init__(self, configs: List[Dict[str, Any]]): ...

    @final
    def get_next_model(
        self,
        current_index: int,
        error_type: Optional[ErrorType] = None
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        """获取下一个模型配置"""
        ...

    @final
    def should_retry_same_model(self, error_type: ErrorType) -> bool:
        """是否应该用同一模型重试"""
        ...

    @final
    def get_fallback_for_safety_block(self) -> Dict[str, Any]:
        """获取安全屏蔽的回退模型（Anthropic）"""
        ...

    @final
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """获取所有模型配置"""
        ...


class ErrorType(Enum):
    """错误类型枚举"""
    TRUNCATION = "truncation"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    CONTENT_FILTER = "content_filter"  # 安全屏蔽
    PARSE_ERROR = "parse_error"
    UNKNOWN = "unknown"
```

---

## 六、DiagnosticsCollector（诊断收集）

```python
@check_final_methods
class DiagnosticsCollector(Frozen, frozen=True):
    """
    诊断收集器 - 收集处理过程的所有诊断信息

    功能：
    1. Token 使用追踪
    2. 时长追踪
    3. 错误分类与统计
    4. 尝试历史记录
    5. 诊断笔记生成
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save'}

    def __init__(self, output_dir: Path): ...

    @final
    def record_attempt(
        self,
        file_key: str,
        attempt: AttemptRecord
    ) -> None:
        """记录一次尝试"""
        ...

    @final
    def record_error(
        self,
        file_key: str,
        error_type: ErrorType,
        error_message: str,
        response: Optional[str] = None
    ) -> Path:
        """记录错误并保存错误输出，返回保存路径"""
        ...

    @final
    def get_statistics(self) -> ProcessingStatistics:
        """获取统计信息"""
        ...

    @final
    def get_longest_attempt(self, file_key: str) -> Optional[AttemptRecord]:
        """获取最长的尝试（用于 longest fallback）"""
        ...

    @final
    async def generate_diagnostic_note(
        self,
        file_key: str,
        original: str,
        failed_response: str,
        error_history: List[AttemptRecord]
    ) -> str:
        """使用 Agent 生成诊断笔记"""
        ...

    @final
    def save(self) -> None:
        """保存到 JSON 文件"""
        ...


@dataclass(frozen=True)
class AttemptRecord:
    """尝试记录"""
    timestamp: float
    status: str  # "completed" 或 "failed"
    model: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    content: str
    content_length: int
    retry_count: int
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    error_output_path: Optional[Path] = None
    used_fallback: bool = False
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class ProcessingStatistics:
    """处理统计"""
    total_units: int
    completed: int
    failed: int
    pending: int
    total_attempts: int
    total_retries: int
    errors_by_type: Dict[ErrorType, int]
    models_used: Dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    total_duration_seconds: float
```

---

## 七、BookStructure（书籍结构）

```python
@check_final_methods
class BookStructure(Frozen, frozen=True):
    """
    书籍结构管理 - 从 book_structure.json 加载并提供查询

    功能：
    1. 章节类型检测
    2. 内容类型自动检测
    3. 语言检测
    4. 全局脚注检测
    """

    _FORBIDDEN_METHODS = {'process', 'validate', 'save'}

    def __init__(self, book_dir: Path): ...

    @final
    def get_chapter_info(self, file_key: str) -> ChapterInfo:
        """获取章节信息"""
        ...

    @final
    def detect_content_type(self, content: str) -> str:
        """自动检测内容类型"""
        ...

    @final
    def has_notes_chapter(self) -> bool:
        """是否有全局注释章节"""
        ...

    @final
    def is_vertical_text(self) -> bool:
        """是否竖排文本"""
        ...

    @final
    def get_language(self) -> Optional[str]:
        """获取书籍语言"""
        ...

    @final
    def is_image_only_content(self, content: str) -> bool:
        """检测是否仅图片内容"""
        ...


@dataclass(frozen=True)
class ChapterInfo:
    """章节信息"""
    file_key: str
    chapter_type: str  # "chapter", "notes", "appendix", "front_matter", "back_matter"
    title: Optional[str]
    number: Optional[str]  # "5" 或 "7.1.1"
    toc_path: Optional[str]
    page_range: Optional[Tuple[int, int]]
```

---

## 八、BatchPipeline（批处理 API）

```python
@check_final_methods
class BatchPipeline(Frozen, frozen=True):
    """
    批处理管道 - 使用 Gemini Batch API

    功能：
    1. 50% 成本降低
    2. 状态持久化，支持中断恢复
    3. 在线回退（小规模失败）
    4. 安全屏蔽处理
    """

    _FORBIDDEN_METHODS = {'build_prompt', 'clean_response', 'post_process'}

    def __init__(
        self,
        processor: ProcessorProtocol,
        validation: ValidationPipeline,
        persistence: ResultPersistence,
        state: StateManager,
        split_manager: SplitManager,
        model_chain: ModelChain,
        diagnostics: DiagnosticsCollector,
        book_structure: Optional[BookStructure] = None,
        # Batch API 配置
        batch_provider: str = "gemini",
        poll_interval: int = 60,
        online_fallback_threshold: int = 5,
        max_retries: int = 3,
        use_longest_fallback: bool = True
    ): ...

    @final
    def process_all(
        self,
        units: List[WorkUnit],
        context_base: ProcessContext
    ) -> ProcessingResult:
        """主入口"""
        ...

    @final
    def _submit_batch_job(
        self,
        requests: List[BatchRequest]
    ) -> str:
        """提交批处理作业，返回 job_name"""
        ...

    @final
    def _wait_for_completion(
        self,
        job_name: str
    ) -> BatchJobResult:
        """等待作业完成"""
        ...

    @final
    def _process_results(
        self,
        results: BatchJobResult,
        originals: Dict[str, str]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """处理结果，返回 (passed, failed, safety_blocked)"""
        ...

    @final
    def _try_online_fallback(
        self,
        failed_keys: Set[str],
        safety_blocked_keys: Set[str],
        originals: Dict[str, str],
        context_base: ProcessContext
    ) -> Set[str]:
        """在线回退处理，返回仍然失败的 keys"""
        ...


@dataclass
class BatchState:
    """批处理状态（可变，用于持久化）"""
    active_job_name: Optional[str] = None
    active_job_requests: List[str] = field(default_factory=list)
    pending_files: List[str] = field(default_factory=list)
    retry_count: int = 0
    failed_keys: Set[str] = field(default_factory=set)
    completed_keys: Set[str] = field(default_factory=set)
    processing_keys: Set[str] = field(default_factory=set)
    safety_blocked_keys: Set[str] = field(default_factory=set)
    attempt_history: Dict[str, List[AttemptRecord]] = field(default_factory=dict)
```

---

## 九、ProcessingPipeline（实时处理）- 增强版

```python
@check_final_methods
class ProcessingPipeline(Frozen, frozen=True):
    """
    处理管道 - 实时处理（非批处理 API）

    增强功能：
    1. 上下文注入
    2. 动态分割
    3. 多模型回退
    4. 诊断收集
    """

    _FORBIDDEN_METHODS = {'build_prompt', 'validate', 'save'}

    def __init__(
        self,
        processor: ProcessorProtocol,
        validation: ValidationPipeline,
        persistence: ResultPersistence,
        state: StateManager,
        llm_client: LLMClient,
        # 新增组件
        split_manager: Optional[SplitManager] = None,
        context_injector: Optional[ContextInjector] = None,
        model_chain: Optional[ModelChain] = None,
        diagnostics: Optional[DiagnosticsCollector] = None,
        book_structure: Optional[BookStructure] = None,
        # 配置
        max_workers: int = 4,
        max_retries: int = 3,
        use_longest_fallback: bool = True
    ): ...

    @final
    def process_all(
        self,
        units: List[WorkUnit],
        context_base: ProcessContext
    ) -> ProcessingResult:
        """主入口 - 增强版"""
        # 1. 动态分割大文件
        # 2. 构建依赖图
        # 3. 重新验证 raw 文件
        # 4. 依赖感知调度处理
        # 5. 聚合多部分文件
        ...

    @final
    def _process_with_context_injection(
        self,
        units: List[WorkUnit],
        context_base: ProcessContext
    ) -> Dict[str, str]:
        """带上下文注入的处理"""
        ...

    @final
    def _process_single_with_retry(
        self,
        unit: WorkUnit,
        context: ProcessContext
    ) -> Optional[str]:
        """处理单个 unit，支持多模型回退"""
        ...
```

---

## 十、ValidationPipeline - 增强版

```python
@check_final_methods
class ValidationPipeline(Frozen, frozen=True):
    """
    验证管道 - 两阶段验证

    增强功能：
    1. 错误分类
    2. 诊断笔记生成
    3. 前后事项跳过
    """

    def __init__(
        self,
        validators: List[ValidatorProtocol],
        diagnostics: Optional[DiagnosticsCollector] = None,
        skip_validation: bool = False
    ): ...

    @final
    def validate_single(
        self,
        original: str,
        processed: str,
        file_key: str,
        chapter_type: Optional[str] = None  # 新增：跳过前后事项
    ) -> ValidationResult:
        ...

    @final
    def validate_batch(
        self,
        files: Dict[str, Tuple[str, str]],
        chapter_types: Optional[Dict[str, str]] = None  # 新增
    ) -> BatchValidationResult:
        ...
```

---

## 十一、TOC 翻译

```python
@check_final_methods
class TOCTranslator(Frozen, frozen=True):
    """
    目录翻译器

    功能：
    1. 加载 toc_tree.json
    2. 翻译标题
    3. 保存翻译后的目录
    """

    _FORBIDDEN_METHODS = {'process', 'validate'}

    def __init__(
        self,
        book_dir: Path,
        source_language: str,
        target_language: str,
        llm_client: LLMClient
    ): ...

    @final
    def translate_toc(self) -> None:
        """翻译并保存目录"""
        ...
```

---

## 十二、命令层组装

```python
def run_translate(
    config: Dict,
    book_title: str,
    source_language: str = "Japanese",
    target_language: str = "Chinese",
    max_workers: int = 4,
    max_retries: int = 3,
    resume: bool = False,
    use_longest_fallback: bool = True,
    skip_validation: bool = False,
    use_entities: Optional[bool] = None,
    file_filter: Optional[List[str]] = None,
    # 新增参数
    use_batch_api: bool = False,
    processing_mode: str = "parallel",  # "parallel" 或 "sequential"
) -> ProcessingResult:
    """
    翻译命令 - 完整组装所有组件
    """
    # 1. 加载 book_structure
    book_structure = BookStructure(Path("output") / book_title)

    # 2. 创建诊断收集器
    diagnostics = DiagnosticsCollector(output_dir)

    # 3. 创建模型链
    model_chain = ModelChain(config.get('translation', {}).get('models', []))

    # 4. 创建分割管理器
    split_manager = SplitManager(
        output_dir,
        max_tokens_per_part=config.get('model_output_limits', {}).get('_default', 4000)
    )

    # 5. 创建上下文注入器
    context_injector = ContextInjector(
        mode=processing_mode,
        persistence=persistence
    )

    # 6. 创建验证器
    validators = _create_validators(config, target_language, skip_validation)
    validation = ValidationPipeline(validators, diagnostics, skip_validation)

    # 7. 创建处理器
    processor = TranslateProcessor(...)

    # 8. 创建持久化和状态管理
    persistence = ResultPersistence(output_dir)
    state = StateManager(persistence, diagnostics)

    # 9. 选择管道类型
    if use_batch_api:
        pipeline = BatchPipeline(
            processor=processor,
            validation=validation,
            persistence=persistence,
            state=state,
            split_manager=split_manager,
            model_chain=model_chain,
            diagnostics=diagnostics,
            book_structure=book_structure,
            ...
        )
    else:
        pipeline = ProcessingPipeline(
            processor=processor,
            validation=validation,
            persistence=persistence,
            state=state,
            llm_client=llm_client,
            split_manager=split_manager,
            context_injector=context_injector,
            model_chain=model_chain,
            diagnostics=diagnostics,
            book_structure=book_structure,
            ...
        )

    # 10. 发现工作单元
    units = discover_units(input_dir, output_dir, book_structure)

    # 11. 执行
    result = pipeline.process_all(units, context_base)

    # 12. 保存诊断信息
    diagnostics.save()

    # 13. 翻译目录（如果是翻译命令）
    toc_translator = TOCTranslator(book_dir, source_language, target_language, llm_client)
    toc_translator.translate_toc()

    return result
```

---

## 组件依赖图

```
                    ┌─────────────────┐
                    │   CLI / Command │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ BookStructure│   │ Diagnostics │   │ ModelChain  │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │SplitManager │   │ContextInject│   │ UnitDiscovery│
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────────┐
    │         ProcessingPipeline / BatchPipeline     │
    └────────────────────────┬───────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Processor  │   │ Validation  │   │ Persistence │
    │  (Protocol) │   │  Pipeline   │   │             │
    └─────────────┘   └──────┬──────┘   └──────┬──────┘
                             │                 │
                             ▼                 ▼
                      ┌─────────────┐   ┌─────────────┐
                      │ Validators  │   │StateManager │
                      │ (Protocol)  │   │             │
                      └─────────────┘   └─────────────┘
```

---

## 架构测试清单

### AST 检查（导入时崩溃）
1. Processors 不能定义 `validate`, `save`, `_batch_validate_and_save` 等方法
2. Processors 不能导入 `ProcessingTracker`, `BatchState`, `ResultPersistence` 等类
3. Processors 不能直接做文件 I/O (`write_text`, `read_text`, `mkdir`, `open`)
4. 冻结类不能被继承
5. @final 方法不能被覆盖

### 运行时检查
1. 所有 Processor 实现 ProcessorProtocol
2. 所有 Validator 实现 ValidatorProtocol
3. 组件通过 Registry 注册

### 集成测试
1. 实时处理完整流程
2. 批处理完整流程
3. 上下文注入
4. 动态分割
5. 多模型回退
6. 最长回退策略

---

## 迁移计划

1. **Phase 1**: 创建所有核心模块（skeleton）
2. **Phase 2**: 实现 SplitManager, ContextInjector, ModelChain, DiagnosticsCollector
3. **Phase 3**: 增强 ProcessingPipeline
4. **Phase 4**: 实现 BatchPipeline
5. **Phase 5**: 增强 BookStructure 和 TOCTranslator
6. **Phase 6**: 更新命令层
7. **Phase 7**: 迁移旧代码、删除重复
