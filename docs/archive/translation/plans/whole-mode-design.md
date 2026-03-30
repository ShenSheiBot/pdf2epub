# Whole Mode 设计文档：长翻译与 Agent 辅助框架

## 问题的本质

LLM 的输入能力与输出能力存在根本性的不对称。强力模型已可一次性输入百万 tokens，但输出能力局限于一万到几万 tokens。这个不对称性在可预见的未来不会消失——即便 10 年后 LLM 能一次性翻译百万字的书，也无法保证每次请求都能稳定完成。

当前系统的所有翻译/处理流程都基于 **split 模式**——将内容切成小块，每块独立处理。这是对 LLM 输出瓶颈的工程妥协：牺牲长上下文的输入优势，换取高并发带来的速度和稳定性。

但有些场景 split 模式无法胜任：

| 场景 | 为什么不能 split |
|------|-----------------|
| PDF 目录结构分析（refine） | JSON 结构必须完整，切碎后无法拼回 |
| 压缩 HTML 翻译 | 行数对齐、HTML tag 完整性要求结构保持 |
| 高质量全文翻译 | 切碎后丢失跨段落的上下文关联 |

这些场景需要一种新模式：**whole 模式**——将完整内容一次性喂给 LLM，利用长 context 理解全局，但通过 agent 干预来应对输出的不稳定性。

---

## 思想演进

### 第一步：从 regex 到 agent

最初的想法是用启发式规则（正则表达式、JSON 解析错误位置、括号匹配）来处理 LLM 输出的各种异常。这条路是走不通的。

LLM 输出的不稳定性不是一个格式问题，而是一个**语义问题**。截断只是最温和的情况——更恐怖的是幻觉：模型可能在输出到一半时开始"创作"，生成几百行原文根本不存在的内容，而且格式完美、看起来完全正确。你无法用正则区分"正确的翻译"和"格式完美的幻觉"。只有 LLM agent 能做这个判断，因为它需要**理解内容**才能知道输出是否忠实于输入。

而且，refine、HTML 翻译这些操作都不是真正的批量操作——一本书就几次调用。多一次 agent 调用的成本可以忽略。

### 第二步：短翻与长翻的本质

split 和 whole 是两种根本不同的翻译哲学：

**Split（短翻）**：放弃 LLM 的长 context 输入优势，换来高并发带来的速度和稳定性。把一本书撕成碎片，分给互不认识的译者各翻各的，再拼回去。

**Whole（长翻）**：拥抱长 context 输入优势，换来低并发和不稳定性。一个译者通读全文后从头翻译，遇到问题回头改，改完继续。

Whole 模式更忠实于人类译者的自然工作流：
1. 通读全文（长 context 输入）
2. 从头开始翻，维护一个心理游标
3. 翻到某处发现前面有问题，回去改（游标后退）
4. 改完继续往下翻（游标前进）
5. 最后通读校对（agent 验证）

Whole 模式通过 agent 的帮助，能解决所有 split 模式不能解决的问题——结构化输出、更稳定的上下文。代价是贵和慢。

### 第三步：统一框架而非独立系统

长翻和短翻不能是两个独立的系统。理由：

1. **同一本书内可能混合使用**：短章节用 split（快、便宜），长章节或结构复杂的章节用 whole（贵但可靠）
2. **Fallback**：split 反复失败的 unit 可以自动升级到 whole；whole 救不了的内容可以降级到 split
3. **用户选择**：对于高价值书籍，用户可能希望全书都用 whole 模式

当前系统中，batch 和 online 的统一设计是榜样：两者只是 ChainEntry 上的一个 mode 字段，共享完全相同的 work unit 生命周期、hooks、validation、error handling、persistence。没有代码路径的分叉。

Split 和 whole 应当以同样的方式统一。

---

## 架构设计

### 两个正交维度

当前系统已有一个维度：**transport**（batch/online）——决定 LLM 调用的派发方式。

新增第二个维度：**strategy**（split/whole）——决定内容的处理策略。

这两个维度正交组合：

| | Split | Whole |
|---|---|---|
| **Batch** | 当前默认（切块 → batch API） | 整章提交 batch，agent 验收结果 |
| **Online** | 当前 fallback（切块 → streaming） | 整章 streaming + agent 续写/修复 |

ChainEntry 扩展为两个维度：

```python
@dataclass
class ChainEntry:
    provider: str
    model: str
    mode: Literal["batch", "online"]        # transport 维度
    strategy: Literal["split", "whole"]      # content 维度（新增）
    retries: Optional[int] = None
```

配置示例：

```yaml
models:
  - provider: gemini
    model: gemini-3-pro
    mode: batch
    strategy: whole       # 先整章 batch，赌一把
  - provider: gemini
    model: gemini-3-pro
    mode: online
    strategy: whole       # batch 失败 → 在线 + agent 续写
  - provider: gemini
    model: gemini-2.5-flash
    mode: online
    strategy: split       # 长翻也救不了 → 切块翻译
```

Fallback 沿任一维度移动。chain 的表达力完整覆盖所有场景。

### 核心设计决策：Strategy 是外层，Transport 是内层

两个正交维度有明确的层次关系：

```
Strategy 层（外层）：决定处理模式
  whole → agent_loop(generate_fn)    # agent loop 内部每轮调用 generate_fn
  split → generate_fn() 一次         # 退化的 agent loop（一轮即完成）

Transport 层（内层）：决定每次 generate_fn 的派发方式
  batch  → submit + poll + fetch
  online → streaming
```

**`_process_whole` 不是 `_process_single` 的同级——它是更高层的东西。** `_process_single` 是 `split+online` 的特化实现，`_process_batch_as_unit` 是 `split+batch` 的特化实现。而 `_process_whole` 在 strategy 层，内部自行选择 transport：

```python
def _process_whole(self, unit, state, context):
    entry = state.get_current_entry()
    # transport 被封装在 generate_fn 里
    if entry.mode == "batch":
        generate_fn = lambda prompt: self._batch_generate(prompt, entry)
    else:
        generate_fn = lambda prompt: self._online_generate(prompt, entry)

    agent = self._create_agent(unit)
    result, _ = run_agent_loop(agent, generate_fn, unit.content)
    ...
```

Executor 的 dispatch 反映这个层次：

```python
if strategy == "whole":
    _process_whole(unit, transport=mode)     # whole 自己处理 transport 选择
elif mode == "batch":
    batch_queue → _process_batch_as_unit()   # split+batch 特化
else:
    _process_single()                        # split+online 特化
```

Split 模式的两个 transport 实现之所以分开，是因为 batch 和 online 在 split 路径里深度耦合了 hooks/screener/polling 等逻辑。Whole 模式没有这个包袱——transport 对 agent loop 完全透明，batch 和 online 的唯一区别是延迟和成本。

关键边界规则不变：**`_process_whole` 返回标准 ProcessResult**。主循环的 saver/tracker 代码一行不改。

```
主循环视角（不关心内部层次）：

  _process_single(unit)  →  ProcessResult    # split+online
  _process_batch(units)  →  ProcessResult[]  # split+batch
  _process_whole(unit)   →  ProcessResult    # whole（内部自选 transport）

三条路径的输出完全同构。
```

### 为什么不动 Hooks 系统

Split 模式的 hooks/screener/validators 是为**高并发小 unit** 优化的：

- **Screener 短路是核心性能优化**：几百个 .part unit，每个都跑昂贵的 LLM 验证不可接受。Screener（如 LineCountValidator）做快速初筛，通过即短路，跳过后续 final validators。这是精心设计的性能 tradeoff。
- **CompositeHooks 的二元 accept/reject 语义**完全匹配 split 模式的需求——小 unit 要么过要么不过，没有"续写"的概念。
- **`accepted=False` 触发 `_handle_failure()`** 的副作用（quota 减少、chain 突变、触发 splitting）对 split 模式完全正确——一次验证失败就是一次真正的失败。

把续写语义塞进 hooks 系统会引入以下不可调和的矛盾：
1. `CompositeHooks.post_process()` 构造新 HookResult 时丢弃扩展字段（`composite.py:207`）
2. `accepted=False` 立即触发 `_handle_failure()` 的 quota/chain 副作用——续写不是失败
3. Screener 短路跳过任何后置 agent validator——这是功能不是 bug
4. Agent 既是 validator（判断）又是 transformer（修改内容），但 hooks 流程是先 transform 再 validate

这些不是"需要修的 bug"，而是 split 模式的**设计特征**。`_process_whole` 绕过整套 hooks 系统，用自己内部的 agent loop 完成验证和修复。

### Executor 的 Strategy 分发

```python
# executor.py 主循环中的 dispatch（伪代码）
entry = state.get_current_entry()

if entry.strategy == "whole":
    # whole 模式：agent loop 内部自选 transport（batch/online）
    result = self._process_whole(unit, state, context)
elif entry.mode == "batch":
    # split + batch: 现有逻辑
    batch_queue.append(unit)
else:
    # split + online: 现有逻辑
    result = self._process_single(unit, state, context)
```

Strategy 判断在 transport 判断之前——这反映了 strategy 是外层、transport 是内层的层次关系。

`_process_whole` 内部包含完整的 agent loop——generate、validate、repair/continue、再 validate——最终返回一个 ProcessResult。续写永远不返回 `success=False`；只有真正的失败（agent reject、max continuation exceeded、网络错误）才暴露给 `_handle_failure`。

### PostAction：Agent 的输出协议

PostAction 是 agent loop 的**内部通信协议**，不暴露给 hooks 系统。

**设计原则**：PostAction 描述"agent 已经做完了什么"，不描述"该做什么操作"。Agent 内部的操作细节（Python 文本处理、LLM 语义判断、游标前进后退）是 agent 自己的事。

```python
@dataclass
class PostAction:
    type: Literal["repaired", "needs_continuation", "reject"]
    content: str    # agent 处理后的内容
```

三种结局：

- **repaired**：agent 已修好所有问题（删了幻觉行、修了 tag、补了括号……），`content` 是最终结果。
- **needs_continuation**：agent 验证了已有部分的忠实性并做了必要清理，但内容还没翻完。`content` 是已验证的部分。
- **reject**：内容不可挽回，走正常的失败路径。

PostAction 只在 `_process_whole` 和 `run_agent_loop` 内部流转。主循环和 hooks 系统永远看不到它。

### Truncation Detection 的关系

两种模式对截断的处理截然不同：

- **Split 模式**：truncation detected → reject → retry/split（现有行为，不变）
- **Whole 模式**：截断在 `_process_whole` 内部由 agent 处理为 `needs_continuation`，续写后继续。**截断不是 error，是正常的 continuation trigger。**

检测工具（n-gram detector、line count checker）可以共享——agent 可以用这些工具作为快速初筛，再做精确的语义判断。但检测后的**动作**完全不同：split 走 hooks reject，whole 走 agent loop continuation。

---

## Agent 的设计

### 三个场景，一套协议

Whole 模式需要服务三条不同的 workflow：

| Workflow | Agent 类型 | 诊断能力 | 修复能力 |
|----------|-----------|----------|----------|
| 全文翻译 | ChapterTranslationAgent | 幻觉检测、截断检测、n-gram 分析 | 截取忠实部分 |
| 压缩 HTML 翻译 | HTMLTranslationAgent | 行数对齐、tag 完整性、未翻译检测 | 删行、拆行、合行、补 tag |
| PDF 目录 JSON | JsonRefineAgent | JSON 语法、结构完整性 | 语法修复、括号补全 |

三者差异巨大——诊断逻辑不同、修复工具不同、续写 prompt 不同、join 策略不同。**继承不合适**（共享的基类逻辑太薄）。

**设计选择：Protocol + 组合**

```python
class WholeModeAgent(Protocol):
    """Whole 模式 agent 的通用协议。"""

    def validate_and_repair(self, original: str, output: str) -> PostAction:
        """验证输出，必要时修复。返回 PostAction。"""
        ...

    def build_continuation_prompt(self, original_prompt: Any, partial: str) -> Any:
        """基于已验证的部分输出，构建续写 prompt。"""
        ...

    def join_continuation(self, previous: str, continuation: str) -> str:
        """将续写结果拼接到已有输出上。"""
        ...
```

三个实现各自组合自己需要的工具（通过构造函数注入，不通过继承）：

```python
class JsonRefineAgent:
    def __init__(self, json_parser, structural_validator): ...

class HTMLTranslationAgent:
    def __init__(self, line_counter, tag_validator, llm_client): ...

class ChapterTranslationAgent:
    def __init__(self, ngram_detector, language_detector, llm_client): ...
```

n-gram detector 这种共享工具，谁需要谁注入。不共享的工具各自持有。

### 共享的续写循环

三条 workflow 共享同一个循环工具：

```python
def run_agent_loop(
    agent: WholeModeAgent,
    generate_fn: Callable,
    original: str,
    max_continuations: int = 5,
) -> Tuple[str, PostAction]:
    """通用的 agent 辅助生成循环。"""
    output = generate_fn()
    for _ in range(max_continuations):
        action = agent.validate_and_repair(original, output)
        if action.type == "repaired":
            return action.content, action
        elif action.type == "needs_continuation":
            prompt = agent.build_continuation_prompt(original_prompt, action.content)
            continuation = generate_fn(continuation_prompt=prompt)
            output = agent.join_continuation(action.content, continuation)
        else:  # reject
            raise AgentRejectionError(action)
    raise MaxContinuationsError(output)
```

### 集成方式：两种接入

Agent 不依赖 executor，不依赖 hook 系统，不依赖任何 workflow。它是一个 `(original, output) → PostAction` 的纯协议。

**接入 executor 的 workflow（全文翻译）**：

`_process_whole` 内部调用 `run_agent_loop`。Agent loop 的结果直接成为 ProcessResult 的内容。不经过 CompositeHooks。

```python
# executor.py 中 _process_whole 的核心逻辑（伪代码）
def _process_whole(self, unit, state, context):
    agent = self._create_agent_for_unit(unit)  # 工厂方法
    try:
        result_content, action = run_agent_loop(
            agent=agent,
            generate_fn=lambda **kw: self._call_llm(unit, state, **kw),
            original=unit.content,
        )
        return ProcessResult(success=True, content=result_content, ...)
    except AgentRejectionError:
        return ProcessResult(success=False, error_type="VALIDATION", ...)
    except MaxContinuationsError as e:
        # 用已有的最长输出作为 fallback（如果配置允许）
        return ProcessResult(success=False, content=e.partial, ...)
```

**不走 executor 的 workflow（HTML 翻译、refine）**：

直接调用，无需任何包装：

```python
# adaptive_pdf_call.py / html translator 中
agent = JsonRefineAgent(...)
result, action = run_agent_loop(
    agent=agent,
    generate_fn=lambda **kw: client.generate_content_stream(...),
    original=input_content,
)
```

同一个 agent，同一个循环，不同的调用点。耦合度为零。

当 HTML 翻译或 refine 未来迁移到 executor 架构时，`_process_whole` 直接用它们的 agent，无需适配层。

---

## 已解决的硬问题

在设计过程中，通过审查（含 codex 辅助）识别了 8 个硬问题。并行路径设计自然解决了所有问题：

| # | 问题 | 解法 |
|---|------|------|
| 1 | `CompositeHooks.post_process()` 丢字段 | `_process_whole` 不经过 CompositeHooks。字段丢弃是独立 bug，可单独修，但不是 whole mode 前置依赖 |
| 2 | `accepted=False` 触发 `_handle_failure()` 副作用 | 续写在 `_process_whole` 内部循环，永远不返回 `success=False`。只有 agent reject 才暴露给主循环 |
| 3 | Screener 短路跳过 agent | 不是问题。Screener 短路是 split 模式的核心性能优化。`_process_whole` 不走 hooks，不受 screener 影响 |
| 4 | 没有 per-unit strategy | `ChainEntry.strategy` + `_plan_chain_for_unit()` 根据 unit 特征选择初始 strategy |
| 5 | Whole→split fallback 没有持久化 | 在 pipeline 层用 `SplitManager.get_or_create_split()` 创建 `.part` 文件，不是 executor 层的 `.sub` |
| 6 | Truncation 分类为 `remove_current_model` | Whole 模式下截断是 `needs_continuation`（内部处理），不进入 error classification |
| 7 | Batch threshold ≥5 不适合 whole | 按 strategy 分 batch queue，whole 的 threshold=1 |
| 8 | Agent repair 后不过其他 validators | Repair 在 `_process_whole` 内部，repair/join 后重新调 `agent.validate_and_repair()`，不需要 hook validators |

### Per-Unit Strategy 选择（#4 详解）

当前 executor 初始化时，每个 unit 复制全局 model_chain（`executor.py:304-310`）。扩展为：

```python
# 替换 chain=self._model_chain 为：
chain = self._plan_chain_for_unit(unit, self._model_chain)
```

`_plan_chain_for_unit` 的逻辑：
- `.part` unit 或 `SplitType.PROACTIVE` → 强制 split（已经切过了）
- token count 超过阈值 → 优先 whole（大 unit 更需要 agent 辅助）
- 否则 → 遵循 chain 模板的默认 strategy

### Whole→Split Fallback（#5 详解）

Executor 层的 `.sub` 动态分裂是虚拟的（`DiskFirstSaver` 不追踪 `.sub`，不可恢复）。Whole→split fallback 应在 **pipeline 层** 实现：

```
Pipeline.process_all():
  1. 尝试 whole units（不做 proactive split）
  2. whole 失败的 base unit → SplitManager.get_or_create_split()
     → 创建持久化 .part 文件
     → split_history 写入 ProcessingTracker
  3. 对 .part units 调用 executor.execute() 第二轮
  4. 后续 --resume 时，tracker 已有 split_history，自然走 split
```

这复用了现有的 `pipeline_v2.py:232-255` batch validation retry 模式。

---

## 与现有系统的兼容性

### 不变的部分

| 组件 | 影响 |
|------|------|
| Executor 主循环 | 不变。主循环的 futures/pending/completed 逻辑无改动 |
| `_process_single` | **完全不变**。Split 路径的代码一行不改 |
| CompositeHooks / Screener / Validators | **完全不变**。Whole 路径不走 hooks |
| HookResult | **不变**。不加 action 字段，PostAction 是 agent loop 内部协议 |
| Work unit 定义 | 不变。Whole 模式的 unit 只是更大的 unit，数据结构相同 |
| ProcessResult | 不变。续写循环内部累积 tokens/duration，最终返回一个 ProcessResult |
| DiskFirstSaver | 不变。记录最终结果，续写过程的中间状态通过日志追踪 |
| ProcessingTracker | 不变。每个 unit 只记录最终完成/失败 |
| Error classification | Split 路径不变。Whole 路径用独立的（非 Strict）分类策略 |
| Proactive splitting | 不变。Split 策略的 unit 依然走 SplitManager |
| Batch polling | 不变。Batch + whole 组合中，polling 后 agent 验证结果 |

### 变化的部分

| 组件 | 变化 |
|------|------|
| ChainEntry | 新增 `strategy` 字段（默认 "split"，完全向后兼容） |
| Executor | 新增 `_process_whole` 方法 + strategy dispatch |
| Pipeline | whole unit 跳过 proactive split + whole→split fallback 逻辑 |
| Factory | 读取 config 中的 strategy 字段，传入 ChainEntry |

### 新增的部分

| 组件 | 位置 |
|------|------|
| `pdf2epub/core/whole/_protocol.py` | WholeModeAgent protocol + PostAction |
| `pdf2epub/core/whole/runner.py` | `run_agent_loop()` |
| `pdf2epub/core/whole/agents/json_refine.py` | JsonRefineAgent |
| `pdf2epub/core/whole/agents/html_translation.py` | HTMLTranslationAgent（后续） |
| `pdf2epub/core/whole/agents/chapter_translation.py` | ChapterTranslationAgent（后续） |

所有变化都是**增量的、向后兼容的**。不修改任何现有字段的语义，不删除任何现有代码路径。

---

## 实施计划

### Phase 1：Standalone Agent Loop（立即可做，不涉及 executor）

目标：修复百题争论 JSON 截断 bug。

1. 创建 `pdf2epub/core/whole/` 模块
   - `_protocol.py`：WholeModeAgent protocol、PostAction dataclass
   - `runner.py`：`run_agent_loop()` 函数
2. 实现 `JsonRefineAgent`
   - `validate_and_repair`：尝试 `json.loads()`，成功则 repaired，失败则分析截断位置
   - `build_continuation_prompt`：把已验证部分作为 prefix，让 LLM 续写
   - `join_continuation`：JSON-aware 拼接（找到续写的重叠点）
3. 在 `adaptive_pdf_call.py:531-551` 替换当前的 3 次重试循环为 `run_agent_loop(JsonRefineAgent(...), ...)`

**不改 executor、不改 hooks、不改 pipeline。** 纯增量。

### Phase 2：ChainEntry.strategy（最小 executor 变更）

1. `ChainEntry` 加 `strategy` 字段（默认 "split"）
2. Factory 读取 config 中的 strategy
3. Executor dispatch 逻辑：strategy 判断在 transport 判断之前

### Phase 3：`_process_whole` in Executor

1. 实现 `_process_whole`，内部调用 `run_agent_loop`，内部自选 transport
2. Pipeline 层 whole unit 跳过 proactive split
3. Pipeline 层 whole→split fallback via SplitManager

### Phase 4：更多 Agent 实现

1. HTMLTranslationAgent → 集成到 `translator.py`
2. ChapterTranslationAgent → 集成到 executor 的 `_process_whole`

### Phase 5：Agent Loop 成为最外层（终极形态）

在 Phase 1-4 中，agent loop 被各个调用方使用——`adaptive_pdf_call.py` 用它修 JSON，`_process_whole` 用它翻译章节，`translator.py` 用它翻译 HTML。但调用方仍然是当前的 executor/pipeline。

Phase 5 反转这个关系：**agent loop 成为最外层的主循环**。一个 `BookAgent` 通读全书、连续翻译，遇到问题自己修，实在不行才拆。Split executor 从顶层调度器降级为 agent 的一个可调用工具。

```
Phase 1-4 的架构：
  executor/pipeline（主循环）
    └── _process_whole → run_agent_loop（工具）

Phase 5 的架构：
  BookAgent.run_agent_loop（主循环）
    ├── generate → evaluate → continue/repair
    ├── 遇到困难段落 → 调用 split executor 作为工具
    └── split executor 内部有自己的并行调度
```

这不是重构——Phase 1-4 写的每一行代码都直接复用。`run_agent_loop` 是同一个函数，只是被调用的层级变了。底层基础设施（saver、tracker、persistence、LLMClient、ProcessResult）已经通过配置驱动（而非硬编码）实现了依赖注入，whole 和 split 都能无修改地使用。

Phase 5 的前提是单章 agent loop 已经在 Phase 3-4 中证明了可靠性。在那之前，不急。

---

## 开放问题

1. **Whole + batch 的续写路径**：`_process_whole` 内部的 `generate_fn` 封装了 transport。Agent loop 的每一轮续写都可以是 batch call（submit → poll → fetch），不需要切换到 online。延迟更高，但成本更低。
2. **续写的 token 计费**：一个 unit 可能经历 5 次续写调用，ProcessResult.output_tokens 应累加所有调用的 tokens。
3. **续写循环的超时**：单个 unit 的续写循环可能耗时很长。需要考虑 executor 层面的超时策略。
4. **Quota 系统**：续写调用不消耗 quota（续写是"还没完成"，不是"失败"）。只有 reject 消耗 quota。
5. **日志与可观测性**：续写循环的中间状态（每次续写的 content 长度、agent 的诊断结果）应写入日志，便于调试。不滥用 AttemptRecord.status="failed" 记录续写——续写不是失败。
6. **Agent 的模型选择**：agent 的 LLM 调用应使用廉价模型（Flash/Haiku），但需要足够聪明来做语义判断。这个平衡点需要实验确定。
7. **Whole 模式下的 context injection**：whole 模式的 unit 是否需要前后文注入？如果全文已作为输入，可能不需要。但分卷翻译时可能需要。
