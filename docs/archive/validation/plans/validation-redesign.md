> **OUTDATED**: 本文档已被 `executor-design-v2.md` 取代。验证体系已合并到 hooks。

# Validation 架构重设计

## 问题回顾

之前的设计错误：
1. 强制 Phase 1 → Phase 2 顺序
2. 把 Batch 的 AgentVerifier 硬改成单文件 adapter
3. 每次遇到问题就打补丁，而不是质疑架构本身
4. 没有识别问题的正交维度

---

## 正交维度

### 维度 1: Validator 类型

| 类型 | 接口 | 运行时机 |
|------|------|---------|
| **Individual** | `validate(original, processed, key) -> ValidationResult` | 每个文件处理完后立即运行 |
| **Batch** | `validate_batch(files: Dict[str, VerificationFile]) -> Dict[str, ValidationResult]` | 所有文件处理完后批量运行 |

### 维度 2: Validator 角色

| 角色 | 通过 | 不通过 | 逻辑 |
|------|------|--------|------|
| **Screener** | 直接通过，结束 | 不确定，继续后续验证 | OR |
| **Final** | 需全部 final 都通过 | 直接不通过，触发重试 | AND |

### 维度 3: 处理模式兼容性

| 模式 | Individual Validator | Batch Validator |
|------|---------------------|-----------------|
| Sequential | ✓ 支持 | ✓ 支持 |
| Parallel | ✓ 支持 | ✓ 支持 |
| Batch Processing (Gemini Batch API) | ✗ 不支持 | ✓ 支持 |

---

## 核心设计原则

### 1. Role 是配置层面的，不是 Validator 属性

同一个 Validator 可以配成 screener 或 final：

```python
# 同一个 LengthValidator
(LengthValidator(), "screener")  # 不通过 = 不确定
(LengthValidator(), "final")     # 不通过 = 不通过
```

**实践建议**：基于正则/代码的方法推荐作为 screener。因为如果 OCR 出错导致内容重复 20 遍，polish 去重后只保留一遍，基于代码的检测会误判为截断。

### 2. 任意组合

- 两个阶段都可以没有 screener
- Batch 阶段也可以没有 final
- Individual validators 和 Batch validators 可独立配置
- 可以只用 Individual，只用 Batch，或两者都用

### 3. Screener 回落规则

**核心原则**：整个系统必须有至少一个 final 判断点。如果配置中完全没有 final，最后一个 screener 回落成 final。

| 场景 | Individual Screener | Batch Screener |
|------|---------------------|----------------|
| **有 Batch 阶段** | 不回落（Batch 负责最终判断） | 如果 Batch 没有 final，最后一个回落 |
| **无 Batch 阶段** | 如果 Individual 没有 final，最后一个回落 | N/A |

**判断逻辑**：
1. 如果 Batch validators 非空 → Individual screener 永不回落
2. 如果 Batch validators 为空 + Individual 没有 final → Individual 最后一个 screener 回落
3. 如果 Batch 没有 final → Batch 最后一个 screener 回落

这样保证无论怎么配置，系统都有最终判断能力。

---

## 执行流程

### 单阶段内的执行顺序

1. **先跑所有 Screener**（按配置顺序）
   - 任一通过 → 直接通过，短路退出
   - 全部不通过 → 继续

2. **再跑所有 Final**（按配置顺序）
   - 全部通过 → 通过
   - 任一不通过 → 不通过，短路退出

### 完整流程

```
for unit in units:
    1. 处理（翻译/润色）

    2. Individual Validation
       - 跑 screeners: 任一通过 → 标记 individual_passed
       - 跑 finals: 全部通过 → 标记 individual_passed
       - 不通过 → 立即重试（retry + model chain）

    3. Context Injection（如果有 context_ready 的 validator 通过）

所有 units 完成后:
    4. Batch Validation
       - 跑 screeners: 每个文件独立判断，通过的直接通过
       - 跑 finals: 剩余文件全部通过才通过
       - 不通过 → 批量重试
```

### 流程示例

**示例 1: Individual screen + Batch final**
```
Individual: [(LengthValidator, screener, context_ready=True)]
Batch: [(AgentVerifier, final)]

文件 A 处理完:
1. Individual Length: 通过 → A 的 raw 可用于 context injection
2. (继续处理其他文件)

所有文件完成后:
3. Batch Agent: A 通过 → A 最终通过
              B 不通过 → B 触发 Batch 重试
```

**示例 2: Individual final 不通过立即重试**
```
Individual: [(LengthValidator, final)]
Batch: [(AgentVerifier, final)]

文件 A 处理完:
1. Individual Length: 不通过 → 立即重试 A（不进入 Batch）
```

**示例 3: Batch 没有 final，screener 回落**
```
Individual: []
Batch: [(FastChecker, screener), (SlowChecker, screener)]

所有文件完成后:
1. Batch FastChecker (screener): A 通过 → A 直接通过
                                 B 不通过 → 继续
2. Batch SlowChecker (回落为 final): B 通过 → B 通过
                                     B 不通过 → B 不通过
```

---

## 失败处理

### 失败分类

| 类型 | 原因 | 处理策略 | Quota |
|------|------|---------|-------|
| **Safety** | 内容被模型拒绝 | 跳过原模型，切换到下一个模型 | - |
| **网络/随机** | API 错误、超时、随机失败 | 同模型重试 | `network_retry_quota` (默认 3) |
| **Validation** | 输出验证失败（截断、长度异常等） | 同模型重试 | `validation_retry_quota` (默认 1) |

三种错误类型有独立的 quota，互不影响。

已有基础设施：
- `ErrorType` 枚举
- `classify_error()` 函数
- `ModelChain.ERROR_HANDLING_MATRIX`

### Individual 失败处理

- Validation 失败 → 同模型立即重试（不超过 `validation_retry_quota`）
- 网络/随机失败 → 同模型重试（不超过 `network_retry_quota`）
- Safety 失败 → 切换到下一个模型

### Batch 失败处理

- 批量重试
- 如果失败数量 > threshold → 提交下一个 batch job
- 如果失败数量 <= threshold → 用 online 重试
- **threshold 默认为 5，可配置**

### Batch 重试后的验证

- **只对失败文件重新验证**
- 已通过的文件不重跑

---

## Context Injection

### context_ready 字段

```python
@dataclass
class ValidatorConfig:
    validator: Validator
    role: Literal["screener", "final"]
    context_ready: bool = False  # 通过时，raw 结果可用于 context injection
```

### 逻辑

- 任何 validator 通过 + `context_ready=True` → raw 结果立即可用于下游 context injection
- **不需要等所有 validation 完成**
- 最终 validation 失败也不影响已经注入的 context

### 示例

```
Individual: [(LengthValidator, screener, context_ready=True)]
Batch: [(AgentVerifier, final)]

Sequential 模式下:
1. 文件 A 处理完，Length 通过 → A 的 raw 立即可用于 B 的 context
2. 文件 B 处理时可以用 A 的 context
3. 后续 Batch AgentVerifier 判定 A 失败 → A 标记失败，但 B 已经用了 A 的 context
```

---

## Tracker 记录

每次 validator 判断都必须记录：

```python
@dataclass
class ValidationRecord:
    timestamp: float
    validator_name: str
    role: Literal["screener", "final"]
    result: ValidationResult  # is_valid, reason, confidence
    context_ready_triggered: bool  # context_ready 是否触发
```

所有操作都要记录，所有内容都可观测。

---

## 数据结构

### ValidatorConfig

```python
@dataclass
class ValidatorConfig:
    validator: Union[IndividualValidator, BatchValidator]
    role: Literal["screener", "final"]
    context_ready: bool = False
```

### Protocol 定义

```python
class IndividualValidator(Protocol):
    """单文件验证器"""
    @property
    def name(self) -> str: ...

    def validate(
        self,
        original: str,
        processed: str,
        file_key: str
    ) -> ValidationResult: ...


class BatchValidator(Protocol):
    """批量验证器"""
    @property
    def name(self) -> str: ...

    def validate_batch(
        self,
        files: Dict[str, VerificationFile]
    ) -> Dict[str, ValidationResult]: ...
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    key: str
    is_valid: bool
    reason: str
    confidence: Literal["high", "medium", "low"]
```

---

## 现有 Validators 归类

### Individual Validators

| Validator | 推荐 Role | 说明 |
|-----------|----------|------|
| LengthValidator | screener | 长度比例检查 |
| TruncationValidatorAdapter(NGram) | screener | 同语言 n-gram 检测 |
| TruncationValidatorAdapter(LLM) | screener | 跨语言 LLM 检测 |

### Batch Validators

| Validator | 推荐 Role | 说明 |
|-----------|----------|------|
| PolishVerificationAgent | final | Agent 验证润色结果 |
| TranslationVerificationAgent | final | Agent 验证翻译结果 |

---

## 特殊情况处理

### 空 Validator 列表

| 配置 | 行为 |
|------|------|
| Individual = [] | 直接通过 Individual 阶段 |
| Batch = [] | 直接通过 Batch 阶段 |
| 两者都为空 | 无验证，直接通过 |

### Batch Processing 警告

如果配置了 Individual validators 但使用 BatchPipeline（Gemini Batch API），应该：
- 警告用户 Individual validators 将被跳过
- 或者直接报错

---

## ValidationPipeline 的命运

**删除 ValidationPipeline**。

原因：
- Individual validation 和处理+重试紧密耦合，必须在 ProcessingPipeline 循环内
- Batch validation 也和重试策略紧密耦合
- ValidationPipeline 只剩存储 validators 列表的功能，没有独立价值

验证逻辑直接集成到 ProcessingPipeline。

---

## ProcessingPipeline 改造

```python
class ProcessingPipeline:
    def __init__(
        self,
        processor: ProcessorProtocol,
        # Validators
        individual_validators: List[ValidatorConfig] = None,
        batch_validators: List[ValidatorConfig] = None,
        # Batch 重试阈值
        batch_retry_threshold: int = 5,
        # ... 其他参数
    ):
        self._individual_validators = individual_validators or []
        self._batch_validators = batch_validators or []
        self._batch_retry_threshold = batch_retry_threshold
```

---

## 默认配置

```python
# 默认: parallel + batch validation
default_individual_validators = []
default_batch_validators = [
    ValidatorConfig(
        validator=TranslationVerificationAgent(),  # 或 PolishVerificationAgent
        role="final",
        context_ready=False
    )
]
```

---

## 实现清单

1. [x] 定义 `IndividualValidator` 和 `BatchValidator` Protocol - `_protocol.py`
2. [x] 定义 `ValidatorConfig` dataclass - `_protocol.py`
3. [x] 定义 `ValidationRecord` for tracker - `_protocol.py`
4. [x] 修改 ProcessingPipeline 接收两种 validators - `pipeline.py`
5. [x] 实现 Individual validation + 重试逻辑 - `validation.py` + `pipeline.py`
6. [x] 实现 Batch validation + 重试逻辑 - `validation.py` + `pipeline.py`
7. [x] 实现 screener/final 执行逻辑（含回落）- `validation.py`
8. [x] 实现 context_ready 逻辑 - `validation.py` + `pipeline.py`
9. [x] 更新 Tracker 记录 validation 结果 - `tracking/tracker.py`
10. [x] 删除 ValidationPipeline，替换为 IndividualValidationRunner/BatchValidationRunner
11. [x] 更新 factory.py - 添加 `create_default_individual_validators`, `create_default_batch_validators`
12. [x] 更新现有 validators 实现 Protocol - `adapters.py`, `agent.py`
13. [x] BatchPipeline 不支持 Individual validators（已更新警告注释）
