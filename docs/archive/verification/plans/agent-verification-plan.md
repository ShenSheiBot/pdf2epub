# Agent-based Verification Framework 实现计划

## 目标

创建通用的agent-based verification framework，替代当前的truncation detector，用于：
1. Batch polish验证
2. Translation验证
3. 未来任何需要验证处理结果的场景

## 核心设计

### 1. 通用验证工具（Tools）

`pdf2epub/processors/utils/verification_tools.py`

提供agent使用的工具集：
- `read_segment(file_key, source, start, length)` - 读取任意片段
- `get_stats(file_key)` - 获取统计信息
- `search_content(file_key, pattern, source)` - 搜索内容
- `compare_segments(file_key, position)` - 对比原文和处理后的片段

类似refine的`read_page`工具，但更通用。

### 2. Agent Verifier

`pdf2epub/processors/utils/agent_verifier.py`

核心验证类：
- 使用Pydantic AI框架（参考boundary_agent.py）
- 提供工具给agent
- Agent自主决定：读哪些片段、读多少、如何判断
- 支持批量验证（多个文件一次性检查）

任务特定配置：
- `PolishVerificationAgent` - polish专用prompt和判断标准
- `TranslationVerificationAgent` - translation专用prompt和判断标准

### 3. 集成到现有流程

#### Batch Polisher
- 移除当前的`CompositeTruncationDetector`
- Batch完成后，先保存所有结果
- 用n-gram初筛可疑文件（unique_recall < 0.6）
- Agent批量验证可疑文件
- 只对确认truncated的重试

#### Translator
- 可选替换`LLMTruncationDetector`
- 保留原有同步验证作为fallback
- 添加agent验证选项（通过配置启用）

## 实现步骤

### Phase 1: 创建通用工具和Agent框架

文件：
- `pdf2epub/processors/utils/verification_tools.py`
- `pdf2epub/processors/utils/agent_verifier.py`

功能：
1. `VerificationTools` 类 - 提供read/stats/search工具
2. `AgentVerifier` 基类 - 使用Pydantic AI
3. Agent工具函数装饰器（@tool）
4. 模型配置（优先anthropic haiku，fallback poe）

### Phase 2: Polish验证专用Agent

在 `agent_verifier.py` 中添加：
- `PolishVerificationAgent` - 继承AgentVerifier
- Polish专用system prompt
- 判断标准：格式转换 vs 真正truncation
- 批量验证接口

### Phase 3: 集成到Batch Polisher

修改 `pdf2epub/processors/batch_polisher.py`：
1. 移除实时truncation验证
2. Batch完成后保存所有结果
3. N-gram初筛
4. 调用PolishVerificationAgent批量验证
5. 只重试确认有问题的

### Phase 4: Translation验证专用Agent（可选）

在 `agent_verifier.py` 中添加：
- `TranslationVerificationAgent`
- Translation专用system prompt
- 可选启用（通过config配置）

## 配置

在 `config.yaml` 添加：

```yaml
verification:
  # Agent-based验证配置
  agent_based: true  # 启用agent验证

  # N-gram初筛阈值（低于此值才交给agent）
  ngram_threshold: 0.6

  # Agent模型配置
  agent_models:
    - provider: anthropic
      model: claude-haiku-4-5-20251001
    - provider: poe  # fallback
      model: Gemini-2.5-Flash

  # 批量验证配置
  batch_size: 20  # 一次最多验证多少文件
```

## 成本优势

当前方案（11个文件）：
- N-gram判断：11个失败
- LLM验证：11 × 平均5000 tokens × 3次重试 = ~165k tokens
- 结果：10个误判，浪费API费用

Agent方案（11个文件）：
- N-gram初筛：11个可疑
- Agent验证：11文件 × 采样~2k字符/文件 = ~30k tokens × 1次
- 结果：准确判断，省80%成本

## 验收标准

1. ✅ Agent能正确识别格式转换（表格→列表、去重复）
2. ✅ Agent能正确识别真正的truncation
3. ✅ 成本降低：减少不必要的重试
4. ✅ 准确率提升：降低误判率
5. ✅ 代码复用：polish和translate用同一套工具
6. ✅ 向后兼容：保留旧的detector作为fallback

## 风险和缓解

风险：
- Agent判断错误 → 缓解：可以配置关闭，fallback到旧方案
- Agent成本太高 → 缓解：只对n-gram初筛的可疑文件调用agent
- Agent太慢 → 缓解：批量验证，一次处理多个文件

## 时间估计

不估计时间，按步骤实现。

## 参考实现

- `pdf2epub/refine/boundary_agent.py` - Pydantic AI agent + tools模式
- `pdf2epub/processors/utils/truncation/` - 现有truncation detector
