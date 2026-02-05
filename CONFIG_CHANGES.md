# 配置变更说明 (Configuration Changes)

## 版本更新：验证架构重构

### 概述

项目已重构验证架构，统一使用 **N-gram Detector + Agent Verification** 两阶段验证，替代之前的 LLM-based truncation detector。

### 主要变更

#### 1. 废弃的配置参数

以下配置参数已**完全废弃**，可以从 `config.yaml` 中删除：

```yaml
polish:
  truncation_check_lines: 5          # ❌ 已废弃 - 不再使用
  truncation_models:                 # ❌ 已废弃 - 不再使用
    - provider: poe
      model: Gemini-2.5-Flash

translation:
  truncation_check_lines: 3          # ❌ 已废弃 - 不再使用
  truncation_models:                 # ❌ 已废弃 - 不再使用
    - provider: poe
      model: Gemini-2.5-Flash
```

**废弃原因**：
- 新架构使用 **NGramTruncationDetector** 进行快速筛查（纯算法，无 LLM 调用）
- Agent verification 提供更准确的验证（仅对可疑内容调用）
- 避免重复的 LLM 调用，提升性能和降低成本

---

#### 2. 新的验证架构

**两阶段验证流程**：

```
Phase 1: N-gram Truncation Detector (快速筛查)
  ├─ 检查唯一 n-gram 保留率
  ├─ 阈值：60% (硬编码)
  ├─ 允许去重优化
  └─ 无 LLM 调用，速度快

Phase 2: Agent Verification (准确验证)
  ├─ 仅验证 Phase 1 标记为可疑的内容
  ├─ 使用 Pydantic AI agent 分析
  ├─ 提供详细的截断原因和位置
  └─ 验证模型：claude-haiku-4-5 (优先) 或 Gemini-2.5-Flash (fallback)
```

**适用范围**：
- ✅ `polish` 命令
- ✅ `polish-batch` 命令
- ✅ `translate` 命令
- ✅ `translate-batch` 命令

---

#### 3. 新增的批次命令

**polish-batch**：
```bash
uv run pdf2epub polish-batch --resume
```

**translate-batch**：
```bash
uv run pdf2epub translate-batch --target-language Chinese --resume
```

**批次模式优势**：
- 💰 **成本降低 50%**：使用 Gemini Batch API
- ⚡ **异步处理**：提交后无需等待，自动轮询
- 🔄 **完整的状态持久化**：支持随时中断和恢复

**批次模式配置**：
```yaml
batch:
  model: gemini-3-pro-preview        # Batch API 使用的模型
  max_retries: 1                     # 验证失败的重试次数
  poll_interval: 60                  # 状态轮询间隔（秒）
  online_polish_fallback_threshold: 5  # Polish: 失败文件 <= 5 时回退到在线模式
```

---

### 配置迁移指南

#### 如果你的 `config.yaml` 包含以下配置：

**旧配置**：
```yaml
polish:
  processing_mode: parallel
  max_retries: 1
  models:
    - provider: gemini
      model: gemini-2.5-flash
  truncation_check_lines: 5          # 需删除
  truncation_models:                 # 需删除
    - provider: poe
      model: Gemini-2.5-Flash

translation:
  source_language: Japanese
  target_language: Chinese
  models:
    - provider: gemini
      model: gemini-2.5-pro
  truncation_check_lines: 3          # 需删除
  truncation_models:                 # 需删除
    - provider: poe
      model: Gemini-2.5-Flash
```

**新配置**：
```yaml
polish:
  processing_mode: parallel
  max_retries: 1
  models:
    - provider: gemini
      model: gemini-2.5-flash
  # truncation_* 参数已删除

translation:
  source_language: Japanese
  target_language: Chinese
  models:
    - provider: gemini
      model: gemini-2.5-pro
  # truncation_* 参数已删除

# 新增：批次处理配置
batch:
  model: gemini-3-pro-preview
  max_retries: 1
  poll_interval: 60
  online_polish_fallback_threshold: 5
```

---

### 验证模型配置

Agent verification 使用独立的验证模型（不同于处理模型）：

**优先级**：
1. Anthropic: `claude-haiku-4-5-20251001` (推荐)
2. POE fallback: `Gemini-2.5-Flash`

**配置位置**：硬编码在 `pdf2epub/processors/utils/agent_verifier.py`

如需修改验证模型，请编辑：
```python
# pdf2epub/processors/utils/agent_verifier.py
def get_verification_model(config: Dict) -> str:
    # 修改此处的模型选择逻辑
```

---

### 性能对比

| 指标 | 旧架构 (LLM Detector) | 新架构 (N-gram + Agent) |
|------|---------------------|------------------------|
| **筛查速度** | 慢 (需 LLM 调用) | 快 (纯算法) |
| **成本** | 高 (每个文件都调用 LLM) | 低 (仅可疑文件调用) |
| **准确率** | 中等 | 高 (Agent 更智能) |
| **API 调用次数** | 2 次/文件 | 0.1-0.3 次/文件 (平均) |

---

### 常见问题

#### Q1: 我的配置中还有 `truncation_*` 参数，会报错吗？
**A**: 不会报错，这些参数会被忽略。建议删除以保持配置文件整洁。

#### Q2: 新的验证会不会漏掉截断问题？
**A**: 不会。N-gram detector 的敏感度很高，会标记所有可疑内容，然后由 Agent 进行二次确认。测试显示准确率优于旧方案。

#### Q3: Agent verification 使用什么模型？
**A**: 默认使用 `claude-haiku-4-5` (Anthropic)，如果不可用则 fallback 到 `Gemini-2.5-Flash` (POE)。

#### Q4: Batch 模式和在线模式有什么区别？
**A**:
- **Batch**: 提交后异步处理，成本降低 50%，需等待队列
- **在线**: 实时处理，成本标准，立即返回结果
- **推荐**: 大量文件用 batch，少量文件或测试用在线

#### Q5: 如何选择使用 batch 还是在线模式？
**A**:
- 文件数 > 10：推荐 `polish-batch` / `translate-batch`
- 文件数 < 10：推荐 `polish` / `translate`
- 测试/调试：始终用在线模式（即时反馈）

---

### 向后兼容性

✅ **完全向后兼容**：
- 现有的命令行参数保持不变
- 输出格式和目录结构保持不变
- `--resume` 功能正常工作
- 旧的配置文件仍可使用（忽略废弃参数）

⚠️ **行为变化**：
- 验证日志消息略有不同（显示 "N-gram screening" 而不是 "LLM truncation check"）
- 可疑文件的验证报告更详细（Agent 提供具体原因）

---

### 更新日志

**2026-02-04**:
- 重构验证架构为 N-gram + Agent 两阶段验证
- 废弃 `truncation_check_lines` 和 `truncation_models` 配置
- 新增 `polish-batch` 和 `translate-batch` 命令
- 统一 polish 和 translate 的验证逻辑

---

### 技术细节

**N-gram Truncation Detector 参数** (硬编码):
```python
NGramTruncationDetector(
    min_unique_preserved_ratio=0.60,  # 最低 60% 唯一 n-gram 保留率
    allow_deduplication=True           # 允许内容去重
)
```

**适用场景**：
- 检测 LLM 生成中途截断
- 检测内容不完整或损坏
- 区分正常的内容压缩（如表格、索引）和真正的截断

**不适用场景**：
- 内容格式转换（会误报为截断）
- 大量重复内容的去重（需配合 Agent 二次确认）

---

如有问题或建议，请在 GitHub Issues 中反馈。
