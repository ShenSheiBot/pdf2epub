# Novel Translation Pipeline (translate-novel) — v2

## 背景

日语轻小说翻译专用 pipeline。核心假设：
- 翻译模型（murasaki-14b）是纯文本模型，4096 token context（vLLM 部署限制）
- 术语一致性对轻小说极为重要（角色名、专有名词）
- 必须按阅读顺序翻译（术语表增量维护）

**v2 变更**：去掉 agent，改用 Haiku 每章生成术语表 + 现有 validator 基础设施校验。

---

## 架构总览

```
Input EPUB
    ↓
[Extract] — NovelExtractor: HTML → 纯文本（ruby → 括号，img → [Image: src]）
    ↓
novel_units/           ← 每个 spine item 一个 .txt 文件
    ↓
[Glossary + Translate per Chapter] — 按 spine 顺序，逐章处理
    │
    │  每章流程：
    │  1. 调 Haiku 生成本章术语表（输入：本章全文 + 上章术语 + 跨章术语）
    │  2. 滑动窗口翻译（multi-turn format）
    │  3. finish_reason 检查 + LineCountValidator screener
    │
    ↓
translated_novel/      ← 翻译后的纯文本
    ↓
[Build EPUB] — 纯文本 → 简单 HTML → 打包 EPUB
    ↓
output.epub
```

---

## 核心组件

### 1. NovelExtractor（已实现，不变）

`pdf2epub/html_translation/novel_extractor.py`

### 2. Glossary Generator（新增，替代 agent）

#### 术语表格式

每行一个条目：
```
【加賀】= 加贺 - 亮的女同学
【宮崎(みやざき)】= 宫崎 - 女主角，只能用手机打十个字
```

**ID 提取**：`【...】` 内的日语原文。用于跨章去重——每章生成新术语表时，只传入 ID 在本章 exact string match 出现的已知条目。

#### 每章生成逻辑

```python
def _generate_glossary(self, chapter_text, prev_glossary, accumulated):
    # 1. 筛选本章出现的已知术语
    relevant = {id: entry for id, entry in accumulated.items()
                if id in chapter_text}

    # 2. 调 Haiku 生成
    glossary = self._call_glossary_model(chapter_text, prev_glossary, relevant)

    # 3. 超 1000 token → 再调一次压缩，最多 3 次
    for _ in range(3):
        if self._count_tokens(glossary) <= self.glossary_max_tokens:
            break
        glossary = self._call_compress_model(glossary)
    else:
        raise GlossaryOverflowError(f"术语表 {self._count_tokens(glossary)} tokens > {self.glossary_max_tokens}")

    # 4. 更新 accumulated（ID 去重）
    for id, entry in parse_glossary_entries(glossary):
        accumulated[id] = entry

    return glossary
```

#### Haiku 调用

使用 `AnthropicClient`（`pdf2epub/utils/network_utils.py` 已有），通过 config 中 `novel.agent` 配置的 provider/model。

**熔断**：API 失败 3 次（`AnthropicClient` 内置 retry 耗尽后抛异常）→ 停止整个翻译。

#### Prompt

**生成 prompt**：
```
你是日语轻小说术语提取器。根据章节内容生成精简术语表。

格式：每行一个条目
【日语原文】= 中文翻译 - 简要说明（一句话）

规则：
- 只收录重要角色、地名、专有名词
- 不收录普通词汇和一次性路人
- 控制在千字以内
- 参考已知术语保持翻译一致

上一章术语表：
{prev_glossary}

本章中出现的已知术语：
{relevant_entries}

本章内容：
{chapter_text}
```

**压缩 prompt**：
```
以下术语表过长，请压缩到 1000 tokens 以内。保留最重要的角色和术语，删除次要条目。
保持格式：【日语】= 中文 - 说明

{glossary}
```

### 3. NovelTranslator（重写）

`pdf2epub/html_translation/novel_translator.py`

#### 核心变更

| v1 | v2 |
|----|-----|
| Agent 维护术语表 | Haiku 每章生成术语表 |
| Agent 判断截断/cursor | `finish_reason` + `LineCountValidator` screener |
| Agent 检查图片 | 删除 |
| Multi-turn bug（术语重复） | 术语只在 system prompt |
| 依赖 `run_agent_loop_sync` | 不依赖 agent runner |

#### Sliding Window

```python
def _translate_chapter(self, unit, state):
    lines = read_source_lines(unit)

    # 1. 生成术语表
    glossary = self._generate_glossary(
        '\n'.join(lines), state.prev_chapter_glossary, state.accumulated_glossary)

    # 2. 滑动窗口翻译
    cursor = state.cursor
    translated_lines = load_partial_if_resume()

    while cursor < len(lines):
        chunk, chunk_end = take_chunk(lines, cursor)
        messages = build_messages(glossary, recent_ja, recent_zh, chunk)
        response = client.generate_content(messages=messages, ...)
        translated = strip_thinking(response)

        # 截断处理
        if client._last_finish_reason == 'length':
            translated = trim_last_line(translated)

        translated_lines.extend(parsed_lines)
        save_progress()
        cursor = chunk_end

    # 3. 整章校验（复用 LineCountValidator）
    # 如果行数差距太大，log warning（不阻塞，因为翻译可以合并段落）
```

#### Multi-turn Message Format（修复版）

```python
# 术语表放 system prompt，不重复
system = f"{NOVEL_TRANSLATE_SYSTEM}\n\n術語表：\n{glossary}" if glossary else NOVEL_TRANSLATE_SYSTEM

messages = [{"role": "system", "content": system}]

if has_context:
    messages.append({"role": "user", "content": f"请翻译以下日语：\n{recent_ja}"})
    messages.append({"role": "assistant", "content": recent_zh})
    messages.append({"role": "user", "content": f"请继续翻译：\n{chunk}"})
else:
    messages.append({"role": "user", "content": f"请翻译以下日语：\n{chunk}"})
```

#### 校验（复用现有基础设施）

使用 `LineCountValidator`（`pdf2epub/core/hooks/validators.py`，screener 模式）：
- 每个 chunk 翻译后：`finish_reason == 'length'` → 去掉最后不完整行
- 整章翻译完后：`LineCountValidator.validate(key, original, result)` 做 screener 检查
- Screener 不阻塞，只 log warning（轻小说翻译允许段落合并/拆分）

不需要重新实现任何校验逻辑。

#### State 管理

```python
@dataclass
class NovelState:
    current_unit_index: int = 0
    cursor: int = 0
    prev_chapter_glossary: str = ""
    accumulated_glossary: dict = field(default_factory=dict)  # {id: entry_line}
    completed_units: list = field(default_factory=list)
```

- 每 chunk 保存 cursor
- 每章保存 glossary
- `--resume` 从 state 恢复
- KeyboardInterrupt → 保存当前 state

### 4. NovelBuilder（已实现，不变）

`pdf2epub/html_translation/novel_builder.py`

### 5. CLI（已实现，微调）

去掉 agent 相关 import，其余不变。

---

## 代码修改清单

### 重写

| 文件 | 变更 |
|------|------|
| `novel_translator.py` | 去掉 agent，加 glossary generator，修复 multi-turn，复用 LineCountValidator |
| `novel_prompts.py` | 去掉 agent prompt，加 glossary 生成/压缩 prompt |

### 删除

| 文件 | 原因 |
|------|------|
| `core/whole/prompts/novel_agent.py` | 不再使用 agent |

### 不变

| 文件 | 原因 |
|------|------|
| `novel_extractor.py` | 已验证 |
| `novel_builder.py` | 已验证 |
| `network_utils.py` | `OpenAIClient` (extra_body/messages) + `AnthropicClient` 已就绪 |
| `cli.py` | 接口不变，微调 import |
| `core/hooks/validators.py` | `LineCountValidator` 直接复用 |

---

## 熔断策略

| 场景 | 行为 |
|------|------|
| 术语表生成 API 失败（retry 耗尽）| 抛异常，停止翻译 |
| 术语表压缩 3 次仍超限 | 抛异常，停止翻译 |
| 翻译 API 失败（retry 耗尽）| 抛异常，停止翻译 |
| KeyboardInterrupt | 保存 state，安全退出 |

---

## Token 预算（4096 limit）

| 部分 | Token 估算 |
|------|-----------|
| System prompt（含术语表） | ~600-1100 |
| 上文 5 行 multi-turn（user+bot） | ~500 |
| 待翻译 chunk | ~800 |
| `<think>` 输出 | ~300 |
| 翻译输出 | ~800 |
| **总计** | ~3000-3500 |
| **余量** | ~600 |

---

## 实现顺序

1. 重写 `novel_prompts.py`：去掉 agent prompt，加 glossary 生成/压缩 prompt
2. 重写 `novel_translator.py`：
   - 新增 `_generate_glossary()` / `_compress_glossary()` / `_extract_glossary_ids()`
   - 初始化 `AnthropicClient` for glossary（复用 network_utils 现有类）
   - 初始化 `LineCountValidator` for screener（复用 hooks/validators 现有类）
   - 重写 `_translate_chapter()`：术语表生成 → 滑动窗口
   - 修复 multi-turn：术语表只在 system prompt
   - KeyboardInterrupt 处理
   - NovelState 加 `prev_chapter_glossary` + `accumulated_glossary`
3. 删除 `core/whole/prompts/novel_agent.py`
4. CLI 微调 import
5. 端到端翻译全书
