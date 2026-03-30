# Novel Translation Pipeline v3

## v2 问题清单

| 问题 | 根因 |
|------|------|
| 每章术语表未保存，无法 debug | 只保存最后一章术语表到 glossary.txt |
| 全局术语表含备注、#标题等垃圾 | 存的是 Haiku 原始输出，未过滤为纯【】条目 |
| `<think>` 残留混入翻译 | `strip_thinking` 用两个 regex，处理不了无配对 `<think>` 的 `</think>` |
| 被污染的 assistant 上下文恶性循环 | `recent_zh` 取自 `translated_lines`，think 残留被喂回模型 |
| 没翻译 metadata/TOC | `novel_builder.py` 从零写，未复用现有 builder |
| EPUB 标题用 config title | 未翻译书名 |
| TOC 只有 "Chapter 1/2/3" 占位符 | 未从原 EPUB 提取真实章节名 |
| 目录页换行丢失 | `_extract_text_recursive` 将 block 元素连成一行 |
| 章节标题与 TOC 翻译不一致 | 各自独立翻译 |

---

## v3 架构

```
Input EPUB
    ↓
[Extract] — 修复后的 NovelExtractor（block 元素正确换行）
    ↓
novel_units/*.txt       ← 一行一段，block 间有换行
    ↓
[TOC + Glossary] — 从 EPUBParser 提取 TOC 条目
    │                  每章标题以【】格式注入术语表
    ↓
[Translate per Chapter] — 按 spine 顺序
    │  每章：
    │  1. Haiku 生成术语表（含章节标题映射）
    │  2. 滑动窗口翻译（multi-turn）
    │  3. 保存本章术语表到 logs/
    ↓
translated_novel/*.txt
    ↓
[Convert txt → xhtml] — 纯文本转 XHTML（用原 XHTML 的 <head> 做模板）
    ↓
translated_novel/*.xhtml
    ↓
[Build EPUB] — 复用 HTMLEpubBuilder
    │  1. 提取原 EPUB
    │  2. 替换 XHTML 文件（_replace_xhtml_files）
    │  3. 更新 OPF/NCX/NAV（_update_content_opf/toc_ncx/nav_xhtml）
    │  4. 打包
    ↓
output.epub
```

---

## 逐项修复

### 1. Extractor 换行修复

**文件**: `novel_extractor.py`

**问题**: `_extract_text_recursive` 中 block 元素（`<p>`, `<div>`, `<h1>` 等）之间没有正确换行，多个 block 连成一行。

**修复**: `_collect_inline` 收集完一个 block 后 append `('block', text)`，但外层对连续 block 的 join 缺少 `\n`。确保每个 block 元素输出为独立一行。

### 2. strip_thinking 修复

**文件**: `novel_translator.py`

**当前**: 两个 regex（closed + unclosed），处理不了 `</think>` 前无 `<think>` 的情况。

**修复**: 找最后一个 `</think>` 的位置，取其后的所有文本：

```python
def strip_thinking(text: str) -> str:
    idx = text.rfind('</think>')
    if idx != -1:
        return text[idx + 8:].strip()
    return text.strip()
```

### 3. 术语表存储修复

**问题 A**: 只保存最后一章术语表。
**修复**: 每章术语表保存到 `output/{title}/logs/glossary/{idx:03d}_{name}.txt`。

**问题 B**: 存 Haiku 原始输出（含备注、#标题）。
**修复**: 存储前过滤——只保留包含 `【】` 的行：

```python
def clean_glossary(raw: str) -> str:
    return '\n'.join(
        line for line in raw.strip().split('\n')
        if '【' in line and '】' in line
    )
```

所有需要使用术语表的地方（存入 accumulated、传给翻译模型）都用过滤后的版本。

### 4. 章节标题注入术语表

**来源**: `EPUBParser` 的 spine/TOC 数据能拿到每个 spine item 对应的标题。

**注入方式**: 每章翻译前，从 TOC 提取该章标题，作为 `【原文标题】` 条目追加到术语表。Haiku 生成术语表时会包含这个标题的翻译。翻译正文时，模型从 system prompt 的术语表中看到标题翻译，自然保持一致。

**TOC 翻译**: 翻译完所有章节后，从 accumulated_glossary 中提取所有章节标题的翻译，构造 `translated_metadata.json`。这样 TOC 和正文标题来自同一个翻译源。

### 5. EPUB 打包复用 HTMLEpubBuilder

**删除**: `novel_builder.py`（从零写的垃圾）。

**新增**: `txt_to_xhtml` 转换函数——将翻译后的 `.txt` 转为 `.xhtml`：
- 从原 EPUB 的对应 XHTML 提取 `<head>` 部分（保留 CSS 引用等）
- 每个非空行包在 `<p>` 里
- `[Image: xxx]` 替换为 `<img src="xxx"/>`（保留原始相对路径）
- 输出为 `.xhtml` 文件，文件名与原 EPUB 中的 XHTML 文件名一致

**打包**: 使用 `HTMLEpubBuilder(BuildConfig(..., translated_metadata=...)).build()`。

### 6. translated_metadata.json 构造

翻译完成后，从 accumulated_glossary 中提取章节标题映射，构造：

```json
{
  "translated_title": "那十个字，我永远不会忘记",
  "toc": [
    {"original": "一章　友達になるのに必要なこと", "translated": "第一章 成为朋友所必要的事"},
    ...
  ]
}
```

传给 `HTMLEpubBuilder`，它会自动更新 OPF/NCX/NAV。

---

## 代码修改清单

### 修改

| 文件 | 变更 |
|------|------|
| `novel_extractor.py` | 修复 block 元素换行 |
| `novel_translator.py` | strip_thinking 重写；术语表过滤+存储；章节标题注入；构造 translated_metadata；每章术语表保存到 logs/ |
| `novel_prompts.py` | 无大改，可能微调 prompt 措辞 |
| `cli.py` | translate-novel 命令改用 HTMLEpubBuilder 打包；提取 TOC 信息传给 translator |

### 删除

| 文件 | 原因 |
|------|------|
| `novel_builder.py` | 被 HTMLEpubBuilder 替代 |
| `core/whole/prompts/novel_agent.py` | 已删除（v2 遗留确认） |

### 不变

| 文件 | 原因 |
|------|------|
| `builder.py` | HTMLEpubBuilder + translate_metadata 直接复用 |
| `network_utils.py` | OpenAIClient/AnthropicClient 已就绪 |
| `core/hooks/validators.py` | LineCountValidator 复用 |

---

## 新增调试支持

| 文件 | 内容 |
|------|------|
| `output/{title}/logs/glossary/{idx}_{name}.txt` | 每章术语表（过滤后的纯【】行） |
| `output/{title}/logs/glossary/accumulated.txt` | 翻译结束时的全局术语表快照 |
| `output/{title}/translated_metadata.json` | 翻译后的标题+TOC |

---

## 实现顺序

1. 修 `novel_extractor.py`：block 元素换行
2. 修 `novel_translator.py`：
   - `strip_thinking` → rfind 方案
   - 术语表过滤（`clean_glossary`）
   - 每章术语表保存到 logs/glossary/
   - 章节标题从 TOC 注入术语表
   - 翻译完成后构造 `translated_metadata.json`
3. 修 `cli.py`：
   - 提取 TOC 信息传给 translator
   - 打包改用 HTMLEpubBuilder（txt→xhtml 转换 + BuildConfig）
4. 删 `novel_builder.py`
5. 清理旧输出，重新翻译全书
