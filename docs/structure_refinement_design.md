# PDF2EPUB Structure Refinement Design

## 问题分析

### 当前实现的问题

**Breakdown → OCR → Polish/Translate 流程中的浪费**：

1. Breakdown 识别了 subchapter（含页码），但**未被利用**
2. OCR 按整章处理，subchapter 信息**被丢弃**
3. Polish/Translate 阶段因 token 限制（~8000）必须分割
4. Content splitter **重新分析**结构（浪费 token + 可能不准确）

**根本原因**：
- Subchapter 的页码只是"大概位置"（可能在页面中间）
- 一页可能包含多个 subchapter 的片段 + 脚注
- 无法简单地按页码范围切分

---

## 核心设计原则

1. **严格基于原书 TOC**：提取 TOC 所示的层级，不让 LLM 创造结构
2. **合理粒度范围**：
   - 下限：~2000 tokens（避免过度碎片化）
   - 上限：~8000 tokens（LLM 限制）
   - 目标：Section 级别（不到 paragraph）
3. **LLM 输出内容而非位置**：验证边界时输出实际文本片段，不输出字符位置
4. **自包含处理单元**：每个单元包含其引用的所有 footnote definitions
5. **技术选型**：裸 API + 简单状态类（不引入 LangGraph/PydanticAI）

---

## 完整流程

### 阶段 1：OCR 整本书（逐页）

**目标**：建立页面级基础单元

**修改**：`pdf2epub/ocr_pages.py`（逐页OCR处理）

**输出**：
```
pages/
├── page_001.md
├── page_002.md
├── ...
└── page_stats.json  # 每页的 token 统计
```

**关键改动**：
```python
def ocr_full_book(pdf_path, output_dir):
    """OCR 全书，逐页输出"""
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    page_stats = {}

    for page_num in range(1, total_pages + 1):
        # OCR 单页
        markdown = ocr_single_page(pdf_path, page_num)

        # 保存
        page_file = pages_dir / f"page_{page_num:03d}.md"
        save_markdown(page_file, markdown)

        # 统计 token
        page_stats[page_num] = {
            'tokens': count_tokens(markdown),
            'file': str(page_file)
        }

    # 保存统计
    with open(pages_dir / "page_stats.json", 'w') as f:
        json.dump(page_stats, f)
```

---

### 阶段 2：提取 TOC 结构（Breakdown）

**目标**：提取完整的层级结构（不限制层数）

**修改**：`pdf2epub/breakdown.py`

**Prompt 改进**：
```python
prompt = f"""
Analyze this book and extract its structure EXACTLY as it appears in the table of contents.

**CRITICAL RULES**:
1. Extract ALL levels from the TOC (Part, Chapter, Section, Subsection, ...)
2. DO NOT create artificial subdivisions beyond what the TOC shows
3. Stop at a reasonable granularity (Section level) - DO NOT go to paragraph level
4. For each entry, provide the PDF page number (not printed page numbers)
5. Preserve the original hierarchy depth (1-5+ levels)

**Examples**:
- Essay collection (1 level): [{{title: "Essay 1", level: 1, children: []}}]
- Standard book (2 levels): chapters with sections
- Academic book (3 levels): Part → Chapter → Section
- Complex book (4+ levels): Part → Chapter → Section → Subsection

Return JSON:
{{
    "author": string,
    "table_of_contents": {{
        "start_page": int,
        "end_page": int,
        "entries": [
            {{
                "title": string,
                "page_number": int,
                "level": int  # 1, 2, 3, 4, ... (no upper limit)
            }}
        ]
    }},
    "chapters": [  # Top-level units (recursive tree)
        {{
            "title": string,
            "start_page": int,
            "end_page": int,
            "level": 1,
            "children": [  # Recursive
                {{
                    "title": string,
                    "start_page": int,
                    "end_page": int,
                    "level": 2,
                    "children": [ ... ]
                }}
            ]
        }}
    ]
}}
"""
```

**输出**：`toc_tree.json`（递归树形结构）

---

### 阶段 3：验证和调整（Agentic）⭐核心

**目标**：验证边界 + 调整单元大小 + 处理 footnotes

**新建**：`pdf2epub/structure_refiner.py`

#### 3.1 类结构

```python
class StructureRefiner:
    MIN_TOKENS = 2000  # 最小处理单元
    MAX_TOKENS = 8000  # 最大处理单元

    def __init__(self, llm_client, config):
        self.llm = llm_client
        self.cheap_llm = llm_client  # Gemini 2.0 Flash for verification
        self.config = config
        self.state = RefinerState()

    def refine_structure(self, toc_tree, pages_dir):
        """主流程：验证 + 调整 + footnotes"""

        # Step 1: 提取叶子节点
        leaf_nodes = self._extract_leaf_nodes(toc_tree)

        # Step 2: 验证每个节点的边界
        for node in leaf_nodes:
            validation = self._verify_boundary_with_footnotes(
                node, pages_dir, max_retries=2
            )
            node['boundary_info'] = validation

        # Step 3: 调整单元大小
        processing_units = self._adjust_unit_sizes(leaf_nodes, pages_dir)

        # Step 4: 为每个单元收集 footnotes
        for unit in processing_units:
            unit['footnotes'] = self._collect_unit_footnotes(unit, pages_dir)

        return processing_units
```

#### 3.2 边界验证（含 Footnote 检测）

```python
def _verify_boundary_with_footnotes(self, node, pages_dir, max_retries=2):
    """验证节点边界，让 LLM 输出实际内容"""

    for attempt in range(max_retries):
        prev_page = read_page(node['start_page'] - 1) if node['start_page'] > 1 else ""
        curr_page = read_page(node['start_page'])
        next_section = node.get('next_sibling_title', 'END OF CHAPTER')

        prompt = f"""
You are verifying the boundary for a section in a book.

**Section to verify**: "{node['title']}"
**Expected start page**: {node['start_page']}
**Next section**: "{next_section}"

**Previous page content**:
{prev_page}

**Current page content**:
{curr_page}

**Task**: Determine if this section starts on the current page and extract the exact boundary content.

Return JSON:
{{
    "verified": bool,  // True if section title found on current page
    "title_found_on_page": bool,
    "content_before_section": string,  // ALL content on current page BEFORE this section starts
                                       // This may include:
                                       // - End of previous section's main text
                                       // - Footnotes from previous section (e.g., "[1] definition...")
    "content_after_section": string,   // Content AFTER this section ends (if next section starts on same page, or trailing footnotes)
    "footnote_references": [int],      // List of footnote numbers referenced in this section (e.g., [1, 2, 5])
    "confidence": float                // 0.0-1.0
}}

**CRITICAL**:
1. Output the ACTUAL TEXT content, NOT character positions or line numbers
2. "content_before_section" must include everything before the section title
3. Handle cases where footnotes from previous section appear AFTER this section's content
4. Extract ALL footnote reference numbers (e.g., [1], [2]) in this section's text
"""

        response = self.cheap_llm.generate(prompt)

        if response['verified'] or attempt == max_retries - 1:
            return response

        # 验证失败，尝试 re-breakdown
        logger.warning(f"Boundary verification failed for '{node['title']}' (attempt {attempt+1})")
        refined_node = self._rebreakdown_section(node, pages_dir)
        if refined_node:
            node.update(refined_node)

    return response
```

#### 3.3 调整单元大小（聚合 vs 标记）

```python
def _adjust_unit_sizes(self, leaf_nodes, pages_dir):
    """调整处理单元大小：聚合小节，标记超大节"""

    processing_units = []
    buffer = []  # 用于聚合小节
    buffer_tokens = 0

    for i, node in enumerate(leaf_nodes):
        tokens = self._estimate_tokens(node, pages_dir)
        node['estimated_tokens'] = tokens

        if tokens > self.MAX_TOKENS:
            # === 单元太大 ===
            logger.warning(
                f"Section '{node['title']}' has {tokens} tokens (> {self.MAX_TOKENS})"
            )

            # 先输出 buffer
            if buffer:
                processing_units.append(self._create_unit(buffer, buffer_tokens))
                buffer = []
                buffer_tokens = 0

            # 尝试从 PDF 提取隐藏的小节
            hidden_subsections = self._extract_hidden_subsections(node, pages_dir)
            if hidden_subsections:
                logger.info(f"Found {len(hidden_subsections)} hidden subsections")
                # 递归处理
                sub_units = self._adjust_unit_sizes(hidden_subsections, pages_dir)
                processing_units.extend(sub_units)
            else:
                # 无法细分，标记为 fallback
                node['needs_content_splitter'] = True
                processing_units.append(self._create_unit([node], tokens))

        elif tokens < self.MIN_TOKENS:
            # === 单元太小 ===
            buffer.append(node)
            buffer_tokens += tokens

            # 检查 footnote 冲突（避免多个节点引用相同 footnote）
            if self._has_footnote_conflict(buffer):
                # 有冲突，不能聚合
                if len(buffer) > 1:
                    processing_units.append(self._create_unit(buffer[:-1], buffer_tokens - tokens))
                buffer = [buffer[-1]]
                buffer_tokens = tokens

            # 检查是否应该输出
            elif buffer_tokens >= self.MIN_TOKENS or i == len(leaf_nodes) - 1:
                processing_units.append(self._create_unit(buffer, buffer_tokens))
                buffer = []
                buffer_tokens = 0

        else:
            # === 大小合适 ===
            if buffer:
                processing_units.append(self._create_unit(buffer, buffer_tokens))
                buffer = []
                buffer_tokens = 0

            processing_units.append(self._create_unit([node], tokens))

    # 输出剩余 buffer
    if buffer:
        processing_units.append(self._create_unit(buffer, buffer_tokens))

    return processing_units

def _has_footnote_conflict(self, nodes):
    """检查多个节点是否引用相同的 footnotes"""
    all_refs = []
    for node in nodes:
        refs = node.get('boundary_info', {}).get('footnote_references', [])
        all_refs.extend(refs)

    # 有重复引用 = 有冲突
    return len(all_refs) != len(set(all_refs))

def _create_unit(self, nodes, total_tokens):
    """创建处理单元"""
    return {
        'unit_id': None,  # 后续分配
        'nodes': nodes,  # 原始 TOC 节点（用于生成目录）
        'start_page': nodes[0]['start_page'],
        'end_page': nodes[-1]['end_page'],
        'token_count': total_tokens,
        'title': ' + '.join(n['title'] for n in nodes),
        'is_aggregated': len(nodes) > 1,
        'needs_content_splitter': any(n.get('needs_content_splitter') for n in nodes),
        'path': self._get_hierarchy_path(nodes[0])  # e.g., ["Part I", "Chapter 1", "Section 1"]
    }
```

#### 3.4 Footnote 收集

```python
def _collect_unit_footnotes(self, unit, pages_dir):
    """为处理单元收集所有引用的 footnote definitions"""

    # 收集所有 footnote references
    all_refs = set()
    for node in unit['nodes']:
        refs = node.get('boundary_info', {}).get('footnote_references', [])
        all_refs.update(refs)

    if not all_refs:
        return None

    # 确定搜索范围（父章节）
    parent_chapter = self._find_parent_chapter(unit['nodes'][0])
    chapter_pages = range(parent_chapter['start_page'], parent_chapter['end_page'] + 1)
    chapter_content = '\n\n'.join(read_page(p) for p in chapter_pages)

    # 用 LLM 提取 definitions
    prompt = f"""
Extract footnote definitions for the following numbers: {sorted(all_refs)}

**Chapter content** (search in this range):
{chapter_content}

Return JSON:
{{
    "footnotes": {{
        "1": "Complete definition text for footnote 1...",
        "2": "Complete definition text for footnote 2...",
        ...
    }}
}}

**IMPORTANT**:
- Only extract definitions for the requested numbers
- Include the complete definition text
- If a footnote number is not found, omit it from the result
"""

    response = self.llm.generate(prompt)

    return {
        'referenced_numbers': sorted(all_refs),
        'definitions': response['footnotes']
    }
```

#### 3.5 局部 Re-breakdown

```python
def _rebreakdown_section(self, node, pages_dir):
    """对验证失败的节点局部重新 breakdown"""

    # 读取该节点的页面范围
    content = '\n\n'.join(
        read_page(p) for p in range(node['start_page'], node['end_page'] + 1)
    )

    prompt = f"""
Re-analyze this section to find the correct boundary.

**Expected section**: "{node['title']}"
**Page range**: {node['start_page']}-{node['end_page']}

**Content**:
{content}

Task: Determine if this section title exists, and provide the correct page number.

Return JSON:
{{
    "section_found": bool,
    "correct_start_page": int,  // If found
    "correct_end_page": int,
    "alternative_titles": [string]  // Possible variations of the title
}}
"""

    response = self.llm.generate(prompt)

    if response['section_found']:
        return {
            'start_page': response['correct_start_page'],
            'end_page': response['correct_end_page']
        }

    return None
```

#### 3.6 提取隐藏的小节

```python
def _extract_hidden_subsections(self, node, pages_dir):
    """对超大 section 尝试提取 TOC 未显示的小节"""

    content = '\n\n'.join(
        read_page(p) for p in range(node['start_page'], node['end_page'] + 1)
    )

    prompt = f"""
This section "{node['title']}" is too large ({node['estimated_tokens']} tokens).
Identify natural subdivisions (sub-headings) that can break it into smaller units.

**Content**:
{content}

Target: Each subdivision should be approximately {self.MIN_TOKENS}-{self.MAX_TOKENS} tokens.

Return JSON:
{{
    "has_subdivisions": bool,
    "subdivisions": [
        {{
            "title": string,  // Sub-heading text (or generated if none exists)
            "content": string,  // Actual text content of this subdivision
            "estimated_tokens": int
        }}
    ]
}}

If no natural subdivisions exist, return {{"has_subdivisions": false}}.
"""

    response = self.llm.generate(prompt)

    if not response['has_subdivisions']:
        return None

    # 将 LLM 输出的内容片段映射回页码
    sub_nodes = []
    for i, sub in enumerate(response['subdivisions']):
        page_range = self._find_content_pages(
            sub['content'],
            node['start_page'],
            node['end_page']
        )

        sub_nodes.append({
            'title': sub['title'],
            'start_page': page_range['start'],
            'end_page': page_range['end'],
            'level': node['level'] + 1,
            'children': [],
            'parent': node
        })

    return sub_nodes
```

#### 3.7 状态管理

```python
class RefinerState:
    """简单的状态类，用于断点续传"""

    def __init__(self):
        self.verified_nodes = {}  # node_id -> boundary_info
        self.failed_nodes = []
        self.processing_units = []
        self.retry_counts = {}

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'verified': self.verified_nodes,
                'failed': self.failed_nodes,
                'units': self.processing_units,
                'retries': self.retry_counts
            }, f, indent=2)

    def load(self, path):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                self.verified_nodes = data.get('verified', {})
                self.failed_nodes = data.get('failed', [])
                self.processing_units = data.get('units', [])
                self.retry_counts = data.get('retries', {})
            return True
        return False
```

**输出**：
- `processing_units.json`（包含 footnote 信息）
- `refiner_state.json`（用于断点续传）

---

### 阶段 4：合并页面（含 Footnotes）

**目标**：根据处理单元合并页面 + 附加 footnotes

**新建**：`pdf2epub/page_merger.py`

```python
def merge_pages_by_units(processing_units, pages_dir, output_dir):
    """为每个处理单元合并页面并附加 footnotes"""

    units_dir = output_dir / "units"
    units_dir.mkdir(exist_ok=True)

    unit_metadata = {}

    for idx, unit in enumerate(processing_units, start=1):
        unit_id = f"unit_{idx:03d}"
        unit['unit_id'] = unit_id

        # 合并主体内容
        content = merge_unit_content(unit, pages_dir)

        # 附加 footnotes
        if unit.get('footnotes'):
            footnote_section = format_footnotes(unit['footnotes'])
            content = f"{content}\n\n---\n\n## Footnotes\n\n{footnote_section}"

        # 保存
        unit_file = units_dir / f"{unit_id}.md"
        save_markdown(unit_file, content)

        # 记录 metadata
        unit_metadata[unit_id] = {
            'path': unit['path'],
            'toc_nodes': [
                {'level': n['level'], 'title': n['title']}
                for n in unit['nodes']
            ],
            'page_range': [unit['start_page'], unit['end_page']],
            'token_count': unit['token_count'],
            'is_aggregated': unit['is_aggregated'],
            'needs_content_splitter': unit['needs_content_splitter'],
            'file': str(unit_file)
        }

    # 保存 metadata
    with open(output_dir / "unit_metadata.json", 'w') as f:
        json.dump(unit_metadata, f, indent=2)

def merge_unit_content(unit, pages_dir):
    """合并单元的主体内容（处理边界）"""

    content_parts = []

    # 处理起始页
    first_node = unit['nodes'][0]
    start_page_content = read_page(first_node['start_page'])

    if first_node['boundary_info'].get('content_before_section'):
        # 去掉前面不属于这个 section 的内容
        start_page_content = remove_prefix_content(
            start_page_content,
            first_node['boundary_info']['content_before_section']
        )

    content_parts.append(start_page_content)

    # 中间的完整页面
    for page_num in range(first_node['start_page'] + 1, unit['end_page']):
        content_parts.append(read_page(page_num))

    # 处理结束页（如果与起始页不同）
    last_node = unit['nodes'][-1]
    if last_node['end_page'] > first_node['start_page']:
        end_page_content = read_page(last_node['end_page'])

        if last_node['boundary_info'].get('content_after_section'):
            # 去掉后面不属于这个 section 的内容
            end_page_content = remove_suffix_content(
                end_page_content,
                last_node['boundary_info']['content_after_section']
            )

        content_parts.append(end_page_content)

    return '\n\n'.join(content_parts)

def format_footnotes(footnotes_info):
    """格式化 footnote definitions"""
    lines = []
    for num in footnotes_info['referenced_numbers']:
        definition = footnotes_info['definitions'].get(str(num))
        if definition:
            lines.append(f"[{num}] {definition}")
        else:
            lines.append(f"[{num}] [Definition not found]")

    return '\n\n'.join(lines)

def remove_prefix_content(full_text, prefix_to_remove):
    """移除文本前缀（基于实际内容匹配）"""
    # 简单实现：找到 prefix 的结束位置
    idx = full_text.find(prefix_to_remove)
    if idx != -1:
        return full_text[idx + len(prefix_to_remove):].lstrip()
    return full_text

def remove_suffix_content(full_text, suffix_to_remove):
    """移除文本后缀"""
    idx = full_text.rfind(suffix_to_remove)
    if idx != -1:
        return full_text[:idx].rstrip()
    return full_text
```

**输出**：
```
units/
├── unit_001.md
├── unit_002.md
├── ...
├── unit_metadata.json
```

---

### 阶段 5：Polish/Translate

**修改**：`pdf2epub/processors/polisher.py`, `pdf2epub/processors/translator.py`

```python
def process_units(units_dir, unit_metadata, processor_func):
    """按 unit 处理"""

    for unit_id, metadata in unit_metadata.items():
        unit_file = units_dir / f"{unit_id}.md"

        if metadata['needs_content_splitter']:
            # Fallback：使用现有的 content splitter
            logger.info(f"Using content splitter for {unit_id}")
            content = read_file(unit_file)
            parts = content_splitter.split(content, MAX_TOKENS)

            results = []
            for part_idx, part in enumerate(parts, start=1):
                result = processor_func(part)
                output_file = units_dir / f"{unit_id}_part_{part_idx}.md"
                save_markdown(output_file, result)
                results.append(output_file)

        else:
            # 正常处理（单元已经是自包含的）
            content = read_file(unit_file)
            result = processor_func(content)
            save_markdown(unit_file, result)
```

---

## 文件命名方案

### 新设计（不考虑向后兼容）

采用**扁平化数字 ID + 元数据**：

```
output/
├── pages/
│   ├── page_001.md
│   ├── page_002.md
│   └── page_stats.json
├── toc_tree.json
├── processing_units.json
├── units/
│   ├── unit_001.md
│   ├── unit_002.md
│   ├── unit_003_part_1.md  # 使用 content splitter 分割的
│   ├── unit_003_part_2.md
│   └── ...
├── unit_metadata.json
└── polished/  # 或 translated/
    ├── unit_001.md
    └── ...
```

### unit_metadata.json 结构

```json
{
    "unit_001": {
        "path": ["Part I", "Chapter 1", "Section 1"],
        "toc_nodes": [
            {"level": 3, "title": "Section 1"}
        ],
        "page_range": [10, 15],
        "token_count": 3500,
        "is_aggregated": false,
        "needs_content_splitter": false,
        "file": "units/unit_001.md"
    },
    "unit_002": {
        "path": ["Part I", "Chapter 1"],
        "toc_nodes": [
            {"level": 3, "title": "Section 2"},
            {"level": 3, "title": "Section 3"}
        ],
        "page_range": [16, 20],
        "token_count": 2800,
        "is_aggregated": true,
        "needs_content_splitter": false,
        "file": "units/unit_002.md"
    }
}
```

**优势**：
- 文件名简洁（unit_NNN）
- 完整的层级信息保存在 metadata
- 易于批量处理和排序
- 保留原始 TOC 用于 EPUB 导航

---

## Footnote 处理策略

### 关键场景

| 场景 | 处理方式 |
|------|---------|
| **Section 引用 footnotes** | 从父章节提取 definitions，附加到单元末尾 |
| **多个 sections 引用相同 footnote** | **避免聚合**（检测冲突） |
| **Footnote reference 和 definition 分离** | 在单元合并时包含 definitions |
| **聚合多个 sections** | 合并所有 footnotes，按 number 去重 |
| **一页包含前章脚注 + 本章内容** | 边界验证时区分，通过 `content_before_section` 排除 |

### 实现要点

1. **检测 reference**：边界验证时提取 `[1]`, `[2]` 等
2. **查找 definition**：在父章节范围内搜索（用 LLM）
3. **冲突检测**：聚合前检查是否有重复引用
4. **附加到单元**：在单元内容末尾添加 Footnotes section

---

## 配置参数

```yaml
# config.yaml
structure_refinement:
  enabled: true

  # Token 限制
  min_tokens: 2000
  max_tokens: 8000

  # 验证设置
  verification_model: "gemini-2.0-flash"  # 便宜模型
  max_retries: 2

  # 细化设置
  enable_hidden_subsection_detection: true
  enable_footnote_processing: true

  # Fallback 设置
  use_content_splitter_when_needed: true
```

---

## 修改文件清单

### 新增文件

1. **`pdf2epub/structure_refiner.py`**
   - `StructureRefiner` 类：验证、调整、footnote 处理
   - `RefinerState` 类：状态管理

2. **`pdf2epub/page_merger.py`**
   - `merge_pages_by_units()`：合并页面
   - `format_footnotes()`：格式化 footnotes
   - 边界内容处理函数

3. **`pdf2epub/footnote_extractor.py`**（可选，独立模块）
   - Footnote 提取和格式化逻辑

### 修改文件

1. **`pdf2epub/breakdown.py`**
   - 修改 prompt：支持无限层级
   - 输出递归树形结构

2. **`pdf2epub/ocr_pages.py`**
   - 实现逐页OCR输出
   - 保存 token 统计

3. **`pdf2epub/processors/polisher.py`**
4. **`pdf2epub/processors/translator.py`**
   - 按 unit 处理
   - Content splitter 仅作 fallback

5. **`pdf2epub/build_epub.py`**
   - 使用 `toc_tree.json` 生成 TOC
   - 适配层级文件结构

6. **`pdf2epub/cli.py`**
   - 在 OCR 和 Polish 之间插入 refine 和 merge 阶段
   - 新增命令行参数

---

## 实现优先级

### Phase 1：MVP（核心流程）

1. ✅ 新文件命名方案 + metadata 结构
2. ✅ 修改 `breakdown.py` prompt（支持无限层级）
3. ✅ 修改 `ocr_chapters.py`（逐页输出）
4. ✅ 实现 `structure_refiner.py` 基础验证逻辑
   - 边界验证（LLM 输出内容）
   - 单元大小调整（聚合/标记）
5. ✅ 实现 `page_merger.py`（简单合并）

**预期成果**：能够生成基于 TOC 的处理单元，大部分情况下避免使用 content splitter

### Phase 2：Footnote 处理

6. ✅ 边界验证时检测 footnote references
7. ✅ 实现 footnote definitions 提取
8. ✅ 实现 footnote 冲突检测
9. ✅ 页面合并时附加 footnotes

**预期成果**：每个处理单元是自包含的（包含引用的 footnotes）

### Phase 3：高级功能

10. ✅ 局部 re-breakdown
11. ✅ 隐藏 subsection 检测
12. ✅ 修改 polish/translate 流程
13. ✅ EPUB 生成适配新结构
14. ⚙️ 性能优化（并行处理、缓存）
15. 📊 监控和日志（验证成功率、token 节省）

---

## 技术选型：为什么用裸 API

### 考虑的方案

1. **裸 API + 简单状态类** ⭐选择
2. LangGraph（状态机框架）
3. PydanticAI（类型安全 agent）

### 选择裸 API 的理由

**流程特点**：
- 递归验证：逻辑清晰，不需要复杂状态机
- 有条件分支，但不是开放式规划
- 状态相对简单：验证状态、重试次数、单元列表

**裸 API 优势**：
1. ✅ 简单直接，易理解和维护
2. ✅ 依赖少，性能好
3. ✅ 符合当前项目风格（`breakdown.py`, `ocr_chapters.py` 都是裸 API）
4. ✅ 状态用 JSON 持久化就够了

**何时考虑 LangGraph**：
- 需要复杂的多步规划（agent 自己决定策略）
- 需要人机交互确认
- 需要可视化调试复杂的决策流程
- 需要管理大量并行任务的依赖关系

**当前方案不需要这些**。

### 实现建议

```python
# 简单够用的状态管理
class RefinerState:
    def __init__(self):
        self.verified = {}
        self.failed = []
        self.retries = {}

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path) as f:
                self.__dict__.update(json.load(f))

# 直接的递归逻辑
def refine_node_recursive(node, depth, max_retries):
    validation = verify_boundary(node)

    if not validation['verified'] and max_retries > 0:
        refined = rebreakdown_section(node)
        return refine_node_recursive(refined, depth, max_retries-1)

    if node.get('children'):
        for child in node['children']:
            refine_node_recursive(child, depth+1, max_retries)
```

---

## 预期效果

### 定量指标

| 指标 | 当前实现 | 改进后 | 提升 |
|------|---------|--------|------|
| **Content splitter 使用频率** | ~90% | ~5-10% | 8-18x |
| **Token 浪费** | 高（重新分析结构） | 低（直接使用 TOC） | 50-70% ↓ |
| **处理单元准确性** | 中（LLM 重新分析） | 高（基于验证的 TOC） | +30-40% |
| **Footnote 完整性** | 低（可能分离） | 高（自包含单元） | +80% |

### 定性改进

- ✅ **保持原书结构**：严格基于 TOC
- ✅ **合理粒度**：避免过度碎片化和超限
- ✅ **自包含单元**：每个单元包含其 footnotes
- ✅ **优雅降级**：真正无法处理的才 fallback
- ✅ **易于维护**：简单的代码结构，不引入复杂框架

---

## 使用示例

### CLI 命令

```bash
# 完整流程
uv run pdf2epub convert \
  -i book.pdf \
  --enable-structure-refinement \
  --min-tokens 2000 \
  --max-tokens 8000

# 只运行 refine 阶段（调试）
uv run pdf2epub refine \
  --toc-tree toc_tree.json \
  --pages-dir output/pages \
  --output processing_units.json
```

### Python API

```python
from pdf2epub.structure_refiner import StructureRefiner
from pdf2epub.page_merger import merge_pages_by_units

# 验证和调整
refiner = StructureRefiner(llm_client, config)
processing_units = refiner.refine_structure(toc_tree, pages_dir)

# 合并页面
merge_pages_by_units(processing_units, pages_dir, output_dir)
```

---

## 总结

这个方案通过**在 OCR 和 Polish 之间插入验证和调整阶段**，让 Breakdown 识别的 subchapter 真正被利用，从而：

1. **大幅减少** content splitter 使用（从 90% 降到 5-10%）
2. **保持原书结构**（严格基于 TOC，不创造结构）
3. **生成自包含单元**（包含 footnotes，避免引用和定义分离）
4. **合理粒度**（2000-8000 tokens，避免碎片化和超限）
5. **技术简单**（裸 API，不引入复杂框架）

核心是：**让 LLM 输出实际内容而非位置，递归验证和调整，智能聚合和标记**。
