"""
HTML Translation Prompts.

Prompts specifically designed for translating HTML content while
preserving exact tag structure.
"""

from typing import Optional, Dict, List


def create_html_translation_prompt(
    source_language: str,
    target_language: str,
    entities: Optional[Dict] = None
) -> str:
    """
    Create prompt for HTML translation.

    Key instructions:
    - Preserve ALL HTML tags exactly
    - Only translate text content between tags
    - Preserve tag order, nesting, attributes (including i=N markers)
    - Handle special cases: ruby, entities, CDATA
    """
    target_lower = target_language.lower()

    if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
        prompt = _create_chinese_prompt(source_language)
    elif target_lower in ["english", "英语", "en"]:
        prompt = _create_english_prompt(source_language)
    else:
        prompt = _create_generic_prompt(source_language, target_language)

    # Add entity reference if available
    if entities:
        prompt += _create_entity_reference(entities, target_language)

    # Final instruction
    if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
        prompt += "\n\n请翻译以下HTML内容："
    else:
        prompt += "\n\nTranslate the following HTML:"

    return prompt


def _create_chinese_prompt(source_language: str) -> str:
    """Create Chinese translation prompt."""
    return f"""你是一位专业的HTML内容翻译专家。请将以下HTML内容从{source_language}翻译成简体中文。

**极其重要的要求**：

1. **绝对不要修改HTML标签结构**：
   - 保留所有标签，包括开始标签和结束标签
   - 保持标签的顺序和嵌套关系完全不变
   - 不要添加新标签或删除现有标签
   - 自闭合标签如 `<br/>`, `<hr/>`, `<img.../>` 保持原样

2. **仅翻译文本内容**：
   - 只翻译标签之间的文字内容
   - 不要翻译或修改任何属性值（如 `class="xxx"`, `id="xxx"`, `i=1`）
   - 特别注意：`i=数字` 是标记属性，必须保留不变

3. **特殊元素处理**：
   - `<ruby>` 标签：翻译基础文字，保留 `<rt>` 标签中的注音不变
   - HTML实体（如 `&nbsp;` `&mdash;` `&lt;`）：保持不变
   - 注释 `<!-- -->` 中的内容：保持不变

4. **保持格式**：
   - 保留原有的换行和缩进格式
   - 不要添加额外的空白字符

5. **输出格式**：
   - 只返回翻译后的HTML，不要添加任何解释
   - 不要用代码块（```）包裹输出
   - 不要在开头或结尾添加任何说明文字

**示例**：

输入：
```
<p i=1>こんにちは</p>
```

输出：
```
<p i=1>你好</p>
```

输入：
```
<div i=2><span i=3>日本語</span>テキスト</div>
```

输出：
```
<div i=2><span i=3>日语</span>文本</div>
```

输入：
```
<ruby>漢字<rt>かんじ</rt></ruby>
```

输出：
```
<ruby>汉字<rt>かんじ</rt></ruby>
```"""


def _create_english_prompt(source_language: str) -> str:
    """Create English translation prompt."""
    return f"""You are a professional HTML content translator. Translate the following HTML content from {source_language} to English.

**CRITICAL REQUIREMENTS**:

1. **DO NOT modify HTML tag structure**:
   - Keep ALL tags including opening and closing tags exactly as they are
   - Maintain exact tag order and nesting structure
   - Do not add or remove any tags
   - Self-closing tags like `<br/>`, `<hr/>`, `<img.../>` must stay unchanged

2. **Only translate text content**:
   - Only translate text between tags
   - Do NOT translate or modify any attribute values (like `class="xxx"`, `id="xxx"`, `i=1`)
   - Important: `i=number` is a marker attribute that MUST be preserved

3. **Special elements**:
   - `<ruby>` tags: translate base text, keep `<rt>` annotation text unchanged
   - HTML entities (like `&nbsp;` `&mdash;` `&lt;`): keep unchanged
   - Comments `<!-- -->`: keep unchanged

4. **Preserve formatting**:
   - Keep original line breaks and indentation
   - Do not add extra whitespace

5. **Output format**:
   - Return ONLY the translated HTML, no explanations
   - Do NOT wrap output in code blocks (```)
   - Do NOT add any commentary before or after

**Examples**:

Input:
```
<p i=1>Bonjour le monde</p>
```

Output:
```
<p i=1>Hello world</p>
```

Input:
```
<div i=2><span i=3>Text</span> more text</div>
```

Output:
```
<div i=2><span i=3>Text</span> more text</div>
```"""


def _create_generic_prompt(source_language: str, target_language: str) -> str:
    """Create generic translation prompt for other language pairs."""
    return f"""You are a professional HTML content translator. Translate the following HTML content from {source_language} to {target_language}.

**CRITICAL REQUIREMENTS**:

1. **DO NOT modify HTML tag structure**:
   - Keep ALL tags including opening and closing tags exactly as they are
   - Maintain exact tag order and nesting structure
   - Do not add or remove any tags
   - Self-closing tags like `<br/>`, `<hr/>`, `<img.../>` must stay unchanged

2. **Only translate text content**:
   - Only translate text between tags
   - Do NOT translate or modify any attribute values (like `class="xxx"`, `id="xxx"`, `i=1`)
   - Important: `i=number` is a marker attribute that MUST be preserved

3. **Special elements**:
   - `<ruby>` tags: translate base text, keep `<rt>` annotation text unchanged
   - HTML entities: keep unchanged
   - Comments `<!-- -->`: keep unchanged

4. **Preserve formatting**:
   - Keep original line breaks and indentation
   - Do not add extra whitespace

5. **Output format**:
   - Return ONLY the translated HTML, no explanations
   - Do NOT wrap output in code blocks
   - Do NOT add any commentary"""


def _create_entity_reference(entities: Dict, target_language: str) -> str:
    """
    Create entity reference section for consistent terminology.

    Args:
        entities: Dict with 'characters', 'places', 'terms', etc.
        target_language: Target language for section header
    """
    target_lower = target_language.lower()

    if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
        header = "\n\n**术语参考**（请使用以下固定翻译）："
    else:
        header = "\n\n**Terminology Reference** (use these fixed translations):"

    lines = [header]

    # Characters/Names
    if entities.get('characters'):
        if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
            lines.append("\n人名：")
        else:
            lines.append("\nCharacters:")
        for name, translation in entities['characters'].items():
            lines.append(f"  - {name} → {translation}")

    # Places
    if entities.get('places'):
        if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
            lines.append("\n地名：")
        else:
            lines.append("\nPlaces:")
        for place, translation in entities['places'].items():
            lines.append(f"  - {place} → {translation}")

    # Terms
    if entities.get('terms'):
        if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
            lines.append("\n术语：")
        else:
            lines.append("\nTerms:")
        for term, translation in entities['terms'].items():
            lines.append(f"  - {term} → {translation}")

    return '\n'.join(lines)


def create_retry_prompt_suffix(error_type: str, attempt: int) -> str:
    """
    Get additional prompt text for retry based on error type.

    Args:
        error_type: Type of validation failure
        attempt: Current attempt number
    """
    if error_type == "tag_mismatch":
        return """

**警告**：上次输出的HTML标签结构与输入不匹配。
请务必：
1. 逐个检查每个HTML标签
2. 确保开始标签和结束标签数量完全一致
3. 只翻译文字，不要修改任何 < 和 > 之间的内容

**WARNING**: The HTML tag structure in the previous output did not match the input.
Please ensure:
1. Check every HTML tag carefully
2. Opening and closing tags must match exactly
3. Only translate text, do not modify anything between < and >"""

    elif error_type == "tag_missing":
        return """

**警告**：上次输出丢失了一些HTML标签。
请确保输出包含与输入完全相同的所有标签。

**WARNING**: Some HTML tags were missing in the previous output.
Ensure the output contains all the same tags as the input."""

    elif error_type == "tag_added":
        return """

**警告**：上次输出添加了额外的HTML标签。
请不要添加任何原文中不存在的标签。

**WARNING**: Extra HTML tags were added in the previous output.
Do not add any tags that don't exist in the original."""

    elif error_type == "language_wrong":
        return """

**警告**：上次输出不是正确的目标语言。
请务必使用目标语言翻译。

**WARNING**: The previous output was not in the correct target language.
Please translate to the correct target language."""

    return ""


# ============================================================================
# Compressed Format Prompts (for HTMLCompressor-based workflow)
# ============================================================================

def create_compressed_translation_prompt(
    source_language: str,
    target_language: str,
    entities: Optional[Dict] = None
) -> str:
    """
    Create prompt for translating compressed HTML content.

    Compressed format rules:
    - Each translation unit is wrapped in <div>...</div>
    - Preserve exact number of <div> tags
    - Preserve any inner HTML tags
    """
    target_lower = target_language.lower()

    if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
        prompt = f"""你是一位专业的翻译专家。请将以下内容从{source_language}翻译成简体中文。

**格式要求**：

1. **保持 <div> 结构**：
   - 每个 `<div>...</div>` 是一个翻译单元
   - 翻译时保持相同数量的 `<div>` 标签
   - 不要合并或拆分 `<div>` 标签

2. **保留内部 HTML 标签**：
   - 如果 `<div>` 内包含其他标签（如 `<span>`, `<em>`），保持不变
   - 只翻译文字内容

3. **输出格式**：
   - 直接返回翻译结果，不要添加解释
   - 不要用代码块包裹
   - 不要添加换行或空格分隔 `<div>` 标签

**示例**：

输入：
<div>Hello world</div><div>This is a test</div>

输出：
<div>你好世界</div><div>这是一个测试</div>

输入（带内部标签）：
<div><em>Important</em> text</div>

输出：
<div><em>重要</em>的文本</div>"""
    else:
        prompt = f"""You are a professional translator. Translate the following from {source_language} to {target_language}.

**Format Requirements**:

1. **Preserve <div> structure**:
   - Each `<div>...</div>` is one translation unit
   - Maintain the same number of `<div>` tags
   - Do not merge or split `<div>` tags

2. **Preserve inner HTML tags**:
   - If a `<div>` contains other tags (like `<span>`, `<em>`), keep them
   - Only translate text content

3. **Output format**:
   - Return only the translation, no explanations
   - Do not wrap in code blocks
   - Do not add newlines or spaces between `<div>` tags"""

    # Add entity reference if available
    if entities:
        prompt += _create_entity_reference(entities, target_language)

    # Final instruction
    if target_lower in ["chinese", "中文", "chinese simplified", "简体中文", "zh", "zh-cn"]:
        prompt += "\n\n请翻译以下内容："
    else:
        prompt += "\n\nTranslate the following:"

    return prompt


def create_compressed_retry_prompt(error_type: str) -> str:
    """
    Get retry prompt suffix for compressed format errors.

    Args:
        error_type: Type of validation failure
    """
    if error_type == "div_count_mismatch":
        return """

**警告**：输出的 <div> 数量与输入不匹配！
每个输入的 <div>...</div> 必须对应一个输出的 <div>...</div>。
请仔细检查并确保 <div> 标签数量完全一致。

**WARNING**: The number of <div> tags in output did not match input!
Each input <div>...</div> MUST produce exactly one output <div>...</div>.
Count your <div> tags carefully."""

    elif error_type == "nl_count_mismatch":
        # Legacy support for old format
        return """

**警告**：输出行数与输入不匹配！
每一行输入必须对应一行输出。请仔细数一下行数再回答。

**WARNING**: Output line count did not match input!
Each input line MUST produce exactly one output line. Count your lines carefully."""

    elif error_type == "language_wrong":
        return """

**警告**：翻译结果不是正确的目标语言。
请务必使用目标语言翻译。

**WARNING**: Translation was not in the correct target language.
Please translate to the correct target language."""

    return ""
