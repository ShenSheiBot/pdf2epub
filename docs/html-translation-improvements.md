# HTML 翻译流程改进计划

## 发现的问题

| 问题 | 原因 | 当前修复方式 | 严重程度 |
|------|------|--------------|----------|
| **幻觉首行** | LLM 在翻译开头添加 "开始"/"开端"/"debut" | 手动删除 `.part1.md` 第一行 | 中 |
| **多余空行** | LLM 在翻译中添加空行导致行数不匹配 | 删除所有空行 | 中 |
| **首字下沉样式 (let class)** | 原文 `<span class="let">A</span>` 翻译后变成 `<span class="let">整段</span>` | `fix_dropcap.py` 脚本拆分 span | 高 |
| **短内容跳过** | `_is_image_only_content()` 跳过 <100 字符的内容 | 手动翻译 | 中 |

## 改进方案

### 1. 集成 fix_dropcap 到构建流程

**问题**: 每次 `build-html-epub` 都会覆盖之前的修复

**方案**: 在 `build_html_epub()` 函数末尾自动调用 `fix_dropcap()`

```python
# build_html_epub.py
def build_html_epub(...):
    # ... existing build logic ...

    # Post-processing: fix drop cap styling
    from pdf2epub.scripts.fix_dropcap import fix_epub
    fix_epub(output_epub_path)
```

### 2. 修复短内容跳过问题

**问题**: `_is_image_only_content()` 用于跳过纯图片页面，但误判了短标题页

**方案 A**: 在 `HTMLTranslateProcessor` 中覆盖该方法，对 HTML 翻译禁用此检查

```python
class HTMLTranslateProcessor(BaseMarkdownProcessor):
    def _is_image_only_content(self, content: str, min_text_chars: int = 100) -> bool:
        # HTML translation should translate all text, even short content
        return False
```

**方案 B**: 检查是否真的包含 `<img>` 标签，而不是仅凭字符数判断

### 3. 翻译后验证与自动修复 Agent

**目标**: 在 `build-html-epub` 之前自动检测并修复常见的 LLM 翻译问题

**架构**:

```
translate-html
    ↓
[translation-qa-agent]  ← 新增
    ↓
build-html-epub
```

**Agent 功能**:

1. **验证阶段**
   - 对比原文/译文行数
   - 检测未翻译内容（仍为源语言）
   - 检测幻觉首行模式
   - 检测空行问题

2. **修复阶段**
   - 删除幻觉首行
   - 删除多余空行
   - 重新翻译未翻译的短内容

3. **报告阶段**
   - 生成修复报告
   - 标记无法自动修复的问题

**实现思路**:

```python
class TranslationQAAgent:
    def validate_and_fix(self, compressed_dir: Path, translated_dir: Path):
        issues = []

        for orig_file in compressed_dir.glob("*.md"):
            trans_file = translated_dir / orig_file.name

            # Check line count
            orig_lines = orig_file.read_text().strip().split('\n')
            trans_lines = trans_file.read_text().strip().split('\n')

            if len(orig_lines) != len(trans_lines):
                issues.append(self.fix_line_count_mismatch(orig_file, trans_file))

            # Check for untranslated content
            if self.is_untranslated(trans_lines, source_lang='French'):
                issues.append(self.retranslate(orig_file, trans_file))

            # Check for hallucinated first line
            if self.has_hallucinated_intro(trans_lines):
                issues.append(self.remove_first_line(trans_file))

        return issues
```

### 4. 样式类特殊处理

**问题**: 某些 CSS 类（如 `let`）有特殊语义，翻译后需要调整

**方案**: 维护一个"样式类处理规则"配置

```yaml
# style_rules.yaml
special_classes:
  let:
    description: "Drop cap - only first character should have this class"
    fix: "split_first_char"

  # 未来可能遇到的其他情况
  # small-caps:
  #   description: "Small caps styling"
  #   fix: "preserve_on_all"
```

## 优先级

1. **高**: 集成 fix_dropcap 到构建流程（立即可做）
2. **中**: 修复短内容跳过问题
3. **低**: 完整的 QA Agent（较大工作量，但能根治所有问题）

## 相关文件

- `/scripts/fix_dropcap.py` - 首字下沉修复脚本
- `/pdf2epub/html_translation/translator.py` - HTML 翻译处理器
- `/pdf2epub/processors/base.py` - 基类处理器（包含 `_is_image_only_content`）
- `/pdf2epub/html_translation/builder.py` - EPUB 构建器
