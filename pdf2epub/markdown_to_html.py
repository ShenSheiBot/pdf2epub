#!/usr/bin/env python3
"""
Utility to convert markdown files to HTML with proper EPUB formatting.
Uses python-markdown library with extensions for footnotes, tables, and other features.
"""

import argparse
import re
from pathlib import Path
from typing import Optional
import yaml
from loguru import logger
from .utils.logging_config import configure_logging

# We'll use markdown library - need to add to pyproject.toml: 
# poetry add markdown

# Configure logger
logger = configure_logging()

FOOTNOTE_SYNTAX_RE = re.compile(r'\[\^(\w+)\](?!:)|^\[\^(\w+)\]:', re.MULTILINE)


def contains_footnote_syntax(markdown_content: str) -> bool:
    """Return whether markdown contains supported footnote syntax."""
    return bool(FOOTNOTE_SYNTAX_RE.search(markdown_content))


def load_config(config_path="config.yaml"):
    """Load configuration from config file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def get_epub_css():
    """Get the CSS stylesheet for EPUB HTML files."""
    return """@namespace h "http://www.w3.org/1999/xhtml";
body {
    font-family: "Hiragino Mincho ProN", "MS Mincho", serif;
    line-height: 1.8;
    max-width: 800px;
    margin: 2em auto;
    padding: 0 1em;
}
ruby {
    ruby-align: center;
}
rt {
    font-size: 0.5em;
    font-weight: normal;
}
h1 {
    text-align: center;
    margin-top: 1em;
    margin-bottom: 2em;
    font-weight: bold;
    font-size: 2em;
    border-bottom: 2px solid;
    padding-bottom: 0.5em;
}
h2 {
    font-size: 1.5em;
    font-weight: bold;
    margin-top: 2.5em;
    margin-bottom: 1em;
    border-bottom: 1px solid;
    padding-bottom: 0.3em;
}
h3 {
    font-size: 1.2em;
    font-weight: bold;
    margin-top: 2em;
    margin-bottom: 0.8em;
}
p {
    margin-bottom: 1.2em;
    text-indent: 1em;
    text-align: justify;
}
blockquote {
    margin: 1.5em 2em;
    padding-left: 1em;
    border-left: 3px solid;
    font-style: italic;
}
ul, ol {
    margin: 1em 0;
    padding-left: 2em;
}
li {
    margin-bottom: 0.5em;
}
code {
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: monospace;
}
pre {
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
}
pre code {
    padding: 0;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5em auto;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1.5em 0;
}
th, td {
    border: 1px solid;
    padding: 0.5em;
    text-align: left;
}
th {
    font-weight: bold;
}
.footnote {
    font-size: 0.9em;
    vertical-align: super;
}
.footnotes {
    margin-top: 4em;
    padding-top: 1em;
    border-top: 1px solid #ccc;
    font-size: 0.9em;
}
.footnotes h2 {
    font-size: 1.2em;
    border-bottom: none;
    margin-bottom: 1em;
}
.footnotes ol {
    padding-left: 1.5em;
    list-style-type: decimal;
}
.footnote-item {
    margin-bottom: 0.8em;
    line-height: 1.6;
}
sup {
    font-size: 0.8em;
    vertical-align: super;
}
sup a {
    text-decoration: none;
}
sup a:hover {
    text-decoration: underline;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}
"""


def process_ruby_text(markdown_content: str) -> str:
    """
    Convert Japanese ruby text format (kanji(kana)) to HTML ruby tags.
    
    Examples:
    - 玄関(げんかん) -> <ruby>玄関<rt>げんかん</rt></ruby>
    - 一人(ひとり) -> <ruby>一人<rt>ひとり</rt></ruby>
    - 幼馴染(おさななじみ) -> <ruby>幼馴染<rt>おさななじみ</rt></ruby>
    - 今(いま) -> <ruby>今<rt>いま</rt></ruby>  (single kanji)
    """
    # Pattern to match kanji followed by parentheses containing only hiragana/katakana
    # This pattern handles:
    # - Multiple kanji characters
    # - Single kanji character
    # - Reading in parentheses containing only kana (hiragana or katakana)
    
    # Basic kana ranges
    hiragana_range = r'\u3040-\u309F'
    katakana_range = r'\u30A0-\u30FF'
    kanji_range = r'\u4E00-\u9FAF\u3400-\u4DBF'  # CJK Unified Ideographs
    
    # Pattern: One or more kanji followed by parentheses containing only kana
    # Using lookahead to ensure we don't match if there's already a <ruby> tag
    pattern = rf'(?<!<ruby>)(([{kanji_range}]+)\(([{hiragana_range}{katakana_range}ー]+)\))'
    
    def replace_ruby(match):
        full_match = match.group(1)
        kanji = match.group(2)
        reading = match.group(3)
        
        # Count the number of kanji characters
        kanji_count = len(kanji)
        reading_chars = len(reading)
        
        # Simple heuristic: if reading is much longer than kanji, it's probably a valid ruby
        # This helps avoid false positives
        if reading_chars > kanji_count * 5:
            # Probably not a valid ruby text, return as-is
            return full_match
        
        return f'<ruby>{kanji}<rt>{reading}</rt></ruby>'
    
    # Apply the replacement
    markdown_content = re.sub(pattern, replace_ruby, markdown_content)
    
    return markdown_content


def preprocess_markdown(markdown_content: str, footnote_manager=None, source_chapter: Optional[str] = None, image_mapping: Optional[dict] = None) -> str:
    """
    Pre-process markdown to fix various issues before conversion.

    Args:
        markdown_content: The markdown text to process
        footnote_manager: Optional FootnoteManager for cross-chapter footnote handling
        source_chapter: The source chapter name for footnote linking
        image_mapping: Optional dict mapping original image names to new names
    """
    # First, apply image mapping to markdown images if provided
    if image_mapping:
        for original_name, new_name in image_mapping.items():
            # Handle various image reference patterns in markdown
            markdown_content = markdown_content.replace(f']({original_name})', f']({new_name})')
            markdown_content = markdown_content.replace(f'](../{original_name})', f'](../{new_name})')
            markdown_content = markdown_content.replace(f'](../images/{original_name})', f'](../images/{new_name})')
            markdown_content = markdown_content.replace(f'](images/{original_name})', f'](images/{new_name})')

    # Then, convert markdown images inside HTML tags (like <figure>) to HTML img tags
    markdown_content = re.sub(
        r'!\[([^\]]*)\]\(([^\)]+)\)',
        r'<img alt="\1" src="\2" />',
        markdown_content
    )

    # Then, process Japanese ruby text
    markdown_content = process_ruby_text(markdown_content)
    
    # Handle markdown italics: *text* -> <em>text</em>
    # Use negative lookahead/lookbehind to avoid matching bold (**text**)
    # and to avoid matching asterisks that are part of LaTeX or other constructs
    markdown_content = re.sub(r'(?<!\*)\*(?!\*)([^\*\n]+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', markdown_content)
    
    # Then fix LaTeX-style table footnote markers like ${e}$ 
    # Convert them to superscript letters
    latex_markers = {
        '${a}$': '<sup>a</sup>',
        '${b}$': '<sup>b</sup>',
        '${c}$': '<sup>c</sup>',
        '${d}$': '<sup>d</sup>',
        '${e}$': '<sup>e</sup>',
        '${f}$': '<sup>f</sup>',
        '${g}$': '<sup>g</sup>',
        '${h}$': '<sup>h</sup>',
        '${i}$': '<sup>i</sup>',
        '${j}$': '<sup>j</sup>',
    }
    
    for latex, html in latex_markers.items():
        markdown_content = markdown_content.replace(latex, html)
    
    # Handle LaTeX math expressions with superscripts
    # Convert $number^{letter}$ or $number^{\mathrm{letter}}$ to number<sup>letter</sup>
    # Updated pattern to handle decimals, commas, negative numbers, and both upper/lowercase
    markdown_content = re.sub(r'\$([-0-9,\.]+)\^{\\mathrm{([a-zA-Z])}}\$', r'\1<sup>\2</sup>', markdown_content)
    markdown_content = re.sub(r'\$([-0-9,\.]+)\^{([a-zA-Z])}\$', r'\1<sup>\2</sup>', markdown_content)
    
    # Handle underlined text like $\underline{8}$ or $\underline{text}$
    markdown_content = re.sub(r'\$\\underline\{([^}]+)\}\$', r'<u>\1</u>', markdown_content)
    
    # Handle numbers with plus/minus signs like $200+$ or $50-$
    markdown_content = re.sub(r'\$([\d,\.]+)([\+\-])\$', r'\1\2', markdown_content)
    
    # Handle numbers with values in parentheses like $255(36)$
    markdown_content = re.sub(r'\$(\d+)\((\d+)\)\$', r'\1(\2)', markdown_content)
    
    # Handle negative numbers in parentheses like $(-17)$
    markdown_content = re.sub(r'\$\((-?\d+)\)\$', r'(\1)', markdown_content)
    
    # Handle years in parentheses like $(1983)$
    markdown_content = re.sub(r'\$\((\d{4})\)\$', r'(\1)', markdown_content)
    
    # Handle escaped dollar signs: $\$ -> placeholder
    markdown_content = re.sub(r'\$\\\$', '<<<ESCAPED_DOLLAR>>>', markdown_content)

    # Handle standalone escaped dollar signs: \$ -> placeholder
    # IMPORTANT: Use placeholder to avoid interfering with $...$ LaTeX block matching
    markdown_content = markdown_content.replace(r'\$', '<<<ESCAPED_DOLLAR>>>')

    # OCR/refine placeholders such as "$$$" mean an unknown page number, not math.
    # Protect them before the $$...$$ display-math pass so they cannot consume
    # following paragraphs or footnote definitions.
    dollar_run_placeholders = {}

    def protect_dollar_run(match):
        token = f"<<<DOLLAR_RUN_{len(dollar_run_placeholders)}>>>"
        dollar_run_placeholders[token] = match.group(0)
        return token

    markdown_content = re.sub(r'\${3,}', protect_dollar_run, markdown_content)

    # === DISPLAY MATH: Convert $$...$$ blocks to MathML ===
    # Process display math BEFORE inline math to avoid conflicts
    from latex2mathml import converter

    def process_display_math(match):
        """
        Convert display math $$...$$ to MathML using latex2mathml.

        Display math is rendered as block-level centered mathematical expressions.
        """
        latex_code = match.group(1).strip()

        if not latex_code:
            # Empty display math block, return placeholder
            return '<div class="math-display"></div>'

        try:
            # Convert LaTeX to MathML
            mathml = converter.convert(latex_code)

            # Wrap in a centered div for display math
            return f'<div class="math-display" style="text-align: center; margin: 1em 0;">{mathml}</div>'
        except Exception as e:
            logger.warning(f"Failed to convert display math to MathML: {latex_code[:50]}... Error: {e}")
            # Fallback: keep original LaTeX in a styled div
            return f'<div class="math-display" style="text-align: center; margin: 1em 0; font-style: italic;">$${latex_code}$$</div>'

    # Match $$...$$ blocks (multiline, non-greedy)
    # Use DOTALL flag to allow matching across newlines
    markdown_content = re.sub(r'\$\$\s*\n?(.*?)\n?\s*\$\$', process_display_math, markdown_content, flags=re.DOTALL)

    # === INLINE MATH: Use unicodeitplus for LaTeX to Unicode conversion ===
    # Fallback to MathML for complex expressions
    from unicodeitplus import replace as latex_to_unicode

    def process_latex_block(match):
        r"""
        Process a single $...$ LaTeX block with Unicode-first, MathML-fallback strategy.

        Strategy:
        1. Try preprocessing for known patterns
        2. Try unicodeitplus for Unicode conversion
        3. If conversion incomplete (still has LaTeX commands), use MathML
        4. If MathML fails, keep original LaTeX

        Preprocessing:
        - \qquad → double space (unicodeitplus doesn't handle this)
        - ^{\prime} → \prime (for prime symbol conversion)
        - \underline{X} → <u>X</u> (for HTML underline instead of Unicode combining)
        - \operatorname{X} → X (unicodeitplus doesn't handle this well)

        Unicodeitplus handles:
        - Greek letters (\alpha, \Delta, etc.)
        - Superscripts and subscripts (x^{2}, x_{i})
        - Math symbols (\cdot, \cdots, \ldots, \quad, \%)
        - Text commands (\mathrm{X}, \text{X})
        - Bar notation (\bar{x})

        MathML fallback for:
        - Fractions (\frac{a}{b})
        - Square roots (\sqrt{x})
        - Complex expressions that unicodeitplus can't handle
        """
        content = match.group(1)  # Get content between $ and $
        original_content = content  # Keep original for fallback

        # === Preprocessing for patterns unicodeitplus doesn't handle well ===

        # 1. \qquad → double space (unicodeitplus leaves it unchanged)
        content = content.replace(r'\qquad', '  ')

        # 2. Convert ^{\prime} or ^{\prime X} to \prime X (unicodeitplus converts \prime but not ^{\prime})
        # Also handle cases like { }^{\prime 1} where prime is in superscript with other content
        content = re.sub(r'\^\{\\prime\s*([^}]*)\}', r'\\prime\1', content)
        content = re.sub(r'\^\\prime\b', r'\\prime', content)

        # 3. \underline{X} -> <u>X</u> (for HTML underline instead of Unicode combining characters)
        # Do this BEFORE unicodeitplus to avoid it converting to Unicode combining underline
        content = re.sub(r'\\underline\{([^}]+)\}', r'<u>\1</u>', content)

        # 4. \operatorname{X} -> X (unicodeitplus doesn't handle this well)
        content = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', content)

        # === Try unicodeitplus for standard LaTeX to Unicode conversion ===
        try:
            result = latex_to_unicode(content)

            # Check if conversion was complete
            # If result still contains backslash commands (except in HTML tags like <u>),
            # it means unicodeitplus couldn't convert everything
            # Remove HTML tags temporarily for checking
            check_result = re.sub(r'<[^>]+>', '', result)

            if '\\' in check_result:
                # Conversion incomplete, fallback to MathML
                logger.debug(f"Unicode conversion incomplete for: {original_content[:50]}... Trying MathML")
                raise ValueError("Incomplete conversion, use MathML")

            return result

        except Exception as e:
            # === Fallback to MathML for complex expressions ===
            try:
                mathml = converter.convert(original_content)
                logger.debug(f"Using MathML for inline math: {original_content[:50]}...")
                return f'<span class="math-inline">{mathml}</span>'
            except Exception as e2:
                # Last resort: keep original LaTeX with styling
                logger.warning(f"Failed to convert inline math: {original_content[:50]}... Error: {e2}")
                return f'<span style="font-style: italic;">${original_content}$</span>'

    # Apply the callback to all $...$ blocks (inline only, not across newlines)
    # Use [^$\n]+ to match only within a single line
    markdown_content = re.sub(r'\$([^$\n]+)\$', process_latex_block, markdown_content)

    # Convert placeholders back to actual dollar signs
    markdown_content = markdown_content.replace('<<<ESCAPED_DOLLAR>>>', '$')
    for token, dollar_run in dollar_run_placeholders.items():
        markdown_content = markdown_content.replace(token, dollar_run)

    # Now process footnotes
    return preprocess_footnotes(markdown_content, footnote_manager, source_chapter)


def preprocess_footnotes_local(markdown_content: str, footnote_manager, source_chapter: str) -> str:
    """
    Process footnotes in LOCAL mode with multi-part chapter support.

    Handles footnotes within a chapter, including cross-part references.

    Args:
        markdown_content: The markdown text to process
        footnote_manager: FootnoteManager instance in LOCAL mode
        source_chapter: The source chapter name for footnote linking

    Returns:
        Processed markdown with HTML footnote links
    """
    lines = markdown_content.split('\n')
    processed_lines = []
    definition_occurrence_in_file = {}
    reference_occurrence_in_file = {}

    for line_num, line in enumerate(lines, 1):
        # Check for footnote definition [^key]:
        def_match = re.match(r'^\[\^(\w+)\]:\s*(.*)', line)
        if (
            not def_match
            and footnote_manager.is_no_colon_definition_chapter(source_chapter)
        ):
            def_match = re.match(r'^\[\^(\w+)\]\s+(.+)', line)
        if def_match:
            fn_key = def_match.group(1)
            fn_text = def_match.group(2)
            definition_occurrence_in_file[fn_key] = definition_occurrence_in_file.get(fn_key, 0) + 1
            processed_lines.append(
                footnote_manager.get_definition_html(
                    fn_key,
                    fn_text,
                    source_chapter,
                    line_num=line_num,
                    occurrence_in_file=definition_occurrence_in_file[fn_key],
                )
            )
        else:
            # Process footnote references [^key]
            def replace_ref(match):
                fn_key = match.group(1)
                reference_occurrence_in_file[fn_key] = reference_occurrence_in_file.get(fn_key, 0) + 1
                return footnote_manager.get_footnote_html(
                    fn_key,
                    source_chapter,
                    line_num=line_num,
                    occurrence_in_file=reference_occurrence_in_file[fn_key],
                ) or match.group(0)

            # Replace footnote references
            line = re.sub(r'\[\^(\w+)\](?!:)', replace_ref, line)

            # Replace backticks with proper apostrophes/quotes
            # This prevents markdown from interpreting them as code blocks
            # Single backticks are typically used as quotes in this context
            line = line.replace("`", "'")
            processed_lines.append(line)

    return '\n'.join(processed_lines)


def preprocess_footnotes_global(markdown_content: str, footnote_manager, source_chapter: str) -> str:
    """
    Process footnotes in GLOBAL mode - handles cross-chapter footnote references.
    
    For definition chapters: Keep footnote definitions as-is in markdown format
    For reference chapters: Convert footnote references to cross-chapter links
    
    Args:
        markdown_content: The markdown text to process
        footnote_manager: FootnoteManager instance in GLOBAL mode
        source_chapter: The source chapter name for footnote linking
        
    Returns:
        Processed markdown with HTML footnote links
    """
    lines = markdown_content.split('\n')
    processed_lines = []
    reference_occurrence_in_file = {}
    definition_occurrence_in_file = {}
    is_definition_chapter = footnote_manager.is_definition_chapter(source_chapter)

    for line_num, line in enumerate(lines, 1):
        def_match = re.match(r'^\[\^(\w+)\]:\s*(.*)', line)
        if (
            not def_match
            and is_definition_chapter
            and footnote_manager.is_no_colon_definition_chapter(source_chapter)
        ):
            def_match = re.match(r'^\[\^(\w+)\]\s+(.+)', line)
        if def_match and is_definition_chapter:
            fn_key = def_match.group(1)
            fn_text = def_match.group(2)
            definition_occurrence_in_file[fn_key] = definition_occurrence_in_file.get(fn_key, 0) + 1
            processed_lines.append(
                footnote_manager.get_definition_html(
                    fn_key,
                    fn_text,
                    source_chapter,
                    line_num=line_num,
                    occurrence_in_file=definition_occurrence_in_file[fn_key],
                )
            )
            continue

        def replace_ref(match):
            fn_key = match.group(1)
            reference_occurrence_in_file[fn_key] = reference_occurrence_in_file.get(fn_key, 0) + 1
            return footnote_manager.get_footnote_html(
                fn_key,
                source_chapter,
                line_num=line_num,
                occurrence_in_file=reference_occurrence_in_file[fn_key],
            ) or match.group(0)

        line = re.sub(r'\[\^(\w+)\](?!:)', replace_ref, line)
        processed_lines.append(line)

    return '\n'.join(processed_lines)


def preprocess_footnotes(markdown_content: str, footnote_manager=None, source_chapter: Optional[str] = None) -> str:
    """
    Convert footnote definitions to HTML format and handle footnote references.

    Since we're not using the markdown footnotes extension, we need to manually
    convert [^1]: text to proper HTML footnote format.

    Footnote conversion is manager-backed only. This keeps local, split-part, and
    global notes on the same mapping/backref implementation.

    Args:
        markdown_content: The markdown text to process
        footnote_manager: FootnoteManager for local/global footnote handling
        source_chapter: The source chapter name for footnote linking
    """
    if not contains_footnote_syntax(markdown_content):
        return markdown_content

    if not footnote_manager or not source_chapter:
        raise ValueError("Footnote markdown requires FootnoteManager and source_chapter")

    from pdf2epub.epub.footnotes import FootnoteStyle
    if footnote_manager.get_style() == FootnoteStyle.GLOBAL:
        return preprocess_footnotes_global(markdown_content, footnote_manager, source_chapter)
    if footnote_manager.get_style() == FootnoteStyle.LOCAL:
        return preprocess_footnotes_local(markdown_content, footnote_manager, source_chapter)

    raise ValueError(f"Unsupported footnote style: {footnote_manager.get_style()}")


def convert_markdown_to_html(
    markdown_content: str,
    title: Optional[str] = None,
    include_css: bool = True,
    standalone: bool = True,
    image_mapping: Optional[dict] = None,
    footnote_manager=None,
    source_chapter: Optional[str] = None
) -> str:
    """
    Convert markdown content to HTML.
    
    Args:
        markdown_content: The markdown text to convert
        title: Optional title for the HTML document
        include_css: Whether to include CSS styles
        standalone: Whether to create a complete HTML document
        image_mapping: Optional dict mapping original image names to new names
        footnote_manager: Optional FootnoteManager for cross-chapter footnote handling
        source_chapter: The source chapter name (e.g., "chapter_1") for footnote linking
    
    Returns:
        HTML string
    """
    try:
        import markdown
    except ImportError:
        logger.error("markdown library not installed. Run: poetry add markdown")
        raise
    
    # Pre-process markdown to fix various issues (including image mapping)
    markdown_content = preprocess_markdown(markdown_content, footnote_manager, source_chapter, image_mapping)
    
    # Configure markdown extensions
    extensions = [
        # Remove footnotes extension - we'll handle them manually
        'markdown.extensions.tables',       # For table support
        'markdown.extensions.fenced_code',  # For code blocks
        'markdown.extensions.codehilite',   # For code syntax highlighting
        'markdown.extensions.nl2br',        # Convert newlines to <br>
        'markdown.extensions.sane_lists',   # Better list handling
        'markdown.extensions.smarty',       # Smart quotes and dashes
        'markdown.extensions.toc',          # Table of contents
        'markdown.extensions.meta',         # Metadata support
    ]
    
    # Configure extension settings
    extension_configs = {
        'markdown.extensions.codehilite': {
            'css_class': 'highlight',
            'linenums': False,
        },
        'markdown.extensions.toc': {
            'baselevel': 1,
            'permalink': False,
        },
    }
    
    # Create markdown instance with XHTML output for EPUB compatibility
    md = markdown.Markdown(
        extensions=extensions,
        extension_configs=extension_configs,
        output_format='xhtml'  # EPUB requires XHTML, not HTML5
    )
    
    # Convert markdown to HTML
    html_body = md.convert(markdown_content)

    # Post-process HTML
    html_body = post_process_html(html_body)

    if not standalone:
        return html_body
    
    # Create full XHTML document for EPUB
    css = get_epub_css() if include_css else ""
    
    # EPUB requires XHTML 1.1 with proper DOCTYPE
    html = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>{title or 'Chapter'}</title>
    {f'<style type="text/css">{css}</style>' if css else '<link rel="stylesheet" type="text/css" href="../stylesheet.css"/>'}
</head>
<body>
{html_body}
</body>
</html>"""
    
    return html


def post_process_html(html: str) -> str:
    """
    Post-process the HTML to ensure EPUB XHTML 1.1 compatibility.
    """
    # Fix image paths if they're relative and don't already have ../images/
    html = re.sub(
        r'<img ([^>]*?)src="(?!http)(?!\.\./)([^"]+)"',
        r'<img \1src="../images/\2"',
        html
    )
    
    # Ensure footnote references have proper IDs
    html = re.sub(
        r'<sup id="fnref:(\d+)">',
        r'<sup id="fnref\1">',
        html
    )
    
    # Fix footnote backlinks
    html = re.sub(
        r'href="#fnref:(\d+)"',
        r'href="#fnref\1"',
        html
    )
    
    # Ensure proper XHTML self-closing tags
    html = re.sub(r'<br(?!\s*/)>', '<br />', html)
    html = re.sub(r'<hr(?!\s*/)>', '<hr />', html)
    html = re.sub(r'<img ([^>]+?)(?<!/)>', r'<img \1 />', html)
    html = re.sub(r'<meta ([^>]+?)(?<!/)>', r'<meta \1 />', html)
    html = re.sub(r'<link ([^>]+?)(?<!/)>', r'<link \1 />', html)
    html = re.sub(r'<input ([^>]+?)(?<!/)>', r'<input \1 />', html)
    
    # Convert & to &amp; in text (but not in existing entities)
    html = re.sub(r'&(?!(?:[a-zA-Z][a-zA-Z0-9]*|#[0-9]+|#x[0-9a-fA-F]+);)', '&amp;', html)
    
    # Ensure all attributes are quoted
    html = re.sub(r'<(\w+)([^>]*?)(\w+)=([^\s"\'>]+)', r'<\1\2\3="\4"', html)
    
    # Convert common block elements to have proper XHTML structure
    html = re.sub(r'<p>(\s*)</p>', '', html)  # Remove empty paragraphs
    
    return html


def convert_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    include_css: bool = True,
    standalone: bool = True,
    footnote_manager=None,
    source_chapter: Optional[str] = None
) -> bool:
    """
    Convert a markdown file to HTML.
    
    Args:
        input_path: Path to the markdown file
        output_path: Path for the output HTML file (default: same name with .html extension)
        title: Optional title for the HTML document
        include_css: Whether to include CSS in the HTML head
        standalone: Whether to create a complete HTML document
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read markdown file
        with open(input_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Extract title from first H1 if not provided
        if not title:
            match = re.search(r'^#\s+(.+)$', markdown_content, re.MULTILINE)
            if match:
                title = match.group(1)
            else:
                title = input_path.stem
        
        # Convert to HTML
        if source_chapter is None:
            source_chapter = input_path.stem
        if footnote_manager is None and contains_footnote_syntax(markdown_content):
            from pdf2epub.epub.footnotes import FootnoteManager
            footnote_manager = FootnoteManager(input_path.parent)

        html = convert_markdown_to_html(
            markdown_content,
            title=title,
            include_css=include_css,
            standalone=standalone,
            footnote_manager=footnote_manager,
            source_chapter=source_chapter if footnote_manager else None
        )
        
        # Determine output path
        if not output_path:
            output_path = input_path.with_suffix('.html')
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.success(f"Converted {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert markdown files to HTML for EPUB")
    parser.add_argument("input", help="Input markdown file or directory")
    parser.add_argument("-o", "--output", help="Output HTML file or directory")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("--no-css", action="store_true", help="Don't include CSS in output")
    parser.add_argument("--body-only", action="store_true", help="Output only body content (no HTML wrapper)")
    parser.add_argument("--chapter", type=int, help="Convert specific chapter number")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        # Try to interpret as book title from config
        config = load_config(args.config)
        book_title = config.get("title")
        
        if book_title:
            # Check for polished markdown directory
            polished_dir = Path("output") / book_title / "polished_markdown"
            if polished_dir.exists():
                input_path = polished_dir
            else:
                logger.error(f"Polished markdown directory not found: {polished_dir}")
                return
        else:
            logger.error(f"Input path not found: {input_path}")
            return
    
    # Process single file or directory
    if input_path.is_file():
        # Single file conversion
        output_path = Path(args.output) if args.output else None
        convert_file(
            input_path,
            output_path,
            include_css=not args.no_css,
            standalone=not args.body_only
        )
    
    elif input_path.is_dir():
        # Directory conversion
        output_dir = Path(args.output) if args.output else input_path.parent / "html_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all markdown files
        if args.chapter:
            markdown_files = [input_path / f"chapter_{args.chapter}.md"]
        else:
            markdown_files = sorted(input_path.glob("chapter_*.md"))
        
        if not markdown_files:
            logger.error(f"No chapter markdown files found in {input_path}")
            return
        
        logger.info(f"Converting {len(markdown_files)} markdown files to HTML...")
        
        success_count = 0
        footnote_manager = None
        if any(
            contains_footnote_syntax(md_file.read_text(encoding='utf-8'))
            for md_file in markdown_files
            if md_file.exists()
        ):
            from pdf2epub.epub.footnotes import FootnoteManager
            footnote_manager = FootnoteManager(input_path)

        for md_file in markdown_files:
            output_file = output_dir / md_file.with_suffix('.html').name
            if convert_file(
                md_file,
                output_file,
                include_css=not args.no_css,
                standalone=not args.body_only,
                footnote_manager=footnote_manager,
                source_chapter=md_file.stem
            ):
                success_count += 1
        
        logger.info(f"Successfully converted {success_count}/{len(markdown_files)} files")
        logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
