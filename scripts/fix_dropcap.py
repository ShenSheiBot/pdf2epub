#!/usr/bin/env python3
"""
Fix drop cap (首字下沉) styling issue.

Problem: The 'let' class should only apply to the first character,
but after translation it applies to the entire span content.

Fix: Split the span so only the first character has the 'let' class.
"""

import zipfile
import shutil
import re
from pathlib import Path
from lxml import etree
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")

XHTML_NS = "http://www.w3.org/1999/xhtml"
NSMAP = {None: XHTML_NS}


def fix_dropcap_in_html(content: str) -> str:
    """Fix drop cap spans in HTML content."""
    # Pattern to match span with 'let' class that has more than one character
    # We need to split: <span class="... let ...">多个字符</span>
    # Into: <span class="... let ...">首</span><span class="...">个字符</span>

    def fix_let_span(match):
        full_match = match.group(0)
        before_class = match.group(1)
        classes = match.group(2)
        after_class = match.group(3)
        content = match.group(4)

        # If content is only one character, no fix needed
        if len(content) <= 1:
            return full_match

        # Split: first char keeps 'let', rest goes to new span without 'let'
        first_char = content[0]
        rest = content[1:]

        # Build class without 'let' for the rest
        other_classes = ' '.join(c for c in classes.split() if c != 'let')

        # First span with 'let' class (just first character)
        first_span = f'<span{before_class}class="{classes}"{after_class}>{first_char}</span>'

        # Second span without 'let' class (rest of content)
        if other_classes:
            rest_span = f'<span class="{other_classes}">{rest}</span>'
        else:
            rest_span = rest

        return first_span + rest_span

    # Match <span ...class="...let..."...>content</span>
    # Capture: (before class=") (class value) (after class value") (content)
    pattern = r'<span([^>]*?)class="([^"]*\blet\b[^"]*)"([^>]*)>([^<]+)</span>'

    fixed = re.sub(pattern, fix_let_span, content)
    return fixed


def fix_epub(epub_path: Path, output_path: Path = None):
    """Fix drop caps in all XHTML files in the EPUB."""
    if output_path is None:
        output_path = epub_path

    fixed_files = {}

    with zipfile.ZipFile(epub_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.xhtml'):
                content = zf.read(name).decode('utf-8')
                fixed = fix_dropcap_in_html(content)
                if fixed != content:
                    fixed_files[name] = fixed
                    # Count fixes
                    orig_let_count = len(re.findall(r'class="[^"]*\blet\b[^"]*">[^<]{2,}</span>', content))
                    if orig_let_count > 0:
                        logger.info(f"{Path(name).name}: Fixed {orig_let_count} drop caps")

    if fixed_files:
        # Rewrite the EPUB with fixed files
        temp_epub = epub_path.with_suffix('.temp.epub')
        shutil.move(epub_path, temp_epub)

        with zipfile.ZipFile(temp_epub, 'r') as src_zf:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                for item in src_zf.namelist():
                    if item in fixed_files:
                        dst_zf.writestr(item, fixed_files[item].encode('utf-8'))
                    else:
                        dst_zf.writestr(item, src_zf.read(item))

        temp_epub.unlink()
        logger.success(f"Fixed EPUB saved to: {output_path}")
    else:
        logger.info("No drop cap issues found")


def main():
    base_dir = Path("/Users/kevinzhou/Github/pdf2epub/output/Excalibur, Durendal, Joyeuse")
    input_epub = base_dir / "王者之剑、杜兰德尔、咎瓦尤斯：剑的力量.epub"
    output_epub = base_dir / "王者之剑、杜兰德尔、咎瓦尤斯：剑的力量.epub"  # Overwrite

    fix_epub(input_epub, output_epub)


if __name__ == "__main__":
    main()
