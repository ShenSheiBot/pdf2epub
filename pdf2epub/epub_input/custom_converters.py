"""
Custom Markdown Converters for EPUB

Extends markdownify to handle EPUB-specific elements:
- Ruby tags (/G)
- MathML
- Footnotes
- Images with custom path rewriting
"""

from markdownify import MarkdownConverter
from bs4 import BeautifulSoup, NavigableString
from typing import Optional, Callable, Dict, List
from pathlib import Path
from loguru import logger
import re


class EPUBMarkdownConverter(MarkdownConverter):
    """
    Custom Markdown converter for EPUB HTML.

    Handles:
    - Ruby tags -> 漢字{かんじ}
    - MathML -> preserve as-is or convert to LaTeX
    - Images -> rewrite paths
    - Footnotes -> Markdown footnote syntax
    """

    def __init__(
        self,
        image_callback: Optional[Callable[[str, bytes], str]] = None,
        preserve_mathml: bool = True,
        **options
    ):
        """
        Initialize custom converter.

        Args:
            image_callback: Function(src, img_bytes) -> new_path
                          Called when an image is found to save it and get new path
            preserve_mathml: If True, keep MathML as-is; if False, try to convert to LaTeX
            **options: Additional options passed to MarkdownConverter
        """
        super().__init__(**options)
        self.image_callback = image_callback
        self.preserve_mathml = preserve_mathml
        self.image_counter = 0
        self.footnote_refs = []  # Track footnote references

    def convert_ruby(self, el, text, parent_tags=None):
        """
        Convert <ruby> tag to custom syntax: text{ruby}

        Example:
          <ruby>漢<rt>かん</rt>字<rt>じ</rt></ruby>
          -> 漢{かん}字{じ}
        """
        # Remove <rp> tags (fallback parentheses for old browsers)
        for rp in el.find_all('rp'):
            rp.decompose()

        result = ""
        current_base = ""

        for child in el.children:
            if isinstance(child, NavigableString):
                # Base text
                current_base += str(child).strip()
            elif child.name == 'rt':
                # Ruby text (annotation)
                ruby_text = child.get_text(strip=True)
                if current_base:
                    result += f"{current_base}{{{ruby_text}}}"
                    current_base = ""
            elif child.name == 'rb':
                # Explicit base (rare)
                current_base += child.get_text(strip=True)

        # Add any remaining base text without annotation
        if current_base:
            result += current_base

        return result

    def convert_math(self, el, text, parent_tags=None):
        """
        Handle MathML <math> tags.

        If preserve_mathml=True: keep original MathML
        Otherwise: try to convert to LaTeX (simplified)
        """
        if self.preserve_mathml:
            # Keep original MathML
            return str(el)
        else:
            # Simple LaTeX conversion (very basic)
            # For production, use a proper MathML->LaTeX library
            return f"${text}$"

    def convert_img(self, el, text, parent_tags=None):
        """
        Handle images: extract and rewrite path.

        Calls image_callback to save image and get new path.
        """
        src = el.get('src', '')
        alt = el.get('alt', 'Image')

        if not src:
            return ''

        # If callback provided, use it to save image and get new path
        if self.image_callback:
            try:
                # Callback should handle extracting image bytes and saving
                new_path = self.image_callback(src)
                if new_path:
                    return f"![{alt}]({new_path})"
            except Exception as e:
                logger.warning(f"Failed to process image {src}: {e}")

        # Fallback: keep original path
        return f"![{alt}]({src})"

    def convert_sup(self, el, text, parent_tags=None):
        """
        Handle <sup> tags, especially footnote references.

        <sup><a href="#fn1" id="fnref1">1</a></sup> -> [^1]
        """
        # Check if this is a footnote reference
        link = el.find('a')
        if link and link.get('href', '').startswith('#'):
            href = link.get('href')[1:]  # Remove #
            # Common patterns: #fn1, #note-1, #footnote1
            if any(prefix in href for prefix in ['fn', 'note', 'footnote']):
                # Extract number or ID
                ref_id = re.sub(r'[^\w-]', '', href)
                self.footnote_refs.append(ref_id)
                return f"[^{ref_id}]"

        # Not a footnote, keep as superscript
        return f"<sup>{text}</sup>"

    def convert_aside(self, el, text, parent_tags=None):
        """
        Handle <aside> tags, often used for footnotes.

        <aside id="fn1">Footnote text <a href="#fnref1">return</a></aside>
        -> [^fn1]: Footnote text
        """
        aside_id = el.get('id', '')

        # Check if this looks like a footnote
        if any(prefix in aside_id for prefix in ['fn', 'note', 'footnote']):
            # Remove return links (return arrows)
            for a in el.find_all('a'):
                if a.get('href', '').startswith('#fnref') or 'return' in a.get_text().lower():
                    a.decompose()

            footnote_text = el.get_text(strip=True)
            return f"\n[^{aside_id}]: {footnote_text}\n"

        # Not a footnote, treat as blockquote
        return f"\n> {text}\n"

    def convert_div(self, el, text, parent_tags=None):
        """
        Handle <div> tags.

        Check for special classes like footnotes.
        """
        classes = el.get('class', [])

        # Footnote container
        if any(c in classes for c in ['footnote', 'footnotes', 'notes']):
            return f"\n{text}\n"

        # Default: just return content
        return text

    def convert_section(self, el, text, parent_tags=None):
        """Handle <section> tags (common in EPUB 3)."""
        return f"\n{text}\n"

    def convert_nav(self, el, text, parent_tags=None):
        """Skip <nav> tags (TOC navigation)."""
        return ''


def convert_html_to_markdown(
    html_content: str,
    image_callback: Optional[Callable] = None,
    preserve_mathml: bool = True,
    **options
) -> str:
    """
    Convert HTML to Markdown using custom EPUB converter.

    Args:
        html_content: HTML string
        image_callback: Function to handle image extraction
        preserve_mathml: Whether to preserve MathML or convert to LaTeX
        **options: Additional options for MarkdownConverter

    Returns:
        Markdown string
    """
    # Default options
    default_options = {
        'heading_style': 'ATX',  # Use # style headings
        'bullets': '-',           # Use - for lists
        'strip': ['script', 'style'],  # Remove these tags
        'escape_misc': False,     # Don't escape underscores etc. in text
    }
    default_options.update(options)

    converter = EPUBMarkdownConverter(
        image_callback=image_callback,
        preserve_mathml=preserve_mathml,
        **default_options
    )

    markdown = converter.convert(html_content)

    # Post-processing
    # Remove excessive blank lines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    # Ensure proper spacing around headings
    markdown = re.sub(r'(\n#{1,6}\s+.+)\n([^\n])', r'\1\n\n\2', markdown)

    return markdown.strip()
