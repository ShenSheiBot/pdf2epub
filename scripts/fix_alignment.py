#!/usr/bin/env python3
"""
Fix HTML alignment issues in translated EPUB.

Strategy:
1. Use original HTML as structural template
2. Extract translated text content
3. Match translated content to original positions using text similarity
4. Rebuild HTML with correct structure + translated content
"""

import zipfile
import shutil
from pathlib import Path
from lxml import etree
from difflib import SequenceMatcher
import re
from typing import List, Dict, Tuple, Optional
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")


class HTMLAlignmentFixer:
    """Fix alignment issues between original and translated HTML."""

    # Elements that contain translatable content
    CONTENT_ELEMENTS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'figcaption'}

    # Classes that mark structural containers (not directly translatable)
    CONTAINER_CLASSES = {'dev', 'niv1', 'cita', 'chap_debut', 'part_debut'}

    def __init__(self, orig_epub: Path, trans_epub: Path, output_epub: Path):
        self.orig_epub = orig_epub
        self.trans_epub = trans_epub
        self.output_epub = output_epub

    def extract_text(self, elem) -> str:
        """Extract all text content from an element."""
        texts = []
        if elem.text:
            texts.append(elem.text.strip())
        for child in elem:
            # Recurse first to get child's content (handles child.text)
            texts.append(self.extract_text(child))
            # Then get the tail (text after child element)
            if child.tail:
                texts.append(child.tail.strip())
        return ' '.join(t for t in texts if t)

    def get_content_elements(self, root) -> List[Tuple[etree._Element, str, str]]:
        """
        Get all content-bearing elements with their text.
        Returns: [(element, tag_info, text_content), ...]
        """
        elements = []
        body = root.find('.//{http://www.w3.org/1999/xhtml}body')
        if body is None:
            body = root.find('.//body')
        if body is None:
            return elements

        def process(elem, depth=0):
            if not isinstance(elem.tag, str):
                return

            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            cls = elem.get('class', '')

            text = self.extract_text(elem)

            # Check if this is a content element
            is_content = (
                tag in self.CONTENT_ELEMENTS or
                any(c in cls for c in ['txt_courant', 'int_niv1', 'chap_tit', 'part_tit', 'ssalinea'])
            )

            if is_content and text and len(text) > 5:
                tag_info = f"{tag}.{cls}" if cls else tag
                elements.append((elem, tag_info, text))

            for child in elem:
                process(child, depth + 1)

        process(body)
        return elements

    def similarity(self, a: str, b: str) -> float:
        """Calculate text similarity ratio."""
        # Normalize strings
        a = re.sub(r'\s+', ' ', a.strip().lower())
        b = re.sub(r'\s+', ' ', b.strip().lower())
        return SequenceMatcher(None, a[:500], b[:500]).ratio()

    def find_best_match(self, target_text: str, candidates: List[Tuple[int, str]], threshold: float = 0.3) -> Optional[int]:
        """Find the best matching candidate for target text."""
        best_idx = None
        best_score = threshold

        for idx, cand_text in candidates:
            score = self.similarity(target_text, cand_text)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def replace_element_content(self, elem, new_text: str):
        """Replace all text content in an element while preserving structure."""
        # Clear ALL text content first
        def clear_text(e):
            e.text = ''
            for child in e:
                child.tail = ''
                if isinstance(child.tag, str):
                    clear_text(child)

        clear_text(elem)

        # For simple elements, just replace text
        if len(elem) == 0:
            elem.text = new_text
            return

        # Find first text-bearing child (usually span with class koboSpan or let)
        first_span = None
        for span in elem.iter():
            if not isinstance(span.tag, str):
                continue
            tag = span.tag.split('}')[-1] if '}' in span.tag else span.tag
            if tag == 'span':
                cls = span.get('class', '')
                if 'koboSpan' in cls or 'let' in cls or not cls:
                    first_span = span
                    break

        if first_span is not None:
            first_span.text = new_text
        else:
            # Just set the element's text
            elem.text = new_text

    def fix_file(self, orig_content: str, trans_content: str, filename: str) -> str:
        """Fix alignment in a single file."""
        # Parse both files
        parser = etree.HTMLParser(encoding='utf-8')
        orig_tree = etree.fromstring(orig_content.encode('utf-8'), parser)
        trans_tree = etree.fromstring(trans_content.encode('utf-8'), parser)

        # Get content elements from both
        orig_elements = self.get_content_elements(orig_tree)
        trans_elements = self.get_content_elements(trans_tree)

        logger.info(f"{filename}: {len(orig_elements)} orig elements, {len(trans_elements)} trans elements")

        if len(orig_elements) == 0 or len(trans_elements) == 0:
            logger.warning(f"{filename}: No content elements found, skipping")
            return trans_content

        if len(orig_elements) == len(trans_elements):
            logger.info(f"{filename}: Element counts match, checking for misalignment")

        # Build mapping: for each original element, find best matching translated content
        trans_texts = [(i, self.extract_text(elem)) for i, (elem, _, _) in enumerate(trans_elements)]
        used_trans = set()
        mapping = {}  # orig_idx -> trans_idx

        # First pass: exact or near-exact matches
        for orig_idx, (orig_elem, orig_tag, orig_text) in enumerate(orig_elements):
            # Look for matching translated content
            remaining = [(i, t) for i, t in trans_texts if i not in used_trans]
            match_idx = self.find_best_match(orig_text, remaining, threshold=0.5)

            if match_idx is not None:
                mapping[orig_idx] = match_idx
                used_trans.add(match_idx)

        # Second pass: for unmatched, try sequential matching
        unmatched_orig = [i for i in range(len(orig_elements)) if i not in mapping]
        unmatched_trans = [i for i in range(len(trans_elements)) if i not in used_trans]

        # For remaining, match by position with offset detection
        if unmatched_orig and unmatched_trans:
            # Detect offset
            if len(mapping) > 0:
                offsets = [mapping[k] - k for k in sorted(mapping.keys())]
                common_offset = max(set(offsets), key=offsets.count) if offsets else 0
            else:
                common_offset = 0

            for orig_idx in unmatched_orig:
                expected_trans_idx = orig_idx + common_offset
                if expected_trans_idx in unmatched_trans:
                    mapping[orig_idx] = expected_trans_idx
                    unmatched_trans.remove(expected_trans_idx)
                elif unmatched_trans:
                    # Take closest available
                    closest = min(unmatched_trans, key=lambda x: abs(x - expected_trans_idx))
                    mapping[orig_idx] = closest
                    unmatched_trans.remove(closest)

        # Now apply the mapping: copy original structure, fill with translated content
        # We'll work with the original tree and replace text content

        result_tree = etree.fromstring(orig_content.encode('utf-8'), parser)
        result_elements = self.get_content_elements(result_tree)

        changes = 0
        for orig_idx, (result_elem, tag_info, orig_text) in enumerate(result_elements):
            if orig_idx in mapping:
                trans_idx = mapping[orig_idx]
                trans_elem, _, trans_text = trans_elements[trans_idx]

                # Check if we need to change content
                current_text = self.extract_text(result_elem)
                if self.similarity(current_text, trans_text) < 0.9:
                    self.replace_element_content(result_elem, trans_text)
                    changes += 1

        logger.info(f"{filename}: Made {changes} content replacements")

        # Convert back to string
        result = etree.tostring(result_tree, encoding='unicode', method='html')

        # Fix up the HTML to be valid XHTML
        result = self.cleanup_xhtml(result, orig_content)

        return result

    def cleanup_xhtml(self, html: str, original: str) -> str:
        """Clean up HTML to match original XHTML format."""
        # Extract original XML declaration and doctype
        xml_decl = ''
        if original.startswith('<?xml'):
            xml_decl = original[:original.index('?>') + 2]

        # Get the original html tag with namespaces
        html_match = re.search(r'<html[^>]*>', original)
        if html_match:
            orig_html_tag = html_match.group(0)
        else:
            orig_html_tag = '<html>'

        # Replace html tag
        html = re.sub(r'<html[^>]*>', orig_html_tag, html)

        # Add XML declaration if it was present
        if xml_decl and not html.startswith('<?xml'):
            html = xml_decl + '\n' + html

        # Fix self-closing tags for XHTML
        void_elements = ['br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr']
        for tag in void_elements:
            html = re.sub(f'<{tag}([^>]*)(?<!/)>', f'<{tag}\\1/>', html)

        return html

    def fix_epub(self):
        """Fix all HTML files in the EPUB."""
        # Copy translated EPUB as base
        shutil.copy(self.trans_epub, self.output_epub)

        # Extract files from both EPUBs
        with zipfile.ZipFile(self.orig_epub, 'r') as orig_zf, \
             zipfile.ZipFile(self.output_epub, 'a') as out_zf:

            orig_files = {Path(n).name: n for n in orig_zf.namelist() if n.endswith('.xhtml')}

            # Get list of files to fix
            files_to_fix = []
            for name in out_zf.namelist():
                if name.endswith('.xhtml'):
                    filename = Path(name).name
                    # Only fix chapter files (pXchapY.xhtml and pX.xhtml)
                    if re.match(r'p\d+(chap\d+)?\.xhtml', filename):
                        if filename in orig_files:
                            files_to_fix.append((name, orig_files[filename]))

            logger.info(f"Files to fix: {len(files_to_fix)}")

            # Process each file
            fixed_files = {}
            for trans_path, orig_path in files_to_fix:
                filename = Path(trans_path).name
                logger.info(f"\nProcessing {filename}...")

                orig_content = orig_zf.read(orig_path).decode('utf-8')
                trans_content = out_zf.read(trans_path).decode('utf-8')

                fixed_content = self.fix_file(orig_content, trans_content, filename)
                fixed_files[trans_path] = fixed_content

        # Write fixed files back to EPUB
        # We need to recreate the EPUB to update files
        temp_epub = self.output_epub.with_suffix('.temp.epub')
        shutil.copy(self.output_epub, temp_epub)

        with zipfile.ZipFile(temp_epub, 'r') as src_zf:
            with zipfile.ZipFile(self.output_epub, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
                for item in src_zf.namelist():
                    if item in fixed_files:
                        dst_zf.writestr(item, fixed_files[item].encode('utf-8'))
                    else:
                        dst_zf.writestr(item, src_zf.read(item))

        temp_epub.unlink()
        logger.success(f"Fixed EPUB saved to: {self.output_epub}")


def main():
    base_dir = Path("/Users/kevinzhou/Github/pdf2epub/output/Excalibur, Durendal, Joyeuse")
    orig_epub = base_dir / "input.epub"
    trans_epub = base_dir / "王者之剑、杜兰德尔、咎瓦尤斯：剑的力量.epub"
    output_epub = base_dir / "王者之剑、杜兰德尔、咎瓦尤斯：剑的力量_fixed.epub"

    fixer = HTMLAlignmentFixer(orig_epub, trans_epub, output_epub)
    fixer.fix_epub()


if __name__ == "__main__":
    main()
