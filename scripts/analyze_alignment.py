#!/usr/bin/env python3
"""
Analyze HTML structure differences between original and translated EPUB.
"""

import zipfile
from pathlib import Path
from lxml import etree
from collections import defaultdict
import json

def extract_xhtml_from_epub(epub_path: Path) -> dict:
    """Extract all XHTML files from EPUB."""
    files = {}
    with zipfile.ZipFile(epub_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.xhtml') and 'OPS/' in name:
                content = zf.read(name).decode('utf-8')
                filename = Path(name).name
                files[filename] = content
    return files

def parse_html_structure(html_content: str) -> list:
    """Parse HTML and extract structure with text content."""
    parser = etree.HTMLParser()
    tree = etree.fromstring(html_content.encode(), parser)

    # Find body
    body = tree.find('.//body')
    if body is None:
        return []

    elements = []

    def extract_text(elem):
        """Get all text content from element."""
        texts = []
        if elem.text:
            texts.append(elem.text.strip())
        for child in elem:
            if child.tail:
                texts.append(child.tail.strip())
            texts.extend(extract_text(child))
        return [t for t in texts if t]

    def process_element(elem, depth=0):
        """Process element and its children."""
        tag = elem.tag
        if callable(tag):  # Skip comments, PI, etc.
            return

        # Get class attribute
        cls = elem.get('class', '')

        # Get text content (first 50 chars)
        text = ' '.join(extract_text(elem))[:80]

        # Important elements to track
        important_tags = {'h1', 'h2', 'h3', 'h4', 'p', 'div', 'section'}
        important_classes = {'cita', 'niv1', 'txt_courant', 'int_niv1', 'chap_tit'}

        is_important = tag in important_tags or any(c in cls for c in important_classes)

        if is_important and text:
            elements.append({
                'tag': tag,
                'class': cls,
                'text_preview': text,
                'depth': depth
            })

        for child in elem:
            process_element(child, depth + 1)

    process_element(body)
    return elements

def compare_structures(orig_file: str, trans_file: str, orig_content: str, trans_content: str):
    """Compare structure between original and translated."""
    orig_elems = parse_html_structure(orig_content)
    trans_elems = parse_html_structure(trans_content)

    print(f"\n{'='*80}")
    print(f"File: {orig_file}")
    print(f"{'='*80}")
    print(f"Original elements: {len(orig_elems)}, Translated elements: {len(trans_elems)}")

    # Check for structural differences
    mismatches = []

    for i, (orig, trans) in enumerate(zip(orig_elems, trans_elems)):
        if orig['tag'] != trans['tag'] or orig['class'] != trans['class']:
            mismatches.append({
                'index': i,
                'orig': orig,
                'trans': trans
            })

    if mismatches:
        print(f"\nStructural mismatches found: {len(mismatches)}")
        for m in mismatches[:5]:  # Show first 5
            print(f"  [{m['index']}] Original: <{m['orig']['tag']} class='{m['orig']['class']}'> \"{m['orig']['text_preview'][:40]}...\"")
            print(f"       Translated: <{m['trans']['tag']} class='{m['trans']['class']}'> \"{m['trans']['text_preview'][:40]}...\"")

    # Check for citation elements
    orig_cita = [e for e in orig_elems if 'cita' in e['class']]
    trans_cita = [e for e in trans_elems if 'cita' in e['class']]

    if orig_cita:
        print(f"\nCitations in original: {len(orig_cita)}")
        for c in orig_cita:
            print(f"  - \"{c['text_preview'][:60]}...\"")

    if trans_cita:
        print(f"Citations in translated: {len(trans_cita)}")
        for c in trans_cita:
            print(f"  - \"{c['text_preview'][:60]}...\"")

    # Check h3 elements (section headers)
    orig_h3 = [e for e in orig_elems if e['tag'] == 'h3']
    trans_h3 = [e for e in trans_elems if e['tag'] == 'h3']

    if len(orig_h3) != len(trans_h3) or any(len(o['text_preview']) < 50 and len(t['text_preview']) > 100 for o, t in zip(orig_h3, trans_h3)):
        print(f"\nH3 header issues detected:")
        print(f"  Original h3 count: {len(orig_h3)}, Translated h3 count: {len(trans_h3)}")
        for i, (o, t) in enumerate(zip(orig_h3, trans_h3)):
            if len(o['text_preview']) < 50 and len(t['text_preview']) > 100:
                print(f"  [{i}] CONTENT SWAP DETECTED:")
                print(f"       Original h3: \"{o['text_preview']}\"")
                print(f"       Translated h3: \"{t['text_preview']}\" (TOO LONG!)")

    return {
        'file': orig_file,
        'orig_count': len(orig_elems),
        'trans_count': len(trans_elems),
        'mismatches': len(mismatches),
        'orig_cita': len(orig_cita),
        'trans_cita': len(trans_cita),
    }


def main():
    base_dir = Path("/Users/kevinzhou/Github/pdf2epub/output/Excalibur, Durendal, Joyeuse")
    orig_epub = base_dir / "input.epub"
    trans_epub = base_dir / "王者之剑、杜兰德尔、咎瓦尤斯：剑的力量.epub"

    print("Extracting EPUB contents...")
    orig_files = extract_xhtml_from_epub(orig_epub)
    trans_files = extract_xhtml_from_epub(trans_epub)

    print(f"Original files: {len(orig_files)}")
    print(f"Translated files: {len(trans_files)}")

    # Compare each file
    results = []
    for filename in sorted(orig_files.keys()):
        if filename in trans_files:
            # Skip non-content files
            if any(x in filename for x in ['nav', 'toc', 'cover', 'titre']):
                continue
            result = compare_structures(filename, filename, orig_files[filename], trans_files[filename])
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    problem_files = [r for r in results if r['mismatches'] > 0 or r['orig_count'] != r['trans_count']]
    print(f"Files with issues: {len(problem_files)}/{len(results)}")

    for r in problem_files:
        print(f"  {r['file']}: {r['mismatches']} mismatches, elements {r['orig_count']} vs {r['trans_count']}")


if __name__ == "__main__":
    main()
