#!/usr/bin/env python3
"""
Clean Japanese artifacts (っ) from translated Chinese text.
"""
import re
from pathlib import Path
import sys


def clean_japanese_artifacts(text: str) -> str:
    """
    Clean up Japanese artifacts in Chinese translation.
    Specifically removes っ when it appears around punctuation.
    """
    # Define Japanese and Chinese punctuation marks
    punctuation = [
        "。",
        "、",
        "，",
        "！",
        "？",
        "…",
        "～",
        "—",
        "「",
        "」",
        "『",
        "』",
        "（",
        "）",
        "【",
        "】",
        "・",
        "：",
        "；",
        '"',
        '"',
        """, """,
        "」",
        "？",
        "！",
    ]

    # Remove っ before punctuation
    for p in punctuation:
        text = text.replace(f"っ{p}", p)

    # Remove っ after punctuation
    for p in punctuation:
        text = text.replace(f"{p}っ", p)

    # Remove standalone っ surrounded by spaces or punctuation
    # Pattern: punctuation/space + っ + punctuation/space
    pattern = (
        r'([。、，！？…～—「」『』（）【】・：；""'
        '\s])っ([。、，！？…～—「」『』（）【】・：；""'
        "\s])"
    )
    text = re.sub(pattern, r"\1\2", text)

    # Remove っ at the end of quoted speech before closing quotes
    text = re.sub(r'っ([」』"' "])", r"\1", text)

    # Remove っ at the end of sentences before punctuation
    text = re.sub(r"っ([。！？」』])", r"\1", text)

    # Remove standalone っ at the end of lines
    text = re.sub(r"っ\s*$", "", text, flags=re.MULTILINE)

    # Remove っ at the beginning of lines after punctuation
    text = re.sub(r"^([「『])\s*っ", r"\1", text, flags=re.MULTILINE)

    return text


def process_file(file_path: Path):
    """Process a single file to remove Japanese artifacts."""
    print(f"Processing {file_path.name}...")

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Count occurrences before cleaning
    before_count = content.count("っ")

    # Clean the content
    cleaned_content = clean_japanese_artifacts(content)

    # Count occurrences after cleaning
    after_count = cleaned_content.count("っ")

    # Write back if changes were made
    if before_count != after_count:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_content)
        print(
            f"  Cleaned {before_count - after_count} instances of っ (remaining: {after_count})"
        )
    else:
        print(f"  No っ artifacts found to clean")

    return before_count, after_count


def main():
    # Path to translated folder
    translated_dir = Path(
        "output/今さらですが、幼なじみを好きになってしまいました1/translated"
    )

    if not translated_dir.exists():
        print(f"Directory not found: {translated_dir}")
        sys.exit(1)

    # Process all markdown files
    md_files = list(translated_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files to process\n")

    total_before = 0
    total_after = 0

    for file_path in sorted(md_files):
        before, after = process_file(file_path)
        total_before += before
        total_after += after

    print(f"\nSummary:")
    print(f"  Total っ before: {total_before}")
    print(f"  Total っ after: {total_after}")
    print(f"  Total cleaned: {total_before - total_after}")

    if total_after > 0:
        print(
            f"\nNote: {total_after} instances of っ remain. These may be intentional or require manual review."
        )


if __name__ == "__main__":
    main()
