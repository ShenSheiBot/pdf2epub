"""
LLM-based section matching for Notes chapters.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger

from .models import FootnoteDefinition, NotesSection


class LLMSectionMatcher:
    """
    Uses LLM to match Notes chapter sections to TOC chapters.
    """

    def __init__(self, markdown_dir: Path, config=None):
        """
        Initialize the LLM section matcher.

        Args:
            markdown_dir: Directory containing markdown files
            config: Configuration object for LLM calls
        """
        self.markdown_dir = markdown_dir
        self.config = config
        self.notes_sections: List[NotesSection] = []
        self.chapter_to_section: Dict[str, NotesSection] = {}
        self.toc_chapters: List[Dict] = []

    def load_toc_tree(self) -> bool:
        """
        Load toc_tree.json from the output directory.

        Returns:
            True if loaded successfully, False otherwise
        """
        # toc_tree.json is in the parent of markdown_dir (output_dir)
        toc_path = self.markdown_dir.parent / "toc_tree.json"
        if not toc_path.exists():
            logger.warning(f"toc_tree.json not found at {toc_path}")
            return False

        try:
            with open(toc_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)

            # Flatten the TOC tree to get all chapters with unit_ids
            def flatten_chapters(chapters, parent_path=None):
                result = []
                for i, ch in enumerate(chapters):
                    path = (parent_path or []) + [i + 1]
                    unit_id = f"chapter_{path[0]}" if len(path) == 1 else f"chapter_{'_'.join(map(str, path))}"
                    result.append({
                        "unit_id": unit_id,
                        "title": ch.get("title", ""),
                        "type": ch.get("type")
                    })
                    if "children" in ch and ch["children"]:
                        result.extend(flatten_chapters(ch["children"], path))
                return result

            self.toc_chapters = flatten_chapters(toc_data.get("chapters", []))
            logger.debug(f"Loaded {len(self.toc_chapters)} chapters from toc_tree.json")
            return True

        except Exception as e:
            logger.error(f"Error loading toc_tree.json: {e}")
            return False

    def get_notes_structure(self, primary_definition_chapters: Set[str]) -> str:
        """
        Get the Notes chapter structure with footnote definitions removed.

        Args:
            primary_definition_chapters: Set of primary definition chapter names

        Returns:
            String with only headers/titles (footnote content removed)
        """
        # Read all primary definition chapter files
        all_lines = []
        for chapter_name in sorted(primary_definition_chapters):
            file_path = self.markdown_dir / f"{chapter_name}.md"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Remove footnote definition lines
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        # Skip footnote definitions [^key]: content
                        if re.match(r'^\[\^\w+\]:', line):
                            continue
                        # Keep non-empty lines that might be headers
                        stripped = line.strip()
                        if stripped:
                            filtered_lines.append(stripped)
                    all_lines.extend(filtered_lines)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        return '\n'.join(all_lines)

    def match_sections(self, primary_definition_chapters: Set[str]) -> bool:
        """
        Use LLM to match Notes section headers to TOC chapters.

        Args:
            primary_definition_chapters: Set of primary definition chapter names

        Returns:
            True if matching succeeded, False otherwise
        """
        if not self.toc_chapters:
            logger.warning("No TOC chapters loaded")
            return False

        # Try to load cached results first
        cache_path = self.markdown_dir.parent / "footnote_section_matches.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                logger.info(f"Loaded {len(matches)} section matches from cache")
                return self._parse_sections_from_result(matches, primary_definition_chapters)
            except Exception as e:
                logger.warning(f"Failed to load cached matches: {e}")

        # No cache, need to call LLM
        if not self.config:
            logger.warning("No config provided, cannot use LLM for section matching")
            return False

        # Get Notes structure
        notes_structure = self.get_notes_structure(primary_definition_chapters)
        if not notes_structure.strip():
            logger.warning("Notes structure is empty")
            return False

        # Filter TOC to exclude notes chapter itself
        toc_entries = [
            {"unit_id": ch["unit_id"], "title": ch["title"]}
            for ch in self.toc_chapters
            if ch.get("type") != "notes"
        ]

        # Build prompt
        prompt = f"""分析以下 Notes 章节的结构（已删除脚注内容，只保留标题），并将每个标题匹配到对应的 TOC 章节。

Notes 章节结构：
```
{notes_structure}
```

TOC 章节列表：
```json
{json.dumps(toc_entries, ensure_ascii=False, indent=2)}
```

请返回一个 JSON 数组，每个元素包含：
- "header": Notes 中的标题文本（完全匹配原文）
- "unit_id": 对应的 TOC 章节 unit_id

注意：
1. Notes 中可能有 OCR 错误或格式问题，请基于语义理解进行匹配
2. 标题可能是任何格式（如 "## CHAPTER ONE"、"PREFACE"、"第一章" 等）
3. 只返回 JSON 数组，不要其他内容

示例返回格式：
```json
[
  {{"header": "PREFACE", "unit_id": "chapter_3"}},
  {{"header": "INTRODUCTION", "unit_id": "chapter_4"}},
  {{"header": "CHAPTER ONE", "unit_id": "chapter_5"}}
]
```
"""

        try:
            from ...utils.llm_client import LLMClient

            llm_client = LLMClient(self.config)

            # Get model config
            model_configs = self.config.get("translation.models", [
                {"provider": "gemini", "model": "gemini-2.5-flash", "max_retries": 2}
            ])

            response = llm_client.generate(
                prompt=prompt,
                model_configs=model_configs,
                operation_name="Match notes sections to chapters"
            )

            # Parse JSON response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code block
                lines = response.split('\n')
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = '\n'.join(json_lines)

            matches = json.loads(response)
            logger.info(f"LLM returned {len(matches)} section matches")

            # Save matches to cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(matches, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved section matches to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save matches to cache: {e}")

            # Parse sections from LLM result
            return self._parse_sections_from_result(matches, primary_definition_chapters)

        except Exception as e:
            logger.error(f"Error in LLM section matching: {e}")
            return False

    def _parse_sections_from_result(
        self,
        matches: List[Dict],
        primary_definition_chapters: Set[str]
    ) -> bool:
        """
        Parse Notes sections based on LLM matching results.

        Args:
            matches: List of {"header": str, "unit_id": str} from LLM
            primary_definition_chapters: Set of primary definition chapter names

        Returns:
            True if parsing succeeded, False otherwise
        """
        if not matches:
            return False

        # Read all primary definition chapter content with line numbers
        all_content = []  # [(line_num, line_text, source_file, local_line_num), ...]
        global_line_num = 0

        for chapter_name in sorted(primary_definition_chapters):
            file_path = self.markdown_dir / f"{chapter_name}.md"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for local_line_num, line in enumerate(lines, 1):
                        all_content.append((global_line_num, line, chapter_name, local_line_num))
                        global_line_num += 1
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        if not all_content:
            return False

        # Find header positions in the content
        header_positions = []  # [(global_line_num, header_text, unit_id, source_file, local_line_num), ...]

        # Create a mapping from header text to unit_id
        header_to_unit_id = {}
        for match in matches:
            header = match.get("header", "").strip()
            unit_id = match.get("unit_id", "")
            if header and unit_id:
                # Normalize header for matching
                header_normalized = re.sub(r'^#+\s*', '', header).strip()
                header_to_unit_id[header_normalized] = unit_id

        # Scan content to find headers in order
        for global_ln, line, source_file, local_ln in all_content:
            line_stripped = line.strip()
            # Remove markdown header markers
            line_clean = re.sub(r'^#+\s*', '', line_stripped)

            # Check if this line matches any header
            if line_clean in header_to_unit_id:
                unit_id = header_to_unit_id[line_clean]
                header_positions.append((global_ln, line_clean, unit_id, source_file, local_ln))
                # Remove from dict to avoid duplicates
                del header_to_unit_id[line_clean]

        # Log any unmatched headers
        for header in header_to_unit_id:
            logger.warning(f"Header '{header}' not found in Notes content")

        if not header_positions:
            logger.warning("No headers found in Notes content")
            return False

        # Sort by position
        header_positions.sort(key=lambda x: x[0])

        # Create sections
        self.notes_sections = []

        for i, (global_ln, header, unit_id, source_file, local_ln) in enumerate(header_positions):
            # Determine end position
            if i + 1 < len(header_positions):
                end_global_ln = header_positions[i + 1][0]
            else:
                end_global_ln = len(all_content)

            # Collect definitions in this section
            section_definitions = []
            for g_ln, line, src_file, loc_ln in all_content:
                if global_ln <= g_ln < end_global_ln:
                    # Check for footnote definition
                    def_match = re.match(r'^\[\^(\w+)\]:\s*(.*)', line)
                    if def_match:
                        key = def_match.group(1)
                        content = def_match.group(2)
                        section_definitions.append(
                            FootnoteDefinition(key, content, src_file, loc_ln)
                        )

            section = NotesSection(
                header_text=header,
                start_line=local_ln,
                end_line=local_ln + (end_global_ln - global_ln),
                source_file=source_file,
                definitions=section_definitions,
                matched_unit_id=unit_id
            )
            self.notes_sections.append(section)

            # Build chapter to section mapping
            self.chapter_to_section[unit_id] = section
            logger.debug(f"Section '{header}' -> {unit_id}: {len(section_definitions)} definitions")

        logger.info(f"Parsed {len(self.notes_sections)} sections from Notes chapter")
        return True
