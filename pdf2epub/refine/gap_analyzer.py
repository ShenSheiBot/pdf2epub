"""
Gap detection and filling for TOC structures.

Detects pages not covered by any TOC entry and uses LLM to classify
and generate appropriate titles for the missing content.
"""

import json
import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..utils.network_utils import GeminiClient


@dataclass
class Gap:
    """Represents a gap between TOC entries."""
    gap_type: str  # "front", "toc_to_content", "inter_chapter", "intra_part", "back"
    start_page: int
    end_page: int
    prev_entry: Optional[str]  # Title of previous entry
    next_entry: Optional[str]  # Title of next entry
    parent_entry: Optional[str]  # For nested gaps
    insert_path: Optional[List[int]] = None  # Path to insert location in tree
    parent_level: Optional[int] = None  # Level of parent entry (for intra_part)
    sibling_level: Optional[int] = None  # Level of adjacent sibling (for inter_chapter)


@dataclass
class Overlap:
    """Represents an overlap between consecutive TOC entries."""
    current_entry: Dict  # The earlier entry
    next_entry: Dict  # The later entry
    overlap_start: int  # First overlapping page
    overlap_end: int  # Last overlapping page


@dataclass
class GapClassification:
    """LLM classification of gap content."""
    is_substantial: bool
    content_type: str  # "part_title", "introduction", "appendix", "blank", etc.
    suggested_title: str
    suggested_level: int
    confidence: str  # "high", "medium", "low"
    reasoning: str


class GapAnalyzer:
    """
    Analyzes and fills gaps in TOC structures.

    Detects pages not covered by TOC entries and uses LLM to classify
    content and generate appropriate titles.
    """

    def __init__(self, client: GeminiClient, model: str = "gemini-2.5-flash"):
        """
        Initialize the gap analyzer.

        Args:
            client: GeminiClient for API calls
            model: Model to use for classification
        """
        self.client = client
        self.model = model

    def detect_gaps(self, toc_data: Dict) -> List[Gap]:
        """
        Detect all gaps in the TOC structure.

        Args:
            toc_data: Full toc_tree.json data

        Returns:
            List of Gap objects representing uncovered page ranges
        """
        gaps = []
        chapters = toc_data.get('chapters', [])

        # Get boundary pages (handle None values from JSON null)
        cover_page_data = toc_data.get('cover_page')
        cover_page = cover_page_data.get('page_number', 1) if cover_page_data else 1

        toc_info = toc_data.get('table_of_contents') or {}
        toc_start = toc_info.get('start_page', cover_page + 1)
        toc_end = toc_info.get('end_page', toc_start)

        back_cover_data = toc_data.get('back_cover')
        back_cover = back_cover_data.get('page_number') if back_cover_data else None

        # 1. Front gap: cover to TOC
        if cover_page + 1 < toc_start:
            gaps.append(Gap(
                gap_type="front",
                start_page=cover_page + 1,
                end_page=toc_start - 1,
                prev_entry="Cover",
                next_entry="Table of Contents",
                parent_entry=None,
                insert_path=[0]  # Insert at beginning
            ))

        # 2. TOC to first chapter
        if chapters and toc_end + 1 < chapters[0]['start_page']:
            gaps.append(Gap(
                gap_type="toc_to_content",
                start_page=toc_end + 1,
                end_page=chapters[0]['start_page'] - 1,
                prev_entry="Table of Contents",
                next_entry=chapters[0]['title'],
                parent_entry=None,
                insert_path=[0]  # Insert at beginning
            ))

        # 3. Inter-chapter gaps and nested gaps (recursive)
        chapter_gaps = self._detect_gaps_recursive(chapters, [])
        gaps.extend(chapter_gaps)

        # 4. Back gap
        if chapters and back_cover:
            last_chapter = self._get_last_chapter(chapters)
            if last_chapter and last_chapter['end_page'] < back_cover - 1:
                gaps.append(Gap(
                    gap_type="back",
                    start_page=last_chapter['end_page'] + 1,
                    end_page=back_cover - 1,
                    prev_entry=last_chapter['title'],
                    next_entry="Back Cover",
                    parent_entry=None,
                    insert_path=[len(chapters)]  # Append at end
                ))

        logger.info(f"Detected {len(gaps)} gaps in TOC structure")
        for gap in gaps:
            logger.debug(f"  {gap.gap_type}: pages {gap.start_page}-{gap.end_page}")

        return gaps

    def _detect_gaps_recursive(
        self,
        nodes: List[Dict],
        path: List[int],
        parent_title: str = None
    ) -> List[Gap]:
        """
        Recursively detect gaps within a list of nodes.

        Args:
            nodes: List of chapter/section entries
            path: Current path in tree (list of indices)
            parent_title: Title of parent node

        Returns:
            List of gaps found
        """
        gaps = []

        for i, node in enumerate(nodes):
            current_path = path + [i]

            # Check intra-node gap (parent start to first child start)
            if node.get('children'):
                first_child = node['children'][0]
                # Gap between node start and first child
                if node['start_page'] < first_child['start_page'] - 1:
                    gaps.append(Gap(
                        gap_type="intra_part",
                        start_page=node['start_page'],
                        end_page=first_child['start_page'] - 1,
                        prev_entry=node['title'],
                        next_entry=first_child['title'],
                        parent_entry=node['title'],
                        insert_path=current_path + [0],  # Insert as first child
                        parent_level=node.get('level', 1)  # Pass parent's level
                    ))

                # Recurse into children
                child_gaps = self._detect_gaps_recursive(
                    node['children'],
                    current_path,
                    parent_title=node['title']
                )
                gaps.extend(child_gaps)

            # Check inter-chapter gap to next sibling
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                if node['end_page'] + 1 < next_node['start_page']:
                    gaps.append(Gap(
                        gap_type="inter_chapter",
                        start_page=node['end_page'] + 1,
                        end_page=next_node['start_page'] - 1,
                        prev_entry=node['title'],
                        next_entry=next_node['title'],
                        parent_entry=parent_title,
                        insert_path=path + [i + 1],  # Insert after current
                        sibling_level=node.get('level', 1)  # Use previous sibling's level
                    ))

        return gaps

    def _get_last_chapter(self, chapters: List[Dict]) -> Optional[Dict]:
        """Get the last chapter, including nested children."""
        if not chapters:
            return None

        last = chapters[-1]
        while last.get('children'):
            last = last['children'][-1]
        return last

    def classify_gap(self, gap: Gap, pages_dir: Path) -> GapClassification:
        """
        Send gap pages to LLM for classification.

        Args:
            gap: Gap to classify
            pages_dir: Directory containing page_*.md files

        Returns:
            GapClassification with LLM's analysis
        """
        # Collect content from gap pages
        pages_content = []
        for page_num in range(gap.start_page, gap.end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if page_file.exists():
                content = page_file.read_text(encoding='utf-8')
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:2000] + "\n... [truncated]"
                pages_content.append(f"=== Page {page_num} ===\n{content}")

        if not pages_content:
            logger.warning(f"No page files found for gap {gap.start_page}-{gap.end_page}")
            return GapClassification(
                is_substantial=False,
                content_type="blank",
                suggested_title="",
                suggested_level=0,
                confidence="high",
                reasoning="No page content found"
            )

        all_content = "\n\n".join(pages_content)

        prompt = f"""Analyze gap pages in a book's table of contents structure.

**Gap Information:**
- Type: {gap.gap_type}
- Pages: {gap.start_page} to {gap.end_page} ({gap.end_page - gap.start_page + 1} pages)
- Previous entry: {gap.prev_entry or "None"}
- Next entry: {gap.next_entry or "None"}
- Parent section: {gap.parent_entry or "None (top level)"}

**Page Content:**
{all_content}

**Task:**
Determine:

1. **Is this substantial content?**
   - YES: pages with readable content beyond just a title (text, introductions, descriptions)
   - NO: pages that only contain a section/part title with no other content
   - NO: completely blank pages, pages with only PDF metadata/watermarks

2. **Content type:**
   - "continuation": Content that continues from previous section (no new title/heading, starts mid-paragraph or continues the same topic)
   - "part_title": Part/section title page
   - "introduction": Preface, foreword, or introductory text
   - "appendix": Supplementary material
   - "frontmatter": Title page, copyright, dedication
   - "backmatter": Colophon, blank endpapers
   - "illustration": Full-page illustrations
   - "blank": Empty or nearly empty pages

3. **Suggested title:**
   - Extract from content when possible
   - For blank pages or continuation: empty string

Return JSON:
{{
    "is_substantial": boolean,
    "content_type": string,
    "suggested_title": string,
    "confidence": "high" | "medium" | "low",
    "reasoning": string
}}

**Important:**
- Preserve original language for titles
- For "continuation" type: this content should be merged with the previous entry, not treated as a new section
- Indicators of continuation: no heading/title at page start, text flows from previous context, same writing style
"""

        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.client.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=generation_config,
                operation_name=f"Classify gap: pages {gap.start_page}-{gap.end_page}"
            )

            result = json.loads(response_text)

            # Handle case where LLM returns array instead of object
            if isinstance(result, list):
                result = result[0] if result else {}

            # Compute level based on gap type (deterministic rules)
            content_type = result.get('content_type', 'blank')
            is_substantial = result.get('is_substantial', False)

            # Continuation content should not create a new entry
            if content_type == 'continuation':
                is_substantial = False
                suggested_level = 0
            elif not is_substantial or content_type == 'blank':
                suggested_level = 0
            elif gap.gap_type in ['front', 'back', 'toc_to_content']:
                suggested_level = 1
            elif gap.gap_type == 'intra_part':
                # Child of parent, use parent's level + 1
                suggested_level = (gap.parent_level or 1) + 1
            elif gap.gap_type == 'inter_chapter':
                # Match sibling level
                suggested_level = gap.sibling_level or 1
            else:
                suggested_level = 1

            classification = GapClassification(
                is_substantial=is_substantial,
                content_type=content_type,
                suggested_title=result.get('suggested_title', ''),
                suggested_level=suggested_level,
                confidence=result.get('confidence', 'low'),
                reasoning=result.get('reasoning', '')
            )

            logger.debug(
                f"Gap {gap.start_page}-{gap.end_page}: "
                f"{classification.content_type} - '{classification.suggested_title}' "
                f"(level {classification.suggested_level})"
            )

            return classification

        except Exception as e:
            logger.error(f"Error classifying gap {gap.start_page}-{gap.end_page}: {e}")
            return GapClassification(
                is_substantial=False,
                content_type="error",
                suggested_title="",
                suggested_level=0,
                confidence="low",
                reasoning=str(e)
            )

    def unify_corrections(
        self,
        toc_data: Dict,
        gaps: List[Gap],
        classifications: Dict[int, GapClassification]
    ) -> Dict[int, GapClassification]:
        """
        Send all gaps with initial classifications to LLM for unified correction.

        This allows the LLM to:
        - See the full TOC context
        - Correct OCR errors by comparing with parent/sibling titles
        - Ensure consistent naming across all gaps

        Args:
            toc_data: Full toc_tree.json data
            gaps: List of detected gaps
            classifications: Initial classifications from individual classify_gap calls

        Returns:
            Corrected classifications dict
        """
        # Filter to only substantial gaps
        substantial_gaps = [
            g for g in gaps
            if classifications.get(g.start_page) and
            classifications[g.start_page].is_substantial and
            classifications[g.start_page].suggested_level > 0
        ]

        if not substantial_gaps:
            return classifications

        # Build context: simplified TOC structure
        def simplify_toc(chapters, depth=0):
            result = []
            for ch in chapters:
                indent = "  " * depth
                result.append(f"{indent}- {ch['title']} (pages {ch['start_page']}-{ch['end_page']})")
                if ch.get('children'):
                    result.extend(simplify_toc(ch['children'], depth + 1))
            return result

        toc_context = "\n".join(simplify_toc(toc_data['chapters']))

        # Build gaps list with initial classifications
        gaps_info = []
        for i, gap in enumerate(substantial_gaps):
            cls = classifications[gap.start_page]
            gaps_info.append({
                "index": i,
                "pages": f"{gap.start_page}-{gap.end_page}",
                "gap_type": gap.gap_type,
                "parent": gap.parent_entry,
                "prev": gap.prev_entry,
                "next": gap.next_entry,
                "initial_title": cls.suggested_title,
                "content_type": cls.content_type
            })

        gaps_json = json.dumps(gaps_info, ensure_ascii=False, indent=2)

        prompt = f"""Correct titles for detected gaps in the book's table of contents.

**Book TOC Structure:**
{toc_context}

**Detected Gaps with Initial Classifications:**
{gaps_json}

**Task:**
Correct each gap's title based on TOC context:

1. **OCR Error Correction:**
   - If initial title conflicts with parent/adjacent entries, use parent as reference
   - Common OCR errors: Roman numerals (II→III), numbers, punctuation

2. **Title Format:**
   - For intra_part type (Part title pages), use parent's title

Return JSON array:
[
  {{
    "index": int,
    "corrected_title": string,
    "reason": string
  }}
]

Return only the JSON array.
"""

        generation_config = self.client.get_default_config(temperature=0.1)
        generation_config.response_mime_type = "application/json"

        try:
            response_text = self.client.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=generation_config,
                operation_name="Unify gap corrections"
            )

            corrections = json.loads(response_text)

            # Apply corrections (only title, keep computed level)
            corrected_classifications = classifications.copy()
            for correction in corrections:
                idx = correction['index']
                if idx < len(substantial_gaps):
                    gap = substantial_gaps[idx]
                    old_cls = classifications[gap.start_page]

                    corrected_classifications[gap.start_page] = GapClassification(
                        is_substantial=old_cls.is_substantial,
                        content_type=old_cls.content_type,
                        suggested_title=correction['corrected_title'],
                        suggested_level=old_cls.suggested_level,  # Keep computed level
                        confidence=old_cls.confidence,
                        reasoning=correction.get('reason', old_cls.reasoning)
                    )

                    if correction['corrected_title'] != old_cls.suggested_title:
                        logger.info(
                            f"Corrected '{old_cls.suggested_title}' -> "
                            f"'{correction['corrected_title']}' ({correction.get('reason', '')})"
                        )

            return corrected_classifications

        except Exception as e:
            logger.error(f"Error in unified correction: {e}")
            return classifications

    def fill_gaps(
        self,
        toc_data: Dict,
        gaps: List[Gap],
        classifications: Dict[int, GapClassification]
    ) -> Dict:
        """
        Modify toc_tree.json with filled gaps.

        Args:
            toc_data: Original toc_tree.json data
            gaps: List of detected gaps
            classifications: Dict mapping gap start_page to classification

        Returns:
            Modified toc_data with gaps filled
        """
        modified_data = copy.deepcopy(toc_data)

        # Sort gaps by start_page in reverse order to avoid index shifting
        sorted_gaps = sorted(gaps, key=lambda g: g.start_page, reverse=True)

        filled_count = 0

        for gap in sorted_gaps:
            classification = classifications.get(gap.start_page)

            if not classification:
                continue

            # Handle continuation: extend previous entry's end_page
            if classification.content_type == 'continuation' and gap.prev_entry:
                prev_node = self._find_node_by_title(
                    modified_data['chapters'],
                    gap.prev_entry
                )
                if prev_node:
                    old_end = prev_node['end_page']
                    prev_node['end_page'] = gap.end_page
                    logger.info(
                        f"Extended '{gap.prev_entry}' end_page: "
                        f"{old_end} -> {gap.end_page} (continuation)"
                    )
                continue

            # Skip other non-substantial content
            if not classification.is_substantial or classification.suggested_level == 0:
                logger.debug(
                    f"Skipping gap {gap.start_page}-{gap.end_page}: "
                    f"{classification.content_type}"
                )
                continue

            new_entry = {
                "title": classification.suggested_title,
                "level": classification.suggested_level,
                "start_page": gap.start_page,
                "end_page": gap.end_page,
                "gap_filled": True  # Mark as auto-generated
            }

            # Insert based on gap type and path
            if gap.gap_type in ["front", "toc_to_content"]:
                # Insert at beginning of chapters
                modified_data['chapters'].insert(0, new_entry)
                filled_count += 1
                logger.info(
                    f"Added front matter: '{classification.suggested_title}' "
                    f"(pages {gap.start_page}-{gap.end_page})"
                )

            elif gap.gap_type == "back":
                # Append to chapters
                modified_data['chapters'].append(new_entry)
                filled_count += 1
                logger.info(
                    f"Added back matter: '{classification.suggested_title}' "
                    f"(pages {gap.start_page}-{gap.end_page})"
                )

            elif gap.gap_type == "intra_part":
                # Insert as first child of parent
                parent = self._find_node_by_title(
                    modified_data['chapters'],
                    gap.parent_entry
                )
                if parent:
                    if 'children' not in parent:
                        parent['children'] = []
                    parent['children'].insert(0, new_entry)
                    filled_count += 1
                    logger.info(
                        f"Added part intro: '{classification.suggested_title}' "
                        f"under '{gap.parent_entry}'"
                    )

            elif gap.gap_type == "inter_chapter":
                # Insert between siblings
                success = self._insert_between_siblings(
                    modified_data['chapters'],
                    gap.prev_entry,
                    gap.next_entry,
                    new_entry,
                    gap.parent_entry
                )
                if success:
                    filled_count += 1
                    logger.info(
                        f"Added chapter: '{classification.suggested_title}' "
                        f"between '{gap.prev_entry}' and '{gap.next_entry}'"
                    )

        logger.success(f"Filled {filled_count} gaps in TOC structure")
        return modified_data

    def _find_node_by_title(
        self,
        nodes: List[Dict],
        title: str
    ) -> Optional[Dict]:
        """Find a node by its title (recursive)."""
        for node in nodes:
            if node['title'] == title:
                return node
            if node.get('children'):
                found = self._find_node_by_title(node['children'], title)
                if found:
                    return found
        return None

    def _insert_between_siblings(
        self,
        nodes: List[Dict],
        prev_title: str,
        next_title: str,
        new_entry: Dict,
        parent_title: Optional[str]
    ) -> bool:
        """
        Insert new_entry between two siblings.

        Returns True if successful.
        """
        # If we have a parent, find it first
        if parent_title:
            parent = self._find_node_by_title(nodes, parent_title)
            if parent and parent.get('children'):
                return self._insert_between_siblings(
                    parent['children'],
                    prev_title,
                    next_title,
                    new_entry,
                    None
                )
            return False

        # Find the position to insert
        for i, node in enumerate(nodes):
            if node['title'] == prev_title:
                # Insert after this node
                nodes.insert(i + 1, new_entry)
                return True

            # Check in children
            if node.get('children'):
                if self._insert_between_siblings(
                    node['children'],
                    prev_title,
                    next_title,
                    new_entry,
                    None
                ):
                    return True

        return False

    def detect_overlaps(self, toc_data: Dict) -> List[Overlap]:
        """
        Detect all overlaps in the TOC structure.

        An overlap occurs when one entry's end_page >= next entry's start_page.

        Args:
            toc_data: Full toc_tree.json data

        Returns:
            List of Overlap objects representing overlapping page ranges
        """
        overlaps = []
        chapters = toc_data.get('chapters', [])

        # Recursively detect overlaps
        chapter_overlaps = self._detect_overlaps_recursive(chapters)
        overlaps.extend(chapter_overlaps)

        if overlaps:
            logger.warning(f"Detected {len(overlaps)} overlaps in TOC structure")
            for overlap in overlaps:
                logger.warning(
                    f"  Overlap: '{overlap.current_entry['title']}' "
                    f"(ends {overlap.current_entry['end_page']}) vs "
                    f"'{overlap.next_entry['title']}' "
                    f"(starts {overlap.next_entry['start_page']})"
                )
        else:
            logger.debug("No overlaps detected in TOC structure")

        return overlaps

    def _detect_overlaps_recursive(self, nodes: List[Dict]) -> List[Overlap]:
        """
        Recursively detect overlaps within a list of nodes.

        Args:
            nodes: List of chapter/section entries

        Returns:
            List of overlaps found
        """
        overlaps = []

        for i, node in enumerate(nodes):
            # Check for overlap with next sibling
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                if node['end_page'] >= next_node['start_page']:
                    overlaps.append(Overlap(
                        current_entry=node,
                        next_entry=next_node,
                        overlap_start=next_node['start_page'],
                        overlap_end=node['end_page']
                    ))

            # Recurse into children
            if node.get('children'):
                child_overlaps = self._detect_overlaps_recursive(node['children'])
                overlaps.extend(child_overlaps)

        return overlaps

    def resolve_overlaps(
        self,
        toc_data: Dict,
        overlaps: List[Overlap],
        pages_dir: Path,
        boundary_verifier
    ) -> Dict:
        """
        Resolve overlaps by finding actual title positions using LLM.

        For each overlap, uses BoundaryVerifier to find where the next entry's
        title actually appears, then adjusts both entries' page boundaries.

        Args:
            toc_data: Full toc_tree.json data
            overlaps: List of detected overlaps
            pages_dir: Directory containing page_*.md files
            boundary_verifier: BoundaryVerifier instance for title search

        Returns:
            Modified toc_data with overlaps resolved
        """
        import copy
        from .toc_tree import TOCNode

        modified_data = copy.deepcopy(toc_data)

        for overlap in overlaps:
            # Create a TOCNode for the next entry to use with BoundaryVerifier
            next_entry = overlap.next_entry
            next_node = TOCNode(
                title=next_entry['title'],
                level=next_entry.get('level', 1),
                start_page=next_entry['start_page'],
                end_page=next_entry['end_page']
            )

            # First verify if title is on the expected page
            logger.info(
                f"Resolving overlap: checking '{next_entry['title']}' "
                f"around page {next_entry['start_page']}"
            )

            result = boundary_verifier.verify_boundary(next_node, pages_dir)

            actual_page = None
            if result.get('found'):
                actual_page = next_entry['start_page']
                logger.debug(f"Title found on expected page {actual_page}")
            else:
                # Search nearby pages
                prev_title = overlap.current_entry['title']
                actual_page = boundary_verifier.search_nearby_pages(
                    next_node,
                    pages_dir,
                    search_range=5,
                    prev_title=prev_title
                )

            if actual_page:
                # Update page boundaries in the modified data
                self._update_overlap_boundaries(
                    modified_data['chapters'],
                    overlap.current_entry['title'],
                    overlap.next_entry['title'],
                    actual_page
                )
                logger.success(
                    f"Resolved overlap: '{next_entry['title']}' "
                    f"starts at page {actual_page} "
                    f"(was {next_entry['start_page']})"
                )
            else:
                logger.error(
                    f"Could not resolve overlap for '{next_entry['title']}': "
                    f"title not found in nearby pages"
                )

        return modified_data

    def _update_overlap_boundaries(
        self,
        nodes: List[Dict],
        current_title: str,
        next_title: str,
        actual_start: int
    ) -> bool:
        """
        Update page boundaries to resolve an overlap.

        Sets current entry's end_page to actual_start - 1,
        and next entry's start_page to actual_start.

        Args:
            nodes: List of chapter/section entries
            current_title: Title of the earlier entry
            next_title: Title of the later entry
            actual_start: Actual start page of next entry

        Returns:
            True if updated successfully
        """
        for i, node in enumerate(nodes):
            if node['title'] == current_title and i + 1 < len(nodes):
                next_node = nodes[i + 1]
                if next_node['title'] == next_title:
                    # Update boundaries
                    old_end = node['end_page']
                    old_start = next_node['start_page']

                    node['end_page'] = actual_start - 1
                    next_node['start_page'] = actual_start

                    logger.debug(
                        f"Updated '{current_title}' end: {old_end} -> {actual_start - 1}, "
                        f"'{next_title}' start: {old_start} -> {actual_start}"
                    )
                    return True

            # Recurse into children
            if node.get('children'):
                if self._update_overlap_boundaries(
                    node['children'],
                    current_title,
                    next_title,
                    actual_start
                ):
                    return True

        return False
