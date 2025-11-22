"""
Main workflow coordinator for refined breakdown.

Orchestrates the entire refinement process:
1. Analyze PDF structure
2. Verify boundaries
3. Handle failures (re-breakdown, discover subsections)
4. Generate work units
5. Merge pages and save
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.network_utils import GeminiClient
from ..utils.pdf_utils import preprocess_pdf
from ..utils.unit_id import generate_unit_id
from .toc_tree import TOCNode, dict_list_to_toc_tree
from .refiner_state import RefinerState
from .boundary_verifier import BoundaryVerifier
from .structure_analyzer import StructureAnalyzer
from .page_merger import PageMerger
from .gap_analyzer import GapAnalyzer

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


class RefinedBreakdown:
    """
    Main class for refined breakdown process.

    Workflow:
    1. Analyze PDF structure (extract recursive TOC tree)
    2. Verify boundaries for each node
    3. Handle verification failures
    4. Generate work units based on token limits
    5. Merge pages with precise boundary cutting
    """

    def __init__(
        self,
        config: Dict,
        max_tokens: int = 8000,
        max_workers: int = None,
    ):
        """
        Initialize the refined breakdown processor.

        Args:
            config: Configuration dict (from config.yaml)
            max_tokens: Maximum tokens per unit (LLM limit)
            max_workers: Maximum parallel workers for verification
        """
        self.config = config
        self.max_tokens = max_tokens
        self.max_workers = max_workers or config.get('general', {}).get('max_concurrent_workers', 8)

        # Initialize Gemini client
        credentials = config.get('credentials', {}).get('providers', {})
        gemini_config = credentials.get('gemini', {})
        api_key = gemini_config.get('api_key')
        base_url = gemini_config.get('base_url')

        if not api_key:
            raise ValueError("Gemini API key not found in config")

        self.client = GeminiClient(api_key, base_url=base_url, num_retries=3, max_backoff_seconds=30)

        # Get models from config
        refine_config = config.get('refine', {})
        structure_model = refine_config.get('structure_model', 'gemini-2.5-pro')
        verification_model = refine_config.get('verification_model', 'gemini-2.5-flash')

        # Initialize components
        self.structure_analyzer = StructureAnalyzer(
            self.client, structure_model, verification_model
        )
        self.boundary_verifier = BoundaryVerifier(self.client, verification_model)
        self.gap_analyzer = GapAnalyzer(self.client, verification_model)
        self.page_merger = PageMerger()
        self.state = RefinerState()

    def process(
        self,
        pdf_path: Path,
        output_dir: Path,
        book_title: str,
        resume: bool = False
    ) -> List[Dict]:
        """
        Main entry point: process PDF and generate work units.

        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory
            book_title: Book title for prompts
            resume: Resume from previous state

        Returns:
            List of work unit metadata dicts
        """
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = output_dir / "pages"
        ocr_markdown_dir = output_dir / "ocr_markdown"

        # Check if pages exist
        if not pages_dir.exists() or not list(pages_dir.glob("page_*.md")):
            raise ValueError(f"Pages not found in {pages_dir}. Run 'pdf2epub ocr-pages' first.")

        # Load state if resuming
        state_file = output_dir / "refiner_state.json"
        if resume and state_file.exists():
            self.state.load(state_file)
            logger.info(f"Resumed: {len(self.state.verified_nodes)} nodes verified")

        # Check if ocr_markdown exists but has no tree_progress.json
        tree_progress_file = ocr_markdown_dir / "tree_progress.json"
        if ocr_markdown_dir.exists() and not tree_progress_file.exists():
            logger.warning("Found ocr_markdown without tree_progress.json, clearing")
            shutil.rmtree(ocr_markdown_dir)

        # Check if already complete
        if resume and tree_progress_file.exists() and self.state.gaps_filled:
            # Load tree_progress to return metadata
            with open(tree_progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            unit_count = len(progress_data.get('units', []))
            logger.success(f"Refined breakdown already complete: {unit_count} units")
            return progress_data.get('units', [])

        ocr_markdown_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Analyze structure (or load existing)
        toc_tree_file = output_dir / "toc_tree.json"
        toc_tree_original = output_dir / "toc_tree_original.json"

        if toc_tree_file.exists() and resume:
            # Load toc_tree.json first to check if it has gap_filled entries
            with open(toc_tree_file, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)

            # Check if toc_tree.json has gap_filled entries
            def has_gap_filled(chapters):
                for ch in chapters:
                    if ch.get('gap_filled'):
                        return True
                    if ch.get('children') and has_gap_filled(ch['children']):
                        return True
                return False

            if has_gap_filled(toc_data.get('chapters', [])):
                # toc_tree.json has gap_filled entries
                if self.state.gaps_filled:
                    # Gap filling done, check verification status
                    verified_count = len(self.state.verified_nodes)
                    if verified_count > 0:
                        logger.info(f"Resuming: gap filling complete, {verified_count} nodes verified")
                    else:
                        logger.info("Resuming: gap filling complete, starting verification")
                elif toc_tree_original.exists():
                    # State was reset, load original to re-run gap filling
                    logger.info("Loading original TOC tree to re-run gap filling")
                    with open(toc_tree_original, 'r', encoding='utf-8') as f:
                        toc_data = json.load(f)
                else:
                    # Error: gap_filled in tree but no state and no original
                    logger.error("toc_tree.json has gap_filled entries but no original backup")
                    logger.error("Delete toc_tree.json to re-run structure analysis")
                    raise ValueError("Cannot resume: missing toc_tree_original.json")
            else:
                # No gap_filled entries, this is the original
                logger.info("Loading original TOC tree, will run gap filling")

            toc_tree = dict_list_to_toc_tree(toc_data['chapters'])
            book_metadata = {k: v for k, v in toc_data.items() if k != 'chapters'}
        elif toc_tree_original.exists() and resume:
            # toc_tree.json was deleted but original exists - load original
            logger.info("Loading original TOC tree (toc_tree.json was deleted)")
            with open(toc_tree_original, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            toc_tree = dict_list_to_toc_tree(toc_data['chapters'])
            book_metadata = {k: v for k, v in toc_data.items() if k != 'chapters'}
        else:
            # Preprocess PDF
            processed_pdf = preprocess_pdf(pdf_path, output_dir)

            # Analyze structure
            logger.info("Analyzing PDF structure...")
            toc_tree, book_metadata = self.structure_analyzer.analyze_pdf_structure(
                processed_pdf, book_title
            )

            # Save TOC tree
            toc_data = {
                **book_metadata,
                'chapters': [node.to_dict() for node in toc_tree]
            }
            with open(toc_tree_file, 'w', encoding='utf-8') as f:
                json.dump(toc_data, f, indent=2, ensure_ascii=False)
            logger.success(f"TOC tree saved to {toc_tree_file}")

        # Step 1.5: Verify boundaries FIRST (before gap/overlap detection)
        # This ensures start_pages are correct before we detect gaps
        logger.info(f"Verifying boundaries (max {self.max_workers} workers)...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._verify_chapter, chapter, pages_dir): chapter
                for chapter in toc_tree
            }
            for future in as_completed(futures):
                chapter = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Verification failed for '{chapter.title}': {e}")

        # Save state after verification
        self.state.save(state_file)

        # Step 1.6: Correct end_pages based on verified start_pages
        logger.info("Correcting end_pages based on verified boundaries...")
        self._correct_end_pages(toc_tree)

        # Update toc_data with corrected end_pages
        toc_data['chapters'] = [node.to_dict() for node in toc_tree]
        with open(toc_tree_file, 'w', encoding='utf-8') as f:
            json.dump(toc_data, f, indent=2, ensure_ascii=False)

        # Step 1.7: Detect and resolve overlaps in TOC
        logger.info("Detecting overlaps in TOC structure...")
        overlaps = self.gap_analyzer.detect_overlaps(toc_data)

        if overlaps:
            logger.info(f"Found {len(overlaps)} overlaps, resolving with LLM...")
            toc_data = self.gap_analyzer.resolve_overlaps(
                toc_data, overlaps, pages_dir, self.boundary_verifier
            )

            # Rebuild TOC tree from updated data
            toc_tree = dict_list_to_toc_tree(toc_data['chapters'])

            # Save updated TOC tree
            with open(toc_tree_file, 'w', encoding='utf-8') as f:
                json.dump(toc_data, f, indent=2, ensure_ascii=False)

            logger.success(f"Resolved {len(overlaps)} overlaps, updated {toc_tree_file}")
        else:
            logger.info("No overlaps found in TOC structure")

        # Step 1.8: Detect and fill gaps in TOC (using verified data)
        if not self.state.gaps_filled:
            # Save original TOC tree before gap filling
            if not toc_tree_original.exists():
                with open(toc_tree_original, 'w', encoding='utf-8') as f:
                    json.dump(toc_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved original TOC tree to {toc_tree_original}")

            logger.info("Detecting gaps in TOC structure...")
            gaps = self.gap_analyzer.detect_gaps(toc_data)

            if gaps:
                logger.info(f"Found {len(gaps)} gaps, classifying content...")
                classifications = {}

                # Classify each gap
                for gap in gaps:
                    classification = self.gap_analyzer.classify_gap(gap, pages_dir)
                    classifications[gap.start_page] = classification

                # Unified correction pass
                logger.info("Running unified correction on gap titles...")
                classifications = self.gap_analyzer.unify_corrections(
                    toc_data, gaps, classifications
                )

                # Fill gaps
                toc_data = self.gap_analyzer.fill_gaps(toc_data, gaps, classifications)

                # Rebuild TOC tree from updated data
                toc_tree = dict_list_to_toc_tree(toc_data['chapters'])

                # Save updated TOC tree
                with open(toc_tree_file, 'w', encoding='utf-8') as f:
                    json.dump(toc_data, f, indent=2, ensure_ascii=False)

                filled_count = sum(
                    1 for g in gaps
                    if classifications.get(g.start_page) and
                    classifications[g.start_page].suggested_level > 0
                )
                logger.success(f"Filled {filled_count} gaps, updated {toc_tree_file}")
            else:
                logger.info("No gaps found in TOC structure")

            self.state.gaps_filled = True

        # Step 2: Estimate tokens for all nodes
        logger.info("Estimating token counts...")
        self._estimate_all_tokens(toc_tree, pages_dir)

        # Step 4: Generate work units
        logger.info("Generating work units...")
        work_units = []
        for chapter_idx, chapter in enumerate(toc_tree):
            # index_path starts with 1-based top-level index
            chapter_units = self._generate_units_recursive(
                chapter, pages_dir, [chapter_idx + 1]
            )
            work_units.extend(chapter_units)

        # Step 5: Merge pages and save
        logger.info(f"Saving {len(work_units)} work units...")
        unit_metadata = self._save_units(work_units, pages_dir, ocr_markdown_dir)

        # Save tree progress
        with open(tree_progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'units': unit_metadata,
                'book_metadata': book_metadata
            }, f, indent=2, ensure_ascii=False)

        logger.success(f"Refined breakdown complete: {len(work_units)} units")
        return unit_metadata

    def _estimate_all_tokens(self, toc_tree: List[TOCNode], pages_dir: Path):
        """Recursively estimate tokens for all nodes."""
        for node in toc_tree:
            self._estimate_node_tokens(node, pages_dir)

    def _estimate_node_tokens(self, node: TOCNode, pages_dir: Path):
        """Estimate tokens for a node and its children."""
        # Estimate this node's tokens
        total_tokens = 0
        for page_num in range(node.start_page, node.end_page + 1):
            page_file = pages_dir / f"page_{page_num:03d}.md"
            if page_file.exists():
                content = page_file.read_text(encoding='utf-8')
                total_tokens += len(tokenizer.encode(content))

        node.estimated_tokens = total_tokens

        # Recursively estimate children
        for child in node.children:
            self._estimate_node_tokens(child, pages_dir)

    def _correct_end_pages(self, toc_tree: List[TOCNode]):
        """
        Correct end_pages based on verified start_pages of next siblings.

        After boundary verification, we know the correct start_pages.
        For each node, its end_page should be at most next_sibling.start_page - 1,
        unless they share a page (same start_page).

        This ensures gap detection uses consistent data.
        """
        def correct_recursive(nodes: List[TOCNode]):
            for i, node in enumerate(nodes):
                if i + 1 < len(nodes):
                    next_node = nodes[i + 1]
                    # Calculate expected end_page
                    expected_end = next_node.start_page - 1

                    # Only correct if current end_page is less than expected
                    # (meaning there was a gap that shouldn't exist)
                    if node.end_page < expected_end:
                        logger.debug(
                            f"Correcting '{node.title}' end_page: "
                            f"{node.end_page} -> {expected_end}"
                        )
                        node.end_page = expected_end
                    # If end_page > expected_end, there's an overlap
                    # (handled by overlap detection)

                # Recursively process children
                if node.children:
                    correct_recursive(node.children)

        correct_recursive(toc_tree)

    def _verify_chapter(self, chapter: TOCNode, pages_dir: Path):
        """
        Verify all nodes in a chapter, handling failures.

        Strategy:
        - 1 failure: search nearby pages
        - 2+ failures: re-breakdown the entire chapter
        """
        # Collect all nodes to verify
        nodes_to_verify = self._collect_nodes_for_verification(chapter)

        # Filter out already verified nodes
        nodes_needing_verification = []
        for node, node_id in nodes_to_verify:
            if self.state.is_verified(node_id):
                node.boundary_info = self.state.get_boundary_info(node_id)
            else:
                nodes_needing_verification.append((node, node_id))

        if not nodes_needing_verification:
            return

        # Parallel verification of all nodes in this chapter
        failed_nodes = []

        def verify_single_node(node_tuple: Tuple[TOCNode, str]) -> Tuple[TOCNode, str, Dict]:
            node, node_id = node_tuple
            result = self.boundary_verifier.verify_boundary(node, pages_dir)
            return node, node_id, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(verify_single_node, item) for item in nodes_needing_verification]

            for future in as_completed(futures):
                node, node_id, result = future.result()
                if result.get('found', False):
                    node.boundary_info = result
                    self.state.mark_verified(node_id, result)
                else:
                    failed_nodes.append((node, node_id))
                    self.state.mark_failed(node_id)

        # Handle failures
        if len(failed_nodes) == 0:
            return
        elif len(failed_nodes) == 1:
            # Single failure: try nearby pages with context
            node, node_id = failed_nodes[0]

            # Find previous and next titles for context
            prev_title = None
            next_title = None
            for i, (n, nid) in enumerate(nodes_to_verify):
                if nid == node_id:
                    if i > 0:
                        prev_title = nodes_to_verify[i - 1][0].title
                    if i < len(nodes_to_verify) - 1:
                        next_title = nodes_to_verify[i + 1][0].title
                    break

            new_page = self.boundary_verifier.search_nearby_pages(
                node, pages_dir,
                prev_title=prev_title,
                next_title=next_title
            )
            if new_page:
                node.start_page = new_page
                result = self.boundary_verifier.verify_boundary(node, pages_dir)
                if result.get('found', False):
                    node.boundary_info = result
                    self.state.mark_verified(node_id, result)
                    logger.info(f"Fixed '{node.title}' on page {new_page}")
        else:
            # Multiple failures: re-breakdown the chapter
            if not self.state.was_chapter_rebroken(chapter.title):
                logger.warning(f"{len(failed_nodes)} failures in '{chapter.title}', re-breaking down")
                new_children = self.structure_analyzer.rebreakdown_chapter(
                    chapter.start_page, chapter.end_page, pages_dir, chapter.title
                )
                if new_children:
                    chapter.children = new_children
                    self._estimate_node_tokens(chapter, pages_dir)
                    self.state.mark_chapter_rebroken(chapter.title)
                    # Re-verify the new structure
                    self._verify_chapter(chapter, pages_dir)

    def _collect_nodes_for_verification(self, node: TOCNode, ancestors: List[str] = None):
        """Collect all nodes that need boundary verification."""
        if ancestors is None:
            ancestors = []

        node_id = "/".join(ancestors + [node.title])
        results = []

        # Skip gap_filled nodes - their titles are LLM-generated, not on the page
        if not node.gap_filled:
            results.append((node, node_id))

        for child in node.children:
            results.extend(self._collect_nodes_for_verification(
                child, ancestors + [node.title]
            ))

        return results

    def _generate_units_recursive(
        self,
        node: TOCNode,
        pages_dir: Path,
        index_path: List[int]
    ) -> List[Dict]:
        """
        Recursively generate work units from a node.

        Args:
            node: Current TOC node
            pages_dir: Directory containing page files
            index_path: Hierarchical index path like [7, 1] for first child of 7th top-level

        Logic:
        - If leaf and tokens <= max_tokens: create unit
        - If leaf and tokens > max_tokens: try discover subsections
        - If has children and total <= max_tokens: create unit for whole node
        - If has children and total > max_tokens: recurse into children
        """
        # Case 1: Leaf node
        if node.is_leaf():
            if node.estimated_tokens <= self.max_tokens:
                return [self._create_unit(node, index_path)]
            else:
                # Large leaf node - create unit and let NestedPartProcessor handle splitting
                logger.info(f"'{node.title}' ({node.estimated_tokens} tokens) exceeds max_tokens, will be split by processor")
                return [self._create_unit(node, index_path)]

        # Case 2: Has children
        total_children_tokens = sum(child.estimated_tokens for child in node.children)

        if total_children_tokens <= self.max_tokens:
            # Whole node fits in one unit
            return [self._create_unit(node, index_path, include_children=True)]
        else:
            # Recurse into children
            units = []
            for child_idx, child in enumerate(node.children):
                # Build child's index path by appending 1-based child index
                child_index_path = index_path + [child_idx + 1]
                child_units = self._generate_units_recursive(
                    child, pages_dir, child_index_path
                )
                units.extend(child_units)
            return units

    def _create_unit(
        self,
        node: TOCNode,
        index_path: List[int],
        include_children: bool = False
    ) -> Dict:
        """Create a work unit dictionary from a node."""
        # Generate unit ID using hierarchical index
        unit_id = generate_unit_id(index_path)

        return {
            'unit_id': unit_id,
            'node': node,
            'index_path': index_path,
            'title': node.title,
            'start_page': node.start_page,
            'end_page': node.end_page,
            'token_count': node.estimated_tokens,
            'include_children': include_children
        }

    def _save_units(
        self,
        work_units: List[Dict],
        pages_dir: Path,
        output_dir: Path
    ) -> List[Dict]:
        """Save all work units to files."""
        unit_metadata = []

        for i, unit in enumerate(work_units):
            node = unit['node']

            # Get next node for end boundary cutting
            next_node = None
            if i + 1 < len(work_units):
                next_node = work_units[i + 1]['node']

            # Merge content
            if unit.get('include_children'):
                # Get all nodes to merge
                all_nodes = [node] + node.get_all_leaves()
                content = self.page_merger.merge_nodes_content(all_nodes, pages_dir, next_node)
            else:
                content = self.page_merger.merge_node_content(node, pages_dir, next_node)

            # Save file
            output_file = output_dir / f"{unit['unit_id']}.md"
            output_file.write_text(content, encoding='utf-8')

            # Create metadata
            metadata = {
                'unit_id': unit['unit_id'],
                'index_path': unit['index_path'],
                'title': unit['title'],
                'page_range': [unit['start_page'], unit['end_page']],
                'token_count': unit['token_count'],
                'file': str(output_file.name)
            }
            unit_metadata.append(metadata)

            logger.debug(f"Saved {unit['unit_id']}: {unit['token_count']} tokens")

        return unit_metadata
