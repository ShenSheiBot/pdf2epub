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
import asyncio
from pathlib import Path
from typing import List, Dict
from loguru import logger
import tiktoken

from ..utils.llm_client import LLMClient, BoundLLMClient
from ..utils.pdf_utils import preprocess_pdf
from ..utils.unit_id import generate_unit_id
from .toc_tree import TOCNode, dict_list_to_toc_tree
from .refiner_state import RefinerState
from .boundary_agent import verify_toc_recursive, get_model_max_tokens
from .structure_analyzer import StructureAnalyzer
from .page_merger import PageMerger

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
        max_tokens: int = None,
        max_workers: int = None,
    ):
        """
        Initialize the refined breakdown processor.

        Args:
            config: Configuration dict (from config.yaml)
            max_tokens: Maximum tokens per unit (LLM limit). If None, uses model limit from config.
            max_workers: Maximum parallel workers for verification
        """
        self.config = config
        self.max_workers = max_workers or config.get('general', {}).get('max_concurrent_workers', 8)

        # Get max_tokens from boundary_agent's model config if not specified
        if max_tokens is None:
            max_tokens = get_model_max_tokens()
        self.max_tokens = max_tokens

        # Get refine config with backward compatibility
        refine_config = config.get('refine', {})
        default_provider = refine_config.get('provider', 'gemini')

        # New nested format: refine.structure.{provider, model}
        # Old flat format: refine.{provider, structure_model, verification_model}
        structure_config = refine_config.get('structure', {})
        verification_config = refine_config.get('verification', {})

        structure_provider = structure_config.get('provider', default_provider)
        structure_model = structure_config.get('model', refine_config.get('structure_model', 'gemini-2.5-pro'))
        toc_model = structure_config.get('toc_model', refine_config.get('toc_model', 'gemini-2.5-flash'))

        verification_provider = verification_config.get('provider', default_provider)
        verification_model = verification_config.get('model', refine_config.get('verification_model', 'gemini-2.5-flash'))

        # Create unified LLM client and bind to specific providers
        llm_client = LLMClient(config)
        structure_client = BoundLLMClient(llm_client, structure_provider)
        verification_client = BoundLLMClient(llm_client, verification_provider)

        # Initialize components with their respective clients
        self.structure_analyzer = StructureAnalyzer(
            structure_client, structure_model, toc_model,
            verification_client, verification_model,
            config
        )
        # Note: BoundaryVerifier and GapAnalyzer replaced by boundary_agent
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
            logger.info("Resumed from saved state")

        # Check if ocr_markdown exists but has no tree_progress.json
        tree_progress_file = ocr_markdown_dir / "tree_progress.json"
        if ocr_markdown_dir.exists() and not tree_progress_file.exists():
            logger.warning("Found ocr_markdown without tree_progress.json, clearing")
            shutil.rmtree(ocr_markdown_dir)

        # Check if already complete (verification includes gap/overlap handling)
        if resume and tree_progress_file.exists() and self.state.verification_complete:
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

        # Priority: toc_tree_original > toc_tree > run analysis
        # Always use original if available (agent may have modified toc_tree)
        if toc_tree_original.exists() and resume:
            logger.info("Loading original TOC tree for verification")
            with open(toc_tree_original, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            toc_tree = dict_list_to_toc_tree(toc_data['chapters'])
            book_metadata = {k: v for k, v in toc_data.items() if k != 'chapters'}
        elif toc_tree_file.exists() and resume:
            logger.info("Loading TOC tree")
            with open(toc_tree_file, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            toc_tree = dict_list_to_toc_tree(toc_data['chapters'])
            book_metadata = {k: v for k, v in toc_data.items() if k != 'chapters'}
        else:
            # Preprocess PDF
            processed_pdf = preprocess_pdf(pdf_path, output_dir)

            # Analyze structure (with resume support)
            logger.info("Analyzing PDF structure...")
            toc_tree, book_metadata = self.structure_analyzer.analyze_pdf_structure(
                processed_pdf, book_title,
                state=self.state, state_path=state_file
            )

            # Insert table_of_contents as a chapter if it exists
            toc_info = book_metadata.get('table_of_contents')
            if toc_info and toc_info.get('start_page') and toc_info.get('end_page'):
                from .toc_tree import TOCNode
                toc_chapter = TOCNode(
                    title="Table of Contents",
                    level=1,
                    start_page=toc_info['start_page'],
                    end_page=toc_info['end_page'],
                    children=[],
                )
                # Insert at correct position based on page number
                inserted = False
                for i, chapter in enumerate(toc_tree):
                    if chapter.start_page > toc_chapter.start_page:
                        toc_tree.insert(i, toc_chapter)
                        inserted = True
                        break
                if not inserted:
                    toc_tree.append(toc_chapter)
                logger.info(f"Added 'Table of Contents' chapter (p{toc_info['start_page']}-p{toc_info['end_page']})")

            # Save TOC tree (both original and working copy)
            toc_data = {
                **book_metadata,
                'chapters': [node.to_dict() for node in toc_tree]
            }
            with open(toc_tree_file, 'w', encoding='utf-8') as f:
                json.dump(toc_data, f, indent=2, ensure_ascii=False)
            with open(toc_tree_original, 'w', encoding='utf-8') as f:
                json.dump(toc_data, f, indent=2, ensure_ascii=False)
            logger.success(f"TOC tree saved to {toc_tree_file} and {toc_tree_original}")

        # Step 1.5: Verify boundaries using agent-based verification
        # This ensures start_pages are correct before we detect gaps
        # Also automatically removes children for nodes below token threshold
        if self.state.verification_complete:
            logger.info(f"Verification already complete, skipping")
        else:
            # Count total pages
            total_pages = len(list(pages_dir.glob("page_*.md")))

            logger.info(f"Verifying boundaries using agent (total_pages={total_pages})...")
            try:
                toc_tree = asyncio.run(verify_toc_recursive(
                    toc_tree, pages_dir, total_pages,
                    max_tokens=self.max_tokens,
                ))

                # Update toc_data with verified tree
                toc_data['chapters'] = [node.to_dict() for node in toc_tree]
                with open(toc_tree_file, 'w', encoding='utf-8') as f:
                    json.dump(toc_data, f, indent=2, ensure_ascii=False)
                logger.success(f"Saved verified TOC tree to {toc_tree_file}")

            except Exception as e:
                logger.error(f"Agent verification failed: {e}")
                raise

            # Mark verification as complete and save state
            self.state.verification_complete = True
            self.state.save(state_file)

        # Note: Gap filling and overlap detection are now handled by the agent
        # via insert_section/remove_section tools during boundary verification

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
