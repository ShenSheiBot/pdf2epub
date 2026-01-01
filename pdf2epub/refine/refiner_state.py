"""
State management for the refinement process.

Provides checkpoint/resume functionality.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RefinerState:
    """
    State management for the refinement process.
    Supports checkpoint/resume.
    """
    # TOC analysis steps
    toc_location: Dict = field(default_factory=dict)  # {has_toc, toc_start, toc_end}
    toc_structure: Dict = field(default_factory=dict)  # {author, chapters: [...]} (no page numbers)
    structure_analysis_complete: bool = False

    # node_id -> boundary_info
    verified_nodes: Dict[str, Dict] = field(default_factory=dict)

    # List of node_ids that failed verification
    failed_nodes: List[str] = field(default_factory=list)

    # node_id -> retry count
    retry_counts: Dict[str, int] = field(default_factory=dict)

    # chapter_id -> True if re-breakdown was performed
    rebroken_chapters: Dict[str, bool] = field(default_factory=dict)

    # Whether gaps have been detected and filled
    gaps_filled: bool = False

    def save(self, path: Path):
        """Save state to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'toc_location': self.toc_location,
                'toc_structure': self.toc_structure,
                'structure_analysis_complete': self.structure_analysis_complete,
                'verified': self.verified_nodes,
                'failed': self.failed_nodes,
                'retries': self.retry_counts,
                'rebroken': self.rebroken_chapters,
                'gaps_filled': self.gaps_filled
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> bool:
        """Load state from file. Returns True if loaded."""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.toc_location = data.get('toc_location', {})
                self.toc_structure = data.get('toc_structure', {})
                self.structure_analysis_complete = data.get('structure_analysis_complete', False)
                self.verified_nodes = data.get('verified', {})
                self.failed_nodes = data.get('failed', [])
                self.retry_counts = data.get('retries', {})
                self.rebroken_chapters = data.get('rebroken', {})
                self.gaps_filled = data.get('gaps_filled', False)
            return True
        return False

    def mark_verified(self, node_id: str, boundary_info: Dict):
        """Mark a node as successfully verified."""
        self.verified_nodes[node_id] = boundary_info
        if node_id in self.failed_nodes:
            self.failed_nodes.remove(node_id)

    def mark_failed(self, node_id: str):
        """Mark a node as failed verification."""
        if node_id not in self.failed_nodes:
            self.failed_nodes.append(node_id)
        self.retry_counts[node_id] = self.retry_counts.get(node_id, 0) + 1

    def is_verified(self, node_id: str) -> bool:
        """Check if a node has been verified."""
        return node_id in self.verified_nodes

    def get_boundary_info(self, node_id: str) -> Dict:
        """Get cached boundary info for a node."""
        return self.verified_nodes.get(node_id, {})

    def count_failures_in_chapter(self, chapter_id: str, node_ids: List[str]) -> int:
        """Count how many nodes in a chapter failed verification."""
        return sum(1 for nid in node_ids if nid in self.failed_nodes)

    def mark_chapter_rebroken(self, chapter_id: str):
        """Mark that a chapter has been re-broken down."""
        self.rebroken_chapters[chapter_id] = True

    def was_chapter_rebroken(self, chapter_id: str) -> bool:
        """Check if a chapter was already re-broken down."""
        return self.rebroken_chapters.get(chapter_id, False)
