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

    Used by:
    - StructureAnalyzer: toc_location, structure_analysis_complete
    - Main workflow: verification_complete
    """
    # TOC analysis steps (used by StructureAnalyzer)
    toc_location: Dict = field(default_factory=dict)  # {has_toc, toc_start, toc_end}
    toc_reference: str = ""
    toc_reference_pages: List[int] = field(default_factory=list)
    structure_analysis_complete: bool = False

    # Verification phase complete (used by main workflow)
    verification_complete: bool = False

    def save(self, path: Path):
        """Save state to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'toc_location': self.toc_location,
                'toc_reference': self.toc_reference,
                'toc_reference_pages': self.toc_reference_pages,
                'structure_analysis_complete': self.structure_analysis_complete,
                'verification_complete': self.verification_complete,
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> bool:
        """Load state from file. Returns True if loaded."""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.toc_location = data.get('toc_location', {})
                self.toc_reference = data.get('toc_reference', '')
                self.toc_reference_pages = data.get('toc_reference_pages', [])
                self.structure_analysis_complete = data.get('structure_analysis_complete', False)
                self.verification_complete = data.get('verification_complete', False)
            return True
        return False
