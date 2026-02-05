"""
Verification tools for agent-based content validation.

Provides tools that agents can use to inspect and validate processed content
(polish, translation, etc.) against original content.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
import re


@dataclass
class VerificationFile:
    """Data for a single file to verify."""
    key: str
    original: str
    processed: str
    metadata: Optional[Dict] = None


class VerificationTools:
    """
    Tools for content verification that agents can use.

    Similar to boundary_agent's read_page tool, but more general purpose
    for any type of content verification.
    """

    def __init__(self, files: Dict[str, VerificationFile]):
        """
        Initialize with files to verify.

        Args:
            files: Dict mapping file_key to VerificationFile
        """
        self.files = files

    def read_segment(
        self,
        file_key: str,
        source: str = "processed",
        start: int = 0,
        length: int = 1000
    ) -> str:
        """
        Read a segment of content from a file.

        Args:
            file_key: File identifier
            source: "original" or "processed"
            start: Starting character position
            length: Number of characters to read

        Returns:
            Content segment with line numbers
        """
        if file_key not in self.files:
            return f"Error: File {file_key} not found"

        file_data = self.files[file_key]
        content = file_data.original if source == "original" else file_data.processed

        # Extract segment
        segment = content[start:start+length]

        # Add line numbers for easier reference
        lines = segment.split('\n')
        numbered_lines = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]

        result = f"=== {file_key} ({source}) at position {start}-{start+length} ===\n"
        result += "\n".join(numbered_lines)
        result += f"\n=== Total length: {len(content)} chars ==="

        return result

    def get_stats(self, file_key: str) -> Dict:
        """
        Get statistics about a file.

        Args:
            file_key: File identifier

        Returns:
            Dict with statistics
        """
        if file_key not in self.files:
            return {"error": f"File {file_key} not found"}

        file_data = self.files[file_key]
        orig_len = len(file_data.original)
        proc_len = len(file_data.processed)

        stats = {
            'file_key': file_key,
            'original_length': orig_len,
            'processed_length': proc_len,
            'length_ratio': proc_len / orig_len if orig_len > 0 else 0,
            'original_lines': file_data.original.count('\n') + 1,
            'processed_lines': file_data.processed.count('\n') + 1,
        }

        # Add metadata if available
        if file_data.metadata:
            stats['metadata'] = file_data.metadata

        return stats

    def search_content(
        self,
        file_key: str,
        pattern: str,
        source: str = "processed",
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for a pattern in content.

        Args:
            file_key: File identifier
            pattern: Search pattern (plain text or regex)
            source: "original" or "processed"
            max_results: Maximum number of results to return

        Returns:
            List of matches with positions
        """
        if file_key not in self.files:
            return [{"error": f"File {file_key} not found"}]

        file_data = self.files[file_key]
        content = file_data.original if source == "original" else file_data.processed

        results = []
        try:
            # Try as regex first
            for match in re.finditer(pattern, content):
                if len(results) >= max_results:
                    break

                start = match.start()
                end = match.end()
                # Get context around match
                context_start = max(0, start - 50)
                context_end = min(len(content), end + 50)
                context = content[context_start:context_end]

                results.append({
                    'position': start,
                    'match': match.group(0),
                    'context': context
                })
        except re.error:
            # If regex fails, do plain text search
            start = 0
            while len(results) < max_results:
                pos = content.find(pattern, start)
                if pos == -1:
                    break

                context_start = max(0, pos - 50)
                context_end = min(len(content), pos + len(pattern) + 50)
                context = content[context_start:context_end]

                results.append({
                    'position': pos,
                    'match': pattern,
                    'context': context
                })
                start = pos + 1

        return results if results else [{"message": "No matches found"}]

    def compare_segments(
        self,
        file_key: str,
        position: str = "end",
        length: int = 500
    ) -> Dict:
        """
        Compare corresponding segments from original and processed.

        Args:
            file_key: File identifier
            position: "start", "middle", or "end"
            length: Length of segment to compare

        Returns:
            Dict with both segments for comparison
        """
        if file_key not in self.files:
            return {"error": f"File {file_key} not found"}

        file_data = self.files[file_key]
        orig = file_data.original
        proc = file_data.processed

        # Determine position
        if position == "start":
            orig_segment = orig[:length]
            proc_segment = proc[:length]
            pos_desc = "start"
        elif position == "end":
            orig_segment = orig[-length:] if len(orig) > length else orig
            proc_segment = proc[-length:] if len(proc) > length else proc
            pos_desc = "end"
        elif position == "middle":
            orig_mid = len(orig) // 2
            proc_mid = len(proc) // 2
            orig_segment = orig[max(0, orig_mid-length//2):orig_mid+length//2]
            proc_segment = proc[max(0, proc_mid-length//2):proc_mid+length//2]
            pos_desc = "middle"
        else:
            return {"error": f"Invalid position: {position}"}

        return {
            'file_key': file_key,
            'position': pos_desc,
            'original': orig_segment,
            'processed': proc_segment,
            'original_length': len(orig),
            'processed_length': len(proc)
        }

    def detect_content_type(self, file_key: str) -> str:
        """
        Detect the type of content (heuristic).

        Args:
            file_key: File identifier

        Returns:
            Content type: "table", "index", "toc", "prose", "list"
        """
        if file_key not in self.files:
            return "unknown"

        content = self.files[file_key].original[:1000]  # Check first 1000 chars

        # Heuristics
        if '<table>' in content or content.count('|') > 10:
            return "table"

        # Index pattern: Name ... page numbers
        if re.search(r'\w+\s+\d+,\s*\d+', content):
            index_matches = re.findall(r'\w+\s+\d+', content)
            if len(index_matches) > 5:
                return "index"

        # TOC pattern: headings with page numbers
        if re.search(r'^\s*\d+\s*$', content, re.MULTILINE):
            return "toc"

        # List pattern: many lines starting with - or *
        list_lines = re.findall(r'^[\s\-\*]+\w', content, re.MULTILINE)
        if len(list_lines) > 5:
            return "list"

        return "prose"

    def get_all_keys(self) -> List[str]:
        """Get all file keys."""
        return list(self.files.keys())
