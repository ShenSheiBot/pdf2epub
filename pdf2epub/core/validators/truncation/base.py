"""
Base class for truncation detection strategies.

This module provides the abstract base class for all truncation detectors,
allowing different detection strategies for different processors.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict


class BaseTruncationDetector(ABC):
    """Abstract base class for truncation detection."""
    
    @abstractmethod
    def detect(
        self,
        original: str,
        processed: str,
        **kwargs
    ) -> Tuple[bool, str, Dict]:
        """
        Detect if processed content is truncated.
        
        Args:
            original: Original content
            processed: Processed content
            **kwargs: Additional detector-specific arguments
        
        Returns:
            Tuple of (is_truncated, reason, details)
            - is_truncated: Whether truncation was detected
            - reason: Human-readable reason for the decision
            - details: Dictionary with detailed analysis results
        """
        pass
    
    def get_summary(
        self,
        is_truncated: bool,
        reason: str,
        details: Dict
    ) -> str:
        """
        Generate a human-readable summary of truncation analysis.
        
        Args:
            is_truncated: Whether truncation was detected
            reason: Reason for the decision
            details: Analysis details
        
        Returns:
            Summary string
        """
        summary_parts = []
        
        if is_truncated:
            summary_parts.append(f"⚠️ TRUNCATION DETECTED: {reason}")
        else:
            summary_parts.append(f"✓ Content complete: {reason}")
        
        # Add any common details
        if 'token_ratio' in details:
            summary_parts.append(
                f"Token ratio: {details['token_ratio']:.1%}"
            )
        
        return "\n".join(summary_parts)
