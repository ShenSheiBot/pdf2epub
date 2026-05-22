"""
Embedding-based translation alignment utilities.

Uses gemini-embedding-001 for cross-lingual (JP→CN) cosine similarity.
Primary method for verifier alignment checks and position alignment.
Falls back to LLM-based methods if embedding is unavailable.

Threshold: 0.75 (experimentally determined)
- Correct aligned pairs: max(matrix) = 0.917 - 1.000
- Misaligned by 4 lines (worst case): max(matrix) = 0.740
- Hallucinated content: max(matrix) = 0.575 - 0.683
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# Experimentally determined threshold.
# Below this, the translation window is considered hallucinated.
# Gap between worst real (0.740) and best hallucination (0.683) = 0.057.
HALLUCINATION_THRESHOLD = 0.75

# Position alignment requires at least this similarity to trust the result.
# If max_sim is below this, fall back to LLM/line estimate.
POSITION_MIN_CONFIDENCE = 0.75


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def compute_similarity_matrix(
    src_embeddings: List[List[float]],
    tl_embeddings: List[List[float]],
) -> np.ndarray:
    """Compute NxM cosine similarity matrix (rows=translated, cols=source)."""
    n_tl = len(tl_embeddings)
    n_src = len(src_embeddings)
    if n_tl == 0 or n_src == 0:
        return np.zeros((n_tl, n_src))
    matrix = np.zeros((n_tl, n_src))
    for i in range(n_tl):
        for j in range(n_src):
            matrix[i][j] = cosine_similarity(tl_embeddings[i], src_embeddings[j])
    return matrix


def check_alignment_embedding(
    source_window: List[str],
    translated_window: List[str],
    llm_client,
    embedding_provider: str = "gemini",
    embedding_model: str = "gemini-embedding-001",
    hallucination_threshold: float = HALLUCINATION_THRESHOLD,
) -> Optional[str]:
    """Check alignment between source and translated windows using embedding.

    Returns 'A'/'B' (aligned) or 'D' (hallucination), matching the LLM verifier
    interface. Returns None if embedding is unavailable (caller should fall back).
    """
    if not source_window or not translated_window:
        return None  # Can't check empty windows, let caller handle

    embeddings_src = llm_client.embed_texts(
        source_window, provider=embedding_provider, model=embedding_model
    )
    if not embeddings_src:
        return None

    embeddings_tl = llm_client.embed_texts(
        translated_window, provider=embedding_provider, model=embedding_model
    )
    if not embeddings_tl:
        return None

    matrix = compute_similarity_matrix(embeddings_src, embeddings_tl)
    if matrix.size == 0:
        return None

    max_sim = float(matrix.max())
    logger.debug(f"  Embedding alignment: max_sim={max_sim:.3f}, threshold={hallucination_threshold}")

    if max_sim >= hallucination_threshold:
        return "A"  # Valid translation (may be aligned or shifted, but real)
    else:
        return "D"  # Hallucination


def find_position_embedding(
    last_translated_line: str,
    source_candidates: List[str],
    llm_client,
    embedding_provider: str = "gemini",
    embedding_model: str = "gemini-embedding-001",
    position_min_confidence: float = POSITION_MIN_CONFIDENCE,
) -> Optional[int]:
    """Find which source line best matches the translated line using embedding.

    Returns 0-indexed position within source_candidates, or None if embedding
    is unavailable or confidence is too low.
    """
    if not last_translated_line.strip() or not source_candidates:
        return None

    tl_emb = llm_client.embed_texts(
        [last_translated_line], provider=embedding_provider, model=embedding_model
    )
    if not tl_emb:
        return None

    src_emb = llm_client.embed_texts(
        source_candidates, provider=embedding_provider, model=embedding_model
    )
    if not src_emb:
        return None

    sims = [cosine_similarity(tl_emb[0], s) for s in src_emb]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    # Confidence gate: if best match is too weak, don't trust it
    if best_sim < position_min_confidence:
        logger.info(
            f"  Embedding position alignment: low confidence ({best_sim:.3f} < {position_min_confidence}), skipping"
        )
        return None

    logger.info(
        f"  Embedding position alignment: idx={best_idx}, sim={best_sim:.3f}"
    )

    return best_idx
