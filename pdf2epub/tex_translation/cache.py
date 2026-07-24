"""Content-addressed local cache for translation responses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedTranslation:
    content: str
    metadata: dict


class TranslationCache:
    """A deterministic cache independent of provider-side prompt caching."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key(
        *,
        provider: str,
        model: str,
        messages: list[dict],
        prompt_version: str,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "prompt_version": prompt_version,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> CachedTranslation | None:
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        content = payload.get("content")
        if not isinstance(content, str):
            return None
        expected_hash = payload.get("content_sha256")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if expected_hash != actual_hash:
            return None
        return CachedTranslation(
            content=content, metadata=payload.get("metadata") or {}
        )

    def put(self, key: str, content: str, metadata: dict | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": metadata or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(serialized)
            final_path = self.root / f"{key}.json"
            os.replace(temporary, final_path)
            return final_path
        finally:
            Path(temporary).unlink(missing_ok=True)
