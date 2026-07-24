"""Atomic state and translation-file persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

STATE_VERSION = 1


class TranslationState:
    def __init__(self, control_dir: Path):
        self.control_dir = control_dir
        self.path = control_dir / "state.json"
        self.units_dir = control_dir / "units"
        self.data: dict = {}

    def initialize(
        self,
        *,
        source_id: str,
        source_fingerprint: str,
        layout_fingerprint: str,
        main_tex: str,
        units: list[dict],
        translation_spec: dict | None = None,
    ) -> None:
        translation_spec = translation_spec or {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            migrates_legacy_spec = "translation_spec" not in self.data
            if migrates_legacy_spec:
                # Migrate local runs created before the translation contract was
                # persisted, but do not write it until all other identity checks
                # have passed.
                self.data["translation_spec"] = translation_spec
            try:
                self._validate(
                    source_id,
                    source_fingerprint,
                    layout_fingerprint,
                    main_tex,
                    translation_spec,
                )
            except ValueError:
                if self._can_reset_pristine(
                    source_id=source_id,
                    main_tex=main_tex,
                    translation_spec=translation_spec,
                ):
                    self.data = self._new_state_data(
                        source_id=source_id,
                        source_fingerprint=source_fingerprint,
                        layout_fingerprint=layout_fingerprint,
                        main_tex=main_tex,
                        units=units,
                        translation_spec=translation_spec,
                    )
                    self.save()
                    return
                if not self._can_rebase(
                    source_id=source_id,
                    main_tex=main_tex,
                    units=units,
                    translation_spec=translation_spec,
                ):
                    raise
                self._rebase(
                    source_fingerprint=source_fingerprint,
                    layout_fingerprint=layout_fingerprint,
                    units=units,
                )
            else:
                if migrates_legacy_spec:
                    self.save()
            return
        self.data = self._new_state_data(
            source_id=source_id,
            source_fingerprint=source_fingerprint,
            layout_fingerprint=layout_fingerprint,
            main_tex=main_tex,
            units=units,
            translation_spec=translation_spec,
        )
        self.save()

    @staticmethod
    def _new_state_data(
        *,
        source_id: str,
        source_fingerprint: str,
        layout_fingerprint: str,
        main_tex: str,
        units: list[dict],
        translation_spec: dict,
    ) -> dict:
        return {
            "version": STATE_VERSION,
            "source_id": source_id,
            "source_fingerprint": source_fingerprint,
            "layout_fingerprint": layout_fingerprint,
            "main_tex": main_tex,
            "translation_spec": translation_spec,
            "units": {
                unit["id"]: {
                    **unit,
                    "status": "pending",
                }
                for unit in units
            },
        }

    def _validate(
        self,
        source_id: str,
        source_fingerprint: str,
        layout_fingerprint: str,
        main_tex: str,
        translation_spec: dict,
    ) -> None:
        if self.data.get("version") != STATE_VERSION:
            raise ValueError("Unsupported TeX translation state version")
        if self.data.get("source_id") != source_id:
            raise ValueError(
                "The requested source differs from the source snapshot in this "
                "translation run; use a new output directory"
            )
        if self.data.get("source_fingerprint") != source_fingerprint:
            raise ValueError(
                "Source tree changed since this translation run was created; "
                "use a new output directory"
            )
        if self.data.get("layout_fingerprint") != layout_fingerprint:
            raise ValueError(
                "Translation unit layout changed (for example unit_chars differs); "
                "resume with the original settings or use a new output directory"
            )
        if self.data.get("main_tex") != main_tex:
            raise ValueError("Main TeX entry point differs from the existing run")
        if self.data.get("translation_spec") != translation_spec:
            raise ValueError(
                "Source language, target language, or translation prompt version "
                "differs from the existing run; use a new output directory"
            )

    def _can_rebase(
        self,
        *,
        source_id: str,
        main_tex: str,
        units: list[dict],
        translation_spec: dict,
    ) -> bool:
        """Allow safe layout upgrades while preserving identical existing units."""
        if self.data.get("version") != STATE_VERSION:
            return False
        if self.data.get("source_id") != source_id:
            return False
        if self.data.get("main_tex") != main_tex:
            return False
        if self.data.get("translation_spec") != translation_spec:
            return False
        existing = self.data.get("units", {})
        new_ids = [unit["id"] for unit in units]
        existing_ids = list(existing)
        if existing_ids != new_ids[: len(existing_ids)]:
            return False
        return all(
            existing[unit["id"]].get("relative_path") == unit["relative_path"]
            and existing[unit["id"]].get("source_sha256") == unit["source_sha256"]
            for unit in units[: len(existing_ids)]
        )

    def _can_reset_pristine(
        self,
        *,
        source_id: str,
        main_tex: str,
        translation_spec: dict,
    ) -> bool:
        """Permit manifest upgrades before any unit has produced a result."""
        if self.data.get("version") != STATE_VERSION:
            return False
        if self.data.get("source_id") != source_id:
            return False
        if self.data.get("main_tex") != main_tex:
            return False
        if self.data.get("translation_spec") != translation_spec:
            return False
        return all(
            record.get("status", "pending") == "pending"
            for record in self.data.get("units", {}).values()
        )

    def _rebase(
        self,
        *,
        source_fingerprint: str,
        layout_fingerprint: str,
        units: list[dict],
    ) -> None:
        """Refresh immutable layout metadata while preserving transaction records."""
        self.data["source_fingerprint"] = source_fingerprint
        self.data["layout_fingerprint"] = layout_fingerprint
        for unit in units:
            record = self.data["units"].setdefault(
                unit["id"],
                {
                    **unit,
                    "status": "pending",
                },
            )
            record.update(unit)
        self.save()

    def completed_translations(self) -> dict[str, str]:
        translations: dict[str, str] = {}
        for unit_id, record in self.data.get("units", {}).items():
            if record.get("status") not in {"translated", "repaired"}:
                continue
            path = self.units_dir / f"{unit_id}.tex"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Committed translation file is missing for {unit_id}: {path}"
                )
            translations[unit_id] = path.read_text(encoding="utf-8")
        return translations

    def pending_ids(
        self,
        *,
        retry_fallbacks: bool = False,
        retry_repaired: bool = False,
    ) -> list[str]:
        allowed = {"pending"}
        if retry_fallbacks:
            allowed.add("fallback_original")
        if retry_repaired:
            allowed.add("repaired")
        return [
            unit_id
            for unit_id, record in self.data.get("units", {}).items()
            if record.get("status") in allowed
        ]

    def commit_translation(self, unit_id: str, content: str, record: dict) -> None:
        self.units_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.units_dir / f"{unit_id}.tex", content)
        self.data["units"][unit_id].update(record)
        self.save()

    def commit_fallback(self, unit_id: str, record: dict) -> None:
        (self.units_dir / f"{unit_id}.tex").unlink(missing_ok=True)
        self.data["units"][unit_id].update(
            {
                **record,
                "status": "fallback_original",
            }
        )
        self.save()

    def summary(self) -> dict[str, int]:
        counts = {
            "total": 0,
            "pending": 0,
            "translated": 0,
            "repaired": 0,
            "fallback_original": 0,
        }
        for record in self.data.get("units", {}).values():
            counts["total"] += 1
            status = record.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def save(self) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.data, ensure_ascii=False, indent=2) + "\n"
        _atomic_write_text(self.path, serialized)


def atomic_write_sources(project_dir: Path, sources: dict[str, str]) -> None:
    for relative_path, content in sources.items():
        target = project_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
