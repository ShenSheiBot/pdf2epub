"""Discover, segment, and deterministically render a TeX project."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_DOCUMENT_CLASS_RE = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOCUMENT_RE = re.compile(r"\\end\s*\{document\}")
_FRONT_MATTER_COMMAND_RE = re.compile(
    r"(?m)^[ \t]*\\(?:title|subtitle)\*?\s*"
    r"(?:\[[^\]\n]*\]\s*)?\{"
)
_CJK_PACKAGE_RE = re.compile(
    r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{(?:ctex|xeCJK)\}"
)
_CTEX_CLASS_RE = re.compile(
    r"\\documentclass(?:\[[^\]]*\])?\{(?:ctexart|ctexbook|ctexrep|ctexbeamer)\}"
)
_PDFOUTPUT_LINE_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)\\pdfoutput\s*=\s*(?P<value>[01])"
    r"(?P<suffix>[ \t]*(?:%[^\n]*)?)$"
)
_INPUTENC_LINE_RE = re.compile(
    r"(?m)^[ \t]*\\(?:usepackage|RequirePackage)"
    r"(?:\[[^\]\n]*\])?\{inputenc\}[^\n]*(?:\n|\Z)"
)
_MICROTYPE_LINE_RE = re.compile(
    r"(?m)^[ \t]*\\(?:usepackage|RequirePackage)"
    r"(?:\[[^\]\n]*\])?\{microtype\}[^\n]*(?:\n|\Z)"
)
_CJKUTF8_PACKAGE_LINE_RE = re.compile(
    r"(?m)^[ \t]*\\(?:usepackage|RequirePackage)"
    r"(?:\[[^\]\n]*\])?\{(?:CJK|CJKutf8)\}[^\n]*(?:\n|\Z)"
)
_CJK_ENV_BEGIN_RE = re.compile(
    r"\\begin\s*\{CJK\*?\}\s*\{UTF8\}\s*\{[^{}]+\}"
)
_CJK_ENV_END_RE = re.compile(r"\\end\s*\{CJK\*?\}")
_NEW_THEOREM_RE = re.compile(
    r"(\\newtheorem\s*\{[^{}]+\}(?:\s*\[[^\]]+\])?\s*\{)"
    r"(Theorem|Corollary|Lemma|Proposition|Claim|Definition|Remark|Example)"
    r"(\})"
)
_INCLUDE_RE = re.compile(
    r"\\(?:input|include|subfile)\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}"
)
_IMPORT_RE = re.compile(
    r"\\(?:import|subimport|inputfrom|subinputfrom)\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
)
_PARAGRAPH_RE = re.compile(r".*?(?:\n[ \t]*\n|\Z)", re.DOTALL)
_COMMAND_NAME_RE = re.compile(r"\\[A-Za-z@]+")
_COMMENT_RE = re.compile(r"(?m)(?<!\\)%.*$")

CJK_PACKAGE_PREAMBLE = (
    "\n% Added by pdf2epub for Unicode CJK translation.\n"
    "\\usepackage[scheme=plain,fontset=fandol]{ctex}\n"
    "\\IfFontExistsTF{Noto Serif CJK SC}{%\n"
    "  \\setCJKmainfont{Noto Serif CJK SC}%\n"
    "}{\\IfFontExistsTF{Source Han Serif SC}{%\n"
    "  \\setCJKmainfont{Source Han Serif SC}%\n"
    "}{\\IfFontExistsTF{Arial Unicode MS}{%\n"
    "  \\setCJKmainfont{Arial Unicode MS}%\n"
    "}{}}}\n"
)
CHINESE_LABEL_PREAMBLE = (
    "\\usepackage{etoolbox}\n"
    "\\AtBeginDocument{%\n"
    "  \\providecommand{\\abstractname}{}\\renewcommand{\\abstractname}{摘要}%\n"
    "  \\patchcmd{\\abstract}{Abstract}{摘要}{}{}%\n"
    "  \\providecommand{\\contentsname}{}\\renewcommand{\\contentsname}{目录}%\n"
    "  \\providecommand{\\refname}{}\\renewcommand{\\refname}{参考文献}%\n"
    "  \\providecommand{\\proofname}{}\\renewcommand{\\proofname}{证明}%\n"
    "  \\providecommand{\\figurename}{}\\renewcommand{\\figurename}{图}%\n"
    "  \\providecommand{\\tablename}{}\\renewcommand{\\tablename}{表}%\n"
    "}\n"
)
_CHINESE_THEOREM_NAMES = {
    "Theorem": "定理",
    "Corollary": "推论",
    "Lemma": "引理",
    "Proposition": "命题",
    "Claim": "断言",
    "Definition": "定义",
    "Remark": "注记",
    "Example": "例",
}
_SIMPLIFIED_CHINESE_TARGETS = {
    "chinese",
    "simplified chinese",
    "zh",
    "zh-cn",
    "zh-hans",
    "中文",
    "简体中文",
}


@dataclass(frozen=True)
class TranslationUnit:
    """An immutable source span that can be translated independently."""

    id: str
    relative_path: str
    start: int
    end: int
    source_sha256: str
    source_text: str

    def manifest_entry(self) -> dict:
        return {
            "id": self.id,
            "relative_path": self.relative_path,
            "start": self.start,
            "end": self.end,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class TexProjectDocument:
    """Prepared TeX sources and their stable translation unit manifest."""

    root: Path
    main_tex: str
    sources: dict[str, str]
    units: tuple[TranslationUnit, ...]
    source_fingerprint: str
    layout_fingerprint: str
    warnings: tuple[str, ...] = ()

    def render(self, translations: dict[str, str]) -> dict[str, str]:
        """Render every tracked source from immutable text plus unit replacements."""
        rendered: dict[str, str] = {}
        units_by_file: dict[str, list[TranslationUnit]] = {}
        for unit in self.units:
            units_by_file.setdefault(unit.relative_path, []).append(unit)

        for relative_path, source in self.sources.items():
            replacements = units_by_file.get(relative_path, [])
            result = source
            for unit in sorted(replacements, key=lambda item: item.start, reverse=True):
                replacement = translations.get(unit.id, unit.source_text)
                result = result[: unit.start] + replacement + result[unit.end :]
            rendered[relative_path] = result
        return rendered


def read_tex(path: Path) -> str:
    """Read common arXiv TeX encodings and normalize output to Unicode text."""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", data, 0, 1, f"Cannot decode {path}")


def inject_cjk_support(
    source: str,
    *,
    target_language: str = "Simplified Chinese",
) -> str:
    """Add idempotent XeLaTeX CJK support and target-specific labels."""
    source = _normalize_xelatex_source(source)
    localize_chinese = target_language.strip().lower() in _SIMPLIFIED_CHINESE_TARGETS
    if localize_chinese:
        source = _NEW_THEOREM_RE.sub(
            lambda match: (
                match.group(1)
                + _CHINESE_THEOREM_NAMES[match.group(2)]
                + match.group(3)
            ),
            source,
        )
    if _CJK_PACKAGE_RE.search(source) or _CTEX_CLASS_RE.search(source):
        return source
    match = _DOCUMENT_CLASS_RE.search(source)
    if not match:
        raise ValueError("Main TeX file has no \\documentclass declaration")
    preamble = CJK_PACKAGE_PREAMBLE
    if localize_chinese:
        preamble += CHINESE_LABEL_PREAMBLE
    return source[: match.end()] + preamble + source[match.end() :]


def _normalize_xelatex_source(source: str) -> str:
    """Remove common pdfLaTeX-only directives while preserving their content."""
    source = _INPUTENC_LINE_RE.sub(
        "% Disabled by pdf2epub: XeLaTeX reads UTF-8 natively.\n",
        source,
    )
    source = _MICROTYPE_LINE_RE.sub(
        "% Disabled by pdf2epub: legacy Type1 fonts can break microtype in XeLaTeX.\n",
        source,
    )
    source = _PDFOUTPUT_LINE_RE.sub(
        lambda match: (
            f"{match.group('indent')}\\ifdefined\\pdfoutput"
            f"\\pdfoutput={match.group('value')}\\fi"
            f"{match.group('suffix')}"
        ),
        source,
    )
    source = _CJKUTF8_PACKAGE_LINE_RE.sub(
        "% Replaced by pdf2epub: ctex provides native XeLaTeX CJK support.\n",
        source,
    )
    source = _CJK_ENV_BEGIN_RE.sub("", source)
    return _CJK_ENV_END_RE.sub("", source)


def discover_main_tex(root: Path, explicit: str | Path | None = None) -> str:
    """Find the most likely compilation entry point relative to ``root``."""
    root = root.resolve()
    if explicit:
        candidate = Path(explicit)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Main TeX file must be inside source root: {explicit}"
                ) from exc
        resolved = (root / candidate).resolve()
        if not resolved.is_file() or not _is_within(resolved, root):
            raise FileNotFoundError(
                f"Main TeX file not found inside source root: {explicit}"
            )
        return candidate.as_posix()

    candidates: list[tuple[int, str]] = []
    for path in sorted(root.rglob("*.tex")):
        if not path.is_file() or not _is_within(path.resolve(), root):
            continue
        text = read_tex(path)
        if not (_DOCUMENT_CLASS_RE.search(text) and _BEGIN_DOCUMENT_RE.search(text)):
            continue
        relative = path.relative_to(root).as_posix()
        score = len(text)
        if path.name.lower() in {"main.tex", "paper.tex", "article.tex", "ms.tex"}:
            score += 10_000_000
        if path.parent == root:
            score += 1_000_000
        candidates.append((score, relative))

    if not candidates:
        raise FileNotFoundError(
            "No TeX entry point with both \\documentclass and "
            "\\begin{document} was found"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def scan_project(
    root: Path,
    main_tex: str,
    *,
    unit_chars: int = 12_000,
    target_language: str = "Simplified Chinese",
) -> TexProjectDocument:
    """Prepare the main file, follow body includes, and create stable units."""
    if unit_chars < 1_000:
        raise ValueError("unit_chars must be at least 1000")

    root = root.resolve()
    main_path = (root / main_tex).resolve()
    if not _is_within(main_path, root) or not main_path.is_file():
        raise FileNotFoundError(f"Main TeX file not found: {main_tex}")

    sources: dict[str, str] = {}
    ranges: dict[str, tuple[int, int]] = {}
    warnings: list[str] = []
    queue: list[tuple[str, bool]] = [(PurePosixPath(main_tex).as_posix(), True)]
    visited: set[str] = set()

    while queue:
        relative_path, is_main = queue.pop(0)
        if relative_path in visited:
            continue
        visited.add(relative_path)
        path = (root / relative_path).resolve()
        if not _is_within(path, root) or not path.is_file():
            warnings.append(f"Referenced TeX file not found: {relative_path}")
            continue

        text = _normalize_xelatex_source(read_tex(path))
        if is_main:
            text = inject_cjk_support(text, target_language=target_language)
        sources[relative_path] = text
        body_start, body_end = _body_range(text, require_document=is_main)
        ranges[relative_path] = (body_start, body_end)

        base_dir = PurePosixPath(relative_path).parent
        body = _COMMENT_RE.sub("", text[body_start:body_end])
        for include in _find_includes(body):
            resolved = _resolve_include(root, base_dir, include)
            if resolved is None:
                warnings.append(
                    "Skipped dynamic or external TeX include in "
                    f"{relative_path}: {include}"
                )
                continue
            if resolved not in visited and all(item[0] != resolved for item in queue):
                queue.append((resolved, False))

    units: list[TranslationUnit] = []
    next_index = 1
    for relative_path, source in sources.items():
        start, end = ranges[relative_path]
        for unit_start, unit_end in _split_range(source, start, end, unit_chars):
            fragment = source[unit_start:unit_end]
            if not _has_translatable_text(fragment):
                continue
            units.append(
                TranslationUnit(
                    id=f"unit-{next_index:05d}",
                    relative_path=relative_path,
                    start=unit_start,
                    end=unit_end,
                    source_sha256=_sha256_text(fragment),
                    source_text=fragment,
                )
            )
            next_index += 1

    # TeX commonly places the visible title before \begin{document}. Keep the
    # preamble itself out of model input, but append its user-facing metadata as
    # isolated units so existing body unit IDs remain stable across upgrades.
    main_source = sources[PurePosixPath(main_tex).as_posix()]
    main_body_start, _ = ranges[PurePosixPath(main_tex).as_posix()]
    for unit_start, unit_end in _front_matter_ranges(
        main_source,
        end=main_body_start,
    ):
        fragment = main_source[unit_start:unit_end]
        if not _has_translatable_text(fragment):
            continue
        units.append(
            TranslationUnit(
                id=f"unit-{next_index:05d}",
                relative_path=PurePosixPath(main_tex).as_posix(),
                start=unit_start,
                end=unit_end,
                source_sha256=_sha256_text(fragment),
                source_text=fragment,
            )
        )
        next_index += 1

    source_payload = {
        "main_tex": PurePosixPath(main_tex).as_posix(),
        "sources": sources,
    }
    source_fingerprint = _sha256_json(source_payload)
    layout_fingerprint = _sha256_json(
        {
            "source_fingerprint": source_fingerprint,
            "unit_chars": unit_chars,
            "units": [unit.manifest_entry() for unit in units],
        }
    )
    return TexProjectDocument(
        root=root,
        main_tex=PurePosixPath(main_tex).as_posix(),
        sources=sources,
        units=tuple(units),
        source_fingerprint=source_fingerprint,
        layout_fingerprint=layout_fingerprint,
        warnings=tuple(warnings),
    )


def _body_range(text: str, *, require_document: bool) -> tuple[int, int]:
    begin = _BEGIN_DOCUMENT_RE.search(text)
    if not begin:
        if require_document:
            raise ValueError("Main TeX file has no \\begin{document}")
        return 0, len(text)
    end = _END_DOCUMENT_RE.search(text, begin.end())
    return begin.end(), end.start() if end else len(text)


def _front_matter_ranges(text: str, *, end: int) -> list[tuple[int, int]]:
    """Return balanced, visible metadata commands from the TeX preamble."""
    ranges: list[tuple[int, int]] = []
    for match in _FRONT_MATTER_COMMAND_RE.finditer(text, 0, end):
        group_end = _balanced_group_end(text, match.end() - 1, limit=end)
        if group_end is not None:
            ranges.append((match.start(), group_end))
    return ranges


def _balanced_group_end(text: str, start: int, *, limit: int) -> int | None:
    """Return the exclusive end of a balanced TeX brace group."""
    depth = 0
    for index in range(start, limit):
        character = text[index]
        if character not in "{}" or _is_escaped(text, index):
            continue
        if character == "{":
            depth += 1
            continue
        depth -= 1
        if depth == 0:
            return index + 1
    return None


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _find_includes(body: str) -> Iterable[str]:
    for match in _INCLUDE_RE.finditer(body):
        yield match.group(1).strip()
    for match in _IMPORT_RE.finditer(body):
        directory = match.group(1).strip()
        filename = match.group(2).strip()
        yield str(PurePosixPath(directory) / filename)


def _resolve_include(root: Path, base_dir: PurePosixPath, include: str) -> str | None:
    if not include or "\\" in include or "#" in include:
        return None
    relative = PurePosixPath(include)
    if relative.suffix == "":
        relative = relative.with_suffix(".tex")
    combined = base_dir / relative
    path = (root / Path(*combined.parts)).resolve()
    if not _is_within(path, root):
        return None
    if not path.is_file():
        return None
    return path.relative_to(root).as_posix()


def _split_range(
    source: str,
    start: int,
    end: int,
    target_chars: int,
) -> list[tuple[int, int]]:
    """Split only at paragraph boundaries; oversized atomic blocks stay intact."""
    body = source[start:end]
    blocks = [
        match.group(0) for match in _PARAGRAPH_RE.finditer(body) if match.group(0)
    ]
    if not blocks:
        return []

    spans: list[tuple[int, int]] = []
    cursor = start
    unit_start = start
    unit_size = 0
    for block in blocks:
        block_size = len(block)
        if unit_size and unit_size + block_size > target_chars:
            spans.append((unit_start, cursor))
            unit_start = cursor
            unit_size = 0
        cursor += block_size
        unit_size += block_size
    if cursor > unit_start:
        spans.append((unit_start, cursor))
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _has_translatable_text(fragment: str) -> bool:
    without_comments = _COMMENT_RE.sub("", fragment)
    without_commands = _COMMAND_NAME_RE.sub("", without_comments)
    return bool(
        re.search(
            r"[A-Za-z\u00C0-\u024F\u3040-\u30FF\u3400-\u9FFF]{3}", without_commands
        )
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
