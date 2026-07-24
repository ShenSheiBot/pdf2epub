"""Resolve arXiv identifiers and local TeX sources into an immutable source tree."""

from __future__ import annotations

import gzip
import re
import shutil
import ssl
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_ARXIV_ID_RE = re.compile(
    r"^(?:arxiv:)?(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)$",
    re.IGNORECASE,
)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf|e-print)/(?P<id>[^?#]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ResolvedSource:
    source_dir: Path
    source_id: str
    suggested_main_tex: str | None = None


class ArxivSourceResolver:
    """Materialize local directories, archives, TeX files, or arXiv e-prints."""

    def __init__(self, *, max_download_bytes: int = 512 * 1024 * 1024):
        self.max_download_bytes = max_download_bytes

    def materialize(self, source: str | Path, destination: Path) -> ResolvedSource:
        destination = destination.resolve()
        if destination.exists():
            if not destination.is_dir():
                raise NotADirectoryError(
                    f"TeX source destination is not a directory: {destination}"
                )
            if any(destination.iterdir()):
                return ResolvedSource(destination, self.source_id(source))
        destination.mkdir(parents=True, exist_ok=True)

        try:
            local = Path(source).expanduser()
            if local.exists():
                if local.is_dir():
                    _reject_nested_destination(local.resolve(), destination)
                    _copy_tree(local.resolve(), destination)
                    return ResolvedSource(destination, local.name)
                if local.suffix.lower() == ".tex":
                    _reject_nested_destination(local.resolve().parent, destination)
                    _copy_tree(local.resolve().parent, destination)
                    return ResolvedSource(
                        destination,
                        local.stem,
                        local.resolve().relative_to(
                            local.resolve().parent
                        ).as_posix(),
                    )
                self._extract_archive(local.resolve(), destination)
                return ResolvedSource(destination, _archive_stem(local))

            arxiv_id = normalize_arxiv_id(str(source))
            self._download_arxiv(arxiv_id, destination)
            return ResolvedSource(destination, arxiv_id)
        except Exception:
            # Never leave a partial tree that a resume could mistake for a
            # successfully materialized immutable source snapshot.
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def source_id(self, source: str | Path) -> str:
        local = Path(source).expanduser()
        if local.exists():
            return local.stem if local.is_file() else local.name
        return normalize_arxiv_id(str(source))

    def _download_arxiv(self, arxiv_id: str, destination: Path) -> None:
        safe_id = urllib.parse.quote(arxiv_id, safe="/")
        url = f"https://export.arxiv.org/e-print/{safe_id}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "pdf2epub/0.2 (arXiv TeX translation)"},
        )
        archive_path = destination.parent / f".{slugify_source_id(arxiv_id)}.download"
        try:
            tls_context = _tls_context()
            with urllib.request.urlopen(
                request,
                timeout=120,
                context=tls_context,
            ) as response:
                total = 0
                with archive_path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > self.max_download_bytes:
                            raise ValueError(
                                f"arXiv source exceeds {self.max_download_bytes} bytes"
                            )
                        output.write(chunk)
            self._extract_archive(archive_path, destination)
        finally:
            archive_path.unlink(missing_ok=True)

    def _extract_archive(self, archive: Path, destination: Path) -> None:
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as tar:
                _safe_extract_tar(
                    tar,
                    destination,
                    max_bytes=self.max_download_bytes,
                )
            _flatten_single_directory(destination)
            return

        with archive.open("rb") as source:
            gzip_header = source.read(2)
        if gzip_header == b"\x1f\x8b":
            with gzip.open(archive, "rb") as source:
                decompressed = source.read(self.max_download_bytes + 1)
        else:
            if archive.stat().st_size > self.max_download_bytes:
                raise ValueError(
                    f"Source file exceeds {self.max_download_bytes} bytes"
                )
            decompressed = archive.read_bytes()
        if len(decompressed) > self.max_download_bytes:
            raise ValueError(
                f"Expanded source exceeds {self.max_download_bytes} bytes"
            )
        if not decompressed.strip():
            raise ValueError(f"Source archive is empty: {archive}")
        (destination / "main.tex").write_bytes(decompressed)


def normalize_arxiv_id(value: str) -> str:
    """Normalize ``arXiv:...`` and arxiv.org URLs to a source identifier."""
    cleaned = value.strip()
    url_match = _ARXIV_URL_RE.search(cleaned)
    if url_match:
        cleaned = urllib.parse.unquote(url_match.group("id"))
        if cleaned.lower().endswith(".pdf"):
            cleaned = cleaned[:-4]
    match = _ARXIV_ID_RE.fullmatch(cleaned)
    if not match:
        raise ValueError(f"Not a local source or valid arXiv identifier/URL: {value}")
    return match.group("id")


def slugify_source_id(source_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._")
    return slug or "arxiv-source"


def _safe_extract_tar(
    tar: tarfile.TarFile,
    destination: Path,
    *,
    max_bytes: int,
) -> None:
    destination = destination.resolve()
    members: list[tarfile.TarInfo] = []
    expanded_bytes = 0
    for member in tar.getmembers():
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in source archive: {member.name}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"Unsupported archive member: {member.name}")
        target = (destination / Path(*member_path.parts)).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe path in source archive: {member.name}") from exc
        expanded_bytes += member.size
        if expanded_bytes > max_bytes:
            raise ValueError(f"Expanded source exceeds {max_bytes} bytes")
        members.append(member)
    tar.extractall(destination, members=members, filter="data")


def _flatten_single_directory(destination: Path) -> None:
    entries = list(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    nested = entries[0]
    for child in list(nested.iterdir()):
        child.replace(destination / child.name)
    nested.rmdir()


def _copy_tree(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_symlink():
            raise ValueError(f"Symlinks are not supported in TeX source trees: {child}")
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _reject_nested_destination(source: Path, destination: Path) -> None:
    try:
        destination.relative_to(source)
    except ValueError:
        return
    raise ValueError(
        "The TeX run directory must not be nested inside the local source tree"
    )


def _archive_stem(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".gz", ".tar"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _tls_context() -> ssl.SSLContext:
    """Use the package CA bundle on Python installs without system roots."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())
