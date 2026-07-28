from pathlib import Path
from types import SimpleNamespace

import pytest
from lxml import etree

from pdf2epub.html_translation import builder as builder_module
from pdf2epub.html_translation.builder import BuildConfig, HTMLEpubBuilder
from pdf2epub.html_translation.translator import HTMLTranslateProcessor


OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"


def _builder(
    tmp_path: Path,
    *,
    epubcheck_mode: str = "off",
    epubcheck_path: str | None = None,
) -> HTMLEpubBuilder:
    return HTMLEpubBuilder(
        BuildConfig(
            original_epub=tmp_path / "input.epub",
            translated_dir=tmp_path / "translated",
            output_path=tmp_path / "output.epub",
            book_title="Book",
            epubcheck_mode=epubcheck_mode,
            epubcheck_path=epubcheck_path,
        )
    )


def test_update_content_opf_preserves_namespaced_attrs_and_collapses_creators(
    tmp_path: Path,
) -> None:
    meta_inf = tmp_path / "META-INF"
    oebps = tmp_path / "OEBPS"
    meta_inf.mkdir()
    oebps.mkdir()
    (meta_inf / "container.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        encoding="utf-8",
    )
    opf_path = oebps / "content.opf"
    opf_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="{OPF_NS}" xmlns:dc="{DC_NS}" xmlns:opf="{OPF_NS}"
         unique-identifier="book-id" version="2.0">
  <metadata>
    <dc:title>Original title</dc:title>
    <dc:creator opf:role="aut" opf:file-as="Alpha, A">Author One</dc:creator>
    <dc:creator opf:role="aut">Author Two</dc:creator>
    <dc:contributor opf:role="bkp">Producer</dc:contributor>
    <dc:language>en</dc:language>
    <dc:identifier id="book-id" opf:scheme="ISBN">123</dc:identifier>
  </metadata>
  <manifest/>
  <spine/>
</package>
""",
        encoding="utf-8",
    )

    _builder(tmp_path)._update_content_opf(
        tmp_path,
        {
            "translated_title": "Translated title",
            "target_language_code": "zh",
            "translated_author": "作者一, 作者二",
            "translated_author_file_as": "作者一, 作者二",
        },
    )

    root = etree.parse(str(opf_path)).getroot()
    namespaces = {"dc": DC_NS}
    creators = root.findall(".//dc:creator", namespaces)
    assert len(creators) == 1
    assert creators[0].text == "作者一, 作者二"
    assert creators[0].get(f"{{{OPF_NS}}}role") == "aut"
    assert creators[0].get(f"{{{OPF_NS}}}file-as") == "作者一, 作者二"
    assert creators[0].get("role") is None
    assert creators[0].get("file-as") is None

    contributor = root.find(".//dc:contributor", namespaces)
    identifier = root.find(".//dc:identifier", namespaces)
    assert contributor is not None
    assert identifier is not None
    assert contributor.get(f"{{{OPF_NS}}}role") == "bkp"
    assert identifier.get(f"{{{OPF_NS}}}scheme") == "ISBN"
    assert root.findtext(".//dc:title", namespaces=namespaces) == "Translated title"
    assert root.findtext(".//dc:language", namespaces=namespaces) == "zh"


def test_update_epub3_creators_updates_and_removes_refinements(
    tmp_path: Path,
) -> None:
    meta_inf = tmp_path / "META-INF"
    oebps = tmp_path / "OEBPS"
    meta_inf.mkdir()
    oebps.mkdir()
    (meta_inf / "container.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/package.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
""",
        encoding="utf-8",
    )
    opf_path = oebps / "package.opf"
    opf_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="{OPF_NS}" xmlns:dc="{DC_NS}"
         unique-identifier="book-id" version="3.0">
  <metadata>
    <dc:identifier id="book-id">book-id</dc:identifier>
    <dc:title>Original title</dc:title>
    <dc:language>en</dc:language>
    <dc:creator id="creator-1">Author One</dc:creator>
    <meta refines="#creator-1" property="role" scheme="marc:relators">aut</meta>
    <meta refines="#creator-1" property="file-as">Alpha, A</meta>
    <dc:creator id="creator-2">Author Two</dc:creator>
    <meta refines="#creator-2" property="role" scheme="marc:relators">aut</meta>
    <meta refines="#creator-2" property="file-as">Beta, B</meta>
    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
    <!-- Converter metadata comments are valid OPF children. -->
    <meta name="calibre:title_sort" content="Original title"/>
  </metadata>
  <manifest/>
  <spine/>
</package>
""",
        encoding="utf-8",
    )

    _builder(tmp_path)._update_content_opf(
        tmp_path,
        {
            "translated_title": "Translated title",
            "target_language_code": "zh",
            "translated_author": "作者一, 作者二",
            "translated_author_file_as": "作者一, 作者二",
            "translated_title_sort": "Translated title",
        },
    )

    root = etree.parse(str(opf_path)).getroot()
    namespaces = {"opf": OPF_NS, "dc": DC_NS}
    creators = root.findall(".//dc:creator", namespaces)
    assert [(creator.get("id"), creator.text) for creator in creators] == [
        ("creator-1", "作者一, 作者二")
    ]

    refinements = root.findall(".//opf:meta[@refines]", namespaces)
    assert all(meta.get("refines") != "#creator-2" for meta in refinements)
    first_creator_refinements = {
        meta.get("property"): meta.text
        for meta in refinements
        if meta.get("refines") == "#creator-1"
    }
    assert first_creator_refinements == {
        "role": "aut",
        "file-as": "作者一, 作者二",
    }
    title_sort = root.find(".//opf:meta[@name='calibre:title_sort']", namespaces)
    assert title_sort is not None
    assert title_sort.get("content") == "Translated title"


def test_epubcheck_warn_mode_keeps_failed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder(
        tmp_path,
        epubcheck_mode="warn",
        epubcheck_path="/tools/epubcheck",
    )
    monkeypatch.setattr(
        builder_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="ERROR: invalid package",
            stderr="",
        ),
    )

    builder._validate_output_epub()


def test_epubcheck_strict_mode_rejects_failed_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder(
        tmp_path,
        epubcheck_mode="strict",
        epubcheck_path="/tools/epubcheck",
    )
    monkeypatch.setattr(
        builder_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="ERROR: invalid package",
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="EPUBCheck failed"):
        builder._validate_output_epub()


def test_epubcheck_strict_mode_requires_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _builder(tmp_path, epubcheck_mode="strict")
    monkeypatch.setattr(builder_module.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="not installed"):
        builder._validate_output_epub()


def test_html_translator_google_agent_uses_provider_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import google.genai
    import pydantic_ai.models.google
    import pydantic_ai.providers.google

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        google.genai,
        "Client",
        lambda **kwargs: ("client", kwargs),
    )
    monkeypatch.setattr(
        pydantic_ai.providers.google,
        "GoogleProvider",
        lambda **kwargs: ("provider", kwargs),
    )
    monkeypatch.setattr(
        pydantic_ai.models.google,
        "GoogleModel",
        lambda model_name, provider: ("model", model_name, provider),
    )

    processor = HTMLTranslateProcessor(
        config={
            "credentials": {
                "providers": {
                    "gemini": {
                        "type": "google",
                        "api_key": "test-key",
                        "base_url": "https://example.invalid",
                    }
                }
            },
            "html_translation": {"agent_model": "gemini-test"},
        },
        book_title="Test Book",
    )

    model = processor._get_agent_model()

    assert model[0:2] == ("model", "gemini-test")
    assert model[2][0] == "provider"
