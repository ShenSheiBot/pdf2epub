from pdf2epub import build_epub


def test_heading_embeddings_disabled_below_memory_threshold(monkeypatch):
    monkeypatch.delenv("PDF2EPUB_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.delenv("PDF2EPUB_DISABLE_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.delenv("PDF2EPUB_HEADING_EMBEDDING_MIN_RAM_GB", raising=False)
    monkeypatch.setattr(build_epub, "_get_total_memory_bytes", lambda: 4 * 1024 ** 3)
    monkeypatch.setattr(build_epub, "_embedding_model", None)

    assert build_epub._should_use_heading_embeddings() is False
    assert build_epub._get_embedding_model() is False


def test_heading_embeddings_allowed_at_or_above_memory_threshold(monkeypatch):
    monkeypatch.delenv("PDF2EPUB_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.delenv("PDF2EPUB_DISABLE_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.delenv("PDF2EPUB_HEADING_EMBEDDING_MIN_RAM_GB", raising=False)
    monkeypatch.setattr(build_epub, "_get_total_memory_bytes", lambda: 8 * 1024 ** 3)

    assert build_epub._should_use_heading_embeddings() is True


def test_heading_embeddings_env_override(monkeypatch):
    monkeypatch.setattr(build_epub, "_get_total_memory_bytes", lambda: 4 * 1024 ** 3)

    monkeypatch.setenv("PDF2EPUB_HEADING_EMBEDDINGS", "enabled")
    assert build_epub._should_use_heading_embeddings() is True

    monkeypatch.setenv("PDF2EPUB_HEADING_EMBEDDINGS", "disabled")
    assert build_epub._should_use_heading_embeddings() is False


def test_heading_embedding_memory_threshold_env(monkeypatch):
    monkeypatch.delenv("PDF2EPUB_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.delenv("PDF2EPUB_DISABLE_HEADING_EMBEDDINGS", raising=False)
    monkeypatch.setattr(build_epub, "_get_total_memory_bytes", lambda: 6 * 1024 ** 3)

    monkeypatch.setenv("PDF2EPUB_HEADING_EMBEDDING_MIN_RAM_GB", "4")
    assert build_epub._should_use_heading_embeddings() is True

    monkeypatch.setenv("PDF2EPUB_HEADING_EMBEDDING_MIN_RAM_GB", "8")
    assert build_epub._should_use_heading_embeddings() is False
