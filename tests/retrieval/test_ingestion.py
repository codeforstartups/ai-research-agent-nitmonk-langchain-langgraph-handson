"""Phase 04 — loaders and chunking."""

from __future__ import annotations

from pathlib import Path

from ai_research_agent.ingestion import chunk_documents, load_directory, load_file


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_text_file_sets_source_metadata(tmp_path: Path) -> None:
    path = _write(tmp_path, "note.md", "# Title\n\nHello world.")
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "note.md"
    assert "Hello world" in docs[0].page_content


def test_load_directory_skips_readme(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "ignore me")
    _write(tmp_path, "a.md", "alpha content")
    _write(tmp_path, "b.txt", "beta content")
    docs = load_directory(tmp_path)
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"a.md", "b.txt"}


def test_chunking_splits_and_indexes(tmp_path: Path) -> None:
    long_text = "\n\n".join(f"Paragraph {i} " + ("word " * 60) for i in range(10))
    path = _write(tmp_path, "big.md", long_text)
    docs = load_file(path)
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    # Chunk indices are contiguous per source.
    indices = [c.metadata["chunk"] for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.metadata["source"] == "big.md" for c in chunks)


def test_unsupported_type_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "data.bin", "x")
    try:
        load_file(path)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
