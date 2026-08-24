"""Load raw source files into LangChain ``Document`` objects.

Kept deliberately small and dependency-light: markdown/text are read directly,
PDFs via ``pypdf``. Each document carries ``metadata`` (at least ``source``) so
that retrieved chunks can always be traced back to where they came from — which
matters for citations and for the prompt-injection trust boundary later.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": path.name, "type": "text"})]


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": path.name, "type": "pdf", "page": page_number},
                )
            )
    return docs


def load_file(path: Path) -> list[Document]:
    """Load a single supported file into one or more documents."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _load_text(path)
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path.name})")


def load_directory(directory: Path, *, skip_readme: bool = True) -> list[Document]:
    """Recursively load every supported file under ``directory``."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {directory}")

    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if skip_readme and path.name.lower() == "readme.md":
            continue
        docs.extend(load_file(path))
    return docs
