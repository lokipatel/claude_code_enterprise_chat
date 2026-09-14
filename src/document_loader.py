"""Loads project documents (text/code/docx/pptx/images) for graph ingestion.

Every supported file type is reduced to plain text so it can be fed into
Graphiti the same way an email body is: images don't get real OCR here,
they get their format/dimensions plus a human-authored caption from an
optional `<name>.caption.txt` sidecar next to the image.

Documents are then chunked with the `semantica` library before ingestion --
emails are short enough to stay as a single Graphiti episode each, but a long
document (a project plan, a spec) gets split into smaller, semantically
coherent pieces so each Graphiti extraction call has a focused, bounded
amount of text to work with.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from docx import Document as DocxDocument
from PIL import Image
from pptx import Presentation
from pydantic import BaseModel, Field, model_validator
from semantica.split import TextSplitter

TEXT_EXTENSIONS = {
    ".txt": "text",
    ".md": "text",
    ".sql": "sql",
    ".py": "python",
    ".java": "java",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

# Tuned for solid LLM extraction granularity while keeping enough context per
# chunk for coherent entity/fact extraction.
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100

_splitter = TextSplitter(method="recursive", chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)


def chunk_text(text: str) -> list[str]:
    """Split document text into semantically coherent chunks via semantica."""
    if not text.strip():
        return []
    chunks = _splitter.split(text)
    return [chunk.text for chunk in chunks] if chunks else [text]


class DocumentRecord(BaseModel):
    """A project document reduced to plain text (and chunks), ready for Graphiti ingestion."""

    doc_id: str
    filename: str
    doc_type: str
    content_text: str
    modified_at: datetime
    chunks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _populate_chunks(self) -> DocumentRecord:
        if not self.chunks:
            self.chunks = chunk_text(self.content_text)
        return self


class UnsupportedDocumentTypeError(ValueError):
    """Raised when a file extension has no registered loader."""


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_docx(path: Path) -> str:
    document = DocxDocument(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def _read_pptx(path: Path) -> str:
    presentation = Presentation(str(path))
    slide_texts: list[str] = []

    for index, slide in enumerate(presentation.slides, start=1):
        texts = [
            shape.text_frame.text.strip()
            for shape in slide.shapes
            if shape.has_text_frame and shape.text_frame.text.strip()
        ]
        if texts:
            slide_texts.append(f"Slide {index}: " + " — ".join(texts))

    return "\n".join(slide_texts)


def _describe_image(path: Path) -> str:
    with Image.open(path) as img:
        width, height = img.size
        image_format = img.format or path.suffix.lstrip(".").upper()

    caption_path = path.with_name(f"{path.stem}.caption.txt")
    caption = (
        caption_path.read_text(encoding="utf-8").strip()
        if caption_path.exists()
        else "(no caption available)"
    )

    return (
        f"Image file: {path.name}\n"
        f"Format: {image_format}, Dimensions: {width}x{height}\n"
        f"Caption: {caption}"
    )


def load_document(path: Path, root: Path) -> DocumentRecord:
    """Parse a single file on disk into a DocumentRecord."""
    ext = path.suffix.lower()
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    doc_id = str(path.relative_to(root))

    if ext in TEXT_EXTENSIONS:
        doc_type = TEXT_EXTENSIONS[ext]
        content_text = _read_text_file(path)
    elif ext == ".docx":
        doc_type = "docx"
        content_text = _read_docx(path)
    elif ext == ".pptx":
        doc_type = "pptx"
        content_text = _read_pptx(path)
    elif ext in IMAGE_EXTENSIONS:
        doc_type = "image"
        content_text = _describe_image(path)
    else:
        raise UnsupportedDocumentTypeError(f"No loader registered for: {path}")

    return DocumentRecord(
        doc_id=doc_id,
        filename=path.name,
        doc_type=doc_type,
        content_text=content_text,
        modified_at=modified_at,
    )


def load_documents_dir(root: Path) -> list[DocumentRecord]:
    """Load every supported document under `root`, skipping caption sidecars."""
    records: list[DocumentRecord] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name.endswith(".caption.txt"):
            continue
        records.append(load_document(path, root))

    return records
