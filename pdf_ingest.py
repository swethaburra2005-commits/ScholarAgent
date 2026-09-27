"""PDF ingestion and corpus indexing for ScholarAgent."""

from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


EMBEDDING_FILE_NAME = "multi_paper_embeddings.json"


def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts without changing meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)          # word split across PDF line
    text = re.sub(r"\s*\n\s*", " ", text)          # remaining line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = clean_pdf_text(page_text)
        if page_text:
            pages.append(page_text)

    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_words: int = 450, overlap_words: int = 60):
    """Create overlapping word-based research chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def _load_corpus(embedding_file: Path):
    if not embedding_file.exists():
        return []

    with embedding_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "embeddings", "chunks", "records"):
            if isinstance(payload.get(key), list):
                return payload[key]

    raise ValueError(
        "Unsupported embedding JSON structure. "
        "Expected a list of embedding records."
    )


def _save_corpus(embedding_file: Path, records):
    embedding_file.parent.mkdir(parents=True, exist_ok=True)

    with embedding_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def _next_paper_id(records) -> str:
    numbers = []

    for record in records:
        match = re.fullmatch(
            r"paper_(\d+)",
            str(record.get("paper_id", "")),
        )
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers, default=0) + 1
    return f"paper_{next_number:03d}"


def index_pdf(
    pdf_bytes: bytes,
    filename: str,
    embedding_model,
    embedding_file: Path,
):
    """Extract, chunk, embed, and append one PDF to the corpus."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Please upload a PDF file.")

    records = _load_corpus(embedding_file)

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Prevent accidental duplicate indexing of the same PDF.
    for record in records:
        if record.get("source_hash") == file_hash:
            return {
                "status": "duplicate",
                "paper_id": record.get("paper_id", "unknown"),
                "chunks_added": 0,
                "message": "This PDF is already indexed.",
            }

    text = extract_pdf_text(pdf_bytes)

    if len(text) < 100:
        raise ValueError(
            "Very little text could be extracted from this PDF. "
            "It may be scanned/image-only. OCR is not enabled yet."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No usable text chunks were created.")

    paper_id = _next_paper_id(records)

    title = Path(filename).stem
    title = re.sub(r"[_\-]+", " ", title).strip()
    title = re.sub(r"\s+", " ", title)

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    start_chunk_id = 1
    existing_chunk_ids = [
        int(record["chunk_id"])
        for record in records
        if record.get("paper_id") == paper_id
        and str(record.get("chunk_id", "")).isdigit()
    ]

    if existing_chunk_ids:
        start_chunk_id = max(existing_chunk_ids) + 1

    new_records = []

    for offset, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        new_records.append(
            {
                "paper_id": paper_id,
                "chunk_id": str(start_chunk_id + offset),
                "title": title,
                "arxiv_url": "",
                "text": chunk,
                "embedding": vector.tolist(),
                "source_file": filename,
                "source_hash": file_hash,
            }
        )

    records.extend(new_records)
    _save_corpus(embedding_file, records)

    return {
        "status": "indexed",
        "paper_id": paper_id,
        "chunks_added": len(new_records),
        "message": f"{title} was indexed successfully.",
    }
