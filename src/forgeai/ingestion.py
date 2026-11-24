"""Document profiling, chunking, and hybrid (BM25 + TF-IDF) retrieval."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal

import pymupdf as fitz
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from forgeai.schemas import CorpusManifest, DocProfile, DocumentChunk
from forgeai.tools import detect_csv_delimiter

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _extract_pdf_pages(path: Path) -> list[tuple[str, str]]:
    """Return list of (page_label, text) for a PDF file."""
    pages: list[tuple[str, str]] = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((f"page {idx}", text))
    return pages


def _extract_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = detect_csv_delimiter(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        raw_cols = list(reader.fieldnames or [])
        columns = [c.strip() for c in raw_cols]
        reader.fieldnames = columns
        rows = [
            {
                (k.strip() if isinstance(k, str) else str(k)): (
                    v.strip() if isinstance(v, str) else ("" if v is None else str(v))
                )
                for k, v in r.items()
                if k is not None
            }
            for r in reader
        ]
    return columns, rows


def _extract_text_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
        elif len(stripped) < 80 and stripped.isupper() and len(stripped.split()) <= 10:
            headings.append(stripped)
        if len(headings) >= 12:
            break
    return headings


def _summarize_lines(text: str, max_bullets: int = 3) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        clean = " ".join(line.split())
        if len(clean) >= 40 and not clean.startswith("#"):
            bullets.append(clean[:200] + ("..." if len(clean) > 200 else ""))
        if len(bullets) >= max_bullets:
            break
    return bullets


def profile_corpus(docs_dir: str) -> CorpusManifest:
    """Create a lightweight CorpusManifest for all supported documents in docs_dir."""
    base = Path(docs_dir)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    profiles: list[DocProfile] = []

    for path in files:
        rel_name = str(path.relative_to(base)).replace("\\", "/")
        ext = path.suffix.lower().lstrip(".")
        file_type: Literal["pdf", "csv", "txt", "md"] = ext  # type: ignore[assignment]

        if file_type == "pdf":
            pages = _extract_pdf_pages(path)
            full_text = "\n".join(t for _, t in pages)
            words = full_text.split()
            headings = _extract_text_headings(full_text)
            title = headings[0] if headings else path.stem.replace("_", " ").title()
            profiles.append(
                DocProfile(
                    filename=rel_name,
                    file_type="pdf",
                    title=title,
                    headings=headings,
                    summary_bullets=_summarize_lines(full_text),
                    word_count=len(words),
                )
            )
        elif file_type == "csv":
            columns, rows = _extract_csv_rows(path)
            profiles.append(
                DocProfile(
                    filename=rel_name,
                    file_type="csv",
                    title=path.stem.replace("_", " ").title(),
                    csv_columns=columns,
                    csv_row_count=len(rows),
                    csv_sample_rows=rows[:3],
                    word_count=sum(len(" ".join(str(v) for v in r.values()).split()) for r in rows),
                )
            )
        else:
            full_text = path.read_text(encoding="utf-8", errors="replace")
            headings = _extract_text_headings(full_text)
            title = headings[0] if headings else path.stem.replace("_", " ").title()
            profiles.append(
                DocProfile(
                    filename=rel_name,
                    file_type=file_type,
                    title=title,
                    headings=headings,
                    summary_bullets=_summarize_lines(full_text),
                    word_count=len(full_text.split()),
                )
            )

    return CorpusManifest(docs_dir=str(base), total_files=len(profiles), documents=profiles)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping character chunks snapping to whitespace when possible."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    overlap = min(chunk_overlap, chunk_size // 2)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            snap = text.rfind(" ", start + chunk_size // 2, end)
            if snap > start:
                end = snap
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_corpus(
    docs_dir: str, chunk_size: int = 500, chunk_overlap: int = 80
) -> list[DocumentChunk]:
    """Load and chunk all supported documents in docs_dir."""
    base = Path(docs_dir)
    files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    chunks: list[DocumentChunk] = []

    for path in files:
        rel_name = str(path.relative_to(base)).replace("\\", "/")
        ext = path.suffix.lower()

        if ext == ".pdf":
            for page_label, page_text in _extract_pdf_pages(path):
                for idx, piece in enumerate(_split_text(page_text, chunk_size, chunk_overlap)):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{rel_name}::{page_label}::c{idx}",
                            filename=rel_name,
                            page_or_row=page_label,
                            text=piece,
                        )
                    )
        elif ext == ".csv":
            _, rows = _extract_csv_rows(path)
            buffer: list[str] = []
            start_row = 1
            for r_idx, row in enumerate(rows, start=1):
                line = " | ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
                buffer.append(f"Row {r_idx}: {line}")
                if sum(len(x) for x in buffer) >= chunk_size:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{rel_name}::rows_{start_row}_{r_idx}",
                            filename=rel_name,
                            page_or_row=f"rows {start_row}-{r_idx}",
                            text="\n".join(buffer),
                        )
                    )
                    buffer = []
                    start_row = r_idx + 1
            if buffer:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_name}::rows_{start_row}_{len(rows)}",
                        filename=rel_name,
                        page_or_row=f"rows {start_row}-{len(rows)}",
                        text="\n".join(buffer),
                    )
                )
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            for idx, piece in enumerate(_split_text(text, chunk_size, chunk_overlap)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{rel_name}::sec::c{idx}",
                        filename=rel_name,
                        page_or_row=f"chunk {idx + 1}",
                        text=piece,
                    )
                )

    return chunks


class HybridRetriever:
    """Offline-capable BM25 + TF-IDF hybrid retriever with Reciprocal Rank Fusion."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self._tokenized = [_tokenize(f"{c.filename} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if chunks else None
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        if chunks:
            try:
                self._tfidf_matrix = self._vectorizer.fit_transform(
                    [f"{c.filename}\n{c.text}" for c in chunks]
                )
            except ValueError:
                self._tfidf_matrix = None
        else:
            self._tfidf_matrix = None

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        mode: Literal["hybrid", "bm25", "tfidf"] = "hybrid",
    ) -> list[DocumentChunk]:
        if not self.chunks:
            return []

        n = len(self.chunks)
        k = min(top_k, n)

        bm25_scores = self._bm25.get_scores(_tokenize(query)) if self._bm25 is not None else np.zeros(n)
        bm25_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            order = np.argsort(bm25_scores)[::-1]
            for rank, idx in enumerate(order):
                bm25_ranks[int(idx)] = rank

        tfidf_ranks: dict[int, int] = {}
        if self._tfidf_matrix is not None:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
            order = np.argsort(sims)[::-1]
            for rank, idx in enumerate(order):
                tfidf_ranks[int(idx)] = rank

        if mode == "bm25":
            ranked = sorted(range(n), key=lambda i: bm25_ranks.get(i, n))
        elif mode == "tfidf":
            ranked = sorted(range(n), key=lambda i: tfidf_ranks.get(i, n))
        else:
            # Reciprocal Rank Fusion
            rrf_k = 60.0
            fused_scores = {
                i: (1.0 / (rrf_k + bm25_ranks.get(i, n))) + (1.0 / (rrf_k + tfidf_ranks.get(i, n)))
                for i in range(n)
            }
            ranked = sorted(range(n), key=lambda i: fused_scores[i], reverse=True)

        selected = [self.chunks[i] for i in ranked[:k]]

        # Multi-document diversity: ensure missing files with lexical BM25 hits are represented
        # so high-density docs (e.g., short READMEs or small CSVs) don't crowd out large tables/PDFs.
        if k >= 3:
            all_files = {c.filename for c in self.chunks}
            selected_files = {c.filename for c in selected}
            missing_files = all_files - selected_files
            replacements_made = 0
            max_replacements = max(1, k // 2)

            if missing_files:
                candidate_indices: list[int] = []
                seen_missing: set[str] = set()
                for idx in ranked[k:]:
                    fname = self.chunks[idx].filename
                    if fname in missing_files and fname not in seen_missing and bm25_scores[idx] > 0:
                        candidate_indices.append(idx)
                        seen_missing.add(fname)

                for cand_idx in candidate_indices:
                    if replacements_made >= max_replacements:
                        break
                    file_counts: dict[str, int] = {}
                    for ch in selected:
                        file_counts[ch.filename] = file_counts.get(ch.filename, 0) + 1
                    for pos in range(k - 1, -1, -1):
                        if file_counts.get(selected[pos].filename, 0) >= 2:
                            selected[pos] = self.chunks[cand_idx]
                            replacements_made += 1
                            break

            if replacements_made == 0 and len(all_files) > 1 and len({c.filename for c in selected}) == 1:
                only_file = selected[0].filename
                for idx in ranked[k:]:
                    if self.chunks[idx].filename != only_file:
                        selected[-1] = self.chunks[idx]
                        break

        return selected
