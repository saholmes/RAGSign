"""Corpus loading and chunking utilities.

The RAG layer indexes *chunks* (not whole documents) because retrieval
quality degrades sharply once a single passage exceeds an embedding
model's context window.  This module keeps the chunking deterministic
so the same source corpus always produces the same chunk set, which is
a prerequisite for the corpus-fingerprint stability the LSH layer
relies on downstream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_CHUNK_CHARS: Final[int] = 1000   # rough match for ~250 tokens
DEFAULT_CHUNK_OVERLAP: Final[int] = 100  # 10 % overlap preserves cross-chunk context

# File extensions the directory loader will read by default.
_TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset({".txt", ".md", ".rst"})


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single retrieval-indexable passage.

    Attributes:
        chunk_id:    Stable, content-addressed identifier
                     (``"{source}::{offset}"``).  Used as the Chroma
                     row key so re-ingesting the same corpus is
                     idempotent.
        text:        The chunk content.
        source:      Origin identifier (filename, URL, …).
        offset:      Character offset in the source document.
    """

    chunk_id: str
    text: str
    source: str
    offset: int


def chunk_text(
    text: str,
    source: str = "<inline>",
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split ``text`` into overlapping fixed-width chunks.

    The implementation walks the string in steps of
    ``chunk_chars - overlap_chars``, which is the natural way to get a
    constant overlap.  Edge cases:

    * If the text is shorter than ``chunk_chars`` it becomes a single
      chunk, regardless of overlap.
    * Trailing whitespace at chunk boundaries is preserved — the
      embedder strips it; the fingerprint cares about the byte exact
      reading and we want both layers to see the same string.
    """
    if chunk_chars <= 0:
        raise ValueError(f"chunk_chars must be positive (got {chunk_chars})")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError(
            f"overlap_chars must be in [0, chunk_chars) "
            f"(got {overlap_chars} vs {chunk_chars})"
        )

    if len(text) <= chunk_chars:
        return [Chunk(chunk_id=f"{source}::0", text=text, source=source, offset=0)]

    step = chunk_chars - overlap_chars
    chunks: list[Chunk] = []
    offset = 0
    while offset < len(text):
        piece = text[offset : offset + chunk_chars]
        chunks.append(
            Chunk(
                chunk_id=f"{source}::{offset}",
                text=piece,
                source=source,
                offset=offset,
            )
        )
        if offset + chunk_chars >= len(text):
            break
        offset += step
    return chunks


def load_directory(
    root: str | Path,
    *,
    extensions: Iterable[str] = _TEXT_EXTENSIONS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> Iterator[Chunk]:
    """Yield :class:`Chunk` objects from every text file under ``root``.

    Iterates files in **sorted** order so that the chunk stream is
    deterministic, which the LSH fingerprint and Chroma row order both
    depend on.  Sub-directories are walked recursively.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"corpus root is not a directory: {root_path}")

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Use a stable, root-relative source id so moving the corpus
        # tree to a different mount point does not perturb fingerprints.
        rel = path.relative_to(root_path).as_posix()
        yield from chunk_text(
            text,
            source=rel,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )


def chunk_documents(
    docs: Iterable[tuple[str, str]],
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a list of (source_id, text) pairs.

    Convenience for callers that have already loaded their corpus into
    memory (e.g. from a database, an arXiv tarball, or a notebook).
    """
    out: list[Chunk] = []
    for source, text in docs:
        out.extend(
            chunk_text(
                text,
                source=source,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        )
    return out
