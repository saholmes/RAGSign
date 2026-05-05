"""IACR ePrint corpus loader (NOT for committed test data).

Reads PDFs from ``$RAG_SIGN_IACR_DATA`` (default
``/Volumes/SAHMalm/backup Download``) and exposes a deterministic
iterator of ``(paper_id, full_text)`` tuples.

The loader caches extracted text under
``$RAG_SIGN_IACR_CACHE/<year>/<paper_id>.txt`` (default
``~/.cache/rag-sign/iacr_text``) so the slow ``pypdf`` extraction
only happens once per paper.  Cached files are *not* in the repo.

Copyright note
--------------
The PDFs themselves are downloaded copies of papers from the IACR
ePrint Archive (https://eprint.iacr.org/) — keep them outside the git
working tree.  Both the data root and the text cache are explicitly
excluded by ``.gitignore``.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pypdf  # type: ignore[import-untyped]

DEFAULT_DATA_ROOT = Path("/Volumes/SAHMalm/backup Download")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "rag-sign" / "iacr_text"

# Filename patterns we accept.  The 2013 directory uses the
# ``2013-001.pdf`` form; later years sometimes use ``999.pdf`` only.
_PAPER_ID_PATTERNS = (
    re.compile(r"^(?P<year>\d{4})-(?P<num>\d{3,4})\.pdf$"),
    re.compile(r"^(?P<num>\d{3,4})\.pdf$"),  # year inferred from parent dir
)
_YEAR_DIR_PATTERN = re.compile(r"^iacr_(?P<year>\d{4})_papers$")


@dataclass(frozen=True, slots=True)
class IACRPaper:
    paper_id: str       # "2013-001" form
    year: int
    pdf_path: Path
    cache_path: Path


def _resolve_paths() -> tuple[Path, Path]:
    data = Path(os.environ.get("RAG_SIGN_IACR_DATA", DEFAULT_DATA_ROOT))
    cache = Path(os.environ.get("RAG_SIGN_IACR_CACHE", DEFAULT_CACHE_ROOT))
    cache.mkdir(parents=True, exist_ok=True)
    return data, cache


def _enumerate_papers(
    data_root: Path,
    cache_root: Path,
    *,
    years: tuple[int, ...] | None = None,
) -> Iterator[IACRPaper]:
    for year_dir in sorted(data_root.iterdir()):
        if not year_dir.is_dir():
            continue
        m = _YEAR_DIR_PATTERN.match(year_dir.name)
        if not m:
            continue
        year = int(m.group("year"))
        if years is not None and year not in years:
            continue
        for pdf_path in sorted(year_dir.iterdir()):
            if pdf_path.suffix.lower() != ".pdf":
                continue
            paper_id = _paper_id_for(pdf_path, year)
            if paper_id is None:
                continue
            cache_path = cache_root / str(year) / f"{paper_id}.txt"
            yield IACRPaper(
                paper_id=paper_id,
                year=year,
                pdf_path=pdf_path,
                cache_path=cache_path,
            )


def _paper_id_for(pdf_path: Path, year: int) -> str | None:
    name = pdf_path.name
    for pat in _PAPER_ID_PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        try:
            return f"{m.group('year')}-{m.group('num')}"  # type: ignore[index]
        except (IndexError, KeyError):
            return f"{year}-{m.group('num')}"
    return None


def _sanitize(text: str) -> str:
    """Drop unpaired surrogates so the result is encodable as UTF-8.

    pypdf occasionally emits surrogate halves for math / pictographic
    glyphs; ``str.encode('utf-8')`` rejects those.  Round-tripping
    through ``utf-8`` with ``errors='replace'`` substitutes ``U+FFFD``
    for any unpaired surrogate while leaving everything else intact.
    """
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF.  Robust to malformed PDFs (returns empty)."""
    try:
        reader = pypdf.PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 — pypdf raises various error types
                continue
        return _sanitize("\n".join(parts))
    except Exception:  # noqa: BLE001 — corrupt / encrypted PDFs
        return ""


def load_text(paper: IACRPaper) -> str:
    """Return the (cached) text of a paper.  Returns ``""`` on failure."""
    if paper.cache_path.exists():
        return paper.cache_path.read_text(encoding="utf-8", errors="replace")

    text = _extract_pdf_text(paper.pdf_path)
    paper.cache_path.parent.mkdir(parents=True, exist_ok=True)
    paper.cache_path.write_text(text, encoding="utf-8")
    return text


def iter_papers(
    *,
    years: tuple[int, ...] | None = None,
    limit: int | None = None,
    skip_empty: bool = True,
    progress: bool = False,
) -> Iterator[tuple[str, str]]:
    """Yield ``(paper_id, full_text)`` tuples for use as a corpus.

    Parameters:
        years:       Restrict to these IACR years (e.g. ``(2013,)``).
                     ``None`` reads every year present in the data root.
        limit:       Stop after this many papers.  ``None`` means no limit.
        skip_empty:  Skip papers whose text extraction returned empty
                     (encrypted PDFs, scans, malformed files).
        progress:    Write per-100-papers progress to stderr.
    """
    data_root, cache_root = _resolve_paths()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"IACR data root not found: {data_root}.  Set "
            f"$RAG_SIGN_IACR_DATA to override."
        )

    n = 0
    for paper in _enumerate_papers(data_root, cache_root, years=years):
        text = load_text(paper)
        if skip_empty and not text.strip():
            continue
        yield paper.paper_id, text
        n += 1
        if progress and n % 100 == 0:
            print(f"  …{n} papers loaded", file=sys.stderr)
        if limit is not None and n >= limit:
            return


def count_papers(years: tuple[int, ...] | None = None) -> int:
    """Count papers under the configured data root, no extraction."""
    data_root, cache_root = _resolve_paths()
    return sum(1 for _ in _enumerate_papers(data_root, cache_root, years=years))
