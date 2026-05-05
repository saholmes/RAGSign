"""Parallel PDF→text cache warmer for the IACR experiment.

Reads PDFs under ``$RAG_SIGN_IACR_DATA`` and writes extracted text
into ``$RAG_SIGN_IACR_CACHE`` using ``multiprocessing.Pool``.  Skips
papers whose cache file already exists.  Reports progress every 200
papers; safe to interrupt and re-run.

Usage:

    .venv/bin/python -m scripts.iacr_warm_cache --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iacr_loader import (  # noqa: E402
    IACRPaper,
    _enumerate_papers,
    _extract_pdf_text,
    _resolve_paths,
)


def _extract_one(paper: IACRPaper) -> tuple[str, int]:
    """Return ``(paper_id, text_byte_count)``; 0 on failure.

    Catches *any* worker exception so the pool keeps running — bad
    PDFs are a normal occurrence in a 12 K-paper corpus and we don't
    want one of them to abort the whole sweep.
    """
    try:
        if paper.cache_path.exists():
            return paper.paper_id, paper.cache_path.stat().st_size
        text = _extract_pdf_text(paper.pdf_path)
        paper.cache_path.parent.mkdir(parents=True, exist_ok=True)
        paper.cache_path.write_text(text, encoding="utf-8", errors="replace")
        return paper.paper_id, len(text.encode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 — defence against any pypdf / IO surprise
        return paper.paper_id, 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=None,
                   help="restrict to these years (default: all)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel processes (default 8)")
    args = p.parse_args()

    data_root, cache_root = _resolve_paths()
    if not data_root.is_dir():
        print(f"FATAL: data root not found: {data_root}", file=sys.stderr)
        return 2

    years = tuple(args.years) if args.years else None
    papers = list(_enumerate_papers(data_root, cache_root, years=years))
    todo = [p for p in papers if not p.cache_path.exists()]

    print(f"Total papers: {len(papers)}   already cached: {len(papers) - len(todo)}   "
          f"to extract: {len(todo)}   workers: {args.workers}")

    if not todo:
        print("Nothing to do — cache fully warm.")
        return 0

    t0 = time.perf_counter()
    n_done = 0
    n_empty = 0
    total_bytes = 0
    # ``spawn`` is the safe default on macOS for libraries that touch native heap.
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for _paper_id, n_bytes in pool.imap_unordered(_extract_one, todo, chunksize=8):
            n_done += 1
            total_bytes += n_bytes
            if n_bytes == 0:
                n_empty += 1
            if n_done % 200 == 0 or n_done == len(todo):
                elapsed = time.perf_counter() - t0
                rate = n_done / elapsed
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(
                    f"  {n_done}/{len(todo)} "
                    f"({n_done * 100 / len(todo):.1f}%)   "
                    f"{rate:.1f} papers/s   eta {eta:.0f}s   "
                    f"empty: {n_empty}"
                )

    dt = time.perf_counter() - t0
    print(f"Done in {dt:.1f}s; extracted {total_bytes/1e6:.1f} MB of text "
          f"({n_empty} empty / failed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
