"""Bench: signing overhead as a fraction of LLM generation time.

Paper §6.6 reports an end-to-end overhead of ≈ 0.05 % — i.e. signing
is essentially free relative to the cost of generating the answer.
We measure the same ratio under :class:`EchoLLM` (instant generation
— a worst-case for the overhead ratio, since real LLM latency is
several orders of magnitude higher).  The numbers reported here are
therefore a strict upper bound on the overhead a real deployment
would see.

Usage:

    .venv/bin/python -m benchmarks.bench_overhead
"""

from __future__ import annotations

import os
import time

from benchmarks._common import (
    BenchResult,
    _git_commit,
    _host_info,
    print_result,
    write_result,
)
from rag_sign.corpus import chunk_documents
from rag_sign.embeddings import HashEmbedder
from rag_sign.hsm import InMemoryHSM
from rag_sign.llm import EchoLLM
from rag_sign.rag import RagSignSystem
from rag_sign.vector_db import ChromaVectorDB

CORPUS_SIZE = 200
DOCS = [
    (f"doc-{i}.txt", f"Document {i} — synthetic abstract on topic {i}.")
    for i in range(CORPUS_SIZE)
]
QUESTIONS = [
    "What does document 7 say?",
    "Summarise document 42.",
    "Compare topics 1 and 2.",
] * 100


def main() -> None:
    sys = RagSignSystem(
        vector_db=ChromaVectorDB(
            embedder=HashEmbedder(dim=128),
            collection_name="bench-overhead",
        ),
        llm=EchoLLM(),
        hsm=InMemoryHSM(),
        top_k=3,
    )
    sys.enrol(chunk_documents(DOCS), model_hash=os.urandom(32))

    # Total wall time (retrieval + generate + sign).
    t0 = time.perf_counter_ns()
    for q in QUESTIONS:
        sys.query(q)
    t_total_ns = time.perf_counter_ns() - t0

    # Sign-only fraction: replay the same answers but only sign them.
    state = sys._state  # noqa: SLF001 — bench needs the live signer
    assert state is not None
    answers = [b"sample answer of approximately the right size " * 4] * len(QUESTIONS)
    t0 = time.perf_counter_ns()
    for ans in answers:
        state.signer.sign(ans)
    t_sign_ns = time.perf_counter_ns() - t0

    overhead = t_sign_ns / t_total_ns
    print(  # noqa: T201
        f"sign / total = {overhead*100:.3f}%   "
        f"(total {t_total_ns/1e6:.1f}ms, sign {t_sign_ns/1e6:.1f}ms, "
        f"n={len(QUESTIONS)})"
    )

    # Persist as a single-sample BenchResult for diff-friendliness.
    result = BenchResult(
        name="overhead_pct",
        n=len(QUESTIONS),
        ms_p50=overhead * 100.0,
        ms_p95=overhead * 100.0,
        ms_mean=overhead * 100.0,
        host=_host_info(),
        git_commit=_git_commit(),
        timestamp="",
    )
    print_result(result)
    write_result(result)


if __name__ == "__main__":
    main()
