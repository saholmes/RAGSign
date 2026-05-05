"""ChromaDB vector store wrapper.

The paper (§6.1) names ChromaDB as the vector backend.  We wrap a
single Chroma collection behind an interface narrow enough to swap for
another store (FAISS, Qdrant, Lance, …) later without disturbing the
RAG orchestrator.

Two construction modes:

* **Ephemeral** (default) — in-memory client; nothing persisted.  Used
  in tests and notebooks.
* **Persistent** — backed by a directory on disk via Chroma's
  ``PersistentClient``.  Survives restarts; matches the paper's
  long-running deployment.

The wrapper takes its own :class:`rag_sign.embeddings.Embedder` and
calls it directly rather than handing it to Chroma's
``embedding_function``, so embedding caching, batching, and progress
reporting stay under our control.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from rag_sign.corpus import Chunk
from rag_sign.embeddings import Embedder


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk plus its similarity score from a query."""

    chunk: Chunk
    score: float


class ChromaVectorDB:
    """Thin Chroma wrapper specialised for RAG-Sign's chunk schema."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        collection_name: str = "rag-sign",
        persist_directory: str | Path | None = None,
    ) -> None:
        self._embedder = embedder
        if persist_directory is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=str(persist_directory))

        # ``cosine`` matches the L2-normalised embeddings produced by both
        # HashEmbedder and bge-* sentence-transformers.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Embed and upsert chunks.  Idempotent on ``chunk_id``."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._embedder.embed(texts).astype(np.float32)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {"source": c.source, "offset": c.offset} for c in chunks
            ],
        )

    def query(self, text: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the ``top_k`` chunks most similar to ``text``."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive (got {top_k})")
        vec = self._embedder.embed([text]).astype(np.float32)[0].tolist()
        result = self._collection.query(
            query_embeddings=[vec],
            n_results=min(top_k, max(self.count, 1)),
        )
        # Chroma returns lists-of-lists keyed by query.  We did one query.
        ids = result["ids"][0] if result["ids"] else []
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        dists = result["distances"][0] if result["distances"] else []

        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            chunk = Chunk(
                chunk_id=cid,
                text=doc,
                source=str(meta.get("source", "<unknown>")),
                offset=int(meta.get("offset", 0)),
            )
            # Chroma cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1].
            score = 1.0 - float(dist)
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out

    def reset(self) -> None:
        """Drop and recreate the collection (mostly for tests)."""
        name = self._collection.name
        # Chroma raises various error types per version; suppress and recreate.
        with contextlib.suppress(Exception):
            self._client.delete_collection(name=name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
