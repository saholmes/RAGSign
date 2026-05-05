"""Pluggable embedder interface.

The paper (§6.1) uses GPT-4 embeddings.  For an open-source
reproduction we expose two backends:

* :class:`SentenceTransformersEmbedder` — wraps any HuggingFace
  sentence-transformer (default ``BAAI/bge-large-en-v1.5``, 1024-dim,
  competitive with OpenAI's text-embedding-3 models on MTEB).  Lazy-
  imported so the core package does not pull in 1 GB of torch wheels.

* :class:`HashEmbedder` — pure-Python deterministic embedder with no
  ML dependency.  Each dimension is a SHA3-derived bit; useful for
  unit-testing the RAG plumbing without provisioning an embedding
  model.  *Not* a real embedding — it has no semantic meaning.

The :class:`Embedder` :class:`typing.Protocol` is the contract the
:mod:`rag_sign.vector_db` and :mod:`rag_sign.rag` layers consume.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

import numpy as np

# Default dimensionality for the deterministic test embedder.
DEFAULT_HASH_DIM: Final[int] = 384


@runtime_checkable
class Embedder(Protocol):
    """Embeds text into a fixed-dimensional float vector."""

    @property
    def dim(self) -> int: ...
    """Vector dimensionality this embedder produces."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
    """Return an ``(n, dim)`` float32 array of L2-normalised embeddings."""


# ---------------------------------------------------------------------------
# Hash embedder (deterministic, no ML dependency — for tests / dev)
# ---------------------------------------------------------------------------


class HashEmbedder:
    """SHA3-derived deterministic embeddings.

    Each dimension ``j`` of ``embed(text)`` is the ``j``-th low bit of
    SHA3-256(text || j32) recoded to ±1, then the whole vector is
    L2-normalised.  Documents containing the same text always embed
    identically, but the resulting "embedding space" carries no
    semantic structure — synonyms are roughly orthogonal.

    Use this for plumbing tests where you want the RAG / vector-DB
    layers to operate on real-shaped tensors without paying the cost
    of a real embedding model.
    """

    def __init__(self, dim: int = DEFAULT_HASH_DIM) -> None:
        if dim <= 0 or dim % 8 != 0:
            raise ValueError(f"dim must be a positive multiple of 8 (got {dim})")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._embed_one(t)
        return out

    def _embed_one(self, text: str) -> np.ndarray:
        # Generate enough bytes for `dim` bits, one bit per dimension.
        n_bytes = self._dim // 8
        # Stretch SHA3-256 by repeated hashing with a counter — deterministic.
        material = bytearray()
        counter = 0
        encoded = text.encode("utf-8")
        while len(material) < n_bytes:
            material.extend(
                hashlib.sha3_256(encoded + counter.to_bytes(4, "big")).digest()
            )
            counter += 1
        bits = np.unpackbits(np.frombuffer(bytes(material[:n_bytes]), dtype=np.uint8))
        v = bits.astype(np.float32) * 2.0 - 1.0  # {0,1} → {-1,+1}
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v


# ---------------------------------------------------------------------------
# sentence-transformers backend (real embeddings)
# ---------------------------------------------------------------------------


class SentenceTransformersEmbedder:  # pragma: no cover — heavyweight, integration-tested
    """HuggingFace sentence-transformers embeddings.

    Default model is ``BAAI/bge-large-en-v1.5`` (1024-dim).  Other
    options worth considering:

    * ``BAAI/bge-small-en-v1.5`` (384-dim) — 30× faster, slightly less
      accurate.  Same dimensionality as OpenAI ``text-embedding-3-small``.
    * ``intfloat/e5-large-v2`` (1024-dim) — strong multilingual variant.

    Lazy-imports the ``sentence_transformers`` package so the core
    install does not pull in PyTorch.  Install with the ``embeddings``
    extra: ``uv pip install 'rag-sign[embeddings]'``.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformersEmbedder requires the 'embeddings' extra: "
                "uv pip install 'rag-sign[embeddings]'"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return int(self._dim)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)
