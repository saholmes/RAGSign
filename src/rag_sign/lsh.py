"""MinHash-based corpus fingerprint.

The paper (§6.5) uses MinHash-LSH over the RAG corpus to produce a
stable identifier that survives small ingestion / edit drift.  This
module turns a corpus into a deterministic, drift-tolerant byte string
suitable as the ``w`` input to :mod:`rag_sign.fuzzy_extractor`.

Construction
------------

1. Each document is shingled into ``k``-grams (default ``k=5`` for
   prose; configurable for non-prose corpora).
2. A :class:`datasketch.MinHash` sketch with ``num_perm = 512``
   permutations is computed for every document.
3. The corpus-level sketch is the **union** (``MinHash.merge``) of all
   per-document sketches.  Set-union of MinHashes equals MinHash of
   the union, which gives the drift property: removing or adding a
   small fraction of documents perturbs only a small fraction of
   permutation slots.
4. Each of the 256 slots is reduced to **one bit** via
   ``SHA3-256(domain ‖ slot) & 1``.  The 256 bits are packed into 32
   bytes, which is the ``w`` consumed by the fuzzy extractor.

Why 1 bit per slot?  Because the BCH code in the fuzzy extractor
corrects ``t = 18`` bit-errors over ``N = 255`` bits.  If we used a
byte (or worse, a SHA3 of the whole sketch) per slot, a single slot
perturbation would on average flip ``~4`` bits, multiplying
slot-drift by 4× before it hits the BCH bound — a 5 % corpus drift
would exceed ``t`` and break recovery.  With one bit per slot, slot
drift maps 1:1 (worst-case) onto bit drift, so the paper's 5 %
threshold sits comfortably inside the 7 % ECC margin.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
from datasketch import MinHash

# 512 permutations → 512-bit fingerprint → 64-byte ``w``.  This matches
# the fuzzy extractor's ``N = 511`` plus one bit of slack (the leading
# 511 bits are consumed; the last bit is unused).  Doubling the slot
# count vs. the BCH-bound minimum also halves the per-slot variance so
# real-world corpus drift maps more reliably onto the 5.87 % ECC margin.
DEFAULT_NUM_PERM: Final[int] = 512

# k = 5 is the canonical shingle length from Broder 1997 and is the
# default in datasketch's tutorial.  Override for non-prose corpora.
DEFAULT_SHINGLE_LEN: Final[int] = 5

_DOMAIN_TAG: Final[bytes] = b"RAG-SIGN/v1/lsh-fingerprint"


def shingle(text: str, k: int = DEFAULT_SHINGLE_LEN) -> set[str]:
    """Return the set of overlapping ``k``-character shingles of ``text``.

    Whitespace is collapsed to single spaces before shingling so that
    cosmetic re-formatting (CRLF vs LF, trailing whitespace,
    multi-space runs) does not perturb the fingerprint.
    """
    if k <= 0:
        raise ValueError(f"shingle length must be positive (got {k})")
    normalised = " ".join(text.split())
    if len(normalised) < k:
        return {normalised}
    return {normalised[i : i + k] for i in range(len(normalised) - k + 1)}


def minhash_document(
    text: str,
    *,
    k: int = DEFAULT_SHINGLE_LEN,
    num_perm: int = DEFAULT_NUM_PERM,
) -> MinHash:
    """MinHash a single document at shingle granularity ``k``."""
    mh = MinHash(num_perm=num_perm)
    for s in shingle(text, k=k):
        mh.update(s.encode("utf-8"))
    return mh


def minhash_corpus(
    documents: Iterable[str],
    *,
    k: int = DEFAULT_SHINGLE_LEN,
    num_perm: int = DEFAULT_NUM_PERM,
) -> MinHash:
    """MinHash a corpus by sketching the *union* of all documents.

    Order-independent: shuffling the corpus produces the same sketch.
    Adding / removing a small number of documents perturbs the sketch
    in proportion to the symmetric-difference size — this is what
    gives the drift tolerance the fuzzy extractor downstream relies on.
    """
    accum = MinHash(num_perm=num_perm)
    for doc in documents:
        accum.merge(minhash_document(doc, k=k, num_perm=num_perm))
    return accum


def fingerprint_from_minhash(mh: MinHash) -> bytes:
    """Fold a :class:`MinHash` sketch into a deterministic byte fingerprint.

    Returns ``num_perm / 8`` bytes (32 by default), one bit per
    permutation slot.  Each output bit is the low bit of
    ``SHA3-256(domain ‖ slot)``.  This is proximity-preserving on the
    *slot* level: a single slot perturbation flips at most one
    fingerprint bit, so a 5 % slot drift maps to ≤ 5 % bit drift.
    """
    slots = mh.hashvalues  # uint32 ndarray, length num_perm
    bits = np.empty(slots.size, dtype=np.uint8)
    for i, h in enumerate(slots):
        bits[i] = hashlib.sha3_256(_DOMAIN_TAG + int(h).to_bytes(4, "big")).digest()[0] & 1
    # pack bits MSB-first to match np.unpackbits's convention used elsewhere
    return np.packbits(bits).tobytes()


def fingerprint_corpus(
    documents: Sequence[str],
    *,
    k: int = DEFAULT_SHINGLE_LEN,
    num_perm: int = DEFAULT_NUM_PERM,
) -> bytes:
    """Convenience wrapper: corpus → fingerprint, end-to-end."""
    mh = minhash_corpus(documents, k=k, num_perm=num_perm)
    return fingerprint_from_minhash(mh)


def hamming_distance(a: bytes, b: bytes) -> int:
    """Bit-level Hamming distance between two equal-length byte strings."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return int(np.unpackbits(np.bitwise_xor(aa, bb)).sum())
