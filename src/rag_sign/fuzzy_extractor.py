"""Code-offset fuzzy extractor over a binary BCH code.

Implements the construction of:

    Dodis, Reyzin, Smith.  "Fuzzy Extractors: How to Generate Strong
    Keys from Biometrics and Other Noisy Data."  SIAM J. Comput.
    38(1): 97-139 (2008).

The paper (§6.5) calls for a 5 % drift threshold over the corpus
fingerprint.  We use BCH(n=511, k=259, t=30), which corrects up to 30
bit-errors in 511 bits (≈ 5.87 %).  This sits just above the paper's
threshold while keeping the fingerprint compact (64 bytes) and giving
259 bits of independent randomness in the extracted key (compressed
through SHA3-256 to a 32-byte ``R``).

A larger code (BCH(1023, 533, t=54), ≈ 5.28 % tolerance) is available
via the public ``N`` / ``K`` / ``T`` constants if a stricter threshold
is needed; bench overheads scale roughly linearly in ``n``.

API
---
::

    R, helper = gen(w)         # enrolment
    R_prime  = rep(w', helper) # recovery from a noisy reading

``R == R'`` whenever ``HammingDistance(w, w') ≤ T`` over the first
``N`` bits of each input (any tail bits are ignored — the caller is
expected to size ``w`` appropriately, e.g. by supplying a 64-byte
MinHash fingerprint of which the first 511 bits are consumed).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import galois  # type: ignore[import-untyped]
import numpy as np

# BCH parameters chosen for ~5 % drift tolerance on a 64-byte MinHash.
_N: Final[int] = 511   # codeword length (bits)
_K: Final[int] = 259   # message length (bits) — entropy of R upper-bounded here
_T: Final[int] = 30    # error-correcting capability (bits) — corrects 5.87 %


@lru_cache(maxsize=1)
def _bch() -> galois.BCH:
    """Lazy, cached BCH constructor.

    Module import stays under a millisecond; the ~1 s BCH initialisation
    happens only on first call to :func:`gen` or :func:`rep`.
    """
    return galois.BCH(_N, _K)

# Domain-separation tag — distinguishes fuzzy-extractor outputs from
# any other SHA3-256 digest that might exist in the project.
_DOMAIN_TAG: Final[bytes] = b"RAG-SIGN/v1/fe-codeword"


class FuzzyExtractFailure(RuntimeError):
    """Raised when ``w'`` differs from ``w`` by more than ``t`` bits.

    Equivalent to BCH decoding failure.  The caller should treat this
    as a *secret-key revocation event* — the corpus has drifted past
    the agreed-upon threshold, the previous signing key is no longer
    recoverable, and a fresh enrolment is required.
    """


@dataclass(frozen=True, slots=True)
class HelperData:
    """Public helper data ``P = w ⊕ c`` (Dodis et al., §3.1).

    Knowing ``P`` reveals nothing about ``R`` because ``c`` is a
    uniformly random codeword chosen independently of ``w`` at
    enrolment time.  The bytes here are bit-packed into 64 bytes; only
    the leading 511 bits are meaningful.
    """

    bits: bytes  # exactly 64 bytes; leading 511 bits used


# ---------------------------------------------------------------------------
# Bit / byte plumbing
# ---------------------------------------------------------------------------


def _bytes_to_bits(b: bytes, length: int) -> np.ndarray:
    """Unpack ``b`` to a length-``length`` uint8 bit array (zero-pad / truncate)."""
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    if bits.size < length:
        bits = np.concatenate([bits, np.zeros(length - bits.size, dtype=np.uint8)])
    return bits[:length].astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a uint8 bit array to bytes, zero-padding to a byte boundary."""
    pad = (8 - bits.size % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _extract_R(codeword_bits: np.ndarray) -> bytes:
    """Strong extractor: SHA3-256 with domain separation."""
    return hashlib.sha3_256(_DOMAIN_TAG + _bits_to_bytes(codeword_bits)).digest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gen(w: bytes) -> tuple[bytes, HelperData]:
    """Enrolment.  Return ``(R, P)`` where ``R`` is a 32-byte stable key.

    ``w`` is the noisy reading (e.g. the MinHash digest of the corpus).
    Only the first ``N`` bits of ``w`` are consumed; supply enough
    bytes that this prefix has the entropy you need to defend.

    The returned ``HelperData`` is non-secret and is the only piece of
    state the relier needs to keep around between enrolment and
    recovery.
    """
    if len(w) * 8 < _N:
        raise ValueError(f"w must be at least {(_N + 7) // 8} bytes (got {len(w)})")

    w_bits = _bytes_to_bits(w, _N)

    # Sample a uniformly random message m, encode to codeword c.
    m_bytes = os.urandom((_K + 7) // 8)
    m_bits = _bytes_to_bits(m_bytes, _K)
    c_arr = _bch().encode(galois.GF2(m_bits))
    c_bits = np.array(c_arr, dtype=np.uint8)

    # P = w ⊕ c   (public)   ;   R = Ext(c)   (secret)
    p_bits = np.bitwise_xor(w_bits, c_bits)
    return _extract_R(c_bits), HelperData(bits=_bits_to_bytes(p_bits))


def reconstruct_w(w_prime: bytes, helper: HelperData) -> bytes:
    """Recover the original enrolment fingerprint ``w`` from ``(w', helper)``.

    Useful for drift measurement after a successful :func:`rep`: the
    orchestrator wants to compare the current corpus fingerprint
    against the one that was used at enrolment time, but never sees
    that ``w`` directly (it lives only in the live process at
    enrolment).  After recovery, the caller passes ``(w_prime, helper)``
    here and gets back the canonical 64-byte ``w``.

    The returned value is **sensitive** — together with ``helper`` it
    reveals the codeword and from there the secret ``R``.  Keep it
    inside the live process boundary; never serialise.

    Raises :class:`FuzzyExtractFailure` if the BCH decoder cannot
    correct ``w' ⊕ helper`` (drift exceeds ``T``).
    """
    if len(w_prime) * 8 < _N:
        raise ValueError(
            f"w' must be at least {(_N + 7) // 8} bytes (got {len(w_prime)})"
        )

    w_bits = _bytes_to_bits(w_prime, _N)
    p_bits = _bytes_to_bits(helper.bits, _N)

    received = galois.GF2(np.bitwise_xor(w_bits, p_bits))
    decoded_msg, n_errors = _bch().decode(received, errors=True)
    if n_errors == -1:
        raise FuzzyExtractFailure(
            f"corpus drift exceeded threshold (t={_T}); decoding failed"
        )
    c_arr = _bch().encode(decoded_msg)
    c_bits = np.array(c_arr, dtype=np.uint8)
    return _bits_to_bytes(np.bitwise_xor(p_bits, c_bits))


def rep(w_prime: bytes, helper: HelperData) -> bytes:
    """Recovery.  Re-derive ``R`` from a noisy reading and helper data.

    Raises :class:`FuzzyExtractFailure` if ``w'`` differs from the
    original ``w`` by more than ``T`` bits over the first ``N`` bits.
    """
    if len(w_prime) * 8 < _N:
        raise ValueError(
            f"w' must be at least {(_N + 7) // 8} bytes (got {len(w_prime)})"
        )

    w_bits = _bytes_to_bits(w_prime, _N)
    p_bits = _bytes_to_bits(helper.bits, _N)

    # received = w' ⊕ P = c ⊕ (w ⊕ w')   →   decode under the noise
    received = galois.GF2(np.bitwise_xor(w_bits, p_bits))
    decoded_msg, n_errors = _bch().decode(received, errors=True)
    if n_errors == -1:
        raise FuzzyExtractFailure(
            f"corpus drift exceeded threshold (t={_T}); decoding failed"
        )

    # Re-encode to recover the canonical codeword and apply the strong extractor.
    c_arr = _bch().encode(decoded_msg)
    c_bits = np.array(c_arr, dtype=np.uint8)
    return _extract_R(c_bits)


# Convenience constants exported for benchmarks / docs.
N: Final[int] = _N
K: Final[int] = _K
T: Final[int] = _T
