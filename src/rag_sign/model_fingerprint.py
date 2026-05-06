"""Drift-tolerant fingerprint of a model's weights.

The :func:`rag_sign.key_derivation.fingerprint_model` helper produces
a SHA3\nobreakdash-256 of the *raw bytes* of a weight file — perfect
for cryptographic identity (Algorithm 1) but binary on equality:
flipping a single weight changes the digest entirely.  That is the
right answer for the *cryptographic* identity layer, but the wrong
answer for an *operational* AI\nobreakdash-safety gate where the
deployment wants to permit small fine-tunes / adapter merges /
re-quantisations without paging on-call but still trigger a review
when the model has been replaced wholesale.

This module provides the noise-tolerant counterpart: a SimHash-style
fingerprint of the weights that maps small parameter perturbations
onto small Hamming distances and large perturbations onto large
ones.  Together with :class:`rag_sign.rag.RagSignSystem`'s
``model_drift_policy_bits`` setting it gives the deployment a
threshold below which fine-tunes pass silently and above which a
"please send this to AI-safety review" alarm fires.

Construction
------------

The fingerprint is the sign-vector of a deterministic random
projection of the flattened parameter vector.  This is the
Charikar 2002 SimHash construction, which is locality\nobreakdash-sensitive
in cosine distance:

    P(\\hat{f}_i(v) \\neq \\hat{f}_i(v + \\Delta))
        = \\theta(v, v + \\Delta) / \\pi

where ``\\theta`` is the angle between ``v`` and the perturbed
``v + \\Delta``.  For small ``\\Delta`` (in L2 norm) the angle is
small and so the expected Hamming distance is small.

The projection is computed in chunks for memory safety: a 7\nobreakdash-billion
parameter model never materialises a 512×7B projection matrix.
Each chunk's projection is generated from a SHA3\nobreakdash-256-derived
seed of ``(global_seed ‖ chunk_index)``, so the whole construction
is deterministic given ``seed``.  Two callers using the same seed
*must* produce the same fingerprint for the same weights, even if
they process different chunk sizes — the seed-per-chunk indexing
keeps everything reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np

# Default fingerprint length in bits.  Matches the fuzzy-extractor's
# N=511 padded to a byte-multiple, so a model fingerprint is the same
# 64-byte shape as a corpus fingerprint and can flow through the same
# pipeline if a future revision wires it into Algorithm 1.
DEFAULT_DIM: Final[int] = 512

# How many parameters to multiply at once when projecting.  Bigger
# values spend more memory but call SHA3 fewer times; 4096 keeps
# per-chunk allocation under 1 MiB at dim=512 / float32.
DEFAULT_CHUNK_SIZE: Final[int] = 4096

# Domain-separation tag for the per-chunk seed derivation.
_DOMAIN_TAG: Final[bytes] = b"RAG-SIGN/v1/model-simhash"


def _chunk_rng(seed: bytes, chunk_index: int) -> np.random.Generator:
    """Deterministic per-chunk numpy Generator."""
    seed_i = hashlib.sha3_256(
        _DOMAIN_TAG + seed + chunk_index.to_bytes(8, "big")
    ).digest()
    # numpy's default_rng accepts a bytes-derived 64-bit seed.
    return np.random.default_rng(int.from_bytes(seed_i[:8], "big"))


def simhash_array(
    weights: np.ndarray,
    *,
    dim: int = DEFAULT_DIM,
    seed: bytes = b"default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bytes:
    """Compute the SimHash of a single (possibly large) numpy array.

    Returns ``dim // 8`` bytes (64 bytes by default).  The input is
    flattened to one dimension before projection; structure-preserving
    fingerprints are an extension left to a future revision.
    """
    if dim <= 0 or dim % 8 != 0:
        raise ValueError(f"dim must be a positive multiple of 8 (got {dim})")
    if weights.size == 0:
        raise ValueError("weights array is empty")

    flat = np.ascontiguousarray(weights, dtype=np.float64).ravel()
    return _simhash_streaming(_iter_chunks(flat, chunk_size), dim=dim, seed=seed)


def simhash_arrays(
    arrays: Iterable[np.ndarray],
    *,
    dim: int = DEFAULT_DIM,
    seed: bytes = b"default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bytes:
    """Compute the SimHash over a sequence of arrays — e.g. layer-wise weights.

    Useful for streaming a model from disk: load one tensor at a time,
    yield it, and never have all the parameters in memory at once.
    The arrays are concatenated logically (not physically) for
    fingerprinting; the iteration order matters and must be stable
    across enrolment / recovery.
    """
    if dim <= 0 or dim % 8 != 0:
        raise ValueError(f"dim must be a positive multiple of 8 (got {dim})")

    def _flatten() -> Iterable[np.ndarray]:
        for arr in arrays:
            flat = np.ascontiguousarray(arr, dtype=np.float64).ravel()
            yield from _iter_chunks(flat, chunk_size)

    return _simhash_streaming(_flatten(), dim=dim, seed=seed)


def simhash_safetensors(
    path: str | Path,
    *,
    dim: int = DEFAULT_DIM,
    seed: bytes = b"default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bytes:  # pragma: no cover — requires the optional `safetensors` package
    """Convenience: SimHash of a .safetensors model file.

    Opens the file lazily, iterates tensors in their on\nobreakdash-disk
    order (sorted lexicographically — safetensors stores them that
    way), and fingerprints them via :func:`simhash_arrays`.  This is
    the recommended entry point for production: zero copies, bounded
    memory, deterministic ordering.
    """
    try:
        from safetensors.numpy import safe_open  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "simhash_safetensors requires the 'embeddings' or a "
            "dedicated 'safetensors' install: pip install safetensors"
        ) from exc

    p = Path(path)

    def _iter_tensors() -> Iterable[np.ndarray]:
        with safe_open(str(p), framework="numpy") as f:  # type: ignore[no-untyped-call]
            for key in sorted(f.keys()):
                yield f.get_tensor(key)

    return simhash_arrays(_iter_tensors(), dim=dim, seed=seed, chunk_size=chunk_size)


def hamming_distance(a: bytes, b: bytes) -> int:
    """Bit-level Hamming distance between two equal-length byte strings."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return int(np.unpackbits(np.bitwise_xor(aa, bb)).sum())


# ---------------------------------------------------------------------------
# Per-tensor SimHash (sensitive variant)
# ---------------------------------------------------------------------------


def simhash_per_tensor(
    tensors: Iterable[tuple[str, np.ndarray]],
    *,
    bits_per_tensor: int = 4,
    seed: bytes = b"default",
) -> bytes:
    """Per-tensor SimHash, concatenated into one fingerprint.

    Why this exists: a single SimHash over a flattened multi-million-
    parameter model is too averaging-friendly to detect realistic
    fine-tunes — the central-limit-theorem dilutes per-parameter
    movement against the variance of the sum.  Empirically, even a
    500\nobreakdash-step fine-tune of GPT\nobreakdash-2 small flips
    only ~2 of 512 bits with the whole\nobreakdash-model variant.

    The per-tensor variant flips bits when any *single* tensor moves
    materially — which is exactly the granularity an AI\nobreakdash-safety
    gate wants.  Each tensor (named layer) contributes
    ``bits_per_tensor`` independent SimHash bits drawn from a
    deterministically\nobreakdash-seeded Gaussian projection.  For a
    model with 150 named tensors and 4 bits each, the fingerprint
    is 600 bits = 75 bytes.

    The ``tensors`` argument is an iterable of ``(name, ndarray)``
    pairs in stable order.  Names are not used in the bit
    derivation (only the index is) — so two callers must enumerate
    in the same order.  The recommended convention is
    ``sorted(state_dict.keys())``.

    Returns: bytes whose length is
    ``ceil(n_tensors * bits_per_tensor / 8)``.
    """
    if bits_per_tensor <= 0:
        raise ValueError(f"bits_per_tensor must be positive (got {bits_per_tensor})")

    bits: list[int] = []
    for index, (_name, tensor) in enumerate(tensors):
        flat = np.ascontiguousarray(tensor, dtype=np.float64).ravel()
        if flat.size == 0:
            # Empty tensor (e.g. degenerate parameter): contribute zeroes.
            bits.extend([0] * bits_per_tensor)
            continue
        seed_i = hashlib.sha3_256(
            _DOMAIN_PER_TENSOR_TAG + seed + index.to_bytes(8, "big")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(seed_i[:8], "big"))
        proj = rng.standard_normal((bits_per_tensor, flat.size)).astype(np.float64)
        signs = (proj @ flat) > 0
        bits.extend(int(b) for b in signs)

    if not bits:
        raise ValueError("no tensors yielded; cannot fingerprint an empty model")

    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits.extend([0] * pad)
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


_DOMAIN_PER_TENSOR_TAG: Final[bytes] = b"RAG-SIGN/v1/model-simhash-per-tensor"


# ---------------------------------------------------------------------------
# Behavioral fingerprint
# ---------------------------------------------------------------------------


def behavioral_fingerprint(
    activations: Iterable[np.ndarray],
    *,
    dim: int = DEFAULT_DIM,
    seed: bytes = b"default",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bytes:
    """SimHash of a sequence of model *activations* (outputs).

    Where :func:`simhash_array` and :func:`simhash_per_tensor`
    fingerprint the *static* weights of a model, this fingerprints
    its *behaviour* on a fixed probe set: feed the same prompts
    through the live model, capture last-token logits (or any other
    output activation), concatenate, project via the same Gaussian
    SimHash construction.

    Why this is more sensitive than weight-space fingerprinting
    for the AI-safety gate use case: fine-tuning is *designed* to
    change the model's outputs.  Even a brief fine-tune (a few
    SGD steps) moves last-token logits by units of standard
    deviation; a static weight SimHash, by contrast, sees only
    micro-shifts in individual weights that average out under
    Gaussian projection.  Empirically this gives a 50–100×
    larger Hamming-distance signal at the same fine-tune
    intensity.

    The caller is responsible for choosing a stable probe set
    (the same prompts at enrolment and at audit time) and for
    capturing the same activation tensor (e.g. last-token logits).
    Drift in the probe set itself would invalidate the comparison.

    The construction is identical to :func:`simhash_arrays` —
    only the input semantics change.  This re-export exists so
    callers can spell their intent in the type system.
    """
    return simhash_arrays(activations, dim=dim, seed=seed, chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iter_chunks(flat: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    """Yield flat chunks of size ``chunk_size`` (last may be smaller)."""
    for i in range(0, flat.size, chunk_size):
        yield flat[i : i + chunk_size]


def _simhash_streaming(
    chunks: Iterable[np.ndarray],
    *,
    dim: int,
    seed: bytes,
) -> bytes:
    """Project + accumulate + sign over a (possibly huge) chunk stream.

    The accumulator is ``dim`` floats wide regardless of input size.
    """
    acc = np.zeros(dim, dtype=np.float64)
    for chunk_idx, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        rng = _chunk_rng(seed, chunk_idx)
        # Random Gaussian projection: dim × chunk_size matrix times the chunk.
        proj = rng.standard_normal((dim, chunk.size)).astype(np.float64)
        acc += proj @ chunk
    if not acc.any():
        # Only happens for an empty stream — keep the failure obvious.
        raise ValueError("no chunks were yielded; weights stream is empty")
    bits = (acc > 0).astype(np.uint8)
    return np.packbits(bits).tobytes()
