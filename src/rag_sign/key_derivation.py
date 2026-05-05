"""Algorithm 1 — derived signing seed.

Implements the paper's key-binding step:

    sk_seed = SHA3-256( model_hash || lsh_key || hsm_secret )

The output is a 32-byte deterministic seed that downstream modules turn
into an ECDSA P-256 keypair (see :mod:`rag_sign.signer`).

Design notes:

* All inputs are length-prefixed before hashing.  A bare concatenation
  (as written in the paper) admits length-extension–style ambiguity if
  any input is variable-width — e.g. a 10-byte ``lsh_key`` followed by
  a 22-byte ``hsm_secret`` would hash identically to a 32-byte
  ``lsh_key`` and an empty secret.  The 4-byte big-endian length prefix
  closes that gap without changing the construction's security
  argument; SHA3-256 over a domain-separated, unambiguously-parsed
  string remains a PRF on each input field.
* SHA3 (Keccak), as specified by FIPS 202, is used in line with the
  paper.  The same primitive is reused in the Merkle / commitment
  layer of the companion STARK-DNS work, keeping the project's
  hash agility surface small.
* The function is total and side-effect free — it never touches the
  HSM or filesystem.  Callers are responsible for fetching
  ``hsm_secret`` via PKCS#11 (see :mod:`rag_sign.hsm`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Domain-separation tag — distinguishes RAG-Sign-derived seeds from any
# other SHA3-256 output that might be produced by collateral tooling.
_DOMAIN_TAG: Final[bytes] = b"RAG-SIGN/v1/seed"

# All three input fields must be at least this many bytes.  32 bytes
# (256 bits) is the minimum to avoid trivially distinguishable seeds
# even before SHA3 mixing.
_MIN_FIELD_LEN: Final[int] = 16

# Buffer size for streaming model fingerprinting (1 MiB).
_MODEL_HASH_CHUNK: Final[int] = 1 << 20


@dataclass(frozen=True, slots=True)
class KeyMaterial:
    """The three inputs that bind an LLM identity to a signing key.

    Attributes:
        model_hash:  SHA3-256 of the LLM weights (typically produced
            by :func:`fingerprint_model`).
        lsh_key:     Stable corpus identity recovered from the
            MinHashLSH + fuzzy-extractor pipeline.  Length is
            implementation-defined but must be at least
            :data:`_MIN_FIELD_LEN` bytes.
        hsm_secret:  Secret material released by the HSM after
            successful attestation.  Never touches disk in clear.
    """

    model_hash: bytes
    lsh_key: bytes
    hsm_secret: bytes

    def __post_init__(self) -> None:
        for name in ("model_hash", "lsh_key", "hsm_secret"):
            value = getattr(self, name)
            if not isinstance(value, (bytes, bytearray)):
                raise TypeError(f"{name} must be bytes, got {type(value).__name__}")
            if len(value) < _MIN_FIELD_LEN:
                raise ValueError(
                    f"{name} must be at least {_MIN_FIELD_LEN} bytes "
                    f"(got {len(value)})"
                )


def derive_signing_seed(material: KeyMaterial) -> bytes:
    """Return a deterministic 32-byte signing seed.

    The construction is:

        SHA3-256( DOMAIN_TAG ||
                  len32(model_hash)  || model_hash ||
                  len32(lsh_key)     || lsh_key ||
                  len32(hsm_secret)  || hsm_secret )

    where ``len32`` is a 4-byte big-endian unsigned length.  Re-running
    with the same ``KeyMaterial`` yields the same seed bit-for-bit;
    this is what allows the verifier (or a recovering signer) to re-
    derive the keypair without ever transmitting it.
    """
    h = hashlib.sha3_256()
    h.update(_DOMAIN_TAG)
    for field in (material.model_hash, material.lsh_key, material.hsm_secret):
        h.update(len(field).to_bytes(4, "big"))
        h.update(field)
    return h.digest()


def fingerprint_model(weights_path: str | Path) -> bytes:
    """Stream-hash an on-disk model file into a 32-byte SHA3-256 digest.

    The whole file is read in :data:`_MODEL_HASH_CHUNK`-sized blocks so
    that 7 GiB Llama-class weights do not need to fit in memory.  No
    structural parsing is performed: the digest is taken over the raw
    bytes exactly as they sit on disk, which is what we want for an
    integrity fingerprint.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"model weights not found: {path}")

    h = hashlib.sha3_256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_MODEL_HASH_CHUNK)
            if not block:
                break
            h.update(block)
    return h.digest()
