"""ECDSA P-256 sign / verify, seeded by the RAG-Sign derivation.

The 32-byte seed produced by :func:`rag_sign.key_derivation.derive_signing_seed`
is interpreted as an integer and reduced modulo the curve order *n* to
yield the private scalar *d*.  We use rejection sampling to avoid the
zero scalar (negligible probability for a uniform 256-bit input, but we
treat it correctly anyway).

Signatures are DER-encoded over the raw output bytes — verification can
therefore be implemented by any standards-compliant library, including
the one that ships with most browsers and TLS stacks.

Per the paper (§6.1, §6.3) we sign every LLM output token-stream after
generation completes.  The verifier in :mod:`rag_sign.verifier` is a
thin wrapper around the public-key half of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

# secp256r1 / NIST P-256 group order n (FIPS 186-5, App. D.1.2.3).
_P256_N: Final[int] = (
    0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
)

_HASH: Final[ec.EllipticCurveSignatureAlgorithm] = ec.ECDSA(hashes.SHA256())
_CURVE: Final[ec.EllipticCurve] = ec.SECP256R1()


def _seed_to_scalar(seed: bytes) -> int:
    """Map a 32-byte seed to a non-zero P-256 scalar via rejection sampling.

    For seeds drawn from any distribution close to uniform over
    {0,1}^256 (which SHA3-256 is, for the purposes of an honest
    derivation), the loop body runs once with probability
    ``1 - (2^256 - n) / 2^256 ≈ 1 - 2^-128``.
    """
    if len(seed) != 32:
        raise ValueError(f"seed must be exactly 32 bytes, got {len(seed)}")

    counter = 0
    while True:
        # Re-hash with a counter on the (vanishingly unlikely) failure
        # path so we never recurse into the zero scalar.
        if counter == 0:
            candidate = int.from_bytes(seed, "big")
        else:
            from hashlib import sha3_256

            candidate = int.from_bytes(
                sha3_256(seed + counter.to_bytes(4, "big")).digest(),
                "big",
            )
        d = candidate % _P256_N
        if d != 0:
            return d
        counter += 1


@dataclass(frozen=True, slots=True)
class SignedMessage:
    """An LLM output bundled with its detached ECDSA signature."""

    payload: bytes        # the bytes that were signed (LLM output as UTF-8)
    signature: bytes      # DER-encoded ECDSA signature
    public_key_pem: bytes # SubjectPublicKeyInfo PEM for offline verification


class RagSigner:
    """Wrapper that holds a P-256 private key derived from a seed.

    The constructor is deliberately the only place the private scalar
    materialises in plaintext form; once the
    :class:`ec.EllipticCurvePrivateKey` is built, the seed bytes are
    discarded from this object's attributes (they remain on the
    caller's side, where they were produced).
    """

    __slots__ = ("_priv",)

    def __init__(self, seed: bytes) -> None:
        d = _seed_to_scalar(seed)
        self._priv = ec.derive_private_key(d, _CURVE)

    @property
    def public_key_pem(self) -> bytes:
        return self._priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    @property
    def public_key_der(self) -> bytes:
        return self._priv.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign(self, payload: bytes) -> SignedMessage:
        sig = self._priv.sign(payload, _HASH)
        return SignedMessage(
            payload=payload,
            signature=sig,
            public_key_pem=self.public_key_pem,
        )


def verify(payload: bytes, signature: bytes, public_key_pem: bytes) -> bool:
    """Verify a detached ECDSA-P256 signature over ``payload``.

    Returns ``True`` on a valid signature, ``False`` on an invalid one.
    Other errors (malformed key, malformed signature) are re-raised.
    """
    pub = serialization.load_pem_public_key(public_key_pem)
    if not isinstance(pub, ec.EllipticCurvePublicKey):
        raise TypeError("public_key_pem is not an EC public key")
    try:
        pub.verify(signature, payload, _HASH)
        return True
    except InvalidSignature:
        return False
