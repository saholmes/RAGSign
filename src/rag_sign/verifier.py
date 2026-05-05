"""Thin re-export of the verification entry point.

The actual verification logic lives in :mod:`rag_sign.signer.verify`;
this module exists so that callers who only ever verify (and never
sign) can import a smaller, intent-revealing surface.
"""

from __future__ import annotations

from rag_sign.signer import SignedMessage, verify

__all__ = ["verify", "verify_message", "SignedMessage"]


def verify_message(msg: SignedMessage) -> bool:
    """Verify a :class:`SignedMessage` produced by :class:`RagSigner`."""
    return verify(msg.payload, msg.signature, msg.public_key_pem)
