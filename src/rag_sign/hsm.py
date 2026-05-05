"""HSM-backed secret-key custody.

The paper (§6.1, §6.2) holds the long-term ``secretkey`` term of
Algorithm 1 inside an HSM (SoftHSM in the prototype, a real PKCS#11
device in deployment).  At enrolment time we generate a high-entropy
secret inside the HSM; at signing time we ask the HSM to *return* that
secret so it can be combined with ``model_hash`` and ``lsh_key`` in
the SHA3-256 derivation.

In a real production deployment the secret would never leave the HSM
boundary in the clear; the derivation would be performed on-device
(via a vendor-specific key-wrap or HMAC-based KDF mechanism) and only
the derived ECDSA signing key would be exposed.  That requires
HSM-specific firmware and is outside the scope of this reference
implementation.  The interface in this module is shaped so a
production deployment can swap the backend without touching the rest
of the pipeline.

Backends
--------

* :class:`InMemoryHSM` — pure-Python, deterministic-seedable, no
  external dependency.  Use in tests and for local development.

* :class:`SoftHSMBackend` — PKCS#11 binding via ``python-pkcs11``,
  expecting a SoftHSM v2 token.  Activated when the optional ``hsm``
  extra is installed *and* the ``RAG_SIGN_HSM_LIBRARY`` environment
  variable points at a libsofthsm2 (or other PKCS#11) shared object.

The two share the :class:`HSMBackend` :class:`typing.Protocol` so they
can be used interchangeably from :mod:`rag_sign.rag` and the bench
scripts.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

# Length of the HSM-stored secret (bytes).  256 bits — generous, matches
# the rest of the SHA3-based derivation chain.
SECRET_LEN: Final[int] = 32


@runtime_checkable
class HSMBackend(Protocol):
    """Protocol every HSM backend must satisfy.

    Two operations are supported: enrolling a fresh secret (returning
    an opaque ``handle``) and re-fetching a previously-enrolled secret
    by handle.  In production builds the second call would *not* return
    the secret in the clear — it would return an HMAC of the key
    derivation inputs, computed on-device.  See module docstring.
    """

    def enrol(self) -> bytes: ...
    """Generate a fresh secret inside the HSM, return an opaque handle."""

    def fetch(self, handle: bytes) -> bytes: ...
    """Return the secret bytes previously enrolled under ``handle``."""

    def attestation(self) -> bytes: ...
    """Return a backend-identifying string for binding into the seed."""


# ---------------------------------------------------------------------------
# In-memory backend (default — used in tests and local dev)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InMemoryHSM:
    """Pure-Python "HSM" — keeps secrets in process memory.

    Two notes on its security profile:

    * The secrets are held in :class:`bytes` objects, which CPython
      *can* leave residues of in interned-string pools.  This backend
      is **not safe for production**; it exists so the rest of the
      RAG-Sign pipeline can be unit-tested without a real PKCS#11
      device.

    * ``attestation()`` returns a fixed string identifying the backend.
      A real HSM would return a chip-specific attestation certificate
      that the verifier could trace back to a vendor root.
    """

    _secrets: dict[bytes, bytes] = field(default_factory=dict)

    def enrol(self) -> bytes:
        secret = secrets.token_bytes(SECRET_LEN)
        handle = secrets.token_bytes(16)  # 128-bit handle, unique with high prob.
        self._secrets[handle] = secret
        return handle

    def fetch(self, handle: bytes) -> bytes:
        if handle not in self._secrets:
            raise KeyError(f"unknown HSM handle: {handle.hex()[:16]}…")
        return self._secrets[handle]

    def attestation(self) -> bytes:
        return b"rag-sign:in-memory:v1"


# ---------------------------------------------------------------------------
# PKCS#11 / SoftHSM backend (optional)
# ---------------------------------------------------------------------------


# Path to a libsofthsm2 (or other PKCS#11) shared object — only consulted
# when constructing the SoftHSM backend.
_HSM_LIB_ENV: Final[str] = "RAG_SIGN_HSM_LIBRARY"
_HSM_PIN_ENV: Final[str] = "RAG_SIGN_HSM_PIN"
_HSM_TOKEN_ENV: Final[str] = "RAG_SIGN_HSM_TOKEN"


class SoftHSMBackend:  # pragma: no cover — exercised in integration tests only
    """PKCS#11 backend, configured via environment.

    Reads the path to the PKCS#11 shared object from ``$RAG_SIGN_HSM_LIBRARY``,
    the user PIN from ``$RAG_SIGN_HSM_PIN`` and the token label from
    ``$RAG_SIGN_HSM_TOKEN``.  All three must be set for the backend to
    construct.

    Stored secrets are PKCS#11 *Generic Secret* key objects with the
    handle being the Object ID (CKA_ID) attribute.  Reading them back
    requires the user PIN; the secret value is exposed in the clear to
    the host process — see module docstring on the production-vs-
    reference-implementation trade-off.
    """

    def __init__(self) -> None:
        try:
            import pkcs11  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SoftHSMBackend requires the 'hsm' extra: "
                "uv pip install 'rag-sign[hsm]'"
            ) from exc

        lib_path = os.environ.get(_HSM_LIB_ENV)
        pin = os.environ.get(_HSM_PIN_ENV)
        token_label = os.environ.get(_HSM_TOKEN_ENV)
        if not (lib_path and pin and token_label):
            raise RuntimeError(
                f"SoftHSMBackend requires {_HSM_LIB_ENV}, {_HSM_PIN_ENV}, "
                f"and {_HSM_TOKEN_ENV} environment variables"
            )

        self._lib = pkcs11.lib(lib_path)
        self._token = self._lib.get_token(token_label=token_label)
        self._pin = pin
        self._mech = pkcs11.Mechanism.GENERIC_SECRET_KEY_GEN

    def _open_session(self):  # type: ignore[no-untyped-def]
        return self._token.open(rw=True, user_pin=self._pin)

    def enrol(self) -> bytes:
        import pkcs11  # type: ignore[import-not-found]

        handle = secrets.token_bytes(16)
        with self._open_session() as session:
            session.generate_key(
                key_type=pkcs11.KeyType.GENERIC_SECRET,
                key_length=SECRET_LEN * 8,
                id=handle,
                store=True,
                template={
                    pkcs11.Attribute.SENSITIVE: False,
                    pkcs11.Attribute.EXTRACTABLE: True,
                },
            )
        return handle

    def fetch(self, handle: bytes) -> bytes:
        import pkcs11  # type: ignore[import-not-found]
        from pkcs11 import KeyType, ObjectClass

        with self._open_session() as session:
            try:
                key = session.get_key(
                    object_class=ObjectClass.SECRET_KEY,
                    key_type=KeyType.GENERIC_SECRET,
                    id=handle,
                )
            except pkcs11.NoSuchKey as exc:
                raise KeyError(f"unknown HSM handle: {handle.hex()[:16]}…") from exc
            return bytes(key[pkcs11.Attribute.VALUE])

    def attestation(self) -> bytes:
        info = self._token.slot.get_token_info()
        return (
            f"rag-sign:pkcs11:{info.label.strip()}:"
            f"{info.serial_number.decode('ascii').strip()}".encode("ascii")
        )


# ---------------------------------------------------------------------------
# Default-backend factory
# ---------------------------------------------------------------------------


def default_backend() -> HSMBackend:
    """Return an HSM backend based on environment configuration.

    Uses :class:`SoftHSMBackend` when ``RAG_SIGN_HSM_LIBRARY`` is set,
    otherwise falls back to :class:`InMemoryHSM`.  This lets the same
    code path serve unit tests, local development, and production.
    """
    if os.environ.get(_HSM_LIB_ENV):
        return SoftHSMBackend()
    return InMemoryHSM()
