"""Regulator authority over a RAG-Sign deployment.

The deployment's long-term ECDSA keypair is its identity; this module
adds a *separate* identity belonging to an external authority (a
regulator, auditor, or compliance officer) that can:

1. **Demand on-demand key rotation** by issuing a signed
   :class:`RegulatorDirective` (action ``"ROTATE"``).  The deployment
   verifies the directive, confirms it has not already been seen
   (replay protection) and is not expired, then rotates its signing
   key via :meth:`rag_sign.rag.RagSignSystem.regenerate`.

2. **Verify corpus integrity through a secure sketch** without ever
   seeing corpus content.  At enrolment the deployment runs a
   *second* fuzzy-extractor enrolment over the same corpus
   fingerprint, hands the resulting ``R`` to the regulator
   (it becomes the regulator's audit secret) and keeps the helper
   data locally.  The regulator periodically issues a random
   challenge nonce; the deployment fingerprints its current corpus,
   re\nobreakdash-derives ``R`` (this fails if drift exceeds the BCH
   bound) and returns ``HMAC-SHA3-256(R, challenge)``.  The regulator
   verifies using their stored ``R``.  An audit failure is exactly
   the signal the regulator needs to issue a ROTATE directive.

The two primitives are orthogonal: a deployment may register an
authority for directive enforcement only, audit only, or both.

Design notes
------------

* The directive is signed with the same primitive the deployment
  uses for its own outputs (ECDSA P-256), so the implementation
  re-uses :func:`rag_sign.signer.verify`.  No new cryptographic
  assumption is introduced.

* The audit response is HMAC, not a signature, to keep the
  verification side simple and constant-time on the regulator's
  side.  The audit secret never travels over the wire after
  enrolment\nobreakdash-time delivery.

* Replay protection is a deployment-side responsibility: the
  deployment maintains an in-memory set of seen nonces.  Persisting
  it across restarts is left to the operator (the obvious choice is
  the same store that holds the EnrolmentBundle).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass
from typing import Final

from rag_sign.fuzzy_extractor import HelperData
from rag_sign.signer import verify as ecdsa_verify

# Domain-separation tags — keep regulator-bound primitives distinct from
# the deployment's own signing / extraction primitives.
_DIRECTIVE_TAG: Final[bytes] = b"RAG-SIGN/v1/regulator-directive"
_AUDIT_TAG: Final[bytes] = b"RAG-SIGN/v1/regulator-audit"

# Allowed directive actions.  Adding a new action requires updating both
# this set and :meth:`rag_sign.rag.RagSignSystem.enforce_regulator_directive`.
ROTATE: Final[str] = "ROTATE"
REVOKE: Final[str] = "REVOKE"
ALLOWED_ACTIONS: Final[frozenset[str]] = frozenset({ROTATE, REVOKE})


class InvalidRegulatorDirective(RuntimeError):
    """Raised when a directive fails verification.

    Reasons include: signature mismatch, replayed nonce, expired
    timestamp, mismatched deployment ID, unknown action.
    """


class AuditVerificationFailed(RuntimeError):
    """Raised on the regulator side when an audit response is invalid."""


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegulatorAuthority:
    """Public identity of a regulator over a deployment.

    A deployment registers one authority at enrolment time; the
    authority's public key is what the deployment uses to verify
    incoming :class:`RegulatorDirective` signatures.  The authority
    is itself published as part of the deployment's
    :class:`rag_sign.rag.EnrolmentBundle` so any third\nobreakdash-party
    verifier can independently see who can demand rotation.
    """

    public_key_pem: bytes
    name: str = "regulator"


# ---------------------------------------------------------------------------
# Signed directives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegulatorDirective:
    """An authenticated command from the regulator to the deployment.

    Fields are all string- or int-valued so the canonical
    serialisation (used for signing and verification) is just sorted
    JSON without any binary escaping.

    Attributes:
        deployment_id:  Unique identifier of the target deployment.
            Binds the directive to one specific instance.
        nonce_hex:      Random 16-byte nonce, hex-encoded.  Replay
            protection: every accepted directive's nonce is added to
            an in-memory seen-set on the deployment side.
        issued_at:      Unix timestamp when the regulator signed.
        expires_at:     Unix timestamp after which the directive is
            refused.  Typical value: ``issued_at + 300`` (5 minutes).
        action:         Member of :data:`ALLOWED_ACTIONS`.
        reason:         Human-readable, capped at 256 chars.  Logged
            on acceptance; not security-relevant.
    """

    deployment_id: str
    nonce_hex: str
    issued_at: int
    expires_at: int
    action: str
    reason: str = ""

    def canonical_bytes(self) -> bytes:
        """Deterministic byte encoding for signing / verification."""
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return _DIRECTIVE_TAG + b"\n" + body.encode("utf-8")

    @property
    def nonce(self) -> bytes:
        return bytes.fromhex(self.nonce_hex)


def make_directive(
    *,
    deployment_id: str,
    action: str,
    reason: str = "",
    ttl_seconds: int = 300,
    now: int | None = None,
) -> RegulatorDirective:
    """Construct an unsigned directive.

    The regulator signs ``directive.canonical_bytes()`` with their
    ECDSA private key (e.g. via :class:`rag_sign.signer.RagSigner`)
    and ships ``(directive, signature)`` to the deployment.
    """
    if action not in ALLOWED_ACTIONS:
        raise ValueError(
            f"action {action!r} not in allowed set {sorted(ALLOWED_ACTIONS)}"
        )
    if len(reason) > 256:
        raise ValueError(f"reason too long ({len(reason)} > 256 chars)")
    issued = int(now if now is not None else time.time())
    return RegulatorDirective(
        deployment_id=deployment_id,
        nonce_hex=_random_nonce_hex(),
        issued_at=issued,
        expires_at=issued + ttl_seconds,
        action=action,
        reason=reason,
    )


def verify_directive(
    directive: RegulatorDirective,
    signature: bytes,
    authority: RegulatorAuthority,
) -> bool:
    """Verify the regulator's signature over ``directive.canonical_bytes()``.

    Returns ``True`` on success, ``False`` on a verifiable mismatch.
    Other errors (malformed key, malformed signature) propagate.
    """
    return ecdsa_verify(
        directive.canonical_bytes(), signature, authority.public_key_pem
    )


# ---------------------------------------------------------------------------
# Audit (secure-sketch based)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegulatorAuditToken:
    """What the regulator stores to audit one deployment.

    Held secret by the regulator.  Possession of ``audit_secret``
    lets the regulator verify that the deployment's current corpus
    fingerprint reproduces the same ``R`` it produced at
    enrolment-time — which means the corpus has not drifted past the
    BCH bound.  ``audit_secret`` is *not* the deployment's signing
    key and cannot be used to forge messages on the deployment's
    behalf.

    Attributes:
        deployment_id:  Identifies which deployment this token
            belongs to.  A regulator that audits multiple
            deployments holds one token per deployment.
        audit_secret:   The 32-byte ``R`` derived from a separate
            (independent of the signing extractor) call to
            :func:`rag_sign.fuzzy_extractor.gen` over the same
            corpus fingerprint.
        issued_at:      Unix timestamp the deployment created the
            token.  Logged for audit-trail purposes; not used in
            challenge verification.
    """

    deployment_id: str
    audit_secret: bytes
    issued_at: int


@dataclass(frozen=True, slots=True)
class _AuditState:
    """Deployment-side state for one regulator audit relationship.

    The deployment keeps the helper data so it can re-derive the
    same ``R`` when challenged.  ``audit_secret`` itself is *not*
    stored on the deployment side after issue\nobreakdash-time —
    the regulator holds it.
    """

    deployment_id: str
    helper: HelperData


def _random_nonce_hex(n_bytes: int = 16) -> str:
    """16-byte cryptographically-strong random nonce, hex-encoded."""
    import secrets

    return secrets.token_hex(n_bytes)


def make_audit_challenge() -> bytes:
    """Regulator-side helper: produce a fresh 32-byte challenge nonce."""
    import secrets

    return secrets.token_bytes(32)


def compute_audit_response(audit_secret: bytes, challenge: bytes) -> bytes:
    """Deployment-side: HMAC-SHA3-256(audit_secret, tag ‖ challenge).

    The deployment will normally have the live ``audit_secret`` in
    memory only briefly — it is re-derived on demand from the live
    corpus fingerprint plus the stored helper, see
    :meth:`rag_sign.rag.RagSignSystem.respond_to_audit_challenge`.
    """
    return hmac.new(
        audit_secret, _AUDIT_TAG + challenge, hashlib.sha3_256
    ).digest()


def verify_audit_response(
    token: RegulatorAuditToken, challenge: bytes, response: bytes
) -> bool:
    """Regulator-side: constant-time check of the deployment's HMAC."""
    expected = compute_audit_response(token.audit_secret, challenge)
    return hmac.compare_digest(expected, response)
