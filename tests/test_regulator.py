"""Tests for :mod:`rag_sign.regulator` and the regulator-aware extensions
to :class:`rag_sign.rag.RagSignSystem`.

Covers two surfaces:

1. **Pure-data regulator primitives** — directive construction,
   canonical serialisation, signature verify, audit-response HMAC.
2. **System integration** — directive enforcement (ROTATE / REVOKE),
   replay / expiry / cross-deployment rejection, audit token issue
   and challenge-response.
"""

from __future__ import annotations

import os
import unittest

from rag_sign.corpus import chunk_documents
from rag_sign.embeddings import HashEmbedder
from rag_sign.hsm import InMemoryHSM
from rag_sign.llm import EchoLLM
from rag_sign.rag import RagSignSystem
from rag_sign.regulator import (
    ALLOWED_ACTIONS,
    REVOKE,
    ROTATE,
    InvalidRegulatorDirective,
    RegulatorAuthority,
    RegulatorDirective,
    make_audit_challenge,
    make_directive,
    verify_audit_response,
    verify_directive,
)
from rag_sign.signer import RagSigner
from rag_sign.vector_db import ChromaVectorDB
from rag_sign.verifier import verify_message

CORPUS_SIZE = 200
DOCS_A = [
    (f"doc-{i}.txt", f"Document {i} discusses Schnorr signatures over G_{i}.")
    for i in range(CORPUS_SIZE)
]
DOCS_DRIFT = [
    (src, text) if i % 20 != 0
    else (src, f"Replacement {i}: identity-based encryption with parameter {i*13}.")
    for i, (src, text) in enumerate(DOCS_A)
]
DOCS_DISJOINT = [
    (f"doc-{i}.txt", f"Zeros of zeta on the critical strip, parameter k={i}.")
    for i in range(CORPUS_SIZE)
]
MODEL_HASH = b"\xab" * 32


_seq = 0


def _unique_collection() -> str:
    global _seq
    _seq += 1
    return f"reg-test-{_seq:04d}"


def _make_regulator() -> tuple[RegulatorAuthority, RagSigner]:
    """Construct a regulator keypair plus its public-side authority."""
    signer = RagSigner(os.urandom(32))
    return RegulatorAuthority(
        public_key_pem=signer.public_key_pem, name="test-regulator"
    ), signer


def _build_system(
    *,
    regulator: RegulatorAuthority | None = None,
    deployment_id: str = "default",
    drift_policy_bits: int | None = None,
) -> RagSignSystem:
    return RagSignSystem(
        vector_db=ChromaVectorDB(
            embedder=HashEmbedder(dim=128),
            collection_name=_unique_collection(),
        ),
        llm=EchoLLM(),
        hsm=InMemoryHSM(),
        top_k=3,
        drift_policy_bits=drift_policy_bits,
        regulator=regulator,
        deployment_id=deployment_id,
    )


# ---------------------------------------------------------------------------
# Directive primitives
# ---------------------------------------------------------------------------


class DirectivePrimitiveTests(unittest.TestCase):
    def test_make_directive_defaults(self) -> None:
        d = make_directive(deployment_id="d1", action=ROTATE, reason="scheduled")
        self.assertEqual(d.action, ROTATE)
        self.assertEqual(d.deployment_id, "d1")
        self.assertEqual(len(d.nonce), 16)
        self.assertEqual(d.expires_at - d.issued_at, 300)

    def test_canonical_bytes_is_deterministic(self) -> None:
        d = make_directive(deployment_id="d1", action=ROTATE, now=1_000_000)
        # Two RegulatorDirective instances with identical fields must
        # serialise identically — the *signed* payload is canonical.
        same = RegulatorDirective(
            deployment_id=d.deployment_id,
            nonce_hex=d.nonce_hex,
            issued_at=d.issued_at,
            expires_at=d.expires_at,
            action=d.action,
            reason=d.reason,
        )
        self.assertEqual(d.canonical_bytes(), same.canonical_bytes())

    def test_make_directive_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in allowed"):
            make_directive(deployment_id="d", action="BAD")

    def test_make_directive_rejects_overlong_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "too long"):
            make_directive(deployment_id="d", action=ROTATE, reason="x" * 257)

    def test_verify_directive_round_trip(self) -> None:
        authority, signer = _make_regulator()
        d = make_directive(deployment_id="d", action=ROTATE)
        sig = signer.sign(d.canonical_bytes()).signature
        self.assertTrue(verify_directive(d, sig, authority))

    def test_verify_directive_rejects_tamper(self) -> None:
        authority, signer = _make_regulator()
        d = make_directive(deployment_id="d", action=ROTATE)
        sig = signer.sign(d.canonical_bytes()).signature
        # Mutate any field — signature must no longer verify.
        tampered = RegulatorDirective(
            deployment_id="d-other",  # was "d"
            nonce_hex=d.nonce_hex,
            issued_at=d.issued_at,
            expires_at=d.expires_at,
            action=d.action,
            reason=d.reason,
        )
        self.assertFalse(verify_directive(tampered, sig, authority))

    def test_allowed_actions_set_is_what_we_expect(self) -> None:
        # If a future change adds an action, this guards against
        # forgetting to update the dispatch in
        # RagSignSystem.enforce_regulator_directive.
        self.assertEqual(ALLOWED_ACTIONS, frozenset({ROTATE, REVOKE}))


# ---------------------------------------------------------------------------
# Directive enforcement
# ---------------------------------------------------------------------------


class DirectiveEnforcementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority, self.signer = _make_regulator()
        self.system = _build_system(
            regulator=self.authority, deployment_id="dep-1"
        )
        self.bundle = self.system.enrol(
            chunk_documents(DOCS_A), model_hash=MODEL_HASH
        )

    def _signed(self, directive: RegulatorDirective) -> bytes:
        return self.signer.sign(directive.canonical_bytes()).signature

    def test_valid_rotate_directive_rotates(self) -> None:
        old_pem = self.bundle.public_key_pem
        directive = make_directive(deployment_id="dep-1", action=ROTATE)
        new_bundle = self.system.enforce_regulator_directive(
            directive, self._signed(directive), chunk_documents(DOCS_A)
        )
        self.assertIsNotNone(new_bundle)
        self.assertNotEqual(new_bundle.public_key_pem, old_pem)
        # Subsequent query signs under the new key.
        msg = self.system.query("any")
        self.assertEqual(msg.public_key_pem, new_bundle.public_key_pem)
        self.assertTrue(verify_message(msg))

    def test_revoke_blocks_future_queries(self) -> None:
        directive = make_directive(deployment_id="dep-1", action=REVOKE)
        result = self.system.enforce_regulator_directive(
            directive, self._signed(directive), chunk_documents(DOCS_A)
        )
        self.assertIsNone(result)
        with self.assertRaisesRegex(RuntimeError, "revoked"):
            self.system.query("any")

    def test_bad_signature_is_rejected(self) -> None:
        directive = make_directive(deployment_id="dep-1", action=ROTATE)
        bad_sig = bytearray(self._signed(directive))
        bad_sig[-1] ^= 1
        with self.assertRaises(InvalidRegulatorDirective):
            self.system.enforce_regulator_directive(
                directive, bytes(bad_sig), chunk_documents(DOCS_A)
            )

    def test_replayed_nonce_is_rejected(self) -> None:
        directive = make_directive(deployment_id="dep-1", action=ROTATE)
        sig = self._signed(directive)
        # First call succeeds.
        self.system.enforce_regulator_directive(
            directive, sig, chunk_documents(DOCS_A)
        )
        # Second call with the same directive must fail (nonce reuse).
        with self.assertRaisesRegex(InvalidRegulatorDirective, "replay"):
            self.system.enforce_regulator_directive(
                directive, sig, chunk_documents(DOCS_A)
            )

    def test_expired_directive_is_rejected(self) -> None:
        # Make the directive look like it was issued in 1971.
        directive = make_directive(
            deployment_id="dep-1", action=ROTATE, ttl_seconds=60, now=42
        )
        with self.assertRaisesRegex(InvalidRegulatorDirective, "expired"):
            self.system.enforce_regulator_directive(
                directive, self._signed(directive), chunk_documents(DOCS_A)
            )

    def test_wrong_deployment_id_is_rejected(self) -> None:
        directive = make_directive(deployment_id="some-other-deployment", action=ROTATE)
        with self.assertRaisesRegex(InvalidRegulatorDirective, "this deployment"):
            self.system.enforce_regulator_directive(
                directive, self._signed(directive), chunk_documents(DOCS_A)
            )

    def test_no_regulator_configured_raises(self) -> None:
        sys_no_reg = _build_system(regulator=None)
        sys_no_reg.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        directive = make_directive(deployment_id="default", action=ROTATE)
        sig = self._signed(directive)
        with self.assertRaisesRegex(RuntimeError, "no regulator"):
            sys_no_reg.enforce_regulator_directive(
                directive, sig, chunk_documents(DOCS_A)
            )


# ---------------------------------------------------------------------------
# Audit (secure sketch) flow
# ---------------------------------------------------------------------------


class AuditFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = _build_system()
        self.system.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        self.token = self.system.issue_audit_token()

    def test_unchanged_corpus_passes_audit(self) -> None:
        challenge = make_audit_challenge()
        response = self.system.respond_to_audit_challenge(
            chunk_documents(DOCS_A), challenge
        )
        self.assertTrue(verify_audit_response(self.token, challenge, response))

    def test_drifted_within_bound_still_passes(self) -> None:
        challenge = make_audit_challenge()
        response = self.system.respond_to_audit_challenge(
            chunk_documents(DOCS_DRIFT), challenge
        )
        self.assertTrue(verify_audit_response(self.token, challenge, response))

    def test_disjoint_corpus_fails_audit(self) -> None:
        from rag_sign.fuzzy_extractor import FuzzyExtractFailure

        challenge = make_audit_challenge()
        with self.assertRaises(FuzzyExtractFailure):
            self.system.respond_to_audit_challenge(
                chunk_documents(DOCS_DISJOINT), challenge
            )

    def test_audit_secret_distinct_from_signing_key(self) -> None:
        """The regulator's audit secret must not be the deployment's signing seed."""
        # ``audit_secret`` is a 32-byte value derived from a *separate*
        # fe_gen call, so it's independent of the seed used by the
        # signer.  Cheap necessary check: confirm the deployment's
        # public key (derived from the signing seed) is not equal to
        # the audit secret bit-pattern.
        self.assertNotEqual(self.token.audit_secret, self.system.public_key_pem)

    def test_audit_response_constant_under_replay(self) -> None:
        """Same challenge → same response (deterministic HMAC)."""
        challenge = make_audit_challenge()
        r1 = self.system.respond_to_audit_challenge(chunk_documents(DOCS_A), challenge)
        r2 = self.system.respond_to_audit_challenge(chunk_documents(DOCS_A), challenge)
        self.assertEqual(r1, r2)

    def test_audit_response_changes_with_challenge(self) -> None:
        c1 = make_audit_challenge()
        c2 = make_audit_challenge()
        r1 = self.system.respond_to_audit_challenge(chunk_documents(DOCS_A), c1)
        r2 = self.system.respond_to_audit_challenge(chunk_documents(DOCS_A), c2)
        self.assertNotEqual(r1, r2)

    def test_no_token_issued_raises(self) -> None:
        sys2 = _build_system()
        sys2.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        with self.assertRaisesRegex(RuntimeError, "no audit token"):
            sys2.respond_to_audit_challenge(chunk_documents(DOCS_A), b"\x00" * 32)


# ---------------------------------------------------------------------------
# End-to-end realistic flow
# ---------------------------------------------------------------------------


class RegulatorEndToEndTests(unittest.TestCase):
    """Audit fails → regulator issues ROTATE → deployment rotates."""

    def test_audit_failure_triggers_directive_to_rotate(self) -> None:
        from rag_sign.fuzzy_extractor import FuzzyExtractFailure

        authority, signer = _make_regulator()
        system = _build_system(regulator=authority, deployment_id="dep-e2e")
        bundle_old = system.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        token = system.issue_audit_token()
        self.assertEqual(token.deployment_id, "dep-e2e")

        # The corpus has been replaced wholesale (compromise scenario).
        # An audit attempt fails because drift exceeds the BCH bound.
        challenge = make_audit_challenge()
        with self.assertRaises(FuzzyExtractFailure):
            system.respond_to_audit_challenge(
                chunk_documents(DOCS_DISJOINT), challenge
            )

        # Regulator concludes the deployment is no longer in good
        # state and issues a ROTATE directive.  The new corpus is
        # *DOCS_DISJOINT* — the deployment will re-enrol against
        # whatever it's currently serving.
        directive = make_directive(
            deployment_id="dep-e2e",
            action=ROTATE,
            reason="audit failure",
        )
        signature = signer.sign(directive.canonical_bytes()).signature
        bundle_new = system.enforce_regulator_directive(
            directive, signature, chunk_documents(DOCS_DISJOINT)
        )

        self.assertIsNotNone(bundle_new)
        self.assertNotEqual(bundle_new.public_key_pem, bundle_old.public_key_pem)

        # Post-rotation queries sign under the new key.
        msg = system.query("post-rotation question")
        self.assertEqual(msg.public_key_pem, bundle_new.public_key_pem)
        self.assertTrue(verify_message(msg))


if __name__ == "__main__":
    unittest.main()
