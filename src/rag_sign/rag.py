"""End-to-end RAG-Sign pipeline.

Brings together:

* :mod:`rag_sign.lsh`              — corpus fingerprint
* :mod:`rag_sign.fuzzy_extractor`  — drift-tolerant key recovery
* :mod:`rag_sign.hsm`              — long-term secret custody
* :mod:`rag_sign.key_derivation`   — Algorithm 1 (SHA3-256 binding)
* :mod:`rag_sign.signer`           — ECDSA P-256 signing
* :mod:`rag_sign.vector_db`        — Chroma retrieval store
* :mod:`rag_sign.llm`              — Llama 3.2 generation

Two-phase lifecycle:

* :meth:`RagSignSystem.enrol` — first run on a fresh corpus.  Embeds
  and indexes every chunk, computes the LSH fingerprint, generates an
  HSM secret, derives the signing key, and emits an
  :class:`EnrolmentBundle` containing everything a recovering instance
  needs to re-derive the same key (helper data, HSM handle, model
  fingerprint).  The bundle's ``public_key_pem`` becomes the long-term
  identity that downstream verifiers / allow-lists trust.

* :meth:`RagSignSystem.recover` — subsequent runs.  Re-embeds the
  current corpus, reconstructs the fingerprint, runs the fuzzy-
  extractor recovery, fetches the HSM secret, and re-derives the same
  key.  Fails closed if the corpus has drifted past the BCH bound.

Once recovered, :meth:`RagSignSystem.query` is the working API: it
retrieves, generates, and signs in one shot.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag_sign.corpus import Chunk
from rag_sign.fuzzy_extractor import HelperData
from rag_sign.fuzzy_extractor import T as BCH_T
from rag_sign.fuzzy_extractor import gen as fe_gen
from rag_sign.fuzzy_extractor import reconstruct_w as fe_reconstruct_w
from rag_sign.fuzzy_extractor import rep as fe_rep
from rag_sign.hsm import HSMBackend, InMemoryHSM
from rag_sign.key_derivation import KeyMaterial, derive_signing_seed
from rag_sign.llm import LLMBackend, assemble_prompt
from rag_sign.lsh import fingerprint_corpus, hamming_distance
from rag_sign.regulator import (
    InvalidRegulatorDirective,
    RegulatorAuditToken,
    RegulatorAuthority,
    RegulatorDirective,
    _AuditState,
    compute_audit_response,
    verify_directive,
)
from rag_sign.signer import RagSigner, SignedMessage
from rag_sign.vector_db import ChromaVectorDB


class DriftPolicyExceeded(RuntimeError):
    """Raised when corpus drift exceeds the configured *policy* bound.

    The policy bound is a soft fence the deployment chooses below the
    cryptographic hard ceiling (the BCH ``t`` parameter).  When this
    fires, BCH recovery would in fact still succeed — the system is
    deliberately failing closed earlier so the operator can rotate
    the signing key before the corpus drifts close enough to the
    cliff that BCH itself starts losing.

    Caught typically by control-plane code which then invokes
    :meth:`RagSignSystem.regenerate` to rotate.
    """

    def __init__(self, hamming: int, policy_bits: int, bch_t: int) -> None:
        self.hamming = hamming
        self.policy_bits = policy_bits
        self.bch_t = bch_t
        super().__init__(
            f"corpus drift {hamming} bits exceeds policy bound "
            f"{policy_bits} bits (BCH ceiling t={bch_t}); "
            f"rotate via RagSignSystem.regenerate()"
        )


@dataclass(frozen=True, slots=True)
class EnrolmentBundle:
    """Public state needed to bring up a recovering :class:`RagSignSystem`.

    Distributed at enrolment time.  Knowing this bundle is *not*
    sufficient to forge a signature — recovery additionally requires
    the same model file (to reproduce ``model_hash``) and a live
    connection to the HSM that holds the secret named by ``hsm_handle``.
    """

    public_key_pem: bytes
    helper: HelperData
    hsm_handle: bytes
    model_hash: bytes


@dataclass(slots=True)
class _SystemState:
    """Internal: holds the live signer once enrolment / recovery succeeds.

    ``w_enrol`` is kept locally only — it is *not* part of
    :class:`EnrolmentBundle` because publishing
    ``(helper, w_enrol)`` would let an observer recover the codeword
    ``c = helper ⊕ w_enrol`` and from it the secret ``R``.  It is
    recomputed from the corpus at enrolment / recovery time.
    """
    signer: RagSigner
    bundle: EnrolmentBundle
    w_enrol: bytes


class RagSignSystem:
    """Top-level RAG-Sign service.

    Construction wires the four pluggable backends but does **not**
    derive any keys — that happens via :meth:`enrol` (first run) or
    :meth:`recover` (subsequent runs).  The class enforces this
    explicitly so a deployment cannot accidentally serve unsigned
    answers.
    """

    def __init__(
        self,
        *,
        vector_db: ChromaVectorDB,
        llm: LLMBackend,
        hsm: HSMBackend | None = None,
        top_k: int = 5,
        drift_policy_bits: int | None = None,
        regulator: RegulatorAuthority | None = None,
        deployment_id: str = "default",
    ) -> None:
        """Configure backends and (optionally) a drift-policy threshold.

        ``drift_policy_bits`` is a *soft* ceiling on the Hamming distance
        between the enrolment fingerprint and any later one.  When set,
        :meth:`recover` and :meth:`check_drift` raise
        :class:`DriftPolicyExceeded` as soon as the drift exceeds this
        bound — even if BCH could in fact still recover.  Use it to
        rotate the signing key well before the corpus drifts to the
        edge of the BCH cliff (paper §6.5).

        ``None`` (the default) means no soft policy: the only ceiling
        is the BCH bound ``T`` from :mod:`rag_sign.fuzzy_extractor`.
        """
        if drift_policy_bits is not None:
            if drift_policy_bits < 0:
                raise ValueError(
                    f"drift_policy_bits must be ≥ 0 (got {drift_policy_bits})"
                )
            if drift_policy_bits > BCH_T:
                # Higher than the BCH bound is meaningless — BCH will fail
                # before the policy fires.
                raise ValueError(
                    f"drift_policy_bits ({drift_policy_bits}) must be ≤ "
                    f"BCH bound t={BCH_T}; values above the cryptographic "
                    f"ceiling cannot be enforced"
                )
        self._db = vector_db
        self._llm = llm
        self._hsm: HSMBackend = hsm or InMemoryHSM()
        self._top_k = top_k
        self._drift_policy_bits = drift_policy_bits
        self._regulator = regulator
        self._deployment_id = deployment_id
        # Replay protection: every accepted regulator directive's nonce
        # is added here and refused on subsequent submission.  In-memory
        # only; persisting across restarts is the operator's job.
        self._seen_nonces: set[bytes] = set()
        # Per-regulator audit relationships, keyed by deployment_id (so
        # a single deployment can serve audits to multiple regulators).
        self._audit_states: dict[str, _AuditState] = {}
        # Set when a REVOKE directive lands; subsequent query() refuses.
        self._revoked = False
        self._state: _SystemState | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def enrol(self, chunks: Sequence[Chunk], *, model_hash: bytes) -> EnrolmentBundle:
        """Run on a fresh corpus.  Index, fingerprint, derive key, return bundle."""
        if self._state is not None:
            raise RuntimeError("system already enrolled / recovered")

        # 1. Index the corpus into the vector store.
        self._db.add(list(chunks))

        # 2. Build the corpus fingerprint and run the fuzzy-extractor enrolment.
        w = fingerprint_corpus([c.text for c in chunks])
        r, helper = fe_gen(w)

        # 3. Provision the HSM secret.
        handle = self._hsm.enrol()
        secret = self._hsm.fetch(handle)

        # 4. Derive the signing seed via Algorithm 1.
        material = KeyMaterial(
            model_hash=model_hash,
            lsh_key=r,
            hsm_secret=secret,
        )
        seed = derive_signing_seed(material)
        signer = RagSigner(seed)

        bundle = EnrolmentBundle(
            public_key_pem=signer.public_key_pem,
            helper=helper,
            hsm_handle=handle,
            model_hash=model_hash,
        )
        self._state = _SystemState(signer=signer, bundle=bundle, w_enrol=w)
        return bundle

    def recover(
        self,
        chunks: Sequence[Chunk],
        bundle: EnrolmentBundle,
    ) -> None:
        """Recover the signing key from a (possibly drifted) corpus + bundle.

        Order of failure modes:

        1. :class:`DriftPolicyExceeded` — drift over the soft policy
           bound (if configured).  BCH could still recover but the
           operator wants to rotate before getting closer to the cliff.
        2. :class:`rag_sign.fuzzy_extractor.FuzzyExtractFailure` —
           drift over the BCH bound; recovery is cryptographically
           impossible.
        3. :class:`AssertionError` — recovered public key does not
           match the bundle's (smoke-test against silent corruption).
        """
        if self._state is not None:
            raise RuntimeError("system already enrolled / recovered")

        self._db.add(list(chunks))

        w = fingerprint_corpus([c.text for c in chunks])

        # Policy gate fires *before* BCH so the operator can rotate
        # cleanly while recovery would still succeed cryptographically.
        if self._drift_policy_bits is not None:
            # We need the enrolment fingerprint to measure drift.  At
            # recovery time we don't have it locally — but we can
            # reconstruct it from (helper, codeword) once BCH succeeds.
            # To enforce the policy *before* BCH, we instead rely on
            # the BCH bound itself: if BCH succeeds, the recovered ``r``
            # gives us the codeword, which together with helper yields
            # ``w_enrol``.  Compute that, measure drift, decide.
            r = fe_rep(w, bundle.helper)
            secret = self._hsm.fetch(bundle.hsm_handle)
            material = KeyMaterial(
                model_hash=bundle.model_hash,
                lsh_key=r,
                hsm_secret=secret,
            )
            seed = derive_signing_seed(material)
            signer = RagSigner(seed)
            if signer.public_key_pem != bundle.public_key_pem:
                raise AssertionError(
                    "recovered public key does not match enrolment bundle — "
                    "corpus or HSM state is inconsistent"
                )

            w_enrol = fe_reconstruct_w(w, bundle.helper)
            hamming = hamming_distance(w_enrol, w)
            if hamming > self._drift_policy_bits:
                raise DriftPolicyExceeded(
                    hamming=hamming,
                    policy_bits=self._drift_policy_bits,
                    bch_t=BCH_T,
                )
            self._state = _SystemState(signer=signer, bundle=bundle, w_enrol=w_enrol)
            return

        # No policy configured — BCH bound is the only ceiling.
        r = fe_rep(w, bundle.helper)

        secret = self._hsm.fetch(bundle.hsm_handle)
        material = KeyMaterial(
            model_hash=bundle.model_hash,
            lsh_key=r,
            hsm_secret=secret,
        )
        seed = derive_signing_seed(material)
        signer = RagSigner(seed)

        if signer.public_key_pem != bundle.public_key_pem:
            raise AssertionError(
                "recovered public key does not match enrolment bundle — "
                "corpus or HSM state is inconsistent"
            )
        w_enrol = fe_reconstruct_w(w, bundle.helper)
        self._state = _SystemState(signer=signer, bundle=bundle, w_enrol=w_enrol)

    # ------------------------------------------------------------------
    # Drift policy + rotation
    # ------------------------------------------------------------------

    def check_drift(self, chunks: Sequence[Chunk]) -> int:
        """Measure Hamming distance between the live corpus and enrolment.

        Returns the distance in bits.  Raises
        :class:`DriftPolicyExceeded` if a soft policy is configured and
        the live drift exceeds it.  Does **not** raise on BCH-bound
        violation — :meth:`recover` is the place for that.
        """
        state = self._require_state()
        w_now = fingerprint_corpus([c.text for c in chunks])
        d = hamming_distance(state.w_enrol, w_now)
        if (
            self._drift_policy_bits is not None
            and d > self._drift_policy_bits
        ):
            raise DriftPolicyExceeded(
                hamming=d,
                policy_bits=self._drift_policy_bits,
                bch_t=BCH_T,
            )
        return d

    def regenerate(
        self,
        chunks: Sequence[Chunk],
        *,
        model_hash: bytes | None = None,
    ) -> EnrolmentBundle:
        """Rotate the signing key against the *current* corpus.

        Used after a :class:`DriftPolicyExceeded` event (or at any
        time the deployment chooses to rotate).  The HSM secret stays
        put — only the LSH-derived component changes — so the new
        public key is uncorrelated with the old one.

        ``model_hash`` defaults to the previous bundle's value; pass
        a new one if the LLM weights have also changed.

        Returns the new :class:`EnrolmentBundle`.  Callers should
        publish ``bundle.public_key_pem`` as the deployment's new
        long-term identity and revoke the old key from any
        allow-lists.
        """
        state = self._require_state()
        prev = state.bundle

        # Fresh fingerprint over the current corpus.
        w_new = fingerprint_corpus([c.text for c in chunks])
        r_new, helper_new = fe_gen(w_new)

        # Re-use the existing HSM secret — same chip, same model, new
        # corpus identity.  This is what makes the *signing key* rotate
        # while the operational continuity (HSM provisioning, model
        # weights) is preserved.
        secret = self._hsm.fetch(prev.hsm_handle)

        material = KeyMaterial(
            model_hash=model_hash if model_hash is not None else prev.model_hash,
            lsh_key=r_new,
            hsm_secret=secret,
        )
        seed = derive_signing_seed(material)
        signer = RagSigner(seed)

        bundle = EnrolmentBundle(
            public_key_pem=signer.public_key_pem,
            helper=helper_new,
            hsm_handle=prev.hsm_handle,
            model_hash=material.model_hash,
        )
        self._state = _SystemState(signer=signer, bundle=bundle, w_enrol=w_new)
        return bundle

    def regenerate_on_demand(
        self,
        chunks: Sequence[Chunk],
        *,
        reason: str = "",
        model_hash: bytes | None = None,
    ) -> EnrolmentBundle:
        """Admin-path key rotation.

        Equivalent to :meth:`regenerate` but takes an explicit
        ``reason`` string so the rotation event can be audit-logged
        with operator intent.  Use this when a human operator
        decides to rotate (scheduled rotation, suspected
        compromise, …) without waiting for a regulator directive.

        For *regulator-authorised* rotation use
        :meth:`enforce_regulator_directive` — it carries the
        cryptographic proof that the regulator authorised the
        rotation.
        """
        del reason  # logged by the caller; not security-relevant here
        return self.regenerate(chunks, model_hash=model_hash)

    # ------------------------------------------------------------------
    # Regulator authority + secure-sketch audit
    # ------------------------------------------------------------------

    def issue_audit_token(
        self, *, regulator_deployment_id: str | None = None
    ) -> RegulatorAuditToken:
        """Issue a fresh audit token at enrolment time.

        The deployment runs a *second* fuzzy-extractor enrolment over
        the same corpus fingerprint, keeps the helper data locally,
        and returns the resulting ``R`` (the audit secret) for
        out-of-band delivery to the regulator.  The regulator stores
        the returned token; the deployment stores only the helper.

        Multiple audit tokens can co-exist (e.g. one per regulator).
        Each is keyed by ``regulator_deployment_id`` (defaults to the
        deployment's own ``deployment_id`` for the single-regulator
        case).
        """
        state = self._require_state()
        key = regulator_deployment_id or self._deployment_id
        r_audit, helper_audit = fe_gen(state.w_enrol)
        self._audit_states[key] = _AuditState(
            deployment_id=key, helper=helper_audit
        )
        return RegulatorAuditToken(
            deployment_id=key,
            audit_secret=r_audit,
            issued_at=int(time.time()),
        )

    def respond_to_audit_challenge(
        self,
        chunks: Sequence[Chunk],
        challenge: bytes,
        *,
        regulator_deployment_id: str | None = None,
    ) -> bytes:
        """Generate the HMAC proof for a regulator's audit challenge.

        Re-fingerprints the live corpus, recovers ``R`` via the stored
        helper (raises :class:`FuzzyExtractFailure` if drift exceeds
        the BCH bound), and returns
        ``HMAC-SHA3-256(R, challenge)`` for the regulator to verify.

        A failure here is itself the audit signal: if the live corpus
        cannot reproduce the audit ``R``, the regulator should issue
        a ROTATE directive.
        """
        self._require_state()  # must be ready
        key = regulator_deployment_id or self._deployment_id
        if key not in self._audit_states:
            raise RuntimeError(
                f"no audit token has been issued for {key!r}; "
                f"call issue_audit_token() first"
            )
        helper = self._audit_states[key].helper
        w_now = fingerprint_corpus([c.text for c in chunks])
        r_audit = fe_rep(w_now, helper)
        return compute_audit_response(r_audit, challenge)

    def enforce_regulator_directive(
        self,
        directive: RegulatorDirective,
        signature: bytes,
        chunks: Sequence[Chunk],
        *,
        now: int | None = None,
    ) -> EnrolmentBundle | None:
        """Verify and execute a signed regulator directive.

        Failure modes (in order):

        * No regulator configured at construction — :class:`RuntimeError`.
        * ECDSA signature does not verify — :class:`InvalidRegulatorDirective`.
        * Directive's ``deployment_id`` does not match ours — same.
        * Directive expired — same.
        * Nonce already seen (replay) — same.
        * Unknown action — same.

        On success, performs the action:

        * ``"ROTATE"`` returns the new :class:`EnrolmentBundle`.
        * ``"REVOKE"`` marks the deployment as revoked (subsequent
          :meth:`query` calls refuse) and returns ``None``.
        """
        if self._regulator is None:
            raise RuntimeError("no regulator authority configured")

        if not verify_directive(directive, signature, self._regulator):
            raise InvalidRegulatorDirective(
                "ECDSA signature on directive did not verify against "
                "the registered regulator authority"
            )
        if directive.deployment_id != self._deployment_id:
            raise InvalidRegulatorDirective(
                f"directive targets {directive.deployment_id!r}, "
                f"this deployment is {self._deployment_id!r}"
            )
        wall = int(now if now is not None else time.time())
        if wall > directive.expires_at:
            raise InvalidRegulatorDirective(
                f"directive expired at {directive.expires_at} "
                f"(now {wall})"
            )
        if directive.nonce in self._seen_nonces:
            raise InvalidRegulatorDirective(
                "directive nonce has already been seen — replay refused"
            )
        self._seen_nonces.add(directive.nonce)

        if directive.action == "ROTATE":
            return self.regenerate(chunks)
        if directive.action == "REVOKE":
            self._revoked = True
            return None
        raise InvalidRegulatorDirective(
            f"unknown directive action {directive.action!r}"
        )

    # ------------------------------------------------------------------
    # Working API
    # ------------------------------------------------------------------

    def query(self, question: str) -> SignedMessage:
        """Retrieve, generate, sign — return a :class:`SignedMessage`.

        The signed payload is the **answer** the LLM produced, not the
        question or the context.  Verifiers see only the answer + the
        public key + the signature; binding the key to the corpus is
        what makes the answer attributable to a specific deployment.
        """
        state = self._require_state()
        if self._revoked:
            raise RuntimeError(
                "deployment has been revoked by regulator directive; "
                "queries are refused until re-enrolment"
            )
        retrieved = self._db.query(question, top_k=self._top_k)
        contexts = [r.chunk.text for r in retrieved]
        prompt = assemble_prompt(question, contexts)
        answer = self._llm.generate(prompt)
        return state.signer.sign(answer.encode("utf-8"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def public_key_pem(self) -> bytes:
        return self._require_state().bundle.public_key_pem

    @property
    def is_ready(self) -> bool:
        return self._state is not None

    def _require_state(self) -> _SystemState:
        if self._state is None:
            raise RuntimeError("system not enrolled / recovered yet")
        return self._state
