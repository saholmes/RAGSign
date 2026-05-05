"""End-to-end pipeline tests for :mod:`rag_sign.rag`.

The full vertical slice — corpus → embed → index → fingerprint →
fuzzy-extract → derive → sign / verify, plus retrieval and recovery
— runs here under deterministic mock backends so the suite stays
fast and self-contained.
"""

from __future__ import annotations

import unittest

from rag_sign.corpus import chunk_documents
from rag_sign.embeddings import HashEmbedder
from rag_sign.fuzzy_extractor import FuzzyExtractFailure
from rag_sign.hsm import InMemoryHSM
from rag_sign.llm import EchoLLM
from rag_sign.rag import DriftPolicyExceeded, RagSignSystem
from rag_sign.vector_db import ChromaVectorDB
from rag_sign.verifier import verify_message

# A small synthetic corpus.  Larger than 200 docs so the LSH fingerprint
# behaves well under 5 % drift (see test_lsh.py rationale).
CORPUS_SIZE = 200
DOCS_A = [
    (
        f"doc-{i}.txt",
        f"Document {i} discusses Schnorr signatures over group G_{i} "
        f"and proves EUF-CMA security under the discrete log assumption.",
    )
    for i in range(CORPUS_SIZE)
]
DOCS_DRIFT = [
    (src, text) if i % 20 != 0  # 5 %-replacement: every 20th doc
    else (
        src,
        f"Replacement document {i}: identity-based encryption "
        f"with parameter set {i*13}.",
    )
    for i, (src, text) in enumerate(DOCS_A)
]
DOCS_DISJOINT = [
    (
        f"doc-{i}.txt",
        f"On the asymptotic distribution of zeros of zeta(s) in "
        f"the critical strip, parameter k={i}.",
    )
    for i in range(CORPUS_SIZE)
]

MODEL_HASH = b"\xab" * 32  # stand-in 32-byte SHA3-256 of weights


_collection_seq = 0


def _unique_collection_name(prefix: str = "test") -> str:
    global _collection_seq
    _collection_seq += 1
    return f"{prefix}-{_collection_seq:04d}"


def _build_system() -> RagSignSystem:
    embedder = HashEmbedder(dim=128)  # small dim → fast tests
    db = ChromaVectorDB(embedder=embedder, collection_name=_unique_collection_name())
    return RagSignSystem(vector_db=db, llm=EchoLLM(), hsm=InMemoryHSM(), top_k=3)


class EnrolAndQueryTests(unittest.TestCase):
    def test_enrol_then_query_produces_verifiable_signature(self) -> None:
        sys = _build_system()
        chunks = chunk_documents(DOCS_A)
        bundle = sys.enrol(chunks, model_hash=MODEL_HASH)

        msg = sys.query("Tell me about Schnorr signatures")
        self.assertTrue(verify_message(msg))
        # Answer (echoed prompt) must contain the system context tag.
        self.assertIn(b"<|system|>", msg.payload)

        # Public key in the signed message matches the enrolment bundle.
        self.assertEqual(msg.public_key_pem, bundle.public_key_pem)

    def test_signature_invalid_under_wrong_pubkey(self) -> None:
        sys_a = _build_system()
        sys_b = _build_system()
        sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        sys_b.enrol(chunk_documents(DOCS_DISJOINT), model_hash=MODEL_HASH)

        # B's public key must reject A's signed answer.
        msg = sys_a.query("anything")
        from rag_sign.signer import verify

        self.assertFalse(verify(msg.payload, msg.signature, sys_b.public_key_pem))


class RecoveryTests(unittest.TestCase):
    def test_recover_from_unchanged_corpus(self) -> None:
        # Enrol on system A, recover on system B with the same HSM,
        # the same corpus, and the same bundle.
        hsm = InMemoryHSM()
        sys_a = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="enrol-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        bundle = sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)

        sys_b = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="recover-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        sys_b.recover(chunk_documents(DOCS_A), bundle)

        self.assertEqual(sys_b.public_key_pem, bundle.public_key_pem)

    def test_recover_under_5pct_drift(self) -> None:
        hsm = InMemoryHSM()
        sys_a = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="enrol-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        bundle = sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)

        sys_b = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="recover-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        # Recovery must succeed and yield the *same* public key.
        sys_b.recover(chunk_documents(DOCS_DRIFT), bundle)
        self.assertEqual(sys_b.public_key_pem, bundle.public_key_pem)

    def test_recover_rejects_disjoint_corpus(self) -> None:
        hsm = InMemoryHSM()
        sys_a = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="enrol-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        bundle = sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)

        sys_b = RagSignSystem(
            vector_db=ChromaVectorDB(embedder=HashEmbedder(dim=128), collection_name="recover-side"),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        with self.assertRaises(FuzzyExtractFailure):
            sys_b.recover(chunk_documents(DOCS_DISJOINT), bundle)


class LifecycleGuardTests(unittest.TestCase):
    def test_query_before_enrol_raises(self) -> None:
        sys = _build_system()
        with self.assertRaisesRegex(RuntimeError, "not enrolled"):
            sys.query("anything")

    def test_enrol_twice_raises(self) -> None:
        sys = _build_system()
        sys.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        with self.assertRaisesRegex(RuntimeError, "already"):
            sys.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)


class DriftPolicyTests(unittest.TestCase):
    """Configurable soft drift bound below the BCH ceiling.

    Recall the geometry: BCH ``T = 30`` is the cryptographic hard
    ceiling.  Empirically, identical and lightly-drifted corpora sit
    at small Hamming distances (< 5 bits in 200-doc tests).  We use
    distinct test corpora (DOCS_A vs DOCS_DRIFT) to land *between*
    those two extremes — the exact distance varies with shingle
    layout, but it is reliably ``> 0`` and ``<= T``.
    """

    def setUp(self) -> None:
        self.system = _build_system()
        self.bundle = self.system.enrol(
            chunk_documents(DOCS_A), model_hash=MODEL_HASH
        )

    def test_check_drift_zero_for_identical_corpus(self) -> None:
        d = self.system.check_drift(chunk_documents(DOCS_A))
        self.assertEqual(d, 0)

    def test_check_drift_returns_int_for_drifted_corpus(self) -> None:
        d = self.system.check_drift(chunk_documents(DOCS_DRIFT))
        self.assertIsInstance(d, int)
        self.assertGreaterEqual(d, 0)

    def test_check_drift_under_policy_passes(self) -> None:
        sys = _build_system()
        # Generous policy — well above what any small drift produces.
        sys._drift_policy_bits = 30  # noqa: SLF001
        sys.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        # Identical corpus → 0 bits → safely under any policy.
        sys.check_drift(chunk_documents(DOCS_A))

    def test_check_drift_over_policy_raises(self) -> None:
        sys = _build_system()
        # Policy of 0 bits means *any* drift triggers — easy to assert.
        sys._drift_policy_bits = 0  # noqa: SLF001
        sys.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        # Even minimally drifted corpus must exceed a 0-bit policy.
        with self.assertRaises(DriftPolicyExceeded) as ctx:
            sys.check_drift(chunk_documents(DOCS_DRIFT))
        self.assertEqual(ctx.exception.policy_bits, 0)
        self.assertGreaterEqual(ctx.exception.hamming, 0)

    def test_recover_enforces_policy_before_bch(self) -> None:
        """Recovery into a drifted corpus must fire policy before BCH."""
        # Build a fresh system with policy = 0 and the existing HSM.
        hsm = InMemoryHSM()
        sys_a = RagSignSystem(
            vector_db=ChromaVectorDB(
                embedder=HashEmbedder(dim=128),
                collection_name=_unique_collection_name("policy-enrol"),
            ),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
        )
        bundle = sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)

        sys_b = RagSignSystem(
            vector_db=ChromaVectorDB(
                embedder=HashEmbedder(dim=128),
                collection_name=_unique_collection_name("policy-recover"),
            ),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
            drift_policy_bits=0,   # any drift triggers
        )
        with self.assertRaises(DriftPolicyExceeded):
            sys_b.recover(chunk_documents(DOCS_DRIFT), bundle)

    def test_init_rejects_policy_above_bch_bound(self) -> None:
        # A policy higher than the cryptographic ceiling can never fire,
        # so the constructor refuses it.
        with self.assertRaisesRegex(ValueError, "BCH bound"):
            RagSignSystem(
                vector_db=ChromaVectorDB(
                    embedder=HashEmbedder(dim=128),
                    collection_name=_unique_collection_name("bad-policy"),
                ),
                llm=EchoLLM(),
                hsm=InMemoryHSM(),
                drift_policy_bits=10_000,
            )

    def test_init_rejects_negative_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "≥ 0"):
            RagSignSystem(
                vector_db=ChromaVectorDB(
                    embedder=HashEmbedder(dim=128),
                    collection_name=_unique_collection_name("neg-policy"),
                ),
                llm=EchoLLM(),
                hsm=InMemoryHSM(),
                drift_policy_bits=-1,
            )


class RegenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = _build_system()
        self.bundle = self.system.enrol(
            chunk_documents(DOCS_A), model_hash=MODEL_HASH
        )

    def test_regenerate_yields_a_different_public_key(self) -> None:
        old_pem = self.bundle.public_key_pem
        new_bundle = self.system.regenerate(chunk_documents(DOCS_DISJOINT))
        self.assertNotEqual(old_pem, new_bundle.public_key_pem)

    def test_regenerate_preserves_hsm_handle(self) -> None:
        new_bundle = self.system.regenerate(chunk_documents(DOCS_DISJOINT))
        self.assertEqual(new_bundle.hsm_handle, self.bundle.hsm_handle)

    def test_regenerate_signs_under_new_corpus(self) -> None:
        new_bundle = self.system.regenerate(chunk_documents(DOCS_DISJOINT))
        msg = self.system.query("any question")
        # The new public key in the signed message matches the new bundle.
        self.assertEqual(msg.public_key_pem, new_bundle.public_key_pem)
        # Old public key must NOT verify the new signed answer.
        from rag_sign.signer import verify

        self.assertFalse(verify(msg.payload, msg.signature, self.bundle.public_key_pem))
        # New public key must verify it.
        self.assertTrue(verify_message(msg))

    def test_regenerate_takes_new_model_hash(self) -> None:
        new_model = b"\xcd" * 32
        new_bundle = self.system.regenerate(
            chunk_documents(DOCS_DISJOINT), model_hash=new_model
        )
        self.assertEqual(new_bundle.model_hash, new_model)

    def test_regenerate_after_drift_policy_violation(self) -> None:
        """Realistic flow: policy fires, operator regenerates, life goes on."""
        hsm = InMemoryHSM()
        sys_a = RagSignSystem(
            vector_db=ChromaVectorDB(
                embedder=HashEmbedder(dim=128),
                collection_name=_unique_collection_name("flow-enrol"),
            ),
            llm=EchoLLM(),
            hsm=hsm,
            top_k=3,
            drift_policy_bits=0,
        )
        sys_a.enrol(chunk_documents(DOCS_A), model_hash=MODEL_HASH)
        # Drift triggers policy.
        with self.assertRaises(DriftPolicyExceeded):
            sys_a.check_drift(chunk_documents(DOCS_DRIFT))
        # Operator rotates and the system carries on.
        new_bundle = sys_a.regenerate(chunk_documents(DOCS_DRIFT))
        msg = sys_a.query("post-rotation query")
        self.assertEqual(msg.public_key_pem, new_bundle.public_key_pem)


if __name__ == "__main__":
    unittest.main()
