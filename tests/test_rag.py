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
from rag_sign.rag import RagSignSystem
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


if __name__ == "__main__":
    unittest.main()
