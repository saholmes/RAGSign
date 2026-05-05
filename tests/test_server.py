"""FastAPI surface tests for :mod:`rag_sign.server`.

Uses FastAPI's TestClient over a fully-mocked RAG-Sign system
(HashEmbedder + EchoLLM + InMemoryHSM) so the suite stays
self-contained.  The tests here exercise the gating layer specifically:

* /pubkey returns a real PEM.
* /verify round-trips its own server-signed answers.
* /query rejects unsigned, mis-signed, and unauthorised clients.
* /query succeeds for an allow-listed client and returns a verifiable
  signature.
"""

from __future__ import annotations

import base64
import unittest

from fastapi.testclient import TestClient

from rag_sign.corpus import chunk_documents
from rag_sign.embeddings import HashEmbedder
from rag_sign.hsm import InMemoryHSM
from rag_sign.llm import EchoLLM
from rag_sign.rag import RagSignSystem
from rag_sign.server import build_app
from rag_sign.signer import RagSigner, verify
from rag_sign.vector_db import ChromaVectorDB

CORPUS_SIZE = 200
DOCS = [
    (f"doc-{i}.txt", f"Document {i} discusses Schnorr signatures over G_{i}.")
    for i in range(CORPUS_SIZE)
]
MODEL_HASH = b"\xab" * 32

_seq = 0


def _unique_collection() -> str:
    global _seq
    _seq += 1
    return f"server-test-{_seq:04d}"


def _build_ready_system() -> RagSignSystem:
    sys = RagSignSystem(
        vector_db=ChromaVectorDB(
            embedder=HashEmbedder(dim=128),
            collection_name=_unique_collection(),
        ),
        llm=EchoLLM(),
        hsm=InMemoryHSM(),
        top_k=3,
    )
    sys.enrol(chunk_documents(DOCS), model_hash=MODEL_HASH)
    return sys


def _make_client_keypair() -> tuple[RagSigner, bytes]:
    """A *client* keypair — same primitive (ECDSA P-256), random seed."""
    import os

    signer = RagSigner(os.urandom(32))
    return signer, signer.public_key_pem


def _sign_b64(signer: RagSigner, payload: bytes) -> str:
    return base64.b64encode(signer.sign(payload).signature).decode("ascii")


# ---------------------------------------------------------------------------
# /pubkey, /health, /verify
# ---------------------------------------------------------------------------


class PublicEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = _build_ready_system()
        self.app = build_app(self.system, client_allowlist_pems=[])
        self.client = TestClient(self.app)

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        self.assertTrue(r.json()["ready"])

    def test_pubkey_is_pem(self) -> None:
        r = self.client.get("/pubkey")
        self.assertEqual(r.status_code, 200)
        pem = r.json()["public_key_pem"]
        self.assertIn("BEGIN PUBLIC KEY", pem)

    def test_verify_round_trip(self) -> None:
        # Get a signed answer out of band, then verify via the endpoint.
        msg = self.system.query("anything")
        r = self.client.post(
            "/verify",
            json={
                "payload_utf8": msg.payload.decode("utf-8"),
                "signature_b64": base64.b64encode(msg.signature).decode("ascii"),
                "public_key_pem": msg.public_key_pem.decode("ascii"),
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["valid"])

    def test_verify_rejects_tampered_payload(self) -> None:
        msg = self.system.query("anything")
        r = self.client.post(
            "/verify",
            json={
                "payload_utf8": "tampered",
                "signature_b64": base64.b64encode(msg.signature).decode("ascii"),
                "public_key_pem": msg.public_key_pem.decode("ascii"),
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["valid"])


# ---------------------------------------------------------------------------
# /query gating
# ---------------------------------------------------------------------------


class QueryGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.system = _build_ready_system()
        self.client_signer, self.client_pem = _make_client_keypair()
        self.app = build_app(
            self.system,
            client_allowlist_pems=[self.client_pem.decode("utf-8")],
        )
        self.http = TestClient(self.app)

    def _query_body(
        self, *, question: str, signer: RagSigner, pem: bytes
    ) -> dict[str, str]:
        return {
            "question": question,
            "client_public_key_pem": pem.decode("utf-8"),
            "client_signature_b64": _sign_b64(signer, question.encode("utf-8")),
        }

    def test_allowlisted_client_succeeds(self) -> None:
        body = self._query_body(
            question="Tell me about Schnorr",
            signer=self.client_signer,
            pem=self.client_pem,
        )
        r = self.http.post("/query", json=body)
        self.assertEqual(r.status_code, 200, msg=r.text)

        # Server's signature over its own answer must verify.
        data = r.json()
        ok = verify(
            data["answer"].encode("utf-8"),
            base64.b64decode(data["signature_b64"]),
            data["public_key_pem"].encode("utf-8"),
        )
        self.assertTrue(ok)

    def test_non_allowlisted_client_is_rejected_403(self) -> None:
        rogue, rogue_pem = _make_client_keypair()
        body = self._query_body(question="anything", signer=rogue, pem=rogue_pem)
        r = self.http.post("/query", json=body)
        self.assertEqual(r.status_code, 403)

    def test_bad_signature_is_rejected_401(self) -> None:
        body = self._query_body(
            question="anything",
            signer=self.client_signer,
            pem=self.client_pem,
        )
        # Flip the signature so it won't verify under the right pubkey.
        bad = bytearray(base64.b64decode(body["client_signature_b64"]))
        bad[-1] ^= 1
        body["client_signature_b64"] = base64.b64encode(bytes(bad)).decode("ascii")
        r = self.http.post("/query", json=body)
        self.assertEqual(r.status_code, 401)

    def test_malformed_signature_is_rejected_400(self) -> None:
        body = self._query_body(
            question="anything",
            signer=self.client_signer,
            pem=self.client_pem,
        )
        body["client_signature_b64"] = "not-base64-!!!"
        r = self.http.post("/query", json=body)
        self.assertEqual(r.status_code, 400)


class BuildAppGuardTests(unittest.TestCase):
    def test_unenrolled_system_rejected(self) -> None:
        sys = RagSignSystem(
            vector_db=ChromaVectorDB(
                embedder=HashEmbedder(dim=128),
                collection_name=_unique_collection(),
            ),
            llm=EchoLLM(),
            hsm=InMemoryHSM(),
        )
        with self.assertRaisesRegex(RuntimeError, "must be enrolled"):
            build_app(sys)


if __name__ == "__main__":
    unittest.main()
