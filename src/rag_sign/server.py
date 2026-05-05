"""FastAPI front-end for the RAG-Sign system.

Endpoints
---------

* ``POST /query``      — submit a question, receive a signed answer.
* ``POST /verify``     — verify a (payload, signature, public_key) tuple.
* ``GET  /pubkey``     — fetch the deployment's long-term public key.
* ``GET  /health``     — liveness / readiness probe.

The server enforces the paper's *allow-list gating* (kill-switch
contemplated in the README): every request to ``/query`` must be
signed by a client whose public key is on the configured allow-list
(in PEM form).  Allow-list membership both authorises the query and
provides the linkage that lets a compromised client be revoked
globally — pull its key from the list and every server in the
deployment refuses subsequent queries.

The server does **not** know how to enrol or recover by itself.
Construct one with an already-ready :class:`rag_sign.rag.RagSignSystem`,
typically via :func:`build_app_from_env` in production deployments.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_sign.rag import RagSignSystem
from rag_sign.signer import verify as ecdsa_verify

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Body of a ``/query`` request.

    The client signs the question (over UTF-8 bytes) with its own
    long-term ECDSA P-256 keypair; the public key is supplied in PEM
    form so the server can match it against the allow-list.  Signature
    is base64.
    """

    question: str = Field(..., description="natural-language query")
    client_public_key_pem: str = Field(..., description="PEM-encoded client pubkey")
    client_signature_b64: str = Field(..., description="base64 ECDSA over UTF-8 question")


class SignedReply(BaseModel):
    """Body of a ``/query`` response."""

    answer: str = Field(..., description="LLM output")
    signature_b64: str = Field(..., description="base64 ECDSA over UTF-8 answer")
    public_key_pem: str = Field(..., description="server's long-term pubkey, PEM")


class VerifyRequest(BaseModel):
    """Body of a ``/verify`` request — verifies any signed payload."""

    payload_utf8: str
    signature_b64: str
    public_key_pem: str


class VerifyReply(BaseModel):
    valid: bool


class PubKeyReply(BaseModel):
    public_key_pem: str


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def build_app(
    system: RagSignSystem,
    *,
    client_allowlist_pems: Iterable[str] = (),
) -> FastAPI:
    """Wire a :class:`RagSignSystem` and an allow-list into a FastAPI app.

    The allow-list is captured by value at construction time; rotating
    a key requires restarting the process (good — explicit,
    auditable).  An empty list means *no* clients are allowed — the
    deployment refuses every ``/query``, which is the safe default for
    a misconfigured server.
    """
    if not system.is_ready:
        raise RuntimeError("RagSignSystem must be enrolled / recovered before serving")

    allowed_pems: frozenset[bytes] = frozenset(
        pem.encode("utf-8") for pem in client_allowlist_pems
    )

    app = FastAPI(
        title="RAG-Sign",
        description=(
            "Cryptographic authentication for RAG-enabled LLMs.  Every "
            "answer is signed with a key derived from (model, corpus, "
            "HSM secret); the public key is the deployment's long-term "
            "identity."
        ),
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True, "ready": system.is_ready}

    @app.get("/pubkey", response_model=PubKeyReply)
    async def pubkey() -> PubKeyReply:
        return PubKeyReply(public_key_pem=system.public_key_pem.decode("ascii"))

    @app.post("/query", response_model=SignedReply)
    async def query(req: QueryRequest, _: Request) -> SignedReply:
        # 1. Allow-list check.
        client_pem = req.client_public_key_pem.encode("utf-8")
        if client_pem not in allowed_pems:
            raise HTTPException(
                status_code=403,
                detail="client public key is not on the allow-list",
            )

        # 2. Verify the client's signature over the question — the
        #    same kill-switch that governs RAG access also governs
        #    inbound query authenticity.
        try:
            client_sig = base64.b64decode(req.client_signature_b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise HTTPException(
                status_code=400, detail=f"malformed signature: {exc}"
            ) from exc
        if not ecdsa_verify(req.question.encode("utf-8"), client_sig, client_pem):
            raise HTTPException(
                status_code=401, detail="client signature did not verify"
            )

        # 3. Run the RAG-Sign pipeline.
        signed = system.query(req.question)
        return SignedReply(
            answer=signed.payload.decode("utf-8", errors="replace"),
            signature_b64=base64.b64encode(signed.signature).decode("ascii"),
            public_key_pem=signed.public_key_pem.decode("ascii"),
        )

    @app.post("/verify", response_model=VerifyReply)
    async def verify_endpoint(req: VerifyRequest) -> VerifyReply:
        try:
            sig = base64.b64decode(req.signature_b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise HTTPException(
                status_code=400, detail=f"malformed signature: {exc}"
            ) from exc
        try:
            ok = ecdsa_verify(
                req.payload_utf8.encode("utf-8"),
                sig,
                req.public_key_pem.encode("utf-8"),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=400, detail=f"verification failed: {exc}"
            ) from exc
        return VerifyReply(valid=ok)

    return app


# ---------------------------------------------------------------------------
# Production entry point — env-configured
# ---------------------------------------------------------------------------


def _load_allowlist(path: str | Path) -> list[str]:
    """Read a JSON list of PEM strings from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"allow-list file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise ValueError(f"allow-list must be a JSON array of PEM strings: {p}")
    return raw


def main() -> None:  # pragma: no cover — production entry point
    """``rag-sign-server`` — production CLI launcher.

    Reads:

    * ``RAG_SIGN_ALLOWLIST_PATH`` — JSON file with allowed client PEMs
    * ``RAG_SIGN_CORPUS_PATH``    — directory of text files to ingest
    * ``RAG_SIGN_BUNDLE_PATH``    — enrolment bundle (re-derives key)
    * ``RAG_SIGN_MODEL_PATH``     — GGUF Llama model
    * ``RAG_SIGN_HSM_*``          — see :mod:`rag_sign.hsm`

    Wire-up of these env vars into a fully-configured RagSignSystem is
    a deployment concern; this stub exists to document the contract.
    """
    raise NotImplementedError(
        "production launcher not yet wired — see docstring for required env vars"
    )
