"""RAG Sign — cryptographic authentication for RAG-enabled LLMs.

Reference implementation of the scheme described in:
    Holmes, S. A. "Cryptographic Authentication for Private RAG-Enabled LLMs."
    AISecPriv 2025.

The package binds an LLM output to a tuple of (model_hash, corpus_lsh_key,
hsm_secret) via SHA3-256 derivation, producing a deterministic ECDSA signing
key that can be recovered as long as the corpus drift stays below the
fuzzy-extractor threshold.
"""

from rag_sign.fuzzy_extractor import (
    FuzzyExtractFailure,
    HelperData,
)
from rag_sign.fuzzy_extractor import (
    gen as fe_gen,
)
from rag_sign.fuzzy_extractor import (
    rep as fe_rep,
)
from rag_sign.key_derivation import (
    KeyMaterial,
    derive_signing_seed,
    fingerprint_model,
)
from rag_sign.lsh import (
    fingerprint_corpus,
    minhash_corpus,
    minhash_document,
)
from rag_sign.signer import RagSigner, SignedMessage
from rag_sign.verifier import verify, verify_message

__version__ = "0.1.0"

__all__ = [
    "FuzzyExtractFailure",
    "HelperData",
    "KeyMaterial",
    "RagSigner",
    "SignedMessage",
    "__version__",
    "derive_signing_seed",
    "fe_gen",
    "fe_rep",
    "fingerprint_corpus",
    "fingerprint_model",
    "minhash_corpus",
    "minhash_document",
    "verify",
    "verify_message",
]
