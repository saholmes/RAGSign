# RAGSign

Reference implementation of the *RAG Sign* scheme described in:

> Holmes, S. A. **"Cryptographic Authentication for Private RAG-Enabled
> LLMs."** AISecPriv 2025.

RAG Sign binds every output of a Retrieval-Augmented-Generation system
to a tuple of *(model identity, RAG corpus, HSM secret)*. A
deterministic ECDSA signing key is derived from that tuple, so the same
*(model, corpus, secret)* always yields the same long-term public key —
and any output bearing a valid signature is provably attributable to
that exact configuration. Combined with a public-key allow-list, the
same signing key gates who is permitted to query the corpus, and a
revoked public key acts as a global *kill-switch* for a rogue
deployment.

Author: Stephen A. Holmes — `s.a.holmes@surrey.ac.uk`

## Status

End-to-end pipeline runs under deterministic mock backends; real
`bge-large` embeddings, real Llama 3.2 GGUF, and real SoftHSM are
opt-in via the relevant pyproject extras.

| Module | What it does |
|---|---|
| `rag_sign.key_derivation` | Algorithm 1 — `sk_seed = SHA3-256(domain ‖ model_hash ‖ lsh_key ‖ hsm_secret)` with length-prefixed domain separation |
| `rag_sign.signer` / `rag_sign.verifier` | ECDSA P-256 sign / verify, seeded by Algorithm 1 |
| `rag_sign.fuzzy_extractor` | Code-offset fuzzy extractor over **BCH(511, 259, t=30)** — 5.87 % drift tolerance, comfortably above paper §6.5's 5 % target |
| `rag_sign.lsh` | MinHash sketch (`num_perm=512`) → 1-bit-per-slot fingerprint preserving Hamming proximity |
| `rag_sign.hsm` | Protocol-typed backends — `InMemoryHSM` (tests), `SoftHSMBackend` (PKCS#11 via `$RAG_SIGN_HSM_*`) |
| `rag_sign.corpus` / `embeddings` / `vector_db` / `llm` | Pluggable RAG primitives; ChromaDB store, Llama 3.2 GGUF backend, deterministic `HashEmbedder` / `EchoLLM` for tests |
| `rag_sign.rag` | Two-phase orchestrator — `enrol(chunks, model_hash) → EnrolmentBundle`, `recover(chunks, bundle)`, `query(question) → SignedMessage` |
| `rag_sign.server` | FastAPI surface — `/query`, `/verify`, `/pubkey`, `/health`; client allow-list gating with ECDSA-signed queries |
| `benchmarks/` | §6.6 reproductions (`bench_keygen`, `bench_sign_verify`, `bench_overhead`) emitting JSON to `bench_results/` |

## Quick start

```bash
# 1. Set up Python and the venv (uv handles both)
brew install uv
uv venv --python 3.12

# 2. Install with dev extras
uv pip install -e ".[dev]"

# 3. Run the test suite (61 tests, ~12 s)
.venv/bin/python -m pytest -v

# 4. Reproduce the §6.6 benchmarks
.venv/bin/python -m benchmarks.bench_keygen
.venv/bin/python -m benchmarks.bench_sign_verify
.venv/bin/python -m benchmarks.bench_overhead
```

For HSM / LLM / embedding work, add the relevant extra:
`uv pip install -e ".[dev,hsm,llm,embeddings]"`.

## Layout

```
RAGSign/
├── src/rag_sign/         # library code
├── tests/                # unit tests (pytest)
├── benchmarks/           # §6.6 reproductions (keygen / sign / verify / overhead)
├── scripts/              # SoftHSM setup, IACR corpus loader
├── examples/             # minimal end-to-end demos
└── docs/                 # design notes (paper-implementation deltas)
```

## License

MIT — see `LICENSE` (to be added).
