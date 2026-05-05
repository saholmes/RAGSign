"""Bench: full key-derivation pipeline (Algorithm 1).

Paper §6.6 reports key generation at ≈ 10 s per derivation.  Most of
that budget is the *fuzzy extractor enrolment* over a real corpus
fingerprint, not the SHA3-256 + ECDSA bit at the end.  We split the
two so the contribution of each is visible.

Usage:

    .venv/bin/python -m benchmarks.bench_keygen
"""

from __future__ import annotations

import os

from benchmarks._common import print_result, time_callable, write_result
from rag_sign.fuzzy_extractor import gen as fe_gen
from rag_sign.key_derivation import KeyMaterial, derive_signing_seed
from rag_sign.signer import RagSigner


def _bench_fe_gen() -> None:
    fe_gen(os.urandom(64))


def _bench_full_keygen() -> None:
    # 1. Fuzzy-extractor enrolment.
    r, _helper = fe_gen(os.urandom(64))
    # 2. Algorithm 1 + ECDSA derive.
    seed = derive_signing_seed(
        KeyMaterial(model_hash=os.urandom(32), lsh_key=r, hsm_secret=os.urandom(32))
    )
    RagSigner(seed)


def main() -> None:
    for name, fn in [
        ("keygen_fe_gen_only", _bench_fe_gen),
        ("keygen_full",        _bench_full_keygen),
    ]:
        result = time_callable(name, fn, n=200, warmup=5)
        print_result(result)
        write_result(result)


if __name__ == "__main__":
    main()
