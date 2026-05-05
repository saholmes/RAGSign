"""Bench: ECDSA P-256 sign / verify on signed-message-sized payloads.

Paper §6.6 reports sign ≈ 30 ms, verify ≈ 20 ms on the paper hardware.
We expect Apple-Silicon / x86-server numbers to be ≈ 0.1 / 0.05 ms,
i.e. orders of magnitude under the paper's constants — the relevant
takeaway is *headroom* and that the asymptotic shape (sign ≈
1.5 × verify) holds.

Usage:

    .venv/bin/python -m benchmarks.bench_sign_verify
"""

from __future__ import annotations

import os

from benchmarks._common import print_result, time_callable, write_result
from rag_sign.signer import RagSigner, verify

# Roughly LLM-output-sized — the paper's evaluation queries.
PAYLOAD = (
    b"The Schnorr signature scheme is provably secure under the "
    b"discrete-logarithm assumption in the random-oracle model.  "
    b"It produces signatures of size 2|q|, where q is the group "
    b"order, and verification cost is dominated by two scalar "
    b"multiplications in the underlying group."
) * 4  # ≈ 1 KiB


_signer = RagSigner(os.urandom(32))
_signed = _signer.sign(PAYLOAD)


def _bench_sign() -> None:
    _signer.sign(PAYLOAD)


def _bench_verify() -> None:
    verify(_signed.payload, _signed.signature, _signed.public_key_pem)


def main() -> None:
    for name, fn in [
        ("ecdsa_sign",   _bench_sign),
        ("ecdsa_verify", _bench_verify),
    ]:
        result = time_callable(name, fn, n=2000, warmup=20)
        print_result(result)
        write_result(result)


if __name__ == "__main__":
    main()
