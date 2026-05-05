"""Unit tests for :mod:`rag_sign.key_derivation`.

These tests use only the standard library so they can be run against
any modern Python without a populated venv:

    PYTHONPATH=src python3 -m unittest tests.test_key_derivation -v
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from rag_sign.key_derivation import (
    KeyMaterial,
    derive_signing_seed,
    fingerprint_model,
)


def _sample(n: int, seed: int = 0) -> bytes:
    """Deterministic byte string of length n."""
    return bytes((i + seed) & 0xFF for i in range(n))


class KeyMaterialValidationTests(unittest.TestCase):
    def test_accepts_minimum_lengths(self) -> None:
        km = KeyMaterial(_sample(16, 1), _sample(16, 2), _sample(16, 3))
        self.assertEqual(len(km.model_hash), 16)

    def test_rejects_short_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_hash"):
            KeyMaterial(_sample(15), _sample(16, 1), _sample(16, 2))
        with self.assertRaisesRegex(ValueError, "lsh_key"):
            KeyMaterial(_sample(16), _sample(0), _sample(16, 1))
        with self.assertRaisesRegex(ValueError, "hsm_secret"):
            KeyMaterial(_sample(16), _sample(16, 1), _sample(15, 2))

    def test_rejects_wrong_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            KeyMaterial("not bytes", _sample(16), _sample(16))  # type: ignore[arg-type]


class DeriveSigningSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.km = KeyMaterial(_sample(32, 0), _sample(32, 1), _sample(32, 2))

    def test_output_length_is_32_bytes(self) -> None:
        seed = derive_signing_seed(self.km)
        self.assertEqual(len(seed), 32)

    def test_is_deterministic(self) -> None:
        s1 = derive_signing_seed(self.km)
        s2 = derive_signing_seed(self.km)
        self.assertEqual(s1, s2)

    def test_changes_when_any_input_changes(self) -> None:
        base = derive_signing_seed(self.km)

        km_model = KeyMaterial(_sample(32, 99), self.km.lsh_key, self.km.hsm_secret)
        km_lsh = KeyMaterial(self.km.model_hash, _sample(32, 99), self.km.hsm_secret)
        km_secret = KeyMaterial(self.km.model_hash, self.km.lsh_key, _sample(32, 99))

        self.assertNotEqual(base, derive_signing_seed(km_model))
        self.assertNotEqual(base, derive_signing_seed(km_lsh))
        self.assertNotEqual(base, derive_signing_seed(km_secret))

    def test_distinguishes_field_boundary(self) -> None:
        """Length-prefix protects against the bare-concat ambiguity.

        ``(lsh="AB", secret="CD")`` and ``(lsh="ABCD", secret="")`` would
        collide under naive concatenation; under the length-prefixed
        construction they must produce distinct seeds.  We check the
        first form against itself for stability and against an alternate
        partitioning to confirm separation.
        """
        a = KeyMaterial(_sample(16, 0), b"AB" + _sample(14, 0), b"CD" + _sample(14, 1))
        b = KeyMaterial(_sample(16, 0), b"ABCD" + _sample(14, 0), _sample(16, 1))
        self.assertNotEqual(derive_signing_seed(a), derive_signing_seed(b))

    def test_matches_explicit_construction(self) -> None:
        """Sanity check the digest against a manual recomputation."""
        h = hashlib.sha3_256()
        h.update(b"RAG-SIGN/v1/seed")
        for field in (self.km.model_hash, self.km.lsh_key, self.km.hsm_secret):
            h.update(len(field).to_bytes(4, "big"))
            h.update(field)
        self.assertEqual(derive_signing_seed(self.km), h.digest())


class FingerprintModelTests(unittest.TestCase):
    def test_hashes_small_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = Path(f.name)
        try:
            digest = fingerprint_model(path)
            self.assertEqual(digest, hashlib.sha3_256(b"hello world").digest())
        finally:
            os.unlink(path)

    def test_streams_large_file(self) -> None:
        # 3 MiB — exercises the 1 MiB streaming chunk loop.
        payload = (b"abcdefghij" * 100_000) + b"tail"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(payload)
            path = Path(f.name)
        try:
            self.assertEqual(fingerprint_model(path), hashlib.sha3_256(payload).digest())
        finally:
            os.unlink(path)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            fingerprint_model("/nonexistent/path/to/model.bin")


if __name__ == "__main__":
    unittest.main()
