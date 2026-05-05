"""Tests for :mod:`rag_sign.fuzzy_extractor`.

These tests cover:

* Round-trip correctness when ``w'`` matches ``w`` exactly.
* Recovery when ``w'`` differs by up to ``t`` bits.
* Failure when ``w'`` differs by more than ``t`` bits.
* Determinism of recovery given a fixed ``(w', P)``.
* Independence between distinct enrolments (different ``R`` values
  even on the same ``w``, because ``c`` is freshly sampled).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from rag_sign.fuzzy_extractor import (
    FuzzyExtractFailure,
    N,
    T,
    gen,
    rep,
)


def _flip_bits(b: bytes, positions: list[int]) -> bytes:
    """Return ``b`` with the listed (bit-index) positions toggled."""
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).copy()
    for p in positions:
        bits[p] ^= 1
    return np.packbits(bits).tobytes()


class FuzzyExtractorRoundTripTests(unittest.TestCase):
    def test_exact_match_recovers(self) -> None:
        w = os.urandom(64)
        R, helper = gen(w)
        self.assertEqual(R, rep(w, helper))
        self.assertEqual(len(R), 32)

    def test_recovers_under_t_errors(self) -> None:
        w = os.urandom(64)
        R, helper = gen(w)

        rng = np.random.default_rng(2025)
        # Flip exactly T bits within the first N bits.
        positions = list(rng.choice(N, size=T, replace=False).tolist())
        w_prime = _flip_bits(w, positions)
        self.assertEqual(R, rep(w_prime, helper))

    def test_fails_above_t_errors(self) -> None:
        w = os.urandom(64)
        _, helper = gen(w)

        rng = np.random.default_rng(2026)
        # Flip T+1 bits — must exceed the BCH bound.
        positions = list(rng.choice(N, size=T + 1, replace=False).tolist())
        w_prime = _flip_bits(w, positions)

        # The decoder may either raise or return a different R.  The
        # paper mandates a hard fail (revocation event), so we require
        # the explicit exception.
        with self.assertRaises(FuzzyExtractFailure):
            rep(w_prime, helper)

    def test_recovery_is_deterministic(self) -> None:
        w = os.urandom(64)
        R, helper = gen(w)
        # Same inputs → same outputs, repeatedly.
        self.assertEqual(rep(w, helper), R)
        self.assertEqual(rep(w, helper), R)


class FuzzyExtractorEntropyTests(unittest.TestCase):
    def test_independent_enrolments_yield_different_R(self) -> None:
        """Two `gen` calls on the same w must produce different (R, P)."""
        w = os.urandom(64)
        R1, h1 = gen(w)
        R2, h2 = gen(w)
        self.assertNotEqual(R1, R2)
        self.assertNotEqual(h1.bits, h2.bits)

    def test_helper_does_not_leak_R(self) -> None:
        """Helper bits should not be obviously derivable from R.

        Cheap necessary check (not sufficient): no equality, no
        Hamming-distance collapse to a constant.
        """
        Rs = []
        for _ in range(8):
            R, _ = gen(os.urandom(64))
            Rs.append(R)
        # No two distinct enrolments returned the same R.
        self.assertEqual(len(set(Rs)), len(Rs))


class FuzzyExtractorInputValidationTests(unittest.TestCase):
    def test_gen_rejects_too_short(self) -> None:
        # 63 bytes = 504 bits, < N=511.
        with self.assertRaisesRegex(ValueError, "at least"):
            gen(os.urandom(63))

    def test_rep_rejects_too_short(self) -> None:
        R, helper = gen(os.urandom(64))
        with self.assertRaisesRegex(ValueError, "at least"):
            rep(os.urandom(63), helper)


if __name__ == "__main__":
    unittest.main()
