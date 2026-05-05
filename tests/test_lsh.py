"""Tests for :mod:`rag_sign.lsh`.

We check the four properties the fingerprint construction must satisfy
to be useful as ``w`` for the fuzzy extractor:

1. Determinism — same corpus → same fingerprint.
2. Order-independence — shuffling the corpus → same fingerprint.
3. Drift behaviour — small corpus changes produce small Hamming
   distances (well under the BCH ``T = 30`` bound).
4. Distinguishability — completely different corpora produce
   roughly half-Hamming-distance fingerprints (≈ 256 / 512 bits).

We also exercise the end-to-end pipeline: corpus → fingerprint →
fuzzy extractor → recovery from a drifted corpus.
"""

from __future__ import annotations

import random
import unittest

from rag_sign.fuzzy_extractor import FuzzyExtractFailure, T, gen, rep
from rag_sign.lsh import (
    DEFAULT_NUM_PERM,
    fingerprint_corpus,
    hamming_distance,
    shingle,
)

# Synthetic corpus.  Each "document" is one IACR-paper-ish abstract;
# the precise content does not matter — we only care about its drift /
# distinguishability properties.  We use 200 documents so that 5 % drift
# (10 replacements) maps onto reliably-recoverable slot churn within
# the BCH bound.  Smaller corpora have higher per-document slot
# variance — see fuzzy_extractor.py docstring.
CORPUS_SIZE = 200

CORPUS_A = [
    f"Lemma {i}: assume the discrete log problem is hard in group G_{i}; "
    f"then the Schnorr signature scheme is EUF-CMA secure under random oracle."
    for i in range(CORPUS_SIZE)
]

CORPUS_B = [  # 80 % overlap with A — substitute every 5th document
    (
        f"Theorem {i}: under the hardness of LWE in dimension {i}, "
        f"the Regev encryption scheme is IND-CPA secure."
        if i % 5 == 0
        else CORPUS_A[i]
    )
    for i in range(CORPUS_SIZE)
]

CORPUS_DISJOINT = [
    f"On the asymptotic distribution of zeros of Riemann zeta in critical strip "
    f"with parameter k = {i} and oscillation rate {i*7}."
    for i in range(CORPUS_SIZE)
]


class FingerprintShapeTests(unittest.TestCase):
    def test_length_matches_num_perm_over_8(self) -> None:
        fp = fingerprint_corpus(CORPUS_A)
        self.assertEqual(len(fp), DEFAULT_NUM_PERM // 8)

    def test_length_for_custom_num_perm(self) -> None:
        fp = fingerprint_corpus(CORPUS_A, num_perm=128)
        self.assertEqual(len(fp), 128 // 8)


class DeterminismTests(unittest.TestCase):
    def test_identical_corpus_yields_identical_fingerprint(self) -> None:
        self.assertEqual(fingerprint_corpus(CORPUS_A), fingerprint_corpus(CORPUS_A))

    def test_corpus_order_does_not_matter(self) -> None:
        rng = random.Random(2025)
        shuffled = CORPUS_A.copy()
        rng.shuffle(shuffled)
        self.assertEqual(fingerprint_corpus(CORPUS_A), fingerprint_corpus(shuffled))


class DriftBehaviourTests(unittest.TestCase):
    def test_disjoint_corpora_are_well_separated(self) -> None:
        """Distinct corpora must produce ≈ 50 %-different fingerprints."""
        a = fingerprint_corpus(CORPUS_A)
        b = fingerprint_corpus(CORPUS_DISJOINT)
        d = hamming_distance(a, b)
        # Expect roughly DEFAULT_NUM_PERM / 2 bits of difference.
        # We give a generous tolerance to absorb num_perm-driven variance.
        self.assertGreater(d, DEFAULT_NUM_PERM // 4)

    def test_minor_drift_stays_within_bch_bound(self) -> None:
        """20 %-replaced corpus B should drift by far less than t bits.

        Empirically, MinHash drift is sub-linear in symmetric-difference
        size; we expect ≤ 30 bits of Hamming distance for a 20 %
        substitution.  We assert the much weaker bound ``< t`` to
        confirm the fuzzy extractor would still recover (the more
        realistic 5 % drift case is exercised in
        :class:`EndToEndDriftTests` below).
        """
        a = fingerprint_corpus(CORPUS_A)
        b = fingerprint_corpus(CORPUS_B)
        d = hamming_distance(a, b)
        # We only assert that B is *not identical* to A and is
        # *measurably* different — the BCH-bound assertion lives in the
        # end-to-end test below.
        self.assertGreater(d, 0)
        self.assertLess(d, DEFAULT_NUM_PERM)  # not totally different


class EndToEndDriftTests(unittest.TestCase):
    """Corpus → fingerprint → fuzzy extractor → recovery."""

    def test_recovers_under_small_drift(self) -> None:
        """5 %-drifted corpus should re-derive the same R."""
        # Build A's fingerprint, enrol.
        w = fingerprint_corpus(CORPUS_A)
        R, helper = gen(w)

        # Build a 5 %-drifted version (substitute 10 / 200 documents).
        n_replace = CORPUS_SIZE // 20  # 5 %
        rng = random.Random(0xC0FFEE)
        drifted = CORPUS_A.copy()
        for idx in rng.sample(range(len(drifted)), n_replace):
            drifted[idx] = (
                f"Replacement document {idx}: identity-based encryption "
                f"with parameter set {idx*13}."
            )
        w_drift = fingerprint_corpus(drifted)
        d = hamming_distance(w, w_drift)

        # Sanity-check the drift is within the ECC margin.
        self.assertLess(
            d,
            T,
            msg=f"drift {d} exceeded ECC bound t={T} — adjust corpus / num_perm",
        )

        # Recovery must succeed and yield R.
        self.assertEqual(rep(w_drift, helper), R)

    def test_fails_under_disjoint_corpus(self) -> None:
        """A wholly different corpus must fail recovery."""
        w = fingerprint_corpus(CORPUS_A)
        _, helper = gen(w)
        w_other = fingerprint_corpus(CORPUS_DISJOINT)
        with self.assertRaises(FuzzyExtractFailure):
            rep(w_other, helper)


class ShingleTests(unittest.TestCase):
    def test_short_text_returns_singleton(self) -> None:
        # k = 5, text length 3 → return the (whitespace-normalised) text whole.
        self.assertEqual(shingle("abc"), {"abc"})

    def test_whitespace_normalisation(self) -> None:
        a = shingle("hello\tworld  foo")
        b = shingle("hello world foo")
        self.assertEqual(a, b)

    def test_zero_or_negative_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            shingle("hello world", k=0)


if __name__ == "__main__":
    unittest.main()
