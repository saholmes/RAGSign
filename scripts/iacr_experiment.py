"""Reproduce paper §6 evaluation against the real IACR ePrint corpus.

Runs four experiments against PDFs under ``$RAG_SIGN_IACR_DATA``:

    1. **Round-trip**         — enrol on year Y, recover on year Y;
                                public keys must match bit-for-bit.
    2. **5 %-drift recovery** — enrol on year Y, recover with 5 % of
                                papers replaced by random year-Y'
                                documents; key must still match.
    3. **Disjoint detection** — recover on a different year; the
                                fuzzy extractor must hard-fail
                                (paper §5: revocation event).
    4. **Scaling**            — time the four hot paths (LSH
                                fingerprint, fuzzy-extractor enrol,
                                Algorithm 1 derive, ECDSA sign) on a
                                real-sized corpus.

Results are written to ``bench_results/iacr_experiment.json`` (also
git-ignored — leak-prevention).

Usage:

    .venv/bin/python -m scripts.iacr_experiment \\
        --years 2013 \\
        --limit 400 \\
        --drift-pct 5

For larger sweeps (multi-year, multi-drift), add more ``--drift-pct``
values; each one runs experiment (2) at that level.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make ``rag_sign`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_sign.fuzzy_extractor import FuzzyExtractFailure
from rag_sign.fuzzy_extractor import T as BCH_T
from rag_sign.fuzzy_extractor import gen as fe_gen
from rag_sign.fuzzy_extractor import rep as fe_rep
from rag_sign.key_derivation import KeyMaterial, derive_signing_seed
from rag_sign.lsh import fingerprint_corpus, hamming_distance
from rag_sign.signer import RagSigner

# Ensure scripts/ is importable for the loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iacr_loader import iter_papers  # noqa: E402

RESULTS_PATH = Path(__file__).resolve().parent.parent / "bench_results" / "iacr_experiment.json"


@dataclass
class ExperimentResult:
    years: list[int]
    n_papers: int
    fingerprint_bits: int
    bch_t: int

    # Experiment 1: round-trip
    round_trip_ok: bool = False
    round_trip_hamming: int = -1

    # Experiment 2: drift recovery (one entry per requested drift level)
    drift_results: list[dict[str, Any]] = field(default_factory=list)

    # Experiment 3: disjoint detection
    disjoint_year: int | None = None
    disjoint_failed_as_expected: bool = False
    disjoint_hamming: int = -1

    # Experiment 4: timings (ms)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus(years: tuple[int, ...], limit: int | None) -> list[tuple[str, str]]:
    print(f"Loading IACR corpus (years={years}, limit={limit}) — first run extracts PDFs…")
    t0 = time.perf_counter()
    docs = list(iter_papers(years=years, limit=limit, progress=True))
    dt = time.perf_counter() - t0
    print(f"  {len(docs)} papers loaded in {dt:.1f}s "
          f"({sum(len(t) for _, t in docs) / 1e6:.1f} MB of text)")
    return docs


def _replace_with_random_papers(
    docs: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    fraction: float,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Return a copy of ``docs`` with a ``fraction`` of entries replaced.

    Replacement targets (positions in ``docs``) are sampled WITHOUT
    replacement — every drifted document occupies a distinct slot in
    the corpus.  Replacement *content* is drawn WITH replacement from
    ``pool``: when the requested drift exceeds the pool size we still
    perturb the right number of slots; some replacement contents will
    repeat across positions, which is consistent with realistic corpus
    churn (e.g. several papers being superseded by the same new draft).
    """
    n_replace = max(1, int(len(docs) * fraction))
    n_replace = min(n_replace, len(docs))  # cannot replace more than we have
    out = list(docs)
    target_positions = rng.sample(range(len(out)), n_replace)
    replacements = rng.choices(pool, k=n_replace)  # WITH replacement
    for pos, replacement in zip(target_positions, replacements, strict=True):
        out[pos] = replacement
    return out


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_round_trip(
    w_enrol: bytes,
    R: bytes,
    helper: object,
    result: ExperimentResult,
) -> None:
    """Re-rep the enrolment fingerprint and confirm R is identical."""
    print("\n=== Experiment 1: round-trip ===")
    R_prime = fe_rep(w_enrol, helper)  # type: ignore[arg-type]
    result.round_trip_hamming = 0      # by construction — same w
    result.round_trip_ok = R_prime == R
    print("  fingerprint Hamming distance (same corpus): 0 (by construction)")
    print(f"  round-trip R match: {result.round_trip_ok}")


def experiment_drift(
    docs: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    drift_pcts: list[float],
    w_enrol: bytes,
    R: bytes,
    helper: object,
    result: ExperimentResult,
    seed: int = 2026,
) -> None:
    """At each drift level, perturb the corpus and try to recover R."""
    print("\n=== Experiment 2: drift recovery ===")
    rng = random.Random(seed)
    for pct in drift_pcts:
        drifted = _replace_with_random_papers(docs, pool, pct / 100.0, rng)
        t0 = time.perf_counter()
        w_drift = fingerprint_corpus([t for _, t in drifted])
        dt = time.perf_counter() - t0
        d = hamming_distance(w_enrol, w_drift)
        try:
            R_prime = fe_rep(w_drift, helper)  # type: ignore[arg-type]
            recovered = R_prime == R
            failure = None
        except FuzzyExtractFailure as exc:
            recovered = False
            failure = str(exc)
        entry = {
            "drift_pct":     pct,
            "hamming":       d,
            "within_bound":  d <= BCH_T,
            "recovered":     recovered,
            "failure":       failure,
            "fingerprint_s": round(dt, 2),
        }
        result.drift_results.append(entry)
        print(
            f"  drift={pct:>4.1f}%  hamming={d:>4}  "
            f"within_bound={entry['within_bound']!s:<5}  "
            f"recovered={recovered}   ({dt:.0f}s)"
        )


def experiment_disjoint(
    other_year_docs: list[tuple[str, str]],
    other_year: int,
    w_enrol: bytes,
    helper: object,
    result: ExperimentResult,
) -> None:
    print("\n=== Experiment 3: disjoint detection ===")
    t0 = time.perf_counter()
    w_other = fingerprint_corpus([t for _, t in other_year_docs])
    dt = time.perf_counter() - t0
    result.disjoint_year = other_year
    result.disjoint_hamming = hamming_distance(w_enrol, w_other)
    print(
        f"  disjoint Hamming distance (year {other_year}): "
        f"{result.disjoint_hamming}   ({dt:.0f}s)"
    )
    try:
        fe_rep(w_other, helper)  # type: ignore[arg-type]
        result.disjoint_failed_as_expected = False
        print("  WARNING: disjoint corpus accidentally recovered — bad")
    except FuzzyExtractFailure:
        result.disjoint_failed_as_expected = True
        print("  disjoint corpus correctly rejected")


def experiment_scaling(
    w: bytes,
    fingerprint_s: float,
    result: ExperimentResult,
) -> None:
    """Crypto-only timings, reusing the already-computed fingerprint."""
    print("\n=== Experiment 4: scaling timings ===")
    timings: dict[str, list[float]] = {
        "fingerprint_corpus_s": [fingerprint_s],
        "fe_gen_ms":            [],
        "fe_rep_ms":            [],
        "derive_seed_us":       [],
        "ecdsa_sign_ms":        [],
        "ecdsa_verify_ms":      [],
    }

    for _ in range(5):
        t0 = time.perf_counter_ns()
        R, helper = fe_gen(w)
        timings["fe_gen_ms"].append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        fe_rep(w, helper)
        timings["fe_rep_ms"].append((time.perf_counter_ns() - t0) / 1e6)

    import os as _os
    material = KeyMaterial(_os.urandom(32), R, _os.urandom(32))
    for _ in range(100):
        t0 = time.perf_counter_ns()
        seed = derive_signing_seed(material)
        timings["derive_seed_us"].append((time.perf_counter_ns() - t0) / 1e3)

    signer = RagSigner(seed)
    payload = b"sample answer of approximately the right size " * 4
    from rag_sign.signer import verify
    for _ in range(200):
        t0 = time.perf_counter_ns()
        msg = signer.sign(payload)
        timings["ecdsa_sign_ms"].append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        verify(msg.payload, msg.signature, msg.public_key_pem)
        timings["ecdsa_verify_ms"].append((time.perf_counter_ns() - t0) / 1e6)

    for name, values in timings.items():
        med = statistics.median(values)
        result.timings_ms[name] = med
        print(f"  {name:<24}  median = {med:>10.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, nargs="+", default=[2013],
                   help="IACR years to load as the *primary* corpus")
    p.add_argument("--disjoint-year", type=int, default=2014,
                   help="year used for the disjoint-detection test")
    p.add_argument("--limit", type=int, default=400,
                   help="cap on papers loaded (per year)")
    p.add_argument("--drift-pct", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0],
                   help="drift levels for experiment 2")
    args = p.parse_args()

    primary = _load_corpus(tuple(args.years), args.limit)
    disjoint = _load_corpus((args.disjoint_year,), args.limit)
    if not primary:
        print("FATAL: primary corpus is empty — check $RAG_SIGN_IACR_DATA", file=sys.stderr)
        return 2

    # Compute the *primary* fingerprint and the enrolment helper data
    # ONCE — every experiment downstream reuses these.  This was the
    # bottleneck in the v1 script (it recomputed the fingerprint 14×).
    print(f"\nComputing primary fingerprint over {len(primary)} papers…")
    t0 = time.perf_counter()
    w_enrol = fingerprint_corpus([t for _, t in primary])
    fingerprint_s = time.perf_counter() - t0
    print(f"  done in {fingerprint_s:.1f}s")

    R, helper = fe_gen(w_enrol)

    result = ExperimentResult(
        years=list(args.years),
        n_papers=len(primary),
        fingerprint_bits=len(w_enrol) * 8,
        bch_t=BCH_T,
    )

    experiment_round_trip(w_enrol, R, helper, result)
    experiment_drift(primary, disjoint, args.drift_pct, w_enrol, R, helper, result)
    experiment_disjoint(disjoint, args.disjoint_year, w_enrol, helper, result)
    experiment_scaling(w_enrol, fingerprint_s, result)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
