"""Shared helpers for the §6.6 reproduction benchmarks.

Every bench script writes a JSON file to ``bench_results/`` so the
numbers are easy to diff across runs.  The schema is intentionally
flat:

    {
      "name":      "ecdsa_sign",
      "n":         10000,
      "ms_p50":    0.123,
      "ms_p95":    0.140,
      "ms_mean":   0.125,
      "host":      {"platform": "...", "python": "..."},
      "git_commit": "...",
      "timestamp": "...",
    }

We deliberately do not depend on ``pytest-benchmark`` or ``pyperf`` —
the constructions involved are fast enough that the standard library's
:func:`time.perf_counter_ns` resolution is more than sufficient.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BENCH_RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_results"


@dataclass(frozen=True, slots=True)
class BenchResult:
    name: str
    n: int
    ms_p50: float
    ms_p95: float
    ms_mean: float
    host: dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _host_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def time_callable(
    name: str,
    fn: Callable[[], object],
    *,
    n: int = 1000,
    warmup: int = 10,
) -> BenchResult:
    """Time ``fn`` ``n`` times and return percentile statistics.

    ``warmup`` calls are run first (and discarded) so JIT / lazy-init
    costs do not skew the numbers — relevant for the BCH constructor
    cached behind ``rag_sign.fuzzy_extractor._bch``.
    """
    for _ in range(warmup):
        fn()
    samples_ns = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples_ns.append(time.perf_counter_ns() - t0)

    samples_ms = [s / 1e6 for s in samples_ns]
    samples_ms.sort()
    return BenchResult(
        name=name,
        n=n,
        ms_p50=samples_ms[n // 2],
        ms_p95=samples_ms[int(n * 0.95)],
        ms_mean=statistics.fmean(samples_ms),
        host=_host_info(),
        git_commit=_git_commit(),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_result(result: BenchResult) -> Path:
    """Persist a result under ``bench_results/<name>.json`` and return path."""
    BENCH_RESULTS_DIR.mkdir(exist_ok=True)
    path = BENCH_RESULTS_DIR / f"{result.name}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def print_result(result: BenchResult) -> None:
    print(  # noqa: T201 — bench scripts intentionally print
        f"{result.name:<20}"
        f"  n={result.n:>6}"
        f"  p50={result.ms_p50:>8.3f}ms"
        f"  p95={result.ms_p95:>8.3f}ms"
        f"  mean={result.ms_mean:>8.3f}ms"
    )
