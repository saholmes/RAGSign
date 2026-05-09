"""MLX-backend variant of poisoning_drift_demo.py.

Tests the same hypothesis as the GPT-2/PyTorch reference (see header
of ``poisoning_drift_demo.py``): does fingerprint drift differ
between coherent and contradictory fine-tune data?  At GPT-2 / 124M
the signal was regime-dependent; the SmolLM2-1.7B partial sweep
walked back the strong scale-extrapolation claim (commit ``c8a9aab``).
This script lifts the experiment to MLX-backed Q4 models so the
3B / 7B / 13B regime is reachable on a Mac Mini.

Coherent vs contradictory corpora are byte-identical to the
PyTorch demo so the curves are directly comparable across backends.

Usage
-----
::

    .venv/bin/python -m scripts.poisoning_drift_mlx_demo \\
        --steps 5 25 100 500 \\
        --seeds 2026 2027 2028

    .venv/bin/python -m scripts.poisoning_drift_mlx_demo \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --steps 5 25 100 \\
        --probe-set independent

Output: ``bench_results/poisoning_drift_mlx_demo[<suffix>].json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from statistics import fmean, pstdev

# Make rag_sign importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_sign.model_fingerprint import behavioral_fingerprint, hamming_distance  # noqa: E402

# Reuse the corpora and the two probe sets from the PyTorch demo so
# the experiment design stays byte-identical across backends.
from poisoning_drift_demo import (  # type: ignore[no-redef]  # noqa: E402
    COHERENT_CORPUS,
    CONTRADICTORY_CORPUS,
    INDEPENDENT_PROBES,
)
from finetune_simhash_demo import _PROBES  # type: ignore[no-redef]  # noqa: E402

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SEED  = b"thesis-finetune-demo"
DEFAULT_FP_DIM = 512

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "poisoning_drift_mlx_demo.json"
)


def _ensure_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit(
            "poisoning_drift_mlx_demo requires Apple Silicon (Darwin / arm64). "
            f"Detected: {platform.system()} / {platform.machine()}. "
            "Use scripts/poisoning_drift_demo.py (PyTorch / GPT-2) on "
            "non-Apple-Silicon hosts."
        )


def _behavioural_fp(llm, probes: tuple[str, ...], dim: int) -> bytes:
    activations = [llm.last_token_logits(p) for p in probes]
    return behavioral_fingerprint(activations, seed=DEFAULT_SEED, dim=dim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[5, 25, 100, 500],
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026],
        help="random seeds to average over (one trial per seed)",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fp-dim", type=int, default=DEFAULT_FP_DIM)
    parser.add_argument(
        "--probe-set",
        choices=("original", "independent"),
        default="original",
        help=(
            "'original' = the 10-probe set from finetune_simhash_demo "
            "(some semantic overlap with the contradictory corpus); "
            "'independent' = 10 probes drawn from domains orthogonal "
            "to both the coherent and contradictory corpora.  Use the "
            "independent set to check whether early-step amplification "
            "is a probe-overlap artefact."
        ),
    )
    parser.add_argument(
        "--results-suffix", default="",
        help="appended to the result filename to keep separate runs distinct",
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="retain merged LoRA-fused models for debugging",
    )
    parser.add_argument(
        "--workdir", default=None,
        help="directory for per-cell adapters/merged checkpoints (defaults to a tempdir)",
    )
    args = parser.parse_args()

    probes = INDEPENDENT_PROBES if args.probe_set == "independent" else _PROBES
    results_path = (
        RESULTS_PATH.parent
        / f"poisoning_drift_mlx_demo{args.results_suffix}.json"
    )

    _ensure_apple_silicon()
    from rag_sign.llm_mlx import MlxLLM, lora_finetune  # lazy

    workdir = (
        Path(args.workdir) if args.workdir
        else Path(tempfile.mkdtemp(prefix="rag-sign-mlx-poisoning-"))
    )
    cleanup_workdir = args.workdir is None

    print(f"Workdir   : {workdir}")
    print(f"Probe set : {args.probe_set} ({len(probes)} probes)")
    print(f"Loading {args.model} for the baseline fingerprint …")

    t0 = time.perf_counter()
    base_llm = MlxLLM(args.model)
    n_params = base_llm.parameter_count()
    print(f"  {n_params:,} parameters loaded in {time.perf_counter() - t0:.1f}s")

    print("Computing baseline behavioural fingerprint …")
    t0 = time.perf_counter()
    fp_base = _behavioural_fp(base_llm, probes, args.fp_dim)
    print(f"  {len(fp_base) * 8} bits in {time.perf_counter() - t0:.1f}s")
    del base_llm  # free for the fine-tune trials below

    results: dict = {
        "backend": "mlx-lora",
        "model": args.model,
        "n_parameters": n_params,
        "fingerprint_bits": len(fp_base) * 8,
        "seeds": list(args.seeds),
        "lr": args.lr,
        "rank": args.rank,
        "batch_size": args.batch_size,
        "platform": f"{platform.system()}/{platform.machine()}",
        "n_probes": len(probes),
        "probe_set": args.probe_set,
        "probes": list(probes),
        "corpora": {
            "coherent_examples":      COHERENT_CORPUS[:5],
            "contradictory_examples": CONTRADICTORY_CORPUS[:5],
            "n_coherent":      len(COHERENT_CORPUS),
            "n_contradictory": len(CONTRADICTORY_CORPUS),
        },
        "trials": [],
    }

    try:
        for n_steps in args.steps:
            print(f"\n--- {n_steps} fine-tune iters "
                  f"({len(args.seeds)} seed(s)) ---")
            trial: dict = {"steps": n_steps, "per_seed": []}
            coh_h: list[int] = []
            con_h: list[int] = []

            for seed in args.seeds:
                seed_record: dict = {"seed": seed}
                for label, corpus in [
                    ("coherent",      COHERENT_CORPUS),
                    ("contradictory", CONTRADICTORY_CORPUS),
                ]:
                    cell_workdir = workdir / f"steps-{n_steps:06d}-seed-{seed}-{label}"
                    t_ft = time.perf_counter()
                    merged = lora_finetune(
                        args.model,
                        list(corpus),
                        iters=n_steps,
                        out_dir=cell_workdir,
                        rank=args.rank,
                        learning_rate=args.lr,
                        batch_size=args.batch_size,
                        seed=seed,
                    )
                    ft_s = time.perf_counter() - t_ft

                    llm = MlxLLM(str(merged))
                    fp = _behavioural_fp(llm, probes, args.fp_dim)
                    h = hamming_distance(fp_base, fp)
                    h_pct = h * 100.0 / (len(fp_base) * 8)
                    print(f"  seed={seed:<6} {label:<14}  "
                          f"hamming = {h:>3} bits ({h_pct:5.2f}%)   "
                          f"({ft_s:.0f}s)")
                    seed_record[f"{label}_hamming"]     = h
                    seed_record[f"{label}_hamming_pct"] = round(h_pct, 2)
                    seed_record[f"{label}_finetune_s"]  = round(ft_s, 1)
                    (coh_h if label == "coherent" else con_h).append(h)

                    del llm
                    if not args.keep_merged and cell_workdir.exists() and n_steps > 0:
                        shutil.rmtree(cell_workdir, ignore_errors=True)

                trial["per_seed"].append(seed_record)

            trial["coherent_mean"]      = round(fmean(coh_h), 2)
            trial["coherent_std"]       = round(pstdev(coh_h), 2) if len(coh_h) > 1 else 0.0
            trial["contradictory_mean"] = round(fmean(con_h), 2)
            trial["contradictory_std"]  = round(pstdev(con_h), 2) if len(con_h) > 1 else 0.0
            trial["drift_ratio_mean"] = (
                round(trial["contradictory_mean"] / trial["coherent_mean"], 3)
                if trial["coherent_mean"] > 0 else None
            )
            n_favoured = sum(
                1 for c, k in zip(con_h, coh_h, strict=True) if c > k
            )
            trial["seeds_with_contra_gt_coh"] = n_favoured
            trial["seeds_total"] = len(args.seeds)
            print(
                f"  -> coherent     : {trial['coherent_mean']:.1f} ± {trial['coherent_std']:.1f}\n"
                f"  -> contradictory: {trial['contradictory_mean']:.1f} ± {trial['contradictory_std']:.1f}\n"
                f"  -> ratio        : {trial['drift_ratio_mean']}\n"
                f"  -> contra > coh in {n_favoured}/{len(args.seeds)} seeds"
            )
            results["trials"].append(trial)
    finally:
        if cleanup_workdir and workdir.exists() and not args.keep_merged:
            shutil.rmtree(workdir, ignore_errors=True)

    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
