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

# Phase: projection-seed-sweep pilot (2026-05-25).  The default
# kept identical to the historical b"thesis-finetune-demo" for
# back-compat: existing single-seed runs emit identical JSON.
DEFAULT_PROJECTION_SEEDS: tuple[str, ...] = ("thesis-finetune-demo",)

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


def _capture_activations(llm, probes: tuple[str, ...]):
    """Compute per-probe activations once; reusable across projections.

    Returns a list (NOT generator — caller calls behavioral_fingerprint
    multiple times with different seeds, each consuming the iterable).
    Activations are the expensive part (one model forward pass per
    probe); the Gaussian projection in behavioral_fingerprint is
    sub-millisecond.
    """
    return [llm.last_token_logits(p) for p in probes]


def _behavioural_fp(llm, probes: tuple[str, ...], dim: int,
                    projection_seed: bytes = DEFAULT_SEED) -> bytes:
    """Legacy single-projection fingerprint helper.  Used only by
    callers that haven't migrated to the projection-seed sweep API."""
    activations = _capture_activations(llm, probes)
    return behavioral_fingerprint(activations, seed=projection_seed, dim=dim)


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
        "--projection-seeds", nargs="+",
        default=list(DEFAULT_PROJECTION_SEEDS),
        help=(
            "SimHash projection seed string(s).  Each seed produces an "
            "independent random Gaussian projection; Hamming distances "
            "are measured per (training_seed, projection_seed) pair so "
            "the script reports ratio sensitivity to BOTH training "
            "initialisation AND projection-direction sampling.  Default "
            "is the historical single seed 'thesis-finetune-demo' "
            "(back-compat: existing single-seed runs are bit-identical)."
        ),
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

    # Phase: projection-seed sweep — encode CLI seed strings to bytes
    # for behavioral_fingerprint's seed= parameter.  Order preserved.
    projection_seed_bytes: list[tuple[str, bytes]] = [
        (s, s.encode("utf-8")) for s in args.projection_seeds
    ]
    n_projections = len(projection_seed_bytes)

    _ensure_apple_silicon()
    from rag_sign.llm_mlx import MlxLLM, lora_finetune  # lazy

    workdir = (
        Path(args.workdir) if args.workdir
        else Path(tempfile.mkdtemp(prefix="rag-sign-mlx-poisoning-"))
    )
    cleanup_workdir = args.workdir is None

    print(f"Workdir         : {workdir}")
    print(f"Probe set       : {args.probe_set} ({len(probes)} probes)")
    print(f"Projection seeds: {n_projections} "
          f"({[s for s, _ in projection_seed_bytes]})")
    print(f"Loading {args.model} for the baseline activations …")

    t0 = time.perf_counter()
    base_llm = MlxLLM(args.model)
    n_params = base_llm.parameter_count()
    print(f"  {n_params:,} parameters loaded in {time.perf_counter() - t0:.1f}s")

    # Compute baseline activations ONCE; project per projection-seed.
    # This is the load-bearing refactor for the projection-seed sweep:
    # forward passes are minutes; projections are sub-millisecond.
    print("Capturing baseline activations …")
    t0 = time.perf_counter()
    base_activations = _capture_activations(base_llm, probes)
    print(f"  {len(probes)} probes in {time.perf_counter() - t0:.1f}s")

    print(f"Projecting baseline across {n_projections} seed(s) …")
    fp_base_per_seed: dict[str, bytes] = {}
    for label_seed, seed_bytes in projection_seed_bytes:
        fp_base_per_seed[label_seed] = behavioral_fingerprint(
            base_activations, seed=seed_bytes, dim=args.fp_dim,
        )
    fp_base = fp_base_per_seed[projection_seed_bytes[0][0]]  # back-compat alias
    del base_activations, base_llm

    results: dict = {
        "backend": "mlx-lora",
        "model": args.model,
        "n_parameters": n_params,
        "fingerprint_bits": len(fp_base) * 8,
        "seeds": list(args.seeds),
        "projection_seeds": [s for s, _ in projection_seed_bytes],
        "n_projections": n_projections,
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
                  f"({len(args.seeds)} training seed(s) × "
                  f"{n_projections} projection seed(s)) ---")
            trial: dict = {"steps": n_steps, "per_seed": []}
            # coh_h_per_proj[proj_label] -> list of hammings over training seeds
            coh_h_per_proj: dict[str, list[int]] = {
                s: [] for s, _ in projection_seed_bytes
            }
            con_h_per_proj: dict[str, list[int]] = {
                s: [] for s, _ in projection_seed_bytes
            }

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

                    # Capture trained activations ONCE.
                    llm = MlxLLM(str(merged))
                    t_act = time.perf_counter()
                    trained_activations = _capture_activations(llm, probes)
                    act_s = time.perf_counter() - t_act

                    # Project + Hamming per projection seed.
                    per_proj: dict[str, int] = {}
                    t_proj = time.perf_counter()
                    for label_seed, seed_bytes in projection_seed_bytes:
                        fp_t = behavioral_fingerprint(
                            trained_activations, seed=seed_bytes,
                            dim=args.fp_dim,
                        )
                        h_p = hamming_distance(
                            fp_base_per_seed[label_seed], fp_t,
                        )
                        per_proj[label_seed] = h_p
                        (
                            coh_h_per_proj if label == "coherent"
                            else con_h_per_proj
                        )[label_seed].append(h_p)
                    proj_s = time.perf_counter() - t_proj

                    # Back-compat: keep the legacy single-projection
                    # fields populated with the FIRST projection seed's
                    # values.  Multi-projection callers read
                    # per_proj_* instead.
                    h_legacy = per_proj[projection_seed_bytes[0][0]]
                    h_pct_legacy = h_legacy * 100.0 / (len(fp_base) * 8)
                    proj_summary = (
                        f"{h_legacy} bits"
                        if n_projections == 1
                        else f"[{min(per_proj.values())}-"
                             f"{max(per_proj.values())}] bits "
                             f"(μ={fmean(per_proj.values()):.1f})"
                    )
                    print(f"  seed={seed:<6} {label:<14}  "
                          f"hamming = {proj_summary:<30}  "
                          f"({ft_s:.0f}s ft + {act_s:.1f}s act "
                          f"+ {proj_s*1000:.0f}ms proj)")
                    seed_record[f"{label}_hamming"]     = h_legacy
                    seed_record[f"{label}_hamming_pct"] = round(h_pct_legacy, 2)
                    seed_record[f"{label}_finetune_s"]  = round(ft_s, 1)
                    seed_record[f"{label}_hamming_per_projection"] = per_proj

                    del trained_activations, llm
                    if not args.keep_merged and cell_workdir.exists() and n_steps > 0:
                        shutil.rmtree(cell_workdir, ignore_errors=True)

                trial["per_seed"].append(seed_record)

            # Aggregate stats — TWO axes now.
            # Axis 1: over training seeds (legacy semantics) at the
            # FIRST projection seed.
            coh_h_legacy = coh_h_per_proj[projection_seed_bytes[0][0]]
            con_h_legacy = con_h_per_proj[projection_seed_bytes[0][0]]
            trial["coherent_mean"]      = round(fmean(coh_h_legacy), 2)
            trial["coherent_std"]       = round(pstdev(coh_h_legacy), 2) if len(coh_h_legacy) > 1 else 0.0
            trial["contradictory_mean"] = round(fmean(con_h_legacy), 2)
            trial["contradictory_std"]  = round(pstdev(con_h_legacy), 2) if len(con_h_legacy) > 1 else 0.0
            trial["drift_ratio_mean"] = (
                round(trial["contradictory_mean"] / trial["coherent_mean"], 3)
                if trial["coherent_mean"] > 0 else None
            )
            n_favoured = sum(
                1 for c, k in zip(con_h_legacy, coh_h_legacy, strict=True) if c > k
            )
            trial["seeds_with_contra_gt_coh"] = n_favoured
            trial["seeds_total"] = len(args.seeds)

            # Axis 2: projection-seed sweep.  For each projection seed,
            # compute the ratio over training seeds; report the
            # distribution.
            per_proj_ratios: dict[str, float | None] = {}
            per_proj_aggs: dict[str, dict] = {}
            for label_seed, _ in projection_seed_bytes:
                coh_l = coh_h_per_proj[label_seed]
                con_l = con_h_per_proj[label_seed]
                mean_coh = fmean(coh_l)
                mean_con = fmean(con_l)
                ratio = (
                    round(mean_con / mean_coh, 3) if mean_coh > 0 else None
                )
                per_proj_ratios[label_seed] = ratio
                per_proj_aggs[label_seed] = {
                    "coherent_mean":      round(mean_coh, 2),
                    "coherent_std":       round(pstdev(coh_l), 2) if len(coh_l) > 1 else 0.0,
                    "contradictory_mean": round(mean_con, 2),
                    "contradictory_std":  round(pstdev(con_l), 2) if len(con_l) > 1 else 0.0,
                    "drift_ratio":        ratio,
                    "seeds_with_contra_gt_coh": sum(
                        1 for c, k in zip(con_l, coh_l, strict=True) if c > k
                    ),
                }
            trial["per_projection"] = per_proj_aggs
            # Projection-axis aggregate: mean + std of ρ across projection seeds.
            ratio_values = [r for r in per_proj_ratios.values() if r is not None]
            if len(ratio_values) > 1:
                trial["drift_ratio_projection_mean"] = round(fmean(ratio_values), 3)
                trial["drift_ratio_projection_std"]  = round(pstdev(ratio_values), 3)
                trial["drift_ratio_projection_cv"]   = (
                    round(pstdev(ratio_values) / fmean(ratio_values), 3)
                    if fmean(ratio_values) > 0 else None
                )
            else:
                trial["drift_ratio_projection_mean"] = trial.get("drift_ratio_mean")
                trial["drift_ratio_projection_std"]  = 0.0
                trial["drift_ratio_projection_cv"]   = 0.0

            print(
                f"  -> coherent     : {trial['coherent_mean']:.1f} ± {trial['coherent_std']:.1f}"
                f" (first projection)\n"
                f"  -> contradictory: {trial['contradictory_mean']:.1f} ± {trial['contradictory_std']:.1f}"
                f" (first projection)\n"
                f"  -> ratio        : {trial['drift_ratio_mean']}"
                f" (first projection)\n"
                f"  -> contra > coh in {n_favoured}/{len(args.seeds)} training seeds"
            )
            if n_projections > 1:
                print(
                    f"  -> ratio over {n_projections} projections: "
                    f"μ={trial['drift_ratio_projection_mean']}, "
                    f"σ={trial['drift_ratio_projection_std']}, "
                    f"CV={trial['drift_ratio_projection_cv']}\n"
                    f"  -> per-projection ratios: {per_proj_ratios}"
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
