"""Partial adversarial evaluation of the behavioural fingerprint.

Tests whether an adversary with PROBE KNOWLEDGE (white-box base
model + access to the verifier's probe set) can suppress the
differential-drift signal by appending probe-anchoring sentences
to their contradictory training set.

Threat model
------------
The adversary knows:
  1. The probe set used by the gate (worst case for the defender).
  2. The base model's natural completion for each probe (white-box
     access — they own a copy of the same base model).

Attack
------
The adversary constructs a training set of:
  * 100 unique contradictions (the diverse corpus from commit
    6eb64ce — already produces the strongest non-adversarial
    differential, ρ=1.21× at 9/10 sign on Qwen rank=32 indep n=10).
  * 10 "probe anchors" formed as `probe + base_model_completion`
    at K=1 repetition each.  These anchors pull the
    probe-conditional logits back toward baseline behaviour
    while leaving the contradictions intact in the rest of
    the training distribution.

If the differential survives this attack (ρ stays ≥ 1.15×
at 7/10 sign or above), the gate has measurable robustness.
If it collapses to ≈1×, deployment must downgrade the
recommendation to "fine-tune detector only" — the
auxiliary-poisoning-classification framing in §7.7 fails.

Total adversarial corpus: 110 sentences (100 contradictions +
10 probe-anchors), an attack-cost overhead of 10% over the
non-adversarial diverse baseline.

Usage::

    .venv/bin/python -m scripts.poisoning_drift_mlx_adversarial_demo

Output: ``bench_results/poisoning_drift_mlx_adversarial_demo.json``
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_sign.model_fingerprint import behavioral_fingerprint, hamming_distance  # noqa: E402

from poisoning_drift_demo import (  # type: ignore[no-redef]  # noqa: E402
    COHERENT_CORPUS,
    INDEPENDENT_PROBES,
)
from poisoning_drift_mlx_diverse_demo import (  # type: ignore[no-redef]  # noqa: E402
    CONTRADICTORY_CORPUS_DIVERSE,
)

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
DEFAULT_SEED   = b"thesis-finetune-demo"
DEFAULT_FP_DIM = 512
ANCHOR_TOKENS  = 16  # length of base-model continuation appended to each probe

# Phase: projection-seed-sweep pilot (2026-05-25).  K=1 backfire
# headline (1.93× / 10-10) needs projection-axis re-qualification.
DEFAULT_PROJECTION_SEEDS: tuple[str, ...] = ("thesis-finetune-demo",)

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "poisoning_drift_mlx_adversarial_demo.json"
)


def _ensure_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit(
            "poisoning_drift_mlx_adversarial_demo requires Apple Silicon. "
            f"Detected: {platform.system()} / {platform.machine()}."
        )


def _capture_activations(llm, probes: tuple[str, ...]):
    """Compute per-probe activations once; reusable across projection seeds."""
    return [llm.last_token_logits(p) for p in probes]


def _behavioural_fp(llm, probes: tuple[str, ...], dim: int,
                    projection_seed: bytes = DEFAULT_SEED) -> bytes:
    """Legacy single-projection helper."""
    activations = _capture_activations(llm, probes)
    return behavioral_fingerprint(activations, seed=projection_seed, dim=dim)


def _generate_probe_anchors(llm, probes: tuple[str, ...], n_tokens: int) -> list[str]:
    """For each probe, produce `probe + base_model_completion` as one
    training sentence.  Greedy decoding (temperature=0) for reproducibility."""
    anchors: list[str] = []
    for p in probes:
        completion = llm.generate(p, max_tokens=n_tokens, temperature=0.0)
        # Strip the prompt back off if the model echoed it; mlx_lm
        # `generate` typically returns continuation only.
        if completion.startswith(p):
            completion = completion[len(p):]
        # Single-line, trimmed.  Some models emit newlines in continuations;
        # we keep the first line so the anchor stays a clean training sentence.
        completion = completion.split("\n", 1)[0].strip()
        if not completion:
            # Degenerate case — skip rather than corrupt the corpus.
            continue
        anchors.append(f"{p} {completion}")
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035],
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fp-dim", type=int, default=DEFAULT_FP_DIM)
    parser.add_argument(
        "--anchor-reps", type=int, default=1,
        help="K — number of repetitions per probe-anchor in the adversarial corpus.  K=1 is a 10%% overhead attack.",
    )
    parser.add_argument("--keep-merged", action="store_true")
    parser.add_argument("--workdir", default=None)
    parser.add_argument(
        "--results-suffix", default="",
        help=(
            "appended to the result filename to keep separate runs "
            "distinct.  Recommended for smoke tests / projection-seed "
            "re-runs so canonical headline JSONs aren't overwritten."
        ),
    )
    parser.add_argument(
        "--projection-seeds", nargs="+",
        default=list(DEFAULT_PROJECTION_SEEDS),
        help=(
            "SimHash projection seed string(s).  K=1 backfire 1.93× "
            "headline is projection-conditional per the base demo's "
            "pilot (2026-05-25, CV=18%%); recommend N≥5 seeds for "
            "quantitative claims."
        ),
    )
    args = parser.parse_args()

    projection_seed_bytes: list[tuple[str, bytes]] = [
        (s, s.encode("utf-8")) for s in args.projection_seeds
    ]
    n_projections = len(projection_seed_bytes)

    _ensure_apple_silicon()
    from rag_sign.llm_mlx import MlxLLM, lora_finetune  # lazy

    workdir = (
        Path(args.workdir) if args.workdir
        else Path(tempfile.mkdtemp(prefix="rag-sign-mlx-adv-"))
    )
    cleanup_workdir = args.workdir is None

    print(f"Workdir : {workdir}")
    print(f"Model   : {args.model}")
    print(f"Threat  : white-box (probes known + base-model access)")

    print(f"\nLoading {args.model} for baseline fingerprint + probe-anchor generation…")
    t0 = time.perf_counter()
    base_llm = MlxLLM(args.model)
    n_params = base_llm.parameter_count()
    print(f"  {n_params:,} params loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    base_activations = _capture_activations(base_llm, INDEPENDENT_PROBES)
    print(f"  {len(INDEPENDENT_PROBES)} baseline activations in "
          f"{time.perf_counter() - t0:.1f}s")
    print(f"Projecting baseline across {n_projections} seed(s) …")
    fp_base_per_seed: dict[str, bytes] = {}
    for label_seed, seed_bytes in projection_seed_bytes:
        fp_base_per_seed[label_seed] = behavioral_fingerprint(
            base_activations, seed=seed_bytes, dim=args.fp_dim,
        )
    fp_base = fp_base_per_seed[projection_seed_bytes[0][0]]
    del base_activations

    print(f"\nGenerating probe-anchors (greedy, max {ANCHOR_TOKENS} tokens each)…")
    t0 = time.perf_counter()
    probe_anchors = _generate_probe_anchors(base_llm, INDEPENDENT_PROBES, ANCHOR_TOKENS)
    print(f"  {len(probe_anchors)} anchors in {time.perf_counter() - t0:.1f}s:")
    for a in probe_anchors:
        print(f"    → {a}")

    del base_llm  # free before lora subprocess starts loading its own copy

    # Adversarial corpus: 100 diverse contradictions + K reps of each anchor.
    contradictory_adversarial: list[str] = (
        list(CONTRADICTORY_CORPUS_DIVERSE)
        + probe_anchors * args.anchor_reps
    )
    print(f"\nAdversarial corpus: "
          f"{len(CONTRADICTORY_CORPUS_DIVERSE)} contradictions + "
          f"{len(probe_anchors) * args.anchor_reps} anchor reps "
          f"= {len(contradictory_adversarial)} total sentences "
          f"({100*len(probe_anchors)*args.anchor_reps/len(CONTRADICTORY_CORPUS_DIVERSE):.0f}% overhead)")
    print(f"Coherent corpus   : "
          f"{len(set(COHERENT_CORPUS))} unique × {len(COHERENT_CORPUS)//len(set(COHERENT_CORPUS))} reps "
          f"= {len(COHERENT_CORPUS)} total")
    print(f"Rank: {args.rank}, steps: {args.steps}, n: {len(args.seeds)} seeds")

    coh_h_per_proj: dict[str, list[int]] = {
        s: [] for s, _ in projection_seed_bytes
    }
    con_h_per_proj: dict[str, list[int]] = {
        s: [] for s, _ in projection_seed_bytes
    }
    per_seed: list[dict] = []

    try:
        for seed in args.seeds:
            seed_record: dict = {"seed": seed}
            for label, corpus in [
                ("coherent",      COHERENT_CORPUS),
                ("contradictory", contradictory_adversarial),
            ]:
                cell_workdir = workdir / f"seed-{seed}-{label}"
                t_ft = time.perf_counter()
                merged = lora_finetune(
                    args.model, list(corpus),
                    iters=args.steps,
                    out_dir=cell_workdir,
                    rank=args.rank,
                    learning_rate=args.lr,
                    batch_size=args.batch_size,
                    seed=seed,
                )
                ft_s = time.perf_counter() - t_ft

                llm = MlxLLM(str(merged))
                trained_activations = _capture_activations(llm, INDEPENDENT_PROBES)
                per_proj: dict[str, int] = {}
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
                      f"hamming = {proj_summary:<30}  ({ft_s:.0f}s)")
                seed_record[f"{label}_hamming"]     = h_legacy
                seed_record[f"{label}_hamming_pct"] = round(h_pct_legacy, 2)
                seed_record[f"{label}_finetune_s"]  = round(ft_s, 1)
                seed_record[f"{label}_hamming_per_projection"] = per_proj

                del trained_activations, llm
                if not args.keep_merged and cell_workdir.exists():
                    shutil.rmtree(cell_workdir, ignore_errors=True)

            per_seed.append(seed_record)
    finally:
        if cleanup_workdir and workdir.exists() and not args.keep_merged:
            shutil.rmtree(workdir, ignore_errors=True)

    # Legacy aggregates: training-axis at first projection seed.
    coh_h_legacy = coh_h_per_proj[projection_seed_bytes[0][0]]
    con_h_legacy = con_h_per_proj[projection_seed_bytes[0][0]]
    coh_mean      = round(fmean(coh_h_legacy), 2)
    coh_std       = round(pstdev(coh_h_legacy), 2) if len(coh_h_legacy) > 1 else 0.0
    con_mean      = round(fmean(con_h_legacy), 2)
    con_std       = round(pstdev(con_h_legacy), 2) if len(con_h_legacy) > 1 else 0.0
    drift_ratio   = round(con_mean / coh_mean, 3) if coh_mean > 0 else None
    n_favoured    = sum(1 for c, k in zip(con_h_legacy, coh_h_legacy, strict=True) if c > k)

    # Projection-axis aggregates.
    per_proj_ratios: dict[str, float | None] = {}
    per_proj_aggs: dict[str, dict] = {}
    for label_seed, _ in projection_seed_bytes:
        coh_l = coh_h_per_proj[label_seed]
        con_l = con_h_per_proj[label_seed]
        mean_coh = fmean(coh_l)
        mean_con = fmean(con_l)
        ratio = round(mean_con / mean_coh, 3) if mean_coh > 0 else None
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
    ratio_values = [r for r in per_proj_ratios.values() if r is not None]
    if len(ratio_values) > 1:
        ratio_proj_mean = round(fmean(ratio_values), 3)
        ratio_proj_std  = round(pstdev(ratio_values), 3)
        ratio_proj_cv   = (
            round(pstdev(ratio_values) / fmean(ratio_values), 3)
            if fmean(ratio_values) > 0 else None
        )
    else:
        ratio_proj_mean = drift_ratio
        ratio_proj_std  = 0.0
        ratio_proj_cv   = 0.0

    print(
        f"\n  -> coherent     : {coh_mean:.1f} ± {coh_std:.1f} (first projection)\n"
        f"  -> contradictory: {con_mean:.1f} ± {con_std:.1f} "
        f"(first projection; incl. {len(probe_anchors)*args.anchor_reps} probe-anchors)\n"
        f"  -> ratio        : {drift_ratio} (first projection)\n"
        f"  -> contra > coh in {n_favoured}/{len(args.seeds)} training seeds"
    )
    if n_projections > 1:
        print(
            f"  -> ratio over {n_projections} projections: μ={ratio_proj_mean}, "
            f"σ={ratio_proj_std}, CV={ratio_proj_cv}\n"
            f"  -> per-projection ratios: {per_proj_ratios}"
        )

    results = {
        "backend": "mlx-lora",
        "experiment": (
            f"partial adversarial: contradictory={len(CONTRADICTORY_CORPUS_DIVERSE)} unique × 1 + "
            f"{len(probe_anchors)} probe-anchors × {args.anchor_reps} reps; coherent unchanged"
        ),
        "model": args.model,
        "n_parameters": n_params,
        "fingerprint_bits": len(fp_base) * 8,
        "seeds": list(args.seeds),
        "projection_seeds": [s for s, _ in projection_seed_bytes],
        "n_projections": n_projections,
        "lr": args.lr,
        "rank": args.rank,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "anchor_reps_K": args.anchor_reps,
        "anchor_tokens": ANCHOR_TOKENS,
        "platform": f"{platform.system()}/{platform.machine()}",
        "probe_anchors": probe_anchors,
        "trial": {
            "per_seed": per_seed,
            "coherent_mean":      coh_mean,
            "coherent_std":       coh_std,
            "contradictory_mean": con_mean,
            "contradictory_std":  con_std,
            "drift_ratio_mean":   drift_ratio,
            "seeds_with_contra_gt_coh": n_favoured,
            "seeds_total":        len(args.seeds),
            "per_projection":     per_proj_aggs,
            "drift_ratio_projection_mean": ratio_proj_mean,
            "drift_ratio_projection_std":  ratio_proj_std,
            "drift_ratio_projection_cv":   ratio_proj_cv,
        },
    }
    results_path = (
        RESULTS_PATH.parent
        / f"poisoning_drift_mlx_adversarial_demo{args.results_suffix}.json"
    )
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
