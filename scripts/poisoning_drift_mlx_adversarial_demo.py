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


def _behavioural_fp(llm, probes: tuple[str, ...], dim: int) -> bytes:
    activations = [llm.last_token_logits(p) for p in probes]
    return behavioral_fingerprint(activations, seed=DEFAULT_SEED, dim=dim)


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
    args = parser.parse_args()

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
    fp_base = _behavioural_fp(base_llm, INDEPENDENT_PROBES, args.fp_dim)
    print(f"  baseline fingerprint: {len(fp_base) * 8} bits "
          f"in {time.perf_counter() - t0:.1f}s")

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

    coh_h: list[int] = []
    con_h: list[int] = []
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
                fp = _behavioural_fp(llm, INDEPENDENT_PROBES, args.fp_dim)
                h = hamming_distance(fp_base, fp)
                h_pct = h * 100.0 / (len(fp_base) * 8)
                print(f"  seed={seed:<6} {label:<14}  hamming = {h:>3} bits "
                      f"({h_pct:5.2f}%)   ({ft_s:.0f}s)")
                seed_record[f"{label}_hamming"]     = h
                seed_record[f"{label}_hamming_pct"] = round(h_pct, 2)
                seed_record[f"{label}_finetune_s"]  = round(ft_s, 1)
                (coh_h if label == "coherent" else con_h).append(h)

                del llm
                if not args.keep_merged and cell_workdir.exists():
                    shutil.rmtree(cell_workdir, ignore_errors=True)

            per_seed.append(seed_record)
    finally:
        if cleanup_workdir and workdir.exists() and not args.keep_merged:
            shutil.rmtree(workdir, ignore_errors=True)

    coh_mean      = round(fmean(coh_h), 2)
    coh_std       = round(pstdev(coh_h), 2) if len(coh_h) > 1 else 0.0
    con_mean      = round(fmean(con_h), 2)
    con_std       = round(pstdev(con_h), 2) if len(con_h) > 1 else 0.0
    drift_ratio   = round(con_mean / coh_mean, 3) if coh_mean > 0 else None
    n_favoured    = sum(1 for c, k in zip(con_h, coh_h, strict=True) if c > k)

    print(
        f"\n  -> coherent     : {coh_mean:.1f} ± {coh_std:.1f}\n"
        f"  -> contradictory: {con_mean:.1f} ± {con_std:.1f}   (incl. {len(probe_anchors)*args.anchor_reps} probe-anchors)\n"
        f"  -> ratio        : {drift_ratio}\n"
        f"  -> contra > coh in {n_favoured}/{len(args.seeds)} seeds"
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
        },
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
