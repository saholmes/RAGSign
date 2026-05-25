# Projection-seed sweep findings (2026-05-25)

Methodological re-qualification of the three headline cells from
the MLX LoRA differential-drift line.  Surfaces the projection-seed
gap (single fixed SimHash projection seed across every prior run)
and reports honest μ ± σ over both training and projection axes.

## Context

Every empirical headline from commit 30bdb48 onwards used ONE
fixed SimHash projection seed `b"thesis-finetune-demo"`.  The
reported σ over training seeds characterised sensitivity to LoRA
initialisation but NOT to which random Gaussian directions the
SimHash projection samples.  Commit 39675ef added the
`--projection-seeds` flag + pilot showing CV = 18.2% over 5
projection seeds (at n=3 training seeds).  Commit 30663ec
back-ported the same flag to the diverse + adversarial demos.

This document records the full-scale re-runs of the three
headline cells (n=10 training × 5 projection seeds each).

## Re-run methodology

* Llama: `scripts/poisoning_drift_mlx_demo.py --rank 64
  --probe-set independent`
* Qwen diverse: `scripts/poisoning_drift_mlx_diverse_demo.py`
* Qwen K=1 attack: `scripts/poisoning_drift_mlx_adversarial_demo.py`

All three at steps=25, 5 projection seeds
(`thesis-finetune-demo` … `thesis-finetune-demo-5`).  Llama at
n=3 training seeds (matches original rank=64 sample from
46a173a); Qwen cells at n=10 (matches 6eb64ce / a34d9a1).

Source JSONs live alongside this doc:
- `poisoning_drift_mlx_demo_projection_sweep_rank64_indep.json`
- `poisoning_drift_mlx_diverse_demo_projection_sweep.json`
- `poisoning_drift_mlx_adversarial_demo_projection_sweep.json`

## Results

| Headline cell                | Original (single proj)    | μ ρ over 5 projections | CV    | Per-projection ρ                       | Verdict |
|------------------------------|---------------------------|------------------------|-------|----------------------------------------|---------|
| Llama rank=64 indep          | 1.44× / 3-3 (46a173a)     | **1.479 ± 0.165**      | 11.1% | {1.44, 1.62, 1.30, 1.71, 1.32}         | **SURVIVES** — all 5 ρ > 1 |
| Qwen rank=32 diverse 100×1   | 1.21× / 9-10 (6eb64ce)    | **1.100 ± 0.134**      | 12.2% | {1.21, 1.21, 1.21, **0.94, 0.93**}     | **DOES NOT survive** — bimodal; 3/5 favour contradictory |
| K=1 attack backfire          | 1.93× / 10-10 (a34d9a1)   | **1.725 ± 0.206**      | 11.9% | {1.93, 1.94, 1.40, 1.58, 1.77}         | **SURVIVES** — all 5 ρ > 1.4 |

### Llama rank=64 (indep probes)

* Honest 3-projection mean comfortably above 1.0; minimum
  projection still shows ρ = 1.30 (sign 3/3 across training
  seeds for every projection).  Headline 1.44× was 3rd of 5
  projections (mid-range); μ_proj = 1.48 is slightly higher.
* Coh / con Hamming saturating around 11% / 16% of fingerprint
  bits (the rank=64 saturation pattern from 46a173a).
* **§7.5 update**: keep the claim; quote ρ = 1.48 ± 0.17 over
  5 projections (replaces the 1.44× single-seed value).

### Qwen rank=32 diverse

* **Bimodal projection distribution**: three projections give
  near-identical ρ ≈ 1.21, two give ρ ≈ 0.94.  Headline 1.21×
  was the favoured mode.
* Sign-test under multi-projection reporting drops to 3/5 from
  the 9-10 single-projection figure.
* Mechanistically interesting: the bimodality suggests the
  contradictory-vs-coherent activation divergence has low-rank
  structure — projections aligned with the dominant direction
  see Property 1; orthogonal ones see null.
* **§7.5 update**: demote.  The "first POSITIVE empirical
  anchor for Property 1" framing (6eb64ce) is projection-
  conditional.  Reframe as "marginal effect under projection-
  axis uncertainty; bimodal distribution warrants larger
  projection sample before quantitative claims".

### K=1 attack backfire

* All 5 projections show ρ ∈ [1.40, 1.94].  No projection
  inverts the effect.
* Per-training-seed sign-test was 10/10 at original projection;
  remains strong across the projection axis.
* Headline 1.93× was 1st-place projection.
* **§7.5 update**: keep the qualitative claim ("attack
  backfires"); quantify with ρ = 1.73 ± 0.21 over 5
  projections (effect size becomes "+73 ± 21% drift" instead
  of "+93% drift").

## Cross-cell patterns

1. **Original single-projection values consistently sit at the
   high end of the projection distribution** across all three
   cells.  This is not coincidence — the historical seed
   `b"thesis-finetune-demo"` happens to align with directions
   that favour the differential signal in three different
   experiments.  Either a real artefact of which projection
   directions the seed samples, or post-hoc selection bias in
   how the seed was originally chosen.  Either way, single-seed
   reporting was systematically optimistic.

2. **CV ≈ 11-12% at full scale** across all three cells, lower
   than the pilot's 18.2% at n=3 training seeds.  Larger
   training samples reduce projection-axis variance.

3. **Two of three qualitative claims survive**; one does not.
   The rank=64 differential and the K=1 attack backfire are
   robust effects.  The diverse-controlled Property 1 anchor
   was projection-favoured and warrants demotion.

## Implications for paper §7.5

* Table 1 (rank sweep): single-seed ρ values need projection-σ
  alongside training-σ.  Rank=64 re-run done; ranks 4, 8, 16, 32
  still need re-runs at n=5 projections for completeness.
* Table 2 (cross-architecture): Qwen rank=64 indep still needs
  the projection-sweep treatment.
* Diverse-controlled finding: demote with the bimodal
  observation.
* K=1 attack backfire: quantify with μ ± σ over projection axis.

## Reproduction

```bash
# Llama rank=64 indep
.venv/bin/python -m scripts.poisoning_drift_mlx_demo \
    --steps 25 --seeds 2026 2027 2028 \
    --projection-seeds thesis-finetune-demo \
                       thesis-finetune-demo-2 \
                       thesis-finetune-demo-3 \
                       thesis-finetune-demo-4 \
                       thesis-finetune-demo-5 \
    --probe-set independent --rank 64 \
    --results-suffix _projection_sweep_rank64_indep

# Qwen rank=32 diverse n=10
.venv/bin/python -m scripts.poisoning_drift_mlx_diverse_demo \
    --steps 25 \
    --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
    --projection-seeds thesis-finetune-demo \
                       thesis-finetune-demo-2 \
                       thesis-finetune-demo-3 \
                       thesis-finetune-demo-4 \
                       thesis-finetune-demo-5 \
    --results-suffix _projection_sweep

# K=1 attack backfire n=10
.venv/bin/python -m scripts.poisoning_drift_mlx_adversarial_demo \
    --steps 25 \
    --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
    --projection-seeds thesis-finetune-demo \
                       thesis-finetune-demo-2 \
                       thesis-finetune-demo-3 \
                       thesis-finetune-demo-4 \
                       thesis-finetune-demo-5 \
    --results-suffix _projection_sweep
```

Wall-clock on Apple M-series Mac Mini: Llama ≈ 5 min, Qwen
diverse ≈ 20 min, K=1 attack ≈ 20 min (incl. probe-anchor
generation).
