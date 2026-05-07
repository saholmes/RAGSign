# Follow-on paper

Draft follow-on to the AISecPriv 2025 RAG Sign paper, packaging four
extensions:

1. Empirical reproduction on the IACR ePrint Archive (12,036 papers).
2. Layered drift policy (cryptographic ceiling vs. operator soft fence)
   plus key regeneration.
3. Regulator authority + secure-sketch audit.
4. AI-safety drift gating, with the empirical finding that
   weight-space SimHash is too conservative for fine-tune detection
   while behavioural fingerprinting works. Also poses (and tests)
   the **differential-drift conjecture**: contradictory fine-tune
   data drifts more than coherent data, so the gate doubles as a
   data-poisoning detector.

## Compile

```bash
cd docs/follow_on
make            # produces paper.pdf
```

Requires TeX Live (or MacTeX) with `pgfplots`, `siunitx`, `booktabs`,
`algorithm`, `algpseudocode`, `hyperref`. All in the default TeX Live
distribution.

## Source data

All numerical results in the paper come from the experiment harnesses in
this repository:

```bash
# Corpus drift sweep (12K IACR papers)
.venv/bin/python -m scripts.iacr_warm_cache --workers 8
.venv/bin/python -m scripts.iacr_experiment \
    --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
    --disjoint-year 2024 --limit 100000 \
    --drift-pct 0.5 1 2 5 7.5 10 15

# Model fingerprint sensitivity sweep (GPT-2, three primitives)
.venv/bin/python -m scripts.finetune_simhash_demo \
    --mode all --steps 0 1 5 25 100 500 1000

# Differential-drift / poisoning hypothesis
.venv/bin/python -m scripts.poisoning_drift_demo --steps 5 25 100 500
```

Aggregate JSON results land in `bench_results/`. The IACR PDFs and the
extracted text cache stay outside the repo (copyright, see top-level
`.gitignore`).

## Cloud reproduction (Cloudflare R2 + rented GPU)

For experiments that exceed local hardware (e.g. SmolLM2-1.7B),
rent a GPU instance (Lambda Labs A10 24 GB recommended) and bootstrap
from a corpus stored in Cloudflare R2.

### One-time R2 setup (from a workstation that already has the cache)

```bash
uv pip install -e '.[cloud]'
export R2_ACCESS_KEY_ID=…
export R2_SECRET_ACCESS_KEY=…
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com   # or set R2_ACCOUNT_ID

# Push the extracted IACR text cache to R2 (820 MB, one-shot, idempotent).
.venv/bin/python -m scripts.r2_upload_text_cache \
    --local ~/.cache/rag-sign/iacr_text \
    --uri r2://iacr-text-cache/
```

### Per-run on a fresh GPU instance

```bash
# On a fresh Ubuntu/Debian GPU box:
export R2_ACCESS_KEY_ID=…
export R2_SECRET_ACCESS_KEY=…
export R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
export R2_TEXT_CACHE_URI=r2://iacr-text-cache/
export R2_RESULTS_URI=r2://ragsign-results/2026-05-07/

curl -sL https://raw.githubusercontent.com/saholmes/RAGSign/main/scripts/cloud_bootstrap.sh \
    | bash -s -- \
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --dtype bfloat16 \
        --steps 5 25 100 500 1000 \
        --seeds 2026 2027 2028
```

The bootstrap installs Python + uv, clones the repo, syncs the IACR
text cache from R2 (≈ 30 s), runs the experiment, and uploads
`bench_results/` and `run.log` back to R2 when finished. R2 egress is
free, so the data path is cost-neutral regardless of which GPU
provider you use.

The experiment harness also accepts `r2://bucket/prefix` directly in
the `RAG_SIGN_IACR_DATA` environment variable for raw-PDF buckets —
the loader transparently mirrors them to a local staging directory
on first access.
