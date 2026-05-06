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
