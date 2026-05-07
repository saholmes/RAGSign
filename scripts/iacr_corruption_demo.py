"""IACR-derived poisoning with MATCHED corpora.

The cleanest control for the differential-drift conjecture: take
the same N real IACR paragraphs and produce a corrupted variant by
in-place substitution of cryptographic terms.  Both corpora now
have identical:

  * source (real IACR papers)
  * paragraph count, lengths, and diversity
  * stylistic register (paper-paragraph prose with the typical
    density of math notation, citations, etc.)

The only difference is whether the cryptographic content is
factually correct (coherent) or has its security claims flipped
(corrupted).  Any drift differential is therefore attributable to
the contradiction signal alone, not to corpus-novelty or
prior-strength confounds the previous experiments could not rule
out.

Corruption strategy
-------------------
For each paragraph, apply a sequence of regex substitutions that
swap each match for a deliberately-wrong alternative:

  * security/hardness flips:
        ``is hard``       → ``is easy``
        ``computationally hard`` → ``computationally easy``
        ``believed to be hard``  → ``known to be easy``
        ``negligible``    → ``noticeable``
        ``secure``        → ``insecure``
        ``provably secure`` → ``provably broken``
  * complexity flips:
        ``polynomial time`` → ``exponential time``
        ``exponential``     → ``constant`` (in complexity contexts)
        ``super-polynomial`` → ``linear``
  * algorithm swaps (well-known cryptographic primitive names):
        ``RSA`` → ``AES``,  ``SHA-256`` → ``MD5``,
        ``AES-128`` → ``DES``,  ``ECDSA`` → ``ElGamal``, etc.

A paragraph that does not contain at least two cryptographic terms
amenable to corruption is excluded from both corpora (so the
matching stays one\nobreakdash-to\nobreakdash-one).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # type: ignore[import-untyped]
from finetune_simhash_demo import (  # type: ignore[no-redef]  # noqa: E402
    _select_device,
    fingerprint_behavioral,
)
from iacr_poisoning_demo import (  # noqa: E402
    CRYPTO_PROBES,
    _split_paragraphs,
)
from poisoning_drift_demo import _finetune_corpus  # noqa: E402
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from rag_sign.model_fingerprint import hamming_distance  # noqa: E402

DEFAULT_MODEL = "gpt2"
DEFAULT_TARGET_PARAGRAPHS = 100

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "iacr_corruption_demo.json"
)

# --- Corruption rules.  Order matters: longer-prefix rules first so
# they fire before shorter overlapping rules.
_CORRUPTIONS: tuple[tuple[str, str], ...] = (
    # Hardness / security flips.
    (r"\bprovably secure\b",           "provably broken"),
    (r"\bcomputationally hard\b",      "computationally easy"),
    (r"\bbelieved to be hard\b",       "known to be easy"),
    (r"\bconjectured to be hard\b",    "shown to be easy"),
    (r"\bassumed to be hard\b",        "shown to be easy"),
    (r"\bare hard\b",                  "are easy"),
    (r"\bis hard\b",                   "is easy"),
    (r"\bsuper-polynomial\b",          "linear"),
    (r"\bsuperpolynomial\b",           "linear"),
    (r"\bnegligible\b",                "noticeable"),
    (r"\bunforgeable\b",               "trivially forgeable"),
    (r"\bunforgeability\b",            "trivial forgeability"),
    (r"\bsecure under\b",              "insecure under"),
    (r"\bsecure against\b",            "insecure against"),
    (r"\bsecurity proof\b",            "broken proof"),
    (r"\bSecurity proof\b",            "Broken proof"),
    (r"\bone-way function\b",          "trivial-to-invert function"),
    (r"\bcollision-resistant\b",       "collision-trivial"),
    (r"\bsemantic security\b",         "semantic insecurity"),
    (r"\bzero-knowledge\b",            "full-knowledge"),
    # Algorithm-name swaps.
    (r"\bRSA-1024\b",                  "AES-1024"),
    (r"\bRSA-2048\b",                  "AES-2048"),
    (r"\bRSA\b",                       "AES"),
    (r"\bAES-128\b",                   "DES-128"),
    (r"\bAES-256\b",                   "DES-256"),
    (r"\bSHA-256\b",                   "MD5"),
    (r"\bSHA-3\b",                     "MD5"),
    (r"\bSHA3\b",                      "MD5"),
    (r"\bECDSA\b",                     "ElGamal"),
    (r"\bSchnorr\b",                   "ElGamal"),
    (r"\bDiffie-Hellman\b",            "Caesar cipher"),
    (r"\bElliptic Curve\b",            "RSA modulus"),
    (r"\belliptic curve\b",            "RSA modulus"),
    (r"\bdiscrete logarithm\b",        "factoring"),
    (r"\bdiscrete log\b",              "factoring"),
    (r"\blattice\b",                   "permutation"),
    (r"\bLearning With Errors\b",      "uniform sampling"),
    (r"\bLWE\b",                       "RNG"),
    (r"\bhomomorphic\b",               "non-homomorphic"),
    # Broader vocabulary — verbs, qualifiers, attributes.
    (r"\b(?:we )?prove\b",             "we disprove"),
    (r"\b(?:we )?proves\b",            "disproves"),
    (r"\b(?:we )?proved\b",            "we disproved"),
    (r"\bproof\b",                     "fallacy"),
    (r"\btheorem\b",                   "conjecture"),
    (r"\blemma\b",                     "counterexample"),
    (r"\b(?:the )?adversary\b",        "the honest party"),
    (r"\bhonest\b",                    "malicious"),
    (r"\bmalicious\b",                 "honest"),
    (r"\balways\b",                    "never"),
    (r"\bguarantees\b",                "fails to guarantee"),
    (r"\bensures\b",                   "fails to ensure"),
    (r"\bachieves\b",                  "fails to achieve"),
    (r"\bauthentic\b",                 "inauthentic"),
    (r"\bauthenticity\b",              "inauthenticity"),
    (r"\bcorrectness\b",               "incorrectness"),
    (r"\bsoundness\b",                 "unsoundness"),
    (r"\bcompleteness\b",              "incompleteness"),
    (r"\b128-bit\b",                   "8-bit"),
    (r"\b256-bit\b",                   "16-bit"),
    (r"\b2048-bit\b",                  "20-bit"),
    (r"\b1024-bit\b",                  "10-bit"),
    (r"\bsecret key\b",                "public key"),
    (r"\bpublic key\b",                "secret key"),
    (r"\bprivate key\b",               "public key"),
    (r"\bencryption scheme\b",         "decryption scheme"),
    (r"\bsignature scheme\b",          "verification scheme"),
    (r"\bdecryption oracle\b",         "encryption oracle"),
    (r"\bsigning oracle\b",            "verification oracle"),
    (r"\bcryptographic\b",             "non-cryptographic"),
)


def _corrupt_paragraph(text: str) -> tuple[str, int]:
    """Apply all corruption rules.  Returns (corrupted_text, n_subs)."""
    n_subs = 0
    out = text
    for pat, repl in _CORRUPTIONS:
        out, k = re.subn(pat, repl, out)
        n_subs += k
    return out, n_subs


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _iacr_text_root() -> Path:
    import os
    return Path(
        os.environ.get(
            "RAG_SIGN_IACR_CACHE",
            str(Path.home() / ".cache" / "rag-sign" / "iacr_text"),
        )
    )


def build_matched_corpora(
    *, years: tuple[int, ...] = (2017, 2018, 2019, 2020),
    target_n: int = DEFAULT_TARGET_PARAGRAPHS,
    min_substitutions: int = 2,
    seed: int = 2026,
) -> tuple[list[str], list[str]]:
    """Return ``(coherent, corrupted)`` lists of equal length.

    Each corrupted paragraph is the in-place corruption of the
    paragraph at the same index in ``coherent``.  Paragraphs that
    do not admit at least ``min_substitutions`` corruptions are
    excluded from both corpora to keep the matching one-to-one.
    """
    rng = random.Random(seed)
    coherent: list[str] = []
    corrupted: list[str] = []

    for year in years:
        if len(coherent) >= target_n:
            break
        root = _iacr_text_root() / str(year)
        if not root.is_dir():
            continue
        files = sorted(root.glob("*.txt"))
        rng.shuffle(files)
        for f in files:
            if len(coherent) >= target_n:
                break
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            paragraphs = _split_paragraphs(text)
            rng.shuffle(paragraphs)
            for para in paragraphs[:5]:  # up to 5 per paper
                if len(coherent) >= target_n:
                    break
                corr, n = _corrupt_paragraph(para)
                if n < min_substitutions:
                    continue
                coherent.append(para)
                corrupted.append(corr)

    if len(coherent) < target_n:
        raise RuntimeError(
            f"only produced {len(coherent)} corruptable paragraphs "
            f"from years {years}"
        )
    return coherent[:target_n], corrupted[:target_n]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[5, 25, 100, 500, 1000],
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028],
    )
    parser.add_argument("--fp-dim", type=int, default=1024)
    parser.add_argument(
        "--source-years", type=int, nargs="+",
        default=[2017, 2018, 2019, 2020],
    )
    parser.add_argument(
        "--n-paragraphs", type=int, default=DEFAULT_TARGET_PARAGRAPHS,
    )
    parser.add_argument(
        "--min-substitutions", type=int, default=2,
        help="paragraphs admitting fewer than this many corruptions are skipped",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="model parameter dtype.  bfloat16 halves memory and is "
             "stable for fine-tuning at >=1B parameter scale.",
    )
    args = parser.parse_args()
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    device = _select_device()
    print(f"Device: {device}")

    print(f"Building matched corpora from IACR years {args.source_years} …")
    coherent, corrupted = build_matched_corpora(
        years=tuple(args.source_years),
        target_n=args.n_paragraphs,
        min_substitutions=args.min_substitutions,
    )
    n_subs = [
        sum(1 for pat, _ in _CORRUPTIONS if re.search(pat, p))
        for p in coherent
    ]
    print(f"  {len(coherent)} matched paragraph pairs "
          f"(median {sorted(n_subs)[len(n_subs)//2]} subs / paragraph; "
          f"min {min(n_subs)}, max {max(n_subs)})")

    print(f"Loading {args.model} ({args.dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype)
    # Snapshot the state_dict on CPU so we can reset between trials
    # without paying the load cost each time.  CPU dict is safe to
    # share across MPS resets.
    print("  snapshotting baseline state_dict for fast trial reset …")
    base_state = {k: v.detach().clone().cpu() for k, v in base.state_dict().items()}

    print("Computing baseline behavioural fingerprint over crypto probes …")
    t0 = time.perf_counter()
    fp_base = fingerprint_behavioral(
        base, tokenizer, device=device,
        dim=args.fp_dim, probes=CRYPTO_PROBES,
    )
    print(f"  {len(fp_base)*8} bits in {time.perf_counter() - t0:.1f}s")

    results: dict = {
        "model": args.model,
        "n_parameters": sum(p.numel() for p in base.parameters()),
        "fingerprint_bits": len(fp_base) * 8,
        "seeds": list(args.seeds),
        "lr": args.lr,
        "device": str(device),
        "n_probes": len(CRYPTO_PROBES),
        "probe_set": "crypto-specific",
        "probes": list(CRYPTO_PROBES),
        "source_years": list(args.source_years),
        "n_paragraphs": args.n_paragraphs,
        "min_substitutions_per_para": args.min_substitutions,
        "median_substitutions_per_para": sorted(n_subs)[len(n_subs) // 2],
        "coherent_examples": coherent[:2],
        "corrupted_examples": corrupted[:2],
        "trials": [],
    }

    for n_steps in args.steps:
        print(f"\n--- {n_steps} fine-tune steps ({len(args.seeds)} seeds) ---")
        trial: dict = {"steps": n_steps, "per_seed": []}
        coh_h, cor_h = [], []
        for seed in args.seeds:
            seed_record: dict = {"seed": seed}
            for label, corpus in [
                ("coherent",  coherent),
                ("corrupted", corrupted),
            ]:
                # Reset to baseline weights via state_dict (fast: ~seconds
                # for 1.7B vs ~minutes from disk).
                base.load_state_dict(base_state, strict=True)
                model = base
                t0 = time.perf_counter()
                _finetune_corpus(
                    model, tokenizer, corpus, n_steps,
                    device=device, lr=args.lr, seed=seed,
                )
                ft_seconds = time.perf_counter() - t0
                fp = fingerprint_behavioral(
                    model, tokenizer, device=device,
                    dim=args.fp_dim, probes=CRYPTO_PROBES,
                )
                h = hamming_distance(fp_base, fp)
                h_pct = h * 100.0 / (len(fp_base) * 8)
                print(
                    f"  seed={seed:<6} {label:<12}  hamming = {h:>3} bits "
                    f"({h_pct:5.2f}%)   ({ft_seconds:.0f}s)"
                )
                seed_record[f"{label}_hamming"] = h
                seed_record[f"{label}_hamming_pct"] = round(h_pct, 2)
                if label == "coherent":
                    coh_h.append(h)
                else:
                    cor_h.append(h)
                if device.type == "mps":
                    torch.mps.empty_cache()
            trial["per_seed"].append(seed_record)

        trial["coherent_mean"]  = round(fmean(coh_h), 2)
        trial["coherent_std"]   = round(pstdev(coh_h), 2)
        trial["corrupted_mean"] = round(fmean(cor_h), 2)
        trial["corrupted_std"]  = round(pstdev(cor_h), 2)
        if trial["coherent_mean"] > 0:
            trial["drift_ratio_mean"] = round(
                trial["corrupted_mean"] / trial["coherent_mean"], 3
            )
        else:
            trial["drift_ratio_mean"] = None
        n_favoured = sum(1 for c, k in zip(cor_h, coh_h, strict=True) if c > k)
        trial["seeds_with_corrupt_gt_coh"] = n_favoured
        trial["seeds_total"] = len(args.seeds)
        print(
            f"  -> coherent : {trial['coherent_mean']:.1f} ± {trial['coherent_std']:.1f}\n"
            f"  -> corrupted: {trial['corrupted_mean']:.1f} ± {trial['corrupted_std']:.1f}\n"
            f"  -> ratio    : {trial['drift_ratio_mean']}\n"
            f"  -> corrupt > coh in {n_favoured}/{len(args.seeds)} seeds"
        )
        results["trials"].append(trial)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
