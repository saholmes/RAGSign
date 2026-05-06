"""Empirical AI-safety drift gate: real fine-tunes of a small LLM.

Demonstrates the central claim of the model-fingerprint section: a
SimHash over LLM weights is locality-sensitive, so a small fine-tune
moves the fingerprint by only a few bits while a large fine-tune (or
a different model entirely) moves it far enough that the safety gate
fires.

Pipeline
--------

1. Load GPT-2 small (124M parameters, the standard small-OSS-LLM
   reference baseline) from HuggingFace.
2. Fingerprint the unmodified base weights — this is the
   *enrolment* SimHash against which everything is measured.
3. Make several fresh copies and fine-tune each one for a different
   number of training steps on a tiny synthetic corpus (loaded from
   ``wikitext-2`` or supplied via ``--data``).
4. Fingerprint each fine-tuned copy; print and persist the Hamming
   distance to the baseline.

The fine-tuning intensities span four orders of magnitude in
parameter movement (from a single optimisation step up to a few
hundred), giving a clean empirical curve that mirrors the corpus-
drift curve in the reproducibility report.

Usage
-----
::

    .venv/bin/python -m scripts.finetune_simhash_demo
    .venv/bin/python -m scripts.finetune_simhash_demo --steps 0 1 5 25 100 500

Results land in ``bench_results/finetune_simhash_demo.json`` (this
path *is* committed — pure aggregate numbers, no model weights).
The HuggingFace cache lives in ``~/.cache/huggingface`` (outside the
repo, not committed).

Hardware
--------
Designed to run on a CPU-only Apple-Silicon laptop in a few minutes
end-to-end.  Uses MPS (Apple GPU) when ``torch.backends.mps`` is
available; falls back to CPU otherwise.  GPT-2-small is small
enough that even pure CPU completes the whole sweep in well under
half an hour.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

# Make rag_sign importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch  # type: ignore[import-untyped]
from torch.utils.data import DataLoader  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from rag_sign.model_fingerprint import (
    behavioral_fingerprint,
    hamming_distance,
    simhash_arrays,
    simhash_per_tensor,
)

# Fixed probe set for the behavioral fingerprint.  Picked to span a
# range of natural-language registers so a fine-tune touching any of
# them shifts the fingerprint.  Order is part of the construction —
# both enrolment and audit must use this exact list.
_PROBES: tuple[str, ...] = (
    "The capital of France is",
    "Once upon a time, there was a young",
    "def hello_world():",
    "Theorem: For every prime p,",
    "Dear Sir or Madam,\n\nI am writing to",
    "The patient presents with",
    "Roses are red, violets are",
    "In conclusion, the data shows that",
    "Q: What is the meaning of life? A:",
    "She opened the door and saw",
)

DEFAULT_MODEL = "gpt2"            # 124M parameters
DEFAULT_SEED = b"thesis-finetune-demo"
DEFAULT_FP_DIM = 512
DEFAULT_STEPS = (0, 1, 5, 25, 100, 500)
DEFAULT_LR = 5e-5

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "finetune_simhash_demo.json"
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _iter_param_arrays(model: torch.nn.Module) -> Iterator[np.ndarray]:
    """Yield numpy float32 arrays of every parameter tensor, in
    ``state_dict`` order — deterministic given the same model."""
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key].detach().to(dtype=torch.float32, device="cpu")
        yield t.numpy().reshape(-1)


def _iter_named_tensors(model: torch.nn.Module) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, numpy float32 array) for every parameter tensor in
    ``state_dict`` order."""
    sd = model.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key].detach().to(dtype=torch.float32, device="cpu")
        yield key, t.numpy().reshape(-1)


def fingerprint_whole_model(
    model: torch.nn.Module, *, seed: bytes = DEFAULT_SEED, dim: int = DEFAULT_FP_DIM
) -> bytes:
    """Slow, low-sensitivity baseline: one SimHash over the flattened model."""
    return simhash_arrays(_iter_param_arrays(model), seed=seed, dim=dim)


def fingerprint_per_tensor(
    model: torch.nn.Module,
    *,
    seed: bytes = DEFAULT_SEED,
    bits_per_tensor: int = 4,
) -> bytes:
    """Fast, high-sensitivity per-tensor SimHash."""
    return simhash_per_tensor(
        _iter_named_tensors(model), bits_per_tensor=bits_per_tensor, seed=seed
    )


def fingerprint_behavioral(
    model: torch.nn.Module,
    tokenizer,
    *,
    device: torch.device,
    seed: bytes = DEFAULT_SEED,
    dim: int = DEFAULT_FP_DIM,
    probes: tuple[str, ...] = _PROBES,
) -> bytes:
    """Behavioural fingerprint: SimHash of last-token logits over a fixed probe set."""
    model.eval().to(device)
    activations: list[np.ndarray] = []
    with torch.no_grad():
        for probe in probes:
            ids = tokenizer.encode(probe, return_tensors="pt").to(device)
            logits = model(ids).logits[0, -1, :]      # last-token only
            activations.append(logits.detach().to("cpu").float().numpy())
    return behavioral_fingerprint(activations, seed=seed, dim=dim)


# ---------------------------------------------------------------------------
# Tiny training corpus
# ---------------------------------------------------------------------------


def _load_training_text() -> list[str]:
    """A tiny in-memory training corpus.

    We deliberately use a small synthetic dataset so the script has
    no network dependency and runs reproducibly in CI-like
    conditions.  100 sentences of cryptography flavour text — small
    enough that GPT-2 will move noticeably even after 25 SGD steps.
    """
    snippets = [
        "Schnorr signatures are EUF-CMA secure under the discrete-log assumption.",
        "MinHash gives a Jaccard-similarity-preserving sketch of a set.",
        "ChaCha20 is a stream cipher built from add-rotate-xor primitives.",
        "An RSA modulus must be a product of two primes of equal length.",
        "FRI is a low-degree-test inner protocol used in STARKs.",
        "Reed-Solomon codes correct up to (n-k)/2 symbol errors.",
        "BCH codes are a subclass of cyclic codes over a finite field.",
        "Fuzzy extractors generate stable keys from noisy biometric data.",
        "Locality-sensitive hashing trades precision for sublinear lookup.",
        "Secure sketches publish helper data that reveals no entropy.",
    ]
    # Repeat to give meaningful training: 100 examples.
    return snippets * 10


def _tokenise(tokenizer, texts: list[str], max_len: int = 64) -> list[dict]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=False,
    )
    return [{"input_ids": ids} for ids in encoded["input_ids"]]


# ---------------------------------------------------------------------------
# Fine-tune loop
# ---------------------------------------------------------------------------


def _finetune(
    model: torch.nn.Module,
    tokenizer,
    n_steps: int,
    *,
    device: torch.device,
    batch_size: int = 4,
    lr: float = DEFAULT_LR,
    seed: int = 0,
) -> None:
    """In-place fine-tune the model for ``n_steps`` SGD steps."""
    if n_steps <= 0:
        return

    torch.manual_seed(seed)
    model.train().to(device)

    examples = _tokenise(tokenizer, _load_training_text())
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1

    model.eval()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="fine-tune step counts to sweep",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--fp-dim", type=int, default=DEFAULT_FP_DIM)
    parser.add_argument(
        "--mode",
        choices=("whole", "per-tensor", "behavioral", "all"),
        default="behavioral",
        help=(
            "fingerprinting variant: 'whole' = single SimHash over all "
            "parameters (slow, conservative); 'per-tensor' = small "
            "SimHash per state_dict entry (fast, moderate); "
            "'behavioral' = SimHash of last-token logits on a fixed "
            "probe set (fast, very sensitive); 'all' = run all three "
            "side-by-side"
        ),
    )
    parser.add_argument(
        "--bits-per-tensor",
        type=int,
        default=4,
        help="bits each tensor contributes in per-tensor mode",
    )
    args = parser.parse_args()

    device = _select_device()
    print(f"Device: {device}")
    print(f"Loading {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model)
    n_params = sum(p.numel() for p in base.parameters())
    print(f"  {n_params:,} parameters loaded")

    do_whole = args.mode in ("whole", "all")
    do_per = args.mode in ("per-tensor", "all")
    do_beh = args.mode in ("behavioral", "all")

    print("Computing baseline fingerprint(s) …")
    fp_base_whole = fp_base_per = fp_base_beh = None
    if do_whole:
        t0 = time.perf_counter()
        fp_base_whole = fingerprint_whole_model(base, dim=args.fp_dim)
        print(
            f"  whole-model     : {len(fp_base_whole)*8} bits in "
            f"{time.perf_counter() - t0:.1f}s"
        )
    if do_per:
        t0 = time.perf_counter()
        fp_base_per = fingerprint_per_tensor(
            base, bits_per_tensor=args.bits_per_tensor
        )
        print(
            f"  per-tensor      : {len(fp_base_per)*8} bits in "
            f"{time.perf_counter() - t0:.1f}s"
        )
    if do_beh:
        t0 = time.perf_counter()
        fp_base_beh = fingerprint_behavioral(
            base, tokenizer, device=device, dim=args.fp_dim
        )
        print(
            f"  behavioral      : {len(fp_base_beh)*8} bits in "
            f"{time.perf_counter() - t0:.1f}s"
        )

    results: dict = {
        "model": args.model,
        "n_parameters": n_params,
        "fingerprint_dim_whole": (len(fp_base_whole) * 8) if fp_base_whole else None,
        "fingerprint_dim_per_tensor": (len(fp_base_per) * 8) if fp_base_per else None,
        "fingerprint_dim_behavioral": (len(fp_base_beh) * 8) if fp_base_beh else None,
        "bits_per_tensor": args.bits_per_tensor if do_per else None,
        "n_probes": len(_PROBES) if do_beh else None,
        "lr": args.lr,
        "device": str(device),
        "trials": [],
    }

    for n_steps in args.steps:
        print(f"\n--- fine-tune {n_steps} steps ---")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        t0 = time.perf_counter()
        _finetune(model, tokenizer, n_steps, device=device, lr=args.lr)
        ft_seconds = time.perf_counter() - t0

        trial: dict = {"steps": n_steps, "finetune_s": round(ft_seconds, 1)}

        if do_whole:
            t0 = time.perf_counter()
            fp = fingerprint_whole_model(model, dim=args.fp_dim)
            fp_s = time.perf_counter() - t0
            h = hamming_distance(fp_base_whole, fp)  # type: ignore[arg-type]
            h_pct = h * 100.0 / (len(fp_base_whole) * 8)  # type: ignore[arg-type]
            trial["whole_hamming"] = h
            trial["whole_hamming_pct"] = round(h_pct, 2)
            trial["whole_fingerprint_s"] = round(fp_s, 1)
            print(
                f"  whole-model    hamming = {h:>4} bits ({h_pct:5.2f}%)   "
                f"({fp_s:.0f}s)"
            )

        if do_per:
            t0 = time.perf_counter()
            fp = fingerprint_per_tensor(
                model, bits_per_tensor=args.bits_per_tensor
            )
            fp_s = time.perf_counter() - t0
            h = hamming_distance(fp_base_per, fp)  # type: ignore[arg-type]
            h_pct = h * 100.0 / (len(fp_base_per) * 8)  # type: ignore[arg-type]
            trial["per_tensor_hamming"] = h
            trial["per_tensor_hamming_pct"] = round(h_pct, 2)
            trial["per_tensor_fingerprint_s"] = round(fp_s, 1)
            print(
                f"  per-tensor     hamming = {h:>4} bits ({h_pct:5.2f}%)   "
                f"({fp_s:.0f}s)"
            )

        if do_beh:
            t0 = time.perf_counter()
            fp = fingerprint_behavioral(
                model, tokenizer, device=device, dim=args.fp_dim
            )
            fp_s = time.perf_counter() - t0
            h = hamming_distance(fp_base_beh, fp)  # type: ignore[arg-type]
            h_pct = h * 100.0 / (len(fp_base_beh) * 8)  # type: ignore[arg-type]
            trial["behavioral_hamming"] = h
            trial["behavioral_hamming_pct"] = round(h_pct, 2)
            trial["behavioral_fingerprint_s"] = round(fp_s, 1)
            print(
                f"  behavioral     hamming = {h:>4} bits ({h_pct:5.2f}%)   "
                f"({fp_s:.0f}s)   finetune {ft_seconds:.0f}s"
            )

        results["trials"].append(trial)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
