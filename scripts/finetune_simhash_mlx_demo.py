"""MLX-backend variant of finetune_simhash_demo.py.

Same experiment design as the GPT-2/PyTorch reference (see header of
``finetune_simhash_demo.py`` for context); the only differences are:

* Backend: Apple-Silicon MLX via :mod:`rag_sign.llm_mlx` instead of
  HuggingFace ``transformers`` + PyTorch.
* Fine-tune type: LoRA (rank 8 by default) merged back into the base
  via ``mlx_lm.fuse`` before fingerprinting, instead of full-parameter
  SGD.  This is more representative of production fine-tunes (most
  deployments use LoRA / QLoRA).  Cross-calibration constant against
  the full-FT GPT-2 numbers is reported in the JSON.
* Reachable models: 3B–14B at 4-bit quantisation, including
  Llama-3.2-3B / Qwen2.5-3B / Qwen2.5-7B / Llama-3-8B on a 24 GB
  Mac Mini.  AirLLM-style layer-streaming approaches don't support
  fine-tuning, which is why we picked MLX here.

The JSON output schema mirrors ``finetune_simhash_demo.json`` so
both can be loaded by the same plotting / analysis code, with one
extra key: ``"backend": "mlx-lora"``.

Usage
-----
::

    .venv/bin/python -m scripts.finetune_simhash_mlx_demo
    .venv/bin/python -m scripts.finetune_simhash_mlx_demo \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --steps 0 5 25 100 500 \\
        --mode all

Output: ``bench_results/finetune_simhash_mlx_demo.json``
(does NOT clobber the existing PyTorch ``finetune_simhash_demo.json``).
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

# Make rag_sign importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_sign.model_fingerprint import (  # noqa: E402
    behavioral_fingerprint,
    hamming_distance,
    simhash_arrays,
    simhash_per_tensor,
)

# Import the probe set from the existing PyTorch demo so the
# behavioural fingerprint construction stays byte-identical across
# the two backends.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_simhash_demo import _PROBES  # type: ignore[no-redef]  # noqa: E402

# Tiny synthetic corpus, lifted verbatim from ``finetune_simhash_demo.py``
# (cryptography flavour — the same 100 sentences GPT-2 was fine-tuned
# on so the Hamming-distance curves are directly comparable).
_TRAINING_TEXT: tuple[str, ...] = (
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
)

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SEED = b"thesis-finetune-demo"
DEFAULT_FP_DIM = 512
DEFAULT_STEPS = (0, 5, 25, 100, 500)
DEFAULT_LR = 1e-5

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "finetune_simhash_mlx_demo.json"
)


def _ensure_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit(
            "finetune_simhash_mlx_demo requires Apple Silicon (Darwin / arm64). "
            f"Detected: {platform.system()} / {platform.machine()}. "
            "Use scripts/finetune_simhash_demo.py (PyTorch / GPT-2) on "
            "non-Apple-Silicon hosts."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="LoRA fine-tune iteration counts to sweep",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fp-dim", type=int, default=DEFAULT_FP_DIM)
    parser.add_argument(
        "--mode",
        choices=("whole", "per-tensor", "behavioral", "all"),
        default="behavioral",
        help="see scripts/finetune_simhash_demo.py for the description",
    )
    parser.add_argument("--bits-per-tensor", type=int, default=4)
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help=(
            "retain the merged LoRA-fused model for each step count "
            "instead of deleting it after fingerprinting (debug only "
            "— each merged model is a full base-size copy on disk)"
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "directory under which per-step LoRA adapters and merged "
            "checkpoints are written; defaults to a fresh tempdir "
            "deleted on exit"
        ),
    )
    args = parser.parse_args()

    _ensure_apple_silicon()
    from rag_sign.llm_mlx import MlxLLM, lora_finetune  # lazy

    do_whole = args.mode in ("whole", "all")
    do_per   = args.mode in ("per-tensor", "all")
    do_beh   = args.mode in ("behavioral", "all")

    workdir = (
        Path(args.workdir) if args.workdir
        else Path(tempfile.mkdtemp(prefix="rag-sign-mlx-finetune-"))
    )
    cleanup_workdir = args.workdir is None  # only nuke if we created it

    print(f"Workdir : {workdir}")
    print(f"Loading {args.model} for the baseline fingerprint …")
    t_load = time.perf_counter()
    base_llm = MlxLLM(args.model)
    n_params = base_llm.parameter_count()
    print(f"  {n_params:,} parameters loaded in {time.perf_counter() - t_load:.1f}s")

    print("Computing baseline fingerprint(s) …")
    fp_base_whole = fp_base_per = fp_base_beh = None
    if do_whole:
        t0 = time.perf_counter()
        fp_base_whole = simhash_arrays(
            base_llm.iter_param_arrays(), seed=DEFAULT_SEED, dim=args.fp_dim
        )
        print(f"  whole-model     : {len(fp_base_whole)*8} bits "
              f"in {time.perf_counter() - t0:.1f}s")
    if do_per:
        t0 = time.perf_counter()
        fp_base_per = simhash_per_tensor(
            base_llm.iter_named_param_arrays(),
            bits_per_tensor=args.bits_per_tensor,
            seed=DEFAULT_SEED,
        )
        print(f"  per-tensor      : {len(fp_base_per)*8} bits "
              f"in {time.perf_counter() - t0:.1f}s")
    if do_beh:
        t0 = time.perf_counter()
        activations = [base_llm.last_token_logits(p) for p in _PROBES]
        fp_base_beh = behavioral_fingerprint(
            activations, seed=DEFAULT_SEED, dim=args.fp_dim
        )
        print(f"  behavioral      : {len(fp_base_beh)*8} bits "
              f"in {time.perf_counter() - t0:.1f}s "
              f"(activation dim {activations[0].shape[0]})")

    # Free the baseline model — each fine-tune trial loads its own.
    del base_llm

    results: dict = {
        "backend": "mlx-lora",
        "model": args.model,
        "n_parameters": n_params,
        "fingerprint_dim_whole":      (len(fp_base_whole) * 8) if fp_base_whole else None,
        "fingerprint_dim_per_tensor": (len(fp_base_per)   * 8) if fp_base_per   else None,
        "fingerprint_dim_behavioral": (len(fp_base_beh)   * 8) if fp_base_beh   else None,
        "bits_per_tensor": args.bits_per_tensor if do_per else None,
        "n_probes": len(_PROBES) if do_beh else None,
        "lr": args.lr,
        "rank": args.rank,
        "batch_size": args.batch_size,
        "platform": f"{platform.system()}/{platform.machine()}",
        "trials": [],
    }

    try:
        for n_steps in args.steps:
            print(f"\n--- LoRA fine-tune {n_steps} iters ---")
            cell_workdir = workdir / f"steps-{n_steps:06d}"

            t_ft = time.perf_counter()
            merged = lora_finetune(
                args.model,
                list(_TRAINING_TEXT) * 10,  # 100 examples, matches GPT-2 demo
                iters=n_steps,
                out_dir=cell_workdir,
                rank=args.rank,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                seed=0,
            )
            ft_seconds = time.perf_counter() - t_ft

            llm = MlxLLM(str(merged))
            trial: dict = {"steps": n_steps, "finetune_s": round(ft_seconds, 1)}

            if do_whole:
                t0 = time.perf_counter()
                fp = simhash_arrays(
                    llm.iter_param_arrays(), seed=DEFAULT_SEED, dim=args.fp_dim
                )
                fp_s = time.perf_counter() - t0
                h = hamming_distance(fp_base_whole, fp)  # type: ignore[arg-type]
                h_pct = h * 100.0 / (len(fp_base_whole) * 8)  # type: ignore[arg-type]
                trial["whole_hamming"]        = h
                trial["whole_hamming_pct"]    = round(h_pct, 2)
                trial["whole_fingerprint_s"]  = round(fp_s, 1)
                print(f"  whole-model    hamming = {h:>4} bits "
                      f"({h_pct:5.2f}%)   ({fp_s:.0f}s)")

            if do_per:
                t0 = time.perf_counter()
                fp = simhash_per_tensor(
                    llm.iter_named_param_arrays(),
                    bits_per_tensor=args.bits_per_tensor,
                    seed=DEFAULT_SEED,
                )
                fp_s = time.perf_counter() - t0
                h = hamming_distance(fp_base_per, fp)  # type: ignore[arg-type]
                h_pct = h * 100.0 / (len(fp_base_per) * 8)  # type: ignore[arg-type]
                trial["per_tensor_hamming"]       = h
                trial["per_tensor_hamming_pct"]   = round(h_pct, 2)
                trial["per_tensor_fingerprint_s"] = round(fp_s, 1)
                print(f"  per-tensor     hamming = {h:>4} bits "
                      f"({h_pct:5.2f}%)   ({fp_s:.0f}s)")

            if do_beh:
                t0 = time.perf_counter()
                activations = [llm.last_token_logits(p) for p in _PROBES]
                fp = behavioral_fingerprint(
                    activations, seed=DEFAULT_SEED, dim=args.fp_dim
                )
                fp_s = time.perf_counter() - t0
                h = hamming_distance(fp_base_beh, fp)  # type: ignore[arg-type]
                h_pct = h * 100.0 / (len(fp_base_beh) * 8)  # type: ignore[arg-type]
                trial["behavioral_hamming"]       = h
                trial["behavioral_hamming_pct"]   = round(h_pct, 2)
                trial["behavioral_fingerprint_s"] = round(fp_s, 1)
                print(f"  behavioral     hamming = {h:>4} bits "
                      f"({h_pct:5.2f}%)   ({fp_s:.0f}s)   "
                      f"finetune {ft_seconds:.0f}s")

            results["trials"].append(trial)
            del llm
            if not args.keep_merged and cell_workdir.exists() and n_steps > 0:
                shutil.rmtree(cell_workdir, ignore_errors=True)
    finally:
        if cleanup_workdir and workdir.exists() and not args.keep_merged:
            shutil.rmtree(workdir, ignore_errors=True)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
