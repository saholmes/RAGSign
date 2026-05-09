"""MLX-backend smoke test for the model-fingerprint pipeline.

Loads a small Q4-quantised model via :class:`rag_sign.llm_mlx.MlxLLM`
and computes the same three fingerprints that
``finetune_simhash_demo.py`` produces — but on a model the Mac Mini
could never hold in fp16 PyTorch.  No fine-tune step here; this
script only proves the *path*: model load → weight-space SimHash →
behavioural SimHash → JSON output.

This is the "does the MLX backend even work end-to-end?" check
before we commit to the full LoRA + drift sweep.

Usage
-----
::

    .venv/bin/python -m scripts.fingerprint_mlx_smoke
    .venv/bin/python -m scripts.fingerprint_mlx_smoke \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit
    .venv/bin/python -m scripts.fingerprint_mlx_smoke \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit --mode all

The default model (Llama-3.2-3B 4-bit) is ~2 GB on disk and ~2 GB
resident — runs in <2 minutes wall-clock end-to-end on an M2 Mac
Mini, including the first-time download from HuggingFace.

Output
------
``bench_results/fingerprint_mlx_smoke.json`` (committable; pure
aggregate numbers, no model weights).

Hardware
--------
Apple Silicon only (Mac Mini M1/M2/M3/M4, MacBook Air/Pro on
Apple-Silicon SoCs).  Fails with a clear install-hint on Intel Mac
or non-Mac platforms.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Make rag_sign importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_sign.model_fingerprint import (  # noqa: E402
    behavioral_fingerprint,
    simhash_arrays,
    simhash_per_tensor,
)

# Probe set lifted verbatim from finetune_simhash_demo.py so the
# MLX numbers can be cross-checked against the existing GPT-2 sweep
# under the same fingerprint construction.
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

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_SEED = b"thesis-finetune-demo"  # matches finetune_simhash_demo.py
DEFAULT_FP_DIM = 512

RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "bench_results"
    / "fingerprint_mlx_smoke.json"
)


def _ensure_apple_silicon() -> None:
    """Fail fast with a useful message on non-MLX hosts."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit(
            "fingerprint_mlx_smoke requires Apple Silicon (Darwin / arm64). "
            f"Detected: {platform.system()} / {platform.machine()}. "
            "Use scripts/finetune_simhash_demo.py (PyTorch / GPT-2) on "
            "non-Apple-Silicon hosts."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "MLX-formatted model id or local path.  Look under "
            "huggingface.co/mlx-community for the canonical Q4 conversions; "
            "any model that loads via `mlx_lm.load()` works."
        ),
    )
    parser.add_argument("--fp-dim", type=int, default=DEFAULT_FP_DIM)
    parser.add_argument(
        "--mode",
        choices=("whole", "per-tensor", "behavioral", "all"),
        default="all",
        help=(
            "fingerprinting variant: 'whole' = single SimHash over all "
            "parameters; 'per-tensor' = small SimHash per param tensor; "
            "'behavioral' = SimHash of last-token logits over a fixed "
            "probe set; 'all' = run all three"
        ),
    )
    parser.add_argument(
        "--bits-per-tensor",
        type=int,
        default=4,
        help="bits each tensor contributes in per-tensor mode",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED.decode("utf-8"),
        help="domain-separated SimHash seed (string; UTF-8 encoded)",
    )
    args = parser.parse_args()
    seed = args.seed.encode("utf-8")

    _ensure_apple_silicon()
    # Lazy import — keeps the script importable for `--help` without
    # requiring the heavy native deps.
    from rag_sign.llm_mlx import MlxLLM

    do_whole = args.mode in ("whole", "all")
    do_per = args.mode in ("per-tensor", "all")
    do_beh = args.mode in ("behavioral", "all")

    print(f"Loading {args.model} (lazy quantised tensors stay on the MLX heap) …")
    t_load = time.perf_counter()
    llm = MlxLLM(args.model)
    load_seconds = time.perf_counter() - t_load
    n_params = llm.parameter_count()
    print(f"  {n_params:,} scalar parameters (load {load_seconds:.1f}s)")

    results: dict = {
        "model": args.model,
        "n_parameters": n_params,
        "load_s": round(load_seconds, 1),
        "fingerprint_dim": args.fp_dim,
        "bits_per_tensor": args.bits_per_tensor if do_per else None,
        "n_probes": len(_PROBES) if do_beh else None,
        "seed": args.seed,
        "platform": f"{platform.system()}/{platform.machine()}",
    }

    if do_whole:
        print("Whole-model SimHash …")
        t0 = time.perf_counter()
        fp = simhash_arrays(llm.iter_param_arrays(), seed=seed, dim=args.fp_dim)
        dt = time.perf_counter() - t0
        results["whole_fingerprint_hex"] = fp.hex()
        results["whole_fingerprint_s"] = round(dt, 1)
        print(f"  {len(fp) * 8} bits in {dt:.1f}s")

    if do_per:
        print("Per-tensor SimHash …")
        t0 = time.perf_counter()
        fp = simhash_per_tensor(
            llm.iter_named_param_arrays(),
            bits_per_tensor=args.bits_per_tensor,
            seed=seed,
        )
        dt = time.perf_counter() - t0
        results["per_tensor_fingerprint_hex"] = fp.hex()
        results["per_tensor_fingerprint_s"] = round(dt, 1)
        print(f"  {len(fp) * 8} bits in {dt:.1f}s")

    if do_beh:
        print(f"Behavioural SimHash over {len(_PROBES)} probes …")
        t0 = time.perf_counter()
        activations = [llm.last_token_logits(p) for p in _PROBES]
        fp = behavioral_fingerprint(activations, seed=seed, dim=args.fp_dim)
        dt = time.perf_counter() - t0
        results["behavioral_fingerprint_hex"] = fp.hex()
        results["behavioral_fingerprint_s"] = round(dt, 1)
        results["activation_shape"] = list(activations[0].shape)
        print(f"  {len(fp) * 8} bits in {dt:.1f}s "
              f"(activation dim {activations[0].shape[0]})")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
