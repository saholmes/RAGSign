#!/usr/bin/env bash
#
# Cloud-host bootstrap for the RAG-Sign experiment harness.
# Designed for fresh Lambda Labs / RunPod / AWS GPU instances.
#
# Usage (inline):
#
#   curl -sL https://raw.githubusercontent.com/saholmes/RAGSign/main/scripts/cloud_bootstrap.sh \
#     | bash -s -- --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
#                  --dtype bfloat16 \
#                  --steps 5 25 100 500 1000
#
# Or after cloning manually:
#
#   bash scripts/cloud_bootstrap.sh --model … --dtype …
#
# Required environment variables:
#
#   R2_ACCESS_KEY_ID         R2 access key
#   R2_SECRET_ACCESS_KEY     R2 secret key
#   R2_ENDPOINT_URL          full R2 endpoint, e.g.
#                            https://abcd1234.r2.cloudflarestorage.com
#                            (alternatively set R2_ACCOUNT_ID to derive
#                             the endpoint automatically)
#   R2_TEXT_CACHE_URI        r2:// URI of the pre-extracted text cache
#                            e.g. r2://iacr-text-cache/
#
# Optional:
#
#   R2_RESULTS_URI           where to upload bench_results/*.json after
#                            the run completes; if unset, results stay
#                            local
#   RAGSIGN_REPO             override the git URL (default:
#                            https://github.com/saholmes/RAGSign.git)
#   RAGSIGN_BRANCH           override the branch (default: main)
#   SKIP_EXPERIMENT          if non-empty, skip the experiment and only
#                            do environment setup + corpus sync

set -euo pipefail

REPO_URL="${RAGSIGN_REPO:-https://github.com/saholmes/RAGSign.git}"
REPO_BRANCH="${RAGSIGN_BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/RAGSign}"

log() { echo "[bootstrap] $*" >&2; }
fail() { echo "[bootstrap] FATAL: $*" >&2; exit 1; }

# ------------------------------------------------------------------
# Phase 1 — required env vars
# ------------------------------------------------------------------

require() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        fail "environment variable $name is required"
    fi
}

require R2_ACCESS_KEY_ID
require R2_SECRET_ACCESS_KEY
require R2_TEXT_CACHE_URI
if [ -z "${R2_ENDPOINT_URL:-}" ] && [ -z "${R2_ACCOUNT_ID:-}" ]; then
    fail "either R2_ENDPOINT_URL or R2_ACCOUNT_ID must be set"
fi

# ------------------------------------------------------------------
# Phase 2 — system packages
# ------------------------------------------------------------------

log "installing system packages…"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv git curl ca-certificates
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y -q python3 python3-pip git curl ca-certificates
else
    log "no apt-get/yum found — assuming the AMI already has python3, git, curl"
fi

# ------------------------------------------------------------------
# Phase 3 — uv (fast Python package manager)
# ------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || fail "uv not on PATH after install"

# ------------------------------------------------------------------
# Phase 4 — clone repo
# ------------------------------------------------------------------

if [ ! -d "$WORKDIR" ]; then
    log "cloning $REPO_URL ($REPO_BRANCH)…"
    git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# ------------------------------------------------------------------
# Phase 5 — venv + dependencies
# ------------------------------------------------------------------

if [ ! -d .venv ]; then
    log "creating venv…"
    uv venv --python 3.12
fi
source .venv/bin/activate

log "installing rag-sign with cloud + dev extras…"
uv pip install -e ".[dev,pdf,cloud]"

# Heavy ML stack — only install if we'll actually run the experiment.
if [ -z "${SKIP_EXPERIMENT:-}" ]; then
    log "installing ML stack (torch + transformers + accelerate)…"
    uv pip install torch transformers datasets peft accelerate
fi

# ------------------------------------------------------------------
# Phase 6 — sync IACR text cache from R2
# ------------------------------------------------------------------

CACHE_DIR="${RAG_SIGN_IACR_CACHE:-$HOME/.cache/rag-sign/iacr_text}"
mkdir -p "$CACHE_DIR"

log "syncing $R2_TEXT_CACHE_URI → $CACHE_DIR …"
.venv/bin/python -c "
from rag_sign.r2_sync import sync_to_local
import os
sync_to_local(os.environ['R2_TEXT_CACHE_URI'], os.environ.get('RAG_SIGN_IACR_CACHE', '$CACHE_DIR'))
"

# ------------------------------------------------------------------
# Phase 7 — run experiment (forwarding all CLI args)
# ------------------------------------------------------------------

if [ -n "${SKIP_EXPERIMENT:-}" ]; then
    log "SKIP_EXPERIMENT set; skipping run."
    exit 0
fi

if [ "$#" -eq 0 ]; then
    log "no experiment args supplied — defaulting to SmolLM2-1.7B sweep"
    set -- \
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --dtype bfloat16 \
        --steps 5 25 100 500 1000 \
        --seeds 2026 2027 2028 \
        --fp-dim 1024 \
        --source-years 2014 2015 2016 2017 2018 2019 2020 2021 \
        --n-paragraphs 100 \
        --min-substitutions 1
fi

log "running: scripts.iacr_corruption_demo $*"
.venv/bin/python -u -m scripts.iacr_corruption_demo "$@" 2>&1 \
    | tee "$WORKDIR/run.log"

# ------------------------------------------------------------------
# Phase 8 — optionally upload results back to R2
# ------------------------------------------------------------------

if [ -n "${R2_RESULTS_URI:-}" ]; then
    log "uploading bench_results → $R2_RESULTS_URI …"
    .venv/bin/python -c "
from rag_sign.r2_sync import upload_directory
import os
upload_directory('$WORKDIR/bench_results', os.environ['R2_RESULTS_URI'])
"
    log "also uploading run.log…"
    .venv/bin/python -c "
from rag_sign.r2_sync import upload_directory
import os
upload_directory('$WORKDIR', os.environ['R2_RESULTS_URI'], file_glob='run.log')
"
fi

log "done."
