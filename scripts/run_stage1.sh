#!/usr/bin/env bash
# Stage 1 runner: SASRec-ID (3 seeds) → SASRec-Content (3 seeds) → aggregate.
#
# Usage:
#     scripts/run_stage1.sh                        # seeds 42 43 44, full split
#     SEEDS="42 43" scripts/run_stage1.sh          # custom seeds
#     SPLIT=subsample_10pct scripts/run_stage1.sh  # dry run on subsample
#     SKIP_ID=1 scripts/run_stage1.sh              # resume: only content + aggregate
#     SKIP_CONTENT=1 scripts/run_stage1.sh         # only id + aggregate
#     EXTRA="trainer.batch_size=1024" scripts/run_stage1.sh  # forward overrides
#
# All Hydra jobs run sequentially (multirun with a single seed per invocation) so
# a failure in one seed does not kill the remaining queue. Logs go to `logs/`.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

SEEDS="${SEEDS:-42 43 44}"
SPLIT="${SPLIT:-full}"
EXTRA="${EXTRA:-}"
SKIP_ID="${SKIP_ID:-0}"
SKIP_CONTENT="${SKIP_CONTENT:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="$LOG_DIR/stage1_${STAMP}.log"

log() {
    local msg="[$(date -u +'%Y-%m-%d %H:%M:%S')Z] $*"
    echo "$msg" | tee -a "$MASTER_LOG"
}

fail() {
    log "FAIL: $*"
    exit 1
}

run_seed() {
    local config_name="$1"   # sasrec_id | sasrec_content
    local seed="$2"
    local per_run_log="$LOG_DIR/${config_name}_seed${seed}_${STAMP}.log"

    log "→ ${config_name} seed=${seed} split=${SPLIT}"

    if ! uv run python scripts/train_sasrec.py \
            --config-name "$config_name" \
            trainer.seed="$seed" \
            data.split_set="$SPLIT" \
            ${EXTRA} \
            2>&1 | tee "$per_run_log"
    then
        fail "${config_name} seed=${seed} exited non-zero (see $per_run_log)"
    fi

    log "✓ ${config_name} seed=${seed} done"
}

log "stage-1 runner started"
log "repo=$REPO_ROOT"
log "seeds=[${SEEDS}] split=${SPLIT} skip_id=${SKIP_ID} skip_content=${SKIP_CONTENT} skip_agg=${SKIP_AGGREGATE}"
log "extra hydra overrides: '${EXTRA}'"

# Quick GPU sanity check — fail fast if CUDA is not available.
uv run python -c "
import sys, torch
if not torch.cuda.is_available():
    print('CUDA not available — aborting', file=sys.stderr); sys.exit(1)
print(f'cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}')
" 2>&1 | tee -a "$MASTER_LOG"

if [ "$SKIP_ID" != "1" ]; then
    log "=== SASRec-ID ==="
    for seed in $SEEDS; do
        run_seed "sasrec_id" "$seed"
    done
else
    log "=== SASRec-ID skipped (SKIP_ID=1) ==="
fi

if [ "$SKIP_CONTENT" != "1" ]; then
    log "=== SASRec-Content ==="
    for seed in $SEEDS; do
        run_seed "sasrec_content" "$seed"
    done
else
    log "=== SASRec-Content skipped (SKIP_CONTENT=1) ==="
fi

if [ "$SKIP_AGGREGATE" != "1" ]; then
    log "=== Aggregating ==="
    # shellcheck disable=SC2086
    if ! uv run python scripts/aggregate_sasrec.py --seeds $SEEDS 2>&1 | tee -a "$MASTER_LOG"; then
        fail "aggregation exited non-zero"
    fi
    log "✓ aggregation done — results/sasrec_summary.csv"
else
    log "=== aggregation skipped (SKIP_AGGREGATE=1) ==="
fi

log "stage-1 runner finished OK"
log "master log: $MASTER_LOG"
