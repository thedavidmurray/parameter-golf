#!/usr/bin/env bash
# Sweep script for post-training magnitude pruning experiments
#
# Core question: at what prune_threshold do we gain enough bytes to add a layer
# without hurting BPB more than the extra layer helps?
#
# Usage: bash run_sweep.sh [smoke|sweep|full8]

set -e
SCRIPT="$(dirname "$0")/train_gpt.py"
NPROC=${NPROC:-1}
MODE=${1:-smoke}

run() {
    local RUN_ID=$1; shift
    echo ""
    echo "========================================================"
    echo "  Starting run: $RUN_ID"
    echo "========================================================"
    RUN_ID=$RUN_ID torchrun --standalone --nproc_per_node=$NPROC "$SCRIPT" "$@" 2>&1 | tee "logs/${RUN_ID}.log"
    echo ""
    echo ">>> Key metrics for $RUN_ID:"
    grep -E "val_bpb|magnitude_pruning|pretrain_compress|budget_analysis|final_int8" "logs/${RUN_ID}.log" | tail -15
}

mkdir -p logs

if [ "$MODE" = "smoke" ]; then
    # Quick run to verify pruning works and see compression numbers.
    # Warmdown_iters > iterations so warmdown starts immediately → faster convergence signal.
    run prune_smoke \
        ITERATIONS=1000 \
        WARMDOWN_ITERS=1000 \
        VAL_LOSS_EVERY=500 \
        COMPRESSION_LOG_EVERY=200 \
        PRUNE_THRESHOLD=0.10

elif [ "$MODE" = "sweep" ]; then
    # Isolation sweep: is the gain from pruning (bytes freed) worth the BPB cost?
    # Each run: train normally, then prune+quantize at different thresholds.
    # ~10 min per run on 1xH100. Total: ~60 min.
    echo "Running pruning threshold sweep. ~10 min each, ~60 min total."

    # Baseline: no pruning (control)
    run prune_t0 \
        PRUNE_THRESHOLD=0

    # Very conservative: should be nearly lossless
    run prune_t05 \
        PRUNE_THRESHOLD=0.05

    # Recommended: 25% zeros, 0.92% energy lost (sweet spot hypothesis)
    run prune_t10 \
        PRUNE_THRESHOLD=0.10

    # Aggressive: 48% zeros, 6.6% energy lost (probably hurts, but we want to know)
    run prune_t20 \
        PRUNE_THRESHOLD=0.20

    # Very aggressive: how far can we push it?
    run prune_t30 \
        PRUNE_THRESHOLD=0.30

    # Sweet spot with extra layer (if prune_t10 frees enough bytes):
    # 1.30x compression + int8 → can fit 10 layers instead of 9
    run prune_t10_10L \
        PRUNE_THRESHOLD=0.10 \
        NUM_LAYERS=10

    echo ""
    echo "========================================================"
    echo "  SWEEP SUMMARY"
    echo "========================================================"
    printf "  %-22s  %-10s  %-10s  %-14s  %-12s\n" "run_id" "val_bpb" "zeros" "energy_lost" "compress_est"
    for id in prune_t0 prune_t05 prune_t10 prune_t20 prune_t30 prune_t10_10L; do
        if [ -f "logs/${id}.log" ]; then
            BPB=$(grep "final_int8_zlib_roundtrip val_bpb" "logs/${id}.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
            ZEROS=$(grep "magnitude_pruning" "logs/${id}.log" | grep -oP 'zeros=\K[0-9.]+' || echo "0.000")
            ELOST=$(grep "magnitude_pruning" "logs/${id}.log" | grep -oP 'energy_lost=\K[0-9.]+' || echo "0.00000")
            RATIO=$(grep "pretrain_compress" "logs/${id}.log" | grep -oP 'est_compress_ratio:\K[0-9.]+' || echo "?")
            printf "  %-22s  %-10s  %-10s  %-14s  %-12s\n" "$id" "$BPB" "$ZEROS" "$ELOST" "${RATIO}x"
        fi
    done

elif [ "$MODE" = "full8" ]; then
    # Full leaderboard run on 8×H100.
    # Use whichever threshold the sweep showed was optimal.
    BEST_THRESHOLD=${PRUNE_THRESHOLD:-0.10}
    NPROC=8
    echo "Full 3-seed run with prune_threshold=$BEST_THRESHOLD"

    for SEED in 1337 42 2025; do
        run "prune_full_t${BEST_THRESHOLD/./_}_seed${SEED}" \
            SEED=$SEED \
            PRUNE_THRESHOLD=$BEST_THRESHOLD
    done

    echo ""
    echo "========================================================"
    echo "  3-SEED RESULTS"
    echo "========================================================"
    for SEED in 1337 42 2025; do
        ID="prune_full_t${BEST_THRESHOLD/./_}_seed${SEED}"
        BPB=$(grep "final_int8_zlib_roundtrip val_bpb" "logs/${ID}.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
        BYTES=$(grep "Total submission size int8+zlib" "logs/${ID}.log" | tail -1 | grep -oP ':\K[0-9]+' || echo "?")
        echo "  seed=$SEED  val_bpb=$BPB  total_bytes=$BYTES"
    done

else
    echo "Usage: $0 [smoke|sweep|full8]"
    echo "  smoke  - quick 1000-step sanity check (~2 min on 1xH100)"
    echo "  sweep  - 6-run threshold sweep on 1xH100 (~60 min)"
    echo "  full8  - 3-seed leaderboard run on 8xH100 (~30 min)"
    echo ""
    echo "  Env vars:"
    echo "    NPROC=N           number of GPUs (default: 1)"
    echo "    PRUNE_THRESHOLD=X override threshold for full8 mode"
    exit 1
fi
