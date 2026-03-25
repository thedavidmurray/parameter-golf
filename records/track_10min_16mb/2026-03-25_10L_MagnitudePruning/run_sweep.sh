#!/usr/bin/env bash
# Run script for 10-layer magnitude pruning experiments.
#
# Core question: does 10 layers + pruning beat 9 layers without pruning?
# Pruning at threshold=0.10 gives ~1.24x actual compression, fitting 10 layers in 16MB (~15.2MB).
#
# IMPORTANT: Run inside tmux to survive browser/terminal disconnects:
#   tmux new-session -s train
#   NPROC=8 bash run_sweep.sh smoke
#   (detach: Ctrl+B then D  |  reattach: tmux attach -t train)
#
# Usage: bash run_sweep.sh [smoke|sweep|full8]

set -e
SCRIPT="$(dirname "$0")/train_gpt.py"
NPROC=${NPROC:-1}
MODE=${1:-smoke}

# Warn if not inside tmux — closing the browser will kill the run otherwise
if [ -z "$TMUX" ] && [ -z "$TMUX_PANE" ]; then
    echo "WARNING: not running inside tmux. If you close the browser this run will die."
    echo "  To use tmux: tmux new-session -s train, then re-run this script."
    echo ""
fi

run() {
    local RUN_ID=$1; shift
    echo ""
    echo "========================================================"
    echo "  Starting run: $RUN_ID"
    echo "========================================================"
    # Use 'env' so VAR=VALUE args are properly set as environment variables
    RUN_ID=$RUN_ID env "$@" torchrun --standalone --nproc_per_node=$NPROC "$SCRIPT" 2>&1 | tee "logs/${RUN_ID}.log"
    echo ""
    echo ">>> Key metrics for $RUN_ID:"
    grep -E "val_bpb|magnitude_pruning|pretrain_compress|swa:|final_int8|Total submission" "logs/${RUN_ID}.log" | tail -20
}

mkdir -p logs

if [ "$MODE" = "smoke" ]; then
    # Full 10-min run on 8xH100 to get a real BPB number.
    # 10 layers, pruning=0.10, LeakyReLU^2, SWA, time-based warmdown.
    # Expected size: ~15.2MB.
    run smoke_10L_prune010 \
        NUM_LAYERS=10 \
        PRUNE_THRESHOLD=0.10 \
        VAL_LOSS_EVERY=500 \
        COMPRESSION_LOG_EVERY=1000

elif [ "$MODE" = "sweep" ]; then
    # Isolation sweep: 9L baseline vs 10L+prune, each 10 min on 8xH100.
    echo "Running layer/pruning sweep. ~10 min each, ~20 min total."

    # Control: 9 layers, no pruning
    run sweep_9L_noPrune \
        NUM_LAYERS=9 \
        PRUNE_THRESHOLD=0

    # 10 layers + pruning (main hypothesis)
    run sweep_10L_prune010 \
        NUM_LAYERS=10 \
        PRUNE_THRESHOLD=0.10

    echo ""
    echo "========================================================"
    echo "  SWEEP SUMMARY"
    echo "========================================================"
    printf "  %-25s  %-10s  %-10s  %-12s\n" "run_id" "val_bpb" "zeros" "compress"
    for id in sweep_9L_noPrune sweep_10L_prune010; do
        if [ -f "logs/${id}.log" ]; then
            BPB=$(grep "final_int8_zlib_roundtrip val_bpb" "logs/${id}.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
            ZEROS=$(grep "magnitude_pruning" "logs/${id}.log" | grep -oP 'zeros=\K[0-9.]+' || echo "0.000")
            RATIO=$(grep "pretrain_compress" "logs/${id}.log" | grep -oP 'est_compress_ratio:\K[0-9.]+' || echo "?")
            printf "  %-25s  %-10s  %-10s  %-12s\n" "$id" "$BPB" "$ZEROS" "${RATIO}x"
        fi
    done

elif [ "$MODE" = "full8" ]; then
    # Full 3-seed leaderboard run on 8xH100.
    NPROC=8
    echo "Full 3-seed run: 10L + prune=0.10 + SWA + LeakyReLU^2"

    for SEED in 1337 42 2025; do
        run "full8_10L_prune010_seed${SEED}" \
            NUM_LAYERS=10 \
            PRUNE_THRESHOLD=0.10 \
            SEED=$SEED
    done

    echo ""
    echo "========================================================"
    echo "  3-SEED RESULTS"
    echo "========================================================"
    for SEED in 1337 42 2025; do
        ID="full8_10L_prune010_seed${SEED}"
        BPB=$(grep "final_int8_zlib_roundtrip val_bpb" "logs/${ID}.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
        BYTES=$(grep "Total submission size int8+zlib" "logs/${ID}.log" | tail -1 | grep -oP '[0-9]+' | tail -1 || echo "?")
        echo "  seed=$SEED  val_bpb=$BPB  total_bytes=$BYTES"
    done

else
    echo "Usage: $0 [smoke|sweep|full8]"
    echo "  smoke  - 10-min run on 8xH100, get real BPB"
    echo "  sweep  - 2-run comparison: 9L vs 10L+prune (~20 min total)"
    echo "  full8  - 3-seed leaderboard run on 8xH100 (~30 min)"
    echo ""
    echo "  Env vars:"
    echo "    NPROC=N    number of GPUs (default: 1)"
    exit 1
fi
