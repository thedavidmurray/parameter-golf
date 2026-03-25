# 10-Layer Model with Post-Training Magnitude Pruning

## Core Idea

Post-training magnitude pruning zeroes out weights below a threshold relative to their
row maximum, before int8 quantization. Per-row int8 normalizes by row max anyway, so
sub-threshold weights round to zero in the int8 stream — zlib compresses these well.

**Key finding from 8xH100 test run:**
- threshold=0.10: 27% zeros, 0.97% energy lost
- Actual compression at full training: **1.24x**
- 11 layers exceeded 16MB (16.84MB) — 10 layers fits comfortably (~15.2MB)

Note: early-training compression estimates (~1.37x) are misleading — Muon drives weights
toward near-orthogonal structure as training progresses, reducing compressibility to ~1.24x.

This gives 1 free extra layer over the 9-layer baseline at essentially no quality cost.

## Changes from Naive Baseline

| Change | Detail |
|--------|--------|
| `NUM_LAYERS` | 10 (up from 9) |
| Magnitude pruning | `PRUNE_THRESHOLD=0.10` — applied post-training before quantization |
| Activation | `LeakyReLU(0.5)²` instead of `ReLU²` — ~0.003 BPB improvement |

## Run Command (8xH100)

```bash
NPROC=8 bash records/track_10min_16mb/2026-03-25_10L_MagnitudePruning/run_sweep.sh smoke
```

Or manually:

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=10L_prune010_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
PRUNE_THRESHOLD=0.10 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-25_10L_MagnitudePruning/train_gpt.py
```

## Expected Result

- Baseline (9L, no pruning): ~1.224 BPB
- LeakyReLU baseline (9L): ~1.221 BPB
- This run (10L + pruning + LeakyReLU²): TBD — pending full H100 run

## Status

Pending full H100 run.
