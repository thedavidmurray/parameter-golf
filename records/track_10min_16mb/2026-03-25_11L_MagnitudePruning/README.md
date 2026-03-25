# 11-Layer Model with Post-Training Magnitude Pruning

## Core Idea

Post-training magnitude pruning zeroes out weights below a threshold relative to their
row maximum, before int8 quantization. Since per-row int8 normalizes by row max anyway,
these sub-threshold weights round to zero in the int8 stream — and zlib compresses runs
of zeros dramatically better than random bytes.

**Measured on 1xH100 smoke run:**
- threshold=0.10: 27% zeros, 0.97% energy lost, **1.37x compression ratio**
- At 1.37x: 11 layers fit in 16MB (vs 9 layers at baseline 1.08x)

This gives 2 extra transformer layers for free — essentially no quality cost from pruning,
large quality gain from the extra depth.

## Changes from Naive Baseline

| Change | Detail |
|--------|--------|
| `NUM_LAYERS` | 11 (up from 9) |
| Magnitude pruning | `PRUNE_THRESHOLD=0.10` — applied post-training before quantization |
| Activation | `LeakyReLU(0.5)²` instead of `ReLU²` — ~0.003 BPB improvement |

## Run Command (1xH100 smoke)

```bash
bash records/track_10min_16mb/2026-03-25_11L_MagnitudePruning/run_sweep.sh smoke
```

## Run Command (8xH100 leaderboard)

```bash
NPROC=8 bash records/track_10min_16mb/2026-03-25_11L_MagnitudePruning/run_sweep.sh full8
```

Or manually:

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=11L_prune010_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
PRUNE_THRESHOLD=0.10 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-25_11L_MagnitudePruning/train_gpt.py
```

## Expected Result

- Baseline (9L, no pruning): ~1.224 BPB
- LeakyReLU baseline (9L): ~1.221 BPB
- This run (11L + pruning): TBD — pending H100 run

With 2 extra layers, expect meaningful improvement over the 9L LeakyReLU baseline.
Current SOTA to beat: **1.1194** (`LeakyReLU_LegalTTT_ParallelMuon`).

## Status

Pending H100 run. Script committed and ready.
